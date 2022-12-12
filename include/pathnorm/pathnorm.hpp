#pragma once

#include <cmath>
#include <limits>

#include <algorithm>
#include <map>
#include <numeric>
#include <set>
#include <vector>

#include "Eigen/Dense"

using Eigen::Array, Eigen::Dynamic;
using Eigen::seq;
using Eigen::placeholders::last;

typedef Array<double, 1, Dynamic> Row;
typedef Array<double, Dynamic, Dynamic> Array2D;

typedef std::map<std::pair<size_t, size_t>, std::pair<Row, Row>> Cache;

class PathNormProximalMap {
public:
  class ProcessRow {
  public:
    Row postprocessRow(Row &row) const {
      // In Postprocessing:
      // The order of the operation matters:
      // 0. Reorder
      // 1. Sign recovery
      return row(_reorder) * _sign;
    }

    template <typename RowXpr> explicit ProcessRow(const RowXpr &row) {
      // In Preprocessing:
      // The order of the operation matters:
      // 1. Sign
      // 2. then Order
      // 3. then Abs
      // 4. then Cumsum
      // Thus, Sign recovery should be taken after recovering the order of
      // elements.

      // 1. Sign
      _sign = Eigen::sign(row);

      // 2. Order Descending-ly w.r.t. absolute vals & Cache Re-ordering
      // indices.
      auto abs = Eigen::abs(row);
      auto sorter = _sort_indexes(abs);
      _reorder = _sort_indexes(sorter);

      // 3. Take the Absolute of the ordered values.
      _abs = abs(sorter);

      // 4. Cumulative Sum
      _cumsum = Row(_abs);
      for (size_t i = 1; i < _cumsum.size(); ++i)
        _cumsum(i) += _cumsum(i - 1);
    }

    const Row &cumsum() const { return _cumsum; }
    const Row &abs() const { return _abs; }

  private:
    Row _sign;
    Row _abs;
    Row _cumsum;
    std::vector<size_t> _reorder;

    template <typename RowXpr>
    static std::vector<size_t> _sort_indexes(const RowXpr &v) {
      // Credits: https://stackoverflow.com/a/12399290

      // initialize original index locations
      std::vector<size_t> idx(v.size());
      std::iota(idx.begin(), idx.end(), 0);

      // sort indexes based on comparing values in v in descending order.
      std::sort(idx.begin(), idx.end(),
                [&v](size_t i1, size_t i2) { return v[i1] > v[i2]; });

      return idx;
    }
  };

private:
  const Array2D _mX;
  const Array2D _mY;
  const Row _v0;
  const Row _w0;
  const std::vector<ProcessRow> _pX;
  const std::vector<ProcessRow> _pY;
  const double _lambda;
  const double _lambda2;

  static double _proxObj(const Row &absx, const Row &absy, const Row &v,
                         const Row &w, double lambda) {
    /*
     * Evaluate the proximal objective:
     * ||v - x||^2 + ||w - y||^2 + lambda * ||v||_1 ||w||_1
     */
    double vt = 0.5 * Eigen::square(v - absx).sum();
    double wt = 0.5 * Eigen::square(w - absy).sum();
    return vt + wt + lambda * v.sum() * w.sum();
  }

  std::pair<Row, Row> _vwStationary(size_t rowIdx, size_t sv, size_t sw) const {
    /**
     * Compute the stationary points v^(sv, sw), w^(sv, sw) using Eq(26)
     * (Latorre et al.; 2020) for the pair (sv, sw) passed as tuple 's_vw'.
     **/

    double sumx = (sv > 0) ? _pX[rowIdx].cumsum()[sv - 1] : 0.0;
    double sumy = (sw > 0) ? _pY[rowIdx].cumsum()[sw - 1] : 0.0;
    double u = 1. / (1 - sv * sw * _lambda);
    Row v = _pX[rowIdx].abs() + u * (_lambda2 * sw * sumx - _lambda * sumy);
    Row w = _pY[rowIdx].abs() + u * (_lambda2 * sv * sumy - _lambda * sumx);
    v(seq(sv, last)).setZero();
    w(seq(sw, last)).setZero();
    return {std::move(v), std::move(w)};
  }

  bool _reject(size_t rowIdx, size_t sv, size_t sw, Cache &cache) const {
    // Check if condition 1 in Lemma 18(Latorre et al.; 2020)is violated.
    if (sv * sw > 1. / _lambda2)
      return true;
    if (cache.count({sv, sw}) == 0)
      cache[{sv, sw}] = _vwStationary(rowIdx, sv, sw);
    auto &[v, w] = cache.at({sv, sw});
    double msv = (sv > 0) ? v[sv - 1] : 0.0;
    double msw = (sw > 0) ? w[sw - 1] : 0.0;
    return msv < 0.0 or msw < 0.0;
  }

  std::set<std::pair<size_t, size_t>> _sparsityPairsMFB(size_t rowIdx,
                                                        Cache &cache) const {
    size_t m = _mX.cols();
    size_t p = _mY.cols();
    int64_t sv = 0, sw = m;
    auto pairs = std::set<std::pair<size_t, size_t>>();
    bool maximal = true;
    while (sv <= p and sw >= 0) {
      if (_reject(rowIdx, sv, sw, cache)) {
        if (maximal) {
          pairs.emplace(sv - 1, sw);
          maximal = false;
        }
        sw -= 1;
      } else {
        sv += 1;
        maximal = true;
      }
    }
    if (sv == p + 1)
      pairs.emplace(sv - 1, sw);
    return pairs;
  }

  std::pair<Row, Row> _vecProximalMap(size_t rowIdx) const {
    auto &ppX = _pX[rowIdx];
    auto &ppY = _pY[rowIdx];

    double optVal = std::numeric_limits<double>::infinity();
    Cache vWCache;
    auto optIterator = vWCache.end();

    for (auto &svw : _sparsityPairsMFB(rowIdx, vWCache)) {
      auto search = vWCache.find(svw);
      if (search == vWCache.end())
        std::tie(search, std::ignore) =
            vWCache.insert({svw, _vwStationary(rowIdx, svw.first, svw.second)});
      const Row &v = search->second.first;
      const Row &w = search->second.second;
      double val = _proxObj(ppX.abs(), ppY.abs(), v, w, _lambda);
      if (val < optVal) {
        optVal = val;
        optIterator = search;
      }
    }

    Row &v = optIterator->second.first;
    Row &w = optIterator->second.second;

    if (_proxObj(ppX.abs(), ppY.abs(), _v0, w, _lambda) < optVal)
      return {_v0, std::move(w)};

    if (_proxObj(ppX.abs(), ppY.abs(), v, _w0, _lambda) < optVal)
      return {std::move(v), _w0};

    return {std::move(v), std::move(w)};
  }

  static std::vector<ProcessRow> _preprocessRows(const Array2D &m) {
    std::vector<ProcessRow> preprocessedRows;
    for (size_t i = 0; i < m.rows(); ++i)
      preprocessedRows.emplace_back(m.row(i));
    return preprocessedRows;
  }

public:
  Array2D mV;
  Array2D mW;

  PathNormProximalMap() = delete;

  PathNormProximalMap(Array2D inX, Array2D inY, double in_lambda)
      : _mX(inX), _mY(inY), _lambda(in_lambda), _lambda2(pow(in_lambda, 2)),
        _pX(_preprocessRows(inX)), _pY(_preprocessRows(inY)),
        _v0(Row::Zero(inX.cols())), _w0(Row::Zero(inY.cols())) {
    eigen_assert(_mX.rows() == _mY.rows());
    mV = Array2D(_mX.rows(), _mX.cols());
    mW = Array2D(_mY.rows(), _mY.cols());
  }

  void run() {
    for (size_t i = 0; i < _mX.rows(); ++i) {
      auto [v, w] = _vecProximalMap(i);
      mV.row(i) = _pX[i].postprocessRow(v);
      mW.row(i) = _pY[i].postprocessRow(w);
    }
  }
};
