#pragma once

#include <cmath>

#include <numeric>
#include <algorithm>

#include <vector>

#include "Eigen/Dense"

using Eigen::Array, Eigen::Dynamic;

typedef Array<double, 1, Dynamic> Row;
typedef Array<double, Dynamic, Dynamic> Array2D;

class PathNormProximalMap {
private:
    struct PreprocessedRow {
        Row sign;
        Row abs;
        Row cumsum;
        std::vector<size_t> reorder;
    };


    const Array2D mX;
    const Array2D mY;
    const double lambda;

    template<typename Row>
    static std::vector<size_t> _sort_indexes(const Row &v) {
        // Credits: https://stackoverflow.com/a/12399290

        // initialize original index locations
        std::vector<size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

        // sort indexes based on comparing values in v
        // using std::stable_sort instead of std::sort
        // to avoid unnecessary index re-orderings
        // when v contains elements of equal values
        std::sort(idx.begin(), idx.end(),
                  [&v](size_t i1, size_t i2) { return v[i1] < v[i2]; });

        return idx;
    }

    template<typename RowType>
    static PreprocessedRow _preprocessRow(const RowType &row) {
        // In Preprocessing:
        // The order of the operation matters:
        // 1. Sign
        // 2. then Order
        // 3. then Abs
        // 4. then Cumsum
        // Thus, Sign recovery should be taken after recovering the order of elements.

        PreprocessedRow r;
        // 1. Sign
        r.sign = Eigen::sign(row);

        // 2. Order Descending-ly w.r.t. absolute vals & Cache Re-ordering indices.
        auto abs = Eigen::abs(row);
        auto sorter = _sort_indexes(-abs);
        r.reorder = _sort_indexes(sorter);

        // 3. Take the Absolute of the ordered values.
        r.abs = abs(sorter);

        // 4. Cumulative Sum
        r.cumsum = Row(r.abs);
        for (size_t i = 1; i < r.cumsum.size(); ++i)
            r.cumsum(i) += r.cumsum(i - 1);
        return r;
    }

    std::vector<PreprocessedRow> _preprocessParams() {
        std::vector<PreprocessedRow> rowsX, rowsY;
        for (size_t i = 0; i < mX.rows(); ++i)
        {
            rowsX.push_back(_preprocessRow(mX.row(i)));
            rowsY.push_back(_preprocessRow(mY.row(i)));
        }




    }

public:

    Array2D mU;
    Array2D mV;

    PathNormProximalMap() = delete;

    PathNormProximalMap(Array2D inX, Array2D inY, double in_lambda) :
            mX(inX), mY(inY), lambda(in_lambda) {
        eigen_assert(mX.rows() == mY.rows());
    }


    void run() {
    }
};
