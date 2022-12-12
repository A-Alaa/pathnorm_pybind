#include "pathnorm/pathnorm.hpp"
#include "pybind11/pybind11.h"
#include "pybind11/eigen.h"

namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;

PYBIND11_MODULE(pathnorm_proxmap, m) {
  m.doc() = "optional module docstring";

  py::class_<PathNormProximalMap>(m, "PathNormProximalMap")
      .def(py::init<const Array2D &, const Array2D &, double>())
      .def("run", &PathNormProximalMap::run,
           py::call_guard<py::gil_scoped_release>())
      .def_readonly("V", &PathNormProximalMap::mV, byref)
      .def_readonly("W", &PathNormProximalMap::mW, byref);
}