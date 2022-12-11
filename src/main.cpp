#include "pathnorm/pathnorm.hpp"
#include "pybind11/pybind11.h"

namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;

PYBIND11_MODULE(proxmap_pathnorm, m) {
  m.doc() = "optional module docstring";

  py::class_<PathNormProximalMap>(m, "PathNormProximalMap")
      .def(py::init<Array2D, Array2D, double>())
      .def("run", &PathNormProximalMap::run,
           py::call_guard<py::gil_scoped_release>())
      .def_readonly("V", &PathNormProximalMap::mV, byref)
      .def_readonly("W", &PathNormProximalMap::mW, byref);
}