#include "pybind11/pybind11.h"
#include "pathnorm.hpp"

namespace py = pybind11;
constexpr auto byref = py::return_value_policy::reference_internal;

PYBIND11_MODULE(proxmap_pathnorm, m) {
    m.doc() = "optional module docstring";

    py::class_<MyClass>(m, "MyClass")
    .def(py::init<double, double, int>())  
    .def("run", &MyClass::run, py::call_guard<py::gil_scoped_release>())
    .def_readonly("v_data", &MyClass::v_data, byref)
    .def_readonly("v_gamma", &MyClass::v_gamma, byref)
    ;
}
