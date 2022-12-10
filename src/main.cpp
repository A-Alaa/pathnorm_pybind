//#include "pybind11/pybind11.h"
//#include "../include/pathnorm/pathnorm.hpp"
//
//namespace py = pybind11;
//constexpr auto byref = py::return_value_policy::reference_internal;
//
//PYBIND11_MODULE(proxmap_pathnorm, m) {
//    m.doc() = "optional module docstring";
//
//    py::class_<MyClass>(m, "MyClass")
//    .def(py::init<double, double, int>())
//    .def("run", &MyClass::run, py::call_guard<py::gil_scoped_release>())
//    .def_readonly("v_data", &MyClass::v_data, byref)
//    .def_readonly("v_gamma", &MyClass::v_gamma, byref)
//    ;
//}

#include <iostream>
#include <numeric>
#include <iterator>

#include "Eigen/Eigen"
#include "pathnorm/pathnorm.hpp"


int main(){
    Array2D x = Array2D::Random(8, 8);
    std::cout << x.rows() << std::endl;

    std::cout << x << std::endl;
    x.row(0) = Eigen::abs(x.row(0));
    std::cout << x << std::endl;
    auto r = x.row(0).array();
    std::cout << r << std::endl;
}