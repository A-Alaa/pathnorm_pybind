cmake_minimum_required(VERSION 3.4...3.18)
project(cmake_example)


add_subdirectory(pybind11)

add_library(cmake_example MODULE src/main.cpp)
target_link_libraries(cmake_example PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)
target_include_directories(cmake_example PRIVATE "${CMAKE_SOURCE_DIR}/third_party/eigen/Eigen")

pybind11_extension(cmake_example)


# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.

target_compile_definitions(cmake_example
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
