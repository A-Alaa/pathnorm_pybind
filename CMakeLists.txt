cmake_minimum_required(VERSION 3.4...3.18)
project(pathnorm_proxmap)

set(PYBIND11_PYTHON_VERSION 3.8.5 CACHE STRING "")
add_subdirectory(third_party/pybind11)

pybind11_add_module(pathnorm_proxmap src/main.cpp)
target_link_libraries(pathnorm_proxmap PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)
target_include_directories(pathnorm_proxmap PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/third_party/eigen")
