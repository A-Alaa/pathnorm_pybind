cmake_minimum_required(VERSION 3.4...3.18)
project(proxmap_pathnorm)

set(PYBIND11_PYTHON_VERSION 3.8.5 CACHE STRING "")
add_subdirectory(pybind11)

add_library(proxmap_pathnorm MODULE src/main.cpp)
target_link_libraries(proxmap_pathnorm PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)
target_include_directories(proxmap_pathnorm PRIVATE "${CMAKE_SOURCE_DIR}/third_party/eigen")

pybind11_extension(proxmap_pathnorm)
pybind11_strip(proxmap_pathnorm)


# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.

set_target_properties(proxmap_pathnorm PROPERTIES CXX_VISIBILITY_PRESET "hidden"
                                         CUDA_VISIBILITY_PRESET "hidden")
target_compile_definitions(proxmap_pathnorm
                           PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})
