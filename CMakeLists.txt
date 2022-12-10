cmake_minimum_required(VERSION 3.4...3.18)
project(proxmap_pathnorm)

set(PYBIND11_PYTHON_VERSION 3.8.5 CACHE STRING "")
add_subdirectory(pybind11)

#pybind11_add_module(proxmap_pathnorm src/main.cpp)
#target_link_libraries(proxmap_pathnorm PRIVATE pybind11::module pybind11::lto pybind11::windows_extras)

add_executable(proxmap_pathnorm src/main.cpp)
target_include_directories(proxmap_pathnorm PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
        "${CMAKE_SOURCE_DIR}/third_party/eigen")

