cmake_minimum_required(VERSION 3.15)
project(spheroids)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_VERBOSE_MAKEFILE ON)

find_package(pybind11 REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

pybind11_add_module(_pkbd src/spheroids/cpp/_pkbd.cpp)
pybind11_add_module(_scauchy src/spheroids/cpp/_scauchy.cpp)
pybind11_add_module(_rpkbd src/spheroids/cpp/_rpkbd.cpp)

target_link_libraries(_pkbd PRIVATE Eigen3::Eigen)
target_link_libraries(_scauchy PRIVATE Eigen3::Eigen)
target_link_libraries(_rpkbd PRIVATE Eigen3::Eigen)

install(TARGETS _pkbd _scauchy _rpkbd
        LIBRARY DESTINATION src/spheroids/cpp)
