cmake_minimum_required(VERSION 3.12)
project(pilot_wrapper)

set(CMAKE_CXX_STANDARD 17)
set(pybind11_DIR /home/rowan/.pyenv/versions/3.10.13/lib/python3.10/site-packages/pybind11/share/cmake/pybind11)

set(Python3_INCLUDE_DIR /home/rowan/.pyenv/versions/3.10.13/include/python3.10)
set(Python3_NumPy_INCLUDE_DIR /home/rowan/.pyenv/versions/3.10.13/lib/python3.10/site-packages/numpy/core/include)


find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 REQUIRED)
find_package(Armadillo REQUIRED)
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
find_package(carma CONFIG REQUIRED)

# Set the output directory for the compiled module
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/pilot)

pybind11_add_module(cpilot pilot_wrapper.cpp tree.cpp)

target_link_libraries(cpilot PRIVATE carma::carma ${ARMADILLO_LIBRARIES} ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} pybind11::module)

