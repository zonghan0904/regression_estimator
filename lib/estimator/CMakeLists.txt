cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

set(CMAKE_PREFIX_PATH "/usr/local/lib/libtorch/share/cmake/Torch")
find_package(Torch REQUIRED)

add_library(estimator SHARED ./estimator.cpp ./estimator.h)
target_link_libraries(estimator "${TORCH_LIBRARIES}")
set_property(TARGET estimator PROPERTY CXX_STANDARD 14)
