cmake_minimum_required(VERSION 3.20)

project(amd_metrics LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(rocprofiler-sdk REQUIRED)
find_package(HIP REQUIRED)

add_subdirectory(lib)
add_subdirectory(demo)
add_subdirectory(python)
