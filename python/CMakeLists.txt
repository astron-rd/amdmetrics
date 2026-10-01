find_package(pybind11 CONFIG REQUIRED)

pybind11_add_module(amdametrics pyamdmetrics.cpp)
target_link_libraries(amdametrics PRIVATE amd_metrics)
target_include_directories(amdametrics PRIVATE ${CMAKE_SOURCE_DIR}/lib)
