add_library(amd_metrics SHARED amd_metrics.h amd_metrics.cpp)
target_include_directories(amd_metrics PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(amd_metrics PRIVATE rocprofiler-sdk::rocprofiler-sdk)
set_target_properties(amd_metrics PROPERTIES BUILD_RPATH ${ROCM_PATH}/lib)
