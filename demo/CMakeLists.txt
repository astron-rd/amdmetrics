add_executable(demo demo.cpp)
set_source_files_properties(demo.cpp PROPERTIES LANGUAGE HIP)
target_link_libraries(demo PRIVATE amd_metrics)
target_include_directories(demo PRIVATE ${CMAKE_SOURCE_DIR}/lib)
