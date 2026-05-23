add_executable(litenn_sdxl_example main.cpp)
target_compile_features(litenn_sdxl_example PRIVATE cxx_std_26)
target_link_libraries(litenn_sdxl_example PRIVATE LiteNN)

if(TARGET LiteNNCompiler)
    target_link_libraries(litenn_sdxl_example PRIVATE LiteNNCompiler)
    target_compile_definitions(litenn_sdxl_example PRIVATE LITENN_ENABLE_MLIR=1)
endif()

if(CMAKE_DL_LIBS)
    target_link_libraries(litenn_sdxl_example PRIVATE ${CMAKE_DL_LIBS})
endif()
