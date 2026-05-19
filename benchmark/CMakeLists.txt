include(${PROJECT_SOURCE_DIR}/cmake/LiteNNGGMLVendor.cmake)
litenn_ensure_ggml_vendor_target()

add_executable(litenn_bench bench.cpp)
target_link_libraries(litenn_bench PRIVATE LiteNN LiteNNGGMLVendor benchmark::benchmark_main)

add_executable(litenn_bench_train bench_train.cpp)
target_link_libraries(litenn_bench_train PRIVATE LiteNN benchmark::benchmark_main)

if(TARGET LiteNNCompiler)
    target_link_libraries(litenn_bench PRIVATE LiteNNCompiler)
    target_compile_definitions(litenn_bench PRIVATE LITENN_BENCH_HAS_AOT)
    target_link_libraries(litenn_bench_train PRIVATE LiteNNCompiler)
    target_compile_definitions(litenn_bench_train PRIVATE LITENN_TRAIN_BENCH_HAS_AOT)

    add_executable(litenn_profile profile.cpp)
    target_link_libraries(litenn_profile PRIVATE LiteNN LiteNNCompiler)
else()
    message(STATUS "LiteNNCompiler not available; AOT benchmark disabled")
endif()
