add_executable(litenn_bench bench.cpp)
target_link_libraries(litenn_bench PRIVATE LiteNN)

if(TARGET LiteNNCompiler)
    target_link_libraries(litenn_bench PRIVATE LiteNNCompiler)
    target_compile_definitions(litenn_bench PRIVATE LITENN_BENCH_HAS_AOT)

    add_executable(litenn_profile profile.cpp)
    target_link_libraries(litenn_profile PRIVATE LiteNN LiteNNCompiler)
else()
    message(STATUS "LiteNNCompiler not available; AOT benchmark disabled")
endif()
