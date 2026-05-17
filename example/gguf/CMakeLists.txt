if(NOT TARGET LiteNNGGUFImporter)
    message(STATUS "LiteNNGGUFImporter is not available; skipping GGUF examples")
    return()
endif()

add_executable(litenn_gguf_conversion_example conversion_example.cpp)
target_compile_features(litenn_gguf_conversion_example PRIVATE cxx_std_26)
target_link_libraries(litenn_gguf_conversion_example PRIVATE LiteNNGGUFImporter LiteNNGGMLVendor)

add_custom_target(litenn_gguf_example DEPENDS litenn_gguf_conversion_example)
