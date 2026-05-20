include(${PROJECT_SOURCE_DIR}/cmake/LiteNNGGMLVendor.cmake)
litenn_ensure_ggml_vendor_target()

add_library(LiteNNGGUFImporter STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/GGUFImporter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/GGUFImporter.h
    ${CMAKE_CURRENT_SOURCE_DIR}/LLaMABuilder.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LLaMABuilder.h
)
target_compile_features(LiteNNGGUFImporter PUBLIC cxx_std_26)
target_include_directories(LiteNNGGUFImporter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(LiteNNGGUFImporter PUBLIC LiteNN PRIVATE LiteNNGGMLVendor)
litenn_enable_sanitizers(LiteNNGGUFImporter)

add_executable(litenn_gguf_convert ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp)
target_compile_features(litenn_gguf_convert PRIVATE cxx_std_26)
target_link_libraries(litenn_gguf_convert PRIVATE LiteNNGGUFImporter)
if(TARGET LiteNNCompiler)
    target_link_libraries(litenn_gguf_convert PRIVATE LiteNNCompiler)
    target_compile_definitions(litenn_gguf_convert PRIVATE LITENN_GGUF_CONVERT_ENABLE_AOT=1)
endif()
litenn_enable_sanitizers(litenn_gguf_convert)

install(TARGETS litenn_gguf_convert
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
