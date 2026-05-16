find_package(Threads REQUIRED)

set(LITENN_GGML_VENDOR_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/llama.cpp/ggml)
set(LITENN_GGML_VENDOR_SOURCES
    ${LITENN_GGML_VENDOR_DIR}/src/ggml.c
    ${LITENN_GGML_VENDOR_DIR}/src/ggml.cpp
    ${LITENN_GGML_VENDOR_DIR}/src/ggml-alloc.c
    ${LITENN_GGML_VENDOR_DIR}/src/ggml-backend.cpp
    ${LITENN_GGML_VENDOR_DIR}/src/ggml-backend-meta.cpp
    ${LITENN_GGML_VENDOR_DIR}/src/ggml-opt.cpp
    ${LITENN_GGML_VENDOR_DIR}/src/ggml-threading.cpp
    ${LITENN_GGML_VENDOR_DIR}/src/ggml-quants.c
    ${LITENN_GGML_VENDOR_DIR}/src/gguf.cpp
)

add_library(LiteNNGGMLVendor STATIC ${LITENN_GGML_VENDOR_SOURCES})
target_include_directories(LiteNNGGMLVendor
    PUBLIC
        ${LITENN_GGML_VENDOR_DIR}/include
    PRIVATE
        ${LITENN_GGML_VENDOR_DIR}/src
)
target_compile_definitions(LiteNNGGMLVendor PRIVATE
    GGML_VERSION="LiteNN-vendored"
    GGML_COMMIT="vendored"
)
target_link_libraries(LiteNNGGMLVendor PRIVATE Threads::Threads)
if(WIN32)
    target_compile_definitions(LiteNNGGMLVendor PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
if(CMAKE_SYSTEM_NAME MATCHES "Linux")
    target_link_libraries(LiteNNGGMLVendor PRIVATE dl)
endif()
if(APPLE)
    target_compile_definitions(LiteNNGGMLVendor PUBLIC _DARWIN_C_SOURCE)
endif()
if(NOT WIN32)
    find_library(LITENN_GGML_MATH_LIBRARY m)
    if(LITENN_GGML_MATH_LIBRARY)
        target_link_libraries(LiteNNGGMLVendor PRIVATE ${LITENN_GGML_MATH_LIBRARY})
    endif()
endif()
litenn_enable_sanitizers(LiteNNGGMLVendor)

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
litenn_enable_sanitizers(litenn_gguf_convert)

install(TARGETS litenn_gguf_convert
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)