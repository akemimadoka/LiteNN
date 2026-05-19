function(litenn_ensure_ggml_vendor_target)
    if(TARGET LiteNNGGMLVendor)
        return()
    endif()

    find_package(Threads REQUIRED)

    set(litenn_ggml_vendor_dir ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../third_party/llama.cpp/ggml)
    set(litenn_ggml_vendor_sources
        ${litenn_ggml_vendor_dir}/src/ggml.c
        ${litenn_ggml_vendor_dir}/src/ggml.cpp
        ${litenn_ggml_vendor_dir}/src/ggml-alloc.c
        ${litenn_ggml_vendor_dir}/src/ggml-backend.cpp
        ${litenn_ggml_vendor_dir}/src/ggml-backend-meta.cpp
        ${litenn_ggml_vendor_dir}/src/ggml-opt.cpp
        ${litenn_ggml_vendor_dir}/src/ggml-threading.cpp
        ${litenn_ggml_vendor_dir}/src/ggml-quants.c
        ${litenn_ggml_vendor_dir}/src/gguf.cpp
        ${litenn_ggml_vendor_dir}/src/ggml-cpu/ggml-cpu.c
        ${litenn_ggml_vendor_dir}/src/ggml-cpu/ggml-cpu.cpp
        ${litenn_ggml_vendor_dir}/src/ggml-cpu/repack.cpp
        ${litenn_ggml_vendor_dir}/src/ggml-cpu/hbm.cpp
        ${litenn_ggml_vendor_dir}/src/ggml-cpu/quants.c
        ${litenn_ggml_vendor_dir}/src/ggml-cpu/traits.cpp
        ${litenn_ggml_vendor_dir}/src/ggml-cpu/binary-ops.cpp
        ${litenn_ggml_vendor_dir}/src/ggml-cpu/unary-ops.cpp
        ${litenn_ggml_vendor_dir}/src/ggml-cpu/vec.cpp
        ${litenn_ggml_vendor_dir}/src/ggml-cpu/ops.cpp
    )

    if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(AMD64|amd64|x86_64|X86_64|x64|i[3-6]86)$")
        list(APPEND litenn_ggml_vendor_sources
            ${litenn_ggml_vendor_dir}/src/ggml-cpu/amx/amx.cpp
            ${litenn_ggml_vendor_dir}/src/ggml-cpu/amx/mmq.cpp
            ${litenn_ggml_vendor_dir}/src/ggml-cpu/arch/x86/quants.c
            ${litenn_ggml_vendor_dir}/src/ggml-cpu/arch/x86/repack.cpp
        )
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|AARCH64|aarch64|armv[0-9].*)$")
        list(APPEND litenn_ggml_vendor_sources
            ${litenn_ggml_vendor_dir}/src/ggml-cpu/arch/arm/quants.c
            ${litenn_ggml_vendor_dir}/src/ggml-cpu/arch/arm/repack.cpp
        )
    endif()

    add_library(LiteNNGGMLVendor STATIC ${litenn_ggml_vendor_sources})
    target_compile_features(LiteNNGGMLVendor PRIVATE c_std_11 cxx_std_17)
    target_include_directories(LiteNNGGMLVendor
        PUBLIC
            ${litenn_ggml_vendor_dir}/include
        PRIVATE
            ${litenn_ggml_vendor_dir}/src
            ${litenn_ggml_vendor_dir}/src/ggml-cpu
    )
    target_compile_definitions(LiteNNGGMLVendor PRIVATE
        GGML_VERSION="LiteNN-vendored"
        GGML_COMMIT="vendored"
        GGML_SCHED_MAX_COPIES=4
    )
    target_link_libraries(LiteNNGGMLVendor PRIVATE Threads::Threads)

    litenn_detect_msys2_mingw(using_msys2_mingw)
    if(using_msys2_mingw)
        target_compile_options(LiteNNGGMLVendor PRIVATE -Wa,-mbig-obj)
    endif()

    if(WIN32)
        target_compile_definitions(LiteNNGGMLVendor PRIVATE _CRT_SECURE_NO_WARNINGS)
    endif()
    if(CMAKE_SYSTEM_NAME MATCHES "Linux|Android")
        target_compile_definitions(LiteNNGGMLVendor PRIVATE _GNU_SOURCE _XOPEN_SOURCE=600)
        target_link_libraries(LiteNNGGMLVendor PRIVATE dl)
    endif()
    if(APPLE)
        target_compile_definitions(LiteNNGGMLVendor PUBLIC _DARWIN_C_SOURCE)
    endif()
    if(NOT WIN32)
        find_library(litenn_ggml_math_library m)
        if(litenn_ggml_math_library)
            target_link_libraries(LiteNNGGMLVendor PRIVATE ${litenn_ggml_math_library})
        endif()
    endif()

    litenn_enable_sanitizers(LiteNNGGMLVendor)
endfunction()