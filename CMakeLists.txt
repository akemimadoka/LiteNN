cmake_minimum_required(VERSION 4.0.0)

project(LiteNN VERSION 0.1.0 LANGUAGES C CXX)

option(LITENN_ENABLE_MLIR "Enable MLIR-based AOT compiler backend" OFF)
option(LITENN_ENABLE_CUDA "Enable CUDA runtime device backend" OFF)
option(LITENN_BUILD_EXAMPLES "Build LiteNN example programs" ON)
option(LITENN_BUILD_BENCHMARKS "Build LiteNN benchmark programs" OFF)
option(LITENN_ENABLE_SANITIZERS "Enable sanitizer instrumentation for LiteNN and its tests" OFF)
option(LITENN_WITH_MODULES "Build LiteNN with C++20 modules support" OFF)
set(LITENN_SANITIZERS "address,undefined" CACHE STRING "Comma-separated list of sanitizers to enable")

include(CheckCXXSourceCompiles)
include(CMakePushCheckState)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
find_program(LITENN_PYTHON_EXECUTABLE NAMES python python3 REQUIRED)

set(LITENN_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/LiteNN" CACHE STRING
    "Install path for LiteNN CMake package files")

function(litenn_parse_sanitizers out_var)
    string(REPLACE "," ";" parsed_sanitizers "${LITENN_SANITIZERS}")
    list(TRANSFORM parsed_sanitizers STRIP)
    list(FILTER parsed_sanitizers EXCLUDE REGEX "^$")
    if(LITENN_ENABLE_SANITIZERS AND NOT parsed_sanitizers)
        message(FATAL_ERROR "LITENN_ENABLE_SANITIZERS requires at least one entry in LITENN_SANITIZERS")
    endif()
    set(${out_var} "${parsed_sanitizers}" PARENT_SCOPE)
endfunction()

function(litenn_detect_msys2_mingw out_var)
    set(is_msys2_mingw FALSE)
    if(MINGW)
        if((DEFINED ENV{MSYSTEM} AND NOT "$ENV{MSYSTEM}" STREQUAL "")
           OR CMAKE_C_COMPILER MATCHES [[(^|[\\/])msys64([\\/])]]
           OR CMAKE_CXX_COMPILER MATCHES [[(^|[\\/])msys64([\\/])]])
            set(is_msys2_mingw TRUE)
        endif()
    endif()
    set(${out_var} ${is_msys2_mingw} PARENT_SCOPE)
endfunction()

function(litenn_enable_sanitizers target)
    if(NOT LITENN_ENABLE_SANITIZERS)
        return()
    endif()

    litenn_detect_msys2_mingw(using_msys2_mingw)
    if(using_msys2_mingw)
        message(FATAL_ERROR
            "LiteNN sanitizers are not supported with MSYS2 MinGW toolchains because the required sanitizer runtimes are not shipped there. Disable LITENN_ENABLE_SANITIZERS or use a supported environment such as WSL/Linux or MSVC AddressSanitizer.\n"
            "See https://github.com/msys2/MINGW-packages/issues/3163")
    endif()

    litenn_parse_sanitizers(requested_sanitizers)
    set(sanitizer_compile_options)
    set(sanitizer_link_options)

    if(MSVC)
        foreach(sanitizer IN LISTS requested_sanitizers)
            if(sanitizer STREQUAL "address")
                list(APPEND sanitizer_compile_options /fsanitize=address /Zi)
            elseif(sanitizer STREQUAL "undefined" OR sanitizer STREQUAL "leak")
                message(WARNING
                    "MSVC does not support the '${sanitizer}' sanitizer; falling back to AddressSanitizer plus Debug CRT leak checks where available")
            else()
                message(FATAL_ERROR "Unsupported sanitizer '${sanitizer}' for MSVC builds")
            endif()
        endforeach()
    else()
        string(JOIN "," joined_sanitizers ${requested_sanitizers})
        list(APPEND sanitizer_compile_options
            -fno-omit-frame-pointer
            "-fsanitize=${joined_sanitizers}"
            -fno-sanitize-recover=all
        )
        list(APPEND sanitizer_link_options "-fsanitize=${joined_sanitizers}")
    endif()

    list(REMOVE_DUPLICATES sanitizer_compile_options)
    list(REMOVE_DUPLICATES sanitizer_link_options)

    get_property(sanitizer_support_checked GLOBAL PROPERTY LITENN_SANITIZER_SUPPORT_CHECKED)
    if(NOT sanitizer_support_checked)
        cmake_push_check_state(RESET)
        string(REPLACE ";" " " CMAKE_REQUIRED_FLAGS "${sanitizer_compile_options}")
        set(CMAKE_REQUIRED_LINK_OPTIONS ${sanitizer_link_options})
        check_cxx_source_compiles("int main() { return 0; }" LITENN_SANITIZER_TOOLCHAIN_SUPPORTED)
        cmake_pop_check_state()
        if(NOT LITENN_SANITIZER_TOOLCHAIN_SUPPORTED)
            message(FATAL_ERROR
                "Requested sanitizers '${LITENN_SANITIZERS}' are not supported by the current C++ toolchain. Ensure the compiler and sanitizer runtime libraries are installed, or choose a supported subset.")
        endif()
        set_property(GLOBAL PROPERTY LITENN_SANITIZER_SUPPORT_CHECKED TRUE)
    endif()

    if(sanitizer_compile_options)
        target_compile_options(${target} PUBLIC ${sanitizer_compile_options})
    endif()
    if(sanitizer_link_options)
        target_link_options(${target} PUBLIC ${sanitizer_link_options})
    endif()

    target_compile_definitions(${target} PUBLIC LITENN_ENABLE_SANITIZERS=1)
endfunction()

function(litenn_configure_sanitizer_test_environment)
    if(NOT LITENN_ENABLE_SANITIZERS)
        return()
    endif()

    if(MSVC)
        return()
    endif()

    litenn_parse_sanitizers(requested_sanitizers)
    set(test_environment)

    list(FIND requested_sanitizers "address" has_address_sanitizer)
    list(FIND requested_sanitizers "leak" has_leak_sanitizer)
    list(FIND requested_sanitizers "undefined" has_undefined_sanitizer)

    if(NOT has_address_sanitizer EQUAL -1 OR NOT has_leak_sanitizer EQUAL -1)
        list(APPEND test_environment
            "ASAN_OPTIONS=detect_leaks=1:strict_string_checks=1"
            "LSAN_OPTIONS=report_objects=1"
        )
    endif()
    if(NOT has_undefined_sanitizer EQUAL -1)
        list(APPEND test_environment "UBSAN_OPTIONS=print_stacktrace=1")
    endif()

    if(test_environment AND ARGN)
        set_tests_properties(${ARGN} PROPERTIES ENVIRONMENT "${test_environment}")
    endif()
endfunction()

file(GLOB_RECURSE LITENN_SOURCES CONFIGURE_DEPENDS src/*.cpp src/*.h)
list(FILTER LITENN_SOURCES EXCLUDE REGEX "src/LiteNN/Compiler/|.*\\.module\\.cpp$")
add_library(LiteNN ${LITENN_SOURCES})

add_library(LiteNN::LiteNN ALIAS LiteNN)

target_include_directories(LiteNN PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_features(LiteNN PUBLIC cxx_std_26)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(LiteNN PUBLIC -freflection-latest)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(LiteNN PUBLIC -freflection)
endif()
litenn_enable_sanitizers(LiteNN)

if(LITENN_ENABLE_CUDA)
    find_package(CUDAToolkit REQUIRED)
    target_compile_definitions(LiteNN PUBLIC LITENN_ENABLE_CUDA=1)
    target_link_libraries(LiteNN PUBLIC CUDA::cudart CUDA::cublas)
    if(TARGET CUDA::cuda_driver)
        target_compile_definitions(LiteNN PUBLIC LITENN_ENABLE_CUDA_DRIVER=1)
        target_link_libraries(LiteNN PUBLIC CUDA::cuda_driver)
    else()
        message(WARNING "CUDA::cuda_driver target is unavailable; CUDA native AOT driver runtime will be disabled")
    endif()
endif()

set(LITENN_HEADERS ${LITENN_SOURCES})
list(FILTER LITENN_HEADERS EXCLUDE REGEX "src/.*\\.cpp$")
set(LITENN_MODULE_HEADERS ${LITENN_HEADERS})

if(LITENN_ENABLE_MLIR)
    add_compile_definitions(LITENN_ENABLE_MLIR=1)
    find_package(LLVM CONFIG REQUIRED)
    find_package(MLIR CONFIG REQUIRED)

    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)

    add_subdirectory(src/LiteNN/Compiler)
    litenn_enable_sanitizers(LiteNNCompiler)

    get_target_property(LITENN_COMPILER_HEADERS LiteNNCompiler SOURCES)
    list(FILTER LITENN_COMPILER_HEADERS EXCLUDE REGEX ".*\\.cpp$")
    list(APPEND LITENN_HEADERS ${LITENN_COMPILER_HEADERS})
endif()

set(LITENN_GENERATED_MODULE_INTERFACE)
if(LITENN_WITH_MODULES)
    set(LITENN_GENERATED_MODULE_INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/src/LiteNN/LiteNN.ixx)
    add_custom_command(
        OUTPUT ${LITENN_GENERATED_MODULE_INTERFACE}
        COMMAND ${CMAKE_COMMAND} -E echo "Generating C++20 module interface for LiteNN"
        COMMAND ${LITENN_PYTHON_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/GenerateModule.py"
            --input-files ${LITENN_MODULE_HEADERS}
                --input-template "${CMAKE_CURRENT_SOURCE_DIR}/src/LiteNN.ixx.template"
                --output ${LITENN_GENERATED_MODULE_INTERFACE}
        DEPENDS ${LITENN_MODULE_HEADERS} "${CMAKE_CURRENT_SOURCE_DIR}/src/LiteNN.ixx.template" "${CMAKE_CURRENT_SOURCE_DIR}/scripts/GenerateModule.py"
        VERBATIM
    )
    target_sources(LiteNN PRIVATE
        src/LiteNN/Device.module.cpp
        src/LiteNN/Debug/Dump.module.cpp
    )
    target_sources(LiteNN PUBLIC
        FILE_SET litenn_modules TYPE CXX_MODULES
        BASE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/src/LiteNN
        FILES ${LITENN_GENERATED_MODULE_INTERFACE}
    )
endif()

set(LITENN_RUNTIME_INSTALL_EXTRA_ARGS)
if(LITENN_WITH_MODULES)
    list(APPEND LITENN_RUNTIME_INSTALL_EXTRA_ARGS
        FILE_SET litenn_modules DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
    )
endif()

install(TARGETS LiteNN
    EXPORT LiteNNRuntimeTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ${LITENN_RUNTIME_INSTALL_EXTRA_ARGS}
)

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/LiteNN/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/LiteNN
    FILES_MATCHING PATTERN "*.h"
)
install(FILES src/LiteNN.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES README.md LICENSE DESTINATION ${CMAKE_INSTALL_DATADIR}/LiteNN)

include(CTest)
include(GoogleTest)

enable_testing()

# gtest 1.16.0 使用了 C++26 中已移除的 std::wstring_convert 和 std::codecvt，
# 需要禁用相关功能以在 C++26 模式下编译
add_compile_definitions(GTEST_HAS_STD_WSTRING=0 GTEST_HAS_FILE_SYSTEM=0)
set(INSTALL_GTEST OFF CACHE BOOL "Disable installation of bundled googletest" FORCE)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_subdirectory(third_party/googletest)

if(LITENN_BUILD_EXAMPLES)
    add_subdirectory(example/mnist)
    add_subdirectory(example/carrier)
endif()

if(LITENN_BUILD_BENCHMARKS)
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable bundled benchmark tests" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable bundled benchmark gtest tests" FORCE)
    set(BENCHMARK_ENABLE_WERROR OFF CACHE BOOL "Disable bundled benchmark -Werror" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable bundled benchmark install targets" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES OFF CACHE BOOL "Disable bundled benchmark dependency downloads" FORCE)
    add_subdirectory(third_party/benchmark EXCLUDE_FROM_ALL)
    add_subdirectory(benchmark)
endif()

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "tests/*.cpp")
if(NOT LITENN_ENABLE_MLIR)
    list(FILTER TEST_SOURCES EXCLUDE REGEX "(CompilerTest|LoweringPassTest|BufferizationPassTest|LLVMCodegenPassTest|CompiledModuleTest|CompiledModuleCUDATest|CompilerDumpTest)\\.cpp$")
endif()
if(NOT LITENN_ENABLE_CUDA)
    list(FILTER TEST_SOURCES EXCLUDE REGEX "(CUDADeviceTest|CompiledModuleCUDATest)\\.cpp$")
endif()

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    if(LITENN_ENABLE_MLIR)
        target_compile_definitions(${TEST_NAME} PRIVATE LITENN_ENABLE_MLIR=1)
    endif()
    if(TEST_NAME STREQUAL "CompilerTest" OR TEST_NAME STREQUAL "LoweringPassTest"
       OR TEST_NAME STREQUAL "BufferizationPassTest" OR TEST_NAME STREQUAL "LLVMCodegenPassTest"
       OR TEST_NAME STREQUAL "CompiledModuleTest" OR TEST_NAME STREQUAL "CompiledModuleCUDATest"
       OR TEST_NAME STREQUAL "CompilerDumpTest"
       OR (LITENN_ENABLE_MLIR AND TEST_NAME STREQUAL "MemorySafetyTest"))
        target_link_libraries(${TEST_NAME} PRIVATE LiteNNCompiler GTest::gtest_main)
    else()
        target_link_libraries(${TEST_NAME} PRIVATE LiteNN GTest::gtest_main)
    endif()
    gtest_add_tests(TARGET ${TEST_NAME} TEST_LIST ${TEST_NAME}_TESTS)
    litenn_configure_sanitizer_test_environment(${${TEST_NAME}_TESTS})
    if(TEST_NAME STREQUAL "MemorySafetyTest")
        set_tests_properties(${${TEST_NAME}_TESTS} PROPERTIES LABELS "memory-safety;sanitizer")
    endif()
endforeach()

if(LITENN_ENABLE_MLIR)
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/src/LiteNN/Compiler/
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/LiteNN/Compiler
        FILES_MATCHING PATTERN "*.h"
    )
    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/src/LiteNN/Compiler/Dialect/LiteNNDialect.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/src/LiteNN/Compiler/Dialect/LiteNNEnums.h.inc
        ${CMAKE_CURRENT_BINARY_DIR}/src/LiteNN/Compiler/Dialect/LiteNNOps.h.inc
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/LiteNN/Compiler/Dialect
    )
endif()

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/LiteNNConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/LiteNNConfig.cmake
    INSTALL_DESTINATION ${LITENN_INSTALL_CMAKEDIR}
)

set(LITENN_CMAKE_PACKAGE_COMPATIBILITY SameMajorVersion)
if(PROJECT_VERSION_MAJOR EQUAL 0)
    set(LITENN_CMAKE_PACKAGE_COMPATIBILITY SameMinorVersion)
endif()

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/LiteNNConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY ${LITENN_CMAKE_PACKAGE_COMPATIBILITY}
)

if(LITENN_WITH_MODULES)
    install(EXPORT LiteNNRuntimeTargets
        FILE LiteNNTargets.cmake
        NAMESPACE LiteNN::
        DESTINATION ${LITENN_INSTALL_CMAKEDIR}
        CXX_MODULES_DIRECTORY modules
    )
else()
    install(EXPORT LiteNNRuntimeTargets
        FILE LiteNNTargets.cmake
        NAMESPACE LiteNN::
        DESTINATION ${LITENN_INSTALL_CMAKEDIR}
    )
endif()

if(LITENN_ENABLE_MLIR)
    install(EXPORT LiteNNCompilerTargets
        FILE LiteNNCompilerTargets.cmake
        NAMESPACE LiteNN::
        DESTINATION ${LITENN_INSTALL_CMAKEDIR}
    )
endif()

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/LiteNNConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/LiteNNConfigVersion.cmake
    DESTINATION ${LITENN_INSTALL_CMAKEDIR}
)
