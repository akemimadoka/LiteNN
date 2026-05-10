cmake_minimum_required(VERSION 4.0.0)

project(LiteNN CXX)

option(LITENN_ENABLE_MLIR "Enable MLIR-based AOT compiler backend" OFF)
option(LITENN_BUILD_EXAMPLES "Build LiteNN example programs" ON)
option(LITENN_BUILD_BENCHMARKS "Build LiteNN benchmark programs" OFF)

file(GLOB_RECURSE LITENN_SOURCES CONFIGURE_DEPENDS src/*.cpp src/*.h)
list(FILTER LITENN_SOURCES EXCLUDE REGEX "src/LiteNN/Compiler/")
add_library(LiteNN ${LITENN_SOURCES})

target_include_directories(LiteNN PUBLIC src)
target_sources(LiteNN PUBLIC FILE_SET modules TYPE CXX_MODULES FILES
    src/LiteNN.ixx
)

target_compile_features(LiteNN PUBLIC cxx_std_26)
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    target_compile_options(LiteNN PUBLIC -freflection-latest)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(LiteNN PUBLIC -freflection)
endif()

if(LITENN_ENABLE_MLIR)
    find_package(LLVM REQUIRED)
    find_package(MLIR REQUIRED)

    list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}" "${LLVM_CMAKE_DIR}")
    include(TableGen)
    include(AddLLVM)
    include(AddMLIR)

    add_subdirectory(src/LiteNN/Compiler)
endif()

include(CTest)
include(GoogleTest)

enable_testing()

# gtest 1.16.0 使用了 C++26 中已移除的 std::wstring_convert 和 std::codecvt，
# 需要禁用相关功能以在 C++26 模式下编译
add_compile_definitions(GTEST_HAS_STD_WSTRING=0 GTEST_HAS_FILE_SYSTEM=0)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_subdirectory(third_party/googletest)

if(LITENN_BUILD_EXAMPLES)
    add_subdirectory(example/mnist)
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
    list(FILTER TEST_SOURCES EXCLUDE REGEX "(CompilerTest|LoweringPassTest|BufferizationPassTest|LLVMCodegenPassTest|CompiledModuleTest)\\.cpp$")
endif()

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    if(TEST_NAME STREQUAL "CompilerTest" OR TEST_NAME STREQUAL "LoweringPassTest"
       OR TEST_NAME STREQUAL "BufferizationPassTest" OR TEST_NAME STREQUAL "LLVMCodegenPassTest"
       OR TEST_NAME STREQUAL "CompiledModuleTest")
        target_link_libraries(${TEST_NAME} PRIVATE LiteNNCompiler GTest::gtest_main)
    else()
        target_link_libraries(${TEST_NAME} PRIVATE LiteNN GTest::gtest_main)
    endif()
    gtest_add_tests(TARGET ${TEST_NAME} TEST_LIST ${TEST_NAME}_TESTS)
endforeach()
