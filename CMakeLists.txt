cmake_minimum_required(VERSION 4.0.0)

project(LiteNN CXX)

file(GLOB_RECURSE LITENN_SOURCES CONFIGURE_DEPENDS src/*.cpp src/*.h)
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

include(CTest)
include(GoogleTest)

enable_testing()

# gtest 1.16.0 使用了 C++26 中已移除的 std::wstring_convert 和 std::codecvt，
# 需要禁用相关功能以在 C++26 模式下编译
add_compile_definitions(GTEST_HAS_STD_WSTRING=0 GTEST_HAS_FILE_SYSTEM=0)
set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
add_subdirectory(third_party/googletest)

# clang-p2996 的 c++.dll 需要在 PATH 中才能运行测试
get_filename_component(COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
set(TEST_PATH_ENV "PATH=${COMPILER_BIN_DIR}")

file(GLOB TEST_SOURCES CONFIGURE_DEPENDS "tests/*.cpp")

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_link_libraries(${TEST_NAME} PRIVATE LiteNN GTest::gtest_main)
    gtest_add_tests(TARGET ${TEST_NAME} TEST_LIST ${TEST_NAME}_TESTS)
    set_tests_properties(${${TEST_NAME}_TESTS} PROPERTIES ENVIRONMENT "${TEST_PATH_ENV}")
endforeach()
