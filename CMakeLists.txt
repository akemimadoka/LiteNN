cmake_minimum_required(VERSION 4.0.0)

project(LiteNN CXX)

add_library(LiteNN
    src/LiteNN.h
    src/LiteNN/Device.cpp
    src/LiteNN/Device.h
    src/LiteNN/Misc.h
    src/LiteNN/Operators.h
    src/LiteNN/Tensor.h
)
target_include_directories(LiteNN PUBLIC src)
target_sources(LiteNN PUBLIC FILE_SET modules TYPE CXX_MODULES FILES
    src/LiteNN.ixx
)

target_compile_features(LiteNN PUBLIC cxx_std_26)
target_compile_options(LiteNN PUBLIC -freflection-latest)

include(CTest)

file(GLOB TEST_SOURCES "tests/*.cpp")

foreach(TEST_SOURCE ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
    target_link_libraries(${TEST_NAME} PRIVATE LiteNN)
endforeach()
