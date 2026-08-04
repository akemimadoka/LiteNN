cmake_minimum_required(VERSION 3.20)

project(LiteNNLlamaCppStageProfile LANGUAGES CXX)

set(LLAMA_CPP_SOURCE_DIR "" CACHE PATH "Path to the llama.cpp source tree")
set(LLAMA_CPP_BUILD_DIR "" CACHE PATH "Path to a completed llama.cpp build tree")

if(NOT IS_DIRECTORY "${LLAMA_CPP_SOURCE_DIR}/include" OR
   NOT IS_DIRECTORY "${LLAMA_CPP_SOURCE_DIR}/ggml/include")
    message(FATAL_ERROR "LLAMA_CPP_SOURCE_DIR must point to a llama.cpp source tree")
endif()
if(NOT IS_DIRECTORY "${LLAMA_CPP_BUILD_DIR}")
    message(FATAL_ERROR "LLAMA_CPP_BUILD_DIR must point to a completed llama.cpp build tree")
endif()

find_library(LLAMA_LIBRARY llama PATHS "${LLAMA_CPP_BUILD_DIR}/src" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_LIBRARY ggml PATHS "${LLAMA_CPP_BUILD_DIR}/ggml/src" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_CPU_LIBRARY ggml-cpu PATHS "${LLAMA_CPP_BUILD_DIR}/ggml/src" NO_DEFAULT_PATH REQUIRED)
find_library(GGML_BASE_LIBRARY ggml-base PATHS "${LLAMA_CPP_BUILD_DIR}/ggml/src" NO_DEFAULT_PATH REQUIRED)
find_package(Threads REQUIRED)

add_executable(litenn_llama_cpp_stage_profile main.cpp)
target_compile_features(litenn_llama_cpp_stage_profile PRIVATE cxx_std_17)
target_include_directories(
    litenn_llama_cpp_stage_profile
    PRIVATE
        "${LLAMA_CPP_SOURCE_DIR}/include"
        "${LLAMA_CPP_SOURCE_DIR}/ggml/include"
)
target_link_libraries(
    litenn_llama_cpp_stage_profile
    PRIVATE
        "${LLAMA_LIBRARY}"
        "${GGML_LIBRARY}"
        "${GGML_CPU_LIBRARY}"
        "${GGML_BASE_LIBRARY}"
        Threads::Threads
)

if(WIN32)
    target_compile_definitions(litenn_llama_cpp_stage_profile PRIVATE _WIN32_WINNT=0x0A00)
    target_link_libraries(litenn_llama_cpp_stage_profile PRIVATE ws2_32)
endif()

