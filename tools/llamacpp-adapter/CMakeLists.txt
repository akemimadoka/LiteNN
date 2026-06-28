cmake_minimum_required(VERSION 3.20)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    project(LiteNNLlamaCppGolden LANGUAGES C CXX)
endif()

set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../third_party/llama.cpp llama.cpp EXCLUDE_FROM_ALL)

add_library(LiteNNLlamaCppTokenizerAdapter STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/LlamaCppTokenizerAdapter.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/LlamaCppTokenizerAdapter.h
)
target_compile_features(LiteNNLlamaCppTokenizerAdapter PUBLIC cxx_std_20)
target_include_directories(LiteNNLlamaCppTokenizerAdapter PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(LiteNNLlamaCppTokenizerAdapter PUBLIC llama)

add_executable(litenn_llamacpp_adapter main.cpp)
target_compile_features(litenn_llamacpp_adapter PRIVATE cxx_std_20)
target_link_libraries(litenn_llamacpp_adapter PRIVATE LiteNNLlamaCppTokenizerAdapter)
