cmake_minimum_required(VERSION 3.20)
project(LiteNNLlamaCppGolden LANGUAGES C CXX)

set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_UI OFF CACHE BOOL "" FORCE)

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../../third_party/llama.cpp llama.cpp EXCLUDE_FROM_ALL)

add_executable(litenn_llamacpp_decode_golden main.cpp)
target_compile_features(litenn_llamacpp_decode_golden PRIVATE cxx_std_20)
target_link_libraries(litenn_llamacpp_decode_golden PRIVATE llama)
