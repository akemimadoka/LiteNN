add_library(litenn_mnist_common INTERFACE)
target_include_directories(litenn_mnist_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(litenn_mnist_common INTERFACE cxx_std_26)
target_compile_definitions(litenn_mnist_common INTERFACE
    LITENN_MNIST_DEFAULT_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data"
)

add_executable(litenn_mnist_interpreter interpreter.cpp)
target_link_libraries(litenn_mnist_interpreter PRIVATE LiteNN litenn_mnist_common)

set(LITENN_MNIST_TARGETS litenn_mnist_interpreter)

if(TARGET LiteNNCompiler)
    add_executable(litenn_mnist_aot aot.cpp)
    target_link_libraries(litenn_mnist_aot PRIVATE LiteNNCompiler litenn_mnist_common)
    list(APPEND LITENN_MNIST_TARGETS litenn_mnist_aot)
else()
    message(STATUS "LiteNNCompiler is not available; skipping litenn_mnist_aot")
endif()

add_custom_target(litenn_mnist DEPENDS ${LITENN_MNIST_TARGETS})
