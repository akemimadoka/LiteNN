cmake_minimum_required(VERSION 4.0)
project(LiteNNPackageSmoke LANGUAGES CXX)

option(LITENN_PACKAGE_SMOKE_WITH_COMPILER "Link the optional LiteNN Compiler component" OFF)

if(LITENN_PACKAGE_SMOKE_WITH_COMPILER)
	find_package(LiteNN CONFIG REQUIRED COMPONENTS Compiler)
else()
	find_package(LiteNN CONFIG REQUIRED)
endif()

add_executable(litenn_package_smoke main.cpp)
target_link_libraries(litenn_package_smoke PRIVATE LiteNN::LiteNN)
target_compile_features(litenn_package_smoke PRIVATE cxx_std_26)

if(LITENN_PACKAGE_SMOKE_WITH_COMPILER)
	add_executable(litenn_package_compiler_smoke compiler_main.cpp)
	target_link_libraries(litenn_package_compiler_smoke PRIVATE LiteNN::LiteNNCompiler)
	target_compile_features(litenn_package_compiler_smoke PRIVATE cxx_std_26)
endif()
