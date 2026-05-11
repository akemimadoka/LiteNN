include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

set(LLVM_TARGET_DEFINITIONS Dialect/LiteNNOps.td)

mlir_tablegen(Dialect/LiteNNOps.h.inc -gen-op-decls)
mlir_tablegen(Dialect/LiteNNOps.cpp.inc -gen-op-defs)
mlir_tablegen(Dialect/LiteNNDialect.h.inc -gen-dialect-decls -dialect=litenn)
mlir_tablegen(Dialect/LiteNNDialect.cpp.inc -gen-dialect-defs -dialect=litenn)
mlir_tablegen(Dialect/LiteNNEnums.h.inc -gen-enum-decls)
mlir_tablegen(Dialect/LiteNNEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(LiteNNCompilerIncGen)

file(GLOB_RECURSE LITENN_COMPILER_SOURCES CONFIGURE_DEPENDS
    *.cpp
    *.h
)

add_library(LiteNNCompiler
    ${LITENN_COMPILER_SOURCES}
)

add_library(LiteNN::LiteNNCompiler ALIAS LiteNNCompiler)

add_dependencies(LiteNNCompiler LiteNNCompilerIncGen)

target_include_directories(LiteNNCompiler PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>
    $<BUILD_INTERFACE:${MLIR_INCLUDE_DIRS}>
    $<BUILD_INTERFACE:${LLVM_INCLUDE_DIRS}>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/LiteNN/Compiler>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(LiteNNCompiler PUBLIC LiteNN)
llvm_map_components_to_libnames(LITENN_LLVM_NATIVE_LIBS
    native
    nativecodegen
    mcjit
    executionengine
    runtimedyld
    object
    mc
    target
    codegen
    passes
    analysis
    scalaropts
    transformutils
    instcombine
    aggressiveinstcombine
    ipo
    vectorize
    targetparser
)
target_link_libraries(LiteNNCompiler PUBLIC
    MLIRIR
    MLIRSupport
    MLIRFuncDialect
    MLIRArithDialect
    MLIRParser
    MLIRPass
    MLIRCallInterfaces
    MLIRControlFlowInterfaces
    MLIRFunctionInterfaces
    MLIRSideEffectInterfaces
    MLIRBytecodeOpInterface
    MLIRBuiltinToLLVMIRTranslation
    LLVMSupport
    MLIRLinalgDialect
    MLIRLinalgTransforms
    MLIRArithTransforms
    MLIRSCFTransforms
    MLIRMathDialect
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRTensorTransforms
    MLIRVectorDialect
    MLIRVectorTransforms
    MLIRMemRefDialect
    MLIRMemRefTransforms
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRBufferizationPipelines
    MLIRBufferizationToMemRef
    MLIRTransforms
    MLIRSCFToControlFlow
    MLIRVectorToLLVM
    MLIRVectorToLLVMPass
    MLIRMathToLLVM
    MLIRArithToLLVM
    MLIRIndexToLLVM
    MLIRMemRefToLLVM
    MLIRFuncToLLVM
    MLIRControlFlowToLLVM
    MLIRReconcileUnrealizedCasts
    MLIRTargetLLVMIRExport
    MLIRLLVMToLLVMIRTranslation
    MLIRLLVMDialect
    MLIRLLVMCommonConversion
    LLVMCore
    LLVMAsmPrinter
    ${LITENN_LLVM_NATIVE_LIBS}
)

install(TARGETS LiteNNCompiler
    EXPORT LiteNNCompilerTargets
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
