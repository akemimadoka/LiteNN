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

add_library(LiteNNCompiler
    CompiledModule.cpp
    Dialect/LiteNNDialect.cpp
    Dialect/LiteNNOps.cpp
    Translation/GraphToMLIR.cpp
    Pass/LowerLiteNNPass.cpp
    Pass/BufferizationPipeline.cpp
    Pass/LLVMCodegenPipeline.cpp
)

add_dependencies(LiteNNCompiler LiteNNCompilerIncGen)

target_include_directories(LiteNNCompiler PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${MLIR_INCLUDE_DIRS}
    ${LLVM_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
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
    MLIRMemRefDialect
    MLIRMemRefTransforms
    MLIRBufferizationDialect
    MLIRBufferizationTransforms
    MLIRBufferizationPipelines
    MLIRBufferizationToMemRef
    MLIRTransforms
    MLIRSCFToControlFlow
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
