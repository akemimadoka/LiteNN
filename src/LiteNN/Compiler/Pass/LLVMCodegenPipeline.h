#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include <memory>

#ifndef LITENN_COMPILER_PASS_LLVMCODEGENPIPELINE_H
#define LITENN_COMPILER_PASS_LLVMCODEGENPIPELINE_H

namespace llvm {
class LLVMContext;
class Module;
} // namespace llvm

namespace mlir {
class DialectRegistry;
} // namespace mlir

namespace litenn
{

// Registers LLVM dialect translation interfaces (LLVMDialect + BuiltinDialect).
// Call before creating the MLIRContext, or via ctx.appendDialectRegistry().
void registerLLVMTranslations(mlir::DialectRegistry& registry);

// Appends the LLVM codegen pipeline to pm (run after addBufferizationPipeline):
//   1. canonicalize/CSE
//   2. convert-linalg-to-loops   — linalg.generic(memref) → scf.for
//   3. mark floating-point arith with reassoc/contract fast-math for SIMD/FMA
//   4. canonicalize/CSE
//   5. convert-scf-to-control-flow
//   6. convert-math-to-llvm
//   7. convert-arith-to-llvm
//   8. convert-index-to-llvm
//   9. expand-strided-metadata   — required before finalize-memref-to-llvm
//  10. finalize-memref-to-llvm
//  11. convert-func-to-llvm
//  12. convert-control-flow-to-llvm
//  13. reconcile-unrealized-casts
void addLLVMCodegenPipeline(mlir::PassManager& pm);

// Translates a fully lowered LLVM dialect ModuleOp to llvm::Module.
// Requires registerLLVMTranslations to have been called on the context's registry.
std::unique_ptr<llvm::Module> translateToLLVMIR(mlir::ModuleOp module,
                                                 llvm::LLVMContext& llvmCtx);

} // namespace litenn

#endif
