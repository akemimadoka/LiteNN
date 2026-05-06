#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include <memory>

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
//   1. convert-linalg-to-loops   — linalg.generic(memref) → scf.for
//   2. convert-scf-to-control-flow
//   3. convert-math-to-llvm
//   4. convert-arith-to-llvm
//   5. convert-index-to-llvm
//   6. expand-strided-metadata   — required before finalize-memref-to-llvm
//   7. finalize-memref-to-llvm
//   8. convert-func-to-llvm
//   9. convert-control-flow-to-llvm
//  10. reconcile-unrealized-casts
void addLLVMCodegenPipeline(mlir::PassManager& pm);

// Translates a fully lowered LLVM dialect ModuleOp to llvm::Module.
// Requires registerLLVMTranslations to have been called on the context's registry.
std::unique_ptr<llvm::Module> translateToLLVMIR(mlir::ModuleOp module,
                                                 llvm::LLVMContext& llvmCtx);

} // namespace litenn
