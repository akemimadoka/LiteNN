#pragma once

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"

namespace litenn
{

// Registers bufferizable op interface external models for all dialects used
// after LowerLiteNNPass. Call this before creating the MLIRContext, or call
// ctx.appendDialectRegistry(registry) with the returned registry.
void registerBufferizationModels(mlir::DialectRegistry& registry);

// Appends the full bufferization pipeline to pm:
//   1. one-shot-bufferize (function-boundary mode)
//   2. ownership-based-buffer-deallocation + simplification + lower-deallocations
//   3. convert-bufferization-to-memref
//   4. promote-buffers-to-stack (small static buffers → alloca)
void addBufferizationPipeline(mlir::PassManager& pm);

} // namespace litenn

