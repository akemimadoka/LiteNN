#ifndef LITENN_COMPILER_TRANSLATION_GRAPHTOMLIR_H
#define LITENN_COMPILER_TRANSLATION_GRAPHTOMLIR_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

namespace LiteNN
{
class Graph;
}

namespace litenn
{

/// Translate a LiteNN Graph to an MLIR module in the litenn dialect.
mlir::OwningOpRef<mlir::ModuleOp> translateGraphToMLIR(const LiteNN::Graph& graph, mlir::MLIRContext& ctx);

} // namespace litenn

#endif
