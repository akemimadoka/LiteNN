#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/OwningOpRef.h"

#ifndef LITENN_COMPILER_TRANSLATION_GRAPHTOMLIR_H
#define LITENN_COMPILER_TRANSLATION_GRAPHTOMLIR_H

namespace LiteNN
{
	struct ExecutablePlan;
}

namespace litenn
{
	struct GraphToMLIROptions
	{
		bool enableNodeProfiling{};
	};

	/// Translate a LiteNN executable plan to an MLIR module in the litenn dialect.
	mlir::OwningOpRef<mlir::ModuleOp> translateExecutablePlanToMLIR(const LiteNN::ExecutablePlan& plan,
	                                                                mlir::MLIRContext& ctx,
	                                                                const GraphToMLIROptions& options = {});

} // namespace litenn

#endif
