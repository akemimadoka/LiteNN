#include "Pass/BufferizationPipeline.h"

#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"
#include "mlir/Dialect/Arith/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/FuncBufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Dialect/Tensor/Transforms/BufferizableOpInterfaceImpl.h"
#include "mlir/Transforms/Passes.h"

namespace litenn
{

void registerBufferizationModels(mlir::DialectRegistry& registry)
{
    mlir::arith::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::bufferization::func_ext::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::linalg::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::scf::registerBufferizableOpInterfaceExternalModels(registry);
    mlir::tensor::registerBufferizableOpInterfaceExternalModels(registry);
}

void addBufferizationPipeline(mlir::PassManager& pm)
{
    mlir::bufferization::OneShotBufferizePassOptions opts;
    opts.bufferizeFunctionBoundaries = true;
    opts.allowReturnAllocsFromLoops = true;
    pm.addPass(mlir::bufferization::createOneShotBufferizePass(opts));

    // Inline of buildBufferDeallocationPipeline — avoids PassPipelineOptions
    // RTTI link issue with MinGW.
    mlir::memref::ExpandReallocPassOptions expandOpts{/*emitDeallocs=*/false};
    pm.addPass(mlir::memref::createExpandReallocPass(expandOpts));
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::bufferization::createOwnershipBasedBufferDeallocationPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::bufferization::createBufferDeallocationSimplificationPass());
    pm.addPass(mlir::bufferization::createLowerDeallocationsPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createCanonicalizerPass());

    pm.addPass(mlir::createConvertBufferizationToMemRefPass());
    pm.nest<mlir::func::FuncOp>().addPass(
        mlir::bufferization::createPromoteBuffersToStackPass());
}

} // namespace litenn
