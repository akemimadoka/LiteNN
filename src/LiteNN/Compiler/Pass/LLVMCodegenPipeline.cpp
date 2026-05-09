#include "Pass/LLVMCodegenPipeline.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVMPass.h"
#include "mlir/Conversion/IndexToLLVM/IndexToLLVM.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

namespace litenn
{
namespace
{

bool isDimMap(mlir::AffineMap map, std::initializer_list<unsigned> dims)
{
    if (map.getNumDims() != 3 || map.getNumSymbols() != 0 ||
        map.getNumResults() != static_cast<unsigned>(dims.size()))
    {
        return false;
    }

    unsigned i = 0;
    for (const unsigned expected : dims)
    {
        auto dim = llvm::dyn_cast<mlir::AffineDimExpr>(map.getResult(i++));
        if (!dim || dim.getPosition() != expected)
        {
            return false;
        }
    }
    return true;
}

bool hasMatMulPayload(mlir::linalg::GenericOp op)
{
    auto& block = op.getRegion().front();
    if (block.getNumArguments() != 3)
    {
        return false;
    }

    auto yield = llvm::dyn_cast<mlir::linalg::YieldOp>(block.getTerminator());
    if (!yield || yield.getValues().size() != 1)
    {
        return false;
    }

    auto add = yield.getValues()[0].getDefiningOp<mlir::arith::AddFOp>();
    if (!add)
    {
        return false;
    }

    mlir::Value product;
    if (add.getLhs() == block.getArgument(2))
    {
        product = add.getRhs();
    }
    else if (add.getRhs() == block.getArgument(2))
    {
        product = add.getLhs();
    }
    else
    {
        return false;
    }

    auto mul = product.getDefiningOp<mlir::arith::MulFOp>();
    if (!mul)
    {
        return false;
    }

    return (mul.getLhs() == block.getArgument(0) && mul.getRhs() == block.getArgument(1)) ||
           (mul.getLhs() == block.getArgument(1) && mul.getRhs() == block.getArgument(0));
}

mlir::arith::FastMathFlagsAttr getContractOnlyFastMath(mlir::MLIRContext* ctx)
{
    auto flags = mlir::arith::FastMathFlags::contract |
                 mlir::arith::FastMathFlags::nnan |
                 mlir::arith::FastMathFlags::ninf |
                 mlir::arith::FastMathFlags::nsz;
    return mlir::arith::FastMathFlagsAttr::get(ctx, flags);
}

mlir::LogicalResult rewriteNarrowMatMul(mlir::linalg::GenericOp op,
                                        mlir::OpBuilder& builder)
{
    if (op->getNumResults() != 0 || op.getInputs().size() != 2 || op.getOutputs().size() != 1)
    {
        return mlir::failure();
    }

    auto lhs = op.getInputs().front();
    auto rhs = op.getInputs().back();
    auto out = op.getOutputs().front();
    auto lhsType = llvm::dyn_cast<mlir::MemRefType>(lhs.getType());
    auto rhsType = llvm::dyn_cast<mlir::MemRefType>(rhs.getType());
    auto outType = llvm::dyn_cast<mlir::MemRefType>(out.getType());
    if (!lhsType || !rhsType || !outType ||
        lhsType.getRank() != 2 || rhsType.getRank() != 2 || outType.getRank() != 2 ||
        !lhsType.getElementType().isF32() || !rhsType.getElementType().isF32() ||
        !outType.getElementType().isF32())
    {
        return mlir::failure();
    }

    const int64_t n = outType.getDimSize(1);
    if (n <= 0 || n > 16)
    {
        return mlir::failure();
    }

    auto maps = op.getIndexingMapsArray();
    if (maps.size() != 3 ||
        !isDimMap(maps[0], { 0, 1 }) ||
        !isDimMap(maps[1], { 1, 2 }) ||
        !isDimMap(maps[2], { 0, 2 }))
    {
        return mlir::failure();
    }

    auto iterTypes = op.getIteratorTypesArray();
    if (iterTypes.size() != 3 ||
        iterTypes[0] != mlir::utils::IteratorType::parallel ||
        iterTypes[1] != mlir::utils::IteratorType::reduction ||
        iterTypes[2] != mlir::utils::IteratorType::parallel)
    {
        return mlir::failure();
    }

    if (!hasMatMulPayload(op))
    {
        return mlir::failure();
    }

    const auto loc = op.getLoc();
    auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto mUpper = builder.create<mlir::memref::DimOp>(loc, out, 0);
    auto kUpper = builder.create<mlir::memref::DimOp>(loc, lhs, 1);
    auto fastMath = getContractOnlyFastMath(builder.getContext());

    auto mLoop = builder.create<mlir::scf::ForOp>(loc, c0, mUpper, c1);
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(mLoop.getBody());
        auto m = mLoop.getInductionVar();

        llvm::SmallVector<mlir::Value, 16> nIndices;
        llvm::SmallVector<mlir::Value, 16> initAccs;
        nIndices.reserve(static_cast<size_t>(n));
        initAccs.reserve(static_cast<size_t>(n));
        for (int64_t col = 0; col < n; ++col)
        {
            auto nIndex = builder.create<mlir::arith::ConstantIndexOp>(loc, col);
            nIndices.push_back(nIndex);
            initAccs.push_back(builder.create<mlir::memref::LoadOp>(
                loc, out, mlir::ValueRange{ m, nIndex }).getResult());
        }

        auto kLoop = builder.create<mlir::scf::ForOp>(
            loc, c0, kUpper, c1, initAccs,
            [&](mlir::OpBuilder& nested, mlir::Location nestedLoc,
                mlir::Value k, mlir::ValueRange accs) {
                auto a = nested.create<mlir::memref::LoadOp>(
                    nestedLoc, lhs, mlir::ValueRange{ m, k }).getResult();

                llvm::SmallVector<mlir::Value, 16> nextAccs;
                nextAccs.reserve(accs.size());
                for (int64_t col = 0; col < n; ++col)
                {
                    auto b = nested.create<mlir::memref::LoadOp>(
                        nestedLoc, rhs, mlir::ValueRange{ k, nIndices[static_cast<size_t>(col)] }).getResult();
                    auto productOp = nested.create<mlir::arith::MulFOp>(nestedLoc, a, b);
                    productOp->setAttr(productOp.getFastMathAttrName(), fastMath);
                    auto sumOp = nested.create<mlir::arith::AddFOp>(
                        nestedLoc, accs[static_cast<size_t>(col)], productOp.getResult());
                    sumOp->setAttr(sumOp.getFastMathAttrName(), fastMath);
                    nextAccs.push_back(sumOp.getResult());
                }
                nested.create<mlir::scf::YieldOp>(nestedLoc, nextAccs);
            });

        for (int64_t col = 0; col < n; ++col)
        {
            builder.create<mlir::memref::StoreOp>(
                loc, kLoop.getResult(static_cast<unsigned>(col)), out,
                mlir::ValueRange{ m, nIndices[static_cast<size_t>(col)] });
        }
    }

    op.erase();
    return mlir::success();
}

struct LowerNarrowMatMulPass
    : mlir::PassWrapper<LowerNarrowMatMulPass, mlir::OperationPass<mlir::ModuleOp>>
{
    llvm::StringRef getName() const override { return "LiteNNLowerNarrowMatMulPass"; }

    void getDependentDialects(mlir::DialectRegistry& registry) const override
    {
        registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                        mlir::scf::SCFDialect>();
    }

    void runOnOperation() override
    {
        llvm::SmallVector<mlir::linalg::GenericOp> candidates;
        getOperation().walk([&](mlir::linalg::GenericOp op) {
            candidates.push_back(op);
        });

        mlir::OpBuilder builder(&getContext());
        for (auto op : candidates)
        {
            builder.setInsertionPoint(op);
            (void)rewriteNarrowMatMul(op, builder);
        }
    }
};

std::unique_ptr<mlir::Pass> createLowerNarrowMatMulPass()
{
    return std::make_unique<LowerNarrowMatMulPass>();
}

struct EnableSIMDFastMathPass
    : mlir::PassWrapper<EnableSIMDFastMathPass, mlir::OperationPass<mlir::ModuleOp>>
{
    llvm::StringRef getName() const override { return "LiteNNEnableSIMDFastMathPass"; }

    void runOnOperation() override
    {
        auto flags = mlir::arith::FastMathFlags::reassoc |
                     mlir::arith::FastMathFlags::contract |
                     mlir::arith::FastMathFlags::nnan |
                     mlir::arith::FastMathFlags::ninf |
                     mlir::arith::FastMathFlags::nsz;
        auto attr = mlir::arith::FastMathFlagsAttr::get(&getContext(), flags);

        getOperation().walk([&](mlir::arith::ArithFastMathInterface op) {
            if (op.getFastMathFlagsAttr())
            {
                return;
            }
            op->setAttr(op.getFastMathAttrName(), attr);
        });
    }
};

std::unique_ptr<mlir::Pass> createEnableSIMDFastMathPass()
{
    return std::make_unique<EnableSIMDFastMathPass>();
}

} // namespace

void registerLLVMTranslations(mlir::DialectRegistry& registry)
{
    mlir::registerBuiltinDialectTranslation(registry);
    mlir::registerLLVMDialectTranslation(registry);
}

void addLLVMCodegenPipeline(mlir::PassManager& pm)
{
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(createLowerNarrowMatMulPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createConvertLinalgToLoopsPass());
    pm.addPass(createEnableSIMDFastMathPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addPass(mlir::createCSEPass());
    pm.addPass(mlir::createSCFToControlFlowPass());
    pm.addPass(mlir::createConvertMathToLLVMPass());
    pm.addPass(mlir::createArithToLLVMConversionPass());
    pm.addPass(mlir::createConvertIndexToLLVMPass());
    pm.addPass(mlir::memref::createExpandStridedMetadataPass());
    pm.addPass(mlir::createFinalizeMemRefToLLVMConversionPass());
    pm.addPass(mlir::createConvertFuncToLLVMPass());
    pm.addPass(mlir::createConvertControlFlowToLLVMPass());
    pm.addPass(mlir::createReconcileUnrealizedCastsPass());
}

std::unique_ptr<llvm::Module> translateToLLVMIR(mlir::ModuleOp module,
                                                  llvm::LLVMContext& llvmCtx)
{
    return mlir::translateModuleToLLVMIR(module, llvmCtx);
}

} // namespace litenn
