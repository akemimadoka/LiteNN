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
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVMPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "mlir/Transforms/Passes.h"

namespace litenn
{
namespace
{
constexpr llvm::StringLiteral kApplyReluAttr = "litenn.apply_relu";

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

mlir::Value applyReluIfNeeded(mlir::OpBuilder& builder,
                              mlir::Location loc,
                              mlir::Value value,
                              bool applyRelu)
{
    if (!applyRelu)
    {
        return value;
    }

    mlir::Value zero;
    if (auto vectorType = llvm::dyn_cast<mlir::VectorType>(value.getType()))
    {
        zero = builder.create<mlir::arith::ConstantOp>(
            loc, vectorType, builder.getZeroAttr(vectorType)).getResult();
    }
    else
    {
        auto floatType = llvm::cast<mlir::FloatType>(value.getType());
        zero = builder.create<mlir::arith::ConstantFloatOp>(
            loc, floatType, llvm::APFloat::getZero(floatType.getFloatSemantics())).getResult();
    }
    return builder.create<mlir::arith::MaximumFOp>(loc, value, zero).getResult();
}

mlir::LogicalResult validateMatMulCandidate(mlir::linalg::GenericOp op,
                                            mlir::Value& lhs,
                                            mlir::Value& rhs,
                                            mlir::Value& out,
                                            mlir::MemRefType& lhsType,
                                            mlir::MemRefType& rhsType,
                                            mlir::MemRefType& outType)
{
    if (op->getNumResults() != 0 || op.getInputs().size() != 2 || op.getOutputs().size() != 1)
    {
        return mlir::failure();
    }

    lhs = op.getInputs().front();
    rhs = op.getInputs().back();
    out = op.getOutputs().front();
    lhsType = llvm::dyn_cast<mlir::MemRefType>(lhs.getType());
    rhsType = llvm::dyn_cast<mlir::MemRefType>(rhs.getType());
    outType = llvm::dyn_cast<mlir::MemRefType>(out.getType());
    if (!lhsType || !rhsType || !outType ||
        lhsType.getRank() != 2 || rhsType.getRank() != 2 || outType.getRank() != 2 ||
        !lhsType.getElementType().isF32() || !rhsType.getElementType().isF32() ||
        !outType.getElementType().isF32())
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

    return mlir::success();
}

mlir::LogicalResult rewriteWideMatMulRowTile(mlir::linalg::GenericOp op,
                                             mlir::OpBuilder& builder)
{
    mlir::Value lhs;
    mlir::Value rhs;
    mlir::Value out;
    mlir::MemRefType lhsType;
    mlir::MemRefType rhsType;
    mlir::MemRefType outType;
    if (mlir::failed(validateMatMulCandidate(op, lhs, rhs, out, lhsType, rhsType, outType)))
    {
        return mlir::failure();
    }

    constexpr int64_t kVectorWidth = 16;
    constexpr int64_t kRowTile = 4;
    const int64_t n = outType.getDimSize(1);
    if (n <= kVectorWidth || n % kVectorWidth != 0 ||
        outType.isDynamicDim(0) || outType.getDimSize(0) % kRowTile != 0)
    {
        return mlir::failure();
    }

    const int64_t tileVectors = (n % (kVectorWidth * 4) == 0) ? 4 :
                                (n % (kVectorWidth * 2) == 0) ? 2 : 1;
    const int64_t nStep = kVectorWidth * tileVectors;

    const auto loc = op.getLoc();
    auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto cMstep = builder.create<mlir::arith::ConstantIndexOp>(loc, kRowTile);
    auto cNStep = builder.create<mlir::arith::ConstantIndexOp>(loc, nStep);
    auto nUpper = builder.create<mlir::arith::ConstantIndexOp>(loc, n);
    auto mUpper = builder.create<mlir::memref::DimOp>(loc, out, 0);
    auto kUpper = builder.create<mlir::memref::DimOp>(loc, lhs, 1);
    auto vecType = mlir::VectorType::get({ kVectorWidth }, outType.getElementType());
    const bool applyRelu = op->hasAttr(kApplyReluAttr);

    auto mLoop = builder.create<mlir::scf::ForOp>(loc, c0, mUpper, cMstep);
    {
        mlir::OpBuilder::InsertionGuard mGuard(builder);
        builder.setInsertionPointToStart(mLoop.getBody());
        auto mBase = mLoop.getInductionVar();
        llvm::SmallVector<mlir::Value, 4> mIndices;
        mIndices.reserve(kRowTile);
        mIndices.push_back(mBase);
        for (int64_t row = 1; row < kRowTile; ++row)
        {
            auto offset = builder.create<mlir::arith::ConstantIndexOp>(loc, row);
            mIndices.push_back(builder.create<mlir::arith::AddIOp>(loc, mBase, offset).getResult());
        }

        auto nLoop = builder.create<mlir::scf::ForOp>(loc, c0, nUpper, cNStep);
        {
            mlir::OpBuilder::InsertionGuard nGuard(builder);
            builder.setInsertionPointToStart(nLoop.getBody());
            auto nBase = nLoop.getInductionVar();

            llvm::SmallVector<mlir::Value, 4> nIndices;
            llvm::SmallVector<mlir::Value, 8> initAccs;
            nIndices.reserve(static_cast<size_t>(tileVectors));
            initAccs.reserve(static_cast<size_t>(kRowTile * tileVectors));
            for (int64_t lane = 0; lane < tileVectors; ++lane)
            {
                mlir::Value nIndex = nBase;
                if (lane != 0)
                {
                    auto offset = builder.create<mlir::arith::ConstantIndexOp>(loc, lane * kVectorWidth);
                    nIndex = builder.create<mlir::arith::AddIOp>(loc, nBase, offset).getResult();
                }
                nIndices.push_back(nIndex);
            }

            for (int64_t row = 0; row < kRowTile; ++row)
            {
                for (int64_t lane = 0; lane < tileVectors; ++lane)
                {
                    initAccs.push_back(builder.create<mlir::vector::LoadOp>(
                        loc, vecType, out,
                        mlir::ValueRange{
                            mIndices[static_cast<size_t>(row)],
                            nIndices[static_cast<size_t>(lane)] }).getResult());
                }
            }

            auto kLoop = builder.create<mlir::scf::ForOp>(
                loc, c0, kUpper, c1, initAccs,
                [&](mlir::OpBuilder& nested, mlir::Location nestedLoc,
                    mlir::Value k, mlir::ValueRange accs) {
                    llvm::SmallVector<mlir::Value, 4> bVecs;
                    bVecs.reserve(static_cast<size_t>(tileVectors));
                    for (int64_t lane = 0; lane < tileVectors; ++lane)
                    {
                        bVecs.push_back(nested.create<mlir::vector::LoadOp>(
                            nestedLoc, vecType, rhs,
                            mlir::ValueRange{ k, nIndices[static_cast<size_t>(lane)] }).getResult());
                    }

                    llvm::SmallVector<mlir::Value, 8> nextAccs;
                    nextAccs.reserve(accs.size());
                    for (int64_t row = 0; row < kRowTile; ++row)
                    {
                        auto a = nested.create<mlir::memref::LoadOp>(
                            nestedLoc, lhs,
                            mlir::ValueRange{ mIndices[static_cast<size_t>(row)], k }).getResult();
                        auto aVec = nested.create<mlir::vector::BroadcastOp>(
                            nestedLoc, vecType, a).getResult();
                        for (int64_t lane = 0; lane < tileVectors; ++lane)
                        {
                            const size_t accIndex =
                                static_cast<size_t>(row * tileVectors + lane);
                            auto next = nested.create<mlir::vector::FMAOp>(
                                nestedLoc, aVec, bVecs[static_cast<size_t>(lane)],
                                accs[accIndex]).getResult();
                            nextAccs.push_back(next);
                        }
                    }
                    nested.create<mlir::scf::YieldOp>(nestedLoc, nextAccs);
                });

            for (int64_t row = 0; row < kRowTile; ++row)
            {
                for (int64_t lane = 0; lane < tileVectors; ++lane)
                {
                    const unsigned accIndex =
                        static_cast<unsigned>(row * tileVectors + lane);
                    auto value = applyReluIfNeeded(
                        builder, loc, kLoop.getResult(accIndex), applyRelu);
                    builder.create<mlir::vector::StoreOp>(
                        loc, value, out,
                        mlir::ValueRange{
                            mIndices[static_cast<size_t>(row)],
                            nIndices[static_cast<size_t>(lane)] });
                }
            }
        }
    }

    op.erase();
    return mlir::success();
}

mlir::LogicalResult rewriteWideMatMul(mlir::linalg::GenericOp op,
                                      mlir::OpBuilder& builder)
{
    mlir::Value lhs;
    mlir::Value rhs;
    mlir::Value out;
    mlir::MemRefType lhsType;
    mlir::MemRefType rhsType;
    mlir::MemRefType outType;
    if (mlir::failed(validateMatMulCandidate(op, lhs, rhs, out, lhsType, rhsType, outType)))
    {
        return mlir::failure();
    }

    constexpr int64_t kVectorWidth = 16;
    const int64_t n = outType.getDimSize(1);
    if (n <= kVectorWidth || n % kVectorWidth != 0)
    {
        return mlir::failure();
    }

    const int64_t tileVectors = (n % (kVectorWidth * 8) == 0) ? 8 :
                                (n % (kVectorWidth * 4) == 0) ? 4 :
                                (n % (kVectorWidth * 2) == 0) ? 2 : 1;
    const int64_t nStep = kVectorWidth * tileVectors;

    const auto loc = op.getLoc();
    auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto cNStep = builder.create<mlir::arith::ConstantIndexOp>(loc, nStep);
    auto nUpper = builder.create<mlir::arith::ConstantIndexOp>(loc, n);
    auto mUpper = builder.create<mlir::memref::DimOp>(loc, out, 0);
    auto kUpper = builder.create<mlir::memref::DimOp>(loc, lhs, 1);
    auto vecType = mlir::VectorType::get({ kVectorWidth }, outType.getElementType());
    const bool applyRelu = op->hasAttr(kApplyReluAttr);

    auto mLoop = builder.create<mlir::scf::ForOp>(loc, c0, mUpper, c1);
    {
        mlir::OpBuilder::InsertionGuard mGuard(builder);
        builder.setInsertionPointToStart(mLoop.getBody());
        auto m = mLoop.getInductionVar();

        auto nLoop = builder.create<mlir::scf::ForOp>(loc, c0, nUpper, cNStep);
        {
            mlir::OpBuilder::InsertionGuard nGuard(builder);
            builder.setInsertionPointToStart(nLoop.getBody());
            auto nBase = nLoop.getInductionVar();

            llvm::SmallVector<mlir::Value, 8> nIndices;
            llvm::SmallVector<mlir::Value, 8> initAccs;
            nIndices.reserve(static_cast<size_t>(tileVectors));
            initAccs.reserve(static_cast<size_t>(tileVectors));
            for (int64_t lane = 0; lane < tileVectors; ++lane)
            {
                mlir::Value nIndex = nBase;
                if (lane != 0)
                {
                    auto offset = builder.create<mlir::arith::ConstantIndexOp>(loc, lane * kVectorWidth);
                    nIndex = builder.create<mlir::arith::AddIOp>(loc, nBase, offset).getResult();
                }
                nIndices.push_back(nIndex);
                initAccs.push_back(builder.create<mlir::vector::LoadOp>(
                    loc, vecType, out, mlir::ValueRange{ m, nIndex }).getResult());
            }

            auto kLoop = builder.create<mlir::scf::ForOp>(
                loc, c0, kUpper, c1, initAccs,
                [&](mlir::OpBuilder& nested, mlir::Location nestedLoc,
                    mlir::Value k, mlir::ValueRange accs) {
                    auto a = nested.create<mlir::memref::LoadOp>(
                        nestedLoc, lhs, mlir::ValueRange{ m, k }).getResult();
                    auto aVec = nested.create<mlir::vector::BroadcastOp>(
                        nestedLoc, vecType, a).getResult();

                    llvm::SmallVector<mlir::Value, 8> nextAccs;
                    nextAccs.reserve(accs.size());
                    for (int64_t lane = 0; lane < tileVectors; ++lane)
                    {
                        auto bVec = nested.create<mlir::vector::LoadOp>(
                            nestedLoc, vecType, rhs,
                            mlir::ValueRange{ k, nIndices[static_cast<size_t>(lane)] }).getResult();
                        auto next = nested.create<mlir::vector::FMAOp>(
                            nestedLoc, aVec, bVec, accs[static_cast<size_t>(lane)]).getResult();
                        nextAccs.push_back(next);
                    }
                    nested.create<mlir::scf::YieldOp>(nestedLoc, nextAccs);
                });

            for (int64_t lane = 0; lane < tileVectors; ++lane)
            {
                auto value = applyReluIfNeeded(
                    builder, loc, kLoop.getResult(static_cast<unsigned>(lane)), applyRelu);
                builder.create<mlir::vector::StoreOp>(
                    loc, value, out,
                    mlir::ValueRange{ m, nIndices[static_cast<size_t>(lane)] });
            }
        }
    }

    op.erase();
    return mlir::success();
}

mlir::LogicalResult rewriteNarrowVectorMatMul(mlir::linalg::GenericOp op,
                                              mlir::OpBuilder& builder)
{
    mlir::Value lhs;
    mlir::Value rhs;
    mlir::Value out;
    mlir::MemRefType lhsType;
    mlir::MemRefType rhsType;
    mlir::MemRefType outType;
    if (mlir::failed(validateMatMulCandidate(op, lhs, rhs, out, lhsType, rhsType, outType)))
    {
        return mlir::failure();
    }

    const int64_t n = outType.getDimSize(1);
    if (n < 4 || n > 16)
    {
        return mlir::failure();
    }

    const auto loc = op.getLoc();
    auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto mUpper = builder.create<mlir::memref::DimOp>(loc, out, 0);
    auto kUpper = builder.create<mlir::memref::DimOp>(loc, lhs, 1);
    auto vecType = mlir::VectorType::get({ n }, outType.getElementType());
    const bool applyRelu = op->hasAttr(kApplyReluAttr);

    auto mLoop = builder.create<mlir::scf::ForOp>(loc, c0, mUpper, c1);
    {
        mlir::OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(mLoop.getBody());
        auto m = mLoop.getInductionVar();

        auto initAcc = builder.create<mlir::vector::LoadOp>(
            loc, vecType, out, mlir::ValueRange{ m, c0 }).getResult();
        auto kLoop = builder.create<mlir::scf::ForOp>(
            loc, c0, kUpper, c1, initAcc,
            [&](mlir::OpBuilder& nested, mlir::Location nestedLoc,
                mlir::Value k, mlir::ValueRange accs) {
                auto a = nested.create<mlir::memref::LoadOp>(
                    nestedLoc, lhs, mlir::ValueRange{ m, k }).getResult();
                auto aVec = nested.create<mlir::vector::BroadcastOp>(
                    nestedLoc, vecType, a).getResult();
                auto bVec = nested.create<mlir::vector::LoadOp>(
                    nestedLoc, vecType, rhs, mlir::ValueRange{ k, c0 }).getResult();
                auto next = nested.create<mlir::vector::FMAOp>(
                    nestedLoc, aVec, bVec, accs.front()).getResult();
                nested.create<mlir::scf::YieldOp>(nestedLoc, next);
            });

        auto value = applyReluIfNeeded(builder, loc, kLoop.getResult(0), applyRelu);
        builder.create<mlir::vector::StoreOp>(
            loc, value, out, mlir::ValueRange{ m, c0 });
    }

    op.erase();
    return mlir::success();
}

mlir::LogicalResult rewriteNarrowMatMul(mlir::linalg::GenericOp op,
                                        mlir::OpBuilder& builder)
{
    mlir::Value lhs;
    mlir::Value rhs;
    mlir::Value out;
    mlir::MemRefType lhsType;
    mlir::MemRefType rhsType;
    mlir::MemRefType outType;
    if (mlir::failed(validateMatMulCandidate(op, lhs, rhs, out, lhsType, rhsType, outType)))
    {
        return mlir::failure();
    }

    const int64_t n = outType.getDimSize(1);
    if (n <= 0 || n > 16)
    {
        return mlir::failure();
    }

    const auto loc = op.getLoc();
    auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
    auto c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
    auto mUpper = builder.create<mlir::memref::DimOp>(loc, out, 0);
    auto kUpper = builder.create<mlir::memref::DimOp>(loc, lhs, 1);
    auto fastMath = getContractOnlyFastMath(builder.getContext());
    const bool applyRelu = op->hasAttr(kApplyReluAttr);

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
            auto value = applyReluIfNeeded(
                builder, loc, kLoop.getResult(static_cast<unsigned>(col)), applyRelu);
            builder.create<mlir::memref::StoreOp>(
                loc, value, out,
                mlir::ValueRange{ m, nIndices[static_cast<size_t>(col)] });
        }
    }

    op.erase();
    return mlir::success();
}

struct LowerNarrowMatMulPass
    : mlir::PassWrapper<LowerNarrowMatMulPass, mlir::OperationPass<mlir::ModuleOp>>
{
    llvm::StringRef getName() const override { return "LiteNNLowerMatMulMicroKernelPass"; }

    void getDependentDialects(mlir::DialectRegistry& registry) const override
    {
        registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect,
                        mlir::scf::SCFDialect, mlir::vector::VectorDialect>();
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
            if (mlir::succeeded(rewriteWideMatMulRowTile(op, builder)))
            {
                continue;
            }
            if (mlir::succeeded(rewriteWideMatMul(op, builder)))
            {
                continue;
            }
            if (mlir::succeeded(rewriteNarrowVectorMatMul(op, builder)))
            {
                continue;
            }
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
    pm.addPass(mlir::createConvertVectorToLLVMPass());
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
