#include "Pass/LowerLiteNNPass.h"
#include "Dialect/LiteNNOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace litenn;

namespace
{

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Build broadcast-aware indexing maps for linalg.generic.
// For each dimension in result: if the input has size 1 at that dim → constant 0,
// otherwise use the corresponding AffineDimExpr.
// Handles the case where input rank < result rank (left-pad with broadcast dims).
AffineMap buildBroadcastMap(MLIRContext* ctx, ArrayRef<int64_t> inputShape,
                             ArrayRef<int64_t> resultShape)
{
	const int64_t resultRank = static_cast<int64_t>(resultShape.size());
	const int64_t inputRank = static_cast<int64_t>(inputShape.size());
	const int64_t rankDiff = resultRank - inputRank;

	SmallVector<AffineExpr> exprs;
	exprs.reserve(inputRank);
	for (int64_t i = 0; i < inputRank; ++i)
	{
		int64_t resultDim = i + rankDiff;
		if (inputShape[i] == 1 && resultShape[resultDim] != 1)
			exprs.push_back(getAffineConstantExpr(0, ctx));
		else
			exprs.push_back(getAffineDimExpr(resultDim, ctx));
	}
	return AffineMap::get(resultRank, 0, exprs, ctx);
}

// Build identity map for result dimensions.
AffineMap identityMap(MLIRContext* ctx, int64_t rank)
{
	return AffineMap::getMultiDimIdentityMap(rank, ctx);
}

// Build all-parallel iterator types for elementwise ops.
SmallVector<utils::IteratorType> parallelIterators(int64_t rank)
{
	return SmallVector<utils::IteratorType>(rank, utils::IteratorType::parallel);
}

// Emit an arith/math binary scalar op matching the given BinaryOpKind.
Value emitBinaryScalar(OpBuilder& b, Location loc, BinaryOpKind kind, Value lhs, Value rhs,
                        Type elemType)
{
	const bool isFloat = isa<FloatType>(elemType);
	const bool isInt = isa<IntegerType>(elemType);

	switch (kind)
	{
	case BinaryOpKind::Add:
		return isFloat ? b.create<arith::AddFOp>(loc, lhs, rhs).getResult()
		               : b.create<arith::AddIOp>(loc, lhs, rhs).getResult();
	case BinaryOpKind::Subtract:
		return isFloat ? b.create<arith::SubFOp>(loc, lhs, rhs).getResult()
		               : b.create<arith::SubIOp>(loc, lhs, rhs).getResult();
	case BinaryOpKind::Multiply:
		return isFloat ? b.create<arith::MulFOp>(loc, lhs, rhs).getResult()
		               : b.create<arith::MulIOp>(loc, lhs, rhs).getResult();
	case BinaryOpKind::Divide:
		return isFloat ? b.create<arith::DivFOp>(loc, lhs, rhs).getResult()
		               : b.create<arith::DivSIOp>(loc, lhs, rhs).getResult();
	case BinaryOpKind::Pow:
		return b.create<math::PowFOp>(loc, lhs, rhs).getResult();
	case BinaryOpKind::Max:
		return isFloat ? b.create<arith::MaximumFOp>(loc, lhs, rhs).getResult()
		               : b.create<arith::MaxSIOp>(loc, lhs, rhs).getResult();
	case BinaryOpKind::Min:
		return isFloat ? b.create<arith::MinimumFOp>(loc, lhs, rhs).getResult()
		               : b.create<arith::MinSIOp>(loc, lhs, rhs).getResult();
	case BinaryOpKind::Less: {
		const bool inputIsFloat = isa<FloatType>(lhs.getType());
		if (inputIsFloat)
			return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OLT, lhs, rhs).getResult();
		return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs, rhs).getResult();
	}
	case BinaryOpKind::Greater: {
		const bool inputIsFloat = isa<FloatType>(lhs.getType());
		if (inputIsFloat)
			return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OGT, lhs, rhs).getResult();
		return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lhs, rhs).getResult();
	}
	case BinaryOpKind::Equal: {
		const bool inputIsFloat = isa<FloatType>(lhs.getType());
		if (inputIsFloat)
			return b.create<arith::CmpFOp>(loc, arith::CmpFPredicate::OEQ, lhs, rhs).getResult();
		return b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, lhs, rhs).getResult();
	}
	case BinaryOpKind::MatMul:
		llvm_unreachable("MatMul handled separately");
	}
	llvm_unreachable("unknown BinaryOpKind");
}

// Emit an arith/math unary scalar op.
Value emitUnaryScalar(OpBuilder& b, Location loc, UnaryOpKind kind, Value input, Type elemType)
{
	const bool isFloat = isa<FloatType>(elemType);
	(void)isFloat;

	switch (kind)
	{
	case UnaryOpKind::Negate:
		return isFloat ? b.create<arith::NegFOp>(loc, input).getResult()
		               : b.create<arith::SubIOp>(loc, b.create<arith::ConstantIntOp>(loc, elemType, (int64_t)0), input)
		                     .getResult();
	case UnaryOpKind::Abs:
		return isFloat ? b.create<math::AbsFOp>(loc, input).getResult()
		               : b.create<math::AbsIOp>(loc, input).getResult();
	case UnaryOpKind::Sqrt:
		return b.create<math::SqrtOp>(loc, input).getResult();
	case UnaryOpKind::Exp:
		return b.create<math::ExpOp>(loc, input).getResult();
	case UnaryOpKind::Log:
		return b.create<math::LogOp>(loc, input).getResult();
	case UnaryOpKind::Sin:
		return b.create<math::SinOp>(loc, input).getResult();
	case UnaryOpKind::Cos:
		return b.create<math::CosOp>(loc, input).getResult();
	case UnaryOpKind::Tan:
		return b.create<math::TanOp>(loc, input).getResult();
	case UnaryOpKind::Arcsin:
		return b.create<math::AsinOp>(loc, input).getResult();
	case UnaryOpKind::Arccos:
		return b.create<math::AcosOp>(loc, input).getResult();
	case UnaryOpKind::Arctan:
		return b.create<math::AtanOp>(loc, input).getResult();
	case UnaryOpKind::LogicalNegation: {
		auto one = b.create<arith::ConstantIntOp>(loc, b.getI1Type(), (int64_t)1);
		return b.create<arith::XOrIOp>(loc, input, one).getResult();
	}
	case UnaryOpKind::Transpose:
		llvm_unreachable("Transpose handled separately");
	}
	llvm_unreachable("unknown UnaryOpKind");
}

// Build a zero scalar constant of the given float/int type.
Value zeroScalar(OpBuilder& b, Location loc, Type elemType)
{
	if (auto ft = dyn_cast<FloatType>(elemType))
		return b.create<arith::ConstantFloatOp>(loc, ft, APFloat::getZero(ft.getFloatSemantics()));
	return b.create<arith::ConstantIntOp>(loc, elemType, (int64_t)0);
}

//===----------------------------------------------------------------------===//
// Structural Op Patterns
//===----------------------------------------------------------------------===//

struct ConvertFuncOp : OpRewritePattern<FuncOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(FuncOp op, PatternRewriter& rewriter) const override
	{
		auto newFunc = rewriter.create<func::FuncOp>(op.getLoc(), op.getSymName(), op.getFunctionType());
		newFunc.setVisibility(op.getVisibility());
		rewriter.inlineRegionBefore(op.getBody(), newFunc.getBody(), newFunc.end());
		rewriter.eraseOp(op);
		return success();
	}
};

struct ConvertReturnOp : OpRewritePattern<ReturnOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(ReturnOp op, PatternRewriter& rewriter) const override
	{
		rewriter.replaceOpWithNewOp<func::ReturnOp>(op, op.getOperands());
		return success();
	}
};

struct ConvertCallOp : OpRewritePattern<CallOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(CallOp op, PatternRewriter& rewriter) const override
	{
		rewriter.replaceOpWithNewOp<func::CallOp>(op, op.getCallee(), op.getResultTypes(),
		                                          op.getOperands());
		return success();
	}
};

//===----------------------------------------------------------------------===//
// Constant / Variable Patterns
//===----------------------------------------------------------------------===//

struct ConvertConstantOp : OpRewritePattern<ConstantOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(ConstantOp op, PatternRewriter& rewriter) const override
	{
		rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, op.getValue());
		return success();
	}
};

struct ConvertVariableOp : OpRewritePattern<VariableOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(VariableOp op, PatternRewriter& rewriter) const override
	{
		auto tensorType = cast<RankedTensorType>(op.getType());
		auto memrefType =
		    MemRefType::get(tensorType.getShape(), tensorType.getElementType());

		// Build initial value attribute (or leave empty for zero-init)
		Attribute initAttr;
		if (auto initVal = op.getInitialValue())
			initAttr = *initVal;

		rewriter.replaceOpWithNewOp<memref::GlobalOp>(
		    op, op.getSymName(),
		    /*sym_visibility=*/rewriter.getStringAttr("public"),
		    memrefType, initAttr,
		    /*constant=*/false,
		    /*alignment=*/IntegerAttr{});
		return success();
	}
};

struct ConvertGetVariableOp : OpRewritePattern<GetVariableOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(GetVariableOp op, PatternRewriter& rewriter) const override
	{
		auto tensorType = cast<RankedTensorType>(op.getType());
		auto memrefType =
		    MemRefType::get(tensorType.getShape(), tensorType.getElementType());

		auto getGlobal = rewriter.create<memref::GetGlobalOp>(op.getLoc(), memrefType, op.getVariable());
		rewriter.replaceOpWithNewOp<bufferization::ToTensorOp>(op, tensorType, getGlobal,
		                                                        /*restrict=*/true,
		                                                        /*writable=*/false);
		return success();
	}
};

//===----------------------------------------------------------------------===//
// Arithmetic / Elementwise Patterns
//===----------------------------------------------------------------------===//

struct ConvertBinaryOp : OpRewritePattern<litenn::BinaryOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(litenn::BinaryOp op, PatternRewriter& rewriter) const override
	{
		auto loc = op.getLoc();
		auto* ctx = rewriter.getContext();
		auto lhs = op.getLhs();
		auto rhs = op.getRhs();
		auto resultType = cast<RankedTensorType>(op.getResult().getType());

		// MatMul: delegate to linalg.matmul
		if (op.getOp() == BinaryOpKind::MatMul)
		{
			auto emptyOut = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
			                                                  resultType.getElementType());
			auto zero = zeroScalar(rewriter, loc, resultType.getElementType());
			auto filled = rewriter.create<linalg::FillOp>(loc, ValueRange{zero}, ValueRange{emptyOut})
			                  .getResult(0);
			rewriter.replaceOpWithNewOp<linalg::MatmulOp>(op, TypeRange{resultType},
			                                               ValueRange{lhs, rhs}, ValueRange{filled});
			return success();
		}

		// Elementwise: linalg.generic with broadcast indexing maps
		auto lhsType = cast<RankedTensorType>(lhs.getType());
		auto rhsType = cast<RankedTensorType>(rhs.getType());

		auto lhsMap = buildBroadcastMap(ctx, lhsType.getShape(), resultType.getShape());
		auto rhsMap = buildBroadcastMap(ctx, rhsType.getShape(), resultType.getShape());
		auto outMap = identityMap(ctx, resultType.getRank());
		SmallVector<AffineMap> maps = {lhsMap, rhsMap, outMap};

		auto emptyOut =
		    rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
		auto iterTypes = parallelIterators(resultType.getRank());
		auto elemType = resultType.getElementType();
		auto kind = op.getOp();

		auto generic = rewriter.create<linalg::GenericOp>(
		    loc, TypeRange{resultType}, ValueRange{lhs, rhs}, ValueRange{emptyOut}, maps, iterTypes,
		    [&](OpBuilder& b, Location l, ValueRange args) {
			    Value result = emitBinaryScalar(b, l, kind, args[0], args[1], elemType);
			    b.create<linalg::YieldOp>(l, result);
		    });
		rewriter.replaceOp(op, generic.getResult(0));
		return success();
	}
};

struct ConvertUnaryOp : OpRewritePattern<litenn::UnaryOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(litenn::UnaryOp op, PatternRewriter& rewriter) const override
	{
		auto loc = op.getLoc();
		auto* ctx = rewriter.getContext();
		auto input = op.getInput();
		auto resultType = cast<RankedTensorType>(op.getResult().getType());
		auto inputType = cast<RankedTensorType>(input.getType());
		const int64_t rank = resultType.getRank();

		// Transpose: use linalg.transpose (permutation {1,0} for 2D)
		if (op.getOp() == UnaryOpKind::Transpose)
		{
			assert(rank == 2 && "Transpose only supports 2D tensors");
			auto emptyOut = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(),
			                                                   resultType.getElementType());
			SmallVector<int64_t> perm = {1, 0};
			rewriter.replaceOpWithNewOp<linalg::TransposeOp>(op, input, emptyOut,
			                                                   ArrayRef<int64_t>(perm));
			return success();
		}

		// Elementwise: linalg.generic with identity map
		auto map = identityMap(ctx, rank);
		auto emptyOut =
		    rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
		auto iterTypes = parallelIterators(rank);
		auto elemType = inputType.getElementType();
		auto kind = op.getOp();

		auto generic = rewriter.create<linalg::GenericOp>(
		    loc, TypeRange{resultType}, ValueRange{input}, ValueRange{emptyOut},
		    SmallVector<AffineMap>{map, map}, iterTypes,
		    [&](OpBuilder& b, Location l, ValueRange args) {
			    Value result = emitUnaryScalar(b, l, kind, args[0], elemType);
			    b.create<linalg::YieldOp>(l, result);
		    });
		rewriter.replaceOp(op, generic.getResult(0));
		return success();
	}
};

struct ConvertCastOp : OpRewritePattern<CastOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(CastOp op, PatternRewriter& rewriter) const override
	{
		auto loc = op.getLoc();
		auto* ctx = rewriter.getContext();
		auto input = op.getInput();
		auto srcType = cast<RankedTensorType>(input.getType());
		auto dstType = cast<RankedTensorType>(op.getResult().getType());
		const int64_t rank = dstType.getRank();

		auto srcElem = srcType.getElementType();
		auto dstElem = dstType.getElementType();

		if (srcElem == dstElem)
		{
			rewriter.replaceOp(op, input);
			return success();
		}

		auto map = identityMap(ctx, rank);
		auto emptyOut = rewriter.create<tensor::EmptyOp>(loc, dstType.getShape(), dstElem);
		auto iterTypes = parallelIterators(rank);

		auto generic = rewriter.create<linalg::GenericOp>(
		    loc, TypeRange{dstType}, ValueRange{input}, ValueRange{emptyOut},
		    SmallVector<AffineMap>{map, map}, iterTypes,
		    [&](OpBuilder& b, Location l, ValueRange args) {
			    Value in = args[0];
			    Value out;
			    const bool srcFloat = isa<FloatType>(srcElem);
			    const bool dstFloat = isa<FloatType>(dstElem);
			    const bool srcInt = isa<IntegerType>(srcElem);
			    const bool dstInt = isa<IntegerType>(dstElem);
			    auto srcFloatTy = dyn_cast<FloatType>(srcElem);
			    auto dstFloatTy = dyn_cast<FloatType>(dstElem);
			    auto srcIntTy = dyn_cast<IntegerType>(srcElem);
			    auto dstIntTy = dyn_cast<IntegerType>(dstElem);

			    if (srcFloat && dstFloat)
			    {
				    if (srcFloatTy.getWidth() < dstFloatTy.getWidth())
					    out = b.create<arith::ExtFOp>(l, dstElem, in);
				    else
					    out = b.create<arith::TruncFOp>(l, dstElem, in);
			    }
			    else if (srcFloat && dstInt)
			    {
				    out = b.create<arith::FPToSIOp>(l, dstElem, in);
			    }
			    else if (srcInt && dstFloat)
			    {
				    out = b.create<arith::SIToFPOp>(l, dstElem, in);
			    }
			    else // int → int
			    {
				    if (srcIntTy.getWidth() < dstIntTy.getWidth())
					    out = b.create<arith::ExtSIOp>(l, dstElem, in);
				    else if (srcIntTy.getWidth() > dstIntTy.getWidth())
					    out = b.create<arith::TruncIOp>(l, dstElem, in);
				    else
					    out = in;
			    }
			    b.create<linalg::YieldOp>(l, out);
		    });
		rewriter.replaceOp(op, generic.getResult(0));
		return success();
	}
};

//===----------------------------------------------------------------------===//
// Reduce / Reshape / Concat / Slice Patterns
//===----------------------------------------------------------------------===//

struct ConvertReduceOp : OpRewritePattern<litenn::ReduceOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(litenn::ReduceOp op, PatternRewriter& rewriter) const override
	{
		auto loc = op.getLoc();
		auto input = op.getInput();
		auto inputType = cast<RankedTensorType>(input.getType());
		auto resultType = cast<RankedTensorType>(op.getResult().getType());
		auto elemType = inputType.getElementType();
		const int64_t axis = static_cast<int64_t>(op.getAxis());
		const auto kind = op.getOp();

		// Init value: 0 for sum/mean, -inf for max
		Value initVal;
		if (kind == ReduceOpKind::Max)
		{
			if (auto ft = dyn_cast<FloatType>(elemType))
			{
				APFloat negInf = APFloat::getInf(ft.getFloatSemantics(), /*negative=*/true);
				initVal = rewriter.create<arith::ConstantFloatOp>(loc, ft, negInf);
			}
			else
			{
				// Integer max: use minimum int value
				auto intTy = cast<IntegerType>(elemType);
				initVal = rewriter.create<arith::ConstantIntOp>(
				    loc, elemType, APInt::getSignedMinValue(intTy.getWidth()).getSExtValue());
			}
		}
		else
		{
			initVal = zeroScalar(rewriter, loc, elemType);
		}

		auto emptyOut = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);
		auto filledOut =
		    rewriter.create<linalg::FillOp>(loc, ValueRange{initVal}, ValueRange{emptyOut}).getResult(0);

		auto reduceOp = rewriter.create<linalg::ReduceOp>(
		    loc, ValueRange{input}, ValueRange{filledOut},
		    ArrayRef<int64_t>{axis},
		    [&](OpBuilder& b, Location l, ValueRange args) {
			    Value acc = args[1]; // accumulator
			    Value cur = args[0]; // current element
			    Value result;
			    if (kind == ReduceOpKind::Sum || kind == ReduceOpKind::Mean)
				    result = isa<FloatType>(elemType) ? b.create<arith::AddFOp>(l, acc, cur).getResult()
				                                      : b.create<arith::AddIOp>(l, acc, cur).getResult();
			    else // Max
				    result = isa<FloatType>(elemType)
				                 ? b.create<arith::MaximumFOp>(l, acc, cur).getResult()
				                 : b.create<arith::MaxSIOp>(l, acc, cur).getResult();
			    b.create<linalg::YieldOp>(l, result);
		    });

		Value output = reduceOp.getResult(0);

		// Mean: divide by the axis size
		if (kind == ReduceOpKind::Mean)
		{
			const int64_t axisSize = inputType.getDimSize(axis);
			auto* ctx = rewriter.getContext();
			auto map = identityMap(ctx, resultType.getRank());
			auto emptyMean = rewriter.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);
			auto iterTypes = parallelIterators(resultType.getRank());
			Value divisor;
			if (isa<FloatType>(elemType))
			{
				auto ft = cast<FloatType>(elemType);
				divisor = rewriter.create<arith::ConstantFloatOp>(
				    loc, ft, APFloat(ft.getFloatSemantics(), axisSize));
			}
			else
			{
				divisor = rewriter.create<arith::ConstantIntOp>(loc, elemType, axisSize);
			}
			auto meanGeneric = rewriter.create<linalg::GenericOp>(
			    loc, TypeRange{resultType}, ValueRange{output}, ValueRange{emptyMean},
			    SmallVector<AffineMap>{map, map}, iterTypes,
			    [&](OpBuilder& b, Location l, ValueRange args) {
				    Value q = isa<FloatType>(elemType) ? b.create<arith::DivFOp>(l, args[0], divisor)
				                                             .getResult()
				                                       : b.create<arith::DivSIOp>(l, args[0], divisor)
				                                             .getResult();
				    b.create<linalg::YieldOp>(l, q);
			    });
			output = meanGeneric.getResult(0);
		}

		rewriter.replaceOp(op, output);
		return success();
	}
};

struct ConvertReshapeOp : OpRewritePattern<ReshapeOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(ReshapeOp op, PatternRewriter& rewriter) const override
	{
		auto loc = op.getLoc();
		auto input = op.getInput();
		auto srcType = cast<RankedTensorType>(input.getType());
		auto dstType = cast<RankedTensorType>(op.getResult().getType());

		// Use tensor.collapse_shape or tensor.expand_shape based on rank change.
		// For arbitrary reshapes we use tensor.reshape (requires a shape operand).
		// Build a constant shape operand from the result type's static dims.
		SmallVector<int64_t> dstShape(dstType.getShape());
		auto i64Type = rewriter.getIntegerType(64);
		auto shapeTensorType = RankedTensorType::get({static_cast<int64_t>(dstShape.size())}, i64Type);
		SmallVector<Attribute> shapeAttrs;
		for (int64_t d : dstShape)
			shapeAttrs.push_back(rewriter.getI64IntegerAttr(d));
		auto shapeAttr = DenseIntElementsAttr::get(shapeTensorType, shapeAttrs);
		auto shapeConst = rewriter.create<arith::ConstantOp>(loc, shapeAttr);

		// tensor.reshape needs a 1D memref shape; we go through tensor.reshape which
		// accepts a tensor<Nxi64> shape operand.
		(void)srcType;
		rewriter.replaceOpWithNewOp<tensor::ReshapeOp>(op, dstType, input, shapeConst);
		return success();
	}
};

struct ConvertConcatOp : OpRewritePattern<litenn::ConcatOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(litenn::ConcatOp op, PatternRewriter& rewriter) const override
	{
		auto resultType = cast<RankedTensorType>(op.getResult().getType());
		rewriter.replaceOpWithNewOp<tensor::ConcatOp>(op, resultType,
		                                               static_cast<int64_t>(op.getAxis()),
		                                               op.getInputs());
		return success();
	}
};

struct ConvertSliceOp : OpRewritePattern<SliceOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(SliceOp op, PatternRewriter& rewriter) const override
	{
		auto input = op.getInput();
		auto srcType = cast<RankedTensorType>(input.getType());
		auto dstType = cast<RankedTensorType>(op.getResult().getType());
		const int64_t rank = srcType.getRank();
		const int64_t axis = static_cast<int64_t>(op.getAxis());
		const int64_t start = static_cast<int64_t>(op.getStart());
		const int64_t length = static_cast<int64_t>(op.getLength());

		SmallVector<int64_t> offsets(rank, 0);
		SmallVector<int64_t> sizes(srcType.getShape().begin(), srcType.getShape().end());
		SmallVector<int64_t> strides(rank, 1);
		offsets[axis] = start;
		sizes[axis] = length;

		rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(op, dstType, input, ValueRange{},
		                                                     ValueRange{}, ValueRange{}, offsets,
		                                                     sizes, strides);
		return success();
	}
};

//===----------------------------------------------------------------------===//
// Control Flow Patterns
//===----------------------------------------------------------------------===//

struct ConvertCondOp : OpRewritePattern<CondOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(CondOp op, PatternRewriter& rewriter) const override
	{
		auto loc = op.getLoc();

		// Extract scalar i1 from condition tensor<1xi1>
		auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
		Value scalarCond =
		    rewriter.create<tensor::ExtractOp>(loc, op.getCondition(), ValueRange{c0});

		// Create scf.if
		auto ifOp = rewriter.create<scf::IfOp>(loc, op.getResultTypes(), scalarCond,
		                                         /*withElseRegion=*/true);

		// Helper: inline a litenn.cond region branch into the corresponding scf.if region.
		// The branch block has arguments (the args passed to cond op); in scf.if those
		// are just captured SSA values — so replace block arg uses and remove them.
		auto inlineBranch = [&](Region& src, Region& dst, ValueRange argValues) {
			Block* srcBlock = &src.front();
			Block* dstBlock = &dst.front();

			// Replace block arg uses with the captured values
			for (auto [blockArg, val] : llvm::zip(srcBlock->getArguments(), argValues))
				blockArg.replaceAllUsesWith(val);
			srcBlock->eraseArguments(0, srcBlock->getNumArguments());

			// Move ops into dst (dstBlock is initially empty, will contain them after merge)
			rewriter.mergeBlocks(srcBlock, dstBlock, {});
		};

		inlineBranch(op.getThenRegion(), ifOp.getThenRegion(), op.getArgs());
		inlineBranch(op.getElseRegion(), ifOp.getElseRegion(), op.getArgs());

		rewriter.replaceOp(op, ifOp.getResults());
		return success();
	}
};

struct ConvertWhileOp : OpRewritePattern<WhileOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(WhileOp op, PatternRewriter& rewriter) const override
	{
		auto loc = op.getLoc();

		// scf.while: before (cond check) + after (body)
		// The init_args and result types are the same as litenn.while's carry.
		auto scfWhile =
		    rewriter.create<scf::WhileOp>(loc, op.getResultTypes(), op.getInitArgs());

		// scf::WhileOp may or may not create entry blocks depending on MLIR version.
		// Ensure both regions have an entry block with the right argument types.
		SmallVector<Type> initTypes;
		SmallVector<Location> initLocs;
		for (auto arg : op.getInitArgs())
		{
			initTypes.push_back(arg.getType());
			initLocs.push_back(loc);
		}
		SmallVector<Type> resultTypes(op.getResultTypes().begin(), op.getResultTypes().end());
		SmallVector<Location> resultLocs(resultTypes.size(), loc);
		if (scfWhile.getBefore().empty())
		{
			OpBuilder::InsertionGuard g(rewriter);
			rewriter.createBlock(&scfWhile.getBefore(), {}, initTypes, initLocs);
		}
		if (scfWhile.getAfter().empty())
		{
			OpBuilder::InsertionGuard g(rewriter);
			rewriter.createBlock(&scfWhile.getAfter(), {}, resultTypes, resultLocs);
		}

		// ---- before region: merge first, then fix terminator ----
		// We merge condBlock into beforeBlock FIRST so that scf::ConditionOp is
		// created inside the correctly-parented scf::WhileOp before region.
		{
			Block* condBlock = &op.getCondRegion().front();
			Block* beforeBlock = &scfWhile.getBefore().front();

			rewriter.mergeBlocks(condBlock, beforeBlock, beforeBlock->getArguments());

			// The terminator is now inside scf::WhileOp's before region.
			// It may be scf.yield (if ConvertYieldOp fired first) or litenn.yield.
			auto* terminator = &beforeBlock->back();
			Value boolTensor = terminator->getOperand(0);
			rewriter.setInsertionPoint(terminator);
			auto c0 = rewriter.create<arith::ConstantIndexOp>(loc, 0);
			Value scalarCond = rewriter.create<tensor::ExtractOp>(loc, boolTensor, ValueRange{c0});
			rewriter.replaceOpWithNewOp<scf::ConditionOp>(terminator, scalarCond,
			                                               beforeBlock->getArguments());
		}

		// ---- after region: merge body, ensure scf.yield terminator ----
		{
			Block* bodyBlock = &op.getBodyRegion().front();
			Block* afterBlock = &scfWhile.getAfter().front();

			rewriter.mergeBlocks(bodyBlock, afterBlock, afterBlock->getArguments());

			// ConvertYieldOp may have already converted litenn.yield → scf.yield;
			// if not, convert it now.
			auto* terminator = &afterBlock->back();
			if (!isa<scf::YieldOp>(terminator))
			{
				rewriter.setInsertionPoint(terminator);
				rewriter.replaceOpWithNewOp<scf::YieldOp>(terminator, terminator->getOperands());
			}
		}

		rewriter.replaceOp(op, scfWhile.getResults());
		return success();
	}
};

struct ConvertYieldOp : OpRewritePattern<YieldOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(YieldOp op, PatternRewriter& rewriter) const override
	{
		rewriter.replaceOpWithNewOp<scf::YieldOp>(op, op.getOperands());
		return success();
	}
};

//===----------------------------------------------------------------------===//
// FusedOp Pattern
//===----------------------------------------------------------------------===//

struct ConvertFusedOp : OpRewritePattern<FusedOp>
{
	using OpRewritePattern::OpRewritePattern;
	LogicalResult matchAndRewrite(FusedOp op, PatternRewriter& rewriter) const override
	{
		Region& body = op.getBody();
		Block* bodyBlock = &body.front();

		// Replace block arg uses with the fused op's args
		for (auto [blockArg, argVal] : llvm::zip(bodyBlock->getArguments(), op.getArgs()))
			blockArg.replaceAllUsesWith(argVal);
		bodyBlock->eraseArguments(0, bodyBlock->getNumArguments());

		// The block's terminator is litenn.yield; capture its operands as replacement values
		auto* terminator = &bodyBlock->back();
		SmallVector<Value> results(terminator->getOperands().begin(), terminator->getOperands().end());
		rewriter.eraseOp(terminator);

		// Inline the body block before this op
		rewriter.inlineBlockBefore(bodyBlock, op);

		rewriter.replaceOp(op, results);
		return success();
	}
};

//===----------------------------------------------------------------------===//
// Pass Definition
//===----------------------------------------------------------------------===//

struct LowerLiteNNPass : PassWrapper<LowerLiteNNPass, OperationPass<ModuleOp>>
{
	StringRef getName() const override { return "LowerLiteNNPass"; }

	void getDependentDialects(DialectRegistry& registry) const override
	{
		registry.insert<
		    arith::ArithDialect,
		    bufferization::BufferizationDialect,
		    func::FuncDialect,
		    linalg::LinalgDialect,
		    math::MathDialect,
		    memref::MemRefDialect,
		    scf::SCFDialect,
		    tensor::TensorDialect>();
	}

	void runOnOperation() override
	{
		RewritePatternSet patterns(&getContext());
		patterns.add<
		    ConvertFuncOp,
		    ConvertReturnOp,
		    ConvertCallOp,
		    ConvertConstantOp,
		    ConvertVariableOp,
		    ConvertGetVariableOp,
		    ConvertBinaryOp,
		    ConvertUnaryOp,
		    ConvertCastOp,
		    ConvertReduceOp,
		    ConvertReshapeOp,
		    ConvertConcatOp,
		    ConvertSliceOp,
		    ConvertCondOp,
		    ConvertWhileOp,
		    ConvertYieldOp,
		    ConvertFusedOp>(&getContext());

		if (failed(applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
			signalPassFailure();
	}
};

} // namespace

namespace litenn
{

std::unique_ptr<mlir::Pass> createLowerLiteNNPass()
{
	return std::make_unique<LowerLiteNNPass>();
}

} // namespace litenn
