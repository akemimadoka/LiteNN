#include "Pass/LLVMCodegenPipeline.h"

#include <LiteNN/Quantization.h>

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
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
#include "llvm/ADT/STLExtras.h"
#include "llvm/TargetParser/Host.h"

#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

namespace litenn
{
	namespace
	{
		constexpr llvm::StringLiteral kApplyReluAttr = "litenn.apply_relu";
		constexpr llvm::StringLiteral kGGMLBlockQuantizedMatMulAttr = "litenn.ggml_block_quantized_matmul";
		constexpr llvm::StringLiteral kGGMLBlockQuantizedMatMulPreparedLayoutAttr =
		    "litenn.ggml_block_quantized_matmul_prepared_layout";
		constexpr llvm::StringLiteral kGGMLBlockGroupedQuantizedMatMulAttr =
		    "litenn.ggml_block_grouped_quantized_matmul";
		constexpr llvm::StringLiteral kGGMLBlockGroupedQuantizedMatMulProjectionCountAttr =
		    "litenn.ggml_block_grouped_quantized_matmul_projection_count";
		constexpr llvm::StringLiteral kGGMLBlockGroupedQuantizedMatMulPreparedLayoutAttr =
		    "litenn.ggml_block_grouped_quantized_matmul_prepared_layout";
		constexpr llvm::StringLiteral kGGMLBlockQuantizedGetRowsAttr = "litenn.ggml_block_quantized_get_rows";
		constexpr llvm::StringLiteral kRoPEAtPositionsBaseAttr = "litenn.rope_at_positions_base";
		constexpr llvm::StringLiteral kRoPEAtPositionsFrequencyScaleAttr = "litenn.rope_at_positions_frequency_scale";
		constexpr llvm::StringLiteral kRoPEAtPositionsHelper = "litenn_cpu_rope_at_positions_f32";
		constexpr llvm::StringLiteral kActivePrefixAttentionAttr = "litenn.active_prefix_attention";
		constexpr llvm::StringLiteral kActivePrefixAttentionKVHeadAttr = "litenn.active_prefix_attention_kv_head";
		constexpr llvm::StringLiteral kActivePrefixAttentionHelper = "litenn_cpu_active_prefix_attention_f32";
		constexpr llvm::StringLiteral kActivePrefixAttentionRank3Helper =
		    "litenn_cpu_active_prefix_attention_f32_rank3";
		constexpr llvm::StringLiteral kGroupedActivePrefixAttentionAttr = "litenn.grouped_active_prefix_attention";
		constexpr llvm::StringLiteral kGroupedActivePrefixAttentionGroupsAttr =
		    "litenn.grouped_active_prefix_attention_query_groups_per_kv_head";
		constexpr llvm::StringLiteral kGroupedActivePrefixAttentionRank3Helper =
		    "litenn_cpu_active_prefix_attention_f32_rank3_grouped";
		constexpr llvm::StringLiteral kGroupedPagedAttentionAttr = "litenn.grouped_paged_attention";
		constexpr llvm::StringLiteral kGroupedPagedAttentionGroupsAttr =
		    "litenn.grouped_paged_attention_query_groups_per_kv_head";
		constexpr llvm::StringLiteral kGroupedPagedAttentionHelper = "litenn_cpu_grouped_paged_attention_f32";
		constexpr llvm::StringLiteral kScatterUpdateAxis0F32Rank3Attr = "litenn.scatter_update_axis0_f32_rank3";
		constexpr llvm::StringLiteral kScatterUpdateAxis0F32Rank3Helper = "litenn_cpu_scatter_update_axis0_f32_rank3";
		constexpr llvm::StringLiteral kGGMLBlockMatMulHelper = "litenn_cpu_ggml_block_matmul_f32";
		constexpr llvm::StringLiteral kGGMLBlockMatMulQ8KStagedHelper = "litenn_cpu_ggml_block_matmul_q8k_staged_f32";
		constexpr llvm::StringLiteral kGGMLBlockMatMulQ4KPrepackedHelper =
		    "litenn_cpu_ggml_block_matmul_q4k_prepacked_f32";
		constexpr llvm::StringLiteral kGGMLBlockMatMulQ6KPrepackedHelper =
		    "litenn_cpu_ggml_block_matmul_q6k_prepacked_f32";
		constexpr llvm::StringLiteral kGGMLBlockMatMulCompactQ8KHelper = "litenn_cpu_ggml_block_matmul_compact_q8k_f32";
		constexpr llvm::StringLiteral kGGMLBlockGroupedMatMul2Helper = "litenn_cpu_ggml_block_grouped_matmul2_f32";
		constexpr llvm::StringLiteral kGGMLBlockGroupedMatMul3Helper = "litenn_cpu_ggml_block_grouped_matmul3_f32";
		constexpr llvm::StringLiteral kGGMLBlockGroupedMatMul2Q8KStagedHelper =
		    "litenn_cpu_ggml_block_grouped_matmul2_q8k_staged_f32";
		constexpr llvm::StringLiteral kGGMLBlockGroupedMatMul3Q8KStagedHelper =
		    "litenn_cpu_ggml_block_grouped_matmul3_q8k_staged_f32";
		constexpr llvm::StringLiteral kGGMLBlockGroupedMatMul2CompactQ8KHelper =
		    "litenn_cpu_ggml_block_grouped_matmul2_compact_q8k_f32";
		constexpr llvm::StringLiteral kGGMLBlockGroupedMatMul3CompactQ8KHelper =
		    "litenn_cpu_ggml_block_grouped_matmul3_compact_q8k_f32";
		constexpr llvm::StringLiteral kGGMLBlockGroupedMatMul2Q4KPrepackedHelper =
		    "litenn_cpu_ggml_block_grouped_matmul2_q4k_prepacked_f32";
		constexpr llvm::StringLiteral kGGMLBlockGroupedMatMul2Q6KPrepackedHelper =
		    "litenn_cpu_ggml_block_grouped_matmul2_q6k_prepacked_f32";
		constexpr llvm::StringLiteral kGGMLBlockGroupedMatMul3Q4KPrepackedHelper =
		    "litenn_cpu_ggml_block_grouped_matmul3_q4k_prepacked_f32";
		constexpr llvm::StringLiteral kGGMLBlockGroupedMatMul3Q6KPrepackedHelper =
		    "litenn_cpu_ggml_block_grouped_matmul3_q6k_prepacked_f32";
		constexpr llvm::StringLiteral kGGMLBlockGetRowsI32Helper = "litenn_cpu_ggml_block_get_rows_i32_f32";
		constexpr llvm::StringLiteral kGGMLBlockGetRowsI64Helper = "litenn_cpu_ggml_block_get_rows_i64_f32";
		constexpr std::int64_t kGGMLPreparedLayoutExpandedF32ScalesV1 = 1;
		constexpr std::int64_t kGGMLPreparedLayoutCompactBlockGroupedV3 = 3;

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

		int64_t nativeF32VectorWidth()
		{
			static const int64_t width = [] {
				const auto features = llvm::sys::getHostCPUFeatures();
				const auto hasFeature = [&](llvm::StringRef name) {
					auto it = features.find(name);
					return it != features.end() && it->second;
				};

				if (hasFeature("avx512f"))
				{
					return int64_t{ 16 };
				}
				if (hasFeature("avx2") || hasFeature("avx"))
				{
					return int64_t{ 8 };
				}
				if (hasFeature("sse2") || hasFeature("sse"))
				{
					return int64_t{ 4 };
				}
				return int64_t{ 1 };
			}();
			return width;
		}

		bool supportsGGMLQ8KStagedMatMul(LiteNN::QuantizedBlockFormat format)
		{
			return format == LiteNN::QuantizedBlockFormat::GGML_Q4_K ||
			       format == LiteNN::QuantizedBlockFormat::GGML_Q5_K ||
			       format == LiteNN::QuantizedBlockFormat::GGML_Q6_K;
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
			auto flags = mlir::arith::FastMathFlags::contract | mlir::arith::FastMathFlags::nnan |
			             mlir::arith::FastMathFlags::ninf | mlir::arith::FastMathFlags::nsz;
			return mlir::arith::FastMathFlagsAttr::get(ctx, flags);
		}

		mlir::Value applyReluIfNeeded(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value value, bool applyRelu)
		{
			if (!applyRelu)
			{
				return value;
			}

			mlir::Value zero;
			if (auto vectorType = llvm::dyn_cast<mlir::VectorType>(value.getType()))
			{
				zero = builder.create<mlir::arith::ConstantOp>(loc, vectorType, builder.getZeroAttr(vectorType))
				           .getResult();
			}
			else
			{
				auto floatType = llvm::cast<mlir::FloatType>(value.getType());
				zero = builder
				           .create<mlir::arith::ConstantFloatOp>(loc, floatType,
				                                                 llvm::APFloat::getZero(floatType.getFloatSemantics()))
				           .getResult();
			}
			auto maxOp = builder.create<mlir::arith::MaxNumFOp>(loc, value, zero);
			maxOp->setAttr(maxOp.getFastMathAttrName(), getContractOnlyFastMath(builder.getContext()));
			return maxOp.getResult();
		}

		mlir::LogicalResult validateMatMulCandidate(mlir::linalg::GenericOp op, mlir::Value& lhs, mlir::Value& rhs,
		                                            mlir::Value& out, mlir::MemRefType& lhsType,
		                                            mlir::MemRefType& rhsType, mlir::MemRefType& outType)
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
			if (!lhsType || !rhsType || !outType || lhsType.getRank() != 2 || rhsType.getRank() != 2 ||
			    outType.getRank() != 2 || !lhsType.getElementType().isF32() || !rhsType.getElementType().isF32() ||
			    !outType.getElementType().isF32())
			{
				return mlir::failure();
			}

			auto maps = op.getIndexingMapsArray();
			if (maps.size() != 3 || !isDimMap(maps[0], { 0, 1 }) || !isDimMap(maps[1], { 1, 2 }) ||
			    !isDimMap(maps[2], { 0, 2 }))
			{
				return mlir::failure();
			}

			auto iterTypes = op.getIteratorTypesArray();
			if (iterTypes.size() != 3 || iterTypes[0] != mlir::utils::IteratorType::parallel ||
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

		mlir::LogicalResult rewriteGGMLBlockQuantizedMatMulCall(mlir::ModuleOp module, mlir::linalg::GenericOp op,
		                                                        mlir::OpBuilder& builder,
		                                                        const LLVMCodegenOptions& options)
		{
			auto formatAttr = op->getAttrOfType<mlir::IntegerAttr>(kGGMLBlockQuantizedMatMulAttr);
			if (!formatAttr || op->getNumResults() != 0 || op.getInputs().size() != 1 || op.getOutputs().size() != 1)
			{
				return mlir::failure();
			}

			auto lhs = op.getInputs()[0];
			auto out = op.getOutputs()[0];
			auto lhsType = llvm::dyn_cast<mlir::MemRefType>(lhs.getType());
			auto outType = llvm::dyn_cast<mlir::MemRefType>(out.getType());
			if (!lhsType || !outType || lhsType.getRank() != 2 || outType.getRank() != 2 ||
			    !lhsType.getElementType().isF32() || !outType.getElementType().isF32())
			{
				return mlir::failure();
			}
			mlir::Value rhs;
			op.getRegion().walk([&](mlir::memref::LoadOp load) {
				if (rhs)
				{
					return;
				}
				auto candidate = load.getMemRef();
				auto candidateType = llvm::dyn_cast<mlir::MemRefType>(candidate.getType());
				if (candidateType && candidateType.getRank() == 1 && candidateType.getElementType().isInteger(8))
				{
					rhs = candidate;
				}
			});
			auto rhsType = rhs ? llvm::dyn_cast<mlir::MemRefType>(rhs.getType()) : mlir::MemRefType{};
			if (!rhs || !rhsType)
			{
				return mlir::failure();
			}

			const auto loc = op.getLoc();
			auto i64 = builder.getI64Type();
			auto dynamicLhsType = mlir::MemRefType::get({ mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic },
			                                            lhsType.getElementType());
			auto dynamicRhsType = mlir::MemRefType::get({ mlir::ShapedType::kDynamic }, rhsType.getElementType());
			auto dynamicOutType = mlir::MemRefType::get({ mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic },
			                                            outType.getElementType());
			const auto blockFormat = static_cast<LiteNN::QuantizedBlockFormat>(formatAttr.getInt());
			auto preparedLayoutAttr = op->getAttrOfType<mlir::IntegerAttr>(kGGMLBlockQuantizedMatMulPreparedLayoutAttr);
			const auto hasPreparedLayout = preparedLayoutAttr != nullptr;
			const auto usesExpandedPreparedLayout =
			    hasPreparedLayout && preparedLayoutAttr.getInt() == kGGMLPreparedLayoutExpandedF32ScalesV1;
			const auto usesCompactPreparedLayout =
			    hasPreparedLayout && preparedLayoutAttr.getInt() == kGGMLPreparedLayoutCompactBlockGroupedV3;
			llvm::StringRef helperName = kGGMLBlockMatMulHelper;
			if (hasPreparedLayout)
			{
				if (!options.enableGGMLPrepackedWeights)
				{
					return mlir::failure();
				}
				if (!usesExpandedPreparedLayout && !usesCompactPreparedLayout)
				{
					return mlir::failure();
				}
				if (usesCompactPreparedLayout && (blockFormat == LiteNN::QuantizedBlockFormat::GGML_Q4_K ||
				                                  blockFormat == LiteNN::QuantizedBlockFormat::GGML_Q6_K))
				{
					helperName = kGGMLBlockMatMulCompactQ8KHelper;
				}
				else if (usesExpandedPreparedLayout && blockFormat == LiteNN::QuantizedBlockFormat::GGML_Q4_K)
				{
					helperName = kGGMLBlockMatMulQ4KPrepackedHelper;
				}
				else if (usesExpandedPreparedLayout && blockFormat == LiteNN::QuantizedBlockFormat::GGML_Q6_K)
				{
					helperName = kGGMLBlockMatMulQ6KPrepackedHelper;
				}
				else
				{
					return mlir::failure();
				}
			}
			else if (options.enableGGMLQ8KStagedMatMul && supportsGGMLQ8KStagedMatMul(blockFormat))
			{
				helperName = kGGMLBlockMatMulQ8KStagedHelper;
			}
			auto funcType = usesExpandedPreparedLayout
			                    ? builder.getFunctionType(
			                          mlir::TypeRange{ dynamicLhsType, dynamicRhsType, dynamicOutType, i64, i64 },
			                          mlir::TypeRange{})
			                    : builder.getFunctionType(
			                          mlir::TypeRange{ dynamicLhsType, dynamicRhsType, dynamicOutType, i64, i64, i64 },
			                          mlir::TypeRange{});
			auto helper = module.lookupSymbol<mlir::func::FuncOp>(helperName);
			if (!helper)
			{
				mlir::OpBuilder::InsertionGuard guard(builder);
				builder.setInsertionPointToStart(module.getBody());
				helper = builder.create<mlir::func::FuncOp>(loc, helperName, funcType);
				helper.setPrivate();
			}
			else if (helper.getFunctionType() != funcType)
			{
				return mlir::failure();
			}

			auto format = builder.create<mlir::arith::ConstantIntOp>(loc, formatAttr.getInt(), 64).getResult();
			auto threadCount =
			    builder.create<mlir::arith::ConstantIntOp>(loc, options.ggmlBlockMatMulThreadCount, 64).getResult();
			auto affinityPolicy =
			    builder.create<mlir::arith::ConstantIntOp>(loc, options.ggmlBlockMatMulAffinityPolicy, 64).getResult();
			auto dynamicLhs = builder.create<mlir::memref::CastOp>(loc, dynamicLhsType, lhs).getResult();
			auto dynamicRhs = builder.create<mlir::memref::CastOp>(loc, dynamicRhsType, rhs).getResult();
			auto dynamicOut = builder.create<mlir::memref::CastOp>(loc, dynamicOutType, out).getResult();
			if (usesExpandedPreparedLayout)
			{
				builder.create<mlir::func::CallOp>(
				    loc, helper, mlir::ValueRange{ dynamicLhs, dynamicRhs, dynamicOut, threadCount, affinityPolicy });
			}
			else
			{
				builder.create<mlir::func::CallOp>(
				    loc, helper,
				    mlir::ValueRange{ dynamicLhs, dynamicRhs, dynamicOut, format, threadCount, affinityPolicy });
			}
			op.erase();
			return mlir::success();
		}

		mlir::LogicalResult rewriteGGMLBlockGroupedQuantizedMatMulCall(mlir::ModuleOp module,
		                                                               mlir::linalg::GenericOp op,
		                                                               mlir::OpBuilder& builder,
		                                                               const LLVMCodegenOptions& options)
		{
			auto formatAttr = op->getAttrOfType<mlir::IntegerAttr>(kGGMLBlockGroupedQuantizedMatMulAttr);
			auto projectionCountAttr =
			    op->getAttrOfType<mlir::IntegerAttr>(kGGMLBlockGroupedQuantizedMatMulProjectionCountAttr);
			if (!formatAttr || !projectionCountAttr || op->getNumResults() != 0 || op.getOutputs().size() != 1)
			{
				return mlir::failure();
			}
			const auto projectionCount = static_cast<std::size_t>(projectionCountAttr.getInt());
			if ((projectionCount != 2 && projectionCount != 3) || op.getInputs().size() != 1)
			{
				return mlir::failure();
			}

			auto lhs = op.getInputs().front();
			auto out = op.getOutputs().front();
			auto lhsType = llvm::dyn_cast<mlir::MemRefType>(lhs.getType());
			auto outType = llvm::dyn_cast<mlir::MemRefType>(out.getType());
			if (!lhsType || !outType || lhsType.getRank() != 2 || outType.getRank() != 2 ||
			    !lhsType.getElementType().isF32() || !outType.getElementType().isF32())
			{
				return mlir::failure();
			}

			constexpr std::array<llvm::StringLiteral, 3> widthAttrs{
				"litenn.ggml_block_grouped_quantized_matmul_output_width0",
				"litenn.ggml_block_grouped_quantized_matmul_output_width1",
				"litenn.ggml_block_grouped_quantized_matmul_output_width2",
			};
			llvm::SmallVector<std::int64_t, 3> outputWidths;
			for (std::size_t i = 0; i < projectionCount; ++i)
			{
				auto widthAttr = op->getAttrOfType<mlir::IntegerAttr>(widthAttrs[i]);
				if (!widthAttr || widthAttr.getInt() <= 0)
				{
					return mlir::failure();
				}
				outputWidths.push_back(widthAttr.getInt());
			}

			llvm::SmallVector<mlir::Value, 3> rhsValues;
			op.getRegion().walk([&](mlir::memref::LoadOp load) {
				auto candidate = load.getMemRef();
				auto candidateType = llvm::dyn_cast<mlir::MemRefType>(candidate.getType());
				if (!candidateType || candidateType.getRank() != 1 || !candidateType.getElementType().isInteger(8))
				{
					return;
				}
				if (llvm::is_contained(rhsValues, candidate))
				{
					return;
				}
				rhsValues.push_back(candidate);
			});
			if (rhsValues.size() != projectionCount)
			{
				return mlir::failure();
			}

			const auto loc = op.getLoc();
			auto i64 = builder.getI64Type();
			auto dynamicLhsType = mlir::MemRefType::get({ mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic },
			                                            lhsType.getElementType());
			auto dynamicRhsType =
			    mlir::MemRefType::get({ mlir::ShapedType::kDynamic },
			                          llvm::cast<mlir::MemRefType>(rhsValues.front().getType()).getElementType());
			auto dynamicOutType = mlir::MemRefType::get({ mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic },
			                                            outType.getElementType());
			const auto blockFormat = static_cast<LiteNN::QuantizedBlockFormat>(formatAttr.getInt());
			auto preparedLayoutAttr =
			    op->getAttrOfType<mlir::IntegerAttr>(kGGMLBlockGroupedQuantizedMatMulPreparedLayoutAttr);
			const auto hasPreparedLayout = preparedLayoutAttr != nullptr;
			const auto usesExpandedPreparedLayout =
			    hasPreparedLayout && preparedLayoutAttr.getInt() == kGGMLPreparedLayoutExpandedF32ScalesV1;
			const auto usesCompactPreparedLayout =
			    hasPreparedLayout && preparedLayoutAttr.getInt() == kGGMLPreparedLayoutCompactBlockGroupedV3;
			llvm::StringRef helperName;
			if (hasPreparedLayout)
			{
				if (!options.enableGGMLPrepackedWeights)
				{
					return mlir::failure();
				}
				if (!usesExpandedPreparedLayout && !usesCompactPreparedLayout)
				{
					return mlir::failure();
				}
				if (usesCompactPreparedLayout && (blockFormat == LiteNN::QuantizedBlockFormat::GGML_Q4_K ||
				                                  blockFormat == LiteNN::QuantizedBlockFormat::GGML_Q6_K))
				{
					helperName = projectionCount == 2 ? kGGMLBlockGroupedMatMul2CompactQ8KHelper
					                                  : kGGMLBlockGroupedMatMul3CompactQ8KHelper;
				}
				else if (usesExpandedPreparedLayout && projectionCount == 2 &&
				         blockFormat == LiteNN::QuantizedBlockFormat::GGML_Q4_K)
				{
					helperName = kGGMLBlockGroupedMatMul2Q4KPrepackedHelper;
				}
				else if (usesExpandedPreparedLayout && projectionCount == 2 &&
				         blockFormat == LiteNN::QuantizedBlockFormat::GGML_Q6_K)
				{
					helperName = kGGMLBlockGroupedMatMul2Q6KPrepackedHelper;
				}
				else if (usesExpandedPreparedLayout && projectionCount == 3 &&
				         blockFormat == LiteNN::QuantizedBlockFormat::GGML_Q4_K)
				{
					helperName = kGGMLBlockGroupedMatMul3Q4KPrepackedHelper;
				}
				else if (usesExpandedPreparedLayout && projectionCount == 3 &&
				         blockFormat == LiteNN::QuantizedBlockFormat::GGML_Q6_K)
				{
					helperName = kGGMLBlockGroupedMatMul3Q6KPrepackedHelper;
				}
				else
				{
					return mlir::failure();
				}
			}
			else
			{
				const auto enableQ8KStaged =
				    options.enableGGMLQ8KStagedMatMul && supportsGGMLQ8KStagedMatMul(blockFormat);
				if (enableQ8KStaged && projectionCount == 2)
				{
					helperName = kGGMLBlockGroupedMatMul2Q8KStagedHelper;
				}
				else if (enableQ8KStaged && projectionCount == 3)
				{
					helperName = kGGMLBlockGroupedMatMul3Q8KStagedHelper;
				}
				else
				{
					helperName = projectionCount == 2 ? kGGMLBlockGroupedMatMul2Helper : kGGMLBlockGroupedMatMul3Helper;
				}
			}
			llvm::SmallVector<mlir::Type> argTypes;
			argTypes.push_back(dynamicLhsType);
			for (std::size_t i = 0; i < projectionCount; ++i)
			{
				argTypes.push_back(dynamicRhsType);
			}
			argTypes.push_back(dynamicOutType);
			const auto scalarArgumentCount = projectionCount + (usesExpandedPreparedLayout ? 2 : 3);
			for (std::size_t i = 0; i < scalarArgumentCount; ++i)
			{
				argTypes.push_back(i64);
			}
			auto funcType = builder.getFunctionType(argTypes, mlir::TypeRange{});
			auto helper = module.lookupSymbol<mlir::func::FuncOp>(helperName);
			if (!helper)
			{
				mlir::OpBuilder::InsertionGuard guard(builder);
				builder.setInsertionPointToStart(module.getBody());
				helper = builder.create<mlir::func::FuncOp>(loc, helperName, funcType);
				helper.setPrivate();
			}
			else if (helper.getFunctionType() != funcType)
			{
				return mlir::failure();
			}

			llvm::SmallVector<mlir::Value> callArgs;
			callArgs.push_back(builder.create<mlir::memref::CastOp>(loc, dynamicLhsType, lhs).getResult());
			for (auto rhs : rhsValues)
			{
				callArgs.push_back(builder.create<mlir::memref::CastOp>(loc, dynamicRhsType, rhs).getResult());
			}
			callArgs.push_back(builder.create<mlir::memref::CastOp>(loc, dynamicOutType, out).getResult());
			if (!usesExpandedPreparedLayout)
			{
				callArgs.push_back(
				    builder.create<mlir::arith::ConstantIntOp>(loc, formatAttr.getInt(), 64).getResult());
			}
			for (const auto width : outputWidths)
			{
				callArgs.push_back(builder.create<mlir::arith::ConstantIntOp>(loc, width, 64).getResult());
			}
			callArgs.push_back(
			    builder.create<mlir::arith::ConstantIntOp>(loc, options.ggmlBlockMatMulThreadCount, 64).getResult());
			callArgs.push_back(
			    builder.create<mlir::arith::ConstantIntOp>(loc, options.ggmlBlockMatMulAffinityPolicy, 64).getResult());
			builder.create<mlir::func::CallOp>(loc, helper, callArgs);
			op.erase();
			return mlir::success();
		}

		mlir::LogicalResult rewriteGGMLBlockQuantizedGetRowsCall(mlir::ModuleOp module, mlir::linalg::GenericOp op,
		                                                         mlir::OpBuilder& builder)
		{
			auto formatAttr = op->getAttrOfType<mlir::IntegerAttr>(kGGMLBlockQuantizedGetRowsAttr);
			if (!formatAttr || op->getNumResults() != 0 || op.getInputs().size() != 1 || op.getOutputs().size() != 1)
			{
				return mlir::failure();
			}

			auto indices = op.getInputs()[0];
			auto out = op.getOutputs()[0];
			auto indicesType = llvm::dyn_cast<mlir::MemRefType>(indices.getType());
			auto outType = llvm::dyn_cast<mlir::MemRefType>(out.getType());
			if (!indicesType || !outType || indicesType.getRank() != 1 || outType.getRank() != 2 ||
			    !outType.getElementType().isF32() ||
			    (!indicesType.getElementType().isInteger(32) && !indicesType.getElementType().isInteger(64)))
			{
				return mlir::failure();
			}

			mlir::Value storage;
			op.getRegion().walk([&](mlir::memref::LoadOp load) {
				if (storage)
				{
					return;
				}
				auto candidate = load.getMemRef();
				auto candidateType = llvm::dyn_cast<mlir::MemRefType>(candidate.getType());
				if (candidateType && candidateType.getRank() == 1 && candidateType.getElementType().isInteger(8))
				{
					storage = candidate;
				}
			});
			auto storageType = storage ? llvm::dyn_cast<mlir::MemRefType>(storage.getType()) : mlir::MemRefType{};
			if (!storage || !storageType)
			{
				return mlir::failure();
			}

			const auto loc = op.getLoc();
			auto i64 = builder.getI64Type();
			auto dynamicStorageType =
			    mlir::MemRefType::get({ mlir::ShapedType::kDynamic }, storageType.getElementType());
			auto dynamicIndicesType =
			    mlir::MemRefType::get({ mlir::ShapedType::kDynamic }, indicesType.getElementType());
			auto dynamicOutType = mlir::MemRefType::get({ mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic },
			                                            outType.getElementType());
			const auto helperName =
			    indicesType.getElementType().isInteger(32) ? kGGMLBlockGetRowsI32Helper : kGGMLBlockGetRowsI64Helper;
			auto funcType = builder.getFunctionType(
			    mlir::TypeRange{ dynamicStorageType, dynamicIndicesType, dynamicOutType, i64 }, mlir::TypeRange{});
			auto helper = module.lookupSymbol<mlir::func::FuncOp>(helperName);
			if (!helper)
			{
				mlir::OpBuilder::InsertionGuard guard(builder);
				builder.setInsertionPointToStart(module.getBody());
				helper = builder.create<mlir::func::FuncOp>(loc, helperName, funcType);
				helper.setPrivate();
			}
			else if (helper.getFunctionType() != funcType)
			{
				return mlir::failure();
			}

			auto dynamicStorage = builder.create<mlir::memref::CastOp>(loc, dynamicStorageType, storage).getResult();
			auto dynamicIndices = builder.create<mlir::memref::CastOp>(loc, dynamicIndicesType, indices).getResult();
			auto dynamicOut = builder.create<mlir::memref::CastOp>(loc, dynamicOutType, out).getResult();
			auto format = builder.create<mlir::arith::ConstantIntOp>(loc, formatAttr.getInt(), 64).getResult();
			builder.create<mlir::func::CallOp>(loc, helper,
			                                   mlir::ValueRange{ dynamicStorage, dynamicIndices, dynamicOut, format });
			op.erase();
			return mlir::success();
		}

		mlir::LogicalResult rewriteRoPEAtPositionsCall(mlir::ModuleOp module, mlir::linalg::GenericOp op,
		                                               mlir::OpBuilder& builder)
		{
			auto baseAttr = op->getAttrOfType<mlir::FloatAttr>(kRoPEAtPositionsBaseAttr);
			auto frequencyScaleAttr = op->getAttrOfType<mlir::FloatAttr>(kRoPEAtPositionsFrequencyScaleAttr);
			if (!baseAttr || !frequencyScaleAttr || op->getNumResults() != 0 || op.getInputs().size() != 2 ||
			    op.getOutputs().size() != 1)
			{
				return mlir::failure();
			}
			auto input = op.getInputs()[0];
			auto positions = op.getInputs()[1];
			auto out = op.getOutputs()[0];
			auto inputType = llvm::dyn_cast<mlir::MemRefType>(input.getType());
			auto positionType = llvm::dyn_cast<mlir::MemRefType>(positions.getType());
			auto outType = llvm::dyn_cast<mlir::MemRefType>(out.getType());
			if (!inputType || !positionType || !outType || inputType.getRank() != 2 || positionType.getRank() != 1 ||
			    outType.getRank() != 2 || !inputType.getElementType().isF32() ||
			    !positionType.getElementType().isInteger(64) || !outType.getElementType().isF32())
			{
				return mlir::failure();
			}

			const auto loc = op.getLoc();
			auto f64 = builder.getF64Type();
			auto* mlirContext = builder.getContext();
			auto dynamicLayoutRank1 =
			    mlir::StridedLayoutAttr::get(mlirContext, mlir::ShapedType::kDynamic, { mlir::ShapedType::kDynamic });
			auto dynamicLayoutRank2 = mlir::StridedLayoutAttr::get(
			    mlirContext, mlir::ShapedType::kDynamic, { mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic });
			auto dynamicF32Rank2 = mlir::MemRefType::get({ mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic },
			                                             inputType.getElementType(), dynamicLayoutRank2);
			auto dynamicI64Rank1 = mlir::MemRefType::get({ mlir::ShapedType::kDynamic }, positionType.getElementType(),
			                                             dynamicLayoutRank1);
			auto funcType = builder.getFunctionType(
			    mlir::TypeRange{ dynamicF32Rank2, dynamicI64Rank1, dynamicF32Rank2, f64, f64 }, mlir::TypeRange{});
			auto helper = module.lookupSymbol<mlir::func::FuncOp>(kRoPEAtPositionsHelper);
			if (!helper)
			{
				mlir::OpBuilder::InsertionGuard guard(builder);
				builder.setInsertionPointToStart(module.getBody());
				helper = builder.create<mlir::func::FuncOp>(loc, kRoPEAtPositionsHelper, funcType);
				helper.setPrivate();
			}
			else if (helper.getFunctionType() != funcType)
			{
				return mlir::failure();
			}

			auto dynamicInput = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank2, input).getResult();
			auto dynamicPositions = builder.create<mlir::memref::CastOp>(loc, dynamicI64Rank1, positions).getResult();
			auto dynamicOut = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank2, out).getResult();
			auto base = builder.create<mlir::arith::ConstantFloatOp>(loc, f64, baseAttr.getValue()).getResult();
			auto frequencyScale =
			    builder.create<mlir::arith::ConstantFloatOp>(loc, f64, frequencyScaleAttr.getValue()).getResult();
			builder.create<mlir::func::CallOp>(
			    loc, helper, mlir::ValueRange{ dynamicInput, dynamicPositions, dynamicOut, base, frequencyScale });
			op.erase();
			return mlir::success();
		}

		mlir::LogicalResult rewriteActivePrefixAttentionCall(mlir::ModuleOp module, mlir::linalg::GenericOp op,
		                                                     mlir::OpBuilder& builder)
		{
			auto scaleAttr = op->getAttrOfType<mlir::FloatAttr>(kActivePrefixAttentionAttr);
			auto kvHeadAttr = op->getAttrOfType<mlir::IntegerAttr>(kActivePrefixAttentionKVHeadAttr);
			if (!scaleAttr || op->getNumResults() != 0 || op.getInputs().size() != 4 || op.getOutputs().size() != 1)
			{
				return mlir::failure();
			}
			auto query = op.getInputs()[0];
			auto keys = op.getInputs()[1];
			auto values = op.getInputs()[2];
			auto position = op.getInputs()[3];
			auto out = op.getOutputs()[0];
			auto queryType = llvm::dyn_cast<mlir::MemRefType>(query.getType());
			auto keysType = llvm::dyn_cast<mlir::MemRefType>(keys.getType());
			auto valuesType = llvm::dyn_cast<mlir::MemRefType>(values.getType());
			auto positionType = llvm::dyn_cast<mlir::MemRefType>(position.getType());
			auto outType = llvm::dyn_cast<mlir::MemRefType>(out.getType());
			if (!queryType || !keysType || !valuesType || !positionType || !outType || queryType.getRank() != 2 ||
			    !((keysType.getRank() == 2 && valuesType.getRank() == 2) ||
			      (keysType.getRank() == 3 && valuesType.getRank() == 3)) ||
			    positionType.getRank() != 1 || outType.getRank() != 2 || !queryType.getElementType().isF32() ||
			    !keysType.getElementType().isF32() || !valuesType.getElementType().isF32() ||
			    !outType.getElementType().isF32() || !positionType.getElementType().isInteger(64))
			{
				return mlir::failure();
			}

			const auto loc = op.getLoc();
			auto f64 = builder.getF64Type();
			auto* mlirContext = builder.getContext();
			auto dynamicLayoutRank1 =
			    mlir::StridedLayoutAttr::get(mlirContext, mlir::ShapedType::kDynamic, { mlir::ShapedType::kDynamic });
			auto dynamicLayoutRank2 = mlir::StridedLayoutAttr::get(
			    mlirContext, mlir::ShapedType::kDynamic, { mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic });
			auto dynamicLayoutRank3 = mlir::StridedLayoutAttr::get(
			    mlirContext, mlir::ShapedType::kDynamic,
			    { mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic });
			auto dynamicF32Rank2 = mlir::MemRefType::get({ mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic },
			                                             queryType.getElementType(), dynamicLayoutRank2);
			auto dynamicF32Rank3 = mlir::MemRefType::get(
			    { mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic },
			    keysType.getElementType(), dynamicLayoutRank3);
			auto dynamicI64Rank1 = mlir::MemRefType::get({ mlir::ShapedType::kDynamic }, positionType.getElementType(),
			                                             dynamicLayoutRank1);
			const bool rank3 = keysType.getRank() == 3;
			llvm::SmallVector<mlir::Type, 8> funcArgTypes;
			funcArgTypes.push_back(dynamicF32Rank2);
			funcArgTypes.push_back(rank3 ? mlir::Type(dynamicF32Rank3) : mlir::Type(dynamicF32Rank2));
			funcArgTypes.push_back(rank3 ? mlir::Type(dynamicF32Rank3) : mlir::Type(dynamicF32Rank2));
			funcArgTypes.push_back(dynamicI64Rank1);
			funcArgTypes.push_back(dynamicF32Rank2);
			funcArgTypes.push_back(f64);
			if (rank3)
			{
				funcArgTypes.push_back(builder.getI64Type());
			}
			auto funcType = builder.getFunctionType(funcArgTypes, mlir::TypeRange{});
			auto helperName = rank3 ? kActivePrefixAttentionRank3Helper : kActivePrefixAttentionHelper;
			auto helper = module.lookupSymbol<mlir::func::FuncOp>(helperName);
			if (!helper)
			{
				mlir::OpBuilder::InsertionGuard guard(builder);
				builder.setInsertionPointToStart(module.getBody());
				helper = builder.create<mlir::func::FuncOp>(loc, helperName, funcType);
				helper.setPrivate();
			}
			else if (helper.getFunctionType() != funcType)
			{
				return mlir::failure();
			}

			auto dynamicQuery = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank2, query).getResult();
			auto dynamicPosition = builder.create<mlir::memref::CastOp>(loc, dynamicI64Rank1, position).getResult();
			auto dynamicOut = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank2, out).getResult();
			auto scale = builder.create<mlir::arith::ConstantFloatOp>(loc, f64, scaleAttr.getValue()).getResult();
			if (rank3)
			{
				auto dynamicKeys = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank3, keys).getResult();
				auto dynamicValues = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank3, values).getResult();
				auto kvHead = builder
				                  .create<mlir::arith::ConstantIntOp>(loc, builder.getI64Type(),
				                                                      kvHeadAttr ? kvHeadAttr.getInt() : 0)
				                  .getResult();
				llvm::SmallVector<mlir::Value, 8> callArgs;
				callArgs.push_back(dynamicQuery);
				callArgs.push_back(dynamicKeys);
				callArgs.push_back(dynamicValues);
				callArgs.push_back(dynamicPosition);
				callArgs.push_back(dynamicOut);
				callArgs.push_back(scale);
				callArgs.push_back(kvHead);
				builder.create<mlir::func::CallOp>(loc, helper, callArgs);
			}
			else
			{
				auto dynamicKeys = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank2, keys).getResult();
				auto dynamicValues = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank2, values).getResult();
				llvm::SmallVector<mlir::Value, 8> callArgs;
				callArgs.push_back(dynamicQuery);
				callArgs.push_back(dynamicKeys);
				callArgs.push_back(dynamicValues);
				callArgs.push_back(dynamicPosition);
				callArgs.push_back(dynamicOut);
				callArgs.push_back(scale);
				builder.create<mlir::func::CallOp>(loc, helper, callArgs);
			}
			op.erase();
			return mlir::success();
		}

		mlir::LogicalResult rewriteGroupedActivePrefixAttentionCall(mlir::ModuleOp module, mlir::linalg::GenericOp op,
		                                                            mlir::OpBuilder& builder)
		{
			auto scaleAttr = op->getAttrOfType<mlir::FloatAttr>(kGroupedActivePrefixAttentionAttr);
			auto groupsAttr = op->getAttrOfType<mlir::IntegerAttr>(kGroupedActivePrefixAttentionGroupsAttr);
			if (!scaleAttr || !groupsAttr || op->getNumResults() != 0 || op.getInputs().size() != 4 ||
			    op.getOutputs().size() != 1)
			{
				return mlir::failure();
			}
			auto query = op.getInputs()[0];
			auto keys = op.getInputs()[1];
			auto values = op.getInputs()[2];
			auto position = op.getInputs()[3];
			auto out = op.getOutputs()[0];
			auto queryType = llvm::dyn_cast<mlir::MemRefType>(query.getType());
			auto keysType = llvm::dyn_cast<mlir::MemRefType>(keys.getType());
			auto valuesType = llvm::dyn_cast<mlir::MemRefType>(values.getType());
			auto positionType = llvm::dyn_cast<mlir::MemRefType>(position.getType());
			auto outType = llvm::dyn_cast<mlir::MemRefType>(out.getType());
			if (!queryType || !keysType || !valuesType || !positionType || !outType || queryType.getRank() != 2 ||
			    keysType.getRank() != 3 || valuesType.getRank() != 3 || positionType.getRank() != 1 ||
			    outType.getRank() != 2 || !queryType.getElementType().isF32() || !keysType.getElementType().isF32() ||
			    !valuesType.getElementType().isF32() || !outType.getElementType().isF32() ||
			    !positionType.getElementType().isInteger(64))
			{
				return mlir::failure();
			}

			const auto loc = op.getLoc();
			auto f64 = builder.getF64Type();
			auto* mlirContext = builder.getContext();
			auto dynamicLayoutRank1 =
			    mlir::StridedLayoutAttr::get(mlirContext, mlir::ShapedType::kDynamic, { mlir::ShapedType::kDynamic });
			auto dynamicLayoutRank2 = mlir::StridedLayoutAttr::get(
			    mlirContext, mlir::ShapedType::kDynamic, { mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic });
			auto dynamicLayoutRank3 = mlir::StridedLayoutAttr::get(
			    mlirContext, mlir::ShapedType::kDynamic,
			    { mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic });
			auto dynamicF32Rank2 = mlir::MemRefType::get({ mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic },
			                                             queryType.getElementType(), dynamicLayoutRank2);
			auto dynamicF32Rank3 = mlir::MemRefType::get(
			    { mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic },
			    keysType.getElementType(), dynamicLayoutRank3);
			auto dynamicI64Rank1 = mlir::MemRefType::get({ mlir::ShapedType::kDynamic }, positionType.getElementType(),
			                                             dynamicLayoutRank1);
			auto funcType =
			    builder.getFunctionType(mlir::TypeRange{ dynamicF32Rank2, dynamicF32Rank3, dynamicF32Rank3,
			                                             dynamicI64Rank1, dynamicF32Rank2, f64, builder.getI64Type() },
			                            mlir::TypeRange{});
			auto helper = module.lookupSymbol<mlir::func::FuncOp>(kGroupedActivePrefixAttentionRank3Helper);
			if (!helper)
			{
				mlir::OpBuilder::InsertionGuard guard(builder);
				builder.setInsertionPointToStart(module.getBody());
				helper = builder.create<mlir::func::FuncOp>(loc, kGroupedActivePrefixAttentionRank3Helper, funcType);
				helper.setPrivate();
			}
			else if (helper.getFunctionType() != funcType)
			{
				return mlir::failure();
			}

			auto dynamicQuery = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank2, query).getResult();
			auto dynamicKeys = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank3, keys).getResult();
			auto dynamicValues = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank3, values).getResult();
			auto dynamicPosition = builder.create<mlir::memref::CastOp>(loc, dynamicI64Rank1, position).getResult();
			auto dynamicOut = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank2, out).getResult();
			auto scale = builder.create<mlir::arith::ConstantFloatOp>(loc, f64, scaleAttr.getValue()).getResult();
			auto groups =
			    builder.create<mlir::arith::ConstantIntOp>(loc, builder.getI64Type(), groupsAttr.getInt()).getResult();
			builder.create<mlir::func::CallOp>(loc, helper,
			                                   mlir::ValueRange{ dynamicQuery, dynamicKeys, dynamicValues,
			                                                     dynamicPosition, dynamicOut, scale, groups });
			op.erase();
			return mlir::success();
		}

		mlir::LogicalResult rewriteGroupedPagedAttentionCall(mlir::ModuleOp module, mlir::linalg::GenericOp op,
		                                                     mlir::OpBuilder& builder)
		{
			auto scaleAttr = op->getAttrOfType<mlir::FloatAttr>(kGroupedPagedAttentionAttr);
			auto groupsAttr = op->getAttrOfType<mlir::IntegerAttr>(kGroupedPagedAttentionGroupsAttr);
			if (!scaleAttr || !groupsAttr || op->getNumResults() != 0 || op.getInputs().size() != 5 ||
			    op.getOutputs().size() != 1)
			{
				return mlir::failure();
			}
			auto queries = op.getInputs()[0];
			auto kvState = op.getInputs()[1];
			auto pageTable = op.getInputs()[2];
			auto pageDescriptors = op.getInputs()[3];
			auto activeLength = op.getInputs()[4];
			auto out = op.getOutputs()[0];
			auto queriesType = llvm::dyn_cast<mlir::MemRefType>(queries.getType());
			auto kvStateType = llvm::dyn_cast<mlir::MemRefType>(kvState.getType());
			auto pageTableType = llvm::dyn_cast<mlir::MemRefType>(pageTable.getType());
			auto pageDescriptorsType = llvm::dyn_cast<mlir::MemRefType>(pageDescriptors.getType());
			auto activeLengthType = llvm::dyn_cast<mlir::MemRefType>(activeLength.getType());
			auto outType = llvm::dyn_cast<mlir::MemRefType>(out.getType());
			if (!queriesType || !kvStateType || !pageTableType || !pageDescriptorsType || !activeLengthType ||
			    !outType || queriesType.getRank() != 2 || kvStateType.getRank() != 5 || pageTableType.getRank() != 1 ||
			    pageDescriptorsType.getRank() != 2 || activeLengthType.getRank() != 1 || outType.getRank() != 2 ||
			    !queriesType.getElementType().isF32() || !kvStateType.getElementType().isF32() ||
			    !outType.getElementType().isF32() || !pageTableType.getElementType().isInteger(64) ||
			    !pageDescriptorsType.getElementType().isInteger(64) || !activeLengthType.getElementType().isInteger(64))
			{
				return mlir::failure();
			}

			const auto loc = op.getLoc();
			auto f64 = builder.getF64Type();
			auto* mlirContext = builder.getContext();
			auto dynamicLayoutRank1 =
			    mlir::StridedLayoutAttr::get(mlirContext, mlir::ShapedType::kDynamic, { mlir::ShapedType::kDynamic });
			auto dynamicLayoutRank2 = mlir::StridedLayoutAttr::get(
			    mlirContext, mlir::ShapedType::kDynamic, { mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic });
			auto dynamicLayoutRank5 = mlir::StridedLayoutAttr::get(
			    mlirContext, mlir::ShapedType::kDynamic,
			    { mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic,
			      mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic });
			auto dynamicF32Rank2 = mlir::MemRefType::get({ mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic },
			                                             queriesType.getElementType(), dynamicLayoutRank2);
			auto dynamicF32Rank5 = mlir::MemRefType::get({ mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic,
			                                               mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic,
			                                               mlir::ShapedType::kDynamic },
			                                             kvStateType.getElementType(), dynamicLayoutRank5);
			auto dynamicI64Rank1 = mlir::MemRefType::get({ mlir::ShapedType::kDynamic }, pageTableType.getElementType(),
			                                             dynamicLayoutRank1);
			auto dynamicI64Rank2 = mlir::MemRefType::get({ mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic },
			                                             pageDescriptorsType.getElementType(), dynamicLayoutRank2);
			auto funcType = builder.getFunctionType(mlir::TypeRange{ dynamicF32Rank2, dynamicF32Rank5, dynamicI64Rank1,
			                                                         dynamicI64Rank2, dynamicI64Rank1, dynamicF32Rank2,
			                                                         f64, builder.getI64Type() },
			                                        mlir::TypeRange{});
			auto helper = module.lookupSymbol<mlir::func::FuncOp>(kGroupedPagedAttentionHelper);
			if (!helper)
			{
				mlir::OpBuilder::InsertionGuard guard(builder);
				builder.setInsertionPointToStart(module.getBody());
				helper = builder.create<mlir::func::FuncOp>(loc, kGroupedPagedAttentionHelper, funcType);
				helper.setPrivate();
			}
			else if (helper.getFunctionType() != funcType)
			{
				return mlir::failure();
			}

			auto dynamicQueries = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank2, queries).getResult();
			auto dynamicKVState = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank5, kvState).getResult();
			auto dynamicPageTable = builder.create<mlir::memref::CastOp>(loc, dynamicI64Rank1, pageTable).getResult();
			auto dynamicPageDescriptors =
			    builder.create<mlir::memref::CastOp>(loc, dynamicI64Rank2, pageDescriptors).getResult();
			auto dynamicActiveLength =
			    builder.create<mlir::memref::CastOp>(loc, dynamicI64Rank1, activeLength).getResult();
			auto dynamicOut = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank2, out).getResult();
			auto scale = builder.create<mlir::arith::ConstantFloatOp>(loc, f64, scaleAttr.getValue()).getResult();
			auto groups =
			    builder.create<mlir::arith::ConstantIntOp>(loc, builder.getI64Type(), groupsAttr.getInt()).getResult();
			builder.create<mlir::func::CallOp>(loc, helper,
			                                   mlir::ValueRange{ dynamicQueries, dynamicKVState, dynamicPageTable,
			                                                     dynamicPageDescriptors, dynamicActiveLength,
			                                                     dynamicOut, scale, groups });
			op.erase();
			return mlir::success();
		}

		mlir::LogicalResult rewriteScatterUpdateAxis0F32Rank3Call(mlir::ModuleOp module, mlir::linalg::GenericOp op,
		                                                          mlir::OpBuilder& builder)
		{
			if (!op->hasAttr(kScatterUpdateAxis0F32Rank3Attr) || op->getNumResults() != 0 ||
			    op.getInputs().size() != 3 || op.getOutputs().size() != 1)
			{
				return mlir::failure();
			}
			auto data = op.getInputs()[0];
			auto indices = op.getInputs()[1];
			auto updates = op.getInputs()[2];
			auto out = op.getOutputs()[0];
			auto dataType = llvm::dyn_cast<mlir::MemRefType>(data.getType());
			auto indicesType = llvm::dyn_cast<mlir::MemRefType>(indices.getType());
			auto updatesType = llvm::dyn_cast<mlir::MemRefType>(updates.getType());
			auto outType = llvm::dyn_cast<mlir::MemRefType>(out.getType());
			if (!dataType || !indicesType || !updatesType || !outType || dataType.getRank() != 3 ||
			    updatesType.getRank() != 3 || indicesType.getRank() != 1 || outType.getRank() != 3 ||
			    !dataType.getElementType().isF32() || !updatesType.getElementType().isF32() ||
			    !outType.getElementType().isF32() || !indicesType.getElementType().isInteger(64))
			{
				return mlir::failure();
			}

			const auto loc = op.getLoc();
			auto dynamicF32Rank3 = mlir::MemRefType::get(
			    { mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic, mlir::ShapedType::kDynamic },
			    dataType.getElementType());
			auto dynamicI64Rank1 = mlir::MemRefType::get({ mlir::ShapedType::kDynamic }, indicesType.getElementType());
			auto funcType = builder.getFunctionType(
			    mlir::TypeRange{ dynamicF32Rank3, dynamicI64Rank1, dynamicF32Rank3, dynamicF32Rank3 },
			    mlir::TypeRange{});
			auto helper = module.lookupSymbol<mlir::func::FuncOp>(kScatterUpdateAxis0F32Rank3Helper);
			if (!helper)
			{
				mlir::OpBuilder::InsertionGuard guard(builder);
				builder.setInsertionPointToStart(module.getBody());
				helper = builder.create<mlir::func::FuncOp>(loc, kScatterUpdateAxis0F32Rank3Helper, funcType);
				helper.setPrivate();
			}
			else if (helper.getFunctionType() != funcType)
			{
				return mlir::failure();
			}

			auto dynamicData = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank3, data).getResult();
			auto dynamicIndices = builder.create<mlir::memref::CastOp>(loc, dynamicI64Rank1, indices).getResult();
			auto dynamicUpdates = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank3, updates).getResult();
			auto dynamicOut = builder.create<mlir::memref::CastOp>(loc, dynamicF32Rank3, out).getResult();
			builder.create<mlir::func::CallOp>(
			    loc, helper, mlir::ValueRange{ dynamicData, dynamicIndices, dynamicUpdates, dynamicOut });
			op.erase();
			return mlir::success();
		}

		bool isStaticPositiveDim(int64_t dim)
		{
			return dim > 0 && dim != mlir::ShapedType::kDynamic;
		}

		bool isConstantF32Global(mlir::Value value, mlir::MemRefType type)
		{
			auto getGlobal = value.getDefiningOp<mlir::memref::GetGlobalOp>();
			if (!getGlobal || type.isDynamicDim(0) || type.isDynamicDim(1))
			{
				return false;
			}

			auto module = getGlobal->getParentOfType<mlir::ModuleOp>();
			auto global = module.lookupSymbol<mlir::memref::GlobalOp>(getGlobal.getName());
			if (!global || !global.getConstant())
			{
				return false;
			}

			auto initialValue = global.getInitialValue();
			if (!initialValue)
			{
				return false;
			}
			auto dense = llvm::dyn_cast<mlir::DenseFPElementsAttr>(*initialValue);
			return dense && dense.getElementType().isF32() &&
			       dense.getNumElements() == type.getDimSize(0) * type.getDimSize(1);
		}

		std::uint64_t saturatedMul(std::uint64_t lhs, std::uint64_t rhs)
		{
			if (lhs == 0 || rhs == 0)
			{
				return 0;
			}
			if (lhs > std::numeric_limits<std::uint64_t>::max() / rhs)
			{
				return std::numeric_limits<std::uint64_t>::max();
			}
			return lhs * rhs;
		}

		std::uint64_t matMulFLOPs(int64_t m, int64_t k, int64_t n)
		{
			auto result = saturatedMul(static_cast<std::uint64_t>(m), static_cast<std::uint64_t>(k));
			result = saturatedMul(result, static_cast<std::uint64_t>(n));
			return saturatedMul(result, 2);
		}

		bool shouldUsePackedRhs(mlir::Value rhs, mlir::MemRefType rhsType, mlir::MemRefType outType, int64_t nStep,
		                        int64_t rowTile)
		{
			const int64_t m = outType.getDimSize(0);
			const int64_t k = rhsType.getDimSize(0);
			const int64_t n = rhsType.getDimSize(1);
			if (!isStaticPositiveDim(m) || !isStaticPositiveDim(k) || !isStaticPositiveDim(n) || m % rowTile != 0 ||
			    n % nStep != 0 || !isConstantF32Global(rhs, rhsType))
			{
				return false;
			}

			const auto flops = matMulFLOPs(m, k, n);
			const auto rhsBytes = static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(n) * sizeof(float);
			const bool enoughReuse = static_cast<std::uint64_t>(m) >= rowTile;
			const bool amortizesPackedGlobal = flops >= rhsBytes * 16;
			const bool wideEnoughForPanel = n >= 128 && k >= 32;
			return enoughReuse && wideEnoughForPanel && amortizesPackedGlobal;
		}

		bool shouldUseKPanelPackedRhs(mlir::Value rhs, mlir::MemRefType rhsType, mlir::MemRefType outType,
		                              int64_t nStep, int64_t rowTile, int64_t kPanel)
		{
			const int64_t m = outType.getDimSize(0);
			const int64_t k = rhsType.getDimSize(0);
			const int64_t n = rhsType.getDimSize(1);
			return shouldUsePackedRhs(rhs, rhsType, outType, nStep, rowTile) && k >= 128 && k % kPanel == 0 &&
			       n >= 128 && m >= rowTile && matMulFLOPs(m, k, n) >= (1ull << 20);
		}

		mlir::Value getOrCreatePackedRhs(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value rhs,
		                                 mlir::MemRefType rhsType, int64_t nStep, mlir::MemRefType& packedType)
		{
			auto getGlobal = rhs.getDefiningOp<mlir::memref::GetGlobalOp>();
			if (!getGlobal || rhsType.isDynamicDim(0) || rhsType.isDynamicDim(1))
			{
				return {};
			}

			auto module = getGlobal->getParentOfType<mlir::ModuleOp>();
			auto global = module.lookupSymbol<mlir::memref::GlobalOp>(getGlobal.getName());
			if (!global || !global.getConstant())
			{
				return {};
			}

			auto initialValue = global.getInitialValue();
			if (!initialValue)
			{
				return {};
			}
			auto dense = llvm::dyn_cast<mlir::DenseFPElementsAttr>(*initialValue);
			if (!dense || !dense.getElementType().isF32())
			{
				return {};
			}

			const int64_t k = rhsType.getDimSize(0);
			const int64_t n = rhsType.getDimSize(1);
			if (k <= 0 || n <= 0 || n % nStep != 0 || dense.getNumElements() != k * n)
			{
				return {};
			}

			const int64_t nTiles = n / nStep;
			packedType = mlir::MemRefType::get({ nTiles, k, nStep }, rhsType.getElementType());

			const std::string packedName =
			    (getGlobal.getName() + llvm::Twine("__litenn_packed_n") + llvm::Twine(nStep)).str();
			if (!module.lookupSymbol<mlir::memref::GlobalOp>(packedName))
			{
				llvm::SmallVector<float> source;
				source.reserve(static_cast<size_t>(dense.getNumElements()));
				for (float value : dense.getValues<float>())
				{
					source.push_back(value);
				}

				llvm::SmallVector<float> packed;
				packed.resize(source.size());
				for (int64_t nTile = 0; nTile < nTiles; ++nTile)
				{
					for (int64_t kk = 0; kk < k; ++kk)
					{
						for (int64_t col = 0; col < nStep; ++col)
						{
							const int64_t srcCol = nTile * nStep + col;
							packed[static_cast<size_t>((nTile * k + kk) * nStep + col)] =
							    source[static_cast<size_t>(kk * n + srcCol)];
						}
					}
				}

				auto packedTensorType = mlir::RankedTensorType::get({ nTiles, k, nStep }, rhsType.getElementType());
				auto packedAttr = mlir::DenseElementsAttr::get(packedTensorType, llvm::ArrayRef(packed));

				mlir::OpBuilder globalBuilder(builder.getContext());
				globalBuilder.setInsertionPointAfter(global);
				globalBuilder.create<mlir::memref::GlobalOp>(loc, packedName,
				                                             /*sym_visibility=*/builder.getStringAttr("private"),
				                                             packedType, packedAttr,
				                                             /*constant=*/true,
				                                             /*alignment=*/builder.getI64IntegerAttr(64));
			}

			return builder.create<mlir::memref::GetGlobalOp>(loc, packedType, packedName).getResult();
		}

		mlir::Value getOrCreateKPanelPackedRhs(mlir::OpBuilder& builder, mlir::Location loc, mlir::Value rhs,
		                                       mlir::MemRefType rhsType, int64_t nStep, int64_t kPanel,
		                                       mlir::MemRefType& packedType)
		{
			auto getGlobal = rhs.getDefiningOp<mlir::memref::GetGlobalOp>();
			if (!getGlobal || rhsType.isDynamicDim(0) || rhsType.isDynamicDim(1))
			{
				return {};
			}

			auto module = getGlobal->getParentOfType<mlir::ModuleOp>();
			auto global = module.lookupSymbol<mlir::memref::GlobalOp>(getGlobal.getName());
			if (!global || !global.getConstant())
			{
				return {};
			}

			auto initialValue = global.getInitialValue();
			if (!initialValue)
			{
				return {};
			}
			auto dense = llvm::dyn_cast<mlir::DenseFPElementsAttr>(*initialValue);
			if (!dense || !dense.getElementType().isF32())
			{
				return {};
			}

			const int64_t k = rhsType.getDimSize(0);
			const int64_t n = rhsType.getDimSize(1);
			if (k <= 0 || n <= 0 || k % kPanel != 0 || n % nStep != 0 || dense.getNumElements() != k * n)
			{
				return {};
			}

			const int64_t nTiles = n / nStep;
			const int64_t kPanels = k / kPanel;
			packedType = mlir::MemRefType::get({ nTiles, kPanels, kPanel, nStep }, rhsType.getElementType());

			const std::string packedName = (getGlobal.getName() + llvm::Twine("__litenn_packed_n") +
			                                llvm::Twine(nStep) + llvm::Twine("_kp") + llvm::Twine(kPanel))
			                                   .str();
			if (!module.lookupSymbol<mlir::memref::GlobalOp>(packedName))
			{
				llvm::SmallVector<float> source;
				source.reserve(static_cast<size_t>(dense.getNumElements()));
				for (float value : dense.getValues<float>())
				{
					source.push_back(value);
				}

				llvm::SmallVector<float> packed;
				packed.resize(source.size());
				for (int64_t nTile = 0; nTile < nTiles; ++nTile)
				{
					for (int64_t kPanelIndex = 0; kPanelIndex < kPanels; ++kPanelIndex)
					{
						for (int64_t kInner = 0; kInner < kPanel; ++kInner)
						{
							const int64_t kk = kPanelIndex * kPanel + kInner;
							for (int64_t col = 0; col < nStep; ++col)
							{
								const int64_t srcCol = nTile * nStep + col;
								packed[static_cast<size_t>(((nTile * kPanels + kPanelIndex) * kPanel + kInner) * nStep +
								                           col)] = source[static_cast<size_t>(kk * n + srcCol)];
							}
						}
					}
				}

				auto packedTensorType =
				    mlir::RankedTensorType::get({ nTiles, kPanels, kPanel, nStep }, rhsType.getElementType());
				auto packedAttr = mlir::DenseElementsAttr::get(packedTensorType, llvm::ArrayRef(packed));

				mlir::OpBuilder globalBuilder(builder.getContext());
				globalBuilder.setInsertionPointAfter(global);
				globalBuilder.create<mlir::memref::GlobalOp>(loc, packedName,
				                                             /*sym_visibility=*/builder.getStringAttr("private"),
				                                             packedType, packedAttr,
				                                             /*constant=*/true,
				                                             /*alignment=*/builder.getI64IntegerAttr(64));
			}

			return builder.create<mlir::memref::GetGlobalOp>(loc, packedType, packedName).getResult();
		}

		mlir::LogicalResult rewriteKPanelPackedWideMatMulRowTile(mlir::linalg::GenericOp op, mlir::OpBuilder& builder,
		                                                         int64_t rowTile, int64_t maxTileVectors,
		                                                         int64_t kPanel)
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

			const int64_t vectorWidth = nativeF32VectorWidth();
			const int64_t n = outType.getDimSize(1);
			if (vectorWidth < 4 || n <= vectorWidth || n % vectorWidth != 0 || outType.isDynamicDim(0) ||
			    outType.getDimSize(0) % rowTile != 0)
			{
				return mlir::failure();
			}

			int64_t tileVectors = 1;
			for (int64_t candidate = maxTileVectors; candidate > 1; candidate /= 2)
			{
				if (n % (vectorWidth * candidate) == 0)
				{
					tileVectors = candidate;
					break;
				}
			}
			const int64_t nStep = vectorWidth * tileVectors;
			if (!shouldUseKPanelPackedRhs(rhs, rhsType, outType, nStep, rowTile, kPanel))
			{
				return mlir::failure();
			}

			const auto loc = op.getLoc();
			mlir::MemRefType packedRhsType;
			auto packedRhs = getOrCreateKPanelPackedRhs(builder, loc, rhs, rhsType, nStep, kPanel, packedRhsType);
			if (!packedRhs)
			{
				return mlir::failure();
			}

			auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
			auto c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
			auto cMstep = builder.create<mlir::arith::ConstantIndexOp>(loc, rowTile);
			auto cNStep = builder.create<mlir::arith::ConstantIndexOp>(loc, nStep);
			auto cKPanel = builder.create<mlir::arith::ConstantIndexOp>(loc, kPanel);
			auto nTileUpper = builder.create<mlir::arith::ConstantIndexOp>(loc, n / nStep);
			auto kPanelUpper = builder.create<mlir::arith::ConstantIndexOp>(loc, rhsType.getDimSize(0) / kPanel);
			auto mUpper = builder.create<mlir::memref::DimOp>(loc, out, 0);
			auto vecType = mlir::VectorType::get({ vectorWidth }, outType.getElementType());
			const bool applyRelu = op->hasAttr(kApplyReluAttr);

			auto mLoop = builder.create<mlir::scf::ForOp>(loc, c0, mUpper, cMstep);
			{
				mlir::OpBuilder::InsertionGuard mGuard(builder);
				builder.setInsertionPointToStart(mLoop.getBody());
				auto mBase = mLoop.getInductionVar();
				llvm::SmallVector<mlir::Value, 8> mIndices;
				mIndices.reserve(static_cast<size_t>(rowTile));
				mIndices.push_back(mBase);
				for (int64_t row = 1; row < rowTile; ++row)
				{
					auto offset = builder.create<mlir::arith::ConstantIndexOp>(loc, row);
					mIndices.push_back(builder.create<mlir::arith::AddIOp>(loc, mBase, offset).getResult());
				}

				auto nLoop = builder.create<mlir::scf::ForOp>(loc, c0, nTileUpper, c1);
				{
					mlir::OpBuilder::InsertionGuard nGuard(builder);
					builder.setInsertionPointToStart(nLoop.getBody());
					auto nTile = nLoop.getInductionVar();
					auto nBase = builder.create<mlir::arith::MulIOp>(loc, nTile, cNStep).getResult();

					llvm::SmallVector<mlir::Value, 8> nIndices;
					llvm::SmallVector<mlir::Value, 8> packedNOffsets;
					llvm::SmallVector<mlir::Value, 16> initAccs;
					nIndices.reserve(static_cast<size_t>(tileVectors));
					packedNOffsets.reserve(static_cast<size_t>(tileVectors));
					initAccs.reserve(static_cast<size_t>(rowTile * tileVectors));
					for (int64_t lane = 0; lane < tileVectors; ++lane)
					{
						mlir::Value nOffset = builder.create<mlir::arith::ConstantIndexOp>(loc, lane * vectorWidth);
						packedNOffsets.push_back(nOffset);

						mlir::Value nIndex = nBase;
						if (lane != 0)
						{
							nIndex = builder.create<mlir::arith::AddIOp>(loc, nBase, nOffset).getResult();
						}
						nIndices.push_back(nIndex);
					}

					for (int64_t row = 0; row < rowTile; ++row)
					{
						for (int64_t lane = 0; lane < tileVectors; ++lane)
						{
							initAccs.push_back(builder
							                       .create<mlir::vector::LoadOp>(
							                           loc, vecType, out,
							                           mlir::ValueRange{ mIndices[static_cast<size_t>(row)],
							                                             nIndices[static_cast<size_t>(lane)] })
							                       .getResult());
						}
					}

					auto kPanelLoop = builder.create<mlir::scf::ForOp>(
					    loc, c0, kPanelUpper, c1, initAccs,
					    [&](mlir::OpBuilder& nested, mlir::Location nestedLoc, mlir::Value kPanelIndex,
					        mlir::ValueRange accs) {
						    llvm::SmallVector<mlir::Value, 16> currentAccs(accs.begin(), accs.end());
						    auto kBase =
						        nested.create<mlir::arith::MulIOp>(nestedLoc, kPanelIndex, cKPanel).getResult();
						    for (int64_t kInner = 0; kInner < kPanel; ++kInner)
						    {
							    auto kInnerOffset = nested.create<mlir::arith::ConstantIndexOp>(nestedLoc, kInner);
							    auto k = kBase;
							    if (kInner != 0)
							    {
								    k = nested.create<mlir::arith::AddIOp>(nestedLoc, kBase, kInnerOffset).getResult();
							    }

							    llvm::SmallVector<mlir::Value, 8> bVecs;
							    bVecs.reserve(static_cast<size_t>(tileVectors));
							    for (int64_t lane = 0; lane < tileVectors; ++lane)
							    {
								    bVecs.push_back(
								        nested
								            .create<mlir::vector::LoadOp>(
								                nestedLoc, vecType, packedRhs,
								                mlir::ValueRange{ nTile, kPanelIndex, kInnerOffset,
								                                  packedNOffsets[static_cast<size_t>(lane)] })
								            .getResult());
							    }

							    llvm::SmallVector<mlir::Value, 16> nextAccs;
							    nextAccs.reserve(currentAccs.size());
							    for (int64_t row = 0; row < rowTile; ++row)
							    {
								    auto a = nested
								                 .create<mlir::memref::LoadOp>(
								                     nestedLoc, lhs,
								                     mlir::ValueRange{ mIndices[static_cast<size_t>(row)], k })
								                 .getResult();
								    auto aVec =
								        nested.create<mlir::vector::BroadcastOp>(nestedLoc, vecType, a).getResult();
								    for (int64_t lane = 0; lane < tileVectors; ++lane)
								    {
									    const size_t accIndex = static_cast<size_t>(row * tileVectors + lane);
									    auto next = nested
									                    .create<mlir::vector::FMAOp>(nestedLoc, aVec,
									                                                 bVecs[static_cast<size_t>(lane)],
									                                                 currentAccs[accIndex])
									                    .getResult();
									    nextAccs.push_back(next);
								    }
							    }
							    currentAccs = std::move(nextAccs);
						    }
						    nested.create<mlir::scf::YieldOp>(nestedLoc, currentAccs);
					    });

					for (int64_t row = 0; row < rowTile; ++row)
					{
						for (int64_t lane = 0; lane < tileVectors; ++lane)
						{
							const unsigned accIndex = static_cast<unsigned>(row * tileVectors + lane);
							auto value = applyReluIfNeeded(builder, loc, kPanelLoop.getResult(accIndex), applyRelu);
							builder.create<mlir::vector::StoreOp>(
							    loc, value, out,
							    mlir::ValueRange{ mIndices[static_cast<size_t>(row)],
							                      nIndices[static_cast<size_t>(lane)] });
						}
					}
				}
			}

			op.erase();
			return mlir::success();
		}

		mlir::LogicalResult rewritePackedWideMatMulRowTile(mlir::linalg::GenericOp op, mlir::OpBuilder& builder,
		                                                   int64_t rowTile, int64_t maxTileVectors)
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

			const int64_t vectorWidth = nativeF32VectorWidth();
			const int64_t n = outType.getDimSize(1);
			if (vectorWidth < 4 || n < 256 || n % vectorWidth != 0 || outType.isDynamicDim(0) ||
			    outType.getDimSize(0) % rowTile != 0)
			{
				return mlir::failure();
			}

			int64_t tileVectors = 1;
			for (int64_t candidate = maxTileVectors; candidate > 1; candidate /= 2)
			{
				if (n % (vectorWidth * candidate) == 0)
				{
					tileVectors = candidate;
					break;
				}
			}
			const int64_t nStep = vectorWidth * tileVectors;
			if (!shouldUsePackedRhs(rhs, rhsType, outType, nStep, rowTile))
			{
				return mlir::failure();
			}

			const auto loc = op.getLoc();
			mlir::MemRefType packedRhsType;
			auto packedRhs = getOrCreatePackedRhs(builder, loc, rhs, rhsType, nStep, packedRhsType);
			if (!packedRhs)
			{
				return mlir::failure();
			}

			auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
			auto c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
			auto cMstep = builder.create<mlir::arith::ConstantIndexOp>(loc, rowTile);
			auto cNStep = builder.create<mlir::arith::ConstantIndexOp>(loc, nStep);
			auto nTileUpper = builder.create<mlir::arith::ConstantIndexOp>(loc, n / nStep);
			auto mUpper = builder.create<mlir::memref::DimOp>(loc, out, 0);
			auto kUpper = builder.create<mlir::memref::DimOp>(loc, lhs, 1);
			auto vecType = mlir::VectorType::get({ vectorWidth }, outType.getElementType());
			const bool applyRelu = op->hasAttr(kApplyReluAttr);

			auto mLoop = builder.create<mlir::scf::ForOp>(loc, c0, mUpper, cMstep);
			{
				mlir::OpBuilder::InsertionGuard mGuard(builder);
				builder.setInsertionPointToStart(mLoop.getBody());
				auto mBase = mLoop.getInductionVar();
				llvm::SmallVector<mlir::Value, 8> mIndices;
				mIndices.reserve(static_cast<size_t>(rowTile));
				mIndices.push_back(mBase);
				for (int64_t row = 1; row < rowTile; ++row)
				{
					auto offset = builder.create<mlir::arith::ConstantIndexOp>(loc, row);
					mIndices.push_back(builder.create<mlir::arith::AddIOp>(loc, mBase, offset).getResult());
				}

				auto nLoop = builder.create<mlir::scf::ForOp>(loc, c0, nTileUpper, c1);
				{
					mlir::OpBuilder::InsertionGuard nGuard(builder);
					builder.setInsertionPointToStart(nLoop.getBody());
					auto nTile = nLoop.getInductionVar();
					auto nBase = builder.create<mlir::arith::MulIOp>(loc, nTile, cNStep).getResult();

					llvm::SmallVector<mlir::Value, 8> nIndices;
					llvm::SmallVector<mlir::Value, 8> packedNOffsets;
					llvm::SmallVector<mlir::Value, 16> initAccs;
					nIndices.reserve(static_cast<size_t>(tileVectors));
					packedNOffsets.reserve(static_cast<size_t>(tileVectors));
					initAccs.reserve(static_cast<size_t>(rowTile * tileVectors));
					for (int64_t lane = 0; lane < tileVectors; ++lane)
					{
						mlir::Value nOffset = builder.create<mlir::arith::ConstantIndexOp>(loc, lane * vectorWidth);
						packedNOffsets.push_back(nOffset);

						mlir::Value nIndex = nBase;
						if (lane != 0)
						{
							nIndex = builder.create<mlir::arith::AddIOp>(loc, nBase, nOffset).getResult();
						}
						nIndices.push_back(nIndex);
					}

					for (int64_t row = 0; row < rowTile; ++row)
					{
						for (int64_t lane = 0; lane < tileVectors; ++lane)
						{
							initAccs.push_back(builder
							                       .create<mlir::vector::LoadOp>(
							                           loc, vecType, out,
							                           mlir::ValueRange{ mIndices[static_cast<size_t>(row)],
							                                             nIndices[static_cast<size_t>(lane)] })
							                       .getResult());
						}
					}

					auto kLoop = builder.create<mlir::scf::ForOp>(
					    loc, c0, kUpper, c1, initAccs,
					    [&](mlir::OpBuilder& nested, mlir::Location nestedLoc, mlir::Value k, mlir::ValueRange accs) {
						    llvm::SmallVector<mlir::Value, 8> bVecs;
						    bVecs.reserve(static_cast<size_t>(tileVectors));
						    for (int64_t lane = 0; lane < tileVectors; ++lane)
						    {
							    bVecs.push_back(
							        nested
							            .create<mlir::vector::LoadOp>(
							                nestedLoc, vecType, packedRhs,
							                mlir::ValueRange{ nTile, k, packedNOffsets[static_cast<size_t>(lane)] })
							            .getResult());
						    }

						    llvm::SmallVector<mlir::Value, 16> nextAccs;
						    nextAccs.reserve(accs.size());
						    for (int64_t row = 0; row < rowTile; ++row)
						    {
							    auto a =
							        nested
							            .create<mlir::memref::LoadOp>(
							                nestedLoc, lhs, mlir::ValueRange{ mIndices[static_cast<size_t>(row)], k })
							            .getResult();
							    auto aVec = nested.create<mlir::vector::BroadcastOp>(nestedLoc, vecType, a).getResult();
							    for (int64_t lane = 0; lane < tileVectors; ++lane)
							    {
								    const size_t accIndex = static_cast<size_t>(row * tileVectors + lane);
								    auto next = nested
								                    .create<mlir::vector::FMAOp>(nestedLoc, aVec,
								                                                 bVecs[static_cast<size_t>(lane)],
								                                                 accs[accIndex])
								                    .getResult();
								    nextAccs.push_back(next);
							    }
						    }
						    nested.create<mlir::scf::YieldOp>(nestedLoc, nextAccs);
					    });

					for (int64_t row = 0; row < rowTile; ++row)
					{
						for (int64_t lane = 0; lane < tileVectors; ++lane)
						{
							const unsigned accIndex = static_cast<unsigned>(row * tileVectors + lane);
							auto value = applyReluIfNeeded(builder, loc, kLoop.getResult(accIndex), applyRelu);
							builder.create<mlir::vector::StoreOp>(
							    loc, value, out,
							    mlir::ValueRange{ mIndices[static_cast<size_t>(row)],
							                      nIndices[static_cast<size_t>(lane)] });
						}
					}
				}
			}

			op.erase();
			return mlir::success();
		}

		mlir::LogicalResult rewriteWideMatMulRowTile(mlir::linalg::GenericOp op, mlir::OpBuilder& builder,
		                                             int64_t rowTile, int64_t maxTileVectors)
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

			const int64_t vectorWidth = nativeF32VectorWidth();
			const int64_t n = outType.getDimSize(1);
			if (vectorWidth < 4 || n <= vectorWidth || n % vectorWidth != 0 || outType.isDynamicDim(0) ||
			    outType.getDimSize(0) % rowTile != 0)
			{
				return mlir::failure();
			}

			int64_t tileVectors = 1;
			for (int64_t candidate = maxTileVectors; candidate > 1; candidate /= 2)
			{
				if (n % (vectorWidth * candidate) == 0)
				{
					tileVectors = candidate;
					break;
				}
			}
			const int64_t nStep = vectorWidth * tileVectors;

			const auto loc = op.getLoc();
			auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
			auto c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
			auto cMstep = builder.create<mlir::arith::ConstantIndexOp>(loc, rowTile);
			auto cNStep = builder.create<mlir::arith::ConstantIndexOp>(loc, nStep);
			auto nUpper = builder.create<mlir::arith::ConstantIndexOp>(loc, n);
			auto mUpper = builder.create<mlir::memref::DimOp>(loc, out, 0);
			auto kUpper = builder.create<mlir::memref::DimOp>(loc, lhs, 1);
			auto vecType = mlir::VectorType::get({ vectorWidth }, outType.getElementType());
			const bool applyRelu = op->hasAttr(kApplyReluAttr);

			auto mLoop = builder.create<mlir::scf::ForOp>(loc, c0, mUpper, cMstep);
			{
				mlir::OpBuilder::InsertionGuard mGuard(builder);
				builder.setInsertionPointToStart(mLoop.getBody());
				auto mBase = mLoop.getInductionVar();
				llvm::SmallVector<mlir::Value, 8> mIndices;
				mIndices.reserve(static_cast<size_t>(rowTile));
				mIndices.push_back(mBase);
				for (int64_t row = 1; row < rowTile; ++row)
				{
					auto offset = builder.create<mlir::arith::ConstantIndexOp>(loc, row);
					mIndices.push_back(builder.create<mlir::arith::AddIOp>(loc, mBase, offset).getResult());
				}

				auto nLoop = builder.create<mlir::scf::ForOp>(loc, c0, nUpper, cNStep);
				{
					mlir::OpBuilder::InsertionGuard nGuard(builder);
					builder.setInsertionPointToStart(nLoop.getBody());
					auto nBase = nLoop.getInductionVar();

					llvm::SmallVector<mlir::Value, 8> nIndices;
					llvm::SmallVector<mlir::Value, 16> initAccs;
					nIndices.reserve(static_cast<size_t>(tileVectors));
					initAccs.reserve(static_cast<size_t>(rowTile * tileVectors));
					for (int64_t lane = 0; lane < tileVectors; ++lane)
					{
						mlir::Value nIndex = nBase;
						if (lane != 0)
						{
							auto offset = builder.create<mlir::arith::ConstantIndexOp>(loc, lane * vectorWidth);
							nIndex = builder.create<mlir::arith::AddIOp>(loc, nBase, offset).getResult();
						}
						nIndices.push_back(nIndex);
					}

					for (int64_t row = 0; row < rowTile; ++row)
					{
						for (int64_t lane = 0; lane < tileVectors; ++lane)
						{
							initAccs.push_back(builder
							                       .create<mlir::vector::LoadOp>(
							                           loc, vecType, out,
							                           mlir::ValueRange{ mIndices[static_cast<size_t>(row)],
							                                             nIndices[static_cast<size_t>(lane)] })
							                       .getResult());
						}
					}

					auto kLoop = builder.create<mlir::scf::ForOp>(
					    loc, c0, kUpper, c1, initAccs,
					    [&](mlir::OpBuilder& nested, mlir::Location nestedLoc, mlir::Value k, mlir::ValueRange accs) {
						    llvm::SmallVector<mlir::Value, 8> bVecs;
						    bVecs.reserve(static_cast<size_t>(tileVectors));
						    for (int64_t lane = 0; lane < tileVectors; ++lane)
						    {
							    bVecs.push_back(nested
							                        .create<mlir::vector::LoadOp>(
							                            nestedLoc, vecType, rhs,
							                            mlir::ValueRange{ k, nIndices[static_cast<size_t>(lane)] })
							                        .getResult());
						    }

						    llvm::SmallVector<mlir::Value, 16> nextAccs;
						    nextAccs.reserve(accs.size());
						    for (int64_t row = 0; row < rowTile; ++row)
						    {
							    auto a =
							        nested
							            .create<mlir::memref::LoadOp>(
							                nestedLoc, lhs, mlir::ValueRange{ mIndices[static_cast<size_t>(row)], k })
							            .getResult();
							    auto aVec = nested.create<mlir::vector::BroadcastOp>(nestedLoc, vecType, a).getResult();
							    for (int64_t lane = 0; lane < tileVectors; ++lane)
							    {
								    const size_t accIndex = static_cast<size_t>(row * tileVectors + lane);
								    auto next = nested
								                    .create<mlir::vector::FMAOp>(nestedLoc, aVec,
								                                                 bVecs[static_cast<size_t>(lane)],
								                                                 accs[accIndex])
								                    .getResult();
								    nextAccs.push_back(next);
							    }
						    }
						    nested.create<mlir::scf::YieldOp>(nestedLoc, nextAccs);
					    });

					for (int64_t row = 0; row < rowTile; ++row)
					{
						for (int64_t lane = 0; lane < tileVectors; ++lane)
						{
							const unsigned accIndex = static_cast<unsigned>(row * tileVectors + lane);
							auto value = applyReluIfNeeded(builder, loc, kLoop.getResult(accIndex), applyRelu);
							builder.create<mlir::vector::StoreOp>(
							    loc, value, out,
							    mlir::ValueRange{ mIndices[static_cast<size_t>(row)],
							                      nIndices[static_cast<size_t>(lane)] });
						}
					}
				}
			}

			op.erase();
			return mlir::success();
		}

		mlir::LogicalResult rewriteWideMatMul(mlir::linalg::GenericOp op, mlir::OpBuilder& builder)
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

			const int64_t vectorWidth = nativeF32VectorWidth();
			const int64_t n = outType.getDimSize(1);
			if (vectorWidth < 4 || n <= vectorWidth || n % vectorWidth != 0)
			{
				return mlir::failure();
			}

			const int64_t tileVectors = (n % (vectorWidth * 8) == 0)   ? 8
			                            : (n % (vectorWidth * 4) == 0) ? 4
			                            : (n % (vectorWidth * 2) == 0) ? 2
			                                                           : 1;
			const int64_t nStep = vectorWidth * tileVectors;

			const auto loc = op.getLoc();
			auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
			auto c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
			auto cNStep = builder.create<mlir::arith::ConstantIndexOp>(loc, nStep);
			auto nUpper = builder.create<mlir::arith::ConstantIndexOp>(loc, n);
			auto mUpper = builder.create<mlir::memref::DimOp>(loc, out, 0);
			auto kUpper = builder.create<mlir::memref::DimOp>(loc, lhs, 1);
			auto vecType = mlir::VectorType::get({ vectorWidth }, outType.getElementType());
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
							auto offset = builder.create<mlir::arith::ConstantIndexOp>(loc, lane * vectorWidth);
							nIndex = builder.create<mlir::arith::AddIOp>(loc, nBase, offset).getResult();
						}
						nIndices.push_back(nIndex);
						initAccs.push_back(
						    builder.create<mlir::vector::LoadOp>(loc, vecType, out, mlir::ValueRange{ m, nIndex })
						        .getResult());
					}

					auto kLoop = builder.create<mlir::scf::ForOp>(
					    loc, c0, kUpper, c1, initAccs,
					    [&](mlir::OpBuilder& nested, mlir::Location nestedLoc, mlir::Value k, mlir::ValueRange accs) {
						    auto a = nested.create<mlir::memref::LoadOp>(nestedLoc, lhs, mlir::ValueRange{ m, k })
						                 .getResult();
						    auto aVec = nested.create<mlir::vector::BroadcastOp>(nestedLoc, vecType, a).getResult();

						    llvm::SmallVector<mlir::Value, 8> nextAccs;
						    nextAccs.reserve(accs.size());
						    for (int64_t lane = 0; lane < tileVectors; ++lane)
						    {
							    auto bVec = nested
							                    .create<mlir::vector::LoadOp>(
							                        nestedLoc, vecType, rhs,
							                        mlir::ValueRange{ k, nIndices[static_cast<size_t>(lane)] })
							                    .getResult();
							    auto next = nested
							                    .create<mlir::vector::FMAOp>(nestedLoc, aVec, bVec,
							                                                 accs[static_cast<size_t>(lane)])
							                    .getResult();
							    nextAccs.push_back(next);
						    }
						    nested.create<mlir::scf::YieldOp>(nestedLoc, nextAccs);
					    });

					for (int64_t lane = 0; lane < tileVectors; ++lane)
					{
						auto value =
						    applyReluIfNeeded(builder, loc, kLoop.getResult(static_cast<unsigned>(lane)), applyRelu);
						builder.create<mlir::vector::StoreOp>(
						    loc, value, out, mlir::ValueRange{ m, nIndices[static_cast<size_t>(lane)] });
					}
				}
			}

			op.erase();
			return mlir::success();
		}

		mlir::LogicalResult rewriteNarrowVectorMatMul(mlir::linalg::GenericOp op, mlir::OpBuilder& builder)
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

				auto initAcc =
				    builder.create<mlir::vector::LoadOp>(loc, vecType, out, mlir::ValueRange{ m, c0 }).getResult();
				auto kLoop = builder.create<mlir::scf::ForOp>(
				    loc, c0, kUpper, c1, initAcc,
				    [&](mlir::OpBuilder& nested, mlir::Location nestedLoc, mlir::Value k, mlir::ValueRange accs) {
					    auto a =
					        nested.create<mlir::memref::LoadOp>(nestedLoc, lhs, mlir::ValueRange{ m, k }).getResult();
					    auto aVec = nested.create<mlir::vector::BroadcastOp>(nestedLoc, vecType, a).getResult();
					    auto bVec =
					        nested.create<mlir::vector::LoadOp>(nestedLoc, vecType, rhs, mlir::ValueRange{ k, c0 })
					            .getResult();
					    auto next = nested.create<mlir::vector::FMAOp>(nestedLoc, aVec, bVec, accs.front()).getResult();
					    nested.create<mlir::scf::YieldOp>(nestedLoc, next);
				    });

				auto value = applyReluIfNeeded(builder, loc, kLoop.getResult(0), applyRelu);
				builder.create<mlir::vector::StoreOp>(loc, value, out, mlir::ValueRange{ m, c0 });
			}

			op.erase();
			return mlir::success();
		}

		mlir::LogicalResult rewriteNarrowVectorMatMulRowTile(mlir::linalg::GenericOp op, mlir::OpBuilder& builder,
		                                                     int64_t rowTile)
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
			if (n < 4 || n > 16 || outType.isDynamicDim(0) || outType.getDimSize(0) % rowTile != 0)
			{
				return mlir::failure();
			}

			const auto loc = op.getLoc();
			auto c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
			auto c1 = builder.create<mlir::arith::ConstantIndexOp>(loc, 1);
			auto cMstep = builder.create<mlir::arith::ConstantIndexOp>(loc, rowTile);
			auto mUpper = builder.create<mlir::memref::DimOp>(loc, out, 0);
			auto kUpper = builder.create<mlir::memref::DimOp>(loc, lhs, 1);
			auto vecType = mlir::VectorType::get({ n }, outType.getElementType());
			const bool applyRelu = op->hasAttr(kApplyReluAttr);

			auto mLoop = builder.create<mlir::scf::ForOp>(loc, c0, mUpper, cMstep);
			{
				mlir::OpBuilder::InsertionGuard guard(builder);
				builder.setInsertionPointToStart(mLoop.getBody());
				auto mBase = mLoop.getInductionVar();

				llvm::SmallVector<mlir::Value, 16> mIndices;
				llvm::SmallVector<mlir::Value, 16> initAccs;
				mIndices.reserve(static_cast<size_t>(rowTile));
				initAccs.reserve(static_cast<size_t>(rowTile));
				mIndices.push_back(mBase);
				for (int64_t row = 1; row < rowTile; ++row)
				{
					auto offset = builder.create<mlir::arith::ConstantIndexOp>(loc, row);
					mIndices.push_back(builder.create<mlir::arith::AddIOp>(loc, mBase, offset).getResult());
				}
				for (int64_t row = 0; row < rowTile; ++row)
				{
					initAccs.push_back(
					    builder
					        .create<mlir::vector::LoadOp>(loc, vecType, out,
					                                      mlir::ValueRange{ mIndices[static_cast<size_t>(row)], c0 })
					        .getResult());
				}

				auto kLoop = builder.create<mlir::scf::ForOp>(
				    loc, c0, kUpper, c1, initAccs,
				    [&](mlir::OpBuilder& nested, mlir::Location nestedLoc, mlir::Value k, mlir::ValueRange accs) {
					    auto bVec =
					        nested.create<mlir::vector::LoadOp>(nestedLoc, vecType, rhs, mlir::ValueRange{ k, c0 })
					            .getResult();

					    llvm::SmallVector<mlir::Value, 16> nextAccs;
					    nextAccs.reserve(accs.size());
					    for (int64_t row = 0; row < rowTile; ++row)
					    {
						    auto a = nested
						                 .create<mlir::memref::LoadOp>(
						                     nestedLoc, lhs, mlir::ValueRange{ mIndices[static_cast<size_t>(row)], k })
						                 .getResult();
						    auto aVec = nested.create<mlir::vector::BroadcastOp>(nestedLoc, vecType, a).getResult();
						    auto next =
						        nested
						            .create<mlir::vector::FMAOp>(nestedLoc, aVec, bVec, accs[static_cast<size_t>(row)])
						            .getResult();
						    nextAccs.push_back(next);
					    }
					    nested.create<mlir::scf::YieldOp>(nestedLoc, nextAccs);
				    });

				for (int64_t row = 0; row < rowTile; ++row)
				{
					auto value =
					    applyReluIfNeeded(builder, loc, kLoop.getResult(static_cast<unsigned>(row)), applyRelu);
					builder.create<mlir::vector::StoreOp>(loc, value, out,
					                                      mlir::ValueRange{ mIndices[static_cast<size_t>(row)], c0 });
				}
			}

			op.erase();
			return mlir::success();
		}

		mlir::LogicalResult rewriteNarrowMatMul(mlir::linalg::GenericOp op, mlir::OpBuilder& builder)
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
					initAccs.push_back(
					    builder.create<mlir::memref::LoadOp>(loc, out, mlir::ValueRange{ m, nIndex }).getResult());
				}

				auto kLoop = builder.create<mlir::scf::ForOp>(
				    loc, c0, kUpper, c1, initAccs,
				    [&](mlir::OpBuilder& nested, mlir::Location nestedLoc, mlir::Value k, mlir::ValueRange accs) {
					    auto a =
					        nested.create<mlir::memref::LoadOp>(nestedLoc, lhs, mlir::ValueRange{ m, k }).getResult();

					    llvm::SmallVector<mlir::Value, 16> nextAccs;
					    nextAccs.reserve(accs.size());
					    for (int64_t col = 0; col < n; ++col)
					    {
						    auto b = nested
						                 .create<mlir::memref::LoadOp>(
						                     nestedLoc, rhs, mlir::ValueRange{ k, nIndices[static_cast<size_t>(col)] })
						                 .getResult();
						    auto productOp = nested.create<mlir::arith::MulFOp>(nestedLoc, a, b);
						    productOp->setAttr(productOp.getFastMathAttrName(), fastMath);
						    auto sumOp = nested.create<mlir::arith::AddFOp>(nestedLoc, accs[static_cast<size_t>(col)],
						                                                    productOp.getResult());
						    sumOp->setAttr(sumOp.getFastMathAttrName(), fastMath);
						    nextAccs.push_back(sumOp.getResult());
					    }
					    nested.create<mlir::scf::YieldOp>(nestedLoc, nextAccs);
				    });

				for (int64_t col = 0; col < n; ++col)
				{
					auto value =
					    applyReluIfNeeded(builder, loc, kLoop.getResult(static_cast<unsigned>(col)), applyRelu);
					builder.create<mlir::memref::StoreOp>(loc, value, out,
					                                      mlir::ValueRange{ m, nIndices[static_cast<size_t>(col)] });
				}
			}

			op.erase();
			return mlir::success();
		}

		struct LowerNarrowMatMulPass : mlir::PassWrapper<LowerNarrowMatMulPass, mlir::OperationPass<mlir::ModuleOp>>
		{
			LowerNarrowMatMulPass() = default;

			explicit LowerNarrowMatMulPass(LLVMCodegenOptions options) : options_(options)
			{
			}

			llvm::StringRef getName() const override
			{
				return "LiteNNLowerMatMulMicroKernelPass";
			}

			void getDependentDialects(mlir::DialectRegistry& registry) const override
			{
				registry.insert<mlir::arith::ArithDialect, mlir::memref::MemRefDialect, mlir::scf::SCFDialect,
				                mlir::vector::VectorDialect>();
			}

			void runOnOperation() override
			{
				llvm::SmallVector<mlir::linalg::GenericOp> candidates;
				getOperation().walk([&](mlir::linalg::GenericOp op) { candidates.push_back(op); });

				mlir::OpBuilder builder(&getContext());
				for (auto op : candidates)
				{
					builder.setInsertionPoint(op);
					if (mlir::succeeded(rewriteRoPEAtPositionsCall(getOperation(), op, builder)))
					{
						continue;
					}
					if (mlir::succeeded(rewriteActivePrefixAttentionCall(getOperation(), op, builder)))
					{
						continue;
					}
					if (mlir::succeeded(rewriteGroupedActivePrefixAttentionCall(getOperation(), op, builder)))
					{
						continue;
					}
					if (mlir::succeeded(rewriteGroupedPagedAttentionCall(getOperation(), op, builder)))
					{
						continue;
					}
					if (mlir::succeeded(rewriteScatterUpdateAxis0F32Rank3Call(getOperation(), op, builder)))
					{
						continue;
					}
					if (mlir::succeeded(rewriteGGMLBlockQuantizedGetRowsCall(getOperation(), op, builder)))
					{
						continue;
					}
					if (mlir::succeeded(
					        rewriteGGMLBlockGroupedQuantizedMatMulCall(getOperation(), op, builder, options_)))
					{
						continue;
					}
					if (mlir::succeeded(rewriteGGMLBlockQuantizedMatMulCall(getOperation(), op, builder, options_)))
					{
						continue;
					}
					if (mlir::succeeded(rewriteKPanelPackedWideMatMulRowTile(op, builder, 8, 2, 8)))
					{
						continue;
					}
					if (mlir::succeeded(rewritePackedWideMatMulRowTile(op, builder, 8, 2)))
					{
						continue;
					}
					if (mlir::succeeded(rewriteWideMatMulRowTile(op, builder, 8, 2)))
					{
						continue;
					}
					if (mlir::succeeded(rewriteWideMatMulRowTile(op, builder, 4, 4)))
					{
						continue;
					}
					if (mlir::succeeded(rewriteWideMatMulRowTile(op, builder, 2, 8)))
					{
						continue;
					}
					if (mlir::succeeded(rewriteWideMatMul(op, builder)))
					{
						continue;
					}
					if (mlir::succeeded(rewriteNarrowVectorMatMulRowTile(op, builder, 16)))
					{
						continue;
					}
					if (mlir::succeeded(rewriteNarrowVectorMatMulRowTile(op, builder, 8)))
					{
						continue;
					}
					if (mlir::succeeded(rewriteNarrowVectorMatMul(op, builder)))
					{
						continue;
					}
					(void) rewriteNarrowMatMul(op, builder);
				}
			}

			LLVMCodegenOptions options_{};
		};

		std::unique_ptr<mlir::Pass> createLowerNarrowMatMulPass(const LLVMCodegenOptions& options)
		{
			return std::make_unique<LowerNarrowMatMulPass>(options);
		}

		struct EnableSIMDFastMathPass : mlir::PassWrapper<EnableSIMDFastMathPass, mlir::OperationPass<mlir::ModuleOp>>
		{
			llvm::StringRef getName() const override
			{
				return "LiteNNEnableSIMDFastMathPass";
			}

			void runOnOperation() override
			{
				auto flags = mlir::arith::FastMathFlags::reassoc | mlir::arith::FastMathFlags::contract |
				             mlir::arith::FastMathFlags::nnan | mlir::arith::FastMathFlags::ninf |
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
		addLLVMCodegenPipeline(pm, LLVMCodegenOptions{});
	}

	void addLLVMCodegenPipeline(mlir::PassManager& pm, const LLVMCodegenOptions& options)
	{
		pm.addPass(mlir::createCanonicalizerPass());
		pm.addPass(mlir::createCSEPass());
		pm.addPass(createLowerNarrowMatMulPass(options));
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

	std::unique_ptr<llvm::Module> translateToLLVMIR(mlir::ModuleOp module, llvm::LLVMContext& llvmCtx)
	{
		return mlir::translateModuleToLLVMIR(module, llvmCtx);
	}

} // namespace litenn
