#include "Translation/GraphToMLIR.h"
#include "Dialect/LiteNNDialect.h"
#include "Dialect/LiteNNOps.h"

#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/Graph.h>
#include <LiteNN/Tensor.h>
#include <LiteNN/Validation/GraphValidator.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

using namespace mlir;
using namespace LiteNN;

namespace litenn
{

	namespace
	{

		// DataType → MLIR element type
		Type convertElementType(MLIRContext& ctx, DataType dt)
		{
			switch (dt)
			{
			case DataType::Float32:
				return Float32Type::get(&ctx);
			case DataType::Float64:
				return Float64Type::get(&ctx);
			case DataType::Float16:
				return Float16Type::get(&ctx);
			case DataType::BFloat16:
				return BFloat16Type::get(&ctx);
			case DataType::Float8E4M3:
			case DataType::Float8E5M2:
				return IntegerType::get(&ctx, 8);
			case DataType::Int32:
				return IntegerType::get(&ctx, 32);
			case DataType::Int64:
				return IntegerType::get(&ctx, 64);
			case DataType::Int8:
			case DataType::UInt8:
				return IntegerType::get(&ctx, 8);
			case DataType::Bool:
				return IntegerType::get(&ctx, 1);
			}
			llvm_unreachable("unknown DataType");
		}

		// DataType + shape → RankedTensorType
		RankedTensorType convertTensorType(MLIRContext& ctx, DataType dt, ShapeView shape)
		{
			SmallVector<int64_t> dims(shape.Dims.begin(), shape.Dims.end());
			return RankedTensorType::get(dims, convertElementType(ctx, dt));
		}

		RankedTensorType convertTensorType(MLIRContext& ctx, const LiteNN::TensorType& type)
		{
			return convertTensorType(ctx, type.dtype, ShapeView{ type.StaticShape() });
		}

		// LiteNN enum → litenn dialect enum
		UnaryOpKind convertUnaryOp(LiteNN::UnaryOp op)
		{
			return static_cast<UnaryOpKind>(op);
		}
		BinaryOpKind convertBinaryOp(LiteNN::BinaryOp op)
		{
			return static_cast<BinaryOpKind>(op);
		}
		ReduceOpKind convertReduceOp(LiteNN::ReduceOp op)
		{
			return static_cast<ReduceOpKind>(op);
		}
		FusionPatternKind convertFusionPattern(FusionPattern pat)
		{
			return static_cast<FusionPatternKind>(pat);
		}

		OutputInfo ToOutputInfo(const LiteNN::TensorType& type)
		{
			return OutputInfo::FromType(type);
		}

		struct PlanSubgraphView
		{
			const ExecutablePlanSubgraph& subgraph;

			std::size_t NodeCount() const
			{
				return subgraph.nodes.size();
			}

			std::span<const LiteNN::TensorType> Params() const
			{
				return subgraph.params;
			}

			std::span<const NodeOutput> Results() const
			{
				return subgraph.results;
			}

			OutputInfo GetOutputInfo(NodeOutput output) const
			{
				if (output.node >= subgraph.nodes.size())
				{
					throw std::runtime_error("ExecutablePlan MLIR lowering output references an out-of-range node");
				}
				const auto& outputs = subgraph.nodes[output.node].outputs;
				if (output.port >= outputs.size())
				{
					throw std::runtime_error("ExecutablePlan MLIR lowering output references an out-of-range port");
				}
				return ToOutputInfo(outputs[output.port]);
			}
		};

		Tensor<PolymorphicDevice> MakeHostTensorValue(const TensorStorageRef& storage)
		{
			if (storage.type.memorySpace != TensorMemorySpace::Host || storage.region.data == nullptr)
			{
				throw std::runtime_error("MLIR translation requires host-backed executable plan variables");
			}
			const auto byteSize = storage.type.ByteSize().value_or(0);
			if (storage.storageOffsetBytes > storage.region.byteSize ||
			    byteSize > storage.region.byteSize - storage.storageOffsetBytes)
			{
				throw std::runtime_error(
				    "MLIR translation executable plan variable storage is smaller than tensor type");
			}
			Tensor<CPU> data(Uninitialized, storage.type.StaticShape(), storage.type.dtype);
			std::memcpy(data.UnsafeRawData(),
			            static_cast<const std::byte*>(storage.region.data) + storage.storageOffsetBytes, byteSize);
			return data.CopyToDevice(PolymorphicDevice{ CPU{} });
		}

		// Extract tensor data to DenseElementsAttr
		DenseElementsAttr convertTensorToAttr(MLIRContext& ctx, const Tensor<PolymorphicDevice>& tensor)
		{
			auto tensorType = convertTensorType(ctx, tensor.DType(), tensor.Shape());
			auto cpuTensor = tensor.CopyToDevice(CPU{});

			const auto numElements = std::max(ShapeView{ cpuTensor.Shape() }.NumElements(), std::size_t(1));
			const auto* rawData = cpuTensor.UnsafeRawData();

			switch (tensor.DType())
			{
			case DataType::Float32: {
				ArrayRef<float> data(static_cast<const float*>(rawData), numElements);
				return DenseElementsAttr::get(tensorType, data);
			}
			case DataType::Float64: {
				ArrayRef<double> data(static_cast<const double*>(rawData), numElements);
				return DenseElementsAttr::get(tensorType, data);
			}
			case DataType::Float16: {
				SmallVector<Attribute> data;
				data.reserve(numElements);
				const auto* src = static_cast<const Float16*>(rawData);
				for (std::size_t i = 0; i < numElements; ++i)
				{
					data.push_back(FloatAttr::get(tensorType.getElementType(), static_cast<float>(src[i])));
				}
				return DenseElementsAttr::get(tensorType, data);
			}
			case DataType::BFloat16: {
				SmallVector<Attribute> data;
				data.reserve(numElements);
				const auto* src = static_cast<const BFloat16*>(rawData);
				for (std::size_t i = 0; i < numElements; ++i)
				{
					data.push_back(FloatAttr::get(tensorType.getElementType(), static_cast<float>(src[i])));
				}
				return DenseElementsAttr::get(tensorType, data);
			}
			case DataType::Float8E4M3: {
				SmallVector<Attribute> data;
				data.reserve(numElements);
				const auto* src = static_cast<const Float8E4M3*>(rawData);
				for (std::size_t i = 0; i < numElements; ++i)
				{
					data.push_back(IntegerAttr::get(tensorType.getElementType(), llvm::APInt(8, src[i].bits)));
				}
				return DenseElementsAttr::get(tensorType, data);
			}
			case DataType::Float8E5M2: {
				SmallVector<Attribute> data;
				data.reserve(numElements);
				const auto* src = static_cast<const Float8E5M2*>(rawData);
				for (std::size_t i = 0; i < numElements; ++i)
				{
					data.push_back(IntegerAttr::get(tensorType.getElementType(), llvm::APInt(8, src[i].bits)));
				}
				return DenseElementsAttr::get(tensorType, data);
			}
			case DataType::Int32: {
				ArrayRef<int32_t> data(static_cast<const int32_t*>(rawData), numElements);
				return DenseElementsAttr::get(tensorType, data);
			}
			case DataType::Int64: {
				ArrayRef<int64_t> data(static_cast<const int64_t*>(rawData), numElements);
				return DenseElementsAttr::get(tensorType, data);
			}
			case DataType::Int8: {
				ArrayRef<int8_t> data(static_cast<const int8_t*>(rawData), numElements);
				return DenseElementsAttr::get(tensorType, data);
			}
			case DataType::UInt8: {
				SmallVector<Attribute> data;
				data.reserve(numElements);
				const auto* src = static_cast<const uint8_t*>(rawData);
				for (std::size_t i = 0; i < numElements; ++i)
				{
					data.push_back(IntegerAttr::get(tensorType.getElementType(), llvm::APInt(8, src[i])));
				}
				return DenseElementsAttr::get(tensorType, data);
			}
			case DataType::Bool: {
				// MLIR DenseElementsAttr for i1 expects bool
				SmallVector<bool> boolData(numElements);
				const auto* src = static_cast<const bool*>(rawData);
				for (std::size_t i = 0; i < numElements; ++i)
				{
					boolData[i] = src[i];
				}
				return DenseElementsAttr::get(tensorType, ArrayRef(boolData));
			}
			}
			llvm_unreachable("unknown DataType");
		}

		class GraphTranslator
		{
		public:
			GraphTranslator(const ExecutablePlan& plan, MLIRContext& ctx) : plan_(plan), ctx_(ctx), builder_(&ctx)
			{
			}

			OwningOpRef<ModuleOp> translate()
			{
				module_ = ModuleOp::create(builder_.getUnknownLoc());
				builder_.setInsertionPointToStart(module_->getBody());

				// Emit variable declarations
				for (std::size_t i = 0; i < plan_.variables.size(); ++i)
				{
					emitVariable(i);
				}

				// Emit subgraph functions
				for (std::size_t i = 0; i < plan_.subgraphs.size(); ++i)
				{
					emitSubgraphFunc(i);
				}

				return std::move(module_);
			}

		private:
			void emitVariable(std::size_t varIndex)
			{
				const auto& variable = plan_.variables[varIndex];
				auto tensorType = convertTensorType(ctx_, variable.type);
				auto initialValue = convertTensorToAttr(ctx_, MakeHostTensorValue(variable));
				auto name = "var_" + std::to_string(varIndex);

				builder_.create<VariableOp>(builder_.getUnknownLoc(), name, tensorType, initialValue);
			}

			void emitSubgraphFunc(std::size_t sgId)
			{
				const PlanSubgraphView sg{ plan_.subgraphs[sgId] };

				// Build function type
				SmallVector<Type> inputTypes;
				for (const auto& param : sg.Params())
				{
					inputTypes.push_back(convertTensorType(ctx_, param));
				}

				SmallVector<Type> resultTypes;
				for (const auto& result : sg.Results())
				{
					const auto& info = sg.GetOutputInfo(result);
					resultTypes.push_back(convertTensorType(ctx_, info.dtype, info.shape));
				}

				auto funcType = builder_.getFunctionType(inputTypes, resultTypes);
				auto name = "subgraph_" + std::to_string(sgId);

				auto funcOp = builder_.create<FuncOp>(builder_.getUnknownLoc(), name, funcType);

				// Create entry block with arguments
				auto& entryBlock = *funcOp.addEntryBlock();
				OpBuilder::InsertionGuard guard(builder_);
				builder_.setInsertionPointToStart(&entryBlock);

				// Value map: nodeId × port → mlir::Value
				std::vector<SmallVector<Value>> valueMap(sg.NodeCount());

				// Activation slot map for SSA化: slotId → Value
				std::map<std::size_t, Value> activationMap;
				std::map<std::size_t, Value> tapeMap;

				emitSubgraphBody(sg, entryBlock, valueMap, activationMap, tapeMap);

				// Emit return
				SmallVector<Value> returnValues;
				for (const auto& result : sg.Results())
				{
					returnValues.push_back(valueMap[result.node][result.port]);
				}

				builder_.create<ReturnOp>(builder_.getUnknownLoc(), returnValues);
			}

			void emitSubgraphBody(const PlanSubgraphView& sg, Block& block, std::vector<SmallVector<Value>>& valueMap,
			                      std::map<std::size_t, Value>& activationMap, std::map<std::size_t, Value>& tapeMap)
			{
				(void) block;
				for (NodeId nodeId = 0; nodeId < sg.NodeCount(); ++nodeId)
				{
					const auto& entry = sg.subgraph.nodes[nodeId];
					std::vector<OutputInfo> outputInfos;
					outputInfos.reserve(entry.outputs.size());
					for (const auto& output : entry.outputs)
					{
						outputInfos.push_back(ToOutputInfo(output));
					}
					std::visit(
					    [&](const auto& node) {
						    emitNode(sg, nodeId, node, outputInfos, valueMap, activationMap, tapeMap);
					    },
					    entry.node);
				}
			}

			Value getVal(const std::vector<SmallVector<Value>>& valueMap, NodeOutput output)
			{
				return valueMap[output.node][output.port];
			}

			Value emitFilledConstant(DataType dtype, std::span<const std::size_t> shape, double value)
			{
				auto ownedShape = std::vector<std::size_t>(shape.begin(), shape.end());
				Tensor<CPU> tensor(Uninitialized, ownedShape, dtype);
				EnumDispatch(dtype, [&]<DataType TypeValue> {
					using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
					auto* data = static_cast<T*>(tensor.UnsafeRawData());
					std::fill(data, data + tensor.NumElements(), static_cast<T>(value));
				});
				auto poly = tensor.CopyToDevice(PolymorphicDevice{ CPU{} });
				auto attr = convertTensorToAttr(ctx_, poly);
				auto op = builder_.create<ConstantOp>(builder_.getUnknownLoc(), attr.getType(), attr);
				return op.getResult();
			}

			Value emitDenseConstant(DataType dtype, std::span<const std::size_t> shape, std::span<const double> values)
			{
				auto ownedShape = std::vector<std::size_t>(shape.begin(), shape.end());
				Tensor<CPU> tensor(Uninitialized, ownedShape, dtype);
				if (tensor.NumElements() != values.size())
				{
					throw std::runtime_error("MLIR dense constant element count does not match shape");
				}
				EnumDispatch(dtype, [&]<DataType TypeValue> {
					using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
					auto* data = static_cast<T*>(tensor.UnsafeRawData());
					for (std::size_t i = 0; i < values.size(); ++i)
					{
						data[i] = static_cast<T>(values[i]);
					}
				});
				auto poly = tensor.CopyToDevice(PolymorphicDevice{ CPU{} });
				auto attr = convertTensorToAttr(ctx_, poly);
				auto op = builder_.create<ConstantOp>(builder_.getUnknownLoc(), attr.getType(), attr);
				return op.getResult();
			}

			std::vector<std::size_t> quantizationAxisBroadcastShape(const QuantizationParams& params,
			                                                        std::span<const std::size_t> outputShape) const
			{
				auto shape = std::vector<std::size_t>(outputShape.size(), std::size_t{ 1 });
				const auto axis = QuantizationDetail::NormalizeAxis(params.axis, ShapeView{ outputShape });
				shape[axis] = outputShape[axis];
				return shape;
			}

			AffineMap buildBroadcastAffineMap(std::span<const std::size_t> inputShape,
			                                  std::span<const std::size_t> resultShape)
			{
				const auto resultRank = static_cast<std::int64_t>(resultShape.size());
				const auto inputRank = static_cast<std::int64_t>(inputShape.size());
				const auto rankDiff = resultRank - inputRank;
				if (rankDiff < 0)
				{
					throw std::runtime_error("MLIR broadcast input rank exceeds result rank");
				}

				SmallVector<AffineExpr> exprs;
				exprs.reserve(inputShape.size());
				for (std::int64_t i = 0; i < inputRank; ++i)
				{
					const auto resultDim = i + rankDiff;
					if (inputShape[static_cast<std::size_t>(i)] == 1 &&
					    resultShape[static_cast<std::size_t>(resultDim)] != 1)
					{
						exprs.push_back(getAffineConstantExpr(0, &ctx_));
					}
					else
					{
						exprs.push_back(getAffineDimExpr(resultDim, &ctx_));
					}
				}
				return AffineMap::get(resultRank, 0, exprs, &ctx_);
			}

			Value emitQuantizationParameterConstant(const QuantizationParams& params, DataType dtype,
			                                        std::span<const std::size_t> outputShape,
			                                        std::span<const double> values)
			{
				if (params.granularity == QuantizationGranularity::PerAxis)
				{
					const auto broadcastShape = quantizationAxisBroadcastShape(params, outputShape);
					if (values.size() != ShapeView{ broadcastShape }.NumElements())
					{
						throw std::runtime_error(
						    "GraphToMLIR DequantizeNode per-axis parameter count does not match output axis shape");
					}
					return emitDenseConstant(dtype, broadcastShape, values);
				}

				const auto shape = ShapeView{ outputShape };
				if (values.size() != QuantizationDetail::ExpectedScaleCount(params, shape))
				{
					throw std::runtime_error(
					    "GraphToMLIR DequantizeNode grouped parameter count does not match output shape");
				}
				std::vector<double> expanded(shape.NumElements());
				for (std::size_t i = 0; i < expanded.size(); ++i)
				{
					expanded[i] = values[QuantizationDetail::ScaleIndexForElement(params, shape, i)];
				}
				return emitDenseConstant(dtype, outputShape, expanded);
			}

			Value emitAffineQuantizeValue(Value input, const QuantizationParams& params, const OutputInfo& output)
			{
				if (params.scheme != QuantizationScheme::Affine ||
				    (params.granularity != QuantizationGranularity::PerTensor &&
				     params.granularity != QuantizationGranularity::PerAxis &&
				     params.granularity != QuantizationGranularity::Grouped) ||
				    (params.granularity == QuantizationGranularity::PerTensor && params.scales.size() != 1) ||
				    (!params.zeroPoints.empty() && params.zeroPoints.size() != params.scales.size()))
				{
					throw std::runtime_error(
					    "GraphToMLIR dynamic QuantizeNode currently supports affine per-tensor/per-axis/grouped "
					    "quantization only");
				}
				if (params.storageType != DataType::Int8 && params.storageType != DataType::UInt8)
				{
					throw std::runtime_error("GraphToMLIR dynamic QuantizeNode currently supports Int8/UInt8 storage");
				}

				Value scaleValue;
				Value zeroPointValue;
				std::vector<std::size_t> scaleShape;
				std::vector<std::size_t> zeroPointShape;
				if (params.granularity == QuantizationGranularity::PerTensor)
				{
					scaleShape = output.shape;
					zeroPointShape = output.shape;
					scaleValue =
					    emitFilledConstant(DataType::Float32, scaleShape, static_cast<double>(params.scales[0]));
					const auto zeroPoint = params.zeroPoints.empty() ? 0 : params.zeroPoints[0];
					zeroPointValue =
					    emitFilledConstant(DataType::Float32, zeroPointShape, static_cast<double>(zeroPoint));
				}
				else
				{
					std::vector<double> scales;
					scales.reserve(params.scales.size());
					for (const auto scale : params.scales)
					{
						scales.push_back(static_cast<double>(scale));
					}
					scaleShape = params.granularity == QuantizationGranularity::PerAxis
					                 ? quantizationAxisBroadcastShape(params, output.shape)
					                 : output.shape;
					scaleValue = emitQuantizationParameterConstant(params, DataType::Float32, output.shape, scales);

					if (params.zeroPoints.empty())
					{
						zeroPointShape = output.shape;
						zeroPointValue = emitFilledConstant(DataType::Float32, zeroPointShape, 0.0);
					}
					else
					{
						std::vector<double> zeroPoints;
						zeroPoints.reserve(params.zeroPoints.size());
						for (const auto zeroPoint : params.zeroPoints)
						{
							zeroPoints.push_back(static_cast<double>(zeroPoint));
						}
						zeroPointShape = params.granularity == QuantizationGranularity::PerAxis
						                     ? quantizationAxisBroadcastShape(params, output.shape)
						                     : output.shape;
						zeroPointValue =
						    emitQuantizationParameterConstant(params, DataType::Float32, output.shape, zeroPoints);
					}
				}

				const auto resultType = convertTensorType(ctx_, params.storageType, output.shape);
				const auto outputMap =
				    AffineMap::getMultiDimIdentityMap(static_cast<int64_t>(output.shape.size()), &ctx_);
				const auto inputMap = outputMap;
				const auto scaleMap = buildBroadcastAffineMap(scaleShape, output.shape);
				const auto zeroPointMap = buildBroadcastAffineMap(zeroPointShape, output.shape);
				auto emptyOut = builder_.create<tensor::EmptyOp>(builder_.getUnknownLoc(), resultType.getShape(),
				                                                 resultType.getElementType());
				SmallVector<utils::IteratorType> iterTypes(resultType.getRank(), utils::IteratorType::parallel);
				const auto minValue = params.storageType == DataType::Int8
				                          ? static_cast<double>(std::numeric_limits<std::int8_t>::min())
				                          : static_cast<double>(std::numeric_limits<std::uint8_t>::min());
				const auto maxValue = params.storageType == DataType::Int8
				                          ? static_cast<double>(std::numeric_limits<std::int8_t>::max())
				                          : static_cast<double>(std::numeric_limits<std::uint8_t>::max());
				auto generic = builder_.create<linalg::GenericOp>(
				    builder_.getUnknownLoc(), TypeRange{ resultType }, ValueRange{ input, scaleValue, zeroPointValue },
				    ValueRange{ emptyOut }, SmallVector<AffineMap>{ inputMap, scaleMap, zeroPointMap, outputMap },
				    iterTypes, [&](OpBuilder& b, Location loc, ValueRange args) {
					    auto inputElemType = cast<RankedTensorType>(input.getType()).getElementType();
					    auto scaled = b.create<arith::DivFOp>(loc, args[0], args[1]).getResult();
					    auto rounded = b.create<math::RoundOp>(loc, scaled).getResult();
					    auto shifted = b.create<arith::AddFOp>(loc, rounded, args[2]).getResult();
					    auto minConst = b.create<arith::ConstantFloatOp>(loc, cast<FloatType>(inputElemType),
					                                                     llvm::APFloat(static_cast<float>(minValue)));
					    auto maxConst = b.create<arith::ConstantFloatOp>(loc, cast<FloatType>(inputElemType),
					                                                     llvm::APFloat(static_cast<float>(maxValue)));
					    auto clampedMin = b.create<arith::MaximumFOp>(loc, shifted, minConst).getResult();
					    auto clamped = b.create<arith::MinimumFOp>(loc, clampedMin, maxConst).getResult();
					    Value result =
					        params.storageType == DataType::UInt8
					            ? b.create<arith::FPToUIOp>(loc, resultType.getElementType(), clamped).getResult()
					            : b.create<arith::FPToSIOp>(loc, resultType.getElementType(), clamped).getResult();
					    b.create<linalg::YieldOp>(loc, result);
				    });
				return generic.getResult(0);
			}

			Value emitUnaryValue(LiteNN::UnaryOp opKind, Value input, DataType dtype,
			                     std::span<const std::size_t> shape)
			{
				auto resultType = convertTensorType(ctx_, dtype, shape);
				auto op = builder_.create<litenn::UnaryOp>(builder_.getUnknownLoc(), resultType, convertUnaryOp(opKind),
				                                           input);
				return op.getResult();
			}

			Value emitBinaryValue(LiteNN::BinaryOp opKind, Value lhs, Value rhs, DataType dtype,
			                      std::span<const std::size_t> shape)
			{
				auto resultType = convertTensorType(ctx_, dtype, shape);
				auto op = builder_.create<litenn::BinaryOp>(builder_.getUnknownLoc(), resultType,
				                                            convertBinaryOp(opKind), lhs, rhs);
				return op.getResult();
			}

			bool shouldUseFloat32Accumulator(DataType dtype) const
			{
				return dtype == DataType::Float16;
			}

			Value emitCastValue(Value input, DataType dtype, std::span<const std::size_t> shape)
			{
				auto resultType = convertTensorType(ctx_, dtype, shape);
				auto op = builder_.create<CastOp>(builder_.getUnknownLoc(), resultType, input);
				return op.getResult();
			}

			Value emitMaybeCastValue(Value input, DataType inputDType, DataType outputDType,
			                         std::span<const std::size_t> shape)
			{
				if (inputDType == outputDType)
				{
					return input;
				}
				return emitCastValue(input, outputDType, shape);
			}

			Value emitReduceValue(LiteNN::ReduceOp opKind, Value input, DataType dtype,
			                      std::span<const std::size_t> shape, std::size_t axis)
			{
				auto inputType = cast<RankedTensorType>(input.getType());
				if (inputType.getRank() == 1 && axis == 0)
				{
					return emitRankOneReduceValue(opKind, input, dtype, shape);
				}

				auto resultType = convertTensorType(ctx_, dtype, shape);
				auto op = builder_.create<litenn::ReduceOp>(
				    builder_.getUnknownLoc(), resultType, convertReduceOp(opKind), input, static_cast<uint64_t>(axis));
				return op.getResult();
			}

			Value emitRankOneReduceValue(LiteNN::ReduceOp opKind, Value input, DataType dtype,
			                             std::span<const std::size_t> shape)
			{
				auto loc = builder_.getUnknownLoc();
				auto inputType = cast<RankedTensorType>(input.getType());
				auto resultType = convertTensorType(ctx_, dtype, shape);
				auto elemType = resultType.getElementType();

				Value initValue;
				if (opKind == LiteNN::ReduceOp::Max || opKind == LiteNN::ReduceOp::Min)
				{
					if (auto floatType = dyn_cast<FloatType>(elemType))
					{
						initValue = builder_.create<arith::ConstantFloatOp>(
						    loc, floatType,
						    llvm::APFloat::getInf(floatType.getFloatSemantics(),
						                          /*negative=*/
						                          opKind == LiteNN::ReduceOp::Max));
					}
					else
					{
						auto intType = cast<IntegerType>(elemType);
						const auto initInt = opKind == LiteNN::ReduceOp::Max
						                         ? llvm::APInt::getSignedMinValue(intType.getWidth()).getSExtValue()
						                         : llvm::APInt::getSignedMaxValue(intType.getWidth()).getSExtValue();
						initValue = builder_.create<arith::ConstantIntOp>(loc, elemType, initInt);
					}
				}
				else
				{
					initValue = emitScalarZero(builder_, loc, elemType);
				}

				auto empty = builder_.create<tensor::EmptyOp>(loc, resultType.getShape(), elemType);
				auto filled =
				    builder_.create<linalg::FillOp>(loc, ValueRange{ initValue }, ValueRange{ empty }).getResult(0);
				auto inputMap = AffineMap::getMultiDimIdentityMap(1, &ctx_);
				auto outputMap = AffineMap::get(1, 0, { getAffineConstantExpr(0, &ctx_) }, &ctx_);

				auto generic = builder_.create<linalg::GenericOp>(
				    loc, TypeRange{ resultType }, ValueRange{ input }, ValueRange{ filled },
				    SmallVector<AffineMap>{ inputMap, outputMap },
				    SmallVector<utils::IteratorType>{ utils::IteratorType::reduction },
				    [&](OpBuilder& b, Location l, ValueRange args) {
					    Value result;
					    if (opKind == LiteNN::ReduceOp::Max)
					    {
						    result = isa<FloatType>(elemType)
						                 ? b.create<arith::MaximumFOp>(l, args[1], args[0]).getResult()
						                 : b.create<arith::MaxSIOp>(l, args[1], args[0]).getResult();
					    }
					    else if (opKind == LiteNN::ReduceOp::Min)
					    {
						    result = isa<FloatType>(elemType)
						                 ? b.create<arith::MinimumFOp>(l, args[1], args[0]).getResult()
						                 : b.create<arith::MinSIOp>(l, args[1], args[0]).getResult();
					    }
					    else
					    {
						    result = emitScalarAdd(b, l, args[1], args[0], elemType);
					    }
					    b.create<linalg::YieldOp>(l, result);
				    });

				Value result = generic.getResult(0);
				if (opKind == LiteNN::ReduceOp::Mean)
				{
					const auto divisor = emitFilledConstant(dtype, shape, static_cast<double>(inputType.getDimSize(0)));
					result = emitBinaryValue(LiteNN::BinaryOp::Divide, result, divisor, dtype, shape);
				}
				return result;
			}

			Value emitReduceAllToSingleValue(LiteNN::ReduceOp opKind, Value input, DataType dtype,
			                                 std::span<const std::size_t> shape)
			{
				(void) shape;
				if (opKind != LiteNN::ReduceOp::Sum)
				{
					throw std::runtime_error("GraphToMLIR reduce-all helper currently supports Sum only");
				}

				auto loc = builder_.getUnknownLoc();
				auto inputType = cast<RankedTensorType>(input.getType());
				const auto inputRank = inputType.getRank();
				const auto scalarShape = std::vector<std::size_t>{ 1 };
				auto resultType = convertTensorType(ctx_, dtype, scalarShape);
				auto empty = builder_.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
				auto zero = emitScalarZero(builder_, loc, resultType.getElementType());
				auto filled =
				    builder_.create<linalg::FillOp>(loc, ValueRange{ zero }, ValueRange{ empty }).getResult(0);

				auto inputMap = AffineMap::getMultiDimIdentityMap(inputRank, &ctx_);
				auto outputMap = AffineMap::get(inputRank, 0, { getAffineConstantExpr(0, &ctx_) }, &ctx_);
				SmallVector<utils::IteratorType> iterTypes(inputRank, utils::IteratorType::reduction);
				auto elemType = resultType.getElementType();

				auto generic = builder_.create<linalg::GenericOp>(
				    loc, TypeRange{ resultType }, ValueRange{ input }, ValueRange{ filled },
				    SmallVector<AffineMap>{ inputMap, outputMap }, iterTypes,
				    [&](OpBuilder& b, Location l, ValueRange args) {
					    b.create<linalg::YieldOp>(l, emitScalarAdd(b, l, args[1], args[0], elemType));
				    });
				return generic.getResult(0);
			}

			Value emitReshapeValue(Value input, DataType dtype, std::span<const std::size_t> shape)
			{
				auto resultType = convertTensorType(ctx_, dtype, shape);
				auto op = builder_.create<ReshapeOp>(builder_.getUnknownLoc(), resultType, input);
				return op.getResult();
			}

			Value emitSoftmaxValueTyped(Value input, DataType dtype, std::span<const std::size_t> shape,
			                            std::size_t axis)
			{
				const auto inputShape = ShapeView{ shape };
				const auto reducedShape = ReducedShape(inputShape, axis);
				const auto broadcastShape = BroadcastShapeForAxis(inputShape, axis);

				auto max = emitReduceValue(LiteNN::ReduceOp::Max, input, dtype, reducedShape, axis);
				auto maxBroadcast = emitReshapeValue(max, dtype, broadcastShape);
				auto shifted = emitBinaryValue(LiteNN::BinaryOp::Subtract, input, maxBroadcast, dtype, shape);
				auto exp = emitUnaryValue(LiteNN::UnaryOp::Exp, shifted, dtype, shape);
				auto denom = emitReduceValue(LiteNN::ReduceOp::Sum, exp, dtype, reducedShape, axis);
				auto denomBroadcast = emitReshapeValue(denom, dtype, broadcastShape);
				return emitBinaryValue(LiteNN::BinaryOp::Divide, exp, denomBroadcast, dtype, shape);
			}

			Value emitSoftmaxValue(Value input, DataType dtype, std::span<const std::size_t> shape, std::size_t axis)
			{
				if (!shouldUseFloat32Accumulator(dtype))
				{
					return emitSoftmaxValueTyped(input, dtype, shape, axis);
				}

				auto wideInput = emitCastValue(input, DataType::Float32, shape);
				auto wideResult = emitSoftmaxValueTyped(wideInput, DataType::Float32, shape, axis);
				return emitCastValue(wideResult, dtype, shape);
			}

			Value emitBroadcastToValue(Value input, DataType dtype, std::span<const std::size_t> targetShape)
			{
				auto loc = builder_.getUnknownLoc();
				auto inputType = cast<RankedTensorType>(input.getType());
				auto resultType = convertTensorType(ctx_, dtype, targetShape);
				const auto inputRank = inputType.getRank();
				const auto resultRank = resultType.getRank();
				const auto rankOffset = resultRank - inputRank;

				SmallVector<AffineExpr> inputExprs;
				inputExprs.reserve(inputRank);
				for (int64_t dim = 0; dim < inputRank; ++dim)
				{
					const auto resultDim = rankOffset + dim;
					const bool isBroadcastDim = inputType.getDimSize(dim) == 1 && resultType.getDimSize(resultDim) != 1;
					inputExprs.push_back(isBroadcastDim ? getAffineConstantExpr(0, &ctx_)
					                                    : getAffineDimExpr(resultDim, &ctx_));
				}

				auto inputMap = AffineMap::get(resultRank, 0, inputExprs, &ctx_);
				auto outputMap = AffineMap::getMultiDimIdentityMap(resultRank, &ctx_);
				auto empty = builder_.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
				SmallVector<utils::IteratorType> iterTypes(resultRank, utils::IteratorType::parallel);

				auto generic = builder_.create<linalg::GenericOp>(
				    loc, TypeRange{ resultType }, ValueRange{ input }, ValueRange{ empty },
				    SmallVector<AffineMap>{ inputMap, outputMap }, iterTypes,
				    [](OpBuilder& b, Location l, ValueRange args) { b.create<linalg::YieldOp>(l, args[0]); });
				return generic.getResult(0);
			}

			static std::vector<std::size_t> ReducedShape(ShapeView inputShape, std::size_t axis)
			{
				std::vector<std::size_t> result;
				result.reserve(inputShape.NumDim() - 1);
				for (auto dim = 0uz; dim < inputShape.NumDim(); ++dim)
				{
					if (dim != axis)
					{
						result.push_back(inputShape[dim]);
					}
				}
				if (result.empty())
				{
					result.push_back(1);
				}
				return result;
			}

			static std::vector<std::size_t> BroadcastShapeForAxis(ShapeView inputShape, std::size_t axis)
			{
				auto result = inputShape.ToOwned();
				result[axis] = 1;
				return result;
			}

			Value emitScalarConstant(OpBuilder& b, Location loc, Type elemType, double value)
			{
				if (auto floatType = dyn_cast<FloatType>(elemType))
				{
					llvm::APFloat floatValue(value);
					bool losesInfo = false;
					floatValue.convert(floatType.getFloatSemantics(), llvm::APFloat::rmNearestTiesToEven, &losesInfo);
					return b.create<arith::ConstantFloatOp>(loc, floatType, floatValue);
				}
				return b.create<arith::ConstantIntOp>(loc, elemType, static_cast<std::int64_t>(value));
			}

			Value emitScalarToF32(OpBuilder& b, Location loc, Value value)
			{
				auto f32 = b.getF32Type();
				if (value.getType() == f32)
				{
					return value;
				}
				if (auto floatType = dyn_cast<FloatType>(value.getType()))
				{
					return floatType.getWidth() < 32 ? b.create<arith::ExtFOp>(loc, f32, value).getResult()
					                                 : b.create<arith::TruncFOp>(loc, f32, value).getResult();
				}
				throw std::runtime_error("GraphToMLIR expected floating-point scalar");
			}

			Value emitIntegerScalarToF32(OpBuilder& b, Location loc, Value value, bool isUnsigned)
			{
				auto f32 = b.getF32Type();
				return isUnsigned ? b.create<arith::UIToFPOp>(loc, f32, value).getResult()
				                  : b.create<arith::SIToFPOp>(loc, f32, value).getResult();
			}

			Value emitScalarFromF32(OpBuilder& b, Location loc, Value value, Type targetType)
			{
				if (value.getType() == targetType)
				{
					return value;
				}
				auto targetFloatType = dyn_cast<FloatType>(targetType);
				if (!targetFloatType)
				{
					throw std::runtime_error("GraphToMLIR expected floating-point output scalar");
				}
				const auto sourceWidth = value.getType().getIntOrFloatBitWidth();
				const auto targetWidth = targetType.getIntOrFloatBitWidth();
				if (sourceWidth < targetWidth)
				{
					return b.create<arith::ExtFOp>(loc, targetType, value).getResult();
				}
				return b.create<arith::TruncFOp>(loc, targetType, value).getResult();
			}

			Value emitScalarZero(OpBuilder& b, Location loc, Type elemType)
			{
				return emitScalarConstant(b, loc, elemType, 0.0);
			}

			Value emitScalarAdd(OpBuilder& b, Location loc, Value lhs, Value rhs, Type elemType)
			{
				return isa<FloatType>(elemType) ? b.create<arith::AddFOp>(loc, lhs, rhs).getResult()
				                                : b.create<arith::AddIOp>(loc, lhs, rhs).getResult();
			}

			Value emitScalarMultiply(OpBuilder& b, Location loc, Value lhs, Value rhs, Type elemType)
			{
				return isa<FloatType>(elemType) ? b.create<arith::MulFOp>(loc, lhs, rhs).getResult()
				                                : b.create<arith::MulIOp>(loc, lhs, rhs).getResult();
			}

			Value emitI64Constant(OpBuilder& b, Location loc, std::int64_t value)
			{
				return b.create<arith::ConstantIntOp>(loc, b.getI64Type(), value);
			}

			Value emitIndexAsI64(OpBuilder& b, Location loc, int64_t dim)
			{
				auto index = b.create<linalg::IndexOp>(loc, dim).getResult();
				return b.create<arith::IndexCastOp>(loc, b.getI64Type(), index).getResult();
			}

			Value emitI64ToIndex(OpBuilder& b, Location loc, Value value)
			{
				return b.create<arith::IndexCastOp>(loc, b.getIndexType(), value).getResult();
			}

			Value emitI64Max(OpBuilder& b, Location loc, Value lhs, Value rhs)
			{
				auto greater = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, lhs, rhs).getResult();
				return b.create<arith::SelectOp>(loc, greater, lhs, rhs).getResult();
			}

			Value emitI64Min(OpBuilder& b, Location loc, Value lhs, Value rhs)
			{
				auto less = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, lhs, rhs).getResult();
				return b.create<arith::SelectOp>(loc, less, lhs, rhs).getResult();
			}

			Value emitClampedI64(OpBuilder& b, Location loc, Value value, std::int64_t lower, std::int64_t upper)
			{
				auto lo = emitI64Constant(b, loc, lower);
				auto hi = emitI64Constant(b, loc, upper);
				return emitI64Min(b, loc, emitI64Max(b, loc, value, lo), hi);
			}

			Value emitReflectedI64(OpBuilder& b, Location loc, Value value, std::int64_t size)
			{
				const auto periodValue = 2 * size - 2;
				auto period = emitI64Constant(b, loc, periodValue);
				auto mod = b.create<arith::RemSIOp>(loc, value, period).getResult();
				auto zero = emitI64Constant(b, loc, 0);
				auto isNegative = b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, mod, zero).getResult();
				auto normalized = b.create<arith::SelectOp>(loc, isNegative,
				                                            b.create<arith::AddIOp>(loc, mod, period).getResult(), mod)
				                      .getResult();
				auto sizeConstant = emitI64Constant(b, loc, size);
				auto inLowHalf =
				    b.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, normalized, sizeConstant).getResult();
				auto mirrored = b.create<arith::SubIOp>(loc, period, normalized).getResult();
				return b.create<arith::SelectOp>(loc, inLowHalf, normalized, mirrored).getResult();
			}

			SmallVector<Type> convertOutputInfos(std::span<const OutputInfo> infos)
			{
				SmallVector<Type> types;
				for (const auto& info : infos)
				{
					types.push_back(convertTensorType(ctx_, info.dtype, info.shape));
				}
				return types;
			}

			// ---- Per-node emission ----

			void emitNode(const PlanSubgraphView& sg, NodeId nodeId, const ParamRefNode& node,
			              std::span<const OutputInfo> /*outputInfos*/, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				// Block arguments correspond to params
				auto* block = builder_.getInsertionBlock();
				valueMap[nodeId] = { block->getArgument(node.paramIndex) };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const ConstantNode& node, std::span<const OutputInfo>,
			              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>&,
			              std::map<std::size_t, Value>&)
			{
				auto attr = convertTensorToAttr(ctx_, node.value);
				auto op = builder_.create<ConstantOp>(builder_.getUnknownLoc(), attr.getType(), attr);
				valueMap[nodeId] = { op.getResult() };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const QuantizedConstantNode& node,
			              std::span<const OutputInfo>, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto attr = convertTensorToAttr(ctx_, node.storage);
				auto op = builder_.create<ConstantOp>(builder_.getUnknownLoc(), attr.getType(), attr);
				valueMap[nodeId] = { op.getResult() };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const VariableRefNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
				auto name = "var_" + std::to_string(node.variableIndex);
				auto op = builder_.create<GetVariableOp>(builder_.getUnknownLoc(), resultType,
				                                         FlatSymbolRefAttr::get(&ctx_, name));
				valueMap[nodeId] = { op.getResult() };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const UnaryOpNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
				auto input = getVal(valueMap, node.input);
				auto op = builder_.create<litenn::UnaryOp>(builder_.getUnknownLoc(), resultType,
				                                           convertUnaryOp(node.op), input);
				valueMap[nodeId] = { op.getResult() };
			}

			void emitNode(const PlanSubgraphView& sg, NodeId nodeId, const BinaryOpNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				const auto dtype = outputInfos[0].dtype;
				auto lhs = getVal(valueMap, node.lhs);
				auto rhs = getVal(valueMap, node.rhs);
				if (node.op == LiteNN::BinaryOp::MatMul && shouldUseFloat32Accumulator(dtype))
				{
					const auto lhsInfo = sg.GetOutputInfo(node.lhs);
					const auto rhsInfo = sg.GetOutputInfo(node.rhs);
					lhs = emitMaybeCastValue(lhs, lhsInfo.dtype, DataType::Float32, lhsInfo.shape);
					rhs = emitMaybeCastValue(rhs, rhsInfo.dtype, DataType::Float32, rhsInfo.shape);
					auto wide = emitBinaryValue(node.op, lhs, rhs, DataType::Float32, outputInfos[0].shape);
					valueMap[nodeId] = { emitCastValue(wide, dtype, outputInfos[0].shape) };
					return;
				}

				auto resultType = convertTensorType(ctx_, dtype, outputInfos[0].shape);
				auto op = builder_.create<litenn::BinaryOp>(builder_.getUnknownLoc(), resultType,
				                                            convertBinaryOp(node.op), lhs, rhs);
				valueMap[nodeId] = { op.getResult() };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const CastNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
				auto input = getVal(valueMap, node.input);
				auto op = builder_.create<CastOp>(builder_.getUnknownLoc(), resultType, input);
				valueMap[nodeId] = { op.getResult() };
			}

			void emitNode(const PlanSubgraphView& sg, NodeId nodeId, const QuantizeNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				const auto inputInfo = sg.GetOutputInfo(node.input);
				auto input = getVal(valueMap, node.input);
				auto computeInput = emitMaybeCastValue(input, inputInfo.dtype, DataType::Float32, outputInfos[0].shape);
				valueMap[nodeId] = { emitAffineQuantizeValue(computeInput, node.params, outputInfos[0]) };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const DequantizeNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				if (node.params.scheme != QuantizationScheme::Affine ||
				    (node.params.granularity != QuantizationGranularity::PerTensor &&
				     node.params.granularity != QuantizationGranularity::PerAxis &&
				     node.params.granularity != QuantizationGranularity::Grouped) ||
				    (node.params.granularity == QuantizationGranularity::PerTensor && node.params.scales.size() != 1) ||
				    (!node.params.zeroPoints.empty() && node.params.zeroPoints.size() != node.params.scales.size()))
				{
					throw std::runtime_error(
					    "GraphToMLIR dynamic DequantizeNode currently supports affine per-tensor/per-axis/grouped "
					    "quantization only; run ConstFoldPass before CPU AOT when dequantizing compile-time constants");
				}
				if (!IsFloatingDataType(node.targetType))
				{
					throw std::runtime_error("GraphToMLIR DequantizeNode target type must be floating-point");
				}
				const auto& output = outputInfos[0];
				auto input = getVal(valueMap, node.input);
				auto casted = emitCastValue(input, node.targetType, output.shape);
				if (node.params.granularity == QuantizationGranularity::PerTensor)
				{
					const auto zeroPoint = node.params.zeroPoints.empty() ? 0 : node.params.zeroPoints[0];
					if (zeroPoint != 0)
					{
						auto zeroPointValue =
						    emitFilledConstant(node.targetType, output.shape, static_cast<double>(zeroPoint));
						casted = emitBinaryValue(LiteNN::BinaryOp::Subtract, casted, zeroPointValue, node.targetType,
						                         output.shape);
					}
					auto scale =
					    emitFilledConstant(node.targetType, output.shape, static_cast<double>(node.params.scales[0]));
					valueMap[nodeId] = { emitBinaryValue(LiteNN::BinaryOp::Multiply, casted, scale, node.targetType,
						                                 output.shape) };
					return;
				}

				if (!node.params.zeroPoints.empty())
				{
					std::vector<double> zeroPoints;
					zeroPoints.reserve(node.params.zeroPoints.size());
					for (const auto zeroPoint : node.params.zeroPoints)
					{
						zeroPoints.push_back(static_cast<double>(zeroPoint));
					}
					auto zeroPointValue =
					    emitQuantizationParameterConstant(node.params, node.targetType, output.shape, zeroPoints);
					casted = emitBinaryValue(LiteNN::BinaryOp::Subtract, casted, zeroPointValue, node.targetType,
					                         output.shape);
				}
				std::vector<double> scales;
				scales.reserve(node.params.scales.size());
				for (const auto scale : node.params.scales)
				{
					scales.push_back(static_cast<double>(scale));
				}
				auto scale = emitQuantizationParameterConstant(node.params, node.targetType, output.shape, scales);
				valueMap[nodeId] = { emitBinaryValue(LiteNN::BinaryOp::Multiply, casted, scale, node.targetType,
					                                 output.shape) };
			}

			void emitNode(const PlanSubgraphView& sg, NodeId nodeId, const QuantizedMatMulNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				const auto isAffineQuantizedMatMul = node.params.scheme == QuantizationScheme::Affine;
				const auto isPackedNibbleQuantizedMatMul = node.params.scheme == QuantizationScheme::Block &&
				                                           IsPackedNibbleQuantizedBlockFormat(node.params.blockFormat);
				const auto isGGMLBlockQuantizedMatMul = node.params.scheme == QuantizationScheme::Block &&
				                                        (node.params.blockFormat == QuantizedBlockFormat::GGML_Q4_K ||
				                                         node.params.blockFormat == QuantizedBlockFormat::GGML_Q5_K ||
				                                         node.params.blockFormat == QuantizedBlockFormat::GGML_Q6_K ||
				                                         node.params.blockFormat == QuantizedBlockFormat::GGML_Q8_0);
				if (node.transposeRhs && !isGGMLBlockQuantizedMatMul)
				{
					throw std::runtime_error(
					    "GraphToMLIR QuantizedMatMulNode CPU AOT lowering currently requires non-transposed rhs");
				}
				if ((!isAffineQuantizedMatMul || (node.params.granularity != QuantizationGranularity::PerTensor &&
				                                  node.params.granularity != QuantizationGranularity::PerAxis &&
				                                  node.params.granularity != QuantizationGranularity::Grouped)) &&
				    !isPackedNibbleQuantizedMatMul && !isGGMLBlockQuantizedMatMul)
				{
					throw std::runtime_error("GraphToMLIR QuantizedMatMulNode CPU AOT lowering currently supports "
					                         "affine per-tensor/per-axis/grouped, packed-nibble, GGML_Q4_K, GGML_Q5_K, "
					                         "GGML_Q6_K, and GGML_Q8_0 weights only");
				}
				if (isAffineQuantizedMatMul && node.params.storageType != DataType::Int8 &&
				    node.params.storageType != DataType::UInt8)
				{
					throw std::runtime_error(
					    "GraphToMLIR QuantizedMatMulNode CPU AOT lowering currently supports Int8/UInt8 storage");
				}
				if (isPackedNibbleQuantizedMatMul &&
				    (node.params.storageType != DataType::UInt8 ||
				     !IsIntegerPackedNibbleFormat(node.params.packedFormat) ||
				     node.params.granularity != QuantizationGranularity::PerTensor || node.params.scales.size() != 1 ||
				     (!node.params.zeroPoints.empty() && node.params.zeroPoints.size() != 1)))
				{
					throw std::runtime_error(
					    "GraphToMLIR QuantizedMatMulNode CPU AOT lowering currently supports per-tensor Int4/UInt4 "
					    "packed-nibble weights only");
				}
				if (outputInfos.size() != 1)
				{
					throw std::runtime_error("GraphToMLIR QuantizedMatMulNode expected one output");
				}
				const auto lhsInfo = sg.GetOutputInfo(node.lhs);
				const auto rhsInfo = sg.GetOutputInfo(node.rhsStorage);
				if (lhsInfo.shape.size() != 2 || outputInfos[0].shape.size() != 2)
				{
					throw std::runtime_error(
					    "GraphToMLIR QuantizedMatMulNode CPU AOT lowering requires rank-2 tensors");
				}
				const auto rhsShape = node.params.expressedShape.empty() ? rhsInfo.shape : node.params.expressedShape;
				const auto expectedRhsRows = node.transposeRhs ? outputInfos[0].shape[1] : lhsInfo.shape[1];
				const auto expectedRhsColumns = node.transposeRhs ? lhsInfo.shape[1] : outputInfos[0].shape[1];
				if (rhsShape.size() != 2 || rhsShape[0] != expectedRhsRows || rhsShape[1] != expectedRhsColumns ||
				    lhsInfo.shape[0] != outputInfos[0].shape[0])
				{
					throw std::runtime_error("GraphToMLIR QuantizedMatMulNode shape metadata is inconsistent");
				}

				const auto loc = builder_.getUnknownLoc();
				const auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
				if (!isa<FloatType>(resultType.getElementType()))
				{
					throw std::runtime_error("GraphToMLIR QuantizedMatMulNode output must be floating-point");
				}
				auto lhs = getVal(valueMap, node.lhs);
				auto rhs = getVal(valueMap, node.rhsStorage);
				const auto k = lhsInfo.shape[1];
				const auto n = outputInfos[0].shape[1];
				if (isAffineQuantizedMatMul &&
				    (rhsInfo.shape.size() != 2 || rhsInfo.shape[0] != k || rhsInfo.shape[1] != n))
				{
					throw std::runtime_error(
					    "GraphToMLIR QuantizedMatMulNode affine storage shape must match expressed rhs shape");
				}
				if (isPackedNibbleQuantizedMatMul &&
				    (rhsInfo.dtype != DataType::UInt8 || rhsInfo.shape.size() != 1 ||
				     rhsInfo.shape[0] != QuantizationDetail::CeilDiv(k * n, std::size_t{ 2 })))
				{
					throw std::runtime_error(
					    "GraphToMLIR QuantizedMatMulNode packed-nibble storage shape must match expressed rhs shape");
				}
				if (isGGMLBlockQuantizedMatMul)
				{
					const auto layout = GetQuantizedBlockLayout(node.params.blockFormat);
					if (!node.transposeRhs || rhsInfo.dtype != DataType::UInt8 || rhsInfo.shape.size() != 1 ||
					    !layout || k % layout->elementsPerBlock != 0 ||
					    rhsInfo.shape[0] != n * (k / layout->elementsPerBlock) * layout->bytesPerBlock)
					{
						throw std::runtime_error(
						    "GraphToMLIR GGML QuantizedMatMulNode requires output-major UInt8 block storage");
					}

					auto empty =
					    builder_.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
					auto zero = emitScalarZero(builder_, loc, resultType.getElementType());
					auto filled =
					    builder_.create<linalg::FillOp>(loc, ValueRange{ zero }, ValueRange{ empty }).getResult(0);
					auto lhsMap =
					    AffineMap::get(3, 0, { getAffineDimExpr(0, &ctx_), getAffineDimExpr(2, &ctx_) }, &ctx_);
					auto outputMap =
					    AffineMap::get(3, 0, { getAffineDimExpr(0, &ctx_), getAffineDimExpr(1, &ctx_) }, &ctx_);
					auto generic = builder_.create<linalg::GenericOp>(
					    loc, TypeRange{ resultType }, ValueRange{ lhs }, ValueRange{ filled },
					    SmallVector<AffineMap>{ lhsMap, outputMap },
					    SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel,
					                                      utils::IteratorType::reduction },
					    [&](OpBuilder& b, Location l, ValueRange args) {
						    const auto i16 = b.getI16Type();
						    const auto i32 = b.getI32Type();
						    auto index = [&](std::int64_t value) {
							    return b.create<arith::ConstantIndexOp>(l, value).getResult();
						    };
						    auto byteAt = [&](Value byteIndex) {
							    return b.create<tensor::ExtractOp>(l, rhs, ValueRange{ byteIndex }).getResult();
						    };
						    auto byteToI32 = [&](Value byte) {
							    return b.create<arith::ExtUIOp>(l, i32, byte).getResult();
						    };
						    auto fp16At = [&](Value byteIndex) {
							    auto low = b.create<arith::ExtUIOp>(l, i16, byteAt(byteIndex)).getResult();
							    auto highIndex = b.create<arith::AddIOp>(l, byteIndex, index(1)).getResult();
							    auto high = b.create<arith::ExtUIOp>(l, i16, byteAt(highIndex)).getResult();
							    auto bits =
							        b.create<arith::OrIOp>(
							             l, low,
							             b.create<arith::ShLIOp>(l, high, b.create<arith::ConstantIntOp>(l, i16, 8))
							                 .getResult())
							            .getResult();
							    auto half = b.create<arith::BitcastOp>(l, b.getF16Type(), bits).getResult();
							    return b.create<arith::ExtFOp>(l, b.getF32Type(), half).getResult();
						    };

						    auto reductionIndex = b.create<linalg::IndexOp>(l, 2).getResult();
						    auto outputIndex = b.create<linalg::IndexOp>(l, 1).getResult();
						    auto blockIndex =
						        b.create<arith::DivUIOp>(l, reductionIndex, index(layout->elementsPerBlock))
						            .getResult();
						    auto rowBlockIndex =
						        b.create<arith::AddIOp>(
						             l,
						             b.create<arith::MulIOp>(l, outputIndex, index(k / layout->elementsPerBlock))
						                 .getResult(),
						             blockIndex)
						            .getResult();
						    auto blockBase =
						        b.create<arith::MulIOp>(l, rowBlockIndex, index(layout->bytesPerBlock)).getResult();
						    auto withinBlock =
						        b.create<arith::RemUIOp>(l, reductionIndex, index(layout->elementsPerBlock))
						            .getResult();
						    auto d = fp16At(node.params.blockFormat == QuantizedBlockFormat::GGML_Q6_K
						                        ? b.create<arith::AddIOp>(l, blockBase, index(208)).getResult()
						                        : blockBase);
						    Value weightF32;
						    if (node.params.blockFormat == QuantizedBlockFormat::GGML_Q8_0)
						    {
							    auto quantIndex =
							        b.create<arith::AddIOp>(
							             l, b.create<arith::AddIOp>(l, blockBase, index(2)).getResult(), withinBlock)
							            .getResult();
							    auto quantI32 = b.create<arith::ExtSIOp>(l, i32, byteAt(quantIndex)).getResult();
							    auto quantF32 = b.create<arith::SIToFPOp>(l, b.getF32Type(), quantI32).getResult();
							    weightF32 = b.create<arith::MulFOp>(l, d, quantF32).getResult();
						    }
						    else if (node.params.blockFormat == QuantizedBlockFormat::GGML_Q6_K)
						    {
							    auto halfBlock = b.create<arith::DivUIOp>(l, withinBlock, index(128)).getResult();
							    auto local = b.create<arith::RemUIOp>(l, withinBlock, index(128)).getResult();
							    auto segment = b.create<arith::DivUIOp>(l, local, index(32)).getResult();
							    auto lane = b.create<arith::RemUIOp>(l, local, index(32)).getResult();
							    auto oddSegment = b.create<arith::RemUIOp>(l, segment, index(2)).getResult();
							    auto qlOffset =
							        b.create<arith::AddIOp>(
							             l, b.create<arith::MulIOp>(l, halfBlock, index(64)).getResult(),
							             b.create<arith::AddIOp>(
							                  l, lane, b.create<arith::MulIOp>(l, oddSegment, index(32)).getResult())
							                 .getResult())
							            .getResult();
							    auto ql =
							        byteToI32(byteAt(b.create<arith::AddIOp>(l, blockBase, qlOffset).getResult()));
							    auto highNibble =
							        b.create<arith::CmpIOp>(l, arith::CmpIPredicate::uge, segment, index(2))
							            .getResult();
							    auto shift4 = b.create<arith::ConstantIntOp>(l, i32, 4);
							    auto mask15 = b.create<arith::ConstantIntOp>(l, i32, 15);
							    auto lowFour = b.create<arith::SelectOp>(
							                        l, highNibble, b.create<arith::ShRUIOp>(l, ql, shift4).getResult(),
							                        b.create<arith::AndIOp>(l, ql, mask15).getResult())
							                       .getResult();
							    auto qhOffset =
							        b.create<arith::AddIOp>(
							             l, b.create<arith::MulIOp>(l, halfBlock, index(32)).getResult(), lane)
							            .getResult();
							    auto qh = byteToI32(byteAt(
							        b.create<arith::AddIOp>(
							             l, b.create<arith::AddIOp>(l, blockBase, index(128)).getResult(), qhOffset)
							            .getResult()));
							    auto segmentI32 = b.create<arith::IndexCastOp>(l, i32, segment).getResult();
							    auto shift =
							        b.create<arith::MulIOp>(l, segmentI32, b.create<arith::ConstantIntOp>(l, i32, 2))
							            .getResult();
							    auto highTwo = b.create<arith::AndIOp>(l, b.create<arith::ShRUIOp>(l, qh, shift),
							                                           b.create<arith::ConstantIntOp>(l, i32, 3))
							                       .getResult();
							    auto quant =
							        b.create<arith::SubIOp>(
							             l,
							             b.create<arith::OrIOp>(l, lowFour, b.create<arith::ShLIOp>(l, highTwo, shift4))
							                 .getResult(),
							             b.create<arith::ConstantIntOp>(l, i32, 32))
							            .getResult();
							    auto scaleOffset =
							        b.create<arith::AddIOp>(
							             l, b.create<arith::MulIOp>(l, halfBlock, index(8)).getResult(),
							             b.create<arith::AddIOp>(
							                  l, b.create<arith::DivUIOp>(l, lane, index(16)).getResult(),
							                  b.create<arith::MulIOp>(l, segment, index(2)).getResult())
							                 .getResult())
							            .getResult();
							    auto scale =
							        b.create<arith::ExtSIOp>(
							             l, i32,
							             byteAt(b.create<arith::AddIOp>(
							                         l, b.create<arith::AddIOp>(l, blockBase, index(192)).getResult(),
							                         scaleOffset)
							                        .getResult()))
							            .getResult();
							    auto quantF32 = b.create<arith::SIToFPOp>(l, b.getF32Type(), quant).getResult();
							    auto scaleF32 = b.create<arith::SIToFPOp>(l, b.getF32Type(), scale).getResult();
							    weightF32 =
							        b.create<arith::MulFOp>(l, b.create<arith::MulFOp>(l, d, scaleF32), quantF32)
							            .getResult();
						    }
						    else
						    {
							    auto dmin = fp16At(b.create<arith::AddIOp>(l, blockBase, index(2)).getResult());
							    auto subblock = b.create<arith::DivUIOp>(l, withinBlock, index(32)).getResult();
							    auto belowFour =
							        b.create<arith::CmpIOp>(l, arith::CmpIPredicate::ult, subblock, index(4))
							            .getResult();
							    auto scalesBase = b.create<arith::AddIOp>(l, blockBase, index(4)).getResult();
							    auto scaleLowOffset = b.create<arith::SelectOp>(
							                               l, belowFour, subblock,
							                               b.create<arith::AddIOp>(l, subblock, index(4)).getResult())
							                              .getResult();
							    auto scaleLow = byteToI32(
							        byteAt(b.create<arith::AddIOp>(l, scalesBase, scaleLowOffset).getResult()));
							    auto minLow = byteToI32(byteAt(
							        b.create<arith::AddIOp>(l, scalesBase,
							                                b.create<arith::AddIOp>(l, subblock, index(4)).getResult())
							            .getResult()));
							    auto mask63 = b.create<arith::ConstantIntOp>(l, i32, 63);
							    auto scaleDirect = b.create<arith::AndIOp>(l, scaleLow, mask63).getResult();
							    auto minDirect = b.create<arith::AndIOp>(l, minLow, mask63).getResult();
							    auto highOffset = b.create<arith::SelectOp>(
							                           l, belowFour, index(0),
							                           b.create<arith::SubIOp>(l, subblock, index(4)).getResult())
							                          .getResult();
							    auto highSource =
							        byteToI32(byteAt(b.create<arith::AddIOp>(l, scalesBase, highOffset).getResult()));
							    auto minHighSource =
							        byteToI32(byteAt(b.create<arith::AddIOp>(l, scalesBase, subblock).getResult()));
							    auto shift4 = b.create<arith::ConstantIntOp>(l, i32, 4);
							    auto shift6 = b.create<arith::ConstantIntOp>(l, i32, 6);
							    auto mask15 = b.create<arith::ConstantIntOp>(l, i32, 15);
							    auto scaleExtended =
							        b.create<arith::OrIOp>(
							             l, b.create<arith::AndIOp>(l, scaleLow, mask15),
							             b.create<arith::ShLIOp>(l, b.create<arith::ShRUIOp>(l, highSource, shift6),
							                                     shift4))
							            .getResult();
							    auto minExtended =
							        b.create<arith::OrIOp>(
							             l, b.create<arith::ShRUIOp>(l, minLow, shift4),
							             b.create<arith::ShLIOp>(l, b.create<arith::ShRUIOp>(l, minHighSource, shift6),
							                                     shift4))
							            .getResult();
							    auto scale =
							        b.create<arith::SelectOp>(l, belowFour, scaleDirect, scaleExtended).getResult();
							    auto minimum =
							        b.create<arith::SelectOp>(l, belowFour, minDirect, minExtended).getResult();
							    auto quantOffset =
							        b.create<arith::AddIOp>(
							             l,
							             b.create<arith::MulIOp>(l, b.create<arith::DivUIOp>(l, withinBlock, index(64)),
							                                     index(32))
							                 .getResult(),
							             b.create<arith::RemUIOp>(l, withinBlock, index(32)).getResult())
							            .getResult();
							    const auto quantBaseOffset =
							        node.params.blockFormat == QuantizedBlockFormat::GGML_Q5_K ? 48 : 16;
							    auto quantByte = byteToI32(byteAt(
							        b.create<arith::AddIOp>(
							             l, b.create<arith::AddIOp>(l, blockBase, index(quantBaseOffset)).getResult(),
							             quantOffset)
							            .getResult()));
							    auto highNibble =
							        b.create<arith::CmpIOp>(
							             l, arith::CmpIPredicate::uge,
							             b.create<arith::RemUIOp>(l, withinBlock, index(64)).getResult(), index(32))
							            .getResult();
							    auto nibble =
							        b.create<arith::SelectOp>(
							             l, highNibble, b.create<arith::ShRUIOp>(l, quantByte, shift4).getResult(),
							             b.create<arith::AndIOp>(l, quantByte, mask15).getResult())
							            .getResult();
							    if (node.params.blockFormat == QuantizedBlockFormat::GGML_Q5_K)
							    {
								    auto highBits = byteToI32(
								        byteAt(b.create<arith::AddIOp>(
								                    l, b.create<arith::AddIOp>(l, blockBase, index(16)).getResult(),
								                    b.create<arith::RemUIOp>(l, withinBlock, index(32)).getResult())
								                   .getResult()));
								    auto subblockI32 = b.create<arith::IndexCastOp>(l, i32, subblock).getResult();
								    auto highBit =
								        b.create<arith::AndIOp>(l, b.create<arith::ShRUIOp>(l, highBits, subblockI32),
								                                b.create<arith::ConstantIntOp>(l, i32, 1))
								            .getResult();
								    nibble =
								        b.create<arith::OrIOp>(l, nibble, b.create<arith::ShLIOp>(l, highBit, shift4))
								            .getResult();
							    }
							    auto scaleF32 = b.create<arith::UIToFPOp>(l, b.getF32Type(), scale).getResult();
							    auto minF32 = b.create<arith::UIToFPOp>(l, b.getF32Type(), minimum).getResult();
							    auto quantF32 = b.create<arith::UIToFPOp>(l, b.getF32Type(), nibble).getResult();
							    auto scaled =
							        b.create<arith::MulFOp>(l, b.create<arith::MulFOp>(l, d, scaleF32), quantF32)
							            .getResult();
							    weightF32 = b.create<arith::SubFOp>(
							                     l, scaled, b.create<arith::MulFOp>(l, dmin, minF32).getResult())
							                    .getResult();
						    }
						    auto product =
						        b.create<arith::MulFOp>(l, emitScalarToF32(b, l, args[0]), weightF32).getResult();
						    auto sum = b.create<arith::AddFOp>(l, emitScalarToF32(b, l, args[1]), product).getResult();
						    b.create<linalg::YieldOp>(l, emitScalarFromF32(b, l, sum, resultType.getElementType()));
					    });
					valueMap[nodeId] = { generic.getResult(0) };
					return;
				}

				if (isPackedNibbleQuantizedMatMul)
				{
					const auto scale = node.params.scales[0];
					const auto zeroPoint = node.params.zeroPoints.empty() ? 0 : node.params.zeroPoints[0];
					const auto lowThenHigh = node.params.packedOrder == PackedNibbleOrder::LowThenHigh;
					const auto signedInt4 = node.params.packedFormat == PackedNibbleFormat::Int4;
					auto empty =
					    builder_.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
					auto zero = emitScalarZero(builder_, loc, resultType.getElementType());
					auto filled =
					    builder_.create<linalg::FillOp>(loc, ValueRange{ zero }, ValueRange{ empty }).getResult(0);
					auto lhsMap =
					    AffineMap::get(3, 0, { getAffineDimExpr(0, &ctx_), getAffineDimExpr(2, &ctx_) }, &ctx_);
					auto outputMap =
					    AffineMap::get(3, 0, { getAffineDimExpr(0, &ctx_), getAffineDimExpr(1, &ctx_) }, &ctx_);
					auto generic = builder_.create<linalg::GenericOp>(
					    loc, TypeRange{ resultType }, ValueRange{ lhs }, ValueRange{ filled },
					    SmallVector<AffineMap>{ lhsMap, outputMap },
					    SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel,
					                                      utils::IteratorType::reduction },
					    [&](OpBuilder& b, Location l, ValueRange args) {
						    const auto i32 = b.getI32Type();
						    const auto lhsF32 = emitScalarToF32(b, l, args[0]);
						    const auto rowIndex = b.create<linalg::IndexOp>(l, 2).getResult();
						    const auto colIndex = b.create<linalg::IndexOp>(l, 1).getResult();
						    auto nIndex = b.create<arith::ConstantIndexOp>(l, n).getResult();
						    auto elementIndex =
						        b.create<arith::AddIOp>(l, b.create<arith::MulIOp>(l, rowIndex, nIndex).getResult(),
						                                colIndex)
						            .getResult();
						    auto twoIndex = b.create<arith::ConstantIndexOp>(l, 2).getResult();
						    auto byteIndex = b.create<arith::DivUIOp>(l, elementIndex, twoIndex).getResult();
						    auto nibbleOffset = b.create<arith::RemUIOp>(l, elementIndex, twoIndex).getResult();
						    auto byte = b.create<tensor::ExtractOp>(l, rhs, ValueRange{ byteIndex }).getResult();
						    auto byteI32 = b.create<arith::ExtUIOp>(l, i32, byte).getResult();
						    auto low = b.create<arith::AndIOp>(l, byteI32, b.create<arith::ConstantIntOp>(l, i32, 15))
						                   .getResult();
						    auto high =
						        b.create<arith::AndIOp>(
						             l,
						             b.create<arith::ShRUIOp>(l, byteI32, b.create<arith::ConstantIntOp>(l, i32, 4))
						                 .getResult(),
						             b.create<arith::ConstantIntOp>(l, i32, 15))
						            .getResult();
						    auto oneIndex = b.create<arith::ConstantIndexOp>(l, 1).getResult();
						    auto takeSecond =
						        b.create<arith::CmpIOp>(l, arith::CmpIPredicate::eq, nibbleOffset, oneIndex)
						            .getResult();
						    Value nibble = lowThenHigh
						                       ? b.create<arith::SelectOp>(l, takeSecond, high, low).getResult()
						                       : b.create<arith::SelectOp>(l, takeSecond, low, high).getResult();
						    if (signedInt4)
						    {
							    auto isNegative = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::sge, nibble,
							                                              b.create<arith::ConstantIntOp>(l, i32, 8))
							                          .getResult();
							    nibble =
							        b.create<arith::SelectOp>(
							             l, isNegative,
							             b.create<arith::SubIOp>(l, nibble, b.create<arith::ConstantIntOp>(l, i32, 16))
							                 .getResult(),
							             nibble)
							            .getResult();
						    }
						    auto rhsF32 = emitIntegerScalarToF32(b, l, nibble, false);
						    rhsF32 = b.create<arith::SubFOp>(
						                  l, rhsF32,
						                  b.create<arith::ConstantFloatOp>(
						                      l, b.getF32Type(), llvm::APFloat(static_cast<float>(zeroPoint))))
						                 .getResult();
						    rhsF32 = b.create<arith::MulFOp>(
						                  l, rhsF32,
						                  b.create<arith::ConstantFloatOp>(l, b.getF32Type(),
						                                                   llvm::APFloat(static_cast<float>(scale))))
						                 .getResult();
						    auto product = b.create<arith::MulFOp>(l, lhsF32, rhsF32).getResult();
						    auto accumulator = emitScalarToF32(b, l, args[1]);
						    auto sum = b.create<arith::AddFOp>(l, accumulator, product).getResult();
						    b.create<linalg::YieldOp>(l, emitScalarFromF32(b, l, sum, resultType.getElementType()));
					    });
					valueMap[nodeId] = { generic.getResult(0) };
					return;
				}

				std::vector<std::size_t> parameterShape{ 1 };
				AffineExpr parameterExpr = getAffineConstantExpr(0, &ctx_);
				bool groupedParameters = false;
				std::size_t groupedAxis = 0;
				std::size_t groupedGroupsPerLine = 0;
				std::size_t groupedGroupSize = 0;
				if (node.params.granularity == QuantizationGranularity::PerAxis ||
				    node.params.granularity == QuantizationGranularity::Grouped)
				{
					const auto axis = QuantizationDetail::NormalizeAxis(node.params.axis, ShapeView{ rhsShape });
					if (axis > 1)
					{
						throw std::runtime_error("GraphToMLIR QuantizedMatMulNode expected rank-2 quantization axis");
					}
					if (node.params.granularity == QuantizationGranularity::PerAxis)
					{
						parameterShape[0] = rhsShape[axis];
						parameterExpr = axis == 0 ? getAffineDimExpr(2, &ctx_) : getAffineDimExpr(1, &ctx_);
					}
					else
					{
						if (node.params.groupSize == 0)
						{
							throw std::runtime_error("GraphToMLIR QuantizedMatMulNode grouped quantization requires "
							                         "groupSize > 0");
						}
						const auto groupsPerLine = QuantizationDetail::CeilDiv(rhsShape[axis], node.params.groupSize);
						parameterShape[0] = (ShapeView{ rhsShape }.NumElements() / rhsShape[axis]) * groupsPerLine;
						groupedParameters = true;
						groupedAxis = axis;
						groupedGroupsPerLine = groupsPerLine;
						groupedGroupSize = node.params.groupSize;
					}
				}
				if (node.params.scales.size() != parameterShape[0] ||
				    (!node.params.zeroPoints.empty() && node.params.zeroPoints.size() != parameterShape[0]))
				{
					throw std::runtime_error("GraphToMLIR QuantizedMatMulNode quantization parameter count mismatch");
				}

				std::vector<double> scales;
				scales.reserve(node.params.scales.size());
				for (const auto scale : node.params.scales)
				{
					scales.push_back(static_cast<double>(scale));
				}
				std::vector<double> zeroPoints(parameterShape[0], 0.0);
				for (std::size_t i = 0; i < node.params.zeroPoints.size(); ++i)
				{
					zeroPoints[i] = static_cast<double>(node.params.zeroPoints[i]);
				}
				auto scaleValue = emitDenseConstant(DataType::Float32, parameterShape, scales);
				auto zeroPointValue = emitDenseConstant(DataType::Float32, parameterShape, zeroPoints);

				auto empty = builder_.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
				auto zero = emitScalarZero(builder_, loc, resultType.getElementType());
				auto filled =
				    builder_.create<linalg::FillOp>(loc, ValueRange{ zero }, ValueRange{ empty }).getResult(0);

				auto lhsMap = AffineMap::get(3, 0, { getAffineDimExpr(0, &ctx_), getAffineDimExpr(2, &ctx_) }, &ctx_);
				auto rhsMap = AffineMap::get(3, 0, { getAffineDimExpr(2, &ctx_), getAffineDimExpr(1, &ctx_) }, &ctx_);
				auto parameterMap = AffineMap::get(3, 0, { parameterExpr }, &ctx_);
				auto outputMap =
				    AffineMap::get(3, 0, { getAffineDimExpr(0, &ctx_), getAffineDimExpr(1, &ctx_) }, &ctx_);
				auto generic = builder_.create<linalg::GenericOp>(
				    loc, TypeRange{ resultType }, ValueRange{ lhs, rhs, scaleValue, zeroPointValue },
				    ValueRange{ filled },
				    SmallVector<AffineMap>{ lhsMap, rhsMap, parameterMap, parameterMap, outputMap },
				    SmallVector<utils::IteratorType>{ utils::IteratorType::parallel, utils::IteratorType::parallel,
				                                      utils::IteratorType::reduction },
				    [&](OpBuilder& b, Location l, ValueRange args) {
					    const auto lhsF32 = emitScalarToF32(b, l, args[0]);
					    auto rhsF32 = emitIntegerScalarToF32(b, l, args[1], node.params.storageType == DataType::UInt8);
					    Value scale = args[2];
					    Value zeroPoint = args[3];
					    if (groupedParameters)
					    {
						    const auto rowIndex = b.create<linalg::IndexOp>(l, 2).getResult();
						    const auto colIndex = b.create<linalg::IndexOp>(l, 1).getResult();
						    const auto axisCoord = groupedAxis == 0 ? rowIndex : colIndex;
						    const auto line = groupedAxis == 0 ? colIndex : rowIndex;
						    auto groupsPerLine = b.create<arith::ConstantIndexOp>(l, groupedGroupsPerLine).getResult();
						    auto groupSize = b.create<arith::ConstantIndexOp>(l, groupedGroupSize).getResult();
						    auto lineOffset = b.create<arith::MulIOp>(l, line, groupsPerLine).getResult();
						    auto groupIndex = b.create<arith::DivUIOp>(l, axisCoord, groupSize).getResult();
						    auto parameterIndex = b.create<arith::AddIOp>(l, lineOffset, groupIndex).getResult();
						    scale =
						        b.create<tensor::ExtractOp>(l, scaleValue, ValueRange{ parameterIndex }).getResult();
						    zeroPoint = b.create<tensor::ExtractOp>(l, zeroPointValue, ValueRange{ parameterIndex })
						                    .getResult();
					    }
					    rhsF32 = b.create<arith::SubFOp>(l, rhsF32, zeroPoint).getResult();
					    rhsF32 = b.create<arith::MulFOp>(l, rhsF32, scale).getResult();
					    auto product = b.create<arith::MulFOp>(l, lhsF32, rhsF32).getResult();
					    auto accumulator = emitScalarToF32(b, l, args[4]);
					    auto sum = b.create<arith::AddFOp>(l, accumulator, product).getResult();
					    b.create<linalg::YieldOp>(l, emitScalarFromF32(b, l, sum, resultType.getElementType()));
				    });
				valueMap[nodeId] = { generic.getResult(0) };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const CallNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto resultTypes = convertOutputInfos(outputInfos);
				SmallVector<Value> args;
				for (const auto& arg : node.args)
				{
					args.push_back(getVal(valueMap, arg));
				}

				auto calleeName = "subgraph_" + std::to_string(node.callee);
				auto op = builder_.create<CallOp>(builder_.getUnknownLoc(), resultTypes,
				                                  FlatSymbolRefAttr::get(&ctx_, calleeName), args,
				                                  /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
				valueMap[nodeId].clear();
				for (auto result : op.getResults())
				{
					valueMap[nodeId].push_back(result);
				}
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const CondNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>& activationMap, std::map<std::size_t, Value>& tapeMap)
			{
				auto resultTypes = convertOutputInfos(outputInfos);
				auto condition = getVal(valueMap, node.condition);
				SmallVector<Value> args;
				for (const auto& arg : node.args)
				{
					args.push_back(getVal(valueMap, arg));
				}

				auto op = builder_.create<CondOp>(builder_.getUnknownLoc(), resultTypes, condition, args);

				// Emit then region
				emitSubgraphIntoRegion(node.thenBranch, op.getThenRegion(), activationMap, tapeMap);

				// Emit else region
				emitSubgraphIntoRegion(node.elseBranch, op.getElseRegion(), activationMap, tapeMap);

				valueMap[nodeId].clear();
				for (auto result : op.getResults())
				{
					valueMap[nodeId].push_back(result);
				}
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const WhileNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>& activationMap, std::map<std::size_t, Value>& tapeMap)
			{
				auto resultTypes = convertOutputInfos(outputInfos);
				SmallVector<Value> initArgs;
				for (const auto& arg : node.initArgs)
				{
					initArgs.push_back(getVal(valueMap, arg));
				}

				auto op = builder_.create<WhileOp>(builder_.getUnknownLoc(), resultTypes, initArgs);

				// Emit cond region
				emitSubgraphIntoRegion(node.condBranch, op.getCondRegion(), activationMap, tapeMap);

				// Emit body region
				emitSubgraphIntoRegion(node.bodyBranch, op.getBodyRegion(), activationMap, tapeMap);

				valueMap[nodeId].clear();
				for (auto result : op.getResults())
				{
					valueMap[nodeId].push_back(result);
				}
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const SaveActivationNode& node,
			              std::span<const OutputInfo>, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>& activationMap, std::map<std::size_t, Value>&)
			{
				// SSA化: passthrough, record mapping
				auto input = getVal(valueMap, node.input);
				activationMap[node.slotId] = input;
				valueMap[nodeId] = { input };
			}

			void emitNode(const PlanSubgraphView& sg, NodeId nodeId, const LoadActivationNode& node,
			              std::span<const OutputInfo>, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>& activationMap, std::map<std::size_t, Value>&)
			{
				// SSA化: lookup saved value
				const auto it = activationMap.find(node.slotId);
				if (it == activationMap.end())
				{
					throw std::runtime_error(std::format(
					    "GraphToMLIR cannot lower LoadActivationNode for slot {} in subgraph {} node {} without an "
					    "explicit saved-activation binding",
					    node.slotId, sg.subgraph.sourceSubgraph, nodeId));
				}
				valueMap[nodeId] = { it->second };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const TapeSaveActivationNode& node,
			              std::span<const OutputInfo>, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>& tapeMap)
			{
				auto input = getVal(valueMap, node.input);
				tapeMap[node.tapeSlotId] = input;
				valueMap[nodeId] = { input };
			}

			void emitNode(const PlanSubgraphView& sg, NodeId nodeId, const TapeLoadActivationNode& node,
			              std::span<const OutputInfo>, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>& tapeMap)
			{
				const auto it = tapeMap.find(node.tapeSlotId);
				if (it == tapeMap.end())
				{
					throw std::runtime_error(std::format(
					    "GraphToMLIR cannot lower TapeLoadActivationNode for slot {} in subgraph {} node {} without "
					    "an explicit tape binding",
					    node.tapeSlotId, sg.subgraph.sourceSubgraph, nodeId));
				}
				valueMap[nodeId] = { it->second };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const ReduceOpNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
				auto input = getVal(valueMap, node.input);
				auto op =
				    builder_.create<litenn::ReduceOp>(builder_.getUnknownLoc(), resultType, convertReduceOp(node.op),
				                                      input, static_cast<uint64_t>(node.axis));
				valueMap[nodeId] = { op.getResult() };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const ReshapeNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
				auto input = getVal(valueMap, node.input);
				auto op = builder_.create<ReshapeOp>(builder_.getUnknownLoc(), resultType, input);
				valueMap[nodeId] = { op.getResult() };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const ConcatNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
				SmallVector<Value> inputs;
				for (const auto& inp : node.inputs)
				{
					inputs.push_back(getVal(valueMap, inp));
				}

				auto op = builder_.create<ConcatOp>(builder_.getUnknownLoc(), resultType, inputs,
				                                    static_cast<uint64_t>(node.axis));
				valueMap[nodeId] = { op.getResult() };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const SliceNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
				auto input = getVal(valueMap, node.input);
				auto op = builder_.create<SliceOp>(builder_.getUnknownLoc(), resultType, input,
				                                   static_cast<uint64_t>(node.axis), static_cast<uint64_t>(node.start),
				                                   static_cast<uint64_t>(node.length));
				valueMap[nodeId] = { op.getResult() };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const GetRowsNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto loc = builder_.getUnknownLoc();
				auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
				auto data = getVal(valueMap, node.data);
				auto indices = getVal(valueMap, node.indices);
				auto indicesType = cast<RankedTensorType>(indices.getType());
				auto empty = builder_.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());

				SmallVector<AffineExpr> indexExprs;
				for (int64_t dim = 0; dim < indicesType.getRank(); ++dim)
				{
					indexExprs.push_back(getAffineDimExpr(dim, &ctx_));
				}
				auto indexMap = AffineMap::get(resultType.getRank(), 0, indexExprs, &ctx_);
				auto outputMap = AffineMap::getMultiDimIdentityMap(resultType.getRank(), &ctx_);
				SmallVector<AffineMap> maps = { indexMap, outputMap };
				SmallVector<utils::IteratorType> iterTypes(resultType.getRank(), utils::IteratorType::parallel);
				const auto indexRank = indicesType.getRank();
				const auto resultRank = resultType.getRank();

				auto generic = builder_.create<linalg::GenericOp>(
				    loc, TypeRange{ resultType }, ValueRange{ indices }, ValueRange{ empty }, maps, iterTypes,
				    [&](OpBuilder& b, Location l, ValueRange args) {
					    Value rowIndex = args[0];
					    if (!isa<IndexType>(rowIndex.getType()))
					    {
						    rowIndex = b.create<arith::IndexCastOp>(l, b.getIndexType(), rowIndex);
					    }

					    SmallVector<Value> dataCoords{ rowIndex };
					    for (int64_t dim = indexRank; dim < resultRank; ++dim)
					    {
						    dataCoords.push_back(b.create<linalg::IndexOp>(l, dim).getResult());
					    }

					    auto element = b.create<tensor::ExtractOp>(l, data, dataCoords).getResult();
					    b.create<linalg::YieldOp>(l, element);
				    });
				valueMap[nodeId] = { generic.getResult(0) };
			}

			void emitNode(const PlanSubgraphView&, NodeId, const ArgsortNode&, std::span<const OutputInfo>,
			              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
			              std::map<std::size_t, Value>&)
			{
				throw std::runtime_error("GraphToMLIR does not support ArgsortNode yet; use the interpreter path");
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const PermuteNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto loc = builder_.getUnknownLoc();
				auto input = getVal(valueMap, node.input);
				auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
				const auto rank = resultType.getRank();

				SmallVector<std::size_t> inverse(node.permutation.size());
				for (auto outputAxis = 0uz; outputAxis < node.permutation.size(); ++outputAxis)
				{
					inverse[node.permutation[outputAxis]] = outputAxis;
				}

				SmallVector<AffineExpr> inputExprs;
				inputExprs.reserve(rank);
				for (int64_t inputAxis = 0; inputAxis < rank; ++inputAxis)
				{
					inputExprs.push_back(getAffineDimExpr(inverse[static_cast<std::size_t>(inputAxis)], &ctx_));
				}

				auto inputMap = AffineMap::get(rank, 0, inputExprs, &ctx_);
				auto outputMap = AffineMap::getMultiDimIdentityMap(rank, &ctx_);
				auto empty = builder_.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
				SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);

				auto generic = builder_.create<linalg::GenericOp>(
				    loc, TypeRange{ resultType }, ValueRange{ input }, ValueRange{ empty },
				    SmallVector<AffineMap>{ inputMap, outputMap }, iterTypes,
				    [](OpBuilder& b, Location l, ValueRange args) { b.create<linalg::YieldOp>(l, args[0]); });
				valueMap[nodeId] = { generic.getResult(0) };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const BroadcastToNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto input = getVal(valueMap, node.input);
				valueMap[nodeId] = { emitBroadcastToValue(input, outputInfos[0].dtype, outputInfos[0].shape) };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const PadNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto loc = builder_.getUnknownLoc();
				auto input = getVal(valueMap, node.input);
				auto inputType = cast<RankedTensorType>(input.getType());
				auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
				const auto rank = resultType.getRank();
				if (node.mode == PadMode::Reflect)
				{
					for (int64_t dim = 0; dim < inputType.getRank(); ++dim)
					{
						if (inputType.getDimSize(dim) < 2)
						{
							throw std::runtime_error(
							    "GraphToMLIR PadNode reflect mode requires padded input dims >= 2");
						}
					}
				}

				auto empty = builder_.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
				auto outputMap = AffineMap::getMultiDimIdentityMap(rank, &ctx_);
				SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);

				auto generic = builder_.create<linalg::GenericOp>(
				    loc, TypeRange{ resultType }, ValueRange{}, ValueRange{ empty },
				    SmallVector<AffineMap>{ outputMap }, iterTypes, [&](OpBuilder& b, Location l, ValueRange) {
					    auto trueValue = b.create<arith::ConstantIntOp>(l, b.getI1Type(), static_cast<std::int64_t>(1));
					    Value inBounds = trueValue;
					    SmallVector<Value> inputCoords;
					    inputCoords.reserve(rank);
					    for (int64_t dim = 0; dim < rank; ++dim)
					    {
						    auto source =
						        b.create<arith::SubIOp>(
						             l, emitIndexAsI64(b, l, dim),
						             emitI64Constant(
						                 b, l, static_cast<std::int64_t>(node.lowPads[static_cast<std::size_t>(dim)])))
						            .getResult();
						    auto zero = emitI64Constant(b, l, 0);
						    auto inputDim = emitI64Constant(b, l, inputType.getDimSize(dim));
						    auto notBefore =
						        b.create<arith::CmpIOp>(l, arith::CmpIPredicate::sge, source, zero).getResult();
						    auto notAfter =
						        b.create<arith::CmpIOp>(l, arith::CmpIPredicate::slt, source, inputDim).getResult();
						    inBounds = b.create<arith::AndIOp>(l, inBounds, notBefore).getResult();
						    inBounds = b.create<arith::AndIOp>(l, inBounds, notAfter).getResult();

						    Value safeSource;
						    switch (node.mode)
						    {
						    case PadMode::Constant:
						    case PadMode::Replicate:
							    safeSource = emitClampedI64(b, l, source, 0, inputType.getDimSize(dim) - 1);
							    break;
						    case PadMode::Reflect:
							    safeSource = emitReflectedI64(b, l, source, inputType.getDimSize(dim));
							    break;
						    }
						    inputCoords.push_back(emitI64ToIndex(b, l, safeSource));
					    }

					    auto inputElement = b.create<tensor::ExtractOp>(l, input, inputCoords).getResult();
					    Value result = inputElement;
					    if (node.mode == PadMode::Constant)
					    {
						    auto padValue = emitScalarConstant(b, l, resultType.getElementType(), node.constantValue);
						    result = b.create<arith::SelectOp>(l, inBounds, inputElement, padValue).getResult();
					    }
					    b.create<linalg::YieldOp>(l, result);
				    });
				valueMap[nodeId] = { generic.getResult(0) };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const GatherNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto loc = builder_.getUnknownLoc();
				auto data = getVal(valueMap, node.data);
				auto indices = getVal(valueMap, node.indices);
				auto indicesType = cast<RankedTensorType>(indices.getType());
				auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
				const auto indexRank = indicesType.getRank();
				const auto resultRank = resultType.getRank();

				SmallVector<AffineExpr> indexExprs;
				indexExprs.reserve(indexRank);
				for (int64_t dim = 0; dim < indexRank; ++dim)
				{
					indexExprs.push_back(getAffineDimExpr(static_cast<int64_t>(node.axis) + dim, &ctx_));
				}
				auto indexMap = AffineMap::get(resultRank, 0, indexExprs, &ctx_);
				auto outputMap = AffineMap::getMultiDimIdentityMap(resultRank, &ctx_);
				auto empty = builder_.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
				SmallVector<utils::IteratorType> iterTypes(resultRank, utils::IteratorType::parallel);

				auto generic = builder_.create<linalg::GenericOp>(
				    loc, TypeRange{ resultType }, ValueRange{ indices }, ValueRange{ empty },
				    SmallVector<AffineMap>{ indexMap, outputMap }, iterTypes,
				    [&](OpBuilder& b, Location l, ValueRange args) {
					    Value gathered = args[0];
					    if (!isa<IndexType>(gathered.getType()))
					    {
						    gathered = b.create<arith::IndexCastOp>(l, b.getIndexType(), gathered).getResult();
					    }

					    SmallVector<Value> dataCoords;
					    dataCoords.reserve(resultRank - indexRank + 1);
					    for (auto dim = 0uz; dim < node.axis; ++dim)
					    {
						    dataCoords.push_back(b.create<linalg::IndexOp>(l, static_cast<int64_t>(dim)).getResult());
					    }
					    dataCoords.push_back(gathered);
					    const auto dataRank = resultRank - indexRank + 1;
					    for (auto dataDim = node.axis + 1; dataDim < static_cast<std::size_t>(dataRank); ++dataDim)
					    {
						    const auto outputDim = dataDim + static_cast<std::size_t>(indexRank) - 1;
						    dataCoords.push_back(
						        b.create<linalg::IndexOp>(l, static_cast<int64_t>(outputDim)).getResult());
					    }

					    auto element = b.create<tensor::ExtractOp>(l, data, dataCoords).getResult();
					    b.create<linalg::YieldOp>(l, element);
				    });
				valueMap[nodeId] = { generic.getResult(0) };
			}

			void emitNode(const PlanSubgraphView& sg, NodeId nodeId, const ScatterNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				const auto indicesInfo = sg.GetOutputInfo(node.indices);
				if (indicesInfo.shape != std::vector<std::size_t>{ 1 })
				{
					throw std::runtime_error("GraphToMLIR currently supports ScatterNode only for one index");
				}
				auto loc = builder_.getUnknownLoc();
				auto data = getVal(valueMap, node.data);
				auto indices = getVal(valueMap, node.indices);
				auto updates = getVal(valueMap, node.updates);
				auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
				const auto rank = resultType.getRank();
				auto identity = AffineMap::getMultiDimIdentityMap(rank, &ctx_);
				auto empty = builder_.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
				SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
				auto generic = builder_.create<linalg::GenericOp>(
				    loc, TypeRange{ resultType }, ValueRange{ data }, ValueRange{ empty },
				    SmallVector<AffineMap>{ identity, identity }, iterTypes,
				    [&](OpBuilder& b, Location l, ValueRange args) {
					    auto zero = b.create<arith::ConstantIndexOp>(l, 0).getResult();
					    auto scatterIndex = b.create<tensor::ExtractOp>(l, indices, ValueRange{ zero }).getResult();
					    if (!isa<IndexType>(scatterIndex.getType()))
					    {
						    scatterIndex = b.create<arith::IndexCastOp>(l, b.getIndexType(), scatterIndex).getResult();
					    }
					    auto axisIndex = b.create<linalg::IndexOp>(l, static_cast<int64_t>(node.axis)).getResult();
					    auto matches = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::eq, axisIndex, scatterIndex);
					    SmallVector<Value> updateCoords;
					    updateCoords.reserve(rank);
					    for (int64_t dim = 0; dim < rank; ++dim)
					    {
						    updateCoords.push_back(static_cast<std::size_t>(dim) == node.axis
						                               ? zero
						                               : b.create<linalg::IndexOp>(l, dim).getResult());
					    }
					    auto update = b.create<tensor::ExtractOp>(l, updates, updateCoords).getResult();
					    Value replacement = update;
					    if (node.mode == ScatterMode::Add)
					    {
						    replacement = isa<FloatType>(update.getType())
						                      ? b.create<arith::AddFOp>(l, args[0], update).getResult()
						                      : b.create<arith::AddIOp>(l, args[0], update).getResult();
					    }
					    b.create<linalg::YieldOp>(
					        l, b.create<arith::SelectOp>(l, matches, replacement, args[0]).getResult());
				    });
				valueMap[nodeId] = { generic.getResult(0) };
			}

			void emitNode(const PlanSubgraphView&, NodeId, const ScanNode&, std::span<const OutputInfo>,
			              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
			              std::map<std::size_t, Value>&)
			{
				throw std::runtime_error("GraphToMLIR does not support ScanNode yet; use the interpreter path");
			}

			void emitNode(const PlanSubgraphView&, NodeId, const SSMScanNode&, std::span<const OutputInfo>,
			              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
			              std::map<std::size_t, Value>&)
			{
				throw std::runtime_error("GraphToMLIR does not support SSMScanNode yet; use the interpreter path");
			}

			void emitNode(const PlanSubgraphView&, NodeId, const RWKVWKVNode&, std::span<const OutputInfo>,
			              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
			              std::map<std::size_t, Value>&)
			{
				throw std::runtime_error("GraphToMLIR does not support RWKVWKVNode yet; use the interpreter path");
			}

			void emitNode(const PlanSubgraphView& sg, NodeId nodeId, const SoftmaxNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				const auto inputInfo = sg.GetOutputInfo(node.input);
				const auto dtype = outputInfos[0].dtype;
				auto input = getVal(valueMap, node.input);
				valueMap[nodeId] = { emitSoftmaxValue(input, dtype, inputInfo.shape, node.axis) };
			}

			void emitNode(const PlanSubgraphView& sg, NodeId nodeId, const CrossEntropyLossNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				const auto logitsInfo = sg.GetOutputInfo(node.logits);
				const auto dtype = outputInfos[0].dtype;
				const auto logitsShape = ShapeView{ logitsInfo.shape };
				const auto axis = logitsShape.NumDim() - 1;
				const auto reducedShape = ReducedShape(logitsShape, axis);
				const auto broadcastShape = BroadcastShapeForAxis(logitsShape, axis);
				const auto classCount = logitsShape[axis];
				const auto rowCount = logitsShape.NumElements() / classCount;

				auto logits = getVal(valueMap, node.logits);
				auto labels = getVal(valueMap, node.labels);
				auto max = emitReduceValue(LiteNN::ReduceOp::Max, logits, dtype, reducedShape, axis);
				auto maxBroadcast = emitReshapeValue(max, dtype, broadcastShape);
				auto shifted =
				    emitBinaryValue(LiteNN::BinaryOp::Subtract, logits, maxBroadcast, dtype, logitsInfo.shape);
				auto exp = emitUnaryValue(LiteNN::UnaryOp::Exp, shifted, dtype, logitsInfo.shape);
				auto sumExp = emitReduceValue(LiteNN::ReduceOp::Sum, exp, dtype, reducedShape, axis);
				auto logSum = emitUnaryValue(LiteNN::UnaryOp::Log, sumExp, dtype, reducedShape);
				auto logSumExp = emitBinaryValue(LiteNN::BinaryOp::Add, logSum, max, dtype, reducedShape);
				auto logSumExpBroadcast = emitReshapeValue(logSumExp, dtype, broadcastShape);
				auto logProb =
				    emitBinaryValue(LiteNN::BinaryOp::Subtract, logits, logSumExpBroadcast, dtype, logitsInfo.shape);
				auto weighted = emitBinaryValue(LiteNN::BinaryOp::Multiply, labels, logProb, dtype, logitsInfo.shape);
				auto negative = emitUnaryValue(LiteNN::UnaryOp::Negate, weighted, dtype, logitsInfo.shape);
				auto total = emitReduceAllToSingleValue(LiteNN::ReduceOp::Sum, negative, dtype, logitsInfo.shape);
				const auto scalarShape = std::vector<std::size_t>{ 1 };
				auto divisor = emitFilledConstant(dtype, scalarShape, static_cast<double>(rowCount));
				valueMap[nodeId] = { emitBinaryValue(LiteNN::BinaryOp::Divide, total, divisor, dtype,
					                                 outputInfos[0].shape) };
			}

			void emitNode(const PlanSubgraphView& sg, NodeId nodeId, const CrossEntropyLossBackwardNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				const auto logitsInfo = sg.GetOutputInfo(node.logits);
				const auto dtype = outputInfos[0].dtype;
				const auto logitsShape = ShapeView{ logitsInfo.shape };
				const auto classCount = logitsShape[logitsShape.NumDim() - 1];
				const auto rowCount = logitsShape.NumElements() / classCount;
				const auto scalarShape = std::vector<std::size_t>{ 1 };

				auto grad = getVal(valueMap, node.grad);
				auto logits = getVal(valueMap, node.logits);
				auto labels = getVal(valueMap, node.labels);
				auto probabilities = emitSoftmaxValue(logits, dtype, logitsInfo.shape, logitsShape.NumDim() - 1);
				auto diff = emitBinaryValue(LiteNN::BinaryOp::Subtract, probabilities, labels, dtype, logitsInfo.shape);
				auto divisor = emitFilledConstant(dtype, scalarShape, static_cast<double>(rowCount));
				auto scale = emitBinaryValue(LiteNN::BinaryOp::Divide, grad, divisor, dtype, scalarShape);
				valueMap[nodeId] = { emitBinaryValue(LiteNN::BinaryOp::Multiply, diff, scale, dtype,
					                                 outputInfos[0].shape) };
			}

			void emitNode(const PlanSubgraphView& sg, NodeId nodeId, const NormalizationNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				if (node.mode == NormalizationMode::GroupNorm)
				{
					throw std::runtime_error(
					    "GraphToMLIR does not support NormalizationNode GroupNorm yet; use the interpreter path");
				}

				const auto inputInfo = sg.GetOutputInfo(node.input);
				const auto dtype = outputInfos[0].dtype;
				const auto computeDType = shouldUseFloat32Accumulator(dtype) ? DataType::Float32 : dtype;
				const auto inputShape = ShapeView{ inputInfo.shape };
				const auto reducedShape = ReducedShape(inputShape, node.axis);
				const auto broadcastShape = BroadcastShapeForAxis(inputShape, node.axis);
				auto input =
				    emitMaybeCastValue(getVal(valueMap, node.input), inputInfo.dtype, computeDType, inputInfo.shape);

				Value centered = input;
				if (node.mode == NormalizationMode::LayerNorm)
				{
					auto mean = emitReduceValue(LiteNN::ReduceOp::Mean, input, computeDType, reducedShape, node.axis);
					auto meanBroadcast = emitReshapeValue(mean, computeDType, broadcastShape);
					centered = emitBinaryValue(LiteNN::BinaryOp::Subtract, input, meanBroadcast, computeDType,
					                           inputInfo.shape);
				}

				auto squared =
				    emitBinaryValue(LiteNN::BinaryOp::Multiply, centered, centered, computeDType, inputInfo.shape);
				auto variance = emitReduceValue(LiteNN::ReduceOp::Mean, squared, computeDType, reducedShape, node.axis);
				auto varianceBroadcast = emitReshapeValue(variance, computeDType, broadcastShape);
				auto epsilon = emitFilledConstant(computeDType, broadcastShape, node.epsilon);
				auto withEpsilon =
				    emitBinaryValue(LiteNN::BinaryOp::Add, varianceBroadcast, epsilon, computeDType, broadcastShape);
				auto denom = emitUnaryValue(LiteNN::UnaryOp::Sqrt, withEpsilon, computeDType, broadcastShape);
				auto normalized =
				    emitBinaryValue(LiteNN::BinaryOp::Divide, centered, denom, computeDType, inputInfo.shape);

				if (node.scale)
				{
					const auto scaleInfo = sg.GetOutputInfo(*node.scale);
					auto scale = emitMaybeCastValue(getVal(valueMap, *node.scale), scaleInfo.dtype, computeDType,
					                                scaleInfo.shape);
					normalized =
					    emitBinaryValue(LiteNN::BinaryOp::Multiply, normalized, scale, computeDType, inputInfo.shape);
				}
				if (node.bias)
				{
					const auto biasInfo = sg.GetOutputInfo(*node.bias);
					auto bias =
					    emitMaybeCastValue(getVal(valueMap, *node.bias), biasInfo.dtype, computeDType, biasInfo.shape);
					normalized =
					    emitBinaryValue(LiteNN::BinaryOp::Add, normalized, bias, computeDType, inputInfo.shape);
				}

				valueMap[nodeId] = { emitMaybeCastValue(normalized, computeDType, dtype, outputInfos[0].shape) };
			}

			void emitNode(const PlanSubgraphView& sg, NodeId nodeId, const BatchMatMulNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto loc = builder_.getUnknownLoc();
				const auto outputDType = outputInfos[0].dtype;
				const auto computeDType = shouldUseFloat32Accumulator(outputDType) ? DataType::Float32 : outputDType;
				const auto lhsInfo = sg.GetOutputInfo(node.lhs);
				const auto rhsInfo = sg.GetOutputInfo(node.rhs);
				auto lhs = emitMaybeCastValue(getVal(valueMap, node.lhs), lhsInfo.dtype, computeDType, lhsInfo.shape);
				auto rhs = emitMaybeCastValue(getVal(valueMap, node.rhs), rhsInfo.dtype, computeDType, rhsInfo.shape);
				auto lhsType = cast<RankedTensorType>(lhs.getType());
				auto rhsType = cast<RankedTensorType>(rhs.getType());
				auto resultType = convertTensorType(ctx_, computeDType, outputInfos[0].shape);
				const auto resultRank = resultType.getRank();
				const auto leadRank = resultRank - 2;
				const auto loopRank = leadRank + 3;
				const auto lhsLeadRank = lhsType.getRank() - 2;
				const auto rhsLeadRank = rhsType.getRank() - 2;

				auto buildInputMap = [&](RankedTensorType inputType, int64_t inputLeadRank, bool isLhs) {
					SmallVector<AffineExpr> exprs;
					exprs.reserve(inputType.getRank());
					for (int64_t dim = 0; dim < inputLeadRank; ++dim)
					{
						const auto outputDim = leadRank - inputLeadRank + dim;
						const bool isBroadcast =
						    inputType.getDimSize(dim) == 1 && resultType.getDimSize(outputDim) != 1;
						exprs.push_back(isBroadcast ? getAffineConstantExpr(0, &ctx_)
						                            : getAffineDimExpr(outputDim, &ctx_));
					}
					exprs.push_back(getAffineDimExpr(isLhs ? leadRank : leadRank + 1, &ctx_));
					exprs.push_back(getAffineDimExpr(isLhs ? leadRank + 1 : leadRank + 2, &ctx_));
					return AffineMap::get(loopRank, 0, exprs, &ctx_);
				};

				SmallVector<AffineExpr> outputExprs;
				outputExprs.reserve(resultRank);
				for (int64_t dim = 0; dim < leadRank; ++dim)
				{
					outputExprs.push_back(getAffineDimExpr(dim, &ctx_));
				}
				outputExprs.push_back(getAffineDimExpr(leadRank, &ctx_));
				outputExprs.push_back(getAffineDimExpr(leadRank + 2, &ctx_));

				auto lhsMap = buildInputMap(lhsType, lhsLeadRank, true);
				auto rhsMap = buildInputMap(rhsType, rhsLeadRank, false);
				auto outputMap = AffineMap::get(loopRank, 0, outputExprs, &ctx_);
				auto empty = builder_.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
				auto zero = emitScalarZero(builder_, loc, resultType.getElementType());
				auto filled =
				    builder_.create<linalg::FillOp>(loc, ValueRange{ zero }, ValueRange{ empty }).getResult(0);

				SmallVector<utils::IteratorType> iterTypes;
				iterTypes.reserve(loopRank);
				for (int64_t dim = 0; dim < leadRank; ++dim)
				{
					iterTypes.push_back(utils::IteratorType::parallel);
				}
				iterTypes.push_back(utils::IteratorType::parallel);
				iterTypes.push_back(utils::IteratorType::reduction);
				iterTypes.push_back(utils::IteratorType::parallel);

				const auto elemType = resultType.getElementType();
				auto generic = builder_.create<linalg::GenericOp>(
				    loc, TypeRange{ resultType }, ValueRange{ lhs, rhs }, ValueRange{ filled },
				    SmallVector<AffineMap>{ lhsMap, rhsMap, outputMap }, iterTypes,
				    [&](OpBuilder& b, Location l, ValueRange args) {
					    auto product = emitScalarMultiply(b, l, args[0], args[1], elemType);
					    auto sum = emitScalarAdd(b, l, args[2], product, elemType);
					    b.create<linalg::YieldOp>(l, sum);
				    });
				valueMap[nodeId] = { emitMaybeCastValue(generic.getResult(0), computeDType, outputDType,
					                                    outputInfos[0].shape) };
			}

			void emitNode(const PlanSubgraphView&, NodeId, const OutProdNode&, std::span<const OutputInfo>,
			              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
			              std::map<std::size_t, Value>&)
			{
				throw std::runtime_error("GraphToMLIR does not support OutProdNode yet; use the interpreter path");
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const TimestepEmbeddingNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto loc = builder_.getUnknownLoc();
				auto timesteps = getVal(valueMap, node.timesteps);
				auto timestepsType = cast<RankedTensorType>(timesteps.getType());
				auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
				if (timestepsType.getRank() != 1 || resultType.getRank() != 2)
				{
					throw std::runtime_error(
					    "GraphToMLIR TimestepEmbeddingNode requires timesteps [T] and output [T, dim]");
				}
				if (resultType.getElementType() != builder_.getF32Type())
				{
					throw std::runtime_error("GraphToMLIR TimestepEmbeddingNode output must be Float32");
				}

				auto empty = builder_.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
				auto inputMap = AffineMap::get(2, 0, { getAffineDimExpr(0, &ctx_) }, &ctx_);
				auto outputMap = AffineMap::getMultiDimIdentityMap(2, &ctx_);
				SmallVector<utils::IteratorType> iterTypes(2, utils::IteratorType::parallel);
				const auto half = static_cast<std::int64_t>(node.dim / 2);

				auto generic = builder_.create<linalg::GenericOp>(
				    loc, TypeRange{ resultType }, ValueRange{ timesteps }, ValueRange{ empty },
				    SmallVector<AffineMap>{ inputMap, outputMap }, iterTypes,
				    [&](OpBuilder& b, Location l, ValueRange args) {
					    if (half == 0)
					    {
						    b.create<linalg::YieldOp>(l, emitScalarZero(b, l, b.getF32Type()));
						    return;
					    }

					    auto dimIndex = b.create<linalg::IndexOp>(l, 1).getResult();
					    auto dimI64 = b.create<arith::IndexCastOp>(l, b.getI64Type(), dimIndex).getResult();
					    auto halfI64 = emitI64Constant(b, l, half);
					    auto twoHalfI64 = emitI64Constant(b, l, 2 * half);
					    auto isSinHalf =
					        b.create<arith::CmpIOp>(l, arith::CmpIPredicate::sge, dimI64, halfI64).getResult();
					    auto isPadding =
					        b.create<arith::CmpIOp>(l, arith::CmpIPredicate::sge, dimI64, twoHalfI64).getResult();
					    auto sinJ = b.create<arith::SubIOp>(l, dimI64, halfI64).getResult();
					    auto freqIndex = b.create<arith::SelectOp>(l, isSinHalf, sinJ, dimI64).getResult();
					    freqIndex = emitClampedI64(b, l, freqIndex, 0, half - 1);

					    auto freqIndexFloat = b.create<arith::SIToFPOp>(l, b.getF32Type(), freqIndex).getResult();
					    auto scale = b.create<arith::ConstantFloatOp>(
					        l, b.getF32Type(),
					        llvm::APFloat(static_cast<float>(-std::log(static_cast<double>(node.maxPeriod)) /
					                                         static_cast<double>(half))));
					    auto exponent = b.create<arith::MulFOp>(l, freqIndexFloat, scale).getResult();
					    auto frequency = b.create<math::ExpOp>(l, exponent).getResult();
					    auto timestep = emitScalarToF32(b, l, args[0]);
					    auto arg = b.create<arith::MulFOp>(l, timestep, frequency).getResult();
					    auto cosValue = b.create<math::CosOp>(l, arg).getResult();
					    auto sinValue = b.create<math::SinOp>(l, arg).getResult();
					    auto wave = b.create<arith::SelectOp>(l, isSinHalf, sinValue, cosValue).getResult();
					    auto result =
					        b.create<arith::SelectOp>(l, isPadding, emitScalarZero(b, l, b.getF32Type()), wave)
					            .getResult();
					    b.create<linalg::YieldOp>(l, result);
				    });
				valueMap[nodeId] = { generic.getResult(0) };
			}

			void emitNode(const PlanSubgraphView&, NodeId, const SolveTriNode&, std::span<const OutputInfo>,
			              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
			              std::map<std::size_t, Value>&)
			{
				throw std::runtime_error("GraphToMLIR does not support SolveTriNode yet; use the interpreter path");
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const SGDStepNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				if (outputInfos.empty())
				{
					throw std::runtime_error("GraphToMLIR SGDStepNode requires an updated-parameter output");
				}
				if (node.momentum != 0.0 || node.velocity.has_value() || outputInfos.size() != 1)
				{
					throw std::runtime_error("GraphToMLIR SGDStepNode currently supports only momentum-free SGD");
				}
				if (outputInfos[0].dtype != DataType::Float32)
				{
					throw std::runtime_error("GraphToMLIR SGDStepNode currently supports Float32 tensors only");
				}

				const auto shape = outputInfos[0].shape;
				auto parameter = getVal(valueMap, node.parameter);
				auto gradient = getVal(valueMap, node.gradient);
				auto regularizedGradient = gradient;
				if (node.weightDecay != 0.0)
				{
					auto weightDecay = emitFilledConstant(DataType::Float32, shape, node.weightDecay);
					auto decay =
					    emitBinaryValue(LiteNN::BinaryOp::Multiply, parameter, weightDecay, DataType::Float32, shape);
					regularizedGradient =
					    emitBinaryValue(LiteNN::BinaryOp::Add, gradient, decay, DataType::Float32, shape);
				}
				auto learningRate = emitFilledConstant(DataType::Float32, shape, node.learningRate);
				auto scaledUpdate = emitBinaryValue(LiteNN::BinaryOp::Multiply, regularizedGradient, learningRate,
				                                    DataType::Float32, shape);
				valueMap[nodeId] = { emitBinaryValue(LiteNN::BinaryOp::Subtract, parameter, scaledUpdate,
					                                 DataType::Float32, shape) };
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const AdamWStepNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				if (outputInfos.size() != 3)
				{
					throw std::runtime_error(
					    "GraphToMLIR AdamWStepNode requires parameter, firstMoment, and secondMoment outputs");
				}
				for (const auto& output : outputInfos)
				{
					if (output.dtype != DataType::Float32)
					{
						throw std::runtime_error("GraphToMLIR AdamWStepNode currently supports Float32 tensors only");
					}
				}

				auto loc = builder_.getUnknownLoc();
				auto parameter = getVal(valueMap, node.parameter);
				auto gradient = getVal(valueMap, node.gradient);
				auto firstMoment = getVal(valueMap, node.firstMoment);
				auto secondMoment = getVal(valueMap, node.secondMoment);
				auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
				const auto rank = resultType.getRank();
				auto elemType = resultType.getElementType();
				SmallVector<Value> outputs;
				outputs.reserve(3);
				for (const auto& output : outputInfos)
				{
					auto type = convertTensorType(ctx_, output.dtype, output.shape);
					outputs.push_back(builder_.create<tensor::EmptyOp>(loc, type.getShape(), type.getElementType()));
				}

				SmallVector<AffineMap> maps;
				maps.reserve(7);
				auto identityMap = AffineMap::getMultiDimIdentityMap(rank, &ctx_);
				for (std::size_t i = 0; i < 7; ++i)
				{
					maps.push_back(identityMap);
				}
				SmallVector<utils::IteratorType> iterTypes(rank, utils::IteratorType::parallel);
				auto biasCorrection1 = 1.0 - std::pow(node.beta1, static_cast<double>(node.step));
				auto biasCorrection2 = 1.0 - std::pow(node.beta2, static_cast<double>(node.step));

				auto generic = builder_.create<linalg::GenericOp>(
				    loc, convertOutputInfos(outputInfos), ValueRange{ parameter, gradient, firstMoment, secondMoment },
				    outputs, maps, iterTypes, [&](OpBuilder& b, Location l, ValueRange args) {
					    auto beta1 = emitScalarConstant(b, l, elemType, node.beta1);
					    auto beta2 = emitScalarConstant(b, l, elemType, node.beta2);
					    auto oneMinusBeta1 = emitScalarConstant(b, l, elemType, 1.0 - node.beta1);
					    auto oneMinusBeta2 = emitScalarConstant(b, l, elemType, 1.0 - node.beta2);
					    auto learningRate = emitScalarConstant(b, l, elemType, node.learningRate);
					    auto weightDecay = emitScalarConstant(b, l, elemType, node.weightDecay);
					    auto bias1 = emitScalarConstant(b, l, elemType, biasCorrection1);
					    auto bias2 = emitScalarConstant(b, l, elemType, biasCorrection2);
					    auto epsilon = emitScalarConstant(b, l, elemType, node.epsilon);
					    auto one = emitScalarConstant(b, l, elemType, 1.0);

					    auto updatedFirst =
					        b.create<arith::AddFOp>(l, b.create<arith::MulFOp>(l, beta1, args[2]).getResult(),
					                                b.create<arith::MulFOp>(l, oneMinusBeta1, args[1]).getResult())
					            .getResult();
					    auto gradientSquared = b.create<arith::MulFOp>(l, args[1], args[1]).getResult();
					    auto updatedSecond =
					        b.create<arith::AddFOp>(
					             l, b.create<arith::MulFOp>(l, beta2, args[3]).getResult(),
					             b.create<arith::MulFOp>(l, oneMinusBeta2, gradientSquared).getResult())
					            .getResult();
					    auto firstHat = b.create<arith::DivFOp>(l, updatedFirst, bias1).getResult();
					    auto secondHat = b.create<arith::DivFOp>(l, updatedSecond, bias2).getResult();
					    auto denom =
					        b.create<arith::AddFOp>(l, b.create<math::SqrtOp>(l, secondHat).getResult(), epsilon)
					            .getResult();
					    auto decayScale =
					        b.create<arith::SubFOp>(l, one,
					                                b.create<arith::MulFOp>(l, learningRate, weightDecay).getResult())
					            .getResult();
					    auto decayedParameter = b.create<arith::MulFOp>(l, args[0], decayScale).getResult();
					    auto normalizedUpdate = b.create<arith::DivFOp>(l, firstHat, denom).getResult();
					    auto scaledUpdate = b.create<arith::MulFOp>(l, learningRate, normalizedUpdate).getResult();
					    auto updatedParameter = b.create<arith::SubFOp>(l, decayedParameter, scaledUpdate).getResult();
					    b.create<linalg::YieldOp>(l, ValueRange{ updatedParameter, updatedFirst, updatedSecond });
				    });
				valueMap[nodeId] = { generic.getResult(0), generic.getResult(1), generic.getResult(2) };
			}

			void emitNode(const PlanSubgraphView&, NodeId, const Im2ColNode&, std::span<const OutputInfo>,
			              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
			              std::map<std::size_t, Value>&)
			{
				throw std::runtime_error("GraphToMLIR does not support Im2ColNode yet; use the interpreter path");
			}

			void emitNode(const PlanSubgraphView& sg, NodeId nodeId, const Conv2DNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				auto loc = builder_.getUnknownLoc();
				const auto outputDType = outputInfos[0].dtype;
				const auto computeDType = shouldUseFloat32Accumulator(outputDType) ? DataType::Float32 : outputDType;
				const auto inputInfo = sg.GetOutputInfo(node.input);
				const auto weightInfo = sg.GetOutputInfo(node.weight);
				auto input =
				    emitMaybeCastValue(getVal(valueMap, node.input), inputInfo.dtype, computeDType, inputInfo.shape);
				auto weight =
				    emitMaybeCastValue(getVal(valueMap, node.weight), weightInfo.dtype, computeDType, weightInfo.shape);
				auto inputType = cast<RankedTensorType>(input.getType());
				auto weightType = cast<RankedTensorType>(weight.getType());
				auto resultType = convertTensorType(ctx_, computeDType, outputInfos[0].shape);
				if (inputType.getRank() != 4 || weightType.getRank() != 4 || resultType.getRank() != 4)
				{
					throw std::runtime_error(
					    "GraphToMLIR Conv2DNode requires rank-4 input, weight, and output tensors");
				}
				if (node.strides.size() != 2 || node.dilations.size() != 2 || node.lowPads.size() != 2 ||
				    node.highPads.size() != 2)
				{
					throw std::runtime_error("GraphToMLIR Conv2DNode requires rank-2 spatial parameters");
				}
				if (node.groupCount == 0)
				{
					throw std::runtime_error("GraphToMLIR Conv2DNode requires groupCount > 0");
				}

				const auto inputChannels = inputType.getDimSize(1);
				const auto outputChannels = weightType.getDimSize(0);
				const auto inChannelsPerGroup = weightType.getDimSize(1);
				if (inputChannels < 0 || outputChannels < 0 || inChannelsPerGroup < 0)
				{
					throw std::runtime_error("GraphToMLIR Conv2DNode requires static channel dimensions");
				}
				if (inputChannels % static_cast<int64_t>(node.groupCount) != 0 ||
				    outputChannels % static_cast<int64_t>(node.groupCount) != 0 ||
				    inputChannels / static_cast<int64_t>(node.groupCount) != inChannelsPerGroup)
				{
					throw std::runtime_error("GraphToMLIR Conv2DNode group/channel dimensions are inconsistent");
				}
				const auto outChannelsPerGroup = outputChannels / static_cast<int64_t>(node.groupCount);

				auto empty = builder_.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
				auto zero = emitScalarZero(builder_, loc, resultType.getElementType());
				auto filled =
				    builder_.create<linalg::FillOp>(loc, ValueRange{ zero }, ValueRange{ empty }).getResult(0);

				constexpr int64_t kLoopRank = 7;
				auto outputMap = AffineMap::get(kLoopRank, 0,
				                                { getAffineDimExpr(0, &ctx_), getAffineDimExpr(1, &ctx_),
				                                  getAffineDimExpr(2, &ctx_), getAffineDimExpr(3, &ctx_) },
				                                &ctx_);
				auto weightMap = AffineMap::get(kLoopRank, 0,
				                                { getAffineDimExpr(1, &ctx_), getAffineDimExpr(4, &ctx_),
				                                  getAffineDimExpr(5, &ctx_), getAffineDimExpr(6, &ctx_) },
				                                &ctx_);
				SmallVector<utils::IteratorType> iterTypes{
					utils::IteratorType::parallel,  utils::IteratorType::parallel,  utils::IteratorType::parallel,
					utils::IteratorType::parallel,  utils::IteratorType::reduction, utils::IteratorType::reduction,
					utils::IteratorType::reduction,
				};

				auto conv = builder_.create<linalg::GenericOp>(
				    loc, TypeRange{ resultType }, ValueRange{ weight }, ValueRange{ filled },
				    SmallVector<AffineMap>{ weightMap, outputMap }, iterTypes,
				    [&](OpBuilder& b, Location l, ValueRange args) {
					    auto n = b.create<linalg::IndexOp>(l, 0).getResult();
					    auto oc = b.create<linalg::IndexOp>(l, 1).getResult();
					    auto oh = b.create<linalg::IndexOp>(l, 2).getResult();
					    auto ow = b.create<linalg::IndexOp>(l, 3).getResult();
					    auto icpg = b.create<linalg::IndexOp>(l, 4).getResult();
					    auto kh = b.create<linalg::IndexOp>(l, 5).getResult();
					    auto kw = b.create<linalg::IndexOp>(l, 6).getResult();

					    auto ocI64 = b.create<arith::IndexCastOp>(l, b.getI64Type(), oc).getResult();
					    auto icpgI64 = b.create<arith::IndexCastOp>(l, b.getI64Type(), icpg).getResult();
					    auto ohI64 = b.create<arith::IndexCastOp>(l, b.getI64Type(), oh).getResult();
					    auto owI64 = b.create<arith::IndexCastOp>(l, b.getI64Type(), ow).getResult();
					    auto khI64 = b.create<arith::IndexCastOp>(l, b.getI64Type(), kh).getResult();
					    auto kwI64 = b.create<arith::IndexCastOp>(l, b.getI64Type(), kw).getResult();

					    auto group =
					        b.create<arith::DivSIOp>(l, ocI64, emitI64Constant(b, l, outChannelsPerGroup)).getResult();
					    auto inputChannel =
					        b.create<arith::AddIOp>(
					             l,
					             b.create<arith::MulIOp>(l, group, emitI64Constant(b, l, inChannelsPerGroup))
					                 .getResult(),
					             icpgI64)
					            .getResult();
					    auto inputY =
					        b.create<arith::SubIOp>(
					             l,
					             b.create<arith::AddIOp>(
					                  l,
					                  b.create<arith::MulIOp>(
					                       l, ohI64, emitI64Constant(b, l, static_cast<std::int64_t>(node.strides[0])))
					                      .getResult(),
					                  b.create<arith::MulIOp>(
					                       l, khI64,
					                       emitI64Constant(b, l, static_cast<std::int64_t>(node.dilations[0])))
					                      .getResult())
					                 .getResult(),
					             emitI64Constant(b, l, static_cast<std::int64_t>(node.lowPads[0])))
					            .getResult();
					    auto inputX =
					        b.create<arith::SubIOp>(
					             l,
					             b.create<arith::AddIOp>(
					                  l,
					                  b.create<arith::MulIOp>(
					                       l, owI64, emitI64Constant(b, l, static_cast<std::int64_t>(node.strides[1])))
					                      .getResult(),
					                  b.create<arith::MulIOp>(
					                       l, kwI64,
					                       emitI64Constant(b, l, static_cast<std::int64_t>(node.dilations[1])))
					                      .getResult())
					                 .getResult(),
					             emitI64Constant(b, l, static_cast<std::int64_t>(node.lowPads[1])))
					            .getResult();

					    auto zeroI64 = emitI64Constant(b, l, 0);
					    auto yInLow =
					        b.create<arith::CmpIOp>(l, arith::CmpIPredicate::sge, inputY, zeroI64).getResult();
					    auto yInHigh = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::slt, inputY,
					                                           emitI64Constant(b, l, inputType.getDimSize(2)))
					                       .getResult();
					    auto xInLow =
					        b.create<arith::CmpIOp>(l, arith::CmpIPredicate::sge, inputX, zeroI64).getResult();
					    auto xInHigh = b.create<arith::CmpIOp>(l, arith::CmpIPredicate::slt, inputX,
					                                           emitI64Constant(b, l, inputType.getDimSize(3)))
					                       .getResult();
					    auto inBounds = b.create<arith::AndIOp>(l, yInLow, yInHigh).getResult();
					    inBounds = b.create<arith::AndIOp>(l, inBounds, xInLow).getResult();
					    inBounds = b.create<arith::AndIOp>(l, inBounds, xInHigh).getResult();

					    SmallVector<Value> inputCoords{
						    n,
						    emitI64ToIndex(b, l, inputChannel),
						    emitI64ToIndex(b, l, emitClampedI64(b, l, inputY, 0, inputType.getDimSize(2) - 1)),
						    emitI64ToIndex(b, l, emitClampedI64(b, l, inputX, 0, inputType.getDimSize(3) - 1)),
					    };
					    auto inputElement = b.create<tensor::ExtractOp>(l, input, inputCoords).getResult();
					    auto product = emitScalarMultiply(b, l, inputElement, args[0], resultType.getElementType());
					    auto maskedProduct =
					        b.create<arith::SelectOp>(l, inBounds, product,
					                                  emitScalarZero(b, l, resultType.getElementType()))
					            .getResult();
					    auto sum = emitScalarAdd(b, l, args[1], maskedProduct, resultType.getElementType());
					    b.create<linalg::YieldOp>(l, sum);
				    });
				Value result = conv.getResult(0);

				if (node.bias)
				{
					const auto biasInfo = sg.GetOutputInfo(*node.bias);
					auto bias =
					    emitMaybeCastValue(getVal(valueMap, *node.bias), biasInfo.dtype, computeDType, biasInfo.shape);
					auto biasType = cast<RankedTensorType>(bias.getType());
					SmallVector<AffineExpr> biasExprs;
					if (biasType.getRank() == 1)
					{
						biasExprs.push_back(getAffineDimExpr(1, &ctx_));
					}
					else if (biasType.getRank() == 4)
					{
						biasExprs = { getAffineConstantExpr(0, &ctx_), getAffineDimExpr(1, &ctx_),
							          getAffineConstantExpr(0, &ctx_), getAffineConstantExpr(0, &ctx_) };
					}
					else
					{
						throw std::runtime_error("GraphToMLIR Conv2DNode bias must be rank-1 or rank-4");
					}

					auto biasMap = AffineMap::get(4, 0, biasExprs, &ctx_);
					auto convMap = AffineMap::getMultiDimIdentityMap(4, &ctx_);
					auto outMap = AffineMap::getMultiDimIdentityMap(4, &ctx_);
					auto biasEmpty =
					    builder_.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
					SmallVector<utils::IteratorType> biasIterTypes(4, utils::IteratorType::parallel);
					auto withBias = builder_.create<linalg::GenericOp>(
					    loc, TypeRange{ resultType }, ValueRange{ result, bias }, ValueRange{ biasEmpty },
					    SmallVector<AffineMap>{ convMap, biasMap, outMap }, biasIterTypes,
					    [&](OpBuilder& b, Location l, ValueRange args) {
						    auto sum = emitScalarAdd(b, l, args[0], args[1], resultType.getElementType());
						    b.create<linalg::YieldOp>(l, sum);
					    });
					result = withBias.getResult(0);
				}

				valueMap[nodeId] = { emitMaybeCastValue(result, computeDType, outputDType, outputInfos[0].shape) };
			}

			void emitNode(const PlanSubgraphView&, NodeId, const ConvTranspose2DNode&, std::span<const OutputInfo>,
			              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
			              std::map<std::size_t, Value>&)
			{
				throw std::runtime_error(
				    "GraphToMLIR does not support ConvTranspose2DNode yet; use the interpreter path");
			}

			void emitNode(const PlanSubgraphView&, NodeId, const Pool2DNode&, std::span<const OutputInfo>,
			              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
			              std::map<std::size_t, Value>&)
			{
				throw std::runtime_error("GraphToMLIR does not support Pool2DNode yet; use the interpreter path");
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const UpsampleNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
			{
				if (node.mode != UpsampleMode::Nearest || node.alignCorners)
				{
					throw std::runtime_error(
					    "GraphToMLIR UpsampleNode currently supports nearest mode with alignCorners=false only");
				}

				auto loc = builder_.getUnknownLoc();
				auto input = getVal(valueMap, node.input);
				auto inputType = cast<RankedTensorType>(input.getType());
				auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
				if (inputType.getRank() != 4 || resultType.getRank() != 4)
				{
					throw std::runtime_error("GraphToMLIR UpsampleNode requires rank-4 NCHW tensors");
				}
				if (inputType.getDimSize(2) <= 0 || inputType.getDimSize(3) <= 0 || resultType.getDimSize(2) <= 0 ||
				    resultType.getDimSize(3) <= 0)
				{
					throw std::runtime_error("GraphToMLIR UpsampleNode requires static non-zero spatial dimensions");
				}

				auto empty = builder_.create<tensor::EmptyOp>(loc, resultType.getShape(), resultType.getElementType());
				auto outputMap = AffineMap::getMultiDimIdentityMap(4, &ctx_);
				SmallVector<utils::IteratorType> iterTypes(4, utils::IteratorType::parallel);
				auto generic = builder_.create<linalg::GenericOp>(
				    loc, TypeRange{ resultType }, ValueRange{}, ValueRange{ empty },
				    SmallVector<AffineMap>{ outputMap }, iterTypes, [&](OpBuilder& b, Location l, ValueRange) {
					    auto n = b.create<linalg::IndexOp>(l, 0).getResult();
					    auto c = b.create<linalg::IndexOp>(l, 1).getResult();
					    auto oh = emitIndexAsI64(b, l, 2);
					    auto ow = emitIndexAsI64(b, l, 3);

					    auto sourceY =
					        b.create<arith::DivSIOp>(
					             l,
					             b.create<arith::MulIOp>(l, oh, emitI64Constant(b, l, inputType.getDimSize(2)))
					                 .getResult(),
					             emitI64Constant(b, l, resultType.getDimSize(2)))
					            .getResult();
					    auto sourceX =
					        b.create<arith::DivSIOp>(
					             l,
					             b.create<arith::MulIOp>(l, ow, emitI64Constant(b, l, inputType.getDimSize(3)))
					                 .getResult(),
					             emitI64Constant(b, l, resultType.getDimSize(3)))
					            .getResult();

					    SmallVector<Value> inputCoords{
						    n,
						    c,
						    emitI64ToIndex(b, l, emitClampedI64(b, l, sourceY, 0, inputType.getDimSize(2) - 1)),
						    emitI64ToIndex(b, l, emitClampedI64(b, l, sourceX, 0, inputType.getDimSize(3) - 1)),
					    };
					    auto element = b.create<tensor::ExtractOp>(l, input, inputCoords).getResult();
					    b.create<linalg::YieldOp>(l, element);
				    });
				valueMap[nodeId] = { generic.getResult(0) };
			}

			void emitNode(const PlanSubgraphView&, NodeId, const MulMatIdNode&, std::span<const OutputInfo>,
			              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
			              std::map<std::size_t, Value>&)
			{
				throw std::runtime_error("GraphToMLIR does not support MulMatIdNode yet; use the interpreter path");
			}

			void emitNode(const PlanSubgraphView&, NodeId nodeId, const FusedOpNode& node,
			              std::span<const OutputInfo> outputInfos, std::vector<SmallVector<Value>>& valueMap,
			              std::map<std::size_t, Value>& activationMap, std::map<std::size_t, Value>& tapeMap)
			{
				auto resultTypes = convertOutputInfos(outputInfos);
				SmallVector<Value> args;
				for (const auto& arg : node.args)
				{
					args.push_back(getVal(valueMap, arg));
				}

				auto op = builder_.create<FusedOp>(builder_.getUnknownLoc(), resultTypes,
				                                   convertFusionPattern(node.pattern), args);

				// Emit body region
				emitSubgraphIntoRegion(node.body, op.getBody(), activationMap, tapeMap);

				valueMap[nodeId].clear();
				for (auto result : op.getResults())
				{
					valueMap[nodeId].push_back(result);
				}
			}

			// Inline a subgraph into a region (for CondNode, WhileNode, FusedOp)
			void emitSubgraphIntoRegion(SubgraphId sgId, Region& region, std::map<std::size_t, Value>& activationMap,
			                            std::map<std::size_t, Value>& tapeMap)
			{
				if (sgId >= plan_.subgraphs.size())
				{
					throw std::runtime_error("ExecutablePlan MLIR lowering region references an out-of-range subgraph");
				}
				const PlanSubgraphView sg{ plan_.subgraphs[sgId] };

				// Create block with params as arguments
				SmallVector<Type> blockArgTypes;
				SmallVector<Location> blockArgLocs;
				for (const auto& param : sg.Params())
				{
					blockArgTypes.push_back(convertTensorType(ctx_, param));
					blockArgLocs.push_back(builder_.getUnknownLoc());
				}

				OpBuilder::InsertionGuard guard(builder_);
				auto* block = builder_.createBlock(&region, {}, blockArgTypes, blockArgLocs);
				builder_.setInsertionPointToStart(block);

				// Emit body
				std::vector<SmallVector<Value>> valueMap(sg.NodeCount());
				emitSubgraphBody(sg, *block, valueMap, activationMap, tapeMap);

				// Emit yield with results
				SmallVector<Value> results;
				for (const auto& result : sg.Results())
				{
					results.push_back(valueMap[result.node][result.port]);
				}

				builder_.create<YieldOp>(builder_.getUnknownLoc(), results);
			}

			const ExecutablePlan& plan_;
			MLIRContext& ctx_;
			OpBuilder builder_;
			OwningOpRef<ModuleOp> module_;
		};

	} // namespace

	OwningOpRef<ModuleOp> TranslateGraphToMLIRInternal(const Graph& graph, MLIRContext& ctx)
	{
		Validation::ValidateGraph(graph);
		return translateExecutablePlanToMLIR(Detail::BuildExecutablePlanFromGraph(graph), ctx);
	}

	OwningOpRef<ModuleOp> translateExecutablePlanToMLIR(const ExecutablePlan& plan, MLIRContext& ctx)
	{
		ctx.loadDialect<litenn::LiteNNDialect, arith::ArithDialect, linalg::LinalgDialect, math::MathDialect,
		                tensor::TensorDialect>();
		ValidateExecutablePlan(plan);
		GraphTranslator translator(plan, ctx);
		return translator.translate();
	}

} // namespace litenn
