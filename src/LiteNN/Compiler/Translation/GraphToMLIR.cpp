#include "Translation/GraphToMLIR.h"
#include "Dialect/LiteNNOps.h"

#include <LiteNN/Graph.h>
#include <LiteNN/Tensor.h>
#include <LiteNN/Validation/GraphValidator.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "llvm/ADT/APInt.h"

#include <algorithm>
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

// LiteNN enum → litenn dialect enum
UnaryOpKind convertUnaryOp(LiteNN::UnaryOp op) { return static_cast<UnaryOpKind>(op); }
BinaryOpKind convertBinaryOp(LiteNN::BinaryOp op) { return static_cast<BinaryOpKind>(op); }
ReduceOpKind convertReduceOp(LiteNN::ReduceOp op) { return static_cast<ReduceOpKind>(op); }
FusionPatternKind convertFusionPattern(FusionPattern pat) { return static_cast<FusionPatternKind>(pat); }

// Extract tensor data to DenseElementsAttr
DenseElementsAttr convertTensorToAttr(MLIRContext& ctx, const Tensor<PolymorphicDevice>& tensor)
{
	auto tensorType = convertTensorType(ctx, tensor.DType(), tensor.Shape());
	auto cpuTensor = tensor.CopyToDevice(CPU{});

	const auto numElements = std::max(ShapeView{ cpuTensor.Shape() }.NumElements(), std::size_t(1));
	const auto* rawData = cpuTensor.RawData();

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
			data.push_back(FloatAttr::get(tensorType.getElementType(), static_cast<float>(src[i])));
		return DenseElementsAttr::get(tensorType, data);
	}
	case DataType::BFloat16: {
		SmallVector<Attribute> data;
		data.reserve(numElements);
		const auto* src = static_cast<const BFloat16*>(rawData);
		for (std::size_t i = 0; i < numElements; ++i)
			data.push_back(FloatAttr::get(tensorType.getElementType(), static_cast<float>(src[i])));
		return DenseElementsAttr::get(tensorType, data);
	}
	case DataType::Float8E4M3: {
		SmallVector<Attribute> data;
		data.reserve(numElements);
		const auto* src = static_cast<const Float8E4M3*>(rawData);
		for (std::size_t i = 0; i < numElements; ++i)
			data.push_back(IntegerAttr::get(tensorType.getElementType(), llvm::APInt(8, src[i].bits)));
		return DenseElementsAttr::get(tensorType, data);
	}
	case DataType::Float8E5M2: {
		SmallVector<Attribute> data;
		data.reserve(numElements);
		const auto* src = static_cast<const Float8E5M2*>(rawData);
		for (std::size_t i = 0; i < numElements; ++i)
			data.push_back(IntegerAttr::get(tensorType.getElementType(), llvm::APInt(8, src[i].bits)));
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
			data.push_back(IntegerAttr::get(tensorType.getElementType(), llvm::APInt(8, src[i])));
		return DenseElementsAttr::get(tensorType, data);
	}
	case DataType::Bool: {
		// MLIR DenseElementsAttr for i1 expects bool
		SmallVector<bool> boolData(numElements);
		const auto* src = static_cast<const bool*>(rawData);
		for (std::size_t i = 0; i < numElements; ++i)
			boolData[i] = src[i];
		return DenseElementsAttr::get(tensorType, ArrayRef(boolData));
	}
	}
	llvm_unreachable("unknown DataType");
}

class GraphTranslator
{
public:
	GraphTranslator(const Graph& graph, MLIRContext& ctx) : graph_(graph), ctx_(ctx), builder_(&ctx) {}

	OwningOpRef<ModuleOp> translate()
	{
		module_ = ModuleOp::create(builder_.getUnknownLoc());
		builder_.setInsertionPointToStart(module_->getBody());

		// Emit variable declarations
		for (std::size_t i = 0; i < graph_.VariableCount(); ++i)
		{
			emitVariable(i);
		}

		// Emit subgraph functions
		for (std::size_t i = 0; i < graph_.SubgraphCount(); ++i)
		{
			emitSubgraphFunc(i);
		}

		return std::move(module_);
	}

private:
	void emitVariable(std::size_t varIndex)
	{
		const auto& var = graph_.GetVariable(varIndex);
		const auto& data = var->Data();
		auto tensorType = convertTensorType(ctx_, data.DType(), data.Shape());
		auto initialValue = convertTensorToAttr(ctx_, data);
		auto name = "var_" + std::to_string(varIndex);

		builder_.create<VariableOp>(builder_.getUnknownLoc(), name, tensorType, initialValue);
	}

	void emitSubgraphFunc(std::size_t sgId)
	{
		const auto& sg = graph_.GetSubgraph(sgId);

		// Build function type
		SmallVector<Type> inputTypes;
		for (const auto& param : sg.Params())
			inputTypes.push_back(convertTensorType(ctx_, param.dtype, param.shape));

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
			returnValues.push_back(valueMap[result.node][result.port]);

		builder_.create<ReturnOp>(builder_.getUnknownLoc(), returnValues);
	}

	void emitSubgraphBody(const Subgraph& sg, Block& block, std::vector<SmallVector<Value>>& valueMap,
	                       std::map<std::size_t, Value>& activationMap, std::map<std::size_t, Value>& tapeMap)
	{
		for (NodeId nodeId = 0; nodeId < sg.NodeCount(); ++nodeId)
		{
			const auto& entry = sg.GetNodeEntry(nodeId);
			std::visit(
			    [&](const auto& node) { emitNode(sg, nodeId, node, entry.outputInfos, valueMap, activationMap, tapeMap); },
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
			auto* data = static_cast<T*>(tensor.RawData());
			std::fill(data, data + tensor.NumElements(), static_cast<T>(value));
		});
		auto poly = tensor.CopyToDevice(PolymorphicDevice{ CPU{} });
		auto attr = convertTensorToAttr(ctx_, poly);
		auto op = builder_.create<ConstantOp>(builder_.getUnknownLoc(), attr.getType(), attr);
		return op.getResult();
	}

	Value emitUnaryValue(LiteNN::UnaryOp opKind, Value input, DataType dtype, std::span<const std::size_t> shape)
	{
		auto resultType = convertTensorType(ctx_, dtype, shape);
		auto op = builder_.create<litenn::UnaryOp>(builder_.getUnknownLoc(), resultType, convertUnaryOp(opKind), input);
		return op.getResult();
	}

	Value emitBinaryValue(LiteNN::BinaryOp opKind, Value lhs, Value rhs, DataType dtype,
	                      std::span<const std::size_t> shape)
	{
		auto resultType = convertTensorType(ctx_, dtype, shape);
		auto op = builder_.create<litenn::BinaryOp>(builder_.getUnknownLoc(), resultType, convertBinaryOp(opKind), lhs,
		                                            rhs);
		return op.getResult();
	}

	Value emitReduceValue(LiteNN::ReduceOp opKind, Value input, DataType dtype,
	                      std::span<const std::size_t> shape, std::size_t axis)
	{
		auto resultType = convertTensorType(ctx_, dtype, shape);
		auto op = builder_.create<litenn::ReduceOp>(builder_.getUnknownLoc(), resultType, convertReduceOp(opKind),
		                                            input, static_cast<uint64_t>(axis));
		return op.getResult();
	}

	Value emitReshapeValue(Value input, DataType dtype, std::span<const std::size_t> shape)
	{
		auto resultType = convertTensorType(ctx_, dtype, shape);
		auto op = builder_.create<ReshapeOp>(builder_.getUnknownLoc(), resultType, input);
		return op.getResult();
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

	SmallVector<Type> convertOutputInfos(std::span<const OutputInfo> infos)
	{
		SmallVector<Type> types;
		for (const auto& info : infos)
			types.push_back(convertTensorType(ctx_, info.dtype, info.shape));
		return types;
	}

	// ---- Per-node emission ----

	void emitNode(const Subgraph& sg, NodeId nodeId, const ParamRefNode& node,
	              std::span<const OutputInfo> /*outputInfos*/, std::vector<SmallVector<Value>>& valueMap,
	              std::map<std::size_t, Value>&, std::map<std::size_t, Value>&)
	{
		// Block arguments correspond to params
		auto* block = builder_.getInsertionBlock();
		valueMap[nodeId] = { block->getArgument(node.paramIndex) };
	}

	void emitNode(const Subgraph&, NodeId nodeId, const ConstantNode& node, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		auto attr = convertTensorToAttr(ctx_, node.value);
		auto op = builder_.create<ConstantOp>(builder_.getUnknownLoc(), attr.getType(), attr);
		valueMap[nodeId] = { op.getResult() };
	}

	void emitNode(const Subgraph&, NodeId nodeId, const QuantizedConstantNode& node, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		auto attr = convertTensorToAttr(ctx_, node.storage);
		auto op = builder_.create<ConstantOp>(builder_.getUnknownLoc(), attr.getType(), attr);
		valueMap[nodeId] = { op.getResult() };
	}

	void emitNode(const Subgraph&, NodeId nodeId, const VariableRefNode& node, std::span<const OutputInfo> outputInfos,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
		auto name = "var_" + std::to_string(node.variableIndex);
		auto op =
		    builder_.create<GetVariableOp>(builder_.getUnknownLoc(), resultType, FlatSymbolRefAttr::get(&ctx_, name));
		valueMap[nodeId] = { op.getResult() };
	}

	void emitNode(const Subgraph&, NodeId nodeId, const UnaryOpNode& node, std::span<const OutputInfo> outputInfos,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
		auto input = getVal(valueMap, node.input);
		auto op = builder_.create<litenn::UnaryOp>(builder_.getUnknownLoc(), resultType, convertUnaryOp(node.op), input);
		valueMap[nodeId] = { op.getResult() };
	}

	void emitNode(const Subgraph&, NodeId nodeId, const BinaryOpNode& node, std::span<const OutputInfo> outputInfos,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
		auto lhs = getVal(valueMap, node.lhs);
		auto rhs = getVal(valueMap, node.rhs);
		auto op = builder_.create<litenn::BinaryOp>(builder_.getUnknownLoc(), resultType, convertBinaryOp(node.op), lhs,
		                                            rhs);
		valueMap[nodeId] = { op.getResult() };
	}

	void emitNode(const Subgraph&, NodeId nodeId, const CastNode& node, std::span<const OutputInfo> outputInfos,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
		auto input = getVal(valueMap, node.input);
		auto op = builder_.create<CastOp>(builder_.getUnknownLoc(), resultType, input);
		valueMap[nodeId] = { op.getResult() };
	}

	void emitNode(const Subgraph&, NodeId, const QuantizeNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support QuantizeNode yet");
	}

	void emitNode(const Subgraph&, NodeId, const DequantizeNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support DequantizeNode yet");
	}

	void emitNode(const Subgraph&, NodeId nodeId, const CallNode& node, std::span<const OutputInfo> outputInfos,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		auto resultTypes = convertOutputInfos(outputInfos);
		SmallVector<Value> args;
		for (const auto& arg : node.args)
			args.push_back(getVal(valueMap, arg));

		auto calleeName = "subgraph_" + std::to_string(node.callee);
		auto op = builder_.create<CallOp>(builder_.getUnknownLoc(), resultTypes,
		                                  FlatSymbolRefAttr::get(&ctx_, calleeName), args,
		                                  /*arg_attrs=*/nullptr, /*res_attrs=*/nullptr);
		valueMap[nodeId].clear();
		for (auto result : op.getResults())
			valueMap[nodeId].push_back(result);
	}

	void emitNode(const Subgraph&, NodeId nodeId, const CondNode& node, std::span<const OutputInfo> outputInfos,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>& activationMap,
	              std::map<std::size_t, Value>& tapeMap)
	{
		auto resultTypes = convertOutputInfos(outputInfos);
		auto condition = getVal(valueMap, node.condition);
		SmallVector<Value> args;
		for (const auto& arg : node.args)
			args.push_back(getVal(valueMap, arg));

		auto op = builder_.create<CondOp>(builder_.getUnknownLoc(), resultTypes, condition, args);

		// Emit then region
		emitSubgraphIntoRegion(node.thenBranch, op.getThenRegion(), activationMap, tapeMap);

		// Emit else region
		emitSubgraphIntoRegion(node.elseBranch, op.getElseRegion(), activationMap, tapeMap);

		valueMap[nodeId].clear();
		for (auto result : op.getResults())
			valueMap[nodeId].push_back(result);
	}

	void emitNode(const Subgraph&, NodeId nodeId, const WhileNode& node, std::span<const OutputInfo> outputInfos,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>& activationMap,
	              std::map<std::size_t, Value>& tapeMap)
	{
		auto resultTypes = convertOutputInfos(outputInfos);
		SmallVector<Value> initArgs;
		for (const auto& arg : node.initArgs)
			initArgs.push_back(getVal(valueMap, arg));

		auto op = builder_.create<WhileOp>(builder_.getUnknownLoc(), resultTypes, initArgs);

		// Emit cond region
		emitSubgraphIntoRegion(node.condBranch, op.getCondRegion(), activationMap, tapeMap);

		// Emit body region
		emitSubgraphIntoRegion(node.bodyBranch, op.getBodyRegion(), activationMap, tapeMap);

		valueMap[nodeId].clear();
		for (auto result : op.getResults())
			valueMap[nodeId].push_back(result);
	}

	void emitNode(const Subgraph&, NodeId nodeId, const SaveActivationNode& node, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>& activationMap,
	              std::map<std::size_t, Value>&)
	{
		// SSA化: passthrough, record mapping
		auto input = getVal(valueMap, node.input);
		activationMap[node.slotId] = input;
		valueMap[nodeId] = { input };
	}

	void emitNode(const Subgraph&, NodeId nodeId, const LoadActivationNode& node,
	              std::span<const OutputInfo>, std::vector<SmallVector<Value>>& valueMap,
	              std::map<std::size_t, Value>& activationMap, std::map<std::size_t, Value>&)
	{
		// SSA化: lookup saved value
		valueMap[nodeId] = { activationMap.at(node.slotId) };
	}

	void emitNode(const Subgraph&, NodeId nodeId, const TapeSaveActivationNode& node, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>& tapeMap)
	{
		auto input = getVal(valueMap, node.input);
		tapeMap[node.tapeSlotId] = input;
		valueMap[nodeId] = { input };
	}

	void emitNode(const Subgraph&, NodeId nodeId, const TapeLoadActivationNode& node,
	              std::span<const OutputInfo>, std::vector<SmallVector<Value>>& valueMap,
	              std::map<std::size_t, Value>&, std::map<std::size_t, Value>& tapeMap)
	{
		valueMap[nodeId] = { tapeMap.at(node.tapeSlotId) };
	}

	void emitNode(const Subgraph&, NodeId nodeId, const ReduceOpNode& node, std::span<const OutputInfo> outputInfos,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
		auto input = getVal(valueMap, node.input);
		auto op = builder_.create<litenn::ReduceOp>(builder_.getUnknownLoc(), resultType, convertReduceOp(node.op),
		                                            input, static_cast<uint64_t>(node.axis));
		valueMap[nodeId] = { op.getResult() };
	}

	void emitNode(const Subgraph&, NodeId nodeId, const ReshapeNode& node, std::span<const OutputInfo> outputInfos,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
		auto input = getVal(valueMap, node.input);
		auto op = builder_.create<ReshapeOp>(builder_.getUnknownLoc(), resultType, input);
		valueMap[nodeId] = { op.getResult() };
	}

	void emitNode(const Subgraph&, NodeId nodeId, const ConcatNode& node, std::span<const OutputInfo> outputInfos,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
		SmallVector<Value> inputs;
		for (const auto& inp : node.inputs)
			inputs.push_back(getVal(valueMap, inp));

		auto op = builder_.create<ConcatOp>(builder_.getUnknownLoc(), resultType, inputs,
		                                    static_cast<uint64_t>(node.axis));
		valueMap[nodeId] = { op.getResult() };
	}

	void emitNode(const Subgraph&, NodeId nodeId, const SliceNode& node, std::span<const OutputInfo> outputInfos,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		auto resultType = convertTensorType(ctx_, outputInfos[0].dtype, outputInfos[0].shape);
		auto input = getVal(valueMap, node.input);
		auto op = builder_.create<SliceOp>(builder_.getUnknownLoc(), resultType, input,
		                                   static_cast<uint64_t>(node.axis),
		                                   static_cast<uint64_t>(node.start),
		                                   static_cast<uint64_t>(node.length));
		valueMap[nodeId] = { op.getResult() };
	}

	void emitNode(const Subgraph&, NodeId nodeId, const GetRowsNode& node, std::span<const OutputInfo> outputInfos,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
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

	void emitNode(const Subgraph&, NodeId, const ArgsortNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support ArgsortNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph&, NodeId, const PermuteNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support PermuteNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph&, NodeId, const BroadcastToNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support BroadcastToNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph&, NodeId, const PadNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support PadNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph&, NodeId, const GatherNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support GatherNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph&, NodeId, const ScatterNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support ScatterNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph&, NodeId, const ScanNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support ScanNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph&, NodeId, const SSMScanNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support SSMScanNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph&, NodeId, const RWKVWKVNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support RWKVWKVNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph&, NodeId, const SoftmaxNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support SoftmaxNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph& sg, NodeId nodeId, const NormalizationNode& node, std::span<const OutputInfo> outputInfos,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		if (node.mode == NormalizationMode::GroupNorm)
		{
			throw std::runtime_error("GraphToMLIR does not support NormalizationNode GroupNorm yet; use the interpreter path");
		}

		const auto inputInfo = sg.GetOutputInfo(node.input);
		const auto dtype = outputInfos[0].dtype;
		const auto inputShape = ShapeView{ inputInfo.shape };
		const auto reducedShape = ReducedShape(inputShape, node.axis);
		const auto broadcastShape = BroadcastShapeForAxis(inputShape, node.axis);
		auto input = getVal(valueMap, node.input);

		Value centered = input;
		if (node.mode == NormalizationMode::LayerNorm)
		{
			auto mean = emitReduceValue(LiteNN::ReduceOp::Mean, input, dtype, reducedShape, node.axis);
			auto meanBroadcast = emitReshapeValue(mean, dtype, broadcastShape);
			centered = emitBinaryValue(LiteNN::BinaryOp::Subtract, input, meanBroadcast, dtype, inputInfo.shape);
		}

		auto squared = emitBinaryValue(LiteNN::BinaryOp::Multiply, centered, centered, dtype, inputInfo.shape);
		auto variance = emitReduceValue(LiteNN::ReduceOp::Mean, squared, dtype, reducedShape, node.axis);
		auto varianceBroadcast = emitReshapeValue(variance, dtype, broadcastShape);
		auto epsilon = emitFilledConstant(dtype, broadcastShape, node.epsilon);
		auto withEpsilon = emitBinaryValue(LiteNN::BinaryOp::Add, varianceBroadcast, epsilon, dtype, broadcastShape);
		auto denom = emitUnaryValue(LiteNN::UnaryOp::Sqrt, withEpsilon, dtype, broadcastShape);
		auto normalized = emitBinaryValue(LiteNN::BinaryOp::Divide, centered, denom, dtype, inputInfo.shape);

		if (node.scale)
		{
			normalized =
			    emitBinaryValue(LiteNN::BinaryOp::Multiply, normalized, getVal(valueMap, *node.scale), dtype, inputInfo.shape);
		}
		if (node.bias)
		{
			normalized =
			    emitBinaryValue(LiteNN::BinaryOp::Add, normalized, getVal(valueMap, *node.bias), dtype, inputInfo.shape);
		}

		valueMap[nodeId] = { normalized };
	}

	void emitNode(const Subgraph&, NodeId, const BatchMatMulNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support BatchMatMulNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph&, NodeId, const Im2ColNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support Im2ColNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph&, NodeId, const Conv2DNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support Conv2DNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph&, NodeId, const ConvTranspose2DNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support ConvTranspose2DNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph&, NodeId, const Pool2DNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support Pool2DNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph&, NodeId, const UpsampleNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support UpsampleNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph&, NodeId, const MulMatIdNode&, std::span<const OutputInfo>,
	              std::vector<SmallVector<Value>>&, std::map<std::size_t, Value>&,
	              std::map<std::size_t, Value>&)
	{
		throw std::runtime_error("GraphToMLIR does not support MulMatIdNode yet; use the interpreter path");
	}

	void emitNode(const Subgraph&, NodeId nodeId, const FusedOpNode& node, std::span<const OutputInfo> outputInfos,
	              std::vector<SmallVector<Value>>& valueMap, std::map<std::size_t, Value>& activationMap,
	              std::map<std::size_t, Value>& tapeMap)
	{
		auto resultTypes = convertOutputInfos(outputInfos);
		SmallVector<Value> args;
		for (const auto& arg : node.args)
			args.push_back(getVal(valueMap, arg));

		auto op = builder_.create<FusedOp>(builder_.getUnknownLoc(), resultTypes, convertFusionPattern(node.pattern),
		                                   args);

		// Emit body region
		emitSubgraphIntoRegion(node.body, op.getBody(), activationMap, tapeMap);

		valueMap[nodeId].clear();
		for (auto result : op.getResults())
			valueMap[nodeId].push_back(result);
	}

	// Inline a subgraph into a region (for CondNode, WhileNode, FusedOp)
	void emitSubgraphIntoRegion(SubgraphId sgId, Region& region, std::map<std::size_t, Value>& activationMap,
	                             std::map<std::size_t, Value>& tapeMap)
	{
		const auto& sg = graph_.GetSubgraph(sgId);

		// Create block with params as arguments
		SmallVector<Type> blockArgTypes;
		SmallVector<Location> blockArgLocs;
		for (const auto& param : sg.Params())
		{
			blockArgTypes.push_back(convertTensorType(ctx_, param.dtype, param.shape));
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
			results.push_back(valueMap[result.node][result.port]);

		builder_.create<YieldOp>(builder_.getUnknownLoc(), results);
	}

	const Graph& graph_;
	MLIRContext& ctx_;
	OpBuilder builder_;
	OwningOpRef<ModuleOp> module_;
};

} // namespace

OwningOpRef<ModuleOp> translateGraphToMLIR(const Graph& graph, MLIRContext& ctx)
{
	Validation::ValidateGraph(graph);
	GraphTranslator translator(graph, ctx);
	return translator.translate();
}

} // namespace litenn
