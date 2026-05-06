#include "Translation/GraphToMLIR.h"
#include "Dialect/LiteNNOps.h"

#include <LiteNN/Graph.h>
#include <LiteNN/Tensor.h>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

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
	case DataType::Int32:
		return IntegerType::get(&ctx, 32);
	case DataType::Int64:
		return IntegerType::get(&ctx, 64);
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
	case DataType::Int32: {
		ArrayRef<int32_t> data(static_cast<const int32_t*>(rawData), numElements);
		return DenseElementsAttr::get(tensorType, data);
	}
	case DataType::Int64: {
		ArrayRef<int64_t> data(static_cast<const int64_t*>(rawData), numElements);
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
	GraphTranslator translator(graph, ctx);
	return translator.translate();
}

} // namespace litenn
