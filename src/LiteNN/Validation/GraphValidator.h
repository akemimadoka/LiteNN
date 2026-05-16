#include <LiteNN/Graph.h>

#include <algorithm>
#include <format>
#include <functional>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#ifndef LITENN_VALIDATION_GRAPHVALIDATOR_H
#define LITENN_VALIDATION_GRAPHVALIDATOR_H

namespace LiteNN::Validation
{
	class GraphValidationError : public std::runtime_error
	{
	public:
		using std::runtime_error::runtime_error;
	};

	inline std::string DataTypeToString(DataType dtype)
	{
		if (IsValidDataTypeValue(dtype))
		{
			return std::string(DataTypeName(dtype));
		}
		return std::format("<invalid:{}>", static_cast<int>(dtype));
	}

	inline bool IsValidDataType(DataType dtype)
	{
		return IsValidDataTypeValue(dtype);
	}

	inline bool IsValidUnaryOp(UnaryOp op)
	{
		switch (op)
		{
		case UnaryOp::Negate:
		case UnaryOp::Abs:
		case UnaryOp::Sqrt:
		case UnaryOp::Exp:
		case UnaryOp::Log:
		case UnaryOp::Sin:
		case UnaryOp::Cos:
		case UnaryOp::Tan:
		case UnaryOp::Arcsin:
		case UnaryOp::Arccos:
		case UnaryOp::Arctan:
		case UnaryOp::Transpose:
		case UnaryOp::LogicalNegation:
			return true;
		}
		return false;
	}

	inline bool IsValidBinaryOp(BinaryOp op)
	{
		switch (op)
		{
		case BinaryOp::Add:
		case BinaryOp::Subtract:
		case BinaryOp::Multiply:
		case BinaryOp::Divide:
		case BinaryOp::MatMul:
		case BinaryOp::Pow:
		case BinaryOp::Max:
		case BinaryOp::Min:
		case BinaryOp::Less:
		case BinaryOp::Greater:
		case BinaryOp::Equal:
			return true;
		}
		return false;
	}

	inline bool IsValidReduceOp(ReduceOp op)
	{
		switch (op)
		{
		case ReduceOp::Sum:
		case ReduceOp::Mean:
		case ReduceOp::Max:
			return true;
		}
		return false;
	}

	inline bool IsValidFusionPattern(FusionPattern pattern)
	{
		switch (pattern)
		{
		case FusionPattern::MatMulBiasAdd:
		case FusionPattern::ElementWiseChain:
		case FusionPattern::MatMulBiasAddReLU:
			return true;
		}
		return false;
	}

	inline std::string ShapeToString(std::span<const std::size_t> shape)
	{
		std::string result = "[";
		for (std::size_t i = 0; i < shape.size(); ++i)
		{
			result += std::format("{}{}", i == 0 ? "" : ", ", shape[i]);
		}
		result += "]";
		return result;
	}

	inline bool SameShape(std::span<const std::size_t> lhs, std::span<const std::size_t> rhs)
	{
		return lhs.size() == rhs.size() && std::ranges::equal(lhs, rhs);
	}

	inline std::size_t NumElements(std::span<const std::size_t> shape)
	{
		return std::accumulate(shape.begin(), shape.end(), 1uz, std::multiplies{});
	}

	inline std::string FormatInfo(DataType dtype, std::span<const std::size_t> shape)
	{
		return std::format("{}{}", DataTypeToString(dtype), ShapeToString(shape));
	}

	inline std::string_view NodeKindName(const NodeVariant& node)
	{
		return std::visit(
		    [](const auto& value) -> std::string_view {
			    using T = std::decay_t<decltype(value)>;
			    if constexpr (std::same_as<T, ParamRefNode>)
				    return "ParamRefNode";
			    else if constexpr (std::same_as<T, ConstantNode>)
				    return "ConstantNode";
			    else if constexpr (std::same_as<T, VariableRefNode>)
				    return "VariableRefNode";
			    else if constexpr (std::same_as<T, UnaryOpNode>)
				    return "UnaryOpNode";
			    else if constexpr (std::same_as<T, BinaryOpNode>)
				    return "BinaryOpNode";
			    else if constexpr (std::same_as<T, CallNode>)
				    return "CallNode";
			    else if constexpr (std::same_as<T, CastNode>)
				    return "CastNode";
			    else if constexpr (std::same_as<T, CondNode>)
				    return "CondNode";
			    else if constexpr (std::same_as<T, WhileNode>)
				    return "WhileNode";
			    else if constexpr (std::same_as<T, SaveActivationNode>)
				    return "SaveActivationNode";
			    else if constexpr (std::same_as<T, LoadActivationNode>)
				    return "LoadActivationNode";
			    else if constexpr (std::same_as<T, TapeSaveActivationNode>)
				    return "TapeSaveActivationNode";
			    else if constexpr (std::same_as<T, TapeLoadActivationNode>)
				    return "TapeLoadActivationNode";
			    else if constexpr (std::same_as<T, ReduceOpNode>)
				    return "ReduceOpNode";
			    else if constexpr (std::same_as<T, ReshapeNode>)
				    return "ReshapeNode";
			    else if constexpr (std::same_as<T, ConcatNode>)
				    return "ConcatNode";
			    else if constexpr (std::same_as<T, SliceNode>)
				    return "SliceNode";
			    else if constexpr (std::same_as<T, FusedOpNode>)
				    return "FusedOpNode";
			    else
				    return "UnknownNode";
		    },
		    node);
	}

	class GraphValidator
	{
	public:
		explicit GraphValidator(const Graph& graph) : graph_(graph) {}

		void Validate() const
		{
			if (graph_.SubgraphCount() == 0)
			{
				throw GraphValidationError("Graph validation failed: graph contains no subgraphs");
			}
			if (graph_.Forward() >= graph_.SubgraphCount())
			{
				throw GraphValidationError(std::format(
				    "Graph validation failed: forward subgraph {} is out of range; subgraphCount={}", graph_.Forward(),
				    graph_.SubgraphCount()));
			}
			if (const auto backward = graph_.Backward(); backward && *backward >= graph_.SubgraphCount())
			{
				throw GraphValidationError(std::format(
				    "Graph validation failed: backward subgraph {} is out of range; subgraphCount={}", *backward,
				    graph_.SubgraphCount()));
			}

			ValidateVariables();
			ValidateSlots();
			for (SubgraphId id = 0; id < graph_.SubgraphCount(); ++id)
			{
				ValidateSubgraph(id);
			}
			ValidatePublicSignatureNames();
		}

	private:
		[[noreturn]] static void Fail(std::string message)
		{
			throw GraphValidationError(std::move(message));
		}

		[[noreturn]] static void Fail(SubgraphId subgraphId, std::string message)
		{
			throw GraphValidationError(std::format("Graph validation failed at subgraph {}: {}", subgraphId, message));
		}

		[[noreturn]] static void Fail(SubgraphId subgraphId, NodeId nodeId, std::string message)
		{
			throw GraphValidationError(
			    std::format("Graph validation failed at subgraph {}, node {}: {}", subgraphId, nodeId, message));
		}

		static void ValidateShape(std::span<const std::size_t> shape, std::string_view context)
		{
			if (std::ranges::any_of(shape, [](std::size_t dim) { return dim == 0; }))
			{
				throw GraphValidationError(std::format("{} has a zero dimension: {}", context, ShapeToString(shape)));
			}
		}

		static void ValidateDataType(DataType dtype, std::string_view context)
		{
			if (!IsValidDataType(dtype))
			{
				throw GraphValidationError(
				    std::format("{} has invalid dtype {}", context, static_cast<int>(dtype)));
			}
		}

		static void ValidateOutputInfo(const OutputInfo& info, std::string_view context)
		{
			ValidateDataType(info.dtype, context);
			ValidateShape(info.shape, context);
		}

		static void ExpectInfo(SubgraphId subgraphId, NodeId nodeId, const OutputInfo& actual,
		                       const OutputInfo& expected, std::string_view context)
		{
			if (actual.dtype != expected.dtype || !SameShape(actual.shape, expected.shape))
			{
				Fail(subgraphId, nodeId,
				     std::format("{} expected {}, got {}", context, FormatInfo(expected.dtype, expected.shape),
				                 FormatInfo(actual.dtype, actual.shape)));
			}
		}

		static void ExpectOutputCount(SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                              std::size_t expected)
		{
			if (entry.outputInfos.size() != expected)
			{
				Fail(subgraphId, nodeId,
				     std::format("expected {} output(s), got {}", expected, entry.outputInfos.size()));
			}
		}

		OutputInfo ValidateNodeOutput(const Subgraph& subgraph, SubgraphId subgraphId, NodeId currentNodeId,
		                              NodeOutput output, std::string_view role, bool requirePreviousNode) const
		{
			if (output.node >= subgraph.NodeCount())
			{
				Fail(subgraphId, currentNodeId,
				     std::format("{} references node {}, but nodeCount={}", role, output.node, subgraph.NodeCount()));
			}
			if (requirePreviousNode && output.node >= currentNodeId)
			{
				Fail(subgraphId, currentNodeId,
				     std::format("{} references node {}, which is not before current node", role, output.node));
			}

			const auto& source = subgraph.GetNodeEntry(output.node);
			if (output.port >= source.outputInfos.size())
			{
				Fail(subgraphId, currentNodeId,
				     std::format("{} references node {} port {}, but outputCount={}", role, output.node, output.port,
				                 source.outputInfos.size()));
			}
			const auto info = source.outputInfos[output.port];
			ValidateOutputInfo(info, std::format("subgraph {} node {} output {}", subgraphId, output.node, output.port));
			return info;
		}

		OutputInfo GetSubgraphResultInfo(SubgraphId subgraphId, std::size_t resultIndex) const
		{
			const auto& subgraph = graph_.GetSubgraph(subgraphId);
			if (resultIndex >= subgraph.Results().size())
			{
				Fail(subgraphId, std::format("result {} is out of range", resultIndex));
			}
			return ValidateNodeOutput(subgraph, subgraphId, subgraph.NodeCount(), subgraph.Results()[resultIndex],
			                          "result", false);
		}

		void ValidateVariables() const
		{
			for (std::size_t i = 0; i < graph_.VariableCount(); ++i)
			{
				const auto& variable = graph_.Variables()[i];
				if (!variable)
				{
					Fail(std::format("Graph validation failed: variable {} is null", i));
				}

				const auto& data = variable->Data();
				const auto& grad = variable->Grad();
				ValidateDataType(data.DType(), std::format("variable {} data", i));
				ValidateShape(data.Shape().Dims, std::format("variable {} data", i));
				ValidateDataType(grad.DType(), std::format("variable {} grad", i));
				ValidateShape(grad.Shape().Dims, std::format("variable {} grad", i));
				if (data.DType() != grad.DType() || !SameShape(data.Shape().Dims, grad.Shape().Dims))
				{
					Fail(std::format("Graph validation failed: variable {} grad metadata {} does not match data {}",
					                 i, FormatInfo(grad.DType(), grad.Shape().Dims),
					                 FormatInfo(data.DType(), data.Shape().Dims)));
				}
			}
		}

		void ValidateSlots() const
		{
			for (std::size_t i = 0; i < graph_.ActivationSlotCount(); ++i)
			{
				const auto& slot = graph_.GetActivationSlot(i);
				ValidateDataType(slot.dtype, std::format("activation slot {}", i));
				ValidateShape(slot.shape, std::format("activation slot {}", i));
			}
			for (std::size_t i = 0; i < graph_.TapeSlotCount(); ++i)
			{
				const auto& slot = graph_.GetTapeSlot(i);
				ValidateDataType(slot.dtype, std::format("tape slot {}", i));
				ValidateShape(slot.shape, std::format("tape slot {}", i));
			}
		}

		static void ValidateNameList(std::span<const std::string> names, std::size_t expectedCount,
		                             std::string_view role)
		{
			if (names.empty())
			{
				return;
			}
			if (names.size() != expectedCount)
			{
				Fail(std::format("Graph validation failed: {} name count {} does not match expected count {}", role,
				                 names.size(), expectedCount));
			}
			for (std::size_t i = 0; i < names.size(); ++i)
			{
				if (names[i].empty())
				{
					Fail(std::format("Graph validation failed: {} name {} is empty", role, i));
				}
				for (std::size_t j = i + 1; j < names.size(); ++j)
				{
					if (names[i] == names[j])
					{
						Fail(std::format("Graph validation failed: duplicate {} name '{}'", role, names[i]));
					}
				}
			}
		}

		void ValidatePublicSignatureNames() const
		{
			const auto& forward = graph_.GetSubgraph(graph_.Forward());
			ValidateNameList(graph_.InputNames(), forward.Params().size(), "input");
			ValidateNameList(graph_.OutputNames(), forward.Results().size(), "output");
		}

		void ValidateSubgraph(SubgraphId subgraphId) const
		{
			const auto& subgraph = graph_.GetSubgraph(subgraphId);
			for (std::size_t i = 0; i < subgraph.Params().size(); ++i)
			{
				const auto& param = subgraph.Params()[i];
				ValidateDataType(param.dtype, std::format("subgraph {} param {}", subgraphId, i));
				ValidateShape(param.shape, std::format("subgraph {} param {}", subgraphId, i));
			}

			for (NodeId nodeId = 0; nodeId < subgraph.NodeCount(); ++nodeId)
			{
				const auto& entry = subgraph.GetNodeEntry(nodeId);
				try
				{
					for (std::size_t port = 0; port < entry.outputInfos.size(); ++port)
					{
						ValidateOutputInfo(entry.outputInfos[port],
						                   std::format("subgraph {} node {} output {}", subgraphId, nodeId, port));
					}

					std::visit([&](const auto& node) { ValidateNode(subgraph, subgraphId, nodeId, entry, node); },
					           entry.node);
				}
				catch (const GraphValidationError& ex)
				{
					throw GraphValidationError(
					    std::format("{} (nodeKind={})", ex.what(), NodeKindName(entry.node)));
				}
			}

			for (std::size_t i = 0; i < subgraph.Results().size(); ++i)
			{
				(void)ValidateNodeOutput(subgraph, subgraphId, subgraph.NodeCount(), subgraph.Results()[i],
				                         std::format("result {}", i), false);
			}
		}

		void ValidateNode(const Subgraph& subgraph, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const ParamRefNode& node) const
		{
			ExpectOutputCount(subgraphId, nodeId, entry, 1);
			if (node.paramIndex >= subgraph.Params().size())
			{
				Fail(subgraphId, nodeId,
				     std::format("ParamRefNode references param {}, but paramCount={}", node.paramIndex,
				                 subgraph.Params().size()));
			}
			const auto& param = subgraph.Params()[node.paramIndex];
			ExpectInfo(subgraphId, nodeId, entry.outputInfos[0], { param.dtype, param.shape }, "ParamRefNode output");
		}

		void ValidateNode(const Subgraph&, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const ConstantNode& node) const
		{
			ExpectOutputCount(subgraphId, nodeId, entry, 1);
			ExpectInfo(subgraphId, nodeId, entry.outputInfos[0],
			           { node.value.DType(), node.value.Shape().ToOwned() }, "ConstantNode output");
		}

		void ValidateNode(const Subgraph&, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const VariableRefNode& node) const
		{
			ExpectOutputCount(subgraphId, nodeId, entry, 1);
			if (node.variableIndex >= graph_.VariableCount())
			{
				Fail(subgraphId, nodeId,
				     std::format("VariableRefNode references variable {}, but variableCount={}", node.variableIndex,
				                 graph_.VariableCount()));
			}
			const auto& data = graph_.GetVariable(node.variableIndex)->Data();
			ExpectInfo(subgraphId, nodeId, entry.outputInfos[0], { data.DType(), data.Shape().ToOwned() },
			           "VariableRefNode output");
		}

		void ValidateNode(const Subgraph& subgraph, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const UnaryOpNode& node) const
		{
			ExpectOutputCount(subgraphId, nodeId, entry, 1);
			if (!IsValidUnaryOp(node.op))
			{
				Fail(subgraphId, nodeId, std::format("invalid UnaryOp {}", static_cast<int>(node.op)));
			}
			const auto input = ValidateNodeOutput(subgraph, subgraphId, nodeId, node.input, "UnaryOp input", true);
			EnumDispatch(node.op, [&]<UnaryOp OpValue> {
				using Traits = UnaryOpTraits<OpValue>;
				const auto dtype = Traits::ResultType(input.dtype);
				const auto shape = Traits::ResultShape(input.shape);
				if (!dtype)
				{
					Fail(subgraphId, nodeId, std::format("UnaryOp dtype error: {}", dtype.error()));
				}
				if (!shape)
				{
					Fail(subgraphId, nodeId, std::format("UnaryOp shape error: {}", shape.error()));
				}
				ExpectInfo(subgraphId, nodeId, entry.outputInfos[0], { *dtype, *shape }, "UnaryOp output");
			});
		}

		void ValidateNode(const Subgraph& subgraph, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const BinaryOpNode& node) const
		{
			ExpectOutputCount(subgraphId, nodeId, entry, 1);
			if (!IsValidBinaryOp(node.op))
			{
				Fail(subgraphId, nodeId, std::format("invalid BinaryOp {}", static_cast<int>(node.op)));
			}
			const auto lhs = ValidateNodeOutput(subgraph, subgraphId, nodeId, node.lhs, "BinaryOp lhs", true);
			const auto rhs = ValidateNodeOutput(subgraph, subgraphId, nodeId, node.rhs, "BinaryOp rhs", true);
			EnumDispatch(node.op, [&]<BinaryOp OpValue> {
				using Traits = BinaryOpTraits<OpValue>;
				const auto dtype = Traits::ResultType(lhs.dtype, rhs.dtype);
				const auto shape = Traits::ResultShape(lhs.shape, rhs.shape);
				if (!dtype)
				{
					Fail(subgraphId, nodeId, std::format("BinaryOp dtype error: {}", dtype.error()));
				}
				if (!shape)
				{
					Fail(subgraphId, nodeId, std::format("BinaryOp shape error: {}", shape.error()));
				}
				ExpectInfo(subgraphId, nodeId, entry.outputInfos[0], { *dtype, *shape }, "BinaryOp output");
			});
		}

		void ValidateNode(const Subgraph& subgraph, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const CallNode& node) const
		{
			if (node.callee >= graph_.SubgraphCount())
			{
				Fail(subgraphId, nodeId,
				     std::format("CallNode callee {} is out of range; subgraphCount={}", node.callee,
				                 graph_.SubgraphCount()));
			}
			ValidateArgsAgainstParams(subgraph, subgraphId, nodeId, node.args, node.callee, "CallNode");
			ValidateOutputsAgainstResults(subgraphId, nodeId, entry, node.callee, "CallNode");
		}

		void ValidateNode(const Subgraph& subgraph, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const CastNode& node) const
		{
			ExpectOutputCount(subgraphId, nodeId, entry, 1);
			ValidateDataType(node.targetType, std::format("subgraph {} node {} CastNode target", subgraphId, nodeId));
			const auto input = ValidateNodeOutput(subgraph, subgraphId, nodeId, node.input, "CastNode input", true);
			ExpectInfo(subgraphId, nodeId, entry.outputInfos[0], { node.targetType, input.shape }, "CastNode output");
		}

		void ValidateNode(const Subgraph& subgraph, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const CondNode& node) const
		{
			const auto cond = ValidateNodeOutput(subgraph, subgraphId, nodeId, node.condition, "CondNode condition", true);
			if (cond.dtype != DataType::Bool || NumElements(cond.shape) != 1)
			{
				Fail(subgraphId, nodeId,
				     std::format("CondNode condition must be Bool with one element, got {}",
				                 FormatInfo(cond.dtype, cond.shape)));
			}
			if (node.thenBranch >= graph_.SubgraphCount() || node.elseBranch >= graph_.SubgraphCount())
			{
				Fail(subgraphId, nodeId,
				     std::format("CondNode branch out of range: then={}, else={}, subgraphCount={}", node.thenBranch,
				                 node.elseBranch, graph_.SubgraphCount()));
			}
			ValidateSameSubgraphSignature(subgraphId, nodeId, node.thenBranch, node.elseBranch, "CondNode branches");
			ValidateArgsAgainstParams(subgraph, subgraphId, nodeId, node.args, node.thenBranch, "CondNode");
			ValidateArgsAgainstParams(subgraph, subgraphId, nodeId, node.args, node.elseBranch, "CondNode");
			ValidateOutputsAgainstResults(subgraphId, nodeId, entry, node.thenBranch, "CondNode");
		}

		void ValidateNode(const Subgraph& subgraph, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const WhileNode& node) const
		{
			if (node.condBranch >= graph_.SubgraphCount() || node.bodyBranch >= graph_.SubgraphCount())
			{
				Fail(subgraphId, nodeId,
				     std::format("WhileNode branch out of range: cond={}, body={}, subgraphCount={}", node.condBranch,
				                 node.bodyBranch, graph_.SubgraphCount()));
			}
			if (entry.outputInfos.size() != node.initArgs.size())
			{
				Fail(subgraphId, nodeId,
				     std::format("WhileNode output count {} must equal init arg count {}", entry.outputInfos.size(),
				                 node.initArgs.size()));
			}
			const auto initInfos = ValidateInputs(subgraph, subgraphId, nodeId, node.initArgs, "WhileNode init");
			ValidateBranchParamsMatchInfos(subgraphId, nodeId, node.condBranch, initInfos, "WhileNode condBranch");
			ValidateBranchParamsMatchInfos(subgraphId, nodeId, node.bodyBranch, initInfos, "WhileNode bodyBranch");

			const auto& condBranch = graph_.GetSubgraph(node.condBranch);
			if (condBranch.Results().size() != 1)
			{
				Fail(subgraphId, nodeId,
				     std::format("WhileNode condBranch must return one result, got {}", condBranch.Results().size()));
			}
			const auto condResult = GetSubgraphResultInfo(node.condBranch, 0);
			if (condResult.dtype != DataType::Bool || NumElements(condResult.shape) != 1)
			{
				Fail(subgraphId, nodeId,
				     std::format("WhileNode condBranch result must be Bool with one element, got {}",
				                 FormatInfo(condResult.dtype, condResult.shape)));
			}

			const auto& bodyBranch = graph_.GetSubgraph(node.bodyBranch);
			if (bodyBranch.Results().size() != initInfos.size())
			{
				Fail(subgraphId, nodeId,
				     std::format("WhileNode bodyBranch result count {} must equal init arg count {}",
				                 bodyBranch.Results().size(), initInfos.size()));
			}
			for (std::size_t i = 0; i < initInfos.size(); ++i)
			{
				const auto bodyResult = GetSubgraphResultInfo(node.bodyBranch, i);
				ExpectInfo(subgraphId, nodeId, bodyResult, initInfos[i], "WhileNode bodyBranch result");
				ExpectInfo(subgraphId, nodeId, entry.outputInfos[i], initInfos[i], "WhileNode output");
			}
		}

		void ValidateNode(const Subgraph& subgraph, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const SaveActivationNode& node) const
		{
			ExpectOutputCount(subgraphId, nodeId, entry, 1);
			if (node.slotId >= graph_.ActivationSlotCount())
			{
				Fail(subgraphId, nodeId,
				     std::format("SaveActivationNode slot {} out of range; activationSlotCount={}", node.slotId,
				                 graph_.ActivationSlotCount()));
			}
			const auto input = ValidateNodeOutput(subgraph, subgraphId, nodeId, node.input, "SaveActivationNode input", true);
			const auto& slot = graph_.GetActivationSlot(node.slotId);
			ExpectInfo(subgraphId, nodeId, { slot.dtype, slot.shape }, input, "SaveActivationNode slot");
			ExpectInfo(subgraphId, nodeId, entry.outputInfos[0], input, "SaveActivationNode output");
		}

		void ValidateNode(const Subgraph&, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const LoadActivationNode& node) const
		{
			ExpectOutputCount(subgraphId, nodeId, entry, 1);
			if (node.slotId >= graph_.ActivationSlotCount())
			{
				Fail(subgraphId, nodeId,
				     std::format("LoadActivationNode slot {} out of range; activationSlotCount={}", node.slotId,
				                 graph_.ActivationSlotCount()));
			}
			const auto& slot = graph_.GetActivationSlot(node.slotId);
			ExpectInfo(subgraphId, nodeId, entry.outputInfos[0], { slot.dtype, slot.shape }, "LoadActivationNode output");
		}

		void ValidateNode(const Subgraph& subgraph, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const TapeSaveActivationNode& node) const
		{
			ExpectOutputCount(subgraphId, nodeId, entry, 1);
			if (node.tapeSlotId >= graph_.TapeSlotCount())
			{
				Fail(subgraphId, nodeId,
				     std::format("TapeSaveActivationNode slot {} out of range; tapeSlotCount={}", node.tapeSlotId,
				                 graph_.TapeSlotCount()));
			}
			const auto input =
			    ValidateNodeOutput(subgraph, subgraphId, nodeId, node.input, "TapeSaveActivationNode input", true);
			const auto& slot = graph_.GetTapeSlot(node.tapeSlotId);
			ExpectInfo(subgraphId, nodeId, { slot.dtype, slot.shape }, input, "TapeSaveActivationNode slot");
			ExpectInfo(subgraphId, nodeId, entry.outputInfos[0], input, "TapeSaveActivationNode output");
		}

		void ValidateNode(const Subgraph&, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const TapeLoadActivationNode& node) const
		{
			ExpectOutputCount(subgraphId, nodeId, entry, 1);
			if (node.tapeSlotId >= graph_.TapeSlotCount())
			{
				Fail(subgraphId, nodeId,
				     std::format("TapeLoadActivationNode slot {} out of range; tapeSlotCount={}", node.tapeSlotId,
				                 graph_.TapeSlotCount()));
			}
			const auto& slot = graph_.GetTapeSlot(node.tapeSlotId);
			ExpectInfo(subgraphId, nodeId, entry.outputInfos[0], { slot.dtype, slot.shape }, "TapeLoadActivationNode output");
		}

		void ValidateNode(const Subgraph& subgraph, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const ReduceOpNode& node) const
		{
			ExpectOutputCount(subgraphId, nodeId, entry, 1);
			if (!IsValidReduceOp(node.op))
			{
				Fail(subgraphId, nodeId, std::format("invalid ReduceOp {}", static_cast<int>(node.op)));
			}
			const auto input = ValidateNodeOutput(subgraph, subgraphId, nodeId, node.input, "ReduceOp input", true);
			EnumDispatch(node.op, [&]<ReduceOp OpValue> {
				using Traits = ReduceOpTraits<OpValue>;
				const auto dtype = Traits::ResultType(input.dtype);
				const auto shape = Traits::ResultShape(input.shape, node.axis);
				if (!dtype)
				{
					Fail(subgraphId, nodeId, std::format("ReduceOp dtype error: {}", dtype.error()));
				}
				if (!shape)
				{
					Fail(subgraphId, nodeId, std::format("ReduceOp shape error: {}", shape.error()));
				}
				ExpectInfo(subgraphId, nodeId, entry.outputInfos[0], { *dtype, *shape }, "ReduceOp output");
			});
		}

		void ValidateNode(const Subgraph& subgraph, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const ReshapeNode& node) const
		{
			ExpectOutputCount(subgraphId, nodeId, entry, 1);
			ValidateShape(node.targetShape, std::format("subgraph {} node {} ReshapeNode target", subgraphId, nodeId));
			const auto input = ValidateNodeOutput(subgraph, subgraphId, nodeId, node.input, "ReshapeNode input", true);
			if (NumElements(input.shape) != NumElements(node.targetShape))
			{
				Fail(subgraphId, nodeId,
				     std::format("ReshapeNode element count mismatch: input {}, target {}",
				                 ShapeToString(input.shape), ShapeToString(node.targetShape)));
			}
			ExpectInfo(subgraphId, nodeId, entry.outputInfos[0], { input.dtype, node.targetShape },
			           "ReshapeNode output");
		}

		void ValidateNode(const Subgraph& subgraph, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const ConcatNode& node) const
		{
			ExpectOutputCount(subgraphId, nodeId, entry, 1);
			if (node.inputs.empty())
			{
				Fail(subgraphId, nodeId, "ConcatNode requires at least one input");
			}
			const auto inputs = ValidateInputs(subgraph, subgraphId, nodeId, node.inputs, "ConcatNode input");
			const auto rank = inputs[0].shape.size();
			if (node.axis >= rank)
			{
				Fail(subgraphId, nodeId, std::format("ConcatNode axis {} out of range for rank {}", node.axis, rank));
			}
			auto outputShape = inputs[0].shape;
			outputShape[node.axis] = 0;
			for (std::size_t i = 0; i < inputs.size(); ++i)
			{
				if (inputs[i].dtype != inputs[0].dtype || inputs[i].shape.size() != rank)
				{
					Fail(subgraphId, nodeId, std::format("ConcatNode input {} metadata does not match first input", i));
				}
				for (std::size_t dim = 0; dim < rank; ++dim)
				{
					if (dim != node.axis && inputs[i].shape[dim] != inputs[0].shape[dim])
					{
						Fail(subgraphId, nodeId,
						     std::format("ConcatNode input {} dim {} mismatch: expected {}, got {}", i, dim,
						                 inputs[0].shape[dim], inputs[i].shape[dim]));
					}
				}
				outputShape[node.axis] += inputs[i].shape[node.axis];
			}
			ExpectInfo(subgraphId, nodeId, entry.outputInfos[0], { inputs[0].dtype, outputShape }, "ConcatNode output");
		}

		void ValidateNode(const Subgraph& subgraph, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const SliceNode& node) const
		{
			ExpectOutputCount(subgraphId, nodeId, entry, 1);
			const auto input = ValidateNodeOutput(subgraph, subgraphId, nodeId, node.input, "SliceNode input", true);
			if (node.axis >= input.shape.size())
			{
				Fail(subgraphId, nodeId,
				     std::format("SliceNode axis {} out of range for rank {}", node.axis, input.shape.size()));
			}
			if (node.length == 0 || node.start > input.shape[node.axis] ||
			    node.length > input.shape[node.axis] - node.start)
			{
				Fail(subgraphId, nodeId,
				     std::format("SliceNode range [{}, {}) out of bounds for axis {} with size {}", node.start,
				                 node.start + node.length, node.axis, input.shape[node.axis]));
			}
			auto outputShape = input.shape;
			outputShape[node.axis] = node.length;
			ExpectInfo(subgraphId, nodeId, entry.outputInfos[0], { input.dtype, outputShape }, "SliceNode output");
		}

		void ValidateNode(const Subgraph& subgraph, SubgraphId subgraphId, NodeId nodeId, const NodeEntry& entry,
		                  const FusedOpNode& node) const
		{
			if (!IsValidFusionPattern(node.pattern))
			{
				Fail(subgraphId, nodeId, std::format("invalid FusionPattern {}", static_cast<int>(node.pattern)));
			}
			if (node.body >= graph_.SubgraphCount())
			{
				Fail(subgraphId, nodeId,
				     std::format("FusedOpNode body {} is out of range; subgraphCount={}", node.body,
				                 graph_.SubgraphCount()));
			}
			ValidateArgsAgainstParams(subgraph, subgraphId, nodeId, node.args, node.body, "FusedOpNode");
			ValidateOutputsAgainstResults(subgraphId, nodeId, entry, node.body, "FusedOpNode");
		}

		std::vector<OutputInfo> ValidateInputs(const Subgraph& subgraph, SubgraphId subgraphId, NodeId nodeId,
		                                       std::span<const NodeOutput> inputs, std::string_view role) const
		{
			std::vector<OutputInfo> infos;
			infos.reserve(inputs.size());
			for (std::size_t i = 0; i < inputs.size(); ++i)
			{
				infos.push_back(ValidateNodeOutput(subgraph, subgraphId, nodeId, inputs[i],
				                                   std::format("{} {}", role, i), true));
			}
			return infos;
		}

		void ValidateArgsAgainstParams(const Subgraph& caller, SubgraphId callerId, NodeId nodeId,
		                               std::span<const NodeOutput> args, SubgraphId calleeId,
		                               std::string_view context) const
		{
			const auto& callee = graph_.GetSubgraph(calleeId);
			if (args.size() != callee.Params().size())
			{
				Fail(callerId, nodeId,
				     std::format("{} arg count {} does not match callee {} param count {}", context, args.size(),
				                 calleeId, callee.Params().size()));
			}
			for (std::size_t i = 0; i < args.size(); ++i)
			{
				const auto arg = ValidateNodeOutput(caller, callerId, nodeId, args[i],
				                                    std::format("{} arg {}", context, i), true);
				const auto& param = callee.Params()[i];
				ExpectInfo(callerId, nodeId, arg, { param.dtype, param.shape },
				           std::format("{} arg {} for callee {}", context, i, calleeId));
			}
		}

		void ValidateOutputsAgainstResults(SubgraphId callerId, NodeId nodeId, const NodeEntry& entry,
		                                   SubgraphId calleeId, std::string_view context) const
		{
			const auto& callee = graph_.GetSubgraph(calleeId);
			if (entry.outputInfos.size() != callee.Results().size())
			{
				Fail(callerId, nodeId,
				     std::format("{} output count {} does not match callee {} result count {}", context,
				                 entry.outputInfos.size(), calleeId, callee.Results().size()));
			}
			for (std::size_t i = 0; i < entry.outputInfos.size(); ++i)
			{
				ExpectInfo(callerId, nodeId, entry.outputInfos[i], GetSubgraphResultInfo(calleeId, i),
				           std::format("{} output {} from callee {}", context, i, calleeId));
			}
		}

		void ValidateSameSubgraphSignature(SubgraphId ownerId, NodeId nodeId, SubgraphId lhsId, SubgraphId rhsId,
		                                   std::string_view context) const
		{
			const auto& lhs = graph_.GetSubgraph(lhsId);
			const auto& rhs = graph_.GetSubgraph(rhsId);
			if (lhs.Params().size() != rhs.Params().size() || lhs.Results().size() != rhs.Results().size())
			{
				Fail(ownerId, nodeId, std::format("{} have different param/result counts", context));
			}
			for (std::size_t i = 0; i < lhs.Params().size(); ++i)
			{
				const OutputInfo lhsParam{ lhs.Params()[i].dtype, lhs.Params()[i].shape };
				const OutputInfo rhsParam{ rhs.Params()[i].dtype, rhs.Params()[i].shape };
				ExpectInfo(ownerId, nodeId, lhsParam, rhsParam, std::format("{} param {}", context, i));
			}
			for (std::size_t i = 0; i < lhs.Results().size(); ++i)
			{
				ExpectInfo(ownerId, nodeId, GetSubgraphResultInfo(lhsId, i), GetSubgraphResultInfo(rhsId, i),
				           std::format("{} result {}", context, i));
			}
		}

		void ValidateBranchParamsMatchInfos(SubgraphId ownerId, NodeId nodeId, SubgraphId branchId,
		                                    std::span<const OutputInfo> infos, std::string_view context) const
		{
			const auto& branch = graph_.GetSubgraph(branchId);
			if (branch.Params().size() != infos.size())
			{
				Fail(ownerId, nodeId,
				     std::format("{} param count {} does not match expected count {}", context,
				                 branch.Params().size(), infos.size()));
			}
			for (std::size_t i = 0; i < infos.size(); ++i)
			{
				const OutputInfo param{ branch.Params()[i].dtype, branch.Params()[i].shape };
				ExpectInfo(ownerId, nodeId, param, infos[i], std::format("{} param {}", context, i));
			}
		}

		const Graph& graph_;
	};

	inline void ValidateGraph(const Graph& graph)
	{
		GraphValidator{ graph }.Validate();
	}
} // namespace LiteNN::Validation

#endif
