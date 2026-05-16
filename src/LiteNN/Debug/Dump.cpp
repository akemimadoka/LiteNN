#ifndef LITENN_MODULE_IMPL
#include <LiteNN/Debug/Dump.h>

#include <LiteNN/Validation/GraphValidator.h>

#include <format>
#include <string_view>
#endif

namespace LiteNN::Debug
{
namespace
{
	template <typename Formatter>
	std::string JoinIndexed(std::size_t count, std::string_view separator, Formatter&& formatter)
	{
		std::string result;
		for (std::size_t index = 0; index < count; ++index)
		{
			if (index != 0)
			{
				result += separator;
			}
			result += formatter(index);
		}
		return result;
	}

	template <typename Range, typename Formatter>
	std::string JoinMapped(const Range& range, std::string_view separator, Formatter&& formatter)
	{
		std::string result;
		bool first = true;
		for (const auto& value : range)
		{
			if (!first)
			{
				result += separator;
			}
			first = false;
			result += formatter(value);
		}
		return result;
	}

	std::string FormatValueRef(NodeOutput output)
	{
		if (output.port == 0)
		{
			return std::format("%{}", output.node);
		}
		return std::format("%{}#{}", output.node, output.port);
	}

	std::string FormatInfo(DataType dtype, std::span<const std::size_t> shape)
	{
		return Validation::FormatInfo(dtype, shape);
	}

	std::string FormatInfo(const OutputInfo& info)
	{
		return FormatInfo(info.dtype, info.shape);
	}

	std::string FormatInfo(const SubgraphParam& param)
	{
		return FormatInfo(param.dtype, param.shape);
	}

	std::string FormatInfo(const ActivationSlot& slot)
	{
		return FormatInfo(slot.dtype, slot.shape);
	}

	std::string FormatInfo(const TapeSlot& slot)
	{
		return FormatInfo(slot.dtype, slot.shape);
	}

	std::string FormatInfo(const NamedTensorSpec& spec)
	{
		return std::format("{}: {}", spec.name, FormatInfo(spec.dtype, spec.shape));
	}

	std::string UnaryOpToString(UnaryOp op)
	{
		switch (op)
		{
		case UnaryOp::Negate:
			return "UnaryOp::Negate";
		case UnaryOp::Abs:
			return "UnaryOp::Abs";
		case UnaryOp::Sqrt:
			return "UnaryOp::Sqrt";
		case UnaryOp::Exp:
			return "UnaryOp::Exp";
		case UnaryOp::Log:
			return "UnaryOp::Log";
		case UnaryOp::Sin:
			return "UnaryOp::Sin";
		case UnaryOp::Cos:
			return "UnaryOp::Cos";
		case UnaryOp::Tan:
			return "UnaryOp::Tan";
		case UnaryOp::Arcsin:
			return "UnaryOp::Arcsin";
		case UnaryOp::Arccos:
			return "UnaryOp::Arccos";
		case UnaryOp::Arctan:
			return "UnaryOp::Arctan";
		case UnaryOp::Transpose:
			return "UnaryOp::Transpose";
		case UnaryOp::LogicalNegation:
			return "UnaryOp::LogicalNegation";
		}
		return std::format("UnaryOp::<invalid:{}>", static_cast<int>(op));
	}

	std::string BinaryOpToString(BinaryOp op)
	{
		switch (op)
		{
		case BinaryOp::Add:
			return "BinaryOp::Add";
		case BinaryOp::Subtract:
			return "BinaryOp::Subtract";
		case BinaryOp::Multiply:
			return "BinaryOp::Multiply";
		case BinaryOp::Divide:
			return "BinaryOp::Divide";
		case BinaryOp::MatMul:
			return "BinaryOp::MatMul";
		case BinaryOp::Pow:
			return "BinaryOp::Pow";
		case BinaryOp::Max:
			return "BinaryOp::Max";
		case BinaryOp::Min:
			return "BinaryOp::Min";
		case BinaryOp::Less:
			return "BinaryOp::Less";
		case BinaryOp::Greater:
			return "BinaryOp::Greater";
		case BinaryOp::Equal:
			return "BinaryOp::Equal";
		}
		return std::format("BinaryOp::<invalid:{}>", static_cast<int>(op));
	}

	std::string ReduceOpToString(ReduceOp op)
	{
		switch (op)
		{
		case ReduceOp::Sum:
			return "ReduceOp::Sum";
		case ReduceOp::Mean:
			return "ReduceOp::Mean";
		case ReduceOp::Max:
			return "ReduceOp::Max";
		}
		return std::format("ReduceOp::<invalid:{}>", static_cast<int>(op));
	}

	std::string FusionPatternToString(FusionPattern pattern)
	{
		switch (pattern)
		{
		case FusionPattern::MatMulBiasAdd:
			return "FusionPattern::MatMulBiasAdd";
		case FusionPattern::ElementWiseChain:
			return "FusionPattern::ElementWiseChain";
		case FusionPattern::MatMulBiasAddReLU:
			return "FusionPattern::MatMulBiasAddReLU";
		}
		return std::format("FusionPattern::<invalid:{}>", static_cast<int>(pattern));
	}

	std::string FormatTensorSummary(const Tensor<PolymorphicDevice>& tensor, const GraphDumpOptions& options)
	{
		if (!options.includeConstantValues)
		{
			return std::format("<{} elements elided>", tensor.NumElements());
		}

		if (tensor.NumElements() > options.maxConstantElements)
		{
			return std::format("<{} elements elided>", tensor.NumElements());
		}

		return std::format("{}", tensor);
	}

	std::string FormatNodeOutputs(NodeId nodeId, std::span<const OutputInfo> outputInfos)
	{
		if (outputInfos.size() == 1)
		{
			return std::format("{}: {}", FormatValueRef({ nodeId, 0 }), FormatInfo(outputInfos.front()));
		}

		return std::format("({})",
		                   JoinIndexed(outputInfos.size(), ", ", [&](std::size_t port) {
			                   return std::format("{}: {}", FormatValueRef({ nodeId, port }),
			                                      FormatInfo(outputInfos[port]));
		                   }));
	}

	std::string FormatNodeArgs(std::span<const NodeOutput> outputs)
	{
		return std::format("[{}]", JoinMapped(outputs, ", ", [](NodeOutput output) { return FormatValueRef(output); }));
	}

	std::string FormatNodePayload(const NodeVariant& node, const GraphDumpOptions& options)
	{
		return std::visit(
		    [&](const auto& value) -> std::string {
				using T = std::decay_t<decltype(value)>;
				if constexpr (std::same_as<T, ParamRefNode>)
				{
					return std::format("ParamRefNode(param={})", value.paramIndex);
				}
				else if constexpr (std::same_as<T, ConstantNode>)
				{
					return std::format("ConstantNode(value={})", FormatTensorSummary(value.value, options));
				}
				else if constexpr (std::same_as<T, QuantizedConstantNode>)
				{
					return std::format("QuantizedConstantNode(storage={}, scheme={}, format={})",
					                   FormatTensorSummary(value.storage, options),
					                   QuantizationSchemeName(value.params.scheme),
					                   QuantizedBlockFormatName(value.params.blockFormat));
				}
				else if constexpr (std::same_as<T, VariableRefNode>)
				{
					return std::format("VariableRefNode(variable={})", value.variableIndex);
				}
				else if constexpr (std::same_as<T, UnaryOpNode>)
				{
					return std::format("UnaryOpNode(op={}, input={})", UnaryOpToString(value.op),
					                   FormatValueRef(value.input));
				}
				else if constexpr (std::same_as<T, BinaryOpNode>)
				{
					return std::format("BinaryOpNode(op={}, lhs={}, rhs={})", BinaryOpToString(value.op),
					                   FormatValueRef(value.lhs), FormatValueRef(value.rhs));
				}
				else if constexpr (std::same_as<T, CallNode>)
				{
					return std::format("CallNode(callee=@{}, args={})", value.callee, FormatNodeArgs(value.args));
				}
				else if constexpr (std::same_as<T, CastNode>)
				{
					return std::format("CastNode(input={}, targetType={})", FormatValueRef(value.input),
					                   Validation::DataTypeToString(value.targetType));
				}
				else if constexpr (std::same_as<T, QuantizeNode>)
				{
					return std::format("QuantizeNode(input={}, storageType={}, granularity={})",
					                   FormatValueRef(value.input),
					                   Validation::DataTypeToString(value.params.storageType),
					                   QuantizationGranularityName(value.params.granularity));
				}
				else if constexpr (std::same_as<T, DequantizeNode>)
				{
					return std::format("DequantizeNode(input={}, targetType={}, scheme={}, format={})",
					                   FormatValueRef(value.input), Validation::DataTypeToString(value.targetType),
					                   QuantizationSchemeName(value.params.scheme),
					                   QuantizedBlockFormatName(value.params.blockFormat));
				}
				else if constexpr (std::same_as<T, CondNode>)
				{
					return std::format("CondNode(condition={}, then=@{}, else=@{}, args={})",
					                   FormatValueRef(value.condition), value.thenBranch, value.elseBranch,
					                   FormatNodeArgs(value.args));
				}
				else if constexpr (std::same_as<T, WhileNode>)
				{
					return std::format("WhileNode(cond=@{}, body=@{}, initArgs={})", value.condBranch,
					                   value.bodyBranch, FormatNodeArgs(value.initArgs));
				}
				else if constexpr (std::same_as<T, SaveActivationNode>)
				{
					return std::format("SaveActivationNode(input={}, slot={})", FormatValueRef(value.input), value.slotId);
				}
				else if constexpr (std::same_as<T, LoadActivationNode>)
				{
					return std::format("LoadActivationNode(slot={})", value.slotId);
				}
				else if constexpr (std::same_as<T, TapeSaveActivationNode>)
				{
					return std::format("TapeSaveActivationNode(input={}, tapeSlot={})", FormatValueRef(value.input),
					                   value.tapeSlotId);
				}
				else if constexpr (std::same_as<T, TapeLoadActivationNode>)
				{
					return std::format("TapeLoadActivationNode(tapeSlot={})", value.tapeSlotId);
				}
				else if constexpr (std::same_as<T, ReduceOpNode>)
				{
					return std::format("ReduceOpNode(op={}, input={}, axis={})", ReduceOpToString(value.op),
					                   FormatValueRef(value.input), value.axis);
				}
				else if constexpr (std::same_as<T, ReshapeNode>)
				{
					return std::format("ReshapeNode(input={}, targetShape={})", FormatValueRef(value.input),
					                   Validation::ShapeToString(value.targetShape));
				}
				else if constexpr (std::same_as<T, ConcatNode>)
				{
					return std::format("ConcatNode(inputs={}, axis={})", FormatNodeArgs(value.inputs), value.axis);
				}
				else if constexpr (std::same_as<T, SliceNode>)
				{
					return std::format("SliceNode(input={}, axis={}, start={}, length={})", FormatValueRef(value.input),
					                   value.axis, value.start, value.length);
				}
				else if constexpr (std::same_as<T, FusedOpNode>)
				{
					return std::format("FusedOpNode(pattern={}, body=@{}, args={})",
					                   FusionPatternToString(value.pattern), value.body, FormatNodeArgs(value.args));
				}
				else
				{
					return std::format("{}", Validation::NodeKindName(node));
				}
		    },
		    node);
	}

	std::string FormatForwardSignatures(const Graph& graph)
	{
		if (graph.SubgraphCount() == 0 || graph.Forward() >= graph.SubgraphCount())
		{
			return "<unavailable>";
		}

		const auto inputSignature = graph.InputSignature();
		const auto outputSignature = graph.OutputSignature();
		return std::format("inputs = [{}]\n  outputs = [{}]",
		                   JoinMapped(inputSignature, ", ", [](const NamedTensorSpec& spec) { return FormatInfo(spec); }),
		                   JoinMapped(outputSignature, ", ", [](const NamedTensorSpec& spec) { return FormatInfo(spec); }));
	}

	std::string FormatSubgraphResults(const Graph& graph, SubgraphId subgraphId, const Subgraph& subgraph)
	{
		return std::format(
		    "[{}]",
		    JoinIndexed(subgraph.Results().size(), ", ", [&](std::size_t index) {
			    const auto result = subgraph.Results()[index];
			    const auto& info = subgraph.GetOutputInfo(result);
			    const auto name = subgraphId == graph.Forward() ? graph.OutputName(index) : std::format("result{}", index);
			    return std::format("{}={}: {}", name, FormatValueRef(result), FormatInfo(info));
		    }));
	}

	std::string FormatSubgraphTags(const Graph& graph, SubgraphId subgraphId)
	{
		std::string tags;
		if (subgraphId == graph.Forward())
		{
			tags = "forward";
		}
		if (graph.Backward() && subgraphId == *graph.Backward())
		{
			if (!tags.empty())
			{
				tags += ", ";
			}
			tags += "backward";
		}
		return tags.empty() ? "" : std::format(" [{}]", tags);
	}
} // namespace

	std::string DumpGraph(const Graph& graph, const GraphDumpOptions& options)
	{
		std::string out = "graph {\n";
		out += std::format("  subgraph_count = {}\n", graph.SubgraphCount());
		out += std::format("  variable_count = {}\n", graph.VariableCount());
		out += std::format("  activation_slot_count = {}\n", graph.ActivationSlotCount());
		out += std::format("  tape_slot_count = {}\n", graph.TapeSlotCount());
		out += std::format("  forward = @{}\n", graph.Forward());
		out += std::format("  backward = {}\n",
		                   graph.Backward() ? std::format("@{}", *graph.Backward()) : std::string("none"));

		const auto signatures = FormatForwardSignatures(graph);
		if (signatures == "<unavailable>")
		{
			out += "  inputs = <unavailable>\n";
			out += "  outputs = <unavailable>\n";
		}
		else
		{
			out += "  ";
			out += signatures;
			out += "\n";
		}

		out += std::format("  variables = [{}]\n", JoinIndexed(graph.VariableCount(), ", ", [&](std::size_t index) {
			const auto& variable = graph.GetVariable(index)->Data();
			return std::format("var{}: {}", index, FormatInfo(variable.DType(), variable.Shape().Dims));
		}));
		out += std::format("  activation_slots = [{}]\n",
		                   JoinIndexed(graph.ActivationSlotCount(), ", ", [&](std::size_t index) {
			                   return std::format("slot{}: {}", index, FormatInfo(graph.GetActivationSlot(index)));
		                   }));
		out += std::format("  tape_slots = [{}]\n", JoinIndexed(graph.TapeSlotCount(), ", ", [&](std::size_t index) {
			return std::format("slot{}: {}", index, FormatInfo(graph.GetTapeSlot(index)));
		}));

		for (std::size_t subgraphId = 0; subgraphId < graph.SubgraphCount(); ++subgraphId)
		{
			const auto& subgraph = graph.GetSubgraph(subgraphId);
			out += "\n";
			out += std::format("  subgraph @{}{} {{\n", subgraphId, FormatSubgraphTags(graph, subgraphId));
			out += std::format("    params = [{}]\n",
			                   JoinIndexed(subgraph.Params().size(), ", ", [&](std::size_t index) {
				                   return std::format("{}: {}", FormatValueRef({ index, 0 }),
				                                      FormatInfo(subgraph.Params()[index]));
			                   }));
			out += std::format("    results = {}\n", FormatSubgraphResults(graph, subgraphId, subgraph));

			for (std::size_t nodeId = 0; nodeId < subgraph.Nodes().size(); ++nodeId)
			{
				const auto& entry = subgraph.Nodes()[nodeId];
				out += std::format("    {} = {}\n", FormatNodeOutputs(nodeId, entry.outputInfos),
				                   FormatNodePayload(entry.node, options));
			}
			out += "  }\n";
		}

		out += "}\n";
		return out;
	}
} // namespace LiteNN::Debug
