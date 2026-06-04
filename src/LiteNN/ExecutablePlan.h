#ifndef LITENN_EXECUTABLE_PLAN_H
#define LITENN_EXECUTABLE_PLAN_H

#include <LiteNN/Graph.h>
#include <LiteNN/OpSchema.h>
#include <LiteNN/Storage.h>
#include <LiteNN/TensorType.h>
#include <cstdint>
#include <format>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace LiteNN
{
	class ModelGraph
	{
	public:
		ModelGraph() = default;

		explicit ModelGraph(Graph graph) : graph_(std::move(graph)) {}

		Graph& MutableGraph() noexcept
		{
			return graph_;
		}

		const Graph& GraphView() const noexcept
		{
			return graph_;
		}

		Graph TakeGraph() noexcept
		{
			return std::move(graph_);
		}

	private:
		Graph graph_;
	};

	struct ExecutablePlanValue
	{
		NodeOutput source;
		TensorType type;
		std::string name;
	};

	struct ExecutablePlanAttribute
	{
		std::string name;
		std::string value;
	};

	struct ExecutablePlanOp
	{
		std::string kind;
		std::uint32_t schemaId{};
		OpCategory category{ OpCategory::Custom };
		OpEffect effect{ OpEffect::Pure };
		std::vector<ExecutablePlanAttribute> attributes;
	};

	struct ExecutablePlanNode
	{
		NodeId sourceNode{};
		ExecutablePlanOp op;
		// Internal execution payload. vNext packages and public runtime/compiler boundaries must use op/schema/type facts.
		NodeVariant node;
		std::string opKind;
		OpCategory category{ OpCategory::Custom };
		OpEffect effect{ OpEffect::Pure };
		std::vector<NodeOutput> inputs;
		std::vector<TensorType> outputs;
	};

	struct ExecutablePlanSubgraph
	{
		SubgraphId sourceSubgraph{};
		std::vector<TensorType> params;
		std::vector<ExecutablePlanNode> nodes;
		std::vector<NodeOutput> results;
	};

	struct ExecutablePlan
	{
		SubgraphId forward{};
		std::optional<SubgraphId> backward;
		std::vector<ExecutablePlanSubgraph> subgraphs;
		std::vector<TensorStorageRef> variables;
		std::vector<TensorType> activationSlots;
		std::vector<TensorType> tapeSlots;
		std::vector<ExecutablePlanValue> inputs;
		std::vector<ExecutablePlanValue> outputs;
	};

	using FunctionId = std::size_t;
	using RegionId = std::size_t;
	using PartitionId = std::size_t;

	struct ExecutableFunction
	{
		FunctionId id{};
		std::string name;
		SubgraphId body{};
		std::vector<TensorType> inputs;
		std::vector<TensorType> outputs;
	};

	struct ExecutableRegion
	{
		RegionId id{};
		std::string name;
		FunctionId function{};
		SubgraphId subgraph{};
		std::vector<NodeId> nodes;
	};

	struct ExecutablePartition
	{
		PartitionId id{};
		std::string backend{ std::string(BackendCPUInterpreter) };
		std::vector<RegionId> regions;
		std::vector<TensorMemorySpace> memorySpaces{ TensorMemorySpace::Host };
	};

	struct ExecutableModule
	{
		ExecutablePlan plan;
		std::vector<ExecutableFunction> functions;
		std::vector<ExecutableRegion> regions;
		std::vector<ExecutablePartition> partitions;
	};

	struct ExecutablePlanBackendIssue
	{
		SubgraphId subgraph{};
		NodeId node{};
		std::string opKind;
		BackendSupportLevel support{ BackendSupportLevel::Unsupported };
		std::string fallback;
	};

	struct ExecutablePlanCompatibilityDiagnostic
	{
		SubgraphId subgraph{};
		NodeId node{};
		std::string opKind;
		OpDomain domain{ OpDomain::Core };
		std::string message;
	};

	inline void AddPlanAttribute(std::vector<ExecutablePlanAttribute>& attributes, std::string name,
	                             std::string value)
	{
		attributes.push_back({ std::move(name), std::move(value) });
	}

	inline void AddPlanAttribute(std::vector<ExecutablePlanAttribute>& attributes, std::string name,
	                             std::size_t value)
	{
		AddPlanAttribute(attributes, std::move(name), std::to_string(value));
	}

	inline void AddPlanAttribute(std::vector<ExecutablePlanAttribute>& attributes, std::string name, double value)
	{
		AddPlanAttribute(attributes, std::move(name), std::format("{}", value));
	}

	inline void AddPlanAttribute(std::vector<ExecutablePlanAttribute>& attributes, std::string name, bool value)
	{
		AddPlanAttribute(attributes, std::move(name), std::string(value ? "true" : "false"));
	}

	template <typename Enum>
		requires std::is_enum_v<Enum>
	inline void AddPlanAttribute(std::vector<ExecutablePlanAttribute>& attributes, std::string name, Enum value)
	{
		AddPlanAttribute(attributes, std::move(name), static_cast<std::size_t>(value));
	}

	inline std::string JoinSizeList(std::span<const std::size_t> values)
	{
		std::string text;
		for (std::size_t i = 0; i < values.size(); ++i)
		{
			if (i != 0)
			{
				text += ',';
			}
			text += std::to_string(values[i]);
		}
		return text;
	}

	inline void AddPlanAttribute(std::vector<ExecutablePlanAttribute>& attributes, std::string name,
	                             const std::vector<std::size_t>& values)
	{
		AddPlanAttribute(attributes, std::move(name), JoinSizeList(values));
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const ParamRefNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "paramIndex", node.paramIndex);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const ConstantNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "dtype", node.value.DType());
		AddPlanAttribute(attrs, "shape", node.value.Shape().ToOwned());
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const QuantizedConstantNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs = PlanAttributesForNode(ConstantNode{ node.storage });
		AddPlanAttribute(attrs, "quantizationScheme", node.params.scheme);
		AddPlanAttribute(attrs, "quantizationStorageType", node.params.storageType);
		AddPlanAttribute(attrs, "quantizationExpressedType", node.params.expressedType);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const VariableRefNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "variableIndex", node.variableIndex);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const UnaryOpNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "op", node.op);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const BinaryOpNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "op", node.op);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const CallNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "callee", node.callee);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const CastNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "targetType", node.targetType);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const QuantizeNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "scheme", node.params.scheme);
		AddPlanAttribute(attrs, "storageType", node.params.storageType);
		AddPlanAttribute(attrs, "expressedType", node.params.expressedType);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const DequantizeNode& node)
	{
		auto attrs = PlanAttributesForNode(QuantizeNode{ node.input, node.params });
		AddPlanAttribute(attrs, "targetType", node.targetType);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const CondNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "thenBranch", node.thenBranch);
		AddPlanAttribute(attrs, "elseBranch", node.elseBranch);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const WhileNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "condBranch", node.condBranch);
		AddPlanAttribute(attrs, "bodyBranch", node.bodyBranch);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const SaveActivationNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "slotId", node.slotId);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const LoadActivationNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "slotId", node.slotId);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const TapeSaveActivationNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "tapeSlotId", node.tapeSlotId);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const TapeLoadActivationNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "tapeSlotId", node.tapeSlotId);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const ReduceOpNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "op", node.op);
		AddPlanAttribute(attrs, "axis", node.axis);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const ReshapeNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "targetShape", node.targetShape);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const PermuteNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "permutation", node.permutation);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const BroadcastToNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "targetShape", node.targetShape);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const PadNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "lowPads", node.lowPads);
		AddPlanAttribute(attrs, "highPads", node.highPads);
		AddPlanAttribute(attrs, "mode", node.mode);
		AddPlanAttribute(attrs, "constantValue", node.constantValue);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const GatherNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "axis", node.axis);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const ScatterNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "axis", node.axis);
		AddPlanAttribute(attrs, "mode", node.mode);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const ScanNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "axis", node.axis);
		AddPlanAttribute(attrs, "op", node.op);
		return attrs;
	}

	template <typename T>
	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const T&)
	{
		return {};
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const SoftmaxNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "axis", node.axis);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const NormalizationNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "mode", node.mode);
		AddPlanAttribute(attrs, "axis", node.axis);
		AddPlanAttribute(attrs, "groupCount", node.groupCount);
		AddPlanAttribute(attrs, "epsilon", node.epsilon);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const TimestepEmbeddingNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "dim", node.dim);
		AddPlanAttribute(attrs, "maxPeriod", node.maxPeriod);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const SolveTriNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "lower", node.lower);
		AddPlanAttribute(attrs, "unitDiagonal", node.unitDiagonal);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const SGDStepNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "learningRate", node.learningRate);
		AddPlanAttribute(attrs, "momentum", node.momentum);
		AddPlanAttribute(attrs, "weightDecay", node.weightDecay);
		AddPlanAttribute(attrs, "nesterov", node.nesterov);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const AdamWStepNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "learningRate", node.learningRate);
		AddPlanAttribute(attrs, "beta1", node.beta1);
		AddPlanAttribute(attrs, "beta2", node.beta2);
		AddPlanAttribute(attrs, "epsilon", node.epsilon);
		AddPlanAttribute(attrs, "weightDecay", node.weightDecay);
		AddPlanAttribute(attrs, "step", node.step);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const Im2ColNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "kernelShape", node.kernelShape);
		AddPlanAttribute(attrs, "strides", node.strides);
		AddPlanAttribute(attrs, "dilations", node.dilations);
		AddPlanAttribute(attrs, "lowPads", node.lowPads);
		AddPlanAttribute(attrs, "highPads", node.highPads);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const Conv2DNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "strides", node.strides);
		AddPlanAttribute(attrs, "dilations", node.dilations);
		AddPlanAttribute(attrs, "lowPads", node.lowPads);
		AddPlanAttribute(attrs, "highPads", node.highPads);
		AddPlanAttribute(attrs, "groupCount", node.groupCount);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const ConvTranspose2DNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "strides", node.strides);
		AddPlanAttribute(attrs, "dilations", node.dilations);
		AddPlanAttribute(attrs, "lowPads", node.lowPads);
		AddPlanAttribute(attrs, "highPads", node.highPads);
		AddPlanAttribute(attrs, "outputPads", node.outputPads);
		AddPlanAttribute(attrs, "groupCount", node.groupCount);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const Pool2DNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "mode", node.mode);
		AddPlanAttribute(attrs, "kernelShape", node.kernelShape);
		AddPlanAttribute(attrs, "strides", node.strides);
		AddPlanAttribute(attrs, "lowPads", node.lowPads);
		AddPlanAttribute(attrs, "highPads", node.highPads);
		AddPlanAttribute(attrs, "countIncludePad", node.countIncludePad);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const UpsampleNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "mode", node.mode);
		AddPlanAttribute(attrs, "outputSpatialShape", node.outputSpatialShape);
		AddPlanAttribute(attrs, "alignCorners", node.alignCorners);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const ConcatNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "axis", node.axis);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const SliceNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "axis", node.axis);
		AddPlanAttribute(attrs, "start", node.start);
		AddPlanAttribute(attrs, "length", node.length);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const ArgsortNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "axis", node.axis);
		AddPlanAttribute(attrs, "order", node.order);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> PlanAttributesForNode(const FusedOpNode& node)
	{
		std::vector<ExecutablePlanAttribute> attrs;
		AddPlanAttribute(attrs, "pattern", node.pattern);
		AddPlanAttribute(attrs, "body", node.body);
		return attrs;
	}

	inline std::vector<ExecutablePlanAttribute> ExtractPlanAttributes(const NodeVariant& node)
	{
		return std::visit([](const auto& concrete) { return PlanAttributesForNode(concrete); }, node);
	}

	inline ExecutablePlanOp BuildExecutablePlanOp(const NodeVariant& node, const OpSchema& schema,
	                                              std::uint32_t schemaId)
	{
		return { .kind = schema.kind,
			     .schemaId = schemaId,
			     .category = schema.category,
			     .effect = schema.effect,
			     .attributes = ExtractPlanAttributes(node) };
	}

	inline void ValidateExecutableTensorType(const TensorType& type, std::string_view context)
	{
		if (!IsValidDataTypeValue(type.dtype))
		{
			throw std::runtime_error(std::format("{} has invalid dtype {}", context, static_cast<int>(type.dtype)));
		}
		for (std::size_t i = 0; i < type.shape.dims.size(); ++i)
		{
			const auto& dim = type.shape.dims[i];
			if (dim.kind == TensorDimKind::Static && dim.extent == 0)
			{
				throw std::runtime_error(std::format("{} has zero static dimension at axis {}", context, i));
			}
			if (dim.kind == TensorDimKind::Symbolic && dim.symbol.empty())
			{
				throw std::runtime_error(std::format("{} has empty symbolic dimension at axis {}", context, i));
			}
		}
		if (type.layout.HasExplicitStrides() && type.layout.strides.size() != type.Rank())
		{
			throw std::runtime_error(std::format("{} has {} explicit strides for rank {}", context,
			                                    type.layout.strides.size(), type.Rank()));
		}
	}

	inline void ValidateExecutablePlanValueRef(const ExecutablePlanSubgraph& subgraph, NodeOutput output,
	                                           std::string_view context, std::optional<NodeId> currentNode = std::nullopt)
	{
		if (output.node >= subgraph.nodes.size())
		{
			throw std::runtime_error(std::format("{} references node {}, but nodeCount={}", context, output.node,
			                                    subgraph.nodes.size()));
		}
		if (currentNode && output.node >= *currentNode)
		{
			throw std::runtime_error(std::format("{} references node {}, which is not before current node {}", context,
			                                    output.node, *currentNode));
		}
		if (output.port >= subgraph.nodes[output.node].outputs.size())
		{
			throw std::runtime_error(std::format("{} references node {} port {}, but outputCount={}", context,
			                                    output.node, output.port, subgraph.nodes[output.node].outputs.size()));
		}
	}

	inline void ValidateExecutablePlan(const ExecutablePlan& plan,
	                                   const OpSchemaRegistry& registry = DefaultOpSchemaRegistry())
	{
		if (plan.subgraphs.empty())
		{
			throw std::runtime_error("ExecutablePlan validation failed: plan contains no subgraphs");
		}
		if (plan.forward >= plan.subgraphs.size())
		{
			throw std::runtime_error(std::format("ExecutablePlan validation failed: forward subgraph {} is out of range",
			                                    plan.forward));
		}
		if (plan.backward && *plan.backward >= plan.subgraphs.size())
		{
			throw std::runtime_error(std::format("ExecutablePlan validation failed: backward subgraph {} is out of range",
			                                    *plan.backward));
		}

		for (std::size_t i = 0; i < plan.variables.size(); ++i)
		{
			ValidateExecutableTensorType(plan.variables[i].type, std::format("variable {}", i));
			if (plan.variables[i].region.alignment == 0)
			{
				throw std::runtime_error(std::format("variable {} storage region has zero alignment", i));
			}
			if (!plan.variables[i].viewStrides.empty() &&
			    plan.variables[i].viewStrides.size() != plan.variables[i].type.Rank())
			{
				throw std::runtime_error(std::format("variable {} view has {} strides for rank {}", i,
				                                    plan.variables[i].viewStrides.size(),
				                                    plan.variables[i].type.Rank()));
			}
			if (const auto logicalBytes = plan.variables[i].LogicalByteSize();
			    logicalBytes && plan.variables[i].region.byteSize != 0 &&
			    plan.variables[i].storageOffsetBytes + *logicalBytes > plan.variables[i].region.byteSize)
			{
				throw std::runtime_error(std::format("variable {} storage view exceeds its buffer region", i));
			}
		}
		for (std::size_t i = 0; i < plan.activationSlots.size(); ++i)
		{
			ValidateExecutableTensorType(plan.activationSlots[i], std::format("activation slot {}", i));
		}
		for (std::size_t i = 0; i < plan.tapeSlots.size(); ++i)
		{
			ValidateExecutableTensorType(plan.tapeSlots[i], std::format("tape slot {}", i));
		}

		for (std::size_t subgraphIndex = 0; subgraphIndex < plan.subgraphs.size(); ++subgraphIndex)
		{
			const auto& subgraph = plan.subgraphs[subgraphIndex];
			for (std::size_t i = 0; i < subgraph.params.size(); ++i)
			{
				ValidateExecutableTensorType(subgraph.params[i], std::format("subgraph {} param {}", subgraphIndex, i));
			}
			for (std::size_t nodeIndex = 0; nodeIndex < subgraph.nodes.size(); ++nodeIndex)
			{
				const auto& node = subgraph.nodes[nodeIndex];
				if (node.sourceNode != nodeIndex)
				{
					throw std::runtime_error(std::format("subgraph {} node {} sourceNode mismatch: {}", subgraphIndex,
					                                    nodeIndex, node.sourceNode));
				}
				if (node.op.kind.empty())
				{
					throw std::runtime_error(std::format("subgraph {} node {} has empty executable op kind",
					                                    subgraphIndex, nodeIndex));
				}
				if (!node.opKind.empty() && node.opKind != node.op.kind)
				{
					throw std::runtime_error(std::format("subgraph {} node {} op kind mismatch: {} vs {}",
					                                    subgraphIndex, nodeIndex, node.op.kind, node.opKind));
				}
				if (node.category != node.op.category || node.effect != node.op.effect)
				{
					throw std::runtime_error(std::format("subgraph {} node {} op metadata mismatch",
					                                    subgraphIndex, nodeIndex));
				}
				const auto expectedSchemaId = static_cast<std::uint32_t>(registry.IndexOf(node.op.kind));
				if (node.op.schemaId != expectedSchemaId)
				{
					throw std::runtime_error(std::format("subgraph {} node {} schema id mismatch: expected {}, got {}",
					                                    subgraphIndex, nodeIndex, expectedSchemaId,
					                                    node.op.schemaId));
				}
				const auto& schema = registry.Require(node.op.kind);
				if (!schema.AllowsInputCount(node.inputs.size()))
				{
					throw std::runtime_error(std::format("subgraph {} node {} {} input count {} violates schema",
					                                    subgraphIndex, nodeIndex, node.op.kind, node.inputs.size()));
				}
				if (!schema.AllowsOutputCount(node.outputs.size()))
				{
					throw std::runtime_error(std::format("subgraph {} node {} {} output count {} violates schema",
					                                    subgraphIndex, nodeIndex, node.op.kind, node.outputs.size()));
				}
				for (std::size_t outputIndex = 0; outputIndex < node.outputs.size(); ++outputIndex)
				{
					ValidateExecutableTensorType(node.outputs[outputIndex],
					                             std::format("subgraph {} node {} output {}", subgraphIndex, nodeIndex,
					                                         outputIndex));
				}
				for (std::size_t inputIndex = 0; inputIndex < node.inputs.size(); ++inputIndex)
				{
					ValidateExecutablePlanValueRef(
					    subgraph, node.inputs[inputIndex],
					    std::format("subgraph {} node {} input {}", subgraphIndex, nodeIndex, inputIndex), nodeIndex);
				}
			}
			for (std::size_t resultIndex = 0; resultIndex < subgraph.results.size(); ++resultIndex)
			{
				ValidateExecutablePlanValueRef(
				    subgraph, subgraph.results[resultIndex],
				    std::format("subgraph {} result {}", subgraphIndex, resultIndex));
			}
		}

		const auto& forward = plan.subgraphs[plan.forward];
		if (plan.inputs.size() != forward.params.size())
		{
			throw std::runtime_error(std::format(
			    "ExecutablePlan public input count {} does not match forward param count {}", plan.inputs.size(),
			    forward.params.size()));
		}
		if (plan.outputs.size() != forward.results.size())
		{
			throw std::runtime_error(std::format(
			    "ExecutablePlan public output count {} does not match forward result count {}", plan.outputs.size(),
			    forward.results.size()));
		}
		for (std::size_t i = 0; i < plan.inputs.size(); ++i)
		{
			ValidateExecutableTensorType(plan.inputs[i].type, std::format("public input {}", i));
			if (plan.inputs[i].type != forward.params[i])
			{
				throw std::runtime_error(std::format("public input {} type does not match forward param", i));
			}
		}
		for (std::size_t i = 0; i < plan.outputs.size(); ++i)
		{
			ValidateExecutableTensorType(plan.outputs[i].type, std::format("public output {}", i));
			const auto output = forward.results[i];
			if (plan.outputs[i].source != output)
			{
				throw std::runtime_error(std::format("public output {} source does not match forward result", i));
			}
			if (plan.outputs[i].type != forward.nodes[output.node].outputs[output.port])
			{
				throw std::runtime_error(std::format("public output {} type does not match forward result", i));
			}
		}
	}

	inline std::vector<ExecutablePlanBackendIssue> CollectExecutablePlanBackendIssues(
	    const ExecutablePlan& plan, std::string_view backend,
	    const OpSchemaRegistry& registry = DefaultOpSchemaRegistry(), bool allowFallback = true)
	{
		ValidateExecutablePlan(plan, registry);
		std::vector<ExecutablePlanBackendIssue> issues;
		for (std::size_t subgraphIndex = 0; subgraphIndex < plan.subgraphs.size(); ++subgraphIndex)
		{
			const auto& subgraph = plan.subgraphs[subgraphIndex];
			for (std::size_t nodeIndex = 0; nodeIndex < subgraph.nodes.size(); ++nodeIndex)
			{
				const auto& node = subgraph.nodes[nodeIndex];
				const auto& schema = registry.Require(node.op.kind);
				const auto* capability = schema.FindCapability(backend);
				const auto support = capability ? capability->support : BackendSupportLevel::Unsupported;
				if (support == BackendSupportLevel::Unsupported ||
				    (!allowFallback && support == BackendSupportLevel::Fallback))
				{
					issues.push_back({ .subgraph = subgraph.sourceSubgraph,
					                   .node = node.sourceNode,
					                   .opKind = node.op.kind,
					                   .support = support,
					                   .fallback = capability ? capability->fallback : std::string{} });
				}
			}
		}
		return issues;
	}

	inline std::vector<ExecutablePlanCompatibilityDiagnostic> CollectExecutablePlanCompatibilityDiagnostics(
	    const ExecutablePlan& plan, const OpSchemaRegistry& registry = DefaultOpSchemaRegistry())
	{
		ValidateExecutablePlan(plan, registry);
		std::vector<ExecutablePlanCompatibilityDiagnostic> diagnostics;
		for (std::size_t subgraphIndex = 0; subgraphIndex < plan.subgraphs.size(); ++subgraphIndex)
		{
			const auto& subgraph = plan.subgraphs[subgraphIndex];
			for (std::size_t nodeIndex = 0; nodeIndex < subgraph.nodes.size(); ++nodeIndex)
			{
				const auto& node = subgraph.nodes[nodeIndex];
				const auto& schema = registry.Require(node.op.kind);
				if (schema.domain == OpDomain::Core)
				{
					continue;
				}
				diagnostics.push_back({
				    .subgraph = subgraph.sourceSubgraph,
				    .node = node.sourceNode,
				    .opKind = node.op.kind,
				    .domain = schema.domain,
				    .message = std::format(
				        "op '{}' remains in compatibility domain; importer must lower it to core ops or keep it in a tagged compatibility partition",
				        node.op.kind),
				});
			}
		}
		return diagnostics;
	}

	inline void RequireExecutablePlanBackendSupport(const ExecutablePlan& plan, std::string_view backend,
	                                                const OpSchemaRegistry& registry = DefaultOpSchemaRegistry(),
	                                                bool allowFallback = true)
	{
		const auto issues = CollectExecutablePlanBackendIssues(plan, backend, registry, allowFallback);
		if (issues.empty())
		{
			return;
		}
		const auto& first = issues.front();
		throw std::runtime_error(std::format(
		    "ExecutablePlan backend '{}' cannot lower {} op(s); first unsupported op is subgraph {} node {} ({})",
		    backend, issues.size(), first.subgraph, first.node, first.opKind));
	}

	inline TensorType ToTensorType(const OutputInfo& info)
	{
		return info.Type();
	}

	inline TensorType ToTensorType(const TensorSpec& spec)
	{
		return spec.Type();
	}

	inline TensorType ToTensorType(const SubgraphParam& param)
	{
		return param.Type();
	}

	inline TensorMemorySpace TensorMemorySpaceFor(const PolymorphicDevice& device)
	{
		return device.Is<CPU>() ? TensorMemorySpace::Host : TensorMemorySpace::Device;
	}

	inline ExecutablePlan BuildExecutablePlan(const Graph& graph,
	                                          const OpSchemaRegistry& registry = DefaultOpSchemaRegistry())
	{
		ExecutablePlan plan;
		plan.forward = graph.Forward();
		plan.backward = graph.Backward();
		plan.subgraphs.reserve(graph.SubgraphCount());

		for (SubgraphId subgraphId = 0; subgraphId < graph.SubgraphCount(); ++subgraphId)
		{
			const auto& subgraph = graph.GetSubgraph(subgraphId);
			ExecutablePlanSubgraph planSubgraph;
			planSubgraph.sourceSubgraph = subgraphId;
			planSubgraph.params.reserve(subgraph.Params().size());
			for (const auto& param : subgraph.Params())
			{
				planSubgraph.params.push_back(ToTensorType(param));
			}

			planSubgraph.nodes.reserve(subgraph.Nodes().size());
			for (NodeId nodeId = 0; nodeId < subgraph.Nodes().size(); ++nodeId)
			{
				const auto& entry = subgraph.Nodes()[nodeId];
				const auto opKind = OpKindName(entry.node);
				const auto& schema = registry.Require(opKind);
				auto inputs = NodeInputs(entry.node);
				if (!schema.AllowsInputCount(inputs.size()))
				{
					throw std::runtime_error("Node input count does not match op schema: " + opKind);
				}
				if (!schema.AllowsOutputCount(entry.outputInfos.size()))
				{
					throw std::runtime_error("Node output count does not match op schema: " + opKind);
				}

				ExecutablePlanNode planNode;
				planNode.sourceNode = nodeId;
				planNode.op = BuildExecutablePlanOp(entry.node, schema,
				                                     static_cast<std::uint32_t>(registry.IndexOf(opKind)));
				planNode.node = entry.node;
				planNode.opKind = opKind;
				planNode.category = schema.category;
				planNode.effect = schema.effect;
				planNode.inputs = std::move(inputs);
				planNode.outputs.reserve(entry.outputInfos.size());
				for (const auto& output : entry.outputInfos)
				{
					planNode.outputs.push_back(ToTensorType(output));
				}
				planSubgraph.nodes.push_back(std::move(planNode));
			}
			planSubgraph.results.assign(subgraph.Results().begin(), subgraph.Results().end());
			plan.subgraphs.push_back(std::move(planSubgraph));
		}

		plan.variables.reserve(graph.VariableCount());
		for (std::size_t i = 0; i < graph.VariableCount(); ++i)
		{
			const auto& tensor = graph.GetVariable(i)->Data();
			TensorStorageRef storage;
			const auto memorySpace = TensorMemorySpaceFor(tensor.CurDevice());
			storage.type = TensorType::Dense(tensor.DType(), tensor.Shape(), memorySpace);
			storage.quantization = graph.GetVariable(i)->Quantization();
			storage.region = MakeBorrowedBufferRegion(tensor.RawData(), storage.type.ByteSize().value_or(0), memorySpace);
			storage.region.name = graph.VariableName(i);
			plan.variables.push_back(std::move(storage));
		}

		plan.activationSlots.reserve(graph.ActivationSlotCount());
		for (std::size_t i = 0; i < graph.ActivationSlotCount(); ++i)
		{
			plan.activationSlots.push_back(graph.GetActivationSlot(i).Type());
		}
		plan.tapeSlots.reserve(graph.TapeSlotCount());
		for (std::size_t i = 0; i < graph.TapeSlotCount(); ++i)
		{
			plan.tapeSlots.push_back(graph.GetTapeSlot(i).Type());
		}

		const auto inputSignature = graph.InputTypeSignature();
		plan.inputs.reserve(inputSignature.size());
		for (std::size_t i = 0; i < inputSignature.size(); ++i)
		{
			plan.inputs.push_back({ { i, 0 }, inputSignature[i].type, inputSignature[i].name });
		}

		const auto outputSignature = graph.OutputTypeSignature();
		plan.outputs.reserve(outputSignature.size());
		for (std::size_t i = 0; i < outputSignature.size(); ++i)
		{
			plan.outputs.push_back(
			    { graph.GetSubgraph(graph.Forward()).Results()[i], outputSignature[i].type, outputSignature[i].name });
		}

		ValidateExecutablePlan(plan, registry);
		return plan;
	}

	inline ExecutablePlan BuildExecutablePlan(const ModelGraph& model,
	                                          const OpSchemaRegistry& registry = DefaultOpSchemaRegistry())
	{
		return BuildExecutablePlan(model.GraphView(), registry);
	}

	inline ExecutableModule BuildExecutableModule(ExecutablePlan plan)
	{
		ValidateExecutablePlan(plan);

		ExecutableModule module;
		module.plan = std::move(plan);
		module.functions.reserve(module.plan.subgraphs.size());
		module.regions.reserve(module.plan.subgraphs.size());

		for (std::size_t i = 0; i < module.plan.subgraphs.size(); ++i)
		{
			const auto& subgraph = module.plan.subgraphs[i];
			ExecutableFunction function;
			function.id = i;
			function.name = i == module.plan.forward ? "forward" : std::format("subgraph_{}", i);
			if (module.plan.backward && i == *module.plan.backward)
			{
				function.name = "backward";
			}
			function.body = subgraph.sourceSubgraph;
			function.inputs = subgraph.params;
			function.outputs.reserve(subgraph.results.size());
			for (const auto& result : subgraph.results)
			{
				function.outputs.push_back(subgraph.nodes[result.node].outputs[result.port]);
			}
			module.functions.push_back(std::move(function));

			ExecutableRegion region;
			region.id = i;
			region.name = std::format("region_{}", i);
			region.function = i;
			region.subgraph = subgraph.sourceSubgraph;
			region.nodes.reserve(subgraph.nodes.size());
			for (NodeId node = 0; node < subgraph.nodes.size(); ++node)
			{
				region.nodes.push_back(node);
			}
			module.regions.push_back(std::move(region));
		}

		ExecutablePartition partition;
		partition.id = 0;
		partition.backend = std::string(BackendCPUInterpreter);
		partition.regions.reserve(module.regions.size());
		for (const auto& region : module.regions)
		{
			partition.regions.push_back(region.id);
		}
		module.partitions.push_back(std::move(partition));
		return module;
	}

	inline ExecutableModule BuildExecutableModule(const Graph& graph,
	                                              const OpSchemaRegistry& registry = DefaultOpSchemaRegistry())
	{
		return BuildExecutableModule(BuildExecutablePlan(graph, registry));
	}

	inline ExecutableModule BuildExecutableModule(const ModelGraph& model,
	                                              const OpSchemaRegistry& registry = DefaultOpSchemaRegistry())
	{
		return BuildExecutableModule(BuildExecutablePlan(model, registry));
	}
} // namespace LiteNN

#endif
