#include <LiteNN/Graph.h>
#include <LiteNN/Pass.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <optional>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#ifndef LITENN_PASS_FORWARDONLYPASS_H
#define LITENN_PASS_FORWARDONLYPASS_H

namespace LiteNN
{
	namespace Detail
	{
		class ForwardOnlyExtractor
		{
		public:
			explicit ForwardOnlyExtractor(const Graph& source) : source_(source) {}

			Graph Extract()
			{
				for (const auto& variable : source_.Variables())
				{
					result_.AddVariable(variable);
				}

				const auto forward = CloneSubgraph(source_.Forward());
				result_.SetForward(forward);
				result_.SetInputNames({ source_.InputNames().begin(), source_.InputNames().end() });
				result_.SetOutputNames({ source_.OutputNames().begin(), source_.OutputNames().end() });
				Validation::ValidateGraph(result_);
				return std::move(result_);
			}

		private:
			SubgraphId CloneSubgraph(SubgraphId oldId)
			{
				if (const auto it = subgraphMap_.find(oldId); it != subgraphMap_.end())
				{
					return it->second;
				}

				const auto newId = result_.AddSubgraph(Subgraph{});
				subgraphMap_.emplace(oldId, newId);

				const auto& oldSubgraph = source_.GetSubgraph(oldId);
				Subgraph cloned;
				for (const auto& param : oldSubgraph.Params())
				{
					(void)cloned.AddParam(param.dtype, param.shape);
				}

				std::vector<std::vector<NodeOutput>> outputMap(oldSubgraph.NodeCount());
				for (NodeId nodeId = 0; nodeId < oldSubgraph.NodeCount(); ++nodeId)
				{
					const auto& entry = oldSubgraph.GetNodeEntry(nodeId);
					if (const auto* param = std::get_if<ParamRefNode>(&entry.node))
					{
						outputMap[nodeId] = { { param->paramIndex, 0 } };
					}
				}

				const auto remapOutput = [&](NodeOutput output) {
					if (output.node >= outputMap.size() || output.port >= outputMap[output.node].size())
					{
						throw std::runtime_error("ForwardOnlyPass encountered an unmapped node output");
					}
					return outputMap[output.node][output.port];
				};

				const auto remapList = [&](std::span<const NodeOutput> outputs) {
					std::vector<NodeOutput> remapped;
					remapped.reserve(outputs.size());
					for (const auto output : outputs)
					{
						remapped.push_back(remapOutput(output));
					}
					return remapped;
				};

				for (NodeId nodeId = 0; nodeId < oldSubgraph.NodeCount(); ++nodeId)
				{
					const auto& entry = oldSubgraph.GetNodeEntry(nodeId);
					if (std::holds_alternative<ParamRefNode>(entry.node))
					{
						continue;
					}

					if (const auto* save = std::get_if<SaveActivationNode>(&entry.node))
					{
						outputMap[nodeId] = { remapOutput(save->input) };
						continue;
					}
					if (const auto* save = std::get_if<TapeSaveActivationNode>(&entry.node))
					{
						outputMap[nodeId] = { remapOutput(save->input) };
						continue;
					}
					if (std::holds_alternative<LoadActivationNode>(entry.node) ||
					    std::holds_alternative<TapeLoadActivationNode>(entry.node))
					{
						throw std::runtime_error("ForwardOnlyPass cannot clone a graph whose forward path loads activations");
					}

					auto clonedNode = std::visit(
					    [&](const auto& node) -> NodeVariant {
						    using T = std::decay_t<decltype(node)>;
						    if constexpr (std::same_as<T, ConstantNode> ||
						                  std::same_as<T, QuantizedConstantNode> ||
						                  std::same_as<T, VariableRefNode>)
						    {
							    return node;
						    }
						    else if constexpr (std::same_as<T, UnaryOpNode>)
						    {
							    return UnaryOpNode{ node.op, remapOutput(node.input) };
						    }
						    else if constexpr (std::same_as<T, BinaryOpNode>)
						    {
							    return BinaryOpNode{ node.op, remapOutput(node.lhs), remapOutput(node.rhs) };
						    }
						    else if constexpr (std::same_as<T, CallNode>)
						    {
							    return CallNode{ CloneSubgraph(node.callee), remapList(node.args) };
						    }
						    else if constexpr (std::same_as<T, CastNode>)
						    {
							    return CastNode{ remapOutput(node.input), node.targetType };
						    }
						    else if constexpr (std::same_as<T, QuantizeNode>)
						    {
							    return QuantizeNode{ remapOutput(node.input), node.params };
						    }
						    else if constexpr (std::same_as<T, DequantizeNode>)
						    {
							    return DequantizeNode{ remapOutput(node.input), node.params, node.targetType };
						    }
						    else if constexpr (std::same_as<T, CondNode>)
						    {
							    return CondNode{ remapOutput(node.condition), CloneSubgraph(node.thenBranch),
							                     CloneSubgraph(node.elseBranch), remapList(node.args) };
						    }
						    else if constexpr (std::same_as<T, WhileNode>)
						    {
							    return WhileNode{ CloneSubgraph(node.condBranch), CloneSubgraph(node.bodyBranch),
							                      remapList(node.initArgs) };
						    }
						    else if constexpr (std::same_as<T, ReduceOpNode>)
						    {
							    return ReduceOpNode{ node.op, remapOutput(node.input), node.axis };
						    }
						    else if constexpr (std::same_as<T, ReshapeNode>)
						    {
							    return ReshapeNode{ remapOutput(node.input), node.targetShape };
						    }
						    else if constexpr (std::same_as<T, PermuteNode>)
						    {
							    return PermuteNode{ remapOutput(node.input), node.permutation };
						    }
						    else if constexpr (std::same_as<T, BroadcastToNode>)
						    {
							    return BroadcastToNode{ remapOutput(node.input), node.targetShape };
						    }
						    else if constexpr (std::same_as<T, PadNode>)
						    {
							    return PadNode{ remapOutput(node.input), node.lowPads, node.highPads, node.mode,
							                    node.constantValue };
						    }
						    else if constexpr (std::same_as<T, GatherNode>)
						    {
							    return GatherNode{ remapOutput(node.data), remapOutput(node.indices), node.axis };
						    }
						    else if constexpr (std::same_as<T, ScatterNode>)
						    {
							    return ScatterNode{ remapOutput(node.data), remapOutput(node.indices),
							                        remapOutput(node.updates), node.axis, node.mode };
						    }
						    else if constexpr (std::same_as<T, ScanNode>)
						    {
							    return ScanNode{ remapOutput(node.input), node.axis, node.op };
						    }
						    else if constexpr (std::same_as<T, SSMScanNode>)
						    {
							    return SSMScanNode{ remapOutput(node.state), remapOutput(node.dt), remapOutput(node.a),
							                        remapOutput(node.b), remapOutput(node.c),
							                        node.d ? std::optional<NodeOutput>{ remapOutput(*node.d) } : std::nullopt };
						    }
						    else if constexpr (std::same_as<T, RWKVWKVNode>)
						    {
							    return RWKVWKVNode{ remapOutput(node.key), remapOutput(node.value),
							                        remapOutput(node.receptance), remapOutput(node.timeDecay),
							                        remapOutput(node.timeFirst) };
						    }
						    else if constexpr (std::same_as<T, SoftmaxNode>)
						    {
							    return SoftmaxNode{ remapOutput(node.input), node.axis };
						    }
						    else if constexpr (std::same_as<T, NormalizationNode>)
						    {
							    return NormalizationNode{
							        remapOutput(node.input),
							        node.scale ? std::optional<NodeOutput>{ remapOutput(*node.scale) } : std::nullopt,
							        node.bias ? std::optional<NodeOutput>{ remapOutput(*node.bias) } : std::nullopt,
							        node.mode, node.axis, node.groupCount, node.epsilon
							    };
						    }
						    else if constexpr (std::same_as<T, BatchMatMulNode>)
						    {
							    return BatchMatMulNode{ remapOutput(node.lhs), remapOutput(node.rhs) };
						    }
						    else if constexpr (std::same_as<T, OutProdNode>)
						    {
							    return OutProdNode{ remapOutput(node.lhs), remapOutput(node.rhs) };
						    }
						    else if constexpr (std::same_as<T, TimestepEmbeddingNode>)
						    {
							    return TimestepEmbeddingNode{ remapOutput(node.timesteps), node.dim, node.maxPeriod };
						    }
						    else if constexpr (std::same_as<T, SolveTriNode>)
						    {
							    return SolveTriNode{ remapOutput(node.a), remapOutput(node.b),
							                         node.lower, node.unitDiagonal };
						    }
						    else if constexpr (std::same_as<T, SGDStepNode>)
						    {
							    return SGDStepNode{
							        remapOutput(node.parameter), remapOutput(node.gradient),
							        node.velocity ? std::optional<NodeOutput>{ remapOutput(*node.velocity) } : std::nullopt,
							        node.learningRate, node.momentum, node.weightDecay, node.nesterov
							    };
						    }
						    else if constexpr (std::same_as<T, AdamWStepNode>)
						    {
							    return AdamWStepNode{ remapOutput(node.parameter), remapOutput(node.gradient),
							                          remapOutput(node.firstMoment), remapOutput(node.secondMoment),
							                          node.learningRate, node.beta1, node.beta2, node.epsilon,
							                          node.weightDecay, node.step };
						    }
						    else if constexpr (std::same_as<T, Im2ColNode>)
						    {
							    return Im2ColNode{ remapOutput(node.input), node.kernelShape, node.strides,
							                       node.dilations, node.lowPads, node.highPads };
						    }
						    else if constexpr (std::same_as<T, Conv2DNode>)
						    {
							    return Conv2DNode{
							        remapOutput(node.input),
							        remapOutput(node.weight),
							        node.bias ? std::optional<NodeOutput>{ remapOutput(*node.bias) } : std::nullopt,
							        node.strides,
							        node.dilations,
							        node.lowPads,
							        node.highPads,
							        node.groupCount
							    };
						    }
						    else if constexpr (std::same_as<T, ConvTranspose2DNode>)
						    {
							    return ConvTranspose2DNode{
							        remapOutput(node.input),
							        remapOutput(node.weight),
							        node.bias ? std::optional<NodeOutput>{ remapOutput(*node.bias) } : std::nullopt,
							        node.strides,
							        node.dilations,
							        node.lowPads,
							        node.highPads,
							        node.outputPads,
							        node.groupCount
							    };
						    }
						    else if constexpr (std::same_as<T, Pool2DNode>)
						    {
							    return Pool2DNode{ remapOutput(node.input), node.mode, node.kernelShape,
							                       node.strides, node.lowPads, node.highPads, node.countIncludePad };
						    }
						    else if constexpr (std::same_as<T, UpsampleNode>)
						    {
							    return UpsampleNode{ remapOutput(node.input), node.mode, node.outputSpatialShape,
							                         node.alignCorners };
						    }
						    else if constexpr (std::same_as<T, ConcatNode>)
						    {
							    return ConcatNode{ remapList(node.inputs), node.axis };
						    }
						    else if constexpr (std::same_as<T, SliceNode>)
						    {
							    return SliceNode{ remapOutput(node.input), node.axis, node.start, node.length };
						    }
						    else if constexpr (std::same_as<T, GetRowsNode>)
						    {
							    return GetRowsNode{ remapOutput(node.data), remapOutput(node.indices) };
						    }
						    else if constexpr (std::same_as<T, ArgsortNode>)
						    {
							    return ArgsortNode{ remapOutput(node.input), node.axis, node.order };
						    }
						    else if constexpr (std::same_as<T, MulMatIdNode>)
						    {
							    return MulMatIdNode{ remapOutput(node.as), remapOutput(node.b), remapOutput(node.ids) };
						    }
						    else if constexpr (std::same_as<T, FusedOpNode>)
						    {
							    return FusedOpNode{ node.pattern, CloneSubgraph(node.body), remapList(node.args) };
						    }
						    else
						    {
							    throw std::runtime_error("ForwardOnlyPass encountered an unsupported node type");
						    }
					    },
					    entry.node);

					const auto newNodeId = cloned.AddNode(std::move(clonedNode), entry.outputInfos);
					outputMap[nodeId].reserve(entry.outputInfos.size());
					for (std::size_t port = 0; port < entry.outputInfos.size(); ++port)
					{
						outputMap[nodeId].push_back({ newNodeId, port });
					}
				}

				cloned.SetResults(remapList(oldSubgraph.Results()));
				result_.GetSubgraph(newId) = std::move(cloned);
				return newId;
			}

			const Graph& source_;
			Graph result_;
			std::unordered_map<SubgraphId, SubgraphId> subgraphMap_;
		};
	} // namespace Detail

	inline Graph ExtractForwardOnlyGraph(const Graph& graph)
	{
		return Detail::ForwardOnlyExtractor(graph).Extract();
	}

	class ForwardOnlyPass : public Pass
	{
	public:
		void Run(Graph& graph) override
		{
			graph = ExtractForwardOnlyGraph(graph);
		}
	};
} // namespace LiteNN

#endif
