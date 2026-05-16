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
						    else if constexpr (std::same_as<T, ConcatNode>)
						    {
							    return ConcatNode{ remapList(node.inputs), node.axis };
						    }
						    else if constexpr (std::same_as<T, SliceNode>)
						    {
							    return SliceNode{ remapOutput(node.input), node.axis, node.start, node.length };
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
