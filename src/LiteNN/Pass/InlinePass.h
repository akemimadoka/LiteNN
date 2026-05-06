#ifndef LITENN_PASS_INLINE_H
#define LITENN_PASS_INLINE_H

#include <LiteNN/Graph.h>
#include <LiteNN/Pass.h>
#include <map>
#include <stdexcept>

namespace LiteNN
{
	// 内联 Pass：将 CallNode 展开为其 callee 子图的内容
	// 消除函数调用开销并暴露跨子图优化机会
	// 应在 AutogradPass 之后、ConstFoldPass/FusionPass 之前运行
	class InlinePass : public Pass
	{
	public:
		void Run(Graph& graph) override
		{
			const auto originalCount = graph.SubgraphCount();

			// 迭代 fixpoint：反复扫描直到没有 CallNode 被内联
			bool changed = true;
			while (changed)
			{
				changed = false;
				for (std::size_t sgId = 0; sgId < originalCount; ++sgId)
				{
					if (ProcessSubgraph(graph, sgId))
					{
						changed = true;
					}
				}
			}
		}

	private:
		using NodeOutputKey = std::pair<NodeId, std::size_t>;

		// ---- 节点输入重映射 ----

		template <typename RemapFn>
		static NodeVariant RemapNodeInputs(const NodeVariant& node, RemapFn&& remap)
		{
			return std::visit(
			    [&](const auto& n) -> NodeVariant {
				    using T = std::decay_t<decltype(n)>;
				    if constexpr (std::same_as<T, ParamRefNode> || std::same_as<T, ConstantNode> ||
				                  std::same_as<T, VariableRefNode> || std::same_as<T, LoadActivationNode>)
				    {
					    return n;
				    }
				    else if constexpr (std::same_as<T, UnaryOpNode>)
				    {
					    return UnaryOpNode{ n.op, remap(n.input) };
				    }
				    else if constexpr (std::same_as<T, BinaryOpNode>)
				    {
					    return BinaryOpNode{ n.op, remap(n.lhs), remap(n.rhs) };
				    }
				    else if constexpr (std::same_as<T, CastNode>)
				    {
					    return CastNode{ remap(n.input), n.targetType };
				    }
				    else if constexpr (std::same_as<T, ReduceOpNode>)
				    {
					    return ReduceOpNode{ n.op, remap(n.input), n.axis };
				    }
				    else if constexpr (std::same_as<T, ReshapeNode>)
				    {
					    return ReshapeNode{ remap(n.input), n.targetShape };
				    }
				    else if constexpr (std::same_as<T, ConcatNode>)
				    {
					    std::vector<NodeOutput> inputs;
					    for (const auto& input : n.inputs)
					    {
						    inputs.push_back(remap(input));
					    }
					    return ConcatNode{ std::move(inputs), n.axis };
				    }
				    else if constexpr (std::same_as<T, SliceNode>)
				    {
					    return SliceNode{ remap(n.input), n.axis, n.start, n.length };
				    }
				    else if constexpr (std::same_as<T, SaveActivationNode>)
				    {
					    return SaveActivationNode{ remap(n.input), n.slotId };
				    }
				    else if constexpr (std::same_as<T, CallNode>)
				    {
					    std::vector<NodeOutput> args;
					    for (const auto& a : n.args)
					    {
						    args.push_back(remap(a));
					    }
					    return CallNode{ n.callee, std::move(args) };
				    }
				    else if constexpr (std::same_as<T, CondNode>)
				    {
					    std::vector<NodeOutput> args;
					    for (const auto& a : n.args)
					    {
						    args.push_back(remap(a));
					    }
					    return CondNode{ remap(n.condition), n.thenBranch, n.elseBranch, std::move(args) };
				    }
				    else if constexpr (std::same_as<T, FusedOpNode>)
				    {
					    std::vector<NodeOutput> args;
					    for (const auto& a : n.args)
					    {
						    args.push_back(remap(a));
					    }
					    return FusedOpNode{ n.pattern, n.body, std::move(args) };
				    }
				    else if constexpr (std::same_as<T, WhileNode>)
				    {
					    std::vector<NodeOutput> args;
					    for (const auto& a : n.initArgs)
					    {
						    args.push_back(remap(a));
					    }
					    return WhileNode{ n.condBranch, n.bodyBranch, std::move(args) };
				    }
				    else if constexpr (std::same_as<T, TapeSaveActivationNode>)
				    {
					    return TapeSaveActivationNode{ remap(n.input), n.tapeSlotId };
				    }
				    else if constexpr (std::same_as<T, TapeLoadActivationNode>)
				    {
					    return n;
				    }
				    else
				    {
					    throw std::runtime_error("InlinePass: unsupported node type in remap");
				    }
			    },
			    node);
		}

		// ---- 检测子图中是否存在 CallNode ----

		static bool HasCallNode(const Subgraph& sg)
		{
			for (NodeId nodeId = 0; nodeId < sg.NodeCount(); ++nodeId)
			{
				if (std::holds_alternative<CallNode>(sg.GetNodeEntry(nodeId).node))
				{
					return true;
				}
			}
			return false;
		}

		// ---- 子图处理 ----

		// 返回 true 表示有 CallNode 被内联
		bool ProcessSubgraph(Graph& graph, SubgraphId sgId)
		{
			const auto& sg = graph.GetSubgraph(sgId);
			if (!HasCallNode(sg))
			{
				return false;
			}

			Subgraph newSg;
			std::vector<NodeId> nodeMap(sg.NodeCount(), static_cast<NodeId>(-1));

			// callResultMap: 对于 CallNode，其输出通过 callee 的 Results 映射
			// key = {callNodeOldId, port}, value = 内联后的 NodeOutput
			std::map<NodeOutputKey, NodeOutput> callResultMap;

			// 重映射函数：处理普通节点和 CallNode 输出
			auto remapOutput = [&](NodeOutput out) -> NodeOutput {
				auto it = callResultMap.find({ out.node, out.port });
				if (it != callResultMap.end())
				{
					return it->second;
				}
				return { nodeMap[out.node], out.port };
			};

			for (NodeId oldId = 0; oldId < sg.NodeCount(); ++oldId)
			{
				const auto& entry = sg.GetNodeEntry(oldId);

				if (auto* callNode = std::get_if<CallNode>(&entry.node))
				{
					// 内联 CallNode
					InlineCallNode(graph, newSg, sg, oldId, *callNode, nodeMap, callResultMap, remapOutput);
				}
				else
				{
					// 普通节点：重映射输入后复制
					auto remapped = RemapNodeInputs(entry.node, remapOutput);
					auto newId = newSg.AddNode(std::move(remapped),
					                           { entry.outputInfos.begin(), entry.outputInfos.end() });
					nodeMap[oldId] = newId;
				}
			}

			// 重映射 Results
			std::vector<NodeOutput> newResults;
			for (const auto& r : sg.Results())
			{
				newResults.push_back(remapOutput(r));
			}
			newSg.SetResults(std::move(newResults));

			graph.GetSubgraph(sgId) = std::move(newSg);
			return true;
		}

		template <typename RemapFn>
		void InlineCallNode(Graph& graph, Subgraph& newSg, const Subgraph& callerSg,
		                    NodeId callNodeOldId, const CallNode& callNode,
		                    std::vector<NodeId>& nodeMap,
		                    std::map<NodeOutputKey, NodeOutput>& callResultMap,
		                    RemapFn& remapOutput)
		{
			const auto& callee = graph.GetSubgraph(callNode.callee);

			// calleeMap: callee 中的 NodeId → newSg 中的 NodeId
			std::vector<NodeId> calleeMap(callee.NodeCount(), static_cast<NodeId>(-1));

			// paramOutputMap: ParamRefNode 的完整 NodeOutput 映射
			// 当实参来自多输出节点的非 0 port 时，calleeMap 仅存 node，
			// 需要 paramOutputMap 来保存完整的 {node, port}
			std::map<NodeId, NodeOutput> paramOutputMap;

			// callee 内部的重映射
			auto remapCalleeOutput = [&](NodeOutput out) -> NodeOutput {
				auto paramIt = paramOutputMap.find(out.node);
				if (paramIt != paramOutputMap.end())
				{
					// ParamRefNode 被直接替换为实参，port 来自实参
					return paramIt->second;
				}
				return { calleeMap[out.node], out.port };
			};

			for (NodeId calleeNodeId = 0; calleeNodeId < callee.NodeCount(); ++calleeNodeId)
			{
				const auto& calleeEntry = callee.GetNodeEntry(calleeNodeId);

				if (auto* paramRef = std::get_if<ParamRefNode>(&calleeEntry.node))
				{
					// ParamRefNode：替换为调用者传入的实参
					const auto& arg = callNode.args[paramRef->paramIndex];
					auto remappedArg = remapOutput(arg);
					// 记录完整的 NodeOutput 映射（包含 port 信息）
					paramOutputMap[calleeNodeId] = remappedArg;
					calleeMap[calleeNodeId] = remappedArg.node;
				}
				else
				{
					// 其他节点：重映射输入后复制
					auto remapped = RemapNodeInputs(calleeEntry.node, remapCalleeOutput);
					auto newId = newSg.AddNode(std::move(remapped),
					                           { calleeEntry.outputInfos.begin(), calleeEntry.outputInfos.end() });
					calleeMap[calleeNodeId] = newId;
				}
			}

			// 将 callee 的 Results 映射为 CallNode 的输出
			const auto& calleeResults = callee.Results();
			for (std::size_t port = 0; port < calleeResults.size(); ++port)
			{
				const auto& r = calleeResults[port];
				callResultMap[{ callNodeOldId, port }] = remapCalleeOutput(r);
			}
		}
	};
} // namespace LiteNN

#endif
