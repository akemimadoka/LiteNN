#ifndef LITENN_PASS_FUSION_H
#define LITENN_PASS_FUSION_H

#include <LiteNN/Graph.h>
#include <LiteNN/Pass.h>
#include <map>
#include <set>
#include <stdexcept>

namespace LiteNN
{
	// 算子融合 Pass：识别可融合的操作模式，将多个节点合并为 FusedOpNode
	// 语义不变：FusedOpNode 持有 body 子图，Interpreter 直接执行 body；AOT 编译器可据 pattern 生成优化内核
	// 应在 AutogradPass 之后运行
	class FusionPass : public Pass
	{
	public:
		void Run(Graph& graph) override
		{
			// 只处理运行前已有的子图，避免处理新创建的 body 子图
			const auto originalCount = graph.SubgraphCount();
			for (std::size_t sgId = 0; sgId < originalCount; ++sgId)
			{
				ProcessSubgraph(graph, sgId);
			}
		}

	private:
		using NodeOutputKey = std::pair<NodeId, std::size_t>;

		struct FusionCandidate
		{
			FusionPattern pattern;
			std::vector<NodeId> fusedNodeIds;       // 拓扑序
			std::vector<NodeOutput> externalInputs;  // 去重，按发现顺序
			NodeId outputNodeId;                     // 融合区域的最后节点
		};

		// ---- 逐元素操作判定 ----

		static bool IsElementWiseBinaryOp(BinaryOp op)
		{
			switch (op)
			{
			case BinaryOp::Add:
			case BinaryOp::Subtract:
			case BinaryOp::Multiply:
			case BinaryOp::Divide:
			case BinaryOp::Pow:
			case BinaryOp::Max:
			case BinaryOp::Min:
				return true;
			default:
				return false;
			}
		}

		static bool IsElementWiseUnaryOp(UnaryOp op)
		{
			return op != UnaryOp::Transpose;
		}

		static bool IsElementWiseNode(const NodeVariant& node)
		{
			if (auto* u = std::get_if<UnaryOpNode>(&node))
			{
				return IsElementWiseUnaryOp(u->op);
			}
			if (auto* b = std::get_if<BinaryOpNode>(&node))
			{
				return IsElementWiseBinaryOp(b->op);
			}
			return false;
		}

		// ---- Consumer 分析 ----

		struct ConsumerInfo
		{
			std::map<NodeOutputKey, std::size_t> counts;
			std::map<NodeOutputKey, std::vector<NodeId>> consumers; // 正向消费者索引
		};

		static ConsumerInfo AnalyzeConsumers(const Subgraph& sg)
		{
			ConsumerInfo info;

			// 统计 Results 中的消费
			for (const auto& r : sg.Results())
			{
				++info.counts[{ r.node, r.port }];
			}

			for (NodeId nodeId = 0; nodeId < sg.NodeCount(); ++nodeId)
			{
				auto countInput = [&](NodeOutput input) {
					auto key = NodeOutputKey{ input.node, input.port };
					++info.counts[key];
					info.consumers[key].push_back(nodeId);
				};

				std::visit(
				    [&](const auto& node) {
					    using T = std::decay_t<decltype(node)>;
					    if constexpr (std::same_as<T, UnaryOpNode>)
					    {
						    countInput(node.input);
					    }
					    else if constexpr (std::same_as<T, BinaryOpNode>)
					    {
						    countInput(node.lhs);
						    countInput(node.rhs);
					    }
					    else if constexpr (std::same_as<T, CastNode>)
					    {
						    countInput(node.input);
					    }
					    else if constexpr (std::same_as<T, CallNode>)
					    {
						    for (const auto& a : node.args)
						    {
							    countInput(a);
						    }
					    }
					    else if constexpr (std::same_as<T, CondNode>)
					    {
						    countInput(node.condition);
						    for (const auto& a : node.args)
						    {
							    countInput(a);
						    }
					    }
					    else if constexpr (std::same_as<T, ReduceOpNode>)
					    {
						    countInput(node.input);
					    }
					    else if constexpr (std::same_as<T, ReshapeNode>)
					    {
						    countInput(node.input);
					    }
					    else if constexpr (std::same_as<T, ConcatNode>)
					    {
						    for (const auto& input : node.inputs)
						    {
							    countInput(input);
						    }
					    }
					    else if constexpr (std::same_as<T, SliceNode>)
					    {
						    countInput(node.input);
					    }
					    else if constexpr (std::same_as<T, SaveActivationNode>)
					    {
						    countInput(node.input);
					    }
					    else if constexpr (std::same_as<T, FusedOpNode>)
					    {
						    for (const auto& a : node.args)
						    {
							    countInput(a);
						    }
					    }
					    else if constexpr (std::same_as<T, WhileNode>)
					    {
						    for (const auto& a : node.initArgs)
						    {
							    countInput(a);
						    }
					    }
					    else if constexpr (std::same_as<T, TapeSaveActivationNode>)
					    {
						    countInput(node.input);
					    }
					    // ParamRefNode, ConstantNode, VariableRefNode, LoadActivationNode, TapeLoadActivationNode: 无输入
				    },
				    sg.GetNodeEntry(nodeId).node);
			}
			return info;
		}

		// ---- 模式检测 ----

		static std::vector<FusionCandidate> DetectCandidates(const Subgraph& sg, const ConsumerInfo& ci)
		{
			std::vector<FusionCandidate> candidates;
			std::set<NodeId> alreadyFused;

			for (NodeId nodeId = 0; nodeId < sg.NodeCount(); ++nodeId)
			{
				if (alreadyFused.contains(nodeId))
				{
					continue;
				}

				const auto& entry = sg.GetNodeEntry(nodeId);

				// 优先匹配 MatMulBiasAdd
				if (auto* bin = std::get_if<BinaryOpNode>(&entry.node);
				    bin && bin->op == BinaryOp::MatMul)
				{
					if (auto c = TryMatchMatMulBiasAdd(sg, ci, nodeId, *bin, alreadyFused))
					{
						for (auto id : c->fusedNodeIds)
						{
							alreadyFused.insert(id);
						}
						candidates.push_back(std::move(*c));
						continue;
					}
				}

				// 匹配逐元素链
				if (IsElementWiseNode(entry.node))
				{
					if (auto c = TryMatchElementWiseChain(sg, ci, nodeId, alreadyFused))
					{
						for (auto id : c->fusedNodeIds)
						{
							alreadyFused.insert(id);
						}
						candidates.push_back(std::move(*c));
					}
				}
			}

			return candidates;
		}

		static std::optional<FusionCandidate> TryMatchMatMulBiasAdd(const Subgraph& sg, const ConsumerInfo& ci,
		                                                             NodeId matmulId, const BinaryOpNode& matmul,
		                                                             const std::set<NodeId>& alreadyFused)
		{
			auto key = NodeOutputKey{ matmulId, 0 };
			auto countIt = ci.counts.find(key);
			if (countIt == ci.counts.end() || countIt->second != 1)
			{
				return std::nullopt; // MatMul 输出有多个消费者
			}

			auto consIt = ci.consumers.find(key);
			if (consIt == ci.consumers.end() || consIt->second.size() != 1)
			{
				return std::nullopt;
			}

			const auto addId = consIt->second[0];
			if (alreadyFused.contains(addId))
			{
				return std::nullopt;
			}

			const auto& addEntry = sg.GetNodeEntry(addId);
			auto* addNode = std::get_if<BinaryOpNode>(&addEntry.node);
			if (!addNode || addNode->op != BinaryOp::Add)
			{
				return std::nullopt;
			}

			// 确定 bias 操作数（Add 是交换的）
			NodeOutput bias;
			if (addNode->lhs.node == matmulId && addNode->lhs.port == 0)
			{
				bias = addNode->rhs;
			}
			else if (addNode->rhs.node == matmulId && addNode->rhs.port == 0)
			{
				bias = addNode->lhs;
			}
			else
			{
				return std::nullopt;
			}

			return FusionCandidate{
				FusionPattern::MatMulBiasAdd,
				{ matmulId, addId },
				{ matmul.lhs, matmul.rhs, bias }, // 外部输入: a, b, c
				addId,
			};
		}

		static std::optional<FusionCandidate> TryMatchElementWiseChain(const Subgraph& sg, const ConsumerInfo& ci,
		                                                                NodeId startId,
		                                                                const std::set<NodeId>& alreadyFused)
		{
			std::vector<NodeId> chain;
			chain.push_back(startId);

			auto currentId = startId;
			while (true)
			{
				auto key = NodeOutputKey{ currentId, 0 };
				auto countIt = ci.counts.find(key);
				if (countIt == ci.counts.end() || countIt->second != 1)
				{
					break; // 多消费者，断链
				}

				auto consIt = ci.consumers.find(key);
				if (consIt == ci.consumers.end() || consIt->second.size() != 1)
				{
					break;
				}

				const auto nextId = consIt->second[0];
				if (alreadyFused.contains(nextId))
				{
					break;
				}

				const auto& nextEntry = sg.GetNodeEntry(nextId);
				if (!IsElementWiseNode(nextEntry.node))
				{
					break;
				}

				chain.push_back(nextId);
				currentId = nextId;
			}

			if (chain.size() < 2)
			{
				return std::nullopt;
			}

			auto externalInputs = CollectExternalInputs(sg, chain);

			return FusionCandidate{
				FusionPattern::ElementWiseChain,
				std::move(chain),
				std::move(externalInputs),
				currentId,
			};
		}

		// ---- 外部输入收集 ----

		static std::vector<NodeOutput> CollectExternalInputs(const Subgraph& sg, const std::vector<NodeId>& fusedIds)
		{
			std::set<NodeId> fusedSet(fusedIds.begin(), fusedIds.end());
			std::vector<NodeOutput> inputs;
			std::set<NodeOutputKey> seen;

			auto tryAdd = [&](NodeOutput output) {
				if (!fusedSet.contains(output.node))
				{
					auto key = NodeOutputKey{ output.node, output.port };
					if (!seen.contains(key))
					{
						seen.insert(key);
						inputs.push_back(output);
					}
				}
			};

			for (auto nodeId : fusedIds)
			{
				const auto& entry = sg.GetNodeEntry(nodeId);
				std::visit(
				    [&](const auto& node) {
					    using T = std::decay_t<decltype(node)>;
					    if constexpr (std::same_as<T, UnaryOpNode>)
					    {
						    tryAdd(node.input);
					    }
					    else if constexpr (std::same_as<T, BinaryOpNode>)
					    {
						    tryAdd(node.lhs);
						    tryAdd(node.rhs);
					    }
					    else if constexpr (std::same_as<T, CastNode>)
					    {
						    tryAdd(node.input);
					    }
				    },
				    entry.node);
			}

			return inputs;
		}

		// ---- Body 子图构建 ----

		static SubgraphId BuildBodySubgraph(Graph& graph, const Subgraph& originalSg,
		                                     const FusionCandidate& candidate)
		{
			Subgraph body;
			std::set<NodeId> fusedSet(candidate.fusedNodeIds.begin(), candidate.fusedNodeIds.end());

			// 外部输入 → body 参数
			std::map<NodeOutputKey, NodeId> externalToParam;
			for (const auto& input : candidate.externalInputs)
			{
				const auto& info = originalSg.GetOutputInfo(input);
				auto paramId = body.AddParam(info.dtype, info.shape);
				externalToParam[{ input.node, input.port }] = paramId;
			}

			// 融合节点 → body 节点
			std::map<NodeId, NodeId> internalMap;

			auto remapInput = [&](NodeOutput output) -> NodeOutput {
				if (fusedSet.contains(output.node))
				{
					return { internalMap.at(output.node), output.port };
				}
				else
				{
					auto paramId = externalToParam.at({ output.node, output.port });
					return { paramId, 0 };
				}
			};

			for (auto nodeId : candidate.fusedNodeIds)
			{
				const auto& entry = originalSg.GetNodeEntry(nodeId);
				auto remapped = RemapNodeInputs(entry.node, remapInput);
				auto newId = body.AddNode(std::move(remapped),
				                          { entry.outputInfos.begin(), entry.outputInfos.end() });
				internalMap[nodeId] = newId;
			}

			body.SetResults({ { internalMap.at(candidate.outputNodeId), 0 } });
			return graph.AddSubgraph(std::move(body));
		}

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
					    throw std::runtime_error("FusionPass: unsupported node type in remap");
				    }
			    },
			    node);
		}

		// ---- 子图重写 ----

		void ProcessSubgraph(Graph& graph, SubgraphId sgId)
		{
			const auto& sg = graph.GetSubgraph(sgId);
			auto ci = AnalyzeConsumers(sg);
			auto candidates = DetectCandidates(sg, ci);

			if (candidates.empty())
			{
				return;
			}

			// 构建融合节点集合和起始节点映射
			std::set<NodeId> fusedNodeSet;
			std::map<NodeId, std::size_t> fusedRegionStart; // 起始节点 → candidate 索引
			for (std::size_t i = 0; i < candidates.size(); ++i)
			{
				fusedRegionStart[candidates[i].fusedNodeIds[0]] = i;
				for (auto id : candidates[i].fusedNodeIds)
				{
					fusedNodeSet.insert(id);
				}
			}

			// 重建子图
			Subgraph newSg;
			std::vector<NodeId> nodeMap(sg.NodeCount(), static_cast<NodeId>(-1));

			for (NodeId oldId = 0; oldId < sg.NodeCount(); ++oldId)
			{
				// 跳过融合区域内部的非起始节点
				if (fusedNodeSet.contains(oldId) && !fusedRegionStart.contains(oldId))
				{
					continue;
				}

				auto regionIt = fusedRegionStart.find(oldId);
				if (regionIt != fusedRegionStart.end())
				{
					// 融合区域起始节点：构建 body 子图并创建 FusedOpNode
					const auto& candidate = candidates[regionIt->second];
					auto bodyId = BuildBodySubgraph(graph, sg, candidate);

					// 重映射外部输入引用
					std::vector<NodeOutput> remappedArgs;
					for (const auto& input : candidate.externalInputs)
					{
						remappedArgs.push_back({ nodeMap[input.node], input.port });
					}

					const auto& outputInfos = sg.GetNodeEntry(candidate.outputNodeId).outputInfos;
					auto newId = newSg.AddNode(
					    FusedOpNode{ candidate.pattern, bodyId, std::move(remappedArgs) },
					    { outputInfos.begin(), outputInfos.end() });

					// 映射输出节点（融合区域的最终输出）
					nodeMap[candidate.outputNodeId] = newId;
				}
				else
				{
					// 非融合节点：重映射输入后复制
					const auto& entry = sg.GetNodeEntry(oldId);
					auto remapFn = [&](NodeOutput output) -> NodeOutput {
						return { nodeMap[output.node], output.port };
					};
					auto remapped = RemapNodeInputs(entry.node, remapFn);
					auto newId = newSg.AddNode(std::move(remapped),
					                           { entry.outputInfos.begin(), entry.outputInfos.end() });
					nodeMap[oldId] = newId;
				}
			}

			// 重映射结果
			std::vector<NodeOutput> newResults;
			for (const auto& r : sg.Results())
			{
				newResults.push_back({ nodeMap[r.node], r.port });
			}
			newSg.SetResults(std::move(newResults));

			graph.GetSubgraph(sgId) = std::move(newSg);
		}
	};
} // namespace LiteNN

#endif
