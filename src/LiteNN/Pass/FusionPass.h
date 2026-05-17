#include <LiteNN/Graph.h>
#include <LiteNN/Pass.h>
#include <LiteNN/Validation/GraphValidator.h>
#include <map>
#include <set>
#include <stdexcept>

#ifndef LITENN_PASS_FUSION_H
#define LITENN_PASS_FUSION_H

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
			Validation::ValidateGraph(graph);
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
					    else if constexpr (std::same_as<T, QuantizeNode>)
					    {
						    countInput(node.input);
					    }
					    else if constexpr (std::same_as<T, DequantizeNode>)
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
					    else if constexpr (std::same_as<T, GetRowsNode>)
					    {
						    countInput(node.data);
						    countInput(node.indices);
					    }
					    else if constexpr (std::same_as<T, ArgsortNode>)
					    {
						    countInput(node.input);
					    }
					    else if constexpr (std::same_as<T, MulMatIdNode>)
					    {
						    countInput(node.as);
						    countInput(node.b);
						    countInput(node.ids);
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
					    // ParamRefNode, ConstantNode, QuantizedConstantNode, VariableRefNode,
					    // LoadActivationNode, TapeLoadActivationNode: 无输入
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

			if (auto relu = TryMatchReLUConsumer(sg, ci, addId, alreadyFused))
			{
				const auto& reluEntry = sg.GetNodeEntry(*relu);
				const auto& reluInfo = reluEntry.outputInfos[0];
				if (CanLowerMatMulBiasAddReLU(reluInfo))
				{
					return FusionCandidate{
						FusionPattern::MatMulBiasAddReLU,
						{ matmulId, addId, *relu },
						{ matmul.lhs, matmul.rhs, bias, FindReLUZeroOperand(sg, *relu, { addId, 0 }).value() },
						*relu,
					};
				}
			}

			return FusionCandidate{
				FusionPattern::MatMulBiasAdd,
				{ matmulId, addId },
				{ matmul.lhs, matmul.rhs, bias }, // 外部输入: a, b, c
				addId,
			};
		}

		static bool CanLowerMatMulBiasAddReLU(const OutputInfo& info)
		{
			if (info.dtype != DataType::Float32 || info.shape.size() != 2)
			{
				return false;
			}
			const auto n = info.shape[1];
			return n > 0 && (n <= 16 || n % 16 == 0);
		}

		static bool IsZeroConstantOutput(const Subgraph& sg, NodeOutput output)
		{
			if (output.port != 0 || output.node >= sg.NodeCount())
			{
				return false;
			}

			const auto* constant = std::get_if<ConstantNode>(&sg.GetNodeEntry(output.node).node);
			if (!constant)
			{
				return false;
			}

			const auto cpuTensor = constant->value.CopyToDevice(CPU{});
			bool allZero = true;
			EnumDispatch(cpuTensor.DType(), [&]<DataType TypeValue> {
				using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
				const auto* data = static_cast<const T*>(cpuTensor.RawData());
				for (std::size_t i = 0; i < cpuTensor.NumElements(); ++i)
				{
					if (data[i] != T{})
					{
						allZero = false;
						break;
					}
				}
			});
			return allZero;
		}

		static std::optional<NodeOutput> FindReLUZeroOperand(const Subgraph& sg, NodeId reluId, NodeOutput value)
		{
			const auto& reluEntry = sg.GetNodeEntry(reluId);
			const auto* relu = std::get_if<BinaryOpNode>(&reluEntry.node);
			if (!relu || relu->op != BinaryOp::Max)
			{
				return std::nullopt;
			}

			if (relu->lhs.node == value.node && relu->lhs.port == value.port && IsZeroConstantOutput(sg, relu->rhs))
			{
				return relu->rhs;
			}
			if (relu->rhs.node == value.node && relu->rhs.port == value.port && IsZeroConstantOutput(sg, relu->lhs))
			{
				return relu->lhs;
			}
			return std::nullopt;
		}

		static std::optional<NodeId> TryMatchReLUConsumer(const Subgraph& sg, const ConsumerInfo& ci,
		                                                   NodeId addId, const std::set<NodeId>& alreadyFused)
		{
			const auto key = NodeOutputKey{ addId, 0 };
			const auto countIt = ci.counts.find(key);
			if (countIt == ci.counts.end() || countIt->second != 1)
			{
				return std::nullopt;
			}

			const auto consIt = ci.consumers.find(key);
			if (consIt == ci.consumers.end() || consIt->second.size() != 1)
			{
				return std::nullopt;
			}

			const auto reluId = consIt->second[0];
			if (alreadyFused.contains(reluId))
			{
				return std::nullopt;
			}

			if (!FindReLUZeroOperand(sg, reluId, { addId, 0 }))
			{
				return std::nullopt;
			}
			return reluId;
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
				                  std::same_as<T, QuantizedConstantNode> || std::same_as<T, VariableRefNode> ||
				                  std::same_as<T, LoadActivationNode>)
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
				    else if constexpr (std::same_as<T, QuantizeNode>)
				    {
					    return QuantizeNode{ remap(n.input), n.params };
				    }
				    else if constexpr (std::same_as<T, DequantizeNode>)
				    {
					    return DequantizeNode{ remap(n.input), n.params, n.targetType };
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
				    else if constexpr (std::same_as<T, GetRowsNode>)
				    {
					    return GetRowsNode{ remap(n.data), remap(n.indices) };
				    }
				    else if constexpr (std::same_as<T, ArgsortNode>)
				    {
					    return ArgsortNode{ remap(n.input), n.axis, n.order };
				    }
				    else if constexpr (std::same_as<T, MulMatIdNode>)
				    {
					    return MulMatIdNode{ remap(n.as), remap(n.b), remap(n.ids) };
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

			// 构建融合节点集合
			std::set<NodeId> fusedNodeSet;
			for (const auto& c : candidates)
			{
				for (auto id : c.fusedNodeIds)
				{
					fusedNodeSet.insert(id);
				}
			}

			// outputNodeId → candidate 索引（FusedOpNode 在此位置发射）
			std::map<NodeId, std::size_t> outputNodeToCandidate;
			for (std::size_t i = 0; i < candidates.size(); ++i)
			{
				outputNodeToCandidate[candidates[i].outputNodeId] = i;
			}

			// 单趟重建：按原始拓扑序扫描
			//   · ParamRefNode / 非融合节点：立即复制（其所有输入已就绪）
			//   · 融合区域的 outputNodeId：在此处发射 FusedOpNode
			//     （保证所有外部输入均已处理：外部输入在拓扑序上先于 outputNodeId）
			//   · 融合区域内其他节点（非 outputNodeId）：跳过
			Subgraph newSg;
			std::vector<NodeId> nodeMap(sg.NodeCount(), static_cast<NodeId>(-1));

			for (NodeId oldId = 0; oldId < sg.NodeCount(); ++oldId)
			{
				const auto& entry = sg.GetNodeEntry(oldId);

				// 融合区域输出节点：发射 FusedOpNode
				if (auto outIt = outputNodeToCandidate.find(oldId); outIt != outputNodeToCandidate.end())
				{
					const auto& candidate = candidates[outIt->second];
					auto bodyId = BuildBodySubgraph(graph, sg, candidate);

					std::vector<NodeOutput> remappedArgs;
					for (const auto& input : candidate.externalInputs)
					{
						remappedArgs.push_back({ nodeMap[input.node], input.port });
					}

					const auto& outputInfos = entry.outputInfos;
					nodeMap[oldId] = newSg.AddNode(
					    FusedOpNode{ candidate.pattern, bodyId, std::move(remappedArgs) },
					    { outputInfos.begin(), outputInfos.end() });
					continue;
				}

				// 融合区域内部节点（非输出端）：跳过
				if (fusedNodeSet.contains(oldId))
				{
					continue;
				}

				// ParamRefNode
				if (const auto* paramRef = std::get_if<ParamRefNode>(&entry.node))
				{
					const auto& param = sg.Params()[paramRef->paramIndex];
					nodeMap[oldId] = newSg.AddParam(param.dtype, param.shape);
					continue;
				}

				// 普通非融合节点：重映射输入后复制
				auto remapFn = [&](NodeOutput output) -> NodeOutput {
					return { nodeMap[output.node], output.port };
				};
				auto remapped = RemapNodeInputs(entry.node, remapFn);
				nodeMap[oldId] = newSg.AddNode(std::move(remapped),
				                               { entry.outputInfos.begin(), entry.outputInfos.end() });
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
