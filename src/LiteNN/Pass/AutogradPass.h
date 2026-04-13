#ifndef LITENN_PASS_AUTOGRAD_H
#define LITENN_PASS_AUTOGRAD_H

#include <LiteNN/Graph.h>
#include <LiteNN/Pass.h>
#include <map>
#include <stdexcept>

namespace LiteNN
{
	// 自动微分 Pass：对 forward 子图生成 backward 子图
	// 采用保存中间激活值方式，通过 ActivationStore 共享前向/反向数据
	class AutogradPass : public Pass
	{
	public:
		void Run(Graph& graph) override
		{
			processed_.clear();
			auto info = ProcessSubgraph(graph, graph.Forward());
			graph.SetForward(info.augForwardId);
			graph.SetBackward(info.backwardId);
		}

	private:
		using NodeOutputKey = std::pair<NodeId, std::size_t>; // (nodeId, port)

		struct VarGradEntry
		{
			std::size_t variableIndex;
			OutputInfo outputInfo;
		};

		struct SubgraphGradInfo
		{
			SubgraphId augForwardId;
			SubgraphId backwardId;
			std::size_t numInputGrads;               // backward 输出中 input 梯度数量
			std::vector<VarGradEntry> variableGrads; // backward 输出中 variable 梯度（按 variableIndex 排序）
		};

		std::map<SubgraphId, SubgraphGradInfo> processed_;

		// 对任意子图生成 augmented forward 和 backward，支持递归处理 CallNode
		SubgraphGradInfo ProcessSubgraph(Graph& graph, SubgraphId fwdId)
		{
			if (auto it = processed_.find(fwdId); it != processed_.end())
			{
				return it->second;
			}

			const auto& fwdSg = graph.GetSubgraph(fwdId);

			// 1. 分析

			std::map<NodeOutputKey, std::size_t> consumerCount;
			CountConsumers(fwdSg, consumerCount);

			std::map<NodeOutputKey, std::size_t> savedActivations;
			AnalyzeSavedValues(fwdSg, graph, savedActivations);

			// 递归处理所有 callee，收集映射
			std::map<SubgraphId, SubgraphGradInfo> calleeInfo;
			for (NodeId nodeId = 0; nodeId < fwdSg.NodeCount(); ++nodeId)
			{
				const auto& entry = fwdSg.GetNodeEntry(nodeId);
				if (auto* cn = std::get_if<CallNode>(&entry.node))
				{
					if (!calleeInfo.contains(cn->callee))
					{
						calleeInfo[cn->callee] = ProcessSubgraph(graph, cn->callee);
					}
				}
				else if (auto* cond = std::get_if<CondNode>(&entry.node))
				{
					if (!calleeInfo.contains(cond->thenBranch))
					{
						calleeInfo[cond->thenBranch] = ProcessSubgraph(graph, cond->thenBranch);
					}
					if (!calleeInfo.contains(cond->elseBranch))
					{
						calleeInfo[cond->elseBranch] = ProcessSubgraph(graph, cond->elseBranch);
					}
				}
			}

			// callee augmented forward 映射
			std::map<SubgraphId, SubgraphId> calleeAugFwdMap;
			for (const auto& [origId, info] : calleeInfo)
			{
				calleeAugFwdMap[origId] = info.augForwardId;
			}

			// 2. 构建 augmented forward

			Subgraph augFwd;
			std::vector<NodeId> augNodeMap(fwdSg.NodeCount());
			BuildAugmentedForward(fwdSg, savedActivations, calleeAugFwdMap, augFwd, augNodeMap);
			const auto augFwdId = graph.AddSubgraph(std::move(augFwd));

			// 3. 构建 backward

			Subgraph bwdSg;

			// backward 参数: [forward_inputs..., grad_outputs...]
			const auto numFwdParams = fwdSg.Params().size();
			std::vector<NodeId> fwdInputInBwd(numFwdParams);
			for (std::size_t i = 0; i < numFwdParams; ++i)
			{
				const auto& param = fwdSg.Params()[i];
				fwdInputInBwd[i] = bwdSg.AddParam(param.dtype, param.shape);
			}

			const auto numFwdOutputs = fwdSg.Results().size();
			std::vector<NodeId> gradOutputParams(numFwdOutputs);
			for (std::size_t i = 0; i < numFwdOutputs; ++i)
			{
				const auto& info = fwdSg.GetOutputInfo(fwdSg.Results()[i]);
				gradOutputParams[i] = bwdSg.AddParam(info.dtype, info.shape);
			}

			std::map<NodeOutputKey, std::vector<NodeOutput>> gradContributions;
			std::map<NodeOutputKey, NodeId> loadMap;

			// 预填充 loadMap: ParamRef → backward param
			for (NodeId fwdNodeId = 0; fwdNodeId < fwdSg.NodeCount(); ++fwdNodeId)
			{
				if (auto* prn = std::get_if<ParamRefNode>(&fwdSg.GetNodeEntry(fwdNodeId).node))
				{
					loadMap[{ fwdNodeId, 0 }] = fwdInputInBwd[prn->paramIndex];
				}
			}

			// 初始化梯度种子
			for (std::size_t i = 0; i < numFwdOutputs; ++i)
			{
				const auto& r = fwdSg.Results()[i];
				gradContributions[{ r.node, r.port }].push_back({ gradOutputParams[i], 0 });
			}

			// 逆拓扑序生成梯度
			std::map<std::size_t, std::vector<NodeOutput>> varGradContribs;
			BuildBackwardNodes(fwdSg, graph, calleeInfo, savedActivations, bwdSg, gradContributions, loadMap,
			                   varGradContribs);

			// 收集 backward 结果
			std::vector<NodeOutput> bwdResults;
			std::size_t numInputGrads = 0;

			for (NodeId fwdNodeId = 0; fwdNodeId < fwdSg.NodeCount(); ++fwdNodeId)
			{
				if (std::holds_alternative<ParamRefNode>(fwdSg.GetNodeEntry(fwdNodeId).node))
				{
					bwdResults.push_back(ResolveGrad(fwdSg, bwdSg, fwdNodeId, 0, gradContributions));
					++numInputGrads;
				}
			}

			// 合并直接出现的 VariableRefNode 梯度和 callee 传播的 variable 梯度
			std::map<std::size_t, std::pair<OutputInfo, std::vector<NodeOutput>>> allVarContribs;

			for (NodeId fwdNodeId = 0; fwdNodeId < fwdSg.NodeCount(); ++fwdNodeId)
			{
				if (auto* vrn = std::get_if<VariableRefNode>(&fwdSg.GetNodeEntry(fwdNodeId).node))
				{
					auto resolved = ResolveGrad(fwdSg, bwdSg, fwdNodeId, 0, gradContributions);
					auto& entry = allVarContribs[vrn->variableIndex];
					entry.first = fwdSg.GetNodeEntry(fwdNodeId).outputInfos[0];
					entry.second.push_back(resolved);
				}
			}

			for (auto& [varIdx, contribs] : varGradContribs)
			{
				auto& entry = allVarContribs[varIdx];
				if (entry.first.shape.empty() && !contribs.empty())
				{
					// 从 calleeInfo 获取 OutputInfo（此 variable 不直接出现在当前子图中）
					for (const auto& [_, ci] : calleeInfo)
					{
						for (const auto& vg : ci.variableGrads)
						{
							if (vg.variableIndex == varIdx)
							{
								entry.first = vg.outputInfo;
								goto found;
							}
						}
					}
				found:;
				}
				for (auto& c : contribs)
				{
					entry.second.push_back(c);
				}
			}

			std::vector<VarGradEntry> variableGrads;
			for (auto& [varIdx, entry] : allVarContribs)
			{
				auto resolved = SumGradContributions(bwdSg, entry.second, entry.first);
				bwdResults.push_back(resolved);
				variableGrads.push_back({ varIdx, entry.first });
			}

			bwdSg.SetResults(std::move(bwdResults));
			const auto bwdId = graph.AddSubgraph(std::move(bwdSg));

			SubgraphGradInfo result{ augFwdId, bwdId, numInputGrads, std::move(variableGrads) };
			processed_[fwdId] = result;
			return result;
		}

		// 分析

		static void CountConsumers(const Subgraph& sg, std::map<NodeOutputKey, std::size_t>& counts)
		{
			for (NodeId nodeId = 0; nodeId < sg.NodeCount(); ++nodeId)
			{
				std::visit(
				    [&](const auto& node) {
					    using T = std::decay_t<decltype(node)>;
					    if constexpr (std::same_as<T, UnaryOpNode>)
					    {
						    ++counts[{ node.input.node, node.input.port }];
					    }
					    else if constexpr (std::same_as<T, BinaryOpNode>)
					    {
						    ++counts[{ node.lhs.node, node.lhs.port }];
						    ++counts[{ node.rhs.node, node.rhs.port }];
					    }
					    else if constexpr (std::same_as<T, CastNode>)
					    {
						    ++counts[{ node.input.node, node.input.port }];
					    }
					    else if constexpr (std::same_as<T, CallNode>)
					    {
						    for (const auto& a : node.args)
						    {
							    ++counts[{ a.node, a.port }];
						    }
					    }
					    else if constexpr (std::same_as<T, ReduceOpNode>)
					    {
						    ++counts[{ node.input.node, node.input.port }];
					    }
					    else if constexpr (std::same_as<T, ReshapeNode>)
					    {
						    ++counts[{ node.input.node, node.input.port }];
					    }
					    else if constexpr (std::same_as<T, CondNode>)
					    {
						    ++counts[{ node.condition.node, node.condition.port }];
						    for (const auto& a : node.args)
						    {
							    ++counts[{ a.node, a.port }];
						    }
					    }
				    },
				    sg.GetNodeEntry(nodeId).node);
			}
		}

		static void AnalyzeSavedValues(const Subgraph& fwdSg, Graph& graph, std::map<NodeOutputKey, std::size_t>& saved)
		{
			for (NodeId nodeId = 0; nodeId < fwdSg.NodeCount(); ++nodeId)
			{
				const auto& entry = fwdSg.GetNodeEntry(nodeId);
				std::visit(
				    [&](const auto& node) {
					    using T = std::decay_t<decltype(node)>;
					    if constexpr (std::same_as<T, UnaryOpNode>)
					    {
						    if (node.op == UnaryOp::Abs || node.op == UnaryOp::Log || node.op == UnaryOp::Sin ||
						        node.op == UnaryOp::Cos || node.op == UnaryOp::Tan || node.op == UnaryOp::Arcsin ||
						        node.op == UnaryOp::Arccos || node.op == UnaryOp::Arctan)
						    {
							    SaveIfNeeded(fwdSg, graph, node.input, saved);
						    }
						    else if (node.op == UnaryOp::Sqrt || node.op == UnaryOp::Exp)
						    {
							    SaveIfNeeded(fwdSg, graph, { nodeId, 0 }, saved); // 保存输出
						    }
					    }
					    else if constexpr (std::same_as<T, BinaryOpNode>)
					    {
						    if (node.op == BinaryOp::Multiply || node.op == BinaryOp::MatMul ||
						        node.op == BinaryOp::Divide || node.op == BinaryOp::Max || node.op == BinaryOp::Min)
						    {
							    SaveIfNeeded(fwdSg, graph, node.lhs, saved);
							    SaveIfNeeded(fwdSg, graph, node.rhs, saved);
						    }
						    else if (node.op == BinaryOp::Pow)
						    {
							    SaveIfNeeded(fwdSg, graph, node.lhs, saved);
							    SaveIfNeeded(fwdSg, graph, node.rhs, saved);
							    SaveIfNeeded(fwdSg, graph, { nodeId, 0 }, saved); // 保存输出 a^b
						    }
					    }
					    else if constexpr (std::same_as<T, ReduceOpNode>)
					    {
						    if (node.op == ReduceOp::Max)
						    {
							    SaveIfNeeded(fwdSg, graph, node.input, saved);
							    SaveIfNeeded(fwdSg, graph, { nodeId, 0 }, saved);
						    }
						    // Sum/Mean 不需要保存前向值
					    }
					    else if constexpr (std::same_as<T, CallNode>)
					    {
						    // CallNode 的 args 在 backward 中需要传给 callee backward
						    for (const auto& a : node.args)
						    {
							    SaveIfNeeded(fwdSg, graph, a, saved);
						    }
					    }
					    else if constexpr (std::same_as<T, CondNode>)
					    {
						    // 保存 condition（backward 需要重新判断走哪个分支）
						    SaveIfNeeded(fwdSg, graph, node.condition, saved);
						    // 保存所有 args（传入分支 backward）
						    for (const auto& a : node.args)
						    {
							    SaveIfNeeded(fwdSg, graph, a, saved);
						    }
					    }
				    },
				    entry.node);
			}
		}

		static void SaveIfNeeded(const Subgraph& fwdSg, Graph& graph, NodeOutput output,
		                         std::map<NodeOutputKey, std::size_t>& saved)
		{
			auto key = std::make_pair(output.node, output.port);
			if (saved.contains(key))
			{
				return;
			}
			const auto& prodNode = fwdSg.GetNodeEntry(output.node).node;
			if (std::holds_alternative<ParamRefNode>(prodNode) || std::holds_alternative<VariableRefNode>(prodNode) ||
			    std::holds_alternative<ConstantNode>(prodNode))
			{
				return;
			}
			const auto& info = fwdSg.GetNodeEntry(output.node).outputInfos[output.port];
			saved[key] = graph.AddActivationSlot({ info.dtype, info.shape });
		}

		// Augmented forward

		static void BuildAugmentedForward(const Subgraph& fwdSg, const std::map<NodeOutputKey, std::size_t>& saved,
		                                  const std::map<SubgraphId, SubgraphId>& calleeAugFwdMap, Subgraph& augFwd,
		                                  std::vector<NodeId>& augNodeMap)
		{
			for (NodeId fwdNodeId = 0; fwdNodeId < fwdSg.NodeCount(); ++fwdNodeId)
			{
				const auto& entry = fwdSg.GetNodeEntry(fwdNodeId);
				auto remapped = RemapNode(entry.node, augNodeMap, calleeAugFwdMap);
				auto augNodeId =
				    augFwd.AddNode(std::move(remapped), { entry.outputInfos.begin(), entry.outputInfos.end() });

				for (std::size_t port = 0; port < entry.outputInfos.size(); ++port)
				{
					auto it = saved.find({ fwdNodeId, port });
					if (it != saved.end() && entry.outputInfos.size() == 1)
					{
						const auto& info = entry.outputInfos[port];
						augNodeId = augFwd.AddNode(SaveActivationNode{ { augNodeId, port }, it->second },
						                           { OutputInfo{ info.dtype, info.shape } });
					}
				}
				augNodeMap[fwdNodeId] = augNodeId;
			}

			std::vector<NodeOutput> augResults;
			for (const auto& r : fwdSg.Results())
			{
				augResults.push_back({ augNodeMap[r.node], r.port });
			}
			augFwd.SetResults(std::move(augResults));
		}

		static NodeVariant RemapNode(const NodeVariant& node, const std::vector<NodeId>& nodeMap,
		                             const std::map<SubgraphId, SubgraphId>& calleeAugMap)
		{
			return std::visit(
			    [&](const auto& n) -> NodeVariant {
				    using T = std::decay_t<decltype(n)>;
				    if constexpr (std::same_as<T, ParamRefNode> || std::same_as<T, ConstantNode> ||
				                  std::same_as<T, VariableRefNode>)
				    {
					    return n;
				    }
				    else if constexpr (std::same_as<T, UnaryOpNode>)
				    {
					    return UnaryOpNode{ n.op, { nodeMap[n.input.node], n.input.port } };
				    }
				    else if constexpr (std::same_as<T, BinaryOpNode>)
				    {
					    return BinaryOpNode{ n.op,
						                     { nodeMap[n.lhs.node], n.lhs.port },
						                     { nodeMap[n.rhs.node], n.rhs.port } };
				    }
				    else if constexpr (std::same_as<T, CastNode>)
				    {
					    return CastNode{ { nodeMap[n.input.node], n.input.port }, n.targetType };
				    }
				    else if constexpr (std::same_as<T, ReduceOpNode>)
				    {
					    return ReduceOpNode{ n.op, { nodeMap[n.input.node], n.input.port }, n.axis };
				    }
				    else if constexpr (std::same_as<T, ReshapeNode>)
				    {
					    return ReshapeNode{ { nodeMap[n.input.node], n.input.port }, n.targetShape };
				    }
				    else if constexpr (std::same_as<T, CallNode>)
				    {
					    std::vector<NodeOutput> args;
					    for (const auto& a : n.args)
					    {
						    args.push_back({ nodeMap[a.node], a.port });
					    }
					    auto it = calleeAugMap.find(n.callee);
					    auto callee = it != calleeAugMap.end() ? it->second : n.callee;
					    return CallNode{ callee, std::move(args) };
				    }
				    else if constexpr (std::same_as<T, CondNode>)
				    {
					    std::vector<NodeOutput> args;
					    for (const auto& a : n.args)
					    {
						    args.push_back({ nodeMap[a.node], a.port });
					    }
					    auto thenIt = calleeAugMap.find(n.thenBranch);
					    auto thenBranch = thenIt != calleeAugMap.end() ? thenIt->second : n.thenBranch;
					    auto elseIt = calleeAugMap.find(n.elseBranch);
					    auto elseBranch = elseIt != calleeAugMap.end() ? elseIt->second : n.elseBranch;
					    return CondNode{
						    { nodeMap[n.condition.node], n.condition.port }, thenBranch, elseBranch, std::move(args)
					    };
				    }
				    else
				    {
					    throw std::runtime_error("AutogradPass: unsupported node type in remap");
				    }
			    },
			    node);
		}

		// Backward 构建

		NodeId GetForwardValue(const Subgraph& fwdSg, Subgraph& bwdSg, NodeId fwdNodeId, std::size_t port,
		                       const std::map<NodeOutputKey, std::size_t>& saved,
		                       std::map<NodeOutputKey, NodeId>& loadMap)
		{
			auto key = std::make_pair(fwdNodeId, port);
			if (auto it = loadMap.find(key); it != loadMap.end())
			{
				return it->second;
			}

			const auto& entry = fwdSg.GetNodeEntry(fwdNodeId);
			const auto& info = entry.outputInfos[port];

			if (std::holds_alternative<VariableRefNode>(entry.node))
			{
				auto& vrn = std::get<VariableRefNode>(entry.node);
				auto id = bwdSg.AddNode(VariableRefNode{ vrn.variableIndex }, { OutputInfo{ info.dtype, info.shape } });
				loadMap[key] = id;
				return id;
			}
			if (std::holds_alternative<ConstantNode>(entry.node))
			{
				auto& cn = std::get<ConstantNode>(entry.node);
				auto id = bwdSg.AddNode(ConstantNode{ Tensor<PolymorphicDevice>(cn.value) },
				                        { OutputInfo{ info.dtype, info.shape } });
				loadMap[key] = id;
				return id;
			}

			if (auto it = saved.find(key); it != saved.end())
			{
				auto id = bwdSg.AddNode(LoadActivationNode{ it->second }, { OutputInfo{ info.dtype, info.shape } });
				loadMap[key] = id;
				return id;
			}

			throw std::runtime_error("AutogradPass: forward value needed but not saved and not accessible");
		}

		static NodeOutput SumGradContributions(Subgraph& bwdSg, const std::vector<NodeOutput>& contribs,
		                                       const OutputInfo& info)
		{
			if (contribs.empty())
			{
				auto zeroTensor = Tensor<CPU>(info.shape, info.dtype);
				auto id = bwdSg.AddNode(ConstantNode{ zeroTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
				                        { OutputInfo{ info.dtype, info.shape } });
				return { id, 0 };
			}

			if (contribs.size() == 1)
			{
				return contribs[0];
			}

			auto acc = contribs[0];
			for (std::size_t i = 1; i < contribs.size(); ++i)
			{
				auto id = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Add, acc, contribs[i] },
				                        { OutputInfo{ info.dtype, info.shape } });
				acc = { id, 0 };
			}
			return acc;
		}

		static NodeOutput ResolveGrad(const Subgraph& fwdSg, Subgraph& bwdSg, NodeId fwdNodeId, std::size_t port,
		                              std::map<NodeOutputKey, std::vector<NodeOutput>>& gradContribs)
		{
			auto key = std::make_pair(fwdNodeId, port);
			auto it = gradContribs.find(key);
			const auto& info = fwdSg.GetNodeEntry(fwdNodeId).outputInfos[port];
			if (it == gradContribs.end())
			{
				return SumGradContributions(bwdSg, {}, info);
			}
			return SumGradContributions(bwdSg, it->second, info);
		}

		void BuildBackwardNodes(const Subgraph& fwdSg, Graph& graph,
		                        const std::map<SubgraphId, SubgraphGradInfo>& calleeInfo,
		                        const std::map<NodeOutputKey, std::size_t>& saved, Subgraph& bwdSg,
		                        std::map<NodeOutputKey, std::vector<NodeOutput>>& gradContribs,
		                        std::map<NodeOutputKey, NodeId>& loadMap,
		                        std::map<std::size_t, std::vector<NodeOutput>>& varGradContribs)
		{
			for (auto i = fwdSg.NodeCount(); i-- > 0;)
			{
				const NodeId fwdNodeId = i;
				const auto& entry = fwdSg.GetNodeEntry(fwdNodeId);
				auto dy = ResolveGrad(fwdSg, bwdSg, fwdNodeId, 0, gradContribs);

				std::visit(
				    [&](const auto& node) {
					    using T = std::decay_t<decltype(node)>;
					    if constexpr (std::same_as<T, ParamRefNode> || std::same_as<T, ConstantNode> ||
					                  std::same_as<T, VariableRefNode>)
					    { /* 叶子节点 */
					    }
					    else if constexpr (std::same_as<T, UnaryOpNode>)
					    {
						    EmitUnaryGrad(fwdSg, bwdSg, fwdNodeId, node, dy, entry, saved, loadMap, gradContribs);
					    }
					    else if constexpr (std::same_as<T, BinaryOpNode>)
					    {
						    EmitBinaryGrad(fwdSg, bwdSg, fwdNodeId, node, dy, entry, saved, loadMap, gradContribs);
					    }
					    else if constexpr (std::same_as<T, CastNode>)
					    {
						    EmitCastGrad(fwdSg, bwdSg, node, dy, gradContribs);
					    }
					    else if constexpr (std::same_as<T, CallNode>)
					    {
						    EmitCallGrad(fwdSg, graph, bwdSg, node, dy, calleeInfo, saved, loadMap, gradContribs,
						                 varGradContribs);
					    }
					    else if constexpr (std::same_as<T, ReduceOpNode>)
					    {
						    EmitReduceGrad(fwdSg, bwdSg, fwdNodeId, node, dy, entry, saved, loadMap, gradContribs);
					    }
					    else if constexpr (std::same_as<T, ReshapeNode>)
					    {
						    EmitReshapeGrad(fwdSg, bwdSg, node, dy, gradContribs);
					    }
					    else if constexpr (std::same_as<T, CondNode>)
					    {
						    EmitCondGrad(fwdSg, graph, bwdSg, node, dy, calleeInfo, saved, loadMap, gradContribs,
						                 varGradContribs);
					    }
					    else
					    {
						    throw std::runtime_error("AutogradPass: unsupported node type for differentiation");
					    }
				    },
				    entry.node);
			}
		}

		// 各节点类型的梯度生成

		void EmitUnaryGrad(const Subgraph& fwdSg, Subgraph& bwdSg, NodeId fwdNodeId, const UnaryOpNode& node,
		                   NodeOutput dy, const NodeEntry& entry, const std::map<NodeOutputKey, std::size_t>& saved,
		                   std::map<NodeOutputKey, NodeId>& loadMap,
		                   std::map<NodeOutputKey, std::vector<NodeOutput>>& gc)
		{
			const auto& inInfo = fwdSg.GetOutputInfo(node.input);
			auto& dst = gc[{ node.input.node, node.input.port }];

			switch (node.op)
			{
			case UnaryOp::Negate: {
				auto id =
				    bwdSg.AddNode(UnaryOpNode{ UnaryOp::Negate, dy }, { OutputInfo{ inInfo.dtype, inInfo.shape } });
				dst.push_back({ id, 0 });
				break;
			}
			case UnaryOp::Transpose: {
				auto id =
				    bwdSg.AddNode(UnaryOpNode{ UnaryOp::Transpose, dy }, { OutputInfo{ inInfo.dtype, inInfo.shape } });
				dst.push_back({ id, 0 });
				break;
			}
			case UnaryOp::Abs: {
				// dx = dy * sign(x) = dy * (Cast(x>0,dtype) - Cast(x<0,dtype))
				auto xVal = GetForwardValue(fwdSg, bwdSg, node.input.node, node.input.port, saved, loadMap);
				auto zero = MakeZeroConstant(bwdSg, inInfo.dtype, inInfo.shape);
				auto gtId = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Greater, { xVal, 0 }, { zero, 0 } },
				                          { OutputInfo{ DataType::Bool, inInfo.shape } });
				auto ltId = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Less, { xVal, 0 }, { zero, 0 } },
				                          { OutputInfo{ DataType::Bool, inInfo.shape } });
				auto gtF =
				    bwdSg.AddNode(CastNode{ { gtId, 0 }, inInfo.dtype }, { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto ltF =
				    bwdSg.AddNode(CastNode{ { ltId, 0 }, inInfo.dtype }, { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto sign = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Subtract, { gtF, 0 }, { ltF, 0 } },
				                          { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto id = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, dy, { sign, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				dst.push_back({ id, 0 });
				break;
			}
			case UnaryOp::Sqrt: {
				// dx = dy * 0.5 / output
				auto outVal = GetForwardValue(fwdSg, bwdSg, fwdNodeId, 0, saved, loadMap);
				auto half = MakeScalarConstant(bwdSg, inInfo.dtype, inInfo.shape, 0.5);
				auto halfOverOut = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Divide, { half, 0 }, { outVal, 0 } },
				                                 { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto id = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, dy, { halfOverOut, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				dst.push_back({ id, 0 });
				break;
			}
			case UnaryOp::Exp: {
				// dx = dy * exp(x) = dy * output
				auto outVal = GetForwardValue(fwdSg, bwdSg, fwdNodeId, 0, saved, loadMap);
				auto id = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, dy, { outVal, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				dst.push_back({ id, 0 });
				break;
			}
			case UnaryOp::Log: {
				// dx = dy / x
				auto xVal = GetForwardValue(fwdSg, bwdSg, node.input.node, node.input.port, saved, loadMap);
				auto id = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Divide, dy, { xVal, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				dst.push_back({ id, 0 });
				break;
			}
			case UnaryOp::Sin: {
				// dx = dy * cos(x)
				auto xVal = GetForwardValue(fwdSg, bwdSg, node.input.node, node.input.port, saved, loadMap);
				auto cosX = bwdSg.AddNode(UnaryOpNode{ UnaryOp::Cos, { xVal, 0 } },
				                          { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto id = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, dy, { cosX, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				dst.push_back({ id, 0 });
				break;
			}
			case UnaryOp::Cos: {
				// dx = -dy * sin(x)
				auto xVal = GetForwardValue(fwdSg, bwdSg, node.input.node, node.input.port, saved, loadMap);
				auto sinX = bwdSg.AddNode(UnaryOpNode{ UnaryOp::Sin, { xVal, 0 } },
				                          { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto negDy =
				    bwdSg.AddNode(UnaryOpNode{ UnaryOp::Negate, dy }, { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto id = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { negDy, 0 }, { sinX, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				dst.push_back({ id, 0 });
				break;
			}
			case UnaryOp::Tan: {
				// dx = dy / cos²(x)
				auto xVal = GetForwardValue(fwdSg, bwdSg, node.input.node, node.input.port, saved, loadMap);
				auto cosX = bwdSg.AddNode(UnaryOpNode{ UnaryOp::Cos, { xVal, 0 } },
				                          { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto cos2 = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { cosX, 0 }, { cosX, 0 } },
				                          { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto id = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Divide, dy, { cos2, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				dst.push_back({ id, 0 });
				break;
			}
			case UnaryOp::Arcsin: {
				// dx = dy / sqrt(1 - x²)
				auto xVal = GetForwardValue(fwdSg, bwdSg, node.input.node, node.input.port, saved, loadMap);
				auto one = MakeScalarConstant(bwdSg, inInfo.dtype, inInfo.shape, 1.0);
				auto x2 = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { xVal, 0 }, { xVal, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto diff = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Subtract, { one, 0 }, { x2, 0 } },
				                          { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto sq = bwdSg.AddNode(UnaryOpNode{ UnaryOp::Sqrt, { diff, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto id = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Divide, dy, { sq, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				dst.push_back({ id, 0 });
				break;
			}
			case UnaryOp::Arccos: {
				// dx = -dy / sqrt(1 - x²)
				auto xVal = GetForwardValue(fwdSg, bwdSg, node.input.node, node.input.port, saved, loadMap);
				auto one = MakeScalarConstant(bwdSg, inInfo.dtype, inInfo.shape, 1.0);
				auto x2 = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { xVal, 0 }, { xVal, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto diff = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Subtract, { one, 0 }, { x2, 0 } },
				                          { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto sq = bwdSg.AddNode(UnaryOpNode{ UnaryOp::Sqrt, { diff, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto negDy =
				    bwdSg.AddNode(UnaryOpNode{ UnaryOp::Negate, dy }, { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto id = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Divide, { negDy, 0 }, { sq, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				dst.push_back({ id, 0 });
				break;
			}
			case UnaryOp::Arctan: {
				// dx = dy / (1 + x²)
				auto xVal = GetForwardValue(fwdSg, bwdSg, node.input.node, node.input.port, saved, loadMap);
				auto one = MakeScalarConstant(bwdSg, inInfo.dtype, inInfo.shape, 1.0);
				auto x2 = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { xVal, 0 }, { xVal, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto sum = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Add, { one, 0 }, { x2, 0 } },
				                         { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto id = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Divide, dy, { sum, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				dst.push_back({ id, 0 });
				break;
			}
			case UnaryOp::LogicalNegation:
				// Bool 输出，不可微
				break;
			}
		}

		void EmitBinaryGrad(const Subgraph& fwdSg, Subgraph& bwdSg, NodeId fwdNodeId, const BinaryOpNode& node,
		                    NodeOutput dy, const NodeEntry& entry, const std::map<NodeOutputKey, std::size_t>& saved,
		                    std::map<NodeOutputKey, NodeId>& loadMap,
		                    std::map<NodeOutputKey, std::vector<NodeOutput>>& gc)
		{
			const auto& li = fwdSg.GetOutputInfo(node.lhs);
			const auto& ri = fwdSg.GetOutputInfo(node.rhs);
			auto& dstL = gc[{ node.lhs.node, node.lhs.port }];
			auto& dstR = gc[{ node.rhs.node, node.rhs.port }];

			switch (node.op)
			{
			case BinaryOp::Add:
				dstL.push_back(dy);
				dstR.push_back(dy);
				break;
			case BinaryOp::Subtract: {
				dstL.push_back(dy);
				auto neg = bwdSg.AddNode(UnaryOpNode{ UnaryOp::Negate, dy }, { OutputInfo{ ri.dtype, ri.shape } });
				dstR.push_back({ neg, 0 });
				break;
			}
			case BinaryOp::Multiply: {
				auto bV = GetForwardValue(fwdSg, bwdSg, node.rhs.node, node.rhs.port, saved, loadMap);
				auto aV = GetForwardValue(fwdSg, bwdSg, node.lhs.node, node.lhs.port, saved, loadMap);
				auto da = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, dy, { bV, 0 } },
				                        { OutputInfo{ li.dtype, li.shape } });
				auto db = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, dy, { aV, 0 } },
				                        { OutputInfo{ ri.dtype, ri.shape } });
				dstL.push_back({ da, 0 });
				dstR.push_back({ db, 0 });
				break;
			}
			case BinaryOp::Divide: {
				// da = dy / b, db = -dy * a / (b * b)
				auto bV = GetForwardValue(fwdSg, bwdSg, node.rhs.node, node.rhs.port, saved, loadMap);
				auto aV = GetForwardValue(fwdSg, bwdSg, node.lhs.node, node.lhs.port, saved, loadMap);
				auto da = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Divide, dy, { bV, 0 } },
				                        { OutputInfo{ li.dtype, li.shape } });
				auto negDy = bwdSg.AddNode(UnaryOpNode{ UnaryOp::Negate, dy }, { OutputInfo{ ri.dtype, ri.shape } });
				auto negDyA = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { negDy, 0 }, { aV, 0 } },
				                            { OutputInfo{ ri.dtype, ri.shape } });
				auto bSq = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { bV, 0 }, { bV, 0 } },
				                         { OutputInfo{ ri.dtype, ri.shape } });
				auto db = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Divide, { negDyA, 0 }, { bSq, 0 } },
				                        { OutputInfo{ ri.dtype, ri.shape } });
				dstL.push_back({ da, 0 });
				dstR.push_back({ db, 0 });
				break;
			}
			case BinaryOp::MatMul: {
				auto bV = GetForwardValue(fwdSg, bwdSg, node.rhs.node, node.rhs.port, saved, loadMap);
				auto aV = GetForwardValue(fwdSg, bwdSg, node.lhs.node, node.lhs.port, saved, loadMap);
				auto bT = bwdSg.AddNode(UnaryOpNode{ UnaryOp::Transpose, { bV, 0 } },
				                        { OutputInfo{ ri.dtype, { ri.shape[1], ri.shape[0] } } });
				auto aT = bwdSg.AddNode(UnaryOpNode{ UnaryOp::Transpose, { aV, 0 } },
				                        { OutputInfo{ li.dtype, { li.shape[1], li.shape[0] } } });
				auto da = bwdSg.AddNode(BinaryOpNode{ BinaryOp::MatMul, dy, { bT, 0 } },
				                        { OutputInfo{ li.dtype, li.shape } });
				auto db = bwdSg.AddNode(BinaryOpNode{ BinaryOp::MatMul, { aT, 0 }, dy },
				                        { OutputInfo{ ri.dtype, ri.shape } });
				dstL.push_back({ da, 0 });
				dstR.push_back({ db, 0 });
				break;
			}
			case BinaryOp::Pow: {
				// a^b
				// da = dy * b * a^(b-1) = dy * b * (a^b) / a
				// db = dy * a^b * log(a)
				auto aV = GetForwardValue(fwdSg, bwdSg, node.lhs.node, node.lhs.port, saved, loadMap);
				auto bV = GetForwardValue(fwdSg, bwdSg, node.rhs.node, node.rhs.port, saved, loadMap);
				auto outV = GetForwardValue(fwdSg, bwdSg, fwdNodeId, 0, saved, loadMap); // a^b

				const auto& outInfo = entry.outputInfos[0];

				// da = dy * b * a^b / a
				auto bMulOut = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { bV, 0 }, { outV, 0 } },
				                             { OutputInfo{ outInfo.dtype, outInfo.shape } });
				auto bMulOutDivA = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Divide, { bMulOut, 0 }, { aV, 0 } },
				                                 { OutputInfo{ outInfo.dtype, outInfo.shape } });
				auto da = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, dy, { bMulOutDivA, 0 } },
				                        { OutputInfo{ li.dtype, li.shape } });
				dstL.push_back({ da, 0 });

				// db = dy * a^b * log(a)
				auto logA = bwdSg.AddNode(UnaryOpNode{ UnaryOp::Log, { aV, 0 } }, { OutputInfo{ li.dtype, li.shape } });
				auto outMulLogA = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { outV, 0 }, { logA, 0 } },
				                                { OutputInfo{ outInfo.dtype, outInfo.shape } });
				auto db = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, dy, { outMulLogA, 0 } },
				                        { OutputInfo{ ri.dtype, ri.shape } });
				dstR.push_back({ db, 0 });
				break;
			}
			case BinaryOp::Max: {
				// d(max(a,b))/da = dy * (a >= b), d(max(a,b))/db = dy * (a < b)
				// 使用 Greater+Cast 实现 indicator
				auto aV = GetForwardValue(fwdSg, bwdSg, node.lhs.node, node.lhs.port, saved, loadMap);
				auto bV = GetForwardValue(fwdSg, bwdSg, node.rhs.node, node.rhs.port, saved, loadMap);

				// a >= b  等价于  !(a < b)
				auto ltId = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Less, { aV, 0 }, { bV, 0 } },
				                          { OutputInfo{ DataType::Bool, li.shape } });
				auto geId = bwdSg.AddNode(UnaryOpNode{ UnaryOp::LogicalNegation, { ltId, 0 } },
				                          { OutputInfo{ DataType::Bool, li.shape } });

				auto geMask = bwdSg.AddNode(CastNode{ { geId, 0 }, li.dtype }, { OutputInfo{ li.dtype, li.shape } });
				auto ltMask = bwdSg.AddNode(CastNode{ { ltId, 0 }, ri.dtype }, { OutputInfo{ ri.dtype, ri.shape } });

				auto da = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, dy, { geMask, 0 } },
				                        { OutputInfo{ li.dtype, li.shape } });
				auto db = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, dy, { ltMask, 0 } },
				                        { OutputInfo{ ri.dtype, ri.shape } });
				dstL.push_back({ da, 0 });
				dstR.push_back({ db, 0 });
				break;
			}
			case BinaryOp::Min: {
				// d(min(a,b))/da = dy * (a <= b), d(min(a,b))/db = dy * (a > b)
				auto aV = GetForwardValue(fwdSg, bwdSg, node.lhs.node, node.lhs.port, saved, loadMap);
				auto bV = GetForwardValue(fwdSg, bwdSg, node.rhs.node, node.rhs.port, saved, loadMap);

				// a <= b  等价于  !(a > b)
				auto gtId = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Greater, { aV, 0 }, { bV, 0 } },
				                          { OutputInfo{ DataType::Bool, li.shape } });
				auto leId = bwdSg.AddNode(UnaryOpNode{ UnaryOp::LogicalNegation, { gtId, 0 } },
				                          { OutputInfo{ DataType::Bool, li.shape } });

				auto leMask = bwdSg.AddNode(CastNode{ { leId, 0 }, li.dtype }, { OutputInfo{ li.dtype, li.shape } });
				auto gtMask = bwdSg.AddNode(CastNode{ { gtId, 0 }, ri.dtype }, { OutputInfo{ ri.dtype, ri.shape } });

				auto da = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, dy, { leMask, 0 } },
				                        { OutputInfo{ li.dtype, li.shape } });
				auto db = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, dy, { gtMask, 0 } },
				                        { OutputInfo{ ri.dtype, ri.shape } });
				dstL.push_back({ da, 0 });
				dstR.push_back({ db, 0 });
				break;
			}
			case BinaryOp::Less:
			case BinaryOp::Greater:
			case BinaryOp::Equal:
				// Bool 输出，不可微
				break;
			}
		}

		static void EmitCastGrad(const Subgraph& fwdSg, Subgraph& bwdSg, const CastNode& node, NodeOutput dy,
		                         std::map<NodeOutputKey, std::vector<NodeOutput>>& gc)
		{
			const auto& inInfo = fwdSg.GetOutputInfo(node.input);
			if (inInfo.dtype == DataType::Bool || inInfo.dtype == DataType::Int32 || inInfo.dtype == DataType::Int64)
			{
				return;
			}
			auto id = bwdSg.AddNode(CastNode{ dy, inInfo.dtype }, { OutputInfo{ inInfo.dtype, inInfo.shape } });
			gc[{ node.input.node, node.input.port }].push_back({ id, 0 });
		}

		void EmitCallGrad(const Subgraph& fwdSg, Graph& graph, Subgraph& bwdSg, const CallNode& node, NodeOutput dy,
		                  const std::map<SubgraphId, SubgraphGradInfo>& calleeInfo,
		                  const std::map<NodeOutputKey, std::size_t>& saved, std::map<NodeOutputKey, NodeId>& loadMap,
		                  std::map<NodeOutputKey, std::vector<NodeOutput>>& gc,
		                  std::map<std::size_t, std::vector<NodeOutput>>& varGradContribs)
		{
			auto it = calleeInfo.find(node.callee);
			if (it == calleeInfo.end())
			{
				throw std::runtime_error("AutogradPass: callee not processed");
			}

			const auto& info = it->second;

			// callee backward 参数: [callee_forward_inputs..., grad_callee_outputs...]
			// 收集 callee_forward_inputs（即 CallNode 的 args 的前向值）
			std::vector<NodeOutput> bwdArgs;
			for (const auto& arg : node.args)
			{
				auto fwdVal = GetForwardValue(fwdSg, bwdSg, arg.node, arg.port, saved, loadMap);
				bwdArgs.push_back({ fwdVal, 0 });
			}
			bwdArgs.push_back(dy); // grad_output

			// 调用 callee 的 backward
			// callee backward 输出: [grad_callee_inputs..., grad_callee_variables...]
			std::vector<OutputInfo> callOutputInfos;
			const auto& calleeFwdSg = graph.GetSubgraph(node.callee);
			for (std::size_t j = 0; j < calleeFwdSg.Params().size(); ++j)
			{
				const auto& p = calleeFwdSg.Params()[j];
				callOutputInfos.push_back({ p.dtype, p.shape });
			}
			// callee 的 variable gradients
			for (const auto& vg : info.variableGrads)
			{
				callOutputInfos.push_back(vg.outputInfo);
			}
			auto callBwdId = bwdSg.AddNode(CallNode{ info.backwardId, std::move(bwdArgs) }, std::move(callOutputInfos));

			// 分发 grad_callee_inputs 回 CallNode 的 args
			for (std::size_t j = 0; j < node.args.size(); ++j)
			{
				const auto& arg = node.args[j];
				gc[{ arg.node, arg.port }].push_back({ callBwdId, j });
			}

			// 分发 callee variable grads
			for (std::size_t j = 0; j < info.variableGrads.size(); ++j)
			{
				const auto outputPort = info.numInputGrads + j;
				varGradContribs[info.variableGrads[j].variableIndex].push_back({ callBwdId, outputPort });
			}
		}

		// 为 CondNode 分支 backward 构建 wrapper 子图，使其输出与 unionVarGrads 布局一致
		// 缺失的 variable grad 用零填充
		static SubgraphId BuildCondBranchBwdWrapper(Graph& graph, const SubgraphGradInfo& branchInfo,
		                                            const std::vector<VarGradEntry>& unionVarGrads)
		{
			const auto& branchBwdSg = graph.GetSubgraph(branchInfo.backwardId);

			Subgraph wrapper;

			// 声明与原 branch backward 相同的参数
			std::vector<NodeOutput> innerArgs;
			for (std::size_t i = 0; i < branchBwdSg.Params().size(); ++i)
			{
				const auto& p = branchBwdSg.Params()[i];
				auto paramId = wrapper.AddParam(p.dtype, p.shape);
				innerArgs.push_back({ paramId, 0 });
			}

			// 调用原 branch backward
			std::vector<OutputInfo> innerOutputInfos;
			for (std::size_t i = 0; i < branchInfo.numInputGrads; ++i)
			{
				const auto& p = branchBwdSg.Params()[i];
				innerOutputInfos.push_back({ p.dtype, p.shape });
			}
			for (const auto& vg : branchInfo.variableGrads)
			{
				innerOutputInfos.push_back(vg.outputInfo);
			}
			auto callId =
			    wrapper.AddNode(CallNode{ branchInfo.backwardId, std::move(innerArgs) }, std::move(innerOutputInfos));

			// 构建 branch variableGrads 的 variableIndex→port 映射
			std::map<std::size_t, std::size_t> branchVarPorts;
			for (std::size_t j = 0; j < branchInfo.variableGrads.size(); ++j)
			{
				branchVarPorts[branchInfo.variableGrads[j].variableIndex] = branchInfo.numInputGrads + j;
			}

			// 构建结果：input grads + union variable grads
			std::vector<NodeOutput> results;
			for (std::size_t i = 0; i < branchInfo.numInputGrads; ++i)
			{
				results.push_back({ callId, i });
			}
			for (const auto& vg : unionVarGrads)
			{
				auto portIt = branchVarPorts.find(vg.variableIndex);
				if (portIt != branchVarPorts.end())
				{
					results.push_back({ callId, portIt->second });
				}
				else
				{
					// 此分支没有此 variable 的梯度，用零填充
					auto zeroId = MakeZeroConstant(wrapper, vg.outputInfo.dtype, vg.outputInfo.shape);
					results.push_back({ zeroId, 0 });
				}
			}
			wrapper.SetResults(std::move(results));
			return graph.AddSubgraph(std::move(wrapper));
		}

		void EmitCondGrad(const Subgraph& fwdSg, Graph& graph, Subgraph& bwdSg, const CondNode& node, NodeOutput dy,
		                  const std::map<SubgraphId, SubgraphGradInfo>& calleeInfo,
		                  const std::map<NodeOutputKey, std::size_t>& saved, std::map<NodeOutputKey, NodeId>& loadMap,
		                  std::map<NodeOutputKey, std::vector<NodeOutput>>& gc,
		                  std::map<std::size_t, std::vector<NodeOutput>>& varGradContribs)
		{
			auto thenIt = calleeInfo.find(node.thenBranch);
			auto elseIt = calleeInfo.find(node.elseBranch);
			if (thenIt == calleeInfo.end() || elseIt == calleeInfo.end())
			{
				throw std::runtime_error("AutogradPass: CondNode branch not processed");
			}

			const auto& thenInfo = thenIt->second;
			const auto& elseInfo = elseIt->second;

			// Reload condition（backward 需要重新判断走哪个分支）
			auto condVal = GetForwardValue(fwdSg, bwdSg, node.condition.node, node.condition.port, saved, loadMap);

			// 收集前向 args 值 + dy，作为分支 backward 的参数
			std::vector<NodeOutput> bwdArgs;
			for (const auto& arg : node.args)
			{
				auto fwdVal = GetForwardValue(fwdSg, bwdSg, arg.node, arg.port, saved, loadMap);
				bwdArgs.push_back({ fwdVal, 0 });
			}
			bwdArgs.push_back(dy); // grad_output

			// 计算两分支 variable grads 的并集（按 variableIndex 排序）
			std::vector<VarGradEntry> unionVarGrads;
			{
				std::map<std::size_t, OutputInfo> varMap;
				for (const auto& vg : thenInfo.variableGrads)
				{
					varMap[vg.variableIndex] = vg.outputInfo;
				}
				for (const auto& vg : elseInfo.variableGrads)
				{
					varMap[vg.variableIndex] = vg.outputInfo;
				}
				for (const auto& [idx, info] : varMap)
				{
					unionVarGrads.push_back({ idx, info });
				}
			}

			// 确定使用的 backward 子图 ID（可能需要 wrapper）
			auto thenBwdId = thenInfo.backwardId;
			auto elseBwdId = elseInfo.backwardId;

			if (!unionVarGrads.empty())
			{
				// 检查两分支是否布局完全相同
				bool sameLayout = (thenInfo.variableGrads.size() == elseInfo.variableGrads.size()) &&
				                  (thenInfo.variableGrads.size() == unionVarGrads.size());
				if (sameLayout)
				{
					for (std::size_t j = 0; j < unionVarGrads.size(); ++j)
					{
						if (thenInfo.variableGrads[j].variableIndex != unionVarGrads[j].variableIndex ||
						    elseInfo.variableGrads[j].variableIndex != unionVarGrads[j].variableIndex)
						{
							sameLayout = false;
							break;
						}
					}
				}

				if (!sameLayout)
				{
					thenBwdId = BuildCondBranchBwdWrapper(graph, thenInfo, unionVarGrads);
					elseBwdId = BuildCondBranchBwdWrapper(graph, elseInfo, unionVarGrads);
				}
			}

			// 构建 backward CondNode 的输出信息
			std::vector<OutputInfo> condOutputInfos;
			const auto& origFwdSg = graph.GetSubgraph(node.thenBranch);
			for (std::size_t j = 0; j < origFwdSg.Params().size(); ++j)
			{
				const auto& p = origFwdSg.Params()[j];
				condOutputInfos.push_back({ p.dtype, p.shape });
			}
			for (const auto& vg : unionVarGrads)
			{
				condOutputInfos.push_back(vg.outputInfo);
			}

			const auto numInputGrads = origFwdSg.Params().size();

			auto condBwdId = bwdSg.AddNode(CondNode{ { condVal, 0 }, thenBwdId, elseBwdId, std::move(bwdArgs) },
			                               std::move(condOutputInfos));

			// 分发梯度回原 args（condition 是 Bool 标量，不传播梯度）
			for (std::size_t j = 0; j < node.args.size(); ++j)
			{
				const auto& arg = node.args[j];
				gc[{ arg.node, arg.port }].push_back({ condBwdId, j });
			}

			// 分发 variable grads
			for (std::size_t j = 0; j < unionVarGrads.size(); ++j)
			{
				const auto outputPort = numInputGrads + j;
				varGradContribs[unionVarGrads[j].variableIndex].push_back({ condBwdId, outputPort });
			}
		}

		static NodeId MakeZeroConstant(Subgraph& sg, DataType dtype, const std::vector<std::size_t>& shape)
		{
			auto t = Tensor<CPU>(shape, dtype); // 默认零初始化
			return sg.AddNode(ConstantNode{ t.CopyToDevice(PolymorphicDevice{ CPU{} }) },
			                  { OutputInfo{ dtype, shape } });
		}

		static NodeId MakeScalarConstant(Subgraph& sg, DataType dtype, const std::vector<std::size_t>& shape,
		                                 double value)
		{
			const auto numElements = ShapeView{ shape }.NumElements();
			std::vector<double> data(numElements, value);
			auto t = Tensor<CPU>(data, shape, dtype);
			return sg.AddNode(ConstantNode{ t.CopyToDevice(PolymorphicDevice{ CPU{} }) },
			                  { OutputInfo{ dtype, shape } });
		}

		static NodeId MakeOnesConstant(Subgraph& sg, DataType dtype, const std::vector<std::size_t>& shape)
		{
			return MakeScalarConstant(sg, dtype, shape, 1.0);
		}

		void EmitReduceGrad(const Subgraph& fwdSg, Subgraph& bwdSg, NodeId fwdNodeId, const ReduceOpNode& node,
		                    NodeOutput dy, const NodeEntry& entry, const std::map<NodeOutputKey, std::size_t>& saved,
		                    std::map<NodeOutputKey, NodeId>& loadMap,
		                    std::map<NodeOutputKey, std::vector<NodeOutput>>& gc)
		{
			const auto& inInfo = fwdSg.GetOutputInfo(node.input);
			auto& dst = gc[{ node.input.node, node.input.port }];

			// dy shape = reduced shape (axis removed). We need to expand back to input shape.
			// Step 1: expandedShape = inputShape with axis dimension set to 1
			auto expandedShape = inInfo.shape;
			expandedShape[node.axis] = 1;

			// Step 2: Reshape dy to expandedShape
			auto dyReshaped =
			    bwdSg.AddNode(ReshapeNode{ dy, expandedShape }, { OutputInfo{ inInfo.dtype, expandedShape } });

			switch (node.op)
			{
			case ReduceOp::Sum: {
				// grad = ones(inputShape) * Reshape(dy, expandedShape) — broadcast handles expansion
				auto ones = MakeOnesConstant(bwdSg, inInfo.dtype, inInfo.shape);
				auto id = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { dyReshaped, 0 }, { ones, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				dst.push_back({ id, 0 });
				break;
			}
			case ReduceOp::Mean: {
				// grad = ones(inputShape) * Reshape(dy, expandedShape) / axisDim
				auto ones = MakeOnesConstant(bwdSg, inInfo.dtype, inInfo.shape);
				auto expanded = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { dyReshaped, 0 }, { ones, 0 } },
				                              { OutputInfo{ inInfo.dtype, inInfo.shape } });
				auto axisDim =
				    MakeScalarConstant(bwdSg, inInfo.dtype, inInfo.shape, static_cast<double>(inInfo.shape[node.axis]));
				auto id = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Divide, { expanded, 0 }, { axisDim, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				dst.push_back({ id, 0 });
				break;
			}
			case ReduceOp::Max: {
				// grad = dy * (input == Reshape(output, expandedShape))
				auto xVal = GetForwardValue(fwdSg, bwdSg, node.input.node, node.input.port, saved, loadMap);
				auto outVal = GetForwardValue(fwdSg, bwdSg, fwdNodeId, 0, saved, loadMap);

				const auto& outInfo = entry.outputInfos[0];
				auto outReshaped = bwdSg.AddNode(ReshapeNode{ { outVal, 0 }, expandedShape },
				                                 { OutputInfo{ outInfo.dtype, expandedShape } });

				// mask = (input == Reshape(output, expandedShape)) — broadcast handles expansion
				auto mask = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Equal, { xVal, 0 }, { outReshaped, 0 } },
				                          { OutputInfo{ DataType::Bool, inInfo.shape } });
				auto maskFloat =
				    bwdSg.AddNode(CastNode{ { mask, 0 }, inInfo.dtype }, { OutputInfo{ inInfo.dtype, inInfo.shape } });

				// expand dy the same way
				auto ones = MakeOnesConstant(bwdSg, inInfo.dtype, inInfo.shape);
				auto dyExpanded = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { dyReshaped, 0 }, { ones, 0 } },
				                                { OutputInfo{ inInfo.dtype, inInfo.shape } });

				auto id = bwdSg.AddNode(BinaryOpNode{ BinaryOp::Multiply, { dyExpanded, 0 }, { maskFloat, 0 } },
				                        { OutputInfo{ inInfo.dtype, inInfo.shape } });
				dst.push_back({ id, 0 });
				break;
			}
			}
		}

		static void EmitReshapeGrad(const Subgraph& fwdSg, Subgraph& bwdSg, const ReshapeNode& node, NodeOutput dy,
		                            std::map<NodeOutputKey, std::vector<NodeOutput>>& gc)
		{
			const auto& inInfo = fwdSg.GetOutputInfo(node.input);
			// Reshape gradient: just reshape back to the input shape
			auto id = bwdSg.AddNode(ReshapeNode{ dy, inInfo.shape }, { OutputInfo{ inInfo.dtype, inInfo.shape } });
			gc[{ node.input.node, node.input.port }].push_back({ id, 0 });
		}
	};
} // namespace LiteNN

#endif
