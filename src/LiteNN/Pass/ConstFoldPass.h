#ifndef LITENN_PASS_CONSTFOLD_H
#define LITENN_PASS_CONSTFOLD_H

#include <LiteNN/Graph.h>
#include <LiteNN/Pass.h>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

namespace LiteNN
{
	// 常量折叠 Pass：在编译期评估所有输入均为常量的节点，
	// 并消除恒等操作（x+0, x*1 等），减少运行时计算量
	// 应在 InlinePass 之后、FusionPass 之前运行
	class ConstFoldPass : public Pass
	{
	public:
		void Run(Graph& graph) override
		{
			const auto originalCount = graph.SubgraphCount();
			for (std::size_t sgId = 0; sgId < originalCount; ++sgId)
			{
				ProcessSubgraph(graph, sgId);
			}
		}

	private:
		using NodeOutputKey = std::pair<NodeId, std::size_t>;

		// ---- CPU 求值工具 ----

		static Tensor<CPU> EvalUnaryOp(UnaryOp op, const Tensor<CPU>& input, const OutputInfo& outInfo)
		{
			CPU device;
			Tensor<CPU> result(Uninitialized, outInfo.shape, outInfo.dtype, device);
			DeviceTraits<CPU>::DoUnaryOp(device, op, result.RawData(), input.DType(), input.Shape(), input.RawData());
			return result;
		}

		static Tensor<CPU> EvalBinaryOp(BinaryOp op, const Tensor<CPU>& lhs, const Tensor<CPU>& rhs,
		                                const OutputInfo& outInfo)
		{
			CPU device;
			Tensor<CPU> result(Uninitialized, outInfo.shape, outInfo.dtype, device);
			DeviceTraits<CPU>::DoBinaryOp(device, op, result.RawData(), lhs.DType(), lhs.Shape(), lhs.RawData(),
			                              rhs.DType(), rhs.Shape(), rhs.RawData());
			return result;
		}

		static Tensor<CPU> EvalCast(const Tensor<CPU>& input, DataType targetType)
		{
			CPU device;
			Tensor<CPU> result(Uninitialized, input.Shape(), targetType, device);
			DeviceTraits<CPU>::ConvertTo(device, input.DType(), input.RawData(), input.NumElements(), targetType,
			                            result.RawData());
			return result;
		}

		static Tensor<CPU> EvalReduceOp(ReduceOp op, const Tensor<CPU>& input, std::size_t axis,
		                                const OutputInfo& outInfo)
		{
			CPU device;
			Tensor<CPU> result(Uninitialized, outInfo.shape, outInfo.dtype, device);
			DeviceTraits<CPU>::DoReduceOp(device, op, result.RawData(), input.DType(), input.Shape(), input.RawData(),
			                              axis);
			return result;
		}

		static Tensor<CPU> EvalReshape(const Tensor<CPU>& input, const OutputInfo& outInfo)
		{
			CPU device;
			Tensor<CPU> result(Uninitialized, outInfo.shape, outInfo.dtype, device);
			DeviceTraits<CPU>::ConvertTo(device, input.DType(), input.RawData(), input.NumElements(), outInfo.dtype,
			                            result.RawData());
			return result;
		}

		static Tensor<CPU> EvalConcat(const std::vector<std::optional<Tensor<CPU>>>& constValues,
		                              const ConcatNode& node, const OutputInfo& outInfo)
		{
			CPU device;
			Tensor<CPU> result(Uninitialized, outInfo.shape, outInfo.dtype, device);
			std::vector<const void*> srcPtrs;
			std::vector<ShapeView> srcShapes;
			for (const auto& input : node.inputs)
			{
				const auto& t = *constValues[input.node];
				srcPtrs.push_back(t.RawData());
				srcShapes.push_back(t.Shape());
			}
			DeviceTraits<CPU>::DoConcatOp(device, result.RawData(), outInfo.dtype, srcPtrs.data(), srcShapes.data(),
			                              srcPtrs.size(), node.axis);
			return result;
		}

		static Tensor<CPU> EvalSlice(const Tensor<CPU>& input, const SliceNode& node, const OutputInfo& outInfo)
		{
			CPU device;
			Tensor<CPU> result(Uninitialized, outInfo.shape, outInfo.dtype, device);
			DeviceTraits<CPU>::DoSliceOp(device, result.RawData(), outInfo.dtype, input.Shape(), input.RawData(),
			                             node.axis, node.start, node.length);
			return result;
		}

		// ---- 常量检测工具 ----

		static bool IsZeroTensor(const Tensor<CPU>& t)
		{
			return EnumDispatch(t.DType(), [&]<DataType TypeValue> -> bool {
				using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
				const auto* data = static_cast<const T*>(t.RawData());
				for (auto i = 0uz; i < t.NumElements(); ++i)
				{
					if (data[i] != T(0))
					{
						return false;
					}
				}
				return true;
			});
		}

		static bool IsOneTensor(const Tensor<CPU>& t)
		{
			return EnumDispatch(t.DType(), [&]<DataType TypeValue> -> bool {
				using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
				const auto* data = static_cast<const T*>(t.RawData());
				for (auto i = 0uz; i < t.NumElements(); ++i)
				{
					if (data[i] != T(1))
					{
						return false;
					}
				}
				return true;
			});
		}

		static Tensor<CPU> MakeZeroTensor(DataType dtype, ShapeView shape)
		{
			return Tensor<CPU>(shape, dtype, CPU{});
		}

		// Shape 兼容检查：非常量操作数的 shape 必须与输出 shape 相同
		static bool ShapeCompatibleForElimination(ShapeView nonConstShape, ShapeView outputShape)
		{
			if (nonConstShape.NumDim() != outputShape.NumDim())
			{
				return false;
			}
			for (auto i = 0uz; i < nonConstShape.NumDim(); ++i)
			{
				if (nonConstShape[i] != outputShape[i])
				{
					return false;
				}
			}
			return true;
		}

		// ---- 获取节点输入的 const 值 ----

		static const Tensor<CPU>& GetConstValue(const std::vector<std::optional<Tensor<CPU>>>& constValues, NodeOutput output)
		{
			return *constValues[output.node];
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
					    throw std::runtime_error("ConstFoldPass: unsupported node type in remap");
				    }
			    },
			    node);
		}

		// ---- 活性分析：从结果反向标记可达节点 ----

		static void MarkReachable(const Subgraph& sg, NodeId nodeId, std::vector<bool>& alive,
		                          const std::vector<bool>& isConst)
		{
			if (alive[nodeId])
			{
				return;
			}
			alive[nodeId] = true;

			// 已折叠为常量的非 ConstantNode 不需要追踪其原始输入
			const auto& entry = sg.GetNodeEntry(nodeId);
			if (isConst[nodeId] && !std::holds_alternative<ConstantNode>(entry.node))
			{
				return;
			}

			std::visit(
			    [&](const auto& node) {
				    using T = std::decay_t<decltype(node)>;
				    auto markInput = [&](NodeOutput input) {
					    MarkReachable(sg, input.node, alive, isConst);
				    };
				    if constexpr (std::same_as<T, UnaryOpNode>)
				    {
					    markInput(node.input);
				    }
				    else if constexpr (std::same_as<T, BinaryOpNode>)
				    {
					    markInput(node.lhs);
					    markInput(node.rhs);
				    }
				    else if constexpr (std::same_as<T, CastNode>)
				    {
					    markInput(node.input);
				    }
				    else if constexpr (std::same_as<T, CallNode>)
				    {
					    for (const auto& a : node.args)
					    {
						    markInput(a);
					    }
				    }
				    else if constexpr (std::same_as<T, CondNode>)
				    {
					    markInput(node.condition);
					    for (const auto& a : node.args)
					    {
						    markInput(a);
					    }
				    }
				    else if constexpr (std::same_as<T, ReduceOpNode>)
				    {
					    markInput(node.input);
				    }
				    else if constexpr (std::same_as<T, ReshapeNode>)
				    {
					    markInput(node.input);
				    }
				    else if constexpr (std::same_as<T, ConcatNode>)
				    {
					    for (const auto& input : node.inputs)
					    {
						    markInput(input);
					    }
				    }
				    else if constexpr (std::same_as<T, SliceNode>)
				    {
					    markInput(node.input);
				    }
				    else if constexpr (std::same_as<T, SaveActivationNode>)
				    {
					    markInput(node.input);
				    }
				    else if constexpr (std::same_as<T, FusedOpNode>)
				    {
					    for (const auto& a : node.args)
					    {
						    markInput(a);
					    }
				    }
				    else if constexpr (std::same_as<T, WhileNode>)
				    {
					    for (const auto& a : node.initArgs)
					    {
						    markInput(a);
					    }
				    }
				    else if constexpr (std::same_as<T, TapeSaveActivationNode>)
				    {
					    markInput(node.input);
				    }
				    // ParamRefNode, ConstantNode, VariableRefNode, LoadActivationNode, TapeLoadActivationNode: 无输入
			    },
			    entry.node);
		}

		// ---- 子图处理 (三阶段) ----

		void ProcessSubgraph(Graph& graph, SubgraphId sgId)
		{
			const auto& sg = graph.GetSubgraph(sgId);
			const auto nodeCount = sg.NodeCount();

			// === Phase 1: 常量传播 + 求值 ===

			std::vector<bool> isConst(nodeCount, false);
			std::vector<std::optional<Tensor<CPU>>> constValues(nodeCount);

			for (NodeId nodeId = 0; nodeId < nodeCount; ++nodeId)
			{
				const auto& entry = sg.GetNodeEntry(nodeId);

				std::visit(
				    [&](const auto& node) {
					    using T = std::decay_t<decltype(node)>;

					    if constexpr (std::same_as<T, ConstantNode>)
					    {
						    isConst[nodeId] = true;
						    constValues[nodeId] = node.value.CopyToDevice(CPU{});
					    }
					    else if constexpr (std::same_as<T, ParamRefNode> || std::same_as<T, VariableRefNode> ||
					                       std::same_as<T, LoadActivationNode> || std::same_as<T, SaveActivationNode>)
					    {
						    // 非常量
					    }
					    else if constexpr (std::same_as<T, UnaryOpNode>)
					    {
						    if (isConst[node.input.node])
						    {
							    isConst[nodeId] = true;
							    const auto& input = GetConstValue(constValues, node.input);
							    constValues[nodeId] = EvalUnaryOp(node.op, input, entry.outputInfos[0]);
						    }
					    }
					    else if constexpr (std::same_as<T, BinaryOpNode>)
					    {
						    if (isConst[node.lhs.node] && isConst[node.rhs.node])
						    {
							    isConst[nodeId] = true;
							    const auto& lhs = GetConstValue(constValues, node.lhs);
							    const auto& rhs = GetConstValue(constValues, node.rhs);
							    constValues[nodeId] = EvalBinaryOp(node.op, lhs, rhs, entry.outputInfos[0]);
						    }
					    }
					    else if constexpr (std::same_as<T, CastNode>)
					    {
						    if (isConst[node.input.node])
						    {
							    isConst[nodeId] = true;
							    const auto& input = GetConstValue(constValues, node.input);
							    constValues[nodeId] = EvalCast(input, node.targetType);
						    }
					    }
					    else if constexpr (std::same_as<T, ReduceOpNode>)
					    {
						    if (isConst[node.input.node])
						    {
							    isConst[nodeId] = true;
							    const auto& input = GetConstValue(constValues, node.input);
							    constValues[nodeId] = EvalReduceOp(node.op, input, node.axis, entry.outputInfos[0]);
						    }
					    }
					    else if constexpr (std::same_as<T, ReshapeNode>)
					    {
						    if (isConst[node.input.node])
						    {
							    isConst[nodeId] = true;
							    const auto& input = GetConstValue(constValues, node.input);
							    constValues[nodeId] = EvalReshape(input, entry.outputInfos[0]);
						    }
					    }
					    else if constexpr (std::same_as<T, ConcatNode>)
					    {
						    bool allConst = true;
						    for (const auto& input : node.inputs)
						    {
							    if (!isConst[input.node])
							    {
								    allConst = false;
								    break;
							    }
						    }
						    if (allConst)
						    {
							    isConst[nodeId] = true;
							    constValues[nodeId] = EvalConcat(constValues, node, entry.outputInfos[0]);
						    }
					    }
					    else if constexpr (std::same_as<T, SliceNode>)
					    {
						    if (isConst[node.input.node])
						    {
							    isConst[nodeId] = true;
							    const auto& input = GetConstValue(constValues, node.input);
							    constValues[nodeId] = EvalSlice(input, node, entry.outputInfos[0]);
						    }
					    }
					    // CallNode, CondNode, FusedOpNode, WhileNode, TapeSaveActivationNode, TapeLoadActivationNode: 不折叠
				    },
				    entry.node);
			}

			// === Phase 2: 恒等消除 ===

			// identityMap: 被消除节点 → 替代 NodeOutput
			std::map<NodeId, NodeOutput> identityMap;

			for (NodeId nodeId = 0; nodeId < nodeCount; ++nodeId)
			{
				if (isConst[nodeId])
				{
					continue; // 已折叠为常量
				}

				const auto& entry = sg.GetNodeEntry(nodeId);
				const auto& outInfo = entry.outputInfos[0];

				if (auto* bin = std::get_if<BinaryOpNode>(&entry.node))
				{
					const bool lhsConst = isConst[bin->lhs.node];
					const bool rhsConst = isConst[bin->rhs.node];

					if (bin->op == BinaryOp::Add)
					{
						// x + 0 → x
						if (rhsConst && IsZeroTensor(GetConstValue(constValues, bin->rhs)))
						{
							const auto& lhsInfo = sg.GetOutputInfo(bin->lhs);
							if (ShapeCompatibleForElimination(lhsInfo.shape, outInfo.shape))
							{
								identityMap[nodeId] = bin->lhs;
							}
						}
						// 0 + x → x
						else if (lhsConst && IsZeroTensor(GetConstValue(constValues, bin->lhs)))
						{
							const auto& rhsInfo = sg.GetOutputInfo(bin->rhs);
							if (ShapeCompatibleForElimination(rhsInfo.shape, outInfo.shape))
							{
								identityMap[nodeId] = bin->rhs;
							}
						}
					}
					else if (bin->op == BinaryOp::Subtract)
					{
						// x - 0 → x
						if (rhsConst && IsZeroTensor(GetConstValue(constValues, bin->rhs)))
						{
							const auto& lhsInfo = sg.GetOutputInfo(bin->lhs);
							if (ShapeCompatibleForElimination(lhsInfo.shape, outInfo.shape))
							{
								identityMap[nodeId] = bin->lhs;
							}
						}
					}
					else if (bin->op == BinaryOp::Multiply)
					{
						// x * 1 → x
						if (rhsConst && IsOneTensor(GetConstValue(constValues, bin->rhs)))
						{
							const auto& lhsInfo = sg.GetOutputInfo(bin->lhs);
							if (ShapeCompatibleForElimination(lhsInfo.shape, outInfo.shape))
							{
								identityMap[nodeId] = bin->lhs;
							}
						}
						// 1 * x → x
						else if (lhsConst && IsOneTensor(GetConstValue(constValues, bin->lhs)))
						{
							const auto& rhsInfo = sg.GetOutputInfo(bin->rhs);
							if (ShapeCompatibleForElimination(rhsInfo.shape, outInfo.shape))
							{
								identityMap[nodeId] = bin->rhs;
							}
						}
						// x * 0 → 0
						else if (rhsConst && IsZeroTensor(GetConstValue(constValues, bin->rhs)))
						{
							const auto& lhsInfo = sg.GetOutputInfo(bin->lhs);
							if (ShapeCompatibleForElimination(lhsInfo.shape, outInfo.shape))
							{
								// 标记为常量
								isConst[nodeId] = true;
								constValues[nodeId] = MakeZeroTensor(outInfo.dtype, outInfo.shape);
							}
						}
						// 0 * x → 0
						else if (lhsConst && IsZeroTensor(GetConstValue(constValues, bin->lhs)))
						{
							const auto& rhsInfo = sg.GetOutputInfo(bin->rhs);
							if (ShapeCompatibleForElimination(rhsInfo.shape, outInfo.shape))
							{
								isConst[nodeId] = true;
								constValues[nodeId] = MakeZeroTensor(outInfo.dtype, outInfo.shape);
							}
						}
					}
				}
				else if (auto* unary = std::get_if<UnaryOpNode>(&entry.node))
				{
					// Negate(Negate(x)) → x
					if (unary->op == UnaryOp::Negate)
					{
						const auto& innerEntry = sg.GetNodeEntry(unary->input.node);
						if (auto* innerUnary = std::get_if<UnaryOpNode>(&innerEntry.node))
						{
							if (innerUnary->op == UnaryOp::Negate)
							{
								identityMap[nodeId] = innerUnary->input;
							}
						}
					}
				}
			}

			// === Phase 3: 活性分析 + 子图重建 ===

			// 先解析恒等链：identityMap 可能出现链式映射
			auto resolveIdentity = [&](NodeOutput out) -> NodeOutput {
				auto it = identityMap.find(out.node);
				while (it != identityMap.end())
				{
					out = it->second;
					it = identityMap.find(out.node);
				}
				return out;
			};

			// 活性分析：从 Results 反向标记可达节点
			std::vector<bool> alive(nodeCount, false);
			for (const auto& r : sg.Results())
			{
				auto resolved = resolveIdentity(r);
				MarkReachable(sg, resolved.node, alive, isConst);
			}

			// 对恒等消除节点：标记其替代目标为活
			for (const auto& [deadId, target] : identityMap)
			{
				// 仅当 deadId 自身被 Results 引用时才需要标记
				// 但由于 resolveIdentity 已经在上面处理了 Results，
				// 被消除节点本身不需要标记为活
			}

			// 检查是否有任何变化（常量折叠、恒等消除、死节点）
			bool hasConstFold = false;
			bool hasIdentity = !identityMap.empty();
			bool hasDeadNodes = false;

			for (NodeId nodeId = 0; nodeId < nodeCount; ++nodeId)
			{
				if (isConst[nodeId] && !std::holds_alternative<ConstantNode>(sg.GetNodeEntry(nodeId).node))
				{
					hasConstFold = true;
				}
				if (!alive[nodeId])
				{
					hasDeadNodes = true;
				}
			}

			if (!hasConstFold && !hasIdentity && !hasDeadNodes)
			{
				return;
			}

			// 重建子图
			Subgraph newSg;
			std::vector<NodeId> nodeMap(nodeCount, static_cast<NodeId>(-1));

			auto remapOutput = [&](NodeOutput out) -> NodeOutput {
				auto resolved = resolveIdentity(out);
				return { nodeMap[resolved.node], resolved.port };
			};

			for (NodeId oldId = 0; oldId < nodeCount; ++oldId)
			{
				if (!alive[oldId])
				{
					continue;
				}

				const auto& entry = sg.GetNodeEntry(oldId);

				if (isConst[oldId] && !std::holds_alternative<ConstantNode>(entry.node))
				{
					// 折叠为 ConstantNode
					auto cpuTensor = std::move(*constValues[oldId]);
					auto polyTensor = cpuTensor.CopyToDevice(PolymorphicDevice{ CPU{} });
					auto newId = newSg.AddNode(ConstantNode{ std::move(polyTensor) },
					                           { entry.outputInfos.begin(), entry.outputInfos.end() });
					nodeMap[oldId] = newId;
				}
				else
				{
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
		}
	};
} // namespace LiteNN

#endif
