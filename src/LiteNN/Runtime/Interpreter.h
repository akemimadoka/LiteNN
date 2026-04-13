#ifndef LITENN_RUNTIME_INTERPRETER_H
#define LITENN_RUNTIME_INTERPRETER_H

#include <LiteNN/Graph.h>
#include <optional>
#include <stdexcept>
#include <vector>

namespace LiteNN::Runtime
{
	// 逐节点解释执行 Graph，用于调试和功能验证
	template <Device D>
	class Interpreter
	{
	public:
		// 执行指定子图
		std::vector<Tensor<D>> RunSubgraph(const Graph& graph, SubgraphId subgraphId, std::span<const Tensor<D>> inputs,
		                                   D device = D{})
		{
			const auto& subgraph = graph.GetSubgraph(subgraphId);

			// slot 表: slots[nodeId] 存储该节点各 port 的输出张量
			// TODO: 如果可能，进行 flatten，直接用一个 vector 存储所有节点的输出，避免多层 vector 的开销
			std::vector<std::vector<Tensor<D>>> slots(subgraph.NodeCount());

			for (NodeId nodeId = 0; nodeId < subgraph.NodeCount(); ++nodeId)
			{
				const auto& entry = subgraph.GetNodeEntry(nodeId);

				std::visit([&](const auto& node) { Execute(graph, entry, nodeId, node, slots, inputs, device); },
				           entry.node);
			}

			// 收集结果
			std::vector<Tensor<D>> results;
			results.reserve(subgraph.Results().size());
			for (const auto& output : subgraph.Results())
			{
				results.push_back(GetValue(slots, output));
			}
			return results;
		}

		// 执行前向子图（自动初始化 activation store）
		std::vector<Tensor<D>> RunForward(const Graph& graph, std::span<const Tensor<D>> inputs, D device = D{})
		{
			activationStore_.clear();
			activationStore_.resize(graph.ActivationSlotCount());
			return RunSubgraph(graph, graph.Forward(), inputs, std::move(device));
		}

		// 执行反向子图（使用前向已填充的 activation store）
		std::vector<Tensor<D>> RunBackward(const Graph& graph, std::span<const Tensor<D>> inputs, D device = D{})
		{
			const auto backwardId = graph.Backward();
			if (!backwardId)
			{
				throw std::runtime_error("Graph has no backward subgraph");
			}
			return RunSubgraph(graph, *backwardId, inputs, std::move(device));
		}

	private:
		static const Tensor<D>& GetValue(const std::vector<std::vector<Tensor<D>>>& slots, NodeOutput output)
		{
			return slots[output.node][output.port];
		}

		static bool ReadScalarBool(const Tensor<D>& tensor)
		{
			assert(tensor.NumElements() == 1 && tensor.DType() == DataType::Bool);
			if constexpr (std::same_as<D, CPU>)
			{
				return *static_cast<const bool*>(tensor.RawData());
			}
			else
			{
				const auto cpuTensor = tensor.CopyToDevice(CPU{});
				return *static_cast<const bool*>(cpuTensor.RawData());
			}
		}

		// 各节点类型的执行逻辑

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const ParamRefNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			slots[nodeId].push_back(Tensor<D>(inputs[node.paramIndex]));
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const ConstantNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			slots[nodeId].push_back(node.value.CopyToDevice(device));
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const VariableRefNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			slots[nodeId].push_back(graph.GetVariable(node.variableIndex)->Data().CopyToDevice(device));
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const UnaryOpNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			const auto& outputInfo = entry.outputInfos[0];

			Tensor<D> result(Uninitialized, outputInfo.shape, outputInfo.dtype, device);
			DeviceTraits<D>::DoUnaryOp(device, node.op, result.RawData(), input.DType(), input.Shape(),
			                           input.RawData());
			slots[nodeId].push_back(std::move(result));
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const BinaryOpNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& lhs = GetValue(slots, node.lhs);
			const auto& rhs = GetValue(slots, node.rhs);
			const auto& outputInfo = entry.outputInfos[0];

			Tensor<D> result(Uninitialized, outputInfo.shape, outputInfo.dtype, device);
			DeviceTraits<D>::DoBinaryOp(device, node.op, result.RawData(), lhs.DType(), lhs.Shape(), lhs.RawData(),
			                            rhs.DType(), rhs.Shape(), rhs.RawData());
			slots[nodeId].push_back(std::move(result));
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const CastNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);

			Tensor<D> result(Uninitialized, input.Shape(), node.targetType, device);
			DeviceTraits<D>::ConvertTo(device, input.DType(), input.RawData(), input.NumElements(), node.targetType,
			                           result.RawData());
			slots[nodeId].push_back(std::move(result));
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const CallNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			// 收集参数
			std::vector<Tensor<D>> args;
			args.reserve(node.args.size());
			for (const auto& arg : node.args)
			{
				args.push_back(GetValue(slots, arg));
			}

			// 递归执行被调用的子图
			slots[nodeId] = RunSubgraph(graph, node.callee, args, device);
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const CondNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& condition = GetValue(slots, node.condition);
			const auto condValue = ReadScalarBool(condition);

			// 收集参数
			std::vector<Tensor<D>> args;
			args.reserve(node.args.size());
			for (const auto& arg : node.args)
			{
				args.push_back(GetValue(slots, arg));
			}

			// 根据条件选择分支执行
			const auto branchId = condValue ? node.thenBranch : node.elseBranch;
			slots[nodeId] = RunSubgraph(graph, branchId, args, device);
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const SaveActivationNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			activationStore_[node.slotId] = Tensor<D>(input);
			// 透传：输出 = 输入
			slots[nodeId].push_back(Tensor<D>(input));
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const LoadActivationNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			assert(activationStore_[node.slotId].has_value());
			slots[nodeId].push_back(Tensor<D>(*activationStore_[node.slotId]));
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const ReduceOpNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			const auto& outputInfo = entry.outputInfos[0];

			Tensor<D> result(Uninitialized, outputInfo.shape, outputInfo.dtype, device);
			DeviceTraits<D>::DoReduceOp(device, node.op, result.RawData(), input.DType(), input.Shape(),
			                            input.RawData(), node.axis);
			slots[nodeId].push_back(std::move(result));
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const ReshapeNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			assert(input.NumElements() == ShapeView{ node.targetShape }.NumElements());

			// 复制数据到新 shape 的 tensor
			const auto& outputInfo = entry.outputInfos[0];
			Tensor<D> result(Uninitialized, outputInfo.shape, outputInfo.dtype, device);
			DeviceTraits<D>::ConvertTo(device, input.DType(), input.RawData(), input.NumElements(), outputInfo.dtype,
			                           result.RawData());
			slots[nodeId].push_back(std::move(result));
		}

		std::vector<std::optional<Tensor<D>>> activationStore_;
	};
} // namespace LiteNN::Runtime

#endif
