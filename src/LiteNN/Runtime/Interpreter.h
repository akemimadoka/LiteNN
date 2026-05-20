#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/DataMovement.h>
#include <LiteNN/Graph.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

#ifndef LITENN_RUNTIME_INTERPRETER_H
#define LITENN_RUNTIME_INTERPRETER_H

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
			Validation::ValidateGraph(graph);
			return RunSubgraphUnchecked(graph, subgraphId, inputs, std::move(device));
		}

		// 执行前向子图（自动初始化 activation store）
		std::vector<Tensor<D>> RunForward(const Graph& graph, std::span<const Tensor<D>> inputs, D device = D{})
		{
			Validation::ValidateGraph(graph);
			activationStore_.clear();
			activationStore_.resize(graph.ActivationSlotCount());
			tapeStore_.clear();
			tapeStore_.resize(graph.TapeSlotCount());
			return RunSubgraphUnchecked(graph, graph.Forward(), inputs, std::move(device));
		}

		// 执行反向子图（使用前向已填充的 activation store）
		std::vector<Tensor<D>> RunBackward(const Graph& graph, std::span<const Tensor<D>> inputs, D device = D{})
		{
			Validation::ValidateGraph(graph);
			const auto backwardId = graph.Backward();
			if (!backwardId)
			{
				throw std::runtime_error("Graph has no backward subgraph");
			}
			return RunSubgraphUnchecked(graph, *backwardId, inputs, std::move(device));
		}

	private:
		static void ValidateRuntimeInputs(const Graph& graph, SubgraphId subgraphId, std::span<const Tensor<D>> inputs)
		{
			const auto& subgraph = graph.GetSubgraph(subgraphId);
			if (inputs.size() != subgraph.Params().size())
			{
				throw std::runtime_error(std::format("RunSubgraph input count mismatch for subgraph {}: expected {}, got {}",
				                                     subgraphId, subgraph.Params().size(), inputs.size()));
			}
			for (std::size_t i = 0; i < inputs.size(); ++i)
			{
				const auto& param = subgraph.Params()[i];
				if (inputs[i].DType() != param.dtype || !Validation::SameShape(inputs[i].Shape().Dims, param.shape))
				{
					throw std::runtime_error(std::format(
					    "RunSubgraph input {} mismatch for subgraph {}: expected {}, got {}", i, subgraphId,
					    Validation::FormatInfo(param.dtype, param.shape),
					    Validation::FormatInfo(inputs[i].DType(), inputs[i].Shape().Dims)));
				}
			}
		}

		std::vector<Tensor<D>> RunSubgraphUnchecked(const Graph& graph, SubgraphId subgraphId,
		                                            std::span<const Tensor<D>> inputs, D device = D{})
		{
			const auto& subgraph = graph.GetSubgraph(subgraphId);
			ValidateRuntimeInputs(graph, subgraphId, inputs);

			// slot 表: slots[nodeId] 存储该节点各 port 的输出张量
			// TODO: 如果可能，进行 flatten，直接用一个 vector 存储所有节点的输出，避免多层 vector 的开销
			std::vector<std::vector<Tensor<D>>> slots(subgraph.NodeCount());

			for (NodeId nodeId = 0; nodeId < subgraph.NodeCount(); ++nodeId)
			{
				const auto& entry = subgraph.GetNodeEntry(nodeId);

				try
				{
					std::visit([&](const auto& node) { Execute(graph, entry, nodeId, node, slots, inputs, device); },
					           entry.node);
				}
				catch (const std::exception& ex)
				{
					throw std::runtime_error(std::format("Interpreter failed at subgraph {}, node {} ({}): {}",
					                                     subgraphId, nodeId, Validation::NodeKindName(entry.node),
					                                     ex.what()));
				}
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

		template <DataType TypeValue, typename T>
		static bool ArgsortComesBefore(const T& lhsValue, std::int32_t lhsIndex, const T& rhsValue,
		                              std::int32_t rhsIndex, SortOrder order)
		{
			if constexpr (TypeValue == DataType::Float32 || TypeValue == DataType::Float64 ||
			              TypeValue == DataType::Float16 || TypeValue == DataType::BFloat16 ||
			              TypeValue == DataType::Float8E4M3 || TypeValue == DataType::Float8E5M2)
			{
				const auto lhsIsNan = std::isnan(static_cast<double>(lhsValue));
				const auto rhsIsNan = std::isnan(static_cast<double>(rhsValue));
				if (lhsIsNan != rhsIsNan)
				{
					return !lhsIsNan;
				}
				if (lhsIsNan && rhsIsNan)
				{
					return lhsIndex < rhsIndex;
				}
			}

			if (lhsValue < rhsValue)
			{
				return order == SortOrder::Ascending;
			}
			if (rhsValue < lhsValue)
			{
				return order == SortOrder::Descending;
			}
			return lhsIndex < rhsIndex;
		}

		static Tensor<CPU> EvalArgsort(const Tensor<CPU>& input, SortOrder order, std::size_t axis)
		{
			if (input.Shape().NumDim() == 0)
			{
				throw std::runtime_error("ArgsortNode requires a rank >= 1 tensor");
			}
			if (axis >= input.Shape().NumDim())
			{
				throw std::runtime_error("ArgsortNode axis out of range");
			}
			if (input.Shape()[axis] > static_cast<std::size_t>(std::numeric_limits<std::int32_t>::max()))
			{
				throw std::runtime_error("ArgsortNode sort dimension exceeds Int32 index range");
			}

			CPU cpu;
			Tensor<CPU> result(Uninitialized, input.Shape(), DataType::Int32, cpu);
			const auto axisSize = input.Shape()[axis];
			auto outerSize = 1uz;
			for (auto dim = 0uz; dim < axis; ++dim)
			{
				outerSize *= input.Shape()[dim];
			}
			auto innerSize = 1uz;
			for (auto dim = axis + 1; dim < input.Shape().NumDim(); ++dim)
			{
				innerSize *= input.Shape()[dim];
			}

			EnumDispatch(input.DType(), [&]<DataType TypeValue> {
				using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
				if constexpr (TypeValue == DataType::Bool)
				{
					throw std::runtime_error("ArgsortNode does not support Bool tensors");
				}
				else
				{
					const auto* src = static_cast<const T*>(input.RawData());
					auto* dst = static_cast<std::int32_t*>(result.RawData());
					std::vector<std::int32_t> orderIndices(axisSize);

					for (auto outer = 0uz; outer < outerSize; ++outer)
					{
						for (auto inner = 0uz; inner < innerSize; ++inner)
						{
							std::iota(orderIndices.begin(), orderIndices.end(), std::int32_t{ 0 });
							std::stable_sort(orderIndices.begin(), orderIndices.end(), [&](std::int32_t lhs, std::int32_t rhs) {
								const auto lhsOffset = (outer * axisSize + static_cast<std::size_t>(lhs)) * innerSize + inner;
								const auto rhsOffset = (outer * axisSize + static_cast<std::size_t>(rhs)) * innerSize + inner;
								return ArgsortComesBefore<TypeValue>(src[lhsOffset], lhs, src[rhsOffset], rhs, order);
							});

							for (auto index = 0uz; index < axisSize; ++index)
							{
								dst[(outer * axisSize + index) * innerSize + inner] = orderIndices[index];
							}
						}
					}
				}
			});

			return result;
		}

		static Tensor<CPU> EvalMulMatId(const Tensor<CPU>& as, const Tensor<CPU>& b, const Tensor<CPU>& ids)
		{
			CPU cpu;
			Tensor<CPU> result(Uninitialized, { as.Shape()[1], ids.Shape()[0], b.Shape()[2] }, DataType::Float32, cpu);
			auto* dst = static_cast<float*>(result.RawData());

			EnumDispatch(as.DType(), [&]<DataType AsTypeValue> {
				using AsT = typename DeviceTraits<CPU>::template DataTypeMapping<AsTypeValue>;
				EnumDispatch(b.DType(), [&]<DataType BTypeValue> {
					using BT = typename DeviceTraits<CPU>::template DataTypeMapping<BTypeValue>;

					auto run = [&]<typename IdT>() {
						const auto* asPtr = static_cast<const AsT*>(as.RawData());
						const auto* bPtr = static_cast<const BT*>(b.RawData());
						const auto* idsPtr = static_cast<const IdT*>(ids.RawData());

						const auto k = as.Shape()[0];
						const auto m = as.Shape()[1];
						const auto matCount = as.Shape()[2];
						const auto usedExperts = ids.Shape()[0];
						const auto tokenCount = ids.Shape()[1];
						const auto bUsed = b.Shape()[1];

						for (auto outM = 0uz; outM < m; ++outM)
						{
							for (auto used = 0uz; used < usedExperts; ++used)
							{
								for (auto token = 0uz; token < tokenCount; ++token)
								{
									const auto rawId = idsPtr[used * tokenCount + token];
									if constexpr (std::is_signed_v<IdT>)
									{
										if (rawId < 0)
										{
											throw std::runtime_error("MulMatIdNode ids must be non-negative");
										}
									}

									const auto expertId = static_cast<std::size_t>(rawId);
									if (expertId >= matCount)
									{
										throw std::runtime_error("MulMatIdNode id out of range for expert tensor");
									}

									float acc = 0.0f;
									const auto bSlot = used % bUsed;
									for (auto kk = 0uz; kk < k; ++kk)
									{
										const auto asIndex = ((kk * m) + outM) * matCount + expertId;
										const auto bIndex = ((kk * bUsed) + bSlot) * tokenCount + token;
										acc += static_cast<float>(asPtr[asIndex]) * static_cast<float>(bPtr[bIndex]);
									}

									dst[((outM * usedExperts) + used) * tokenCount + token] = acc;
								}
							}
						}
					};

					switch (ids.DType())
					{
					case DataType::Int32:
						run.template operator()<std::int32_t>();
						break;
					case DataType::Int64:
						run.template operator()<std::int64_t>();
						break;
					default:
						throw std::runtime_error("MulMatIdNode ids must have dtype Int32 or Int64");
					}
				});
			});

			return result;
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

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const QuantizedConstantNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			slots[nodeId].push_back(node.storage.CopyToDevice(device));
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

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const QuantizeNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			if (node.params.scheme != QuantizationScheme::Affine)
			{
				throw std::runtime_error("Interpreter QuantizeNode currently supports affine quantization only");
			}
			const auto& input = GetValue(slots, node.input);
			const auto cpuInput = input.CopyToDevice(CPU{});
			auto quantized = QuantizeAffine(cpuInput, node.params);
			slots[nodeId].push_back(quantized.Storage().CopyToDevice(device));
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const DequantizeNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			if (node.params.scheme != QuantizationScheme::Affine)
			{
				throw std::runtime_error("Interpreter DequantizeNode currently supports affine quantization only");
			}
			const auto& input = GetValue(slots, node.input);
			const auto cpuInput = input.CopyToDevice(CPU{});
			auto dequantized = DequantizeAffine(cpuInput, node.params, node.targetType);
			slots[nodeId].push_back(dequantized.CopyToDevice(device));
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
			slots[nodeId] = RunSubgraphUnchecked(graph, node.callee, args, device);
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
			slots[nodeId] = RunSubgraphUnchecked(graph, branchId, args, device);
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const WhileNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			// 初始 carry values
			std::vector<Tensor<D>> carry;
			carry.reserve(node.initArgs.size());
			for (const auto& arg : node.initArgs)
			{
				carry.push_back(GetValue(slots, arg));
			}

			// 循环
			while (true)
			{
				auto condResult = RunSubgraphUnchecked(graph, node.condBranch, carry, device);
				if (!ReadScalarBool(condResult[0]))
				{
					break;
				}
				carry = RunSubgraphUnchecked(graph, node.bodyBranch, carry, device);
			}

			// 输出 = 最终 carry
			slots[nodeId] = std::move(carry);
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

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const TapeSaveActivationNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			tapeStore_[node.tapeSlotId].push_back(Tensor<D>(input));
			// 透传：输出 = 输入
			slots[nodeId].push_back(Tensor<D>(input));
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const TapeLoadActivationNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			assert(!tapeStore_[node.tapeSlotId].empty());
			slots[nodeId].push_back(std::move(tapeStore_[node.tapeSlotId].back()));
			tapeStore_[node.tapeSlotId].pop_back();
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

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const PermuteNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			const auto& outputInfo = entry.outputInfos[0];

			Tensor<D> result(Uninitialized, outputInfo.shape, outputInfo.dtype, device);
			DeviceTraits<D>::DoPermuteOp(device, result.RawData(), input.DType(), input.Shape(), input.RawData(),
			                            ShapeView{ node.permutation });
			slots[nodeId].push_back(std::move(result));
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const BroadcastToNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			auto cpuResult = Detail::EvalBroadcastTo(input.CopyToDevice(CPU{}), node.targetShape);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const PadNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			auto cpuResult = Detail::EvalPad(input.CopyToDevice(CPU{}), node.lowPads, node.highPads, node.mode,
			                                 node.constantValue);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const GatherNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& data = GetValue(slots, node.data);
			const auto& indices = GetValue(slots, node.indices);
			auto cpuResult = Detail::EvalGather(data.CopyToDevice(CPU{}), indices.CopyToDevice(CPU{}), node.axis);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const ScatterNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& data = GetValue(slots, node.data);
			const auto& indices = GetValue(slots, node.indices);
			const auto& updates = GetValue(slots, node.updates);
			auto cpuResult = Detail::EvalScatter(data.CopyToDevice(CPU{}), indices.CopyToDevice(CPU{}),
			                                     updates.CopyToDevice(CPU{}), node.axis, node.mode);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const ScanNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			auto cpuResult = Detail::EvalScan(input.CopyToDevice(CPU{}), node.axis, node.op);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const SSMScanNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const Tensor<D>* dTensor = node.d ? &GetValue(slots, *node.d) : nullptr;
			const auto cpuD = dTensor ? std::optional{ dTensor->CopyToDevice(CPU{}) } : std::nullopt;
			auto cpuResult = Detail::EvalSSMScan(GetValue(slots, node.state).CopyToDevice(CPU{}),
			                                     GetValue(slots, node.dt).CopyToDevice(CPU{}),
			                                     GetValue(slots, node.a).CopyToDevice(CPU{}),
			                                     GetValue(slots, node.b).CopyToDevice(CPU{}),
			                                     GetValue(slots, node.c).CopyToDevice(CPU{}),
			                                     cpuD ? &*cpuD : nullptr);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const RWKVWKVNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuResult = Detail::EvalRWKVWKV(GetValue(slots, node.key).CopyToDevice(CPU{}),
			                                     GetValue(slots, node.value).CopyToDevice(CPU{}),
			                                     GetValue(slots, node.receptance).CopyToDevice(CPU{}),
			                                     GetValue(slots, node.timeDecay).CopyToDevice(CPU{}),
			                                     GetValue(slots, node.timeFirst).CopyToDevice(CPU{}));
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const SoftmaxNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			auto cpuResult = Detail::EvalSoftmax(input.CopyToDevice(CPU{}), node.axis);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const NormalizationNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const Tensor<D>* scaleTensor = node.scale ? &GetValue(slots, *node.scale) : nullptr;
			const Tensor<D>* biasTensor = node.bias ? &GetValue(slots, *node.bias) : nullptr;
			const auto cpuScale = scaleTensor ? std::optional{ scaleTensor->CopyToDevice(CPU{}) } : std::nullopt;
			const auto cpuBias = biasTensor ? std::optional{ biasTensor->CopyToDevice(CPU{}) } : std::nullopt;
			auto cpuResult = Detail::EvalNormalization(GetValue(slots, node.input).CopyToDevice(CPU{}),
			                                           cpuScale ? &*cpuScale : nullptr,
			                                           cpuBias ? &*cpuBias : nullptr, node.mode,
			                                           node.axis, node.groupCount, node.epsilon);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const BatchMatMulNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuResult = Detail::EvalBatchMatMul(GetValue(slots, node.lhs).CopyToDevice(CPU{}),
			                                        GetValue(slots, node.rhs).CopyToDevice(CPU{}));
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const OutProdNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuResult = Detail::EvalOutProd(GetValue(slots, node.lhs).CopyToDevice(CPU{}),
			                                     GetValue(slots, node.rhs).CopyToDevice(CPU{}));
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const TimestepEmbeddingNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuResult = Detail::EvalTimestepEmbedding(GetValue(slots, node.timesteps).CopyToDevice(CPU{}),
			                                               node.dim, node.maxPeriod);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const SolveTriNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuResult = Detail::EvalSolveTri(GetValue(slots, node.a).CopyToDevice(CPU{}),
			                                      GetValue(slots, node.b).CopyToDevice(CPU{}),
			                                      node.lower, node.unitDiagonal);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const SGDStepNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const Tensor<D>* velocity = node.velocity ? &GetValue(slots, *node.velocity) : nullptr;
			const auto cpuVelocity = velocity ? std::optional{ velocity->CopyToDevice(CPU{}) } : std::nullopt;
			auto cpuResults = Detail::EvalSGDStep(GetValue(slots, node.parameter).CopyToDevice(CPU{}),
			                                      GetValue(slots, node.gradient).CopyToDevice(CPU{}),
			                                      cpuVelocity ? &*cpuVelocity : nullptr,
			                                      node.learningRate, node.momentum, node.weightDecay,
			                                      node.nesterov);
			for (auto& cpuResult : cpuResults)
			{
				if constexpr (std::same_as<D, CPU>)
				{
					slots[nodeId].push_back(std::move(cpuResult));
				}
				else
				{
					slots[nodeId].push_back(cpuResult.CopyToDevice(device));
				}
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const AdamWStepNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuResults = Detail::EvalAdamWStep(GetValue(slots, node.parameter).CopyToDevice(CPU{}),
			                                        GetValue(slots, node.gradient).CopyToDevice(CPU{}),
			                                        GetValue(slots, node.firstMoment).CopyToDevice(CPU{}),
			                                        GetValue(slots, node.secondMoment).CopyToDevice(CPU{}),
			                                        node.learningRate, node.beta1, node.beta2, node.epsilon,
			                                        node.weightDecay, node.step);
			for (auto& cpuResult : cpuResults)
			{
				if constexpr (std::same_as<D, CPU>)
				{
					slots[nodeId].push_back(std::move(cpuResult));
				}
				else
				{
					slots[nodeId].push_back(cpuResult.CopyToDevice(device));
				}
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const Im2ColNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuResult = Detail::EvalIm2Col(GetValue(slots, node.input).CopyToDevice(CPU{}),
			                                    node.kernelShape, node.strides, node.dilations,
			                                    node.lowPads, node.highPads);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const Conv2DNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuInput = GetValue(slots, node.input).CopyToDevice(CPU{});
			auto cpuWeight = GetValue(slots, node.weight).CopyToDevice(CPU{});
			auto cpuBiasStorage = node.bias ? std::optional<Tensor<CPU>>{ GetValue(slots, *node.bias).CopyToDevice(CPU{}) }
			                                : std::nullopt;
			auto cpuResult = Detail::EvalConv2D(cpuInput, cpuWeight,
			                                    cpuBiasStorage ? &*cpuBiasStorage : nullptr,
			                                    node.strides, node.dilations, node.lowPads, node.highPads,
			                                    node.groupCount);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const ConvTranspose2DNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuInput = GetValue(slots, node.input).CopyToDevice(CPU{});
			auto cpuWeight = GetValue(slots, node.weight).CopyToDevice(CPU{});
			auto cpuBiasStorage = node.bias ? std::optional<Tensor<CPU>>{ GetValue(slots, *node.bias).CopyToDevice(CPU{}) }
			                                : std::nullopt;
			auto cpuResult = Detail::EvalConvTranspose2D(cpuInput, cpuWeight,
			                                             cpuBiasStorage ? &*cpuBiasStorage : nullptr,
			                                             node.strides, node.dilations, node.lowPads, node.highPads,
			                                             node.outputPads, node.groupCount);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const Pool2DNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuResult = Detail::EvalPool2D(GetValue(slots, node.input).CopyToDevice(CPU{}),
			                                    node.mode, node.kernelShape, node.strides,
			                                    node.lowPads, node.highPads, node.countIncludePad);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const UpsampleNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuResult = Detail::EvalUpsample(GetValue(slots, node.input).CopyToDevice(CPU{}),
			                                      node.mode, node.outputSpatialShape, node.alignCorners);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const ConcatNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& outputInfo = entry.outputInfos[0];

			std::vector<const void*> srcPtrs;
			std::vector<ShapeView> srcShapes;
			srcPtrs.reserve(node.inputs.size());
			srcShapes.reserve(node.inputs.size());
			for (const auto& input : node.inputs)
			{
				const auto& t = GetValue(slots, input);
				srcPtrs.push_back(t.RawData());
				srcShapes.push_back(t.Shape());
			}

			Tensor<D> result(Uninitialized, outputInfo.shape, outputInfo.dtype, device);
			DeviceTraits<D>::DoConcatOp(device, result.RawData(), outputInfo.dtype, srcPtrs.data(), srcShapes.data(),
			                            srcPtrs.size(), node.axis);
			slots[nodeId].push_back(std::move(result));
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const SliceNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			const auto& outputInfo = entry.outputInfos[0];

			Tensor<D> result(Uninitialized, outputInfo.shape, outputInfo.dtype, device);
			DeviceTraits<D>::DoSliceOp(device, result.RawData(), outputInfo.dtype, input.Shape(), input.RawData(),
			                           node.axis, node.start, node.length);
			slots[nodeId].push_back(std::move(result));
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const GetRowsNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& data = GetValue(slots, node.data);
			const auto& indices = GetValue(slots, node.indices);
			const auto& outputInfo = entry.outputInfos[0];

			Tensor<D> result(Uninitialized, outputInfo.shape, outputInfo.dtype, device);
			DeviceTraits<D>::DoGetRowsOp(device, result.RawData(), data.DType(), data.Shape(), data.RawData(),
			                            indices.DType(), indices.Shape(), indices.RawData());
			slots[nodeId].push_back(std::move(result));
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const ArgsortNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			const auto cpuInput = input.CopyToDevice(CPU{});
			auto cpuResult = EvalArgsort(cpuInput, node.order, node.axis);

			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const MulMatIdNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& as = GetValue(slots, node.as);
			const auto& b = GetValue(slots, node.b);
			const auto& ids = GetValue(slots, node.ids);
			const auto cpuAs = as.CopyToDevice(CPU{});
			const auto cpuB = b.CopyToDevice(CPU{});
			const auto cpuIds = ids.CopyToDevice(CPU{});
			auto cpuResult = EvalMulMatId(cpuAs, cpuB, cpuIds);

			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		void Execute(const Graph& graph, const NodeEntry& entry, NodeId nodeId, const FusedOpNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			// 收集参数
			std::vector<Tensor<D>> args;
			args.reserve(node.args.size());
			for (const auto& arg : node.args)
			{
				args.push_back(GetValue(slots, arg));
			}

			// 执行 body 子图（语义等价于融合前的原语操作）
			slots[nodeId] = RunSubgraphUnchecked(graph, node.body, args, device);
		}

		std::vector<std::optional<Tensor<D>>> activationStore_;
		std::vector<std::vector<Tensor<D>>> tapeStore_;
	};
} // namespace LiteNN::Runtime

#endif
