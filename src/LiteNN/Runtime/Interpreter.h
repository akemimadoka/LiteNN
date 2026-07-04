#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/DataMovement.h>
#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/Graph.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <format>
#include <functional>
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
		using TraceCallback = std::function<void(SubgraphId, NodeId, const NodeEntry&, std::span<const Tensor<D>>)>;
		using QuantizedMatMulCallback = std::function<std::optional<Tensor<CPU>>(const Tensor<CPU>&, const Tensor<CPU>&,
		                                                                         const QuantizationParams&, bool)>;

		Interpreter() = default;

		explicit Interpreter(QuantizedMatMulCallback quantizedMatMulCallback)
		    : quantizedMatMulCallback_(std::move(quantizedMatMulCallback))
		{
		}

		std::vector<Tensor<D>> RunSubgraph(const ExecutablePlan& plan, SubgraphId subgraphId,
		                                   std::span<const Tensor<D>> inputs, D device = D{})
		{
			ValidateExecutablePlan(plan);
			return RunSubgraphUnchecked(plan, subgraphId, inputs, std::move(device));
		}

		std::vector<Tensor<D>> RunForward(const ExecutablePlan& plan, std::span<const Tensor<D>> inputs, D device = D{})
		{
			ValidateExecutablePlan(plan);
			activationStore_.clear();
			activationStore_.resize(plan.activationSlots.size());
			tapeStore_.clear();
			tapeStore_.resize(plan.tapeSlots.size());
			return RunSubgraphUnchecked(plan, plan.forward, inputs, std::move(device));
		}

		std::vector<Tensor<D>> RunForwardWithTrace(const ExecutablePlan& plan, std::span<const Tensor<D>> inputs,
		                                           TraceCallback callback, D device = D{})
		{
			ValidateExecutablePlan(plan);
			activationStore_.clear();
			activationStore_.resize(plan.activationSlots.size());
			tapeStore_.clear();
			tapeStore_.resize(plan.tapeSlots.size());
			auto previousCallback = std::move(traceCallback_);
			traceCallback_ = std::move(callback);
			try
			{
				auto results = RunSubgraphUnchecked(plan, plan.forward, inputs, std::move(device));
				traceCallback_ = std::move(previousCallback);
				return results;
			}
			catch (...)
			{
				traceCallback_ = std::move(previousCallback);
				throw;
			}
		}

		std::vector<Tensor<D>> RunBackward(const ExecutablePlan& plan, std::span<const Tensor<D>> inputs,
		                                   D device = D{})
		{
			ValidateExecutablePlan(plan);
			if (!plan.backward)
			{
				throw std::runtime_error("Graph has no backward subgraph");
			}
			return RunSubgraphUnchecked(plan, *plan.backward, inputs, std::move(device));
		}

	private:
		static void ValidateRuntimeInputs(const Graph& graph, SubgraphId subgraphId, std::span<const Tensor<D>> inputs)
		{
			const auto& subgraph = graph.GetSubgraph(subgraphId);
			if (inputs.size() != subgraph.Params().size())
			{
				throw std::runtime_error(
				    std::format("RunSubgraph input count mismatch for subgraph {}: expected {}, got {}", subgraphId,
				                subgraph.Params().size(), inputs.size()));
			}
			for (std::size_t i = 0; i < inputs.size(); ++i)
			{
				const auto& param = subgraph.Params()[i];
				if (inputs[i].DType() != param.dtype || inputs[i].Shape() != param.shape)
				{
					throw std::runtime_error(
					    std::format("RunSubgraph input {} mismatch for subgraph {}: expected {}, got {}", i, subgraphId,
					                Validation::FormatInfo(param.dtype, param.shape),
					                Validation::FormatInfo(inputs[i].DType(), inputs[i].Shape().Dims)));
				}
			}
		}

		static void ValidateRuntimeInputs(const ExecutablePlan& plan, SubgraphId subgraphId,
		                                  std::span<const Tensor<D>> inputs)
		{
			if (subgraphId >= plan.subgraphs.size())
			{
				throw std::runtime_error(std::format("RunSubgraph subgraph {} is out of range; subgraphCount={}",
				                                     subgraphId, plan.subgraphs.size()));
			}
			const auto& subgraph = plan.subgraphs[subgraphId];
			if (inputs.size() != subgraph.params.size())
			{
				throw std::runtime_error(
				    std::format("RunSubgraph input count mismatch for subgraph {}: expected {}, got {}", subgraphId,
				                subgraph.params.size(), inputs.size()));
			}
			for (std::size_t i = 0; i < inputs.size(); ++i)
			{
				const auto& param = subgraph.params[i];
				if (!param.IsFullyStatic())
				{
					throw std::runtime_error(std::format(
					    "RunSubgraph input {} for subgraph {} has non-static executable type", i, subgraphId));
				}
				const auto expectedShape = param.StaticShape();
				if (inputs[i].DType() != param.dtype || inputs[i].Shape() != ShapeView{ expectedShape })
				{
					throw std::runtime_error(
					    std::format("RunSubgraph input {} mismatch for subgraph {}: expected {}, got {}", i, subgraphId,
					                Validation::FormatInfo(param.dtype, expectedShape),
					                Validation::FormatInfo(inputs[i].DType(), inputs[i].Shape().Dims)));
				}
			}
		}

		static OutputInfo ToOutputInfo(const TensorType& type)
		{
			return OutputInfo::FromType(type);
		}

		static NodeEntry MakeNodeEntry(const ExecutablePlanNode& node)
		{
			std::vector<OutputInfo> outputs;
			outputs.reserve(node.outputs.size());
			for (const auto& output : node.outputs)
			{
				outputs.push_back(ToOutputInfo(output));
			}
			return { node.node, std::move(outputs) };
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
				if (traceCallback_)
				{
					traceCallback_(subgraphId, nodeId, entry,
					               std::span<const Tensor<D>>(slots[nodeId].data(), slots[nodeId].size()));
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

		std::vector<Tensor<D>> RunSubgraphUnchecked(const ExecutablePlan& plan, SubgraphId subgraphId,
		                                            std::span<const Tensor<D>> inputs, D device = D{})
		{
			const auto& subgraph = plan.subgraphs[subgraphId];
			ValidateRuntimeInputs(plan, subgraphId, inputs);

			std::vector<std::vector<Tensor<D>>> slots(subgraph.nodes.size());

			for (NodeId nodeId = 0; nodeId < subgraph.nodes.size(); ++nodeId)
			{
				const auto& planNode = subgraph.nodes[nodeId];
				auto entry = MakeNodeEntry(planNode);

				try
				{
					std::visit([&](const auto& node) { Execute(plan, entry, nodeId, node, slots, inputs, device); },
					           entry.node);
				}
				catch (const std::exception& ex)
				{
					throw std::runtime_error(std::format("Interpreter failed at subgraph {}, node {} ({}): {}",
					                                     subgraphId, nodeId, Validation::NodeKindName(entry.node),
					                                     ex.what()));
				}
				if (traceCallback_)
				{
					traceCallback_(subgraphId, nodeId, entry,
					               std::span<const Tensor<D>>(slots[nodeId].data(), slots[nodeId].size()));
				}
			}

			std::vector<Tensor<D>> results;
			results.reserve(subgraph.results.size());
			for (const auto& output : subgraph.results)
			{
				results.push_back(GetValue(slots, output));
			}
			return results;
		}
		static const Tensor<D>& GetValue(const std::vector<std::vector<Tensor<D>>>& slots, NodeOutput output)
		{
			return slots[output.node][output.port];
		}

		static Tensor<D> LoadVariable(const Graph& graph, std::size_t variableIndex, D& device)
		{
			return graph.GetVariable(variableIndex)->Data().CopyToDevice(device);
		}

		static Tensor<D> LoadVariable(const ExecutablePlan& plan, std::size_t variableIndex, D& device)
		{
			if (variableIndex >= plan.variables.size())
			{
				throw std::runtime_error(std::format("VariableRefNode references variable {}, but variableCount={}",
				                                     variableIndex, plan.variables.size()));
			}
			const auto& storage = plan.variables[variableIndex];
			if (!storage.type.IsFullyStatic())
			{
				throw std::runtime_error("VariableRefNode requires a fully static tensor type");
			}
			if (!storage.region.data)
			{
				throw std::runtime_error("VariableRefNode storage region is not bound");
			}
			const auto shape = storage.type.StaticShape();
			const auto* bytes = static_cast<const std::byte*>(storage.region.data) + storage.region.byteOffset +
			                    storage.storageOffsetBytes;
			if (storage.type.memorySpace == TensorMemorySpace::Host ||
			    storage.type.memorySpace == TensorMemorySpace::Constant ||
			    storage.type.memorySpace == TensorMemorySpace::External ||
			    storage.type.memorySpace == TensorMemorySpace::Unified)
			{
				auto hostView = Tensor<CPU>::UnsafeBorrowed(const_cast<std::byte*>(bytes), ShapeView{ shape },
				                                            storage.type.dtype, CPU{});
				if constexpr (std::same_as<D, CPU>)
				{
					return hostView;
				}
				else
				{
					return hostView.CopyToDevice(device);
				}
			}

			if constexpr (std::same_as<D, CPU>)
			{
				throw std::runtime_error("VariableRefNode device storage cannot be read by a CPU interpreter plan");
			}
			else
			{
				Tensor<D> deviceView(const_cast<std::byte*>(bytes), ShapeView{ shape }, storage.type.dtype, device);
				return Tensor<D>(deviceView);
			}
		}

		static bool ReadScalarBool(const Tensor<D>& tensor)
		{
			assert(tensor.NumElements() == 1 && tensor.DType() == DataType::Bool);
			if constexpr (std::same_as<D, CPU>)
			{
				return *static_cast<const bool*>(tensor.UnsafeRawData());
			}
			else
			{
				const auto cpuTensor = tensor.CopyToDevice(CPU{});
				return *static_cast<const bool*>(cpuTensor.UnsafeRawData());
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
					const auto* src = static_cast<const T*>(input.UnsafeRawData());
					auto* dst = static_cast<std::int32_t*>(result.UnsafeRawData());
					std::vector<std::int32_t> orderIndices(axisSize);

					for (auto outer = 0uz; outer < outerSize; ++outer)
					{
						for (auto inner = 0uz; inner < innerSize; ++inner)
						{
							std::iota(orderIndices.begin(), orderIndices.end(), std::int32_t{ 0 });
							std::stable_sort(
							    orderIndices.begin(), orderIndices.end(), [&](std::int32_t lhs, std::int32_t rhs) {
								    const auto lhsOffset =
								        (outer * axisSize + static_cast<std::size_t>(lhs)) * innerSize + inner;
								    const auto rhsOffset =
								        (outer * axisSize + static_cast<std::size_t>(rhs)) * innerSize + inner;
								    return ArgsortComesBefore<TypeValue>(src[lhsOffset], lhs, src[rhsOffset], rhs,
								                                         order);
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
			auto* dst = static_cast<float*>(result.UnsafeRawData());

			EnumDispatch(as.DType(), [&]<DataType AsTypeValue> {
				using AsT = typename DeviceTraits<CPU>::template DataTypeMapping<AsTypeValue>;
				EnumDispatch(b.DType(), [&]<DataType BTypeValue> {
					using BT = typename DeviceTraits<CPU>::template DataTypeMapping<BTypeValue>;

					auto run = [&]<typename IdT>() {
						const auto* asPtr = static_cast<const AsT*>(as.UnsafeRawData());
						const auto* bPtr = static_cast<const BT*>(b.UnsafeRawData());
						const auto* idsPtr = static_cast<const IdT*>(ids.UnsafeRawData());

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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const ParamRefNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			slots[nodeId].push_back(Tensor<D>(inputs[node.paramIndex]));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const ConstantNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			slots[nodeId].push_back(node.value.CopyToDevice(device));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId,
		             const QuantizedConstantNode& node, std::vector<std::vector<Tensor<D>>>& slots,
		             std::span<const Tensor<D>> inputs, D& device)
		{
			slots[nodeId].push_back(node.storage.CopyToDevice(device));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const VariableRefNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			slots[nodeId].push_back(LoadVariable(graph, node.variableIndex, device));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const UnaryOpNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			const auto& outputInfo = entry.outputInfos[0];

			Tensor<D> result(Uninitialized, outputInfo.shape, outputInfo.dtype, device);
			DeviceTraits<D>::DoUnaryOp(device, node.op, result.UnsafeRawData(), input.DType(), input.Shape(),
			                           input.UnsafeRawData());
			slots[nodeId].push_back(std::move(result));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const BinaryOpNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& lhs = GetValue(slots, node.lhs);
			const auto& rhs = GetValue(slots, node.rhs);
			const auto& outputInfo = entry.outputInfos[0];

			Tensor<D> result(Uninitialized, outputInfo.shape, outputInfo.dtype, device);
			DeviceTraits<D>::DoBinaryOp(device, node.op, result.UnsafeRawData(), lhs.DType(), lhs.Shape(),
			                            lhs.UnsafeRawData(), rhs.DType(), rhs.Shape(), rhs.UnsafeRawData());
			slots[nodeId].push_back(std::move(result));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const CastNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);

			Tensor<D> result(Uninitialized, input.Shape(), node.targetType, device);
			DeviceTraits<D>::ConvertTo(device, input.DType(), input.UnsafeRawData(), input.NumElements(),
			                           node.targetType, result.UnsafeRawData());
			slots[nodeId].push_back(std::move(result));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const QuantizeNode& node,
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const DequantizeNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			const auto cpuInput = input.CopyToDevice(CPU{});
			Tensor<CPU> dequantized = [&] {
				if (node.params.scheme == QuantizationScheme::Affine)
				{
					return DequantizeAffine(cpuInput, node.params, node.targetType);
				}
				if (node.params.scheme == QuantizationScheme::Block &&
				    IsPackedNibbleQuantizedBlockFormat(node.params.blockFormat))
				{
					return DequantizePackedNibble(cpuInput, node.params, node.targetType);
				}
				if (node.params.scheme == QuantizationScheme::Block &&
				    IsGGMLQuantizedBlockFormat(node.params.blockFormat))
				{
					return DequantizeGGMLBlock(cpuInput, node.params, node.targetType);
				}
				throw std::runtime_error(
				    "Interpreter DequantizeNode currently supports affine, packed nibble, and selected GGML block "
				    "quantization only");
			}();
			slots[nodeId].push_back(dequantized.CopyToDevice(device));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId,
		             const QuantizedMatMulNode& node, std::vector<std::vector<Tensor<D>>>& slots,
		             std::span<const Tensor<D>> inputs, D& device)
		{
			auto evalOnCPU = [&](const Tensor<CPU>& lhs, const Tensor<CPU>& rhsStorage) {
				std::optional<Tensor<CPU>> cpuResult;
				if (quantizedMatMulCallback_)
				{
					cpuResult = quantizedMatMulCallback_(lhs, rhsStorage, node.params, node.transposeRhs);
				}
				if (!cpuResult)
				{
					if (node.transposeRhs)
					{
						throw std::runtime_error(
						    "Interpreter QuantizedMatMulNode requires a backend callback for transposed block weights");
					}
					cpuResult = EvalQuantizedMatMul(lhs, rhsStorage, node.params, node.params.expressedType);
				}
				return std::move(*cpuResult);
			};
			std::optional<Tensor<CPU>> cpuResult;
			if constexpr (std::same_as<D, CPU>)
			{
				cpuResult = evalOnCPU(GetValue(slots, node.lhs), GetValue(slots, node.rhsStorage));
			}
			else
			{
				const auto lhs = GetValue(slots, node.lhs).CopyToDevice(CPU{});
				const auto rhsStorage = GetValue(slots, node.rhsStorage).CopyToDevice(CPU{});
				cpuResult = evalOnCPU(lhs, rhsStorage);
			}
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(*cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult->CopyToDevice(device));
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId,
		             const GroupedQuantizedMatMulNode& node, std::vector<std::vector<Tensor<D>>>& slots,
		             std::span<const Tensor<D>> inputs, D& device)
		{
			if (node.rhsStorages.size() != node.outputWidths.size() || node.rhsStorages.empty())
			{
				throw std::runtime_error("GroupedQuantizedMatMulNode projection metadata is inconsistent");
			}
			auto lhsCPU = [&]() {
				if constexpr (std::same_as<D, CPU>)
				{
					return GetValue(slots, node.lhs);
				}
				else
				{
					return GetValue(slots, node.lhs).CopyToDevice(CPU{});
				}
			}();
			Tensor<CPU> cpuResult(Uninitialized, entry.outputInfos[0].shape, entry.outputInfos[0].dtype);
			auto* output = static_cast<std::byte*>(cpuResult.UnsafeRawData());
			const auto rows = entry.outputInfos[0].shape[0];
			const auto totalColumns = entry.outputInfos[0].shape[1];
			const auto elementBytes = ElementByteSize(entry.outputInfos[0].dtype);
			std::size_t columnOffset = 0;
			for (std::size_t projection = 0; projection < node.rhsStorages.size(); ++projection)
			{
				auto params = node.params;
				params.expressedShape = { node.outputWidths[projection], lhsCPU.Shape()[1] };
				auto rhsCPU = [&]() {
					if constexpr (std::same_as<D, CPU>)
					{
						return GetValue(slots, node.rhsStorages[projection]);
					}
					else
					{
						return GetValue(slots, node.rhsStorages[projection]).CopyToDevice(CPU{});
					}
				}();
				std::optional<Tensor<CPU>> projectionResult;
				if (quantizedMatMulCallback_)
				{
					projectionResult = quantizedMatMulCallback_(lhsCPU, rhsCPU, params, node.transposeRhs);
				}
				if (!projectionResult)
				{
					if (node.transposeRhs)
					{
						throw std::runtime_error(
						    "Interpreter GroupedQuantizedMatMulNode requires a backend callback for transposed block "
						    "weights");
					}
					projectionResult = EvalQuantizedMatMul(lhsCPU, rhsCPU, params, params.expressedType);
				}
				const auto projectionColumns = node.outputWidths[projection];
				const auto* projectionData = static_cast<const std::byte*>(projectionResult->UnsafeRawData());
				for (std::size_t row = 0; row < rows; ++row)
				{
					std::memcpy(output + (row * totalColumns + columnOffset) * elementBytes,
					            projectionData + row * projectionColumns * elementBytes,
					            projectionColumns * elementBytes);
				}
				columnOffset += projectionColumns;
			}
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		template <typename IdT>
		static void FillQuantizedRows(const Tensor<CPU>& storage, const Tensor<CPU>& indices,
		                              const QuantizationParams& params, Tensor<CPU>& result)
		{
			const auto* ids = static_cast<const IdT*>(indices.UnsafeRawData());
			auto* output = static_cast<float*>(result.UnsafeRawData());
			const auto rowCount = params.expressedShape[0];
			const auto rowWidth = params.expressedShape[1];
			std::vector<float> row(rowWidth);
			for (std::size_t i = 0; i < indices.NumElements(); ++i)
			{
				const auto rawId = ids[i];
				if constexpr (std::is_signed_v<IdT>)
				{
					if (rawId < 0)
					{
						throw std::runtime_error("QuantizedGetRowsNode index must be non-negative");
					}
				}
				const auto rowIndex = static_cast<std::size_t>(rawId);
				if (rowIndex >= rowCount)
				{
					throw std::runtime_error("QuantizedGetRowsNode index is out of range");
				}
				if (params.scheme == QuantizationScheme::Block && IsGGMLQuantizedBlockFormat(params.blockFormat))
				{
					DequantizeGGMLBlockRowToFloat32(storage, params, rowIndex, row.data());
				}
				else
				{
					const auto full =
					    params.scheme == QuantizationScheme::Affine
					        ? DequantizeAffine(storage, params, DataType::Float32)
					    : IsPackedNibbleQuantizedBlockFormat(params.blockFormat)
					        ? DequantizePackedNibble(storage, params, DataType::Float32)
					        : throw std::runtime_error("QuantizedGetRowsNode unsupported quantization format");
					const auto* fullData = static_cast<const float*>(full.UnsafeRawData());
					std::copy_n(fullData + rowIndex * rowWidth, rowWidth, row.data());
				}
				std::copy_n(row.data(), rowWidth, output + i * rowWidth);
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId,
		             const QuantizedGetRowsNode& node, std::vector<std::vector<Tensor<D>>>& slots,
		             std::span<const Tensor<D>> inputs, D& device)
		{
			const auto runOnCPU = [&](const Tensor<CPU>& storage, const Tensor<CPU>& indices) {
				Tensor<CPU> cpuResult(Uninitialized, entry.outputInfos[0].shape, DataType::Float32);
				switch (indices.DType())
				{
				case DataType::Int32:
					FillQuantizedRows<std::int32_t>(storage, indices, node.params, cpuResult);
					break;
				case DataType::Int64:
					FillQuantizedRows<std::int64_t>(storage, indices, node.params, cpuResult);
					break;
				default:
					throw std::runtime_error("QuantizedGetRowsNode indices must have dtype Int32 or Int64");
				}
				return cpuResult;
			};
			if (node.params.expressedType != DataType::Float32)
			{
				throw std::runtime_error("Interpreter QuantizedGetRowsNode currently emits Float32 rows only");
			}
			std::optional<Tensor<CPU>> cpuResult;
			if constexpr (std::same_as<D, CPU>)
			{
				cpuResult = runOnCPU(GetValue(slots, node.storage), GetValue(slots, node.indices));
			}
			else
			{
				const auto storage = GetValue(slots, node.storage).CopyToDevice(CPU{});
				const auto indices = GetValue(slots, node.indices).CopyToDevice(CPU{});
				cpuResult = runOnCPU(storage, indices);
			}
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(*cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult->CopyToDevice(device));
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const CallNode& node,
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const CondNode& node,
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const WhileNode& node,
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const SaveActivationNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			activationStore_[node.slotId] = Tensor<D>(input);
			// 透传：输出 = 输入
			slots[nodeId].push_back(Tensor<D>(input));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const LoadActivationNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			assert(activationStore_[node.slotId].has_value());
			slots[nodeId].push_back(Tensor<D>(*activationStore_[node.slotId]));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId,
		             const TapeSaveActivationNode& node, std::vector<std::vector<Tensor<D>>>& slots,
		             std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			tapeStore_[node.tapeSlotId].push_back(Tensor<D>(input));
			// 透传：输出 = 输入
			slots[nodeId].push_back(Tensor<D>(input));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId,
		             const TapeLoadActivationNode& node, std::vector<std::vector<Tensor<D>>>& slots,
		             std::span<const Tensor<D>> inputs, D& device)
		{
			assert(!tapeStore_[node.tapeSlotId].empty());
			slots[nodeId].push_back(std::move(tapeStore_[node.tapeSlotId].back()));
			tapeStore_[node.tapeSlotId].pop_back();
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const ReduceOpNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			const auto& outputInfo = entry.outputInfos[0];

			Tensor<D> result(Uninitialized, outputInfo.shape, outputInfo.dtype, device);
			DeviceTraits<D>::DoReduceOp(device, node.op, result.UnsafeRawData(), input.DType(), input.Shape(),
			                            input.UnsafeRawData(), node.axis);
			slots[nodeId].push_back(std::move(result));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const ReshapeNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			assert(input.NumElements() == ShapeView{ node.targetShape }.NumElements());

			// 复制数据到新 shape 的 tensor
			const auto& outputInfo = entry.outputInfos[0];
			Tensor<D> result(Uninitialized, outputInfo.shape, outputInfo.dtype, device);
			DeviceTraits<D>::ConvertTo(device, input.DType(), input.UnsafeRawData(), input.NumElements(),
			                           outputInfo.dtype, result.UnsafeRawData());
			slots[nodeId].push_back(std::move(result));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const PermuteNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			const auto& outputInfo = entry.outputInfos[0];

			Tensor<D> result(Uninitialized, outputInfo.shape, outputInfo.dtype, device);
			DeviceTraits<D>::DoPermuteOp(device, result.UnsafeRawData(), input.DType(), input.Shape(),
			                             input.UnsafeRawData(), ShapeView{ node.permutation });
			slots[nodeId].push_back(std::move(result));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const BroadcastToNode& node,
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const PadNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			auto cpuResult =
			    Detail::EvalPad(input.CopyToDevice(CPU{}), node.lowPads, node.highPads, node.mode, node.constantValue);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const GatherNode& node,
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const ScatterNode& node,
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const ScanNode& node,
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const SSMScanNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const Tensor<D>* dTensor = node.d ? &GetValue(slots, *node.d) : nullptr;
			const auto cpuD = dTensor ? std::optional{ dTensor->CopyToDevice(CPU{}) } : std::nullopt;
			auto cpuResult = Detail::EvalSSMScan(
			    GetValue(slots, node.state).CopyToDevice(CPU{}), GetValue(slots, node.dt).CopyToDevice(CPU{}),
			    GetValue(slots, node.a).CopyToDevice(CPU{}), GetValue(slots, node.b).CopyToDevice(CPU{}),
			    GetValue(slots, node.c).CopyToDevice(CPU{}), cpuD ? &*cpuD : nullptr);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const RWKVWKVNode& node,
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId,
		             const ActivePrefixAttentionNode& node, std::vector<std::vector<Tensor<D>>>& slots,
		             std::span<const Tensor<D>> inputs, D& device)
		{
			const auto query = GetValue(slots, node.query).CopyToDevice(CPU{});
			const auto keys = GetValue(slots, node.keys).CopyToDevice(CPU{});
			const auto values = GetValue(slots, node.values).CopyToDevice(CPU{});
			const auto position = GetValue(slots, node.currentPosition).CopyToDevice(CPU{});
			if (query.DType() != DataType::Float32 || keys.DType() != DataType::Float32 ||
			    values.DType() != DataType::Float32 || position.DType() != DataType::Int64)
			{
				throw std::runtime_error(
				    "Interpreter ActivePrefixAttentionNode currently supports Float32 + Int64 only");
			}
			const auto& queryShape = query.Shape();
			const auto& keysShape = keys.Shape();
			const auto& valuesShape = values.Shape();
			const auto capacity = keysShape[0];
			const auto keysRank = keysShape.Dims.size();
			const auto headDim = keysRank == 2 ? keysShape[1] : keysShape[2];
			const auto valueDim = valuesShape.Dims.size() == 2 ? valuesShape[1] : valuesShape[2];
			const auto kvHeads = keysRank == 2 ? 1uz : keysShape[1];
			if (node.kvHeadIndex >= kvHeads)
			{
				throw std::runtime_error("ActivePrefixAttentionNode kvHeadIndex is out of range");
			}
			const auto rawPosition = *static_cast<const std::int64_t*>(position.UnsafeRawData());
			if (rawPosition < 0)
			{
				throw std::runtime_error("ActivePrefixAttentionNode currentPosition must be non-negative");
			}
			const auto active = std::min<std::size_t>(capacity, static_cast<std::size_t>(rawPosition) + 1);
			const auto* q = static_cast<const float*>(query.UnsafeRawData());
			const auto* k = static_cast<const float*>(keys.UnsafeRawData());
			const auto* v = static_cast<const float*>(values.UnsafeRawData());
			Tensor<CPU> cpuResult(Uninitialized, entry.outputInfos[0].shape, DataType::Float32);
			auto* out = static_cast<float*>(cpuResult.UnsafeRawData());
			std::fill_n(out, cpuResult.NumElements(), 0.0F);
			if (queryShape[0] != 1 || active == 0)
			{
				throw std::runtime_error("ActivePrefixAttentionNode expects one query and at least one active key");
			}
			float maxScore = -std::numeric_limits<float>::infinity();
			std::vector<float> scores(active);
			for (std::size_t row = 0; row < active; ++row)
			{
				float score = 0.0F;
				for (std::size_t col = 0; col < headDim; ++col)
				{
					const auto keyIndex =
					    keysRank == 2 ? row * headDim + col : (row * kvHeads + node.kvHeadIndex) * headDim + col;
					score += q[col] * k[keyIndex];
				}
				score *= static_cast<float>(node.scale);
				scores[row] = score;
				maxScore = std::max(maxScore, score);
			}
			float denom = 0.0F;
			for (auto& score : scores)
			{
				score = std::exp(score - maxScore);
				denom += score;
			}
			for (std::size_t row = 0; row < active; ++row)
			{
				const auto weight = scores[row] / denom;
				for (std::size_t col = 0; col < valueDim; ++col)
				{
					const auto valueIndex = valuesShape.Dims.size() == 2
					                            ? row * valueDim + col
					                            : (row * kvHeads + node.kvHeadIndex) * valueDim + col;
					out[col] += weight * v[valueIndex];
				}
			}
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId,
		             const GroupedActivePrefixAttentionNode& node, std::vector<std::vector<Tensor<D>>>& slots,
		             std::span<const Tensor<D>> inputs, D& device)
		{
			const auto queries = GetValue(slots, node.queries).CopyToDevice(CPU{});
			const auto keys = GetValue(slots, node.keys).CopyToDevice(CPU{});
			const auto values = GetValue(slots, node.values).CopyToDevice(CPU{});
			const auto position = GetValue(slots, node.currentPosition).CopyToDevice(CPU{});
			if (queries.DType() != DataType::Float32 || keys.DType() != DataType::Float32 ||
			    values.DType() != DataType::Float32 || position.DType() != DataType::Int64)
			{
				throw std::runtime_error(
				    "Interpreter GroupedActivePrefixAttentionNode currently supports Float32 + Int64 only");
			}
			const auto& queryShape = queries.Shape();
			const auto& keysShape = keys.Shape();
			const auto& valuesShape = values.Shape();
			if (queryShape.Dims.size() != 2 || keysShape.Dims.size() != 3 || valuesShape.Dims.size() != 3)
			{
				throw std::runtime_error(
				    "GroupedActivePrefixAttentionNode requires rank-2 queries and rank-3 KV cache");
			}
			const auto queryHeads = queryShape[0];
			const auto headDim = queryShape[1];
			const auto capacity = keysShape[0];
			const auto kvHeads = keysShape[1];
			const auto valueDim = valuesShape[2];
			if (node.queryGroupsPerKVHead == 0 || queryHeads > kvHeads * node.queryGroupsPerKVHead ||
			    keysShape[2] != headDim || valuesShape[0] != capacity || valuesShape[1] != kvHeads)
			{
				throw std::runtime_error("GroupedActivePrefixAttentionNode received incompatible query/KV shapes");
			}
			const auto rawPosition = *static_cast<const std::int64_t*>(position.UnsafeRawData());
			if (rawPosition < 0)
			{
				throw std::runtime_error("GroupedActivePrefixAttentionNode currentPosition must be non-negative");
			}
			const auto active = std::min<std::size_t>(capacity, static_cast<std::size_t>(rawPosition) + 1);
			if (active == 0)
			{
				throw std::runtime_error("GroupedActivePrefixAttentionNode expects at least one active key");
			}

			const auto* q = static_cast<const float*>(queries.UnsafeRawData());
			const auto* k = static_cast<const float*>(keys.UnsafeRawData());
			const auto* v = static_cast<const float*>(values.UnsafeRawData());
			Tensor<CPU> cpuResult(Uninitialized, entry.outputInfos[0].shape, DataType::Float32);
			auto* out = static_cast<float*>(cpuResult.UnsafeRawData());
			std::fill_n(out, cpuResult.NumElements(), 0.0F);
			std::vector<float> scores(active);
			for (std::size_t queryHead = 0; queryHead < queryHeads; ++queryHead)
			{
				const auto kvHead = queryHead / node.queryGroupsPerKVHead;
				const auto* query = q + queryHead * headDim;
				float maxScore = -std::numeric_limits<float>::infinity();
				for (std::size_t row = 0; row < active; ++row)
				{
					float score = 0.0F;
					for (std::size_t col = 0; col < headDim; ++col)
					{
						score += query[col] * k[(row * kvHeads + kvHead) * headDim + col];
					}
					score *= static_cast<float>(node.scale);
					scores[row] = score;
					maxScore = std::max(maxScore, score);
				}
				float denom = 0.0F;
				for (auto& score : scores)
				{
					score = std::exp(score - maxScore);
					denom += score;
				}
				for (std::size_t row = 0; row < active; ++row)
				{
					const auto weight = scores[row] / denom;
					for (std::size_t col = 0; col < valueDim; ++col)
					{
						out[queryHead * valueDim + col] += weight * v[(row * kvHeads + kvHead) * valueDim + col];
					}
				}
			}
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const SoftmaxNode& node,
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const RoPENode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const Tensor<D>* positions = node.positions ? &GetValue(slots, *node.positions) : nullptr;
			const auto cpuPositions = positions ? std::optional{ positions->CopyToDevice(CPU{}) } : std::nullopt;
			auto cpuResult = Detail::EvalRoPE(GetValue(slots, node.input).CopyToDevice(CPU{}),
			                                  cpuPositions ? &*cpuPositions : nullptr, node.base, node.frequencyScale,
			                                  node.positionOffset);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId,
		             const CrossEntropyLossNode& node, std::vector<std::vector<Tensor<D>>>& slots,
		             std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuResult = Detail::EvalCrossEntropyLoss(GetValue(slots, node.logits).CopyToDevice(CPU{}),
			                                              GetValue(slots, node.labels).CopyToDevice(CPU{}));
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId,
		             const CrossEntropyLossBackwardNode& node, std::vector<std::vector<Tensor<D>>>& slots,
		             std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuResult = Detail::EvalCrossEntropyLossBackward(GetValue(slots, node.grad).CopyToDevice(CPU{}),
			                                                      GetValue(slots, node.logits).CopyToDevice(CPU{}),
			                                                      GetValue(slots, node.labels).CopyToDevice(CPU{}));
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const NormalizationNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const Tensor<D>* scaleTensor = node.scale ? &GetValue(slots, *node.scale) : nullptr;
			const Tensor<D>* biasTensor = node.bias ? &GetValue(slots, *node.bias) : nullptr;
			const auto cpuScale = scaleTensor ? std::optional{ scaleTensor->CopyToDevice(CPU{}) } : std::nullopt;
			const auto cpuBias = biasTensor ? std::optional{ biasTensor->CopyToDevice(CPU{}) } : std::nullopt;
			auto cpuResult = Detail::EvalNormalization(GetValue(slots, node.input).CopyToDevice(CPU{}),
			                                           cpuScale ? &*cpuScale : nullptr, cpuBias ? &*cpuBias : nullptr,
			                                           node.mode, node.axis, node.groupCount, node.epsilon);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const BatchMatMulNode& node,
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const OutProdNode& node,
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId,
		             const TimestepEmbeddingNode& node, std::vector<std::vector<Tensor<D>>>& slots,
		             std::span<const Tensor<D>> inputs, D& device)
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const SolveTriNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuResult =
			    Detail::EvalSolveTri(GetValue(slots, node.a).CopyToDevice(CPU{}),
			                         GetValue(slots, node.b).CopyToDevice(CPU{}), node.lower, node.unitDiagonal);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const SGDStepNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const Tensor<D>* velocity = node.velocity ? &GetValue(slots, *node.velocity) : nullptr;
			const auto cpuVelocity = velocity ? std::optional{ velocity->CopyToDevice(CPU{}) } : std::nullopt;
			auto cpuResults = Detail::EvalSGDStep(GetValue(slots, node.parameter).CopyToDevice(CPU{}),
			                                      GetValue(slots, node.gradient).CopyToDevice(CPU{}),
			                                      cpuVelocity ? &*cpuVelocity : nullptr, node.learningRate,
			                                      node.momentum, node.weightDecay, node.nesterov);
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const AdamWStepNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuResults = Detail::EvalAdamWStep(
			    GetValue(slots, node.parameter).CopyToDevice(CPU{}), GetValue(slots, node.gradient).CopyToDevice(CPU{}),
			    GetValue(slots, node.firstMoment).CopyToDevice(CPU{}),
			    GetValue(slots, node.secondMoment).CopyToDevice(CPU{}), node.learningRate, node.beta1, node.beta2,
			    node.epsilon, node.weightDecay, node.step);
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const Im2ColNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuResult = Detail::EvalIm2Col(GetValue(slots, node.input).CopyToDevice(CPU{}), node.kernelShape,
			                                    node.strides, node.dilations, node.lowPads, node.highPads);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const Conv2DNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuInput = GetValue(slots, node.input).CopyToDevice(CPU{});
			auto cpuWeight = GetValue(slots, node.weight).CopyToDevice(CPU{});
			auto cpuBiasStorage = node.bias
			                          ? std::optional<Tensor<CPU>>{ GetValue(slots, *node.bias).CopyToDevice(CPU{}) }
			                          : std::nullopt;
			auto cpuResult =
			    Detail::EvalConv2D(cpuInput, cpuWeight, cpuBiasStorage ? &*cpuBiasStorage : nullptr, node.strides,
			                       node.dilations, node.lowPads, node.highPads, node.groupCount);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId,
		             const ConvTranspose2DNode& node, std::vector<std::vector<Tensor<D>>>& slots,
		             std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuInput = GetValue(slots, node.input).CopyToDevice(CPU{});
			auto cpuWeight = GetValue(slots, node.weight).CopyToDevice(CPU{});
			auto cpuBiasStorage = node.bias
			                          ? std::optional<Tensor<CPU>>{ GetValue(slots, *node.bias).CopyToDevice(CPU{}) }
			                          : std::nullopt;
			auto cpuResult = Detail::EvalConvTranspose2D(
			    cpuInput, cpuWeight, cpuBiasStorage ? &*cpuBiasStorage : nullptr, node.strides, node.dilations,
			    node.lowPads, node.highPads, node.outputPads, node.groupCount);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const Pool2DNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuResult =
			    Detail::EvalPool2D(GetValue(slots, node.input).CopyToDevice(CPU{}), node.mode, node.kernelShape,
			                       node.strides, node.lowPads, node.highPads, node.countIncludePad);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const UpsampleNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			auto cpuResult = Detail::EvalUpsample(GetValue(slots, node.input).CopyToDevice(CPU{}), node.mode,
			                                      node.outputSpatialShape, node.alignCorners);
			if constexpr (std::same_as<D, CPU>)
			{
				slots[nodeId].push_back(std::move(cpuResult));
			}
			else
			{
				slots[nodeId].push_back(cpuResult.CopyToDevice(device));
			}
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const ConcatNode& node,
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
				srcPtrs.push_back(t.UnsafeRawData());
				srcShapes.push_back(t.Shape());
			}

			Tensor<D> result(Uninitialized, outputInfo.shape, outputInfo.dtype, device);
			DeviceTraits<D>::DoConcatOp(device, result.UnsafeRawData(), outputInfo.dtype, srcPtrs.data(),
			                            srcShapes.data(), srcPtrs.size(), node.axis);
			slots[nodeId].push_back(std::move(result));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const SliceNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& input = GetValue(slots, node.input);
			const auto& outputInfo = entry.outputInfos[0];

			Tensor<D> result(Uninitialized, outputInfo.shape, outputInfo.dtype, device);
			DeviceTraits<D>::DoSliceOp(device, result.UnsafeRawData(), outputInfo.dtype, input.Shape(),
			                           input.UnsafeRawData(), node.axis, node.start, node.length);
			slots[nodeId].push_back(std::move(result));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const GetRowsNode& node,
		             std::vector<std::vector<Tensor<D>>>& slots, std::span<const Tensor<D>> inputs, D& device)
		{
			const auto& data = GetValue(slots, node.data);
			const auto& indices = GetValue(slots, node.indices);
			const auto& outputInfo = entry.outputInfos[0];

			Tensor<D> result(Uninitialized, outputInfo.shape, outputInfo.dtype, device);
			DeviceTraits<D>::DoGetRowsOp(device, result.UnsafeRawData(), data.DType(), data.Shape(),
			                             data.UnsafeRawData(), indices.DType(), indices.Shape(),
			                             indices.UnsafeRawData());
			slots[nodeId].push_back(std::move(result));
		}

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const ArgsortNode& node,
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const MulMatIdNode& node,
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

		template <typename ExecutionModel>
		void Execute(const ExecutionModel& graph, const NodeEntry& entry, NodeId nodeId, const FusedOpNode& node,
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
		TraceCallback traceCallback_;
		QuantizedMatMulCallback quantizedMatMulCallback_;
	};
} // namespace LiteNN::Runtime

#endif
