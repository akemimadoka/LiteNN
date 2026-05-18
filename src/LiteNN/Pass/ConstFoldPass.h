#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/DataMovement.h>
#include <LiteNN/Graph.h>
#include <LiteNN/Pass.h>
#include <LiteNN/Validation/GraphValidator.h>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <vector>

#ifndef LITENN_PASS_CONSTFOLD_H
#define LITENN_PASS_CONSTFOLD_H

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
			Validation::ValidateGraph(graph);
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

		static Tensor<CPU> EvalQuantize(const Tensor<CPU>& input, const QuantizationParams& params)
		{
			if (params.scheme != QuantizationScheme::Affine)
			{
				throw std::runtime_error("ConstFoldPass only folds affine QuantizeNode");
			}
			auto quantized = QuantizeAffine(input, params);
			return std::move(quantized.Storage());
		}

		static Tensor<CPU> EvalDequantize(const Tensor<CPU>& input, const QuantizationParams& params,
		                                  DataType targetType)
		{
			if (params.scheme != QuantizationScheme::Affine)
			{
				throw std::runtime_error("ConstFoldPass only folds affine DequantizeNode");
			}
			return DequantizeAffine(input, params, targetType);
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

		static Tensor<CPU> EvalPermute(const Tensor<CPU>& input, const std::vector<std::size_t>& permutation,
		                              const OutputInfo& outInfo)
		{
			CPU device;
			Tensor<CPU> result(Uninitialized, outInfo.shape, outInfo.dtype, device);
			DeviceTraits<CPU>::DoPermuteOp(device, result.RawData(), input.DType(), input.Shape(), input.RawData(),
			                              ShapeView{ permutation });
			return result;
		}

		static Tensor<CPU> EvalBroadcastTo(const Tensor<CPU>& input, const BroadcastToNode& node)
		{
			return Detail::EvalBroadcastTo(input, node.targetShape);
		}

		static Tensor<CPU> EvalPad(const Tensor<CPU>& input, const PadNode& node)
		{
			return Detail::EvalPad(input, node.lowPads, node.highPads, node.mode, node.constantValue);
		}

		static Tensor<CPU> EvalGather(const Tensor<CPU>& data, const Tensor<CPU>& indices, const GatherNode& node)
		{
			return Detail::EvalGather(data, indices, node.axis);
		}

		static Tensor<CPU> EvalScatter(const Tensor<CPU>& data, const Tensor<CPU>& indices, const Tensor<CPU>& updates,
		                              const ScatterNode& node)
		{
			return Detail::EvalScatter(data, indices, updates, node.axis, node.mode);
		}

		static Tensor<CPU> EvalScan(const Tensor<CPU>& input, const ScanNode& node)
		{
			return Detail::EvalScan(input, node.axis, node.op);
		}

		static Tensor<CPU> EvalSSMScan(const Tensor<CPU>& state, const Tensor<CPU>& dt, const Tensor<CPU>& a,
		                              const Tensor<CPU>& b, const Tensor<CPU>& c, const Tensor<CPU>* d)
		{
			return Detail::EvalSSMScan(state, dt, a, b, c, d);
		}

		static Tensor<CPU> EvalRWKVWKV(const Tensor<CPU>& key, const Tensor<CPU>& value,
		                              const Tensor<CPU>& receptance, const Tensor<CPU>& timeDecay,
		                              const Tensor<CPU>& timeFirst)
		{
			return Detail::EvalRWKVWKV(key, value, receptance, timeDecay, timeFirst);
		}

		static Tensor<CPU> EvalSoftmax(const Tensor<CPU>& input, const SoftmaxNode& node)
		{
			return Detail::EvalSoftmax(input, node.axis);
		}

		static Tensor<CPU> EvalNormalization(const Tensor<CPU>& input, const Tensor<CPU>* scale,
		                                    const Tensor<CPU>* bias, const NormalizationNode& node)
		{
			return Detail::EvalNormalization(input, scale, bias, node.mode, node.axis, node.groupCount,
			                                 node.epsilon);
		}

		static Tensor<CPU> EvalBatchMatMul(const Tensor<CPU>& lhs, const Tensor<CPU>& rhs)
		{
			return Detail::EvalBatchMatMul(lhs, rhs);
		}

		static Tensor<CPU> EvalIm2Col(const Tensor<CPU>& input, const Im2ColNode& node)
		{
			return Detail::EvalIm2Col(input, node.kernelShape, node.strides, node.dilations,
			                          node.lowPads, node.highPads);
		}

		static Tensor<CPU> EvalConv2D(const Tensor<CPU>& input, const Tensor<CPU>& weight,
		                              const Tensor<CPU>* bias, const Conv2DNode& node)
		{
			return Detail::EvalConv2D(input, weight, bias, node.strides, node.dilations,
			                          node.lowPads, node.highPads, node.groupCount);
		}

		static Tensor<CPU> EvalConvTranspose2D(const Tensor<CPU>& input, const Tensor<CPU>& weight,
		                                       const Tensor<CPU>* bias, const ConvTranspose2DNode& node)
		{
			return Detail::EvalConvTranspose2D(input, weight, bias, node.strides, node.dilations,
			                                   node.lowPads, node.highPads, node.outputPads, node.groupCount);
		}

		static Tensor<CPU> EvalPool2D(const Tensor<CPU>& input, const Pool2DNode& node)
		{
			return Detail::EvalPool2D(input, node.mode, node.kernelShape, node.strides,
			                          node.lowPads, node.highPads, node.countIncludePad);
		}

		static Tensor<CPU> EvalUpsample(const Tensor<CPU>& input, const UpsampleNode& node)
		{
			return Detail::EvalUpsample(input, node.mode, node.outputSpatialShape, node.alignCorners);
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

		static Tensor<CPU> EvalGetRows(const Tensor<CPU>& data, const Tensor<CPU>& indices, const OutputInfo& outInfo)
		{
			CPU device;
			Tensor<CPU> result(Uninitialized, outInfo.shape, outInfo.dtype, device);
			DeviceTraits<CPU>::DoGetRowsOp(device, result.RawData(), data.DType(), data.Shape(), data.RawData(),
			                              indices.DType(), indices.Shape(), indices.RawData());
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
				    else if constexpr (std::same_as<T, PermuteNode>)
				    {
					    return PermuteNode{ remap(n.input), n.permutation };
				    }
				    else if constexpr (std::same_as<T, BroadcastToNode>)
				    {
					    return BroadcastToNode{ remap(n.input), n.targetShape };
				    }
				    else if constexpr (std::same_as<T, PadNode>)
				    {
					    return PadNode{ remap(n.input), n.lowPads, n.highPads, n.mode, n.constantValue };
				    }
				    else if constexpr (std::same_as<T, GatherNode>)
				    {
					    return GatherNode{ remap(n.data), remap(n.indices), n.axis };
				    }
				    else if constexpr (std::same_as<T, ScatterNode>)
				    {
					    return ScatterNode{ remap(n.data), remap(n.indices), remap(n.updates), n.axis, n.mode };
				    }
				    else if constexpr (std::same_as<T, ScanNode>)
				    {
					    return ScanNode{ remap(n.input), n.axis, n.op };
				    }
				    else if constexpr (std::same_as<T, SSMScanNode>)
				    {
					    return SSMScanNode{ remap(n.state), remap(n.dt), remap(n.a), remap(n.b), remap(n.c),
					                        n.d ? std::optional<NodeOutput>{ remap(*n.d) } : std::nullopt };
				    }
				    else if constexpr (std::same_as<T, RWKVWKVNode>)
				    {
					    return RWKVWKVNode{ remap(n.key), remap(n.value), remap(n.receptance),
					                        remap(n.timeDecay), remap(n.timeFirst) };
				    }
				    else if constexpr (std::same_as<T, SoftmaxNode>)
				    {
					    return SoftmaxNode{ remap(n.input), n.axis };
				    }
				    else if constexpr (std::same_as<T, NormalizationNode>)
				    {
					    return NormalizationNode{ remap(n.input),
					                              n.scale ? std::optional<NodeOutput>{ remap(*n.scale) } : std::nullopt,
					                              n.bias ? std::optional<NodeOutput>{ remap(*n.bias) } : std::nullopt,
					                              n.mode, n.axis, n.groupCount, n.epsilon };
				    }
				    else if constexpr (std::same_as<T, BatchMatMulNode>)
				    {
					    return BatchMatMulNode{ remap(n.lhs), remap(n.rhs) };
				    }
				    else if constexpr (std::same_as<T, Im2ColNode>)
				    {
					    return Im2ColNode{ remap(n.input), n.kernelShape, n.strides, n.dilations,
					                       n.lowPads, n.highPads };
				    }
				    else if constexpr (std::same_as<T, Conv2DNode>)
				    {
					    return Conv2DNode{ remap(n.input), remap(n.weight),
					                       n.bias ? std::optional<NodeOutput>{ remap(*n.bias) } : std::nullopt,
					                       n.strides, n.dilations, n.lowPads, n.highPads, n.groupCount };
				    }
				    else if constexpr (std::same_as<T, ConvTranspose2DNode>)
				    {
					    return ConvTranspose2DNode{
					        remap(n.input),
					        remap(n.weight),
					        n.bias ? std::optional<NodeOutput>{ remap(*n.bias) } : std::nullopt,
					        n.strides,
					        n.dilations,
					        n.lowPads,
					        n.highPads,
					        n.outputPads,
					        n.groupCount
					    };
				    }
				    else if constexpr (std::same_as<T, Pool2DNode>)
				    {
					    return Pool2DNode{ remap(n.input), n.mode, n.kernelShape, n.strides,
					                       n.lowPads, n.highPads, n.countIncludePad };
				    }
				    else if constexpr (std::same_as<T, UpsampleNode>)
				    {
					    return UpsampleNode{ remap(n.input), n.mode, n.outputSpatialShape, n.alignCorners };
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
				    else if constexpr (std::same_as<T, QuantizeNode>)
				    {
					    markInput(node.input);
				    }
				    else if constexpr (std::same_as<T, DequantizeNode>)
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
				    else if constexpr (std::same_as<T, PermuteNode>)
				    {
					    markInput(node.input);
				    }
				    else if constexpr (std::same_as<T, BroadcastToNode>)
				    {
					    markInput(node.input);
				    }
				    else if constexpr (std::same_as<T, PadNode>)
				    {
					    markInput(node.input);
				    }
				    else if constexpr (std::same_as<T, GatherNode>)
				    {
					    markInput(node.data);
					    markInput(node.indices);
				    }
				    else if constexpr (std::same_as<T, ScatterNode>)
				    {
					    markInput(node.data);
					    markInput(node.indices);
					    markInput(node.updates);
				    }
				    else if constexpr (std::same_as<T, ScanNode>)
				    {
					    markInput(node.input);
				    }
				    else if constexpr (std::same_as<T, SSMScanNode>)
				    {
					    markInput(node.state);
					    markInput(node.dt);
					    markInput(node.a);
					    markInput(node.b);
					    markInput(node.c);
					    if (node.d)
					    {
						    markInput(*node.d);
					    }
				    }
				    else if constexpr (std::same_as<T, RWKVWKVNode>)
				    {
					    markInput(node.key);
					    markInput(node.value);
					    markInput(node.receptance);
					    markInput(node.timeDecay);
					    markInput(node.timeFirst);
				    }
				    else if constexpr (std::same_as<T, SoftmaxNode>)
				    {
					    markInput(node.input);
				    }
				    else if constexpr (std::same_as<T, NormalizationNode>)
				    {
					    markInput(node.input);
					    if (node.scale)
					    {
						    markInput(*node.scale);
					    }
					    if (node.bias)
					    {
						    markInput(*node.bias);
					    }
				    }
				    else if constexpr (std::same_as<T, BatchMatMulNode>)
				    {
					    markInput(node.lhs);
					    markInput(node.rhs);
				    }
				    else if constexpr (std::same_as<T, Im2ColNode>)
				    {
					    markInput(node.input);
				    }
				    else if constexpr (std::same_as<T, Conv2DNode>)
				    {
					    markInput(node.input);
					    markInput(node.weight);
					    if (node.bias)
					    {
						    markInput(*node.bias);
					    }
				    }
				    else if constexpr (std::same_as<T, ConvTranspose2DNode>)
				    {
					    markInput(node.input);
					    markInput(node.weight);
					    if (node.bias)
					    {
						    markInput(*node.bias);
					    }
				    }
				    else if constexpr (std::same_as<T, Pool2DNode>)
				    {
					    markInput(node.input);
				    }
				    else if constexpr (std::same_as<T, UpsampleNode>)
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
				    else if constexpr (std::same_as<T, GetRowsNode>)
				    {
					    markInput(node.data);
					    markInput(node.indices);
				    }
				    else if constexpr (std::same_as<T, ArgsortNode>)
				    {
					    markInput(node.input);
				    }
				    else if constexpr (std::same_as<T, MulMatIdNode>)
				    {
					    markInput(node.as);
					    markInput(node.b);
					    markInput(node.ids);
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
				    // ParamRefNode, ConstantNode, QuantizedConstantNode, VariableRefNode,
				    // LoadActivationNode, TapeLoadActivationNode: 无输入
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
					    else if constexpr (std::same_as<T, QuantizedConstantNode>)
					    {
						    isConst[nodeId] = true;
						    constValues[nodeId] = node.storage.CopyToDevice(CPU{});
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
					    else if constexpr (std::same_as<T, QuantizeNode>)
					    {
						    if (isConst[node.input.node] && node.params.scheme == QuantizationScheme::Affine)
						    {
							    isConst[nodeId] = true;
							    const auto& input = GetConstValue(constValues, node.input);
							    constValues[nodeId] = EvalQuantize(input, node.params);
						    }
					    }
					    else if constexpr (std::same_as<T, DequantizeNode>)
					    {
						    if (isConst[node.input.node] && node.params.scheme == QuantizationScheme::Affine)
						    {
							    isConst[nodeId] = true;
							    const auto& input = GetConstValue(constValues, node.input);
							    constValues[nodeId] = EvalDequantize(input, node.params, node.targetType);
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
					    else if constexpr (std::same_as<T, PermuteNode>)
					    {
						    if (isConst[node.input.node])
						    {
							    isConst[nodeId] = true;
							    const auto& input = GetConstValue(constValues, node.input);
							    constValues[nodeId] = EvalPermute(input, node.permutation, entry.outputInfos[0]);
						    }
					    }
					    else if constexpr (std::same_as<T, BroadcastToNode>)
					    {
						    if (isConst[node.input.node])
						    {
							    isConst[nodeId] = true;
							    const auto& input = GetConstValue(constValues, node.input);
							    constValues[nodeId] = EvalBroadcastTo(input, node);
						    }
					    }
					    else if constexpr (std::same_as<T, PadNode>)
					    {
						    if (isConst[node.input.node])
						    {
							    isConst[nodeId] = true;
							    const auto& input = GetConstValue(constValues, node.input);
							    constValues[nodeId] = EvalPad(input, node);
						    }
					    }
					    else if constexpr (std::same_as<T, GatherNode>)
					    {
						    if (isConst[node.data.node] && isConst[node.indices.node])
						    {
							    isConst[nodeId] = true;
							    const auto& data = GetConstValue(constValues, node.data);
							    const auto& indices = GetConstValue(constValues, node.indices);
							    constValues[nodeId] = EvalGather(data, indices, node);
						    }
					    }
					    else if constexpr (std::same_as<T, ScatterNode>)
					    {
						    if (isConst[node.data.node] && isConst[node.indices.node] && isConst[node.updates.node])
						    {
							    isConst[nodeId] = true;
							    const auto& data = GetConstValue(constValues, node.data);
							    const auto& indices = GetConstValue(constValues, node.indices);
							    const auto& updates = GetConstValue(constValues, node.updates);
							    constValues[nodeId] = EvalScatter(data, indices, updates, node);
						    }
					    }
					    else if constexpr (std::same_as<T, ScanNode>)
					    {
						    if (isConst[node.input.node])
						    {
							    isConst[nodeId] = true;
							    const auto& input = GetConstValue(constValues, node.input);
							    constValues[nodeId] = EvalScan(input, node);
						    }
					    }
					    else if constexpr (std::same_as<T, SSMScanNode>)
					    {
						    const auto dConst = !node.d || isConst[node.d->node];
						    if (isConst[node.state.node] && isConst[node.dt.node] && isConst[node.a.node] &&
						        isConst[node.b.node] && isConst[node.c.node] && dConst)
						    {
							    isConst[nodeId] = true;
							    const auto& state = GetConstValue(constValues, node.state);
							    const auto& dt = GetConstValue(constValues, node.dt);
							    const auto& a = GetConstValue(constValues, node.a);
							    const auto& b = GetConstValue(constValues, node.b);
							    const auto& c = GetConstValue(constValues, node.c);
							    const auto* d = node.d ? &GetConstValue(constValues, *node.d) : nullptr;
							    constValues[nodeId] = EvalSSMScan(state, dt, a, b, c, d);
						    }
					    }
					    else if constexpr (std::same_as<T, RWKVWKVNode>)
					    {
						    if (isConst[node.key.node] && isConst[node.value.node] && isConst[node.receptance.node] &&
						        isConst[node.timeDecay.node] && isConst[node.timeFirst.node])
						    {
							    isConst[nodeId] = true;
							    constValues[nodeId] =
							        EvalRWKVWKV(GetConstValue(constValues, node.key), GetConstValue(constValues, node.value),
							                    GetConstValue(constValues, node.receptance),
							                    GetConstValue(constValues, node.timeDecay),
							                    GetConstValue(constValues, node.timeFirst));
						    }
					    }
					    else if constexpr (std::same_as<T, SoftmaxNode>)
					    {
						    if (isConst[node.input.node])
						    {
							    isConst[nodeId] = true;
							    const auto& input = GetConstValue(constValues, node.input);
							    constValues[nodeId] = EvalSoftmax(input, node);
						    }
					    }
					    else if constexpr (std::same_as<T, NormalizationNode>)
					    {
						    const auto scaleConst = !node.scale || isConst[node.scale->node];
						    const auto biasConst = !node.bias || isConst[node.bias->node];
						    if (isConst[node.input.node] && scaleConst && biasConst)
						    {
							    isConst[nodeId] = true;
							    const auto& input = GetConstValue(constValues, node.input);
							    const auto* scale = node.scale ? &GetConstValue(constValues, *node.scale) : nullptr;
							    const auto* bias = node.bias ? &GetConstValue(constValues, *node.bias) : nullptr;
							    constValues[nodeId] = EvalNormalization(input, scale, bias, node);
						    }
					    }
					    else if constexpr (std::same_as<T, BatchMatMulNode>)
					    {
						    if (isConst[node.lhs.node] && isConst[node.rhs.node])
						    {
							    isConst[nodeId] = true;
							    const auto& lhs = GetConstValue(constValues, node.lhs);
							    const auto& rhs = GetConstValue(constValues, node.rhs);
							    constValues[nodeId] = EvalBatchMatMul(lhs, rhs);
						    }
					    }
					    else if constexpr (std::same_as<T, Im2ColNode>)
					    {
						    if (isConst[node.input.node])
						    {
							    isConst[nodeId] = true;
							    constValues[nodeId] = EvalIm2Col(GetConstValue(constValues, node.input), node);
						    }
					    }
					    else if constexpr (std::same_as<T, Conv2DNode>)
					    {
						    const auto allConst =
						        isConst[node.input.node] && isConst[node.weight.node] &&
						        (!node.bias || isConst[node.bias->node]);
						    if (allConst)
						    {
							    isConst[nodeId] = true;
							    const auto& input = GetConstValue(constValues, node.input);
							    const auto& weight = GetConstValue(constValues, node.weight);
							    const auto* bias =
							        node.bias ? &GetConstValue(constValues, *node.bias) : nullptr;
							    constValues[nodeId] = EvalConv2D(input, weight, bias, node);
						    }
					    }
					    else if constexpr (std::same_as<T, ConvTranspose2DNode>)
					    {
						    const auto allConst =
						        isConst[node.input.node] && isConst[node.weight.node] &&
						        (!node.bias || isConst[node.bias->node]);
						    if (allConst)
						    {
							    isConst[nodeId] = true;
							    const auto& input = GetConstValue(constValues, node.input);
							    const auto& weight = GetConstValue(constValues, node.weight);
							    const auto* bias =
							        node.bias ? &GetConstValue(constValues, *node.bias) : nullptr;
							    constValues[nodeId] = EvalConvTranspose2D(input, weight, bias, node);
						    }
					    }
					    else if constexpr (std::same_as<T, Pool2DNode>)
					    {
						    if (isConst[node.input.node])
						    {
							    isConst[nodeId] = true;
							    constValues[nodeId] = EvalPool2D(GetConstValue(constValues, node.input), node);
						    }
					    }
					    else if constexpr (std::same_as<T, UpsampleNode>)
					    {
						    if (isConst[node.input.node])
						    {
							    isConst[nodeId] = true;
							    constValues[nodeId] = EvalUpsample(GetConstValue(constValues, node.input), node);
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
					    else if constexpr (std::same_as<T, GetRowsNode>)
					    {
						    if (isConst[node.data.node] && isConst[node.indices.node])
						    {
							    isConst[nodeId] = true;
							    const auto& data = GetConstValue(constValues, node.data);
							    const auto& indices = GetConstValue(constValues, node.indices);
							    constValues[nodeId] = EvalGetRows(data, indices, entry.outputInfos[0]);
						    }
					    }
					    else if constexpr (std::same_as<T, ArgsortNode>)
					    {
						    // Argsort currently only runs through the interpreter execution path.
					    }
					    else if constexpr (std::same_as<T, MulMatIdNode>)
					    {
						    // MulMatId currently only runs through the interpreter execution path.
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
			for (NodeId oldId = 0; oldId < nodeCount; ++oldId)
			{
				if (const auto* paramRef = std::get_if<ParamRefNode>(&sg.GetNodeEntry(oldId).node))
				{
					const auto& param = sg.Params()[paramRef->paramIndex];
					nodeMap[oldId] = newSg.AddParam(param.dtype, param.shape);
				}
			}

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
				if (std::holds_alternative<ParamRefNode>(entry.node))
				{
					continue;
				}

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
