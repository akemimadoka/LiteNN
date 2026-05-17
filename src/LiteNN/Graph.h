#include <LiteNN/Device.h>
#include <LiteNN/Metadata.h>
#include <LiteNN/Quantization.h>
#include <LiteNN/Tensor.h>
#include <deque>
#include <memory>
#include <meta>
#include <optional>
#include <ranges>
#include <string>
#include <string_view>

#ifndef LITENN_GRAPH_H
#define LITENN_GRAPH_H

namespace LiteNN
{
	// 基础 ID 类型
	using NodeId = std::size_t;
	using SubgraphId = std::size_t;

	// 引用某个节点的某个输出
	struct NodeOutput
	{
		NodeId node;
		std::size_t port; // 该节点的第几个输出
	};

	// 节点输出的类型和形状元信息
	struct OutputInfo
	{
		DataType dtype;
		std::vector<std::size_t> shape;
	};

	struct TensorSpec
	{
		DataType dtype;
		std::vector<std::size_t> shape;
	};

	struct NamedTensorSpec
	{
		std::string name;
		DataType dtype;
		std::vector<std::size_t> shape;
	};

	// 节点类型
	inline namespace Node
	{
		// 引用当前 Subgraph 的第 paramIndex 个参数
		struct ParamRefNode
		{
			std::size_t paramIndex;
		};

		// 常量张量
		struct ConstantNode
		{
			Tensor<PolymorphicDevice> value;
		};

		// 量化常量载荷：输出原始 storage tensor，同时携带其量化元数据。
		// 对于 block 格式，storage 通常是 UInt8 raw byte payload，逻辑 shape 在 params.expressedShape 中。
		struct QuantizedConstantNode
		{
			Tensor<PolymorphicDevice> storage;
			QuantizationParams params;
		};

		// 引用 Graph 的第 variableIndex 个可训练参数
		struct VariableRefNode
		{
			std::size_t variableIndex;
		};

		// 原语一元操作
		struct UnaryOpNode
		{
			UnaryOp op;
			NodeOutput input;
		};

		// 原语二元操作
		struct BinaryOpNode
		{
			BinaryOp op;
			NodeOutput lhs;
			NodeOutput rhs;
		};

		// 调用另一个 Subgraph，用于复合操作
		struct CallNode
		{
			SubgraphId callee;
			std::vector<NodeOutput> args;
		};

		// 类型转换
		struct CastNode
		{
			NodeOutput input;
			DataType targetType;
		};

		// 将浮点张量量化为 storage tensor。目前执行路径支持 scalar affine int8/uint8。
		struct QuantizeNode
		{
			NodeOutput input;
			QuantizationParams params;
		};

		// 将量化 storage tensor 还原到浮点张量。目前执行路径支持 scalar affine int8/uint8。
		struct DequantizeNode
		{
			NodeOutput input;
			QuantizationParams params;
			DataType targetType{ DataType::Float32 };
		};

		// 条件节点：根据 condition 选择执行 thenBranch 或 elseBranch
		// 两个分支的子图必须有相同数量和类型的输出
		// condition 必须是标量 Bool
		struct CondNode
		{
			NodeOutput condition;
			SubgraphId thenBranch;
			SubgraphId elseBranch;
			std::vector<NodeOutput> args; // 传入两个分支的参数
		};

		// 函数式循环：while condBranch(carry) do carry = bodyBranch(carry)
		// condBranch: params 与 initArgs 类型/形状一致，返回单个 Bool 标量
		// bodyBranch: params 和 results 与 initArgs 类型/形状一致
		// 输出 = 最终 carry values（多输出节点，outputInfos.size() == initArgs.size()）
		struct WhileNode
		{
			SubgraphId condBranch;
			SubgraphId bodyBranch;
			std::vector<NodeOutput> initArgs;
		};

		// 保存激活值到 ActivationStore（前向使用）
		// 透传输入值，不阻断数据流（输出 = 输入）
		struct SaveActivationNode
		{
			NodeOutput input;
			std::size_t slotId;
		};

		// 从 ActivationStore 加载激活值（反向使用）
		struct LoadActivationNode
		{
			std::size_t slotId;
		};

		// 栈式保存激活值（循环体前向使用）
		// 每次执行 push 一个值到 TapeStore[tapeSlotId]
		// 透传输入值，不阻断数据流（输出 = 输入）
		struct TapeSaveActivationNode
		{
			NodeOutput input;
			std::size_t tapeSlotId;
		};

		// 栈式加载激活值（循环体反向使用）
		// 每次执行 pop 最近的值（LIFO）
		struct TapeLoadActivationNode
		{
			std::size_t tapeSlotId;
		};

		// 归约操作（沿指定轴）
		struct ReduceOpNode
		{
			ReduceOp op;
			NodeOutput input;
			std::size_t axis;
		};

		// 改变形状（不改变数据，元素数量必须相同）
		struct ReshapeNode
		{
			NodeOutput input;
			std::vector<std::size_t> targetShape;
		};

		// 按 permutation 重排 axes（多维转置）
		// 约定：output.shape[d] == input.shape[permutation[d]]
		// permutation 必须是 [0, rank) 的合法置换
		struct PermuteNode
		{
			NodeOutput input;
			std::vector<std::size_t> permutation;
		};

		// 显式广播到 targetShape，按 trailing dimensions 对齐。
		// 输入维度必须等于目标维度或为 1；target 可插入前导维。
		struct BroadcastToNode
		{
			NodeOutput input;
			std::vector<std::size_t> targetShape;
		};

		// 任意轴 padding。lowPads/highPads 长度必须等于输入 rank。
		struct PadNode
		{
			NodeOutput input;
			std::vector<std::size_t> lowPads;
			std::vector<std::size_t> highPads;
			PadMode mode{ PadMode::Constant };
			double constantValue{ 0.0 };
		};

		// 沿 axis 做 gather，输出 shape = data[:axis] + indices.shape + data[axis+1:].
		struct GatherNode
		{
			NodeOutput data;
			NodeOutput indices;
			std::size_t axis;
		};

		// 沿 axis 写回 updates，输出 shape 与 data 相同。
		// updates.shape 必须等于 data[:axis] + indices.shape + data[axis+1:]。
		struct ScatterNode
		{
			NodeOutput data;
			NodeOutput indices;
			NodeOutput updates;
			std::size_t axis;
			ScatterMode mode{ ScatterMode::Update };
		};

		// Inclusive scan along an axis. Sum/max are the primary G5.2 targets;
		// prod/logsumexp are kept as explicit interface values for future lowering.
		struct ScanNode
		{
			NodeOutput input;
			std::size_t axis;
			ScanOp op{ ScanOp::Sum };
		};

		// Reference selective scan primitive for Mamba-style state-space layers.
		// state, dt, a, b, c, and optional d must broadcast to state.shape.
		struct SSMScanNode
		{
			NodeOutput state;
			NodeOutput dt;
			NodeOutput a;
			NodeOutput b;
			NodeOutput c;
			std::optional<NodeOutput> d;
		};

		// RWKV-style weighted key/value recurrence. Inputs are key/value/receptance
		// sequences plus time-decay/time-first parameters broadcast over channels.
		struct RWKVWKVNode
		{
			NodeOutput key;
			NodeOutput value;
			NodeOutput receptance;
			NodeOutput timeDecay;
			NodeOutput timeFirst;
		};

		// Numerically stable softmax along one axis.
		struct SoftmaxNode
		{
			NodeOutput input;
			std::size_t axis;
		};

		// Hot-path normalization primitive. scale/bias are optional and broadcast
		// to input.shape. GroupNorm keeps the existing ggml-oriented layout rule:
		// rank-4 tensors use the last axis as batch and normalize per batch.
		struct NormalizationNode
		{
			NodeOutput input;
			std::optional<NodeOutput> scale;
			std::optional<NodeOutput> bias;
			NormalizationMode mode{ NormalizationMode::LayerNorm };
			std::size_t axis;
			std::size_t groupCount{ 1 };
			double epsilon{ 1e-5 };
		};

		// Batch matmul with NumPy-style broadcasting on leading dimensions.
		// lhs shape [..., M, K], rhs shape [..., K, N] -> [..., M, N].
		struct BatchMatMulNode
		{
			NodeOutput lhs;
			NodeOutput rhs;
		};

		// 沿指定轴拼接多个张量
		// 所有输入除 axis 维度外 shape 必须相同
		struct ConcatNode
		{
			std::vector<NodeOutput> inputs;
			std::size_t axis;
		};

		// 沿指定轴提取连续切片
		struct SliceNode
		{
			NodeOutput input;
			std::size_t axis;
			std::size_t start;  // 起始索引（含）
			std::size_t length; // 切片长度
		};

		// 按第一维做行查找，输出 shape = indices.shape + data.shape[1:]
		struct GetRowsNode
		{
			NodeOutput data;
			NodeOutput indices;
		};

		// 沿指定轴对每个剩余位置独立排序，输出对应的 Int32 索引
		struct ArgsortNode
		{
			NodeOutput input;
			std::size_t axis = 0;
			SortOrder order;
		};

		// ggml-compatible MoE helper: 多 expert 矩阵中按 ids 选择对应矩阵，与每个 token 的输入向量做矩阵乘。
		// 该节点保留 ggml MUL_MAT_ID 的维度约定，并使用 Float32 累加/输出。
		struct MulMatIdNode
		{
			NodeOutput as;
			NodeOutput b;
			NodeOutput ids;
		};

		// 融合操作：语义等价于执行 body 子图，
		// 但向后端发出信号以生成优化的融合内核
		// body 子图参数与 args 按序 1:1 对应
		struct FusedOpNode
		{
			FusionPattern pattern;
			SubgraphId body;
			std::vector<NodeOutput> args;
		};
	} // namespace Node

	consteval std::meta::info CreateVariantFromMemberTypes(std::meta::info info)
	{
		return std::meta::substitute(
		    ^^std::variant, std::meta::members_of(info, std::meta::access_context::current()) |
		                        std::views::filter([](std::meta::info member) { return std::meta::is_type(member); }));
	}

	using NodeVariant = [:CreateVariantFromMemberTypes(^^Node):];

	// 节点及其输出元信息
	struct NodeEntry
	{
		NodeVariant node;
		std::vector<OutputInfo> outputInfos;
	};

	// 可训练参数，持有数据和梯度
	class Variable
	{
	public:
		Variable(const Variable&) = delete;
		Variable& operator=(const Variable&) = delete;

		template <Device D>
		static std::shared_ptr<Variable> Create(Tensor<D> data)
		{
			return std::shared_ptr<Variable>(new Variable(std::move(data)));
		}

		template <Device D>
		static std::shared_ptr<Variable> CreateQuantized(Tensor<D> storage, QuantizationParams quantization)
		{
			if constexpr (std::same_as<D, PolymorphicDevice>)
			{
				return std::shared_ptr<Variable>(new Variable(std::move(storage), std::move(quantization)));
			}
			else
			{
				return std::shared_ptr<Variable>(
				    new Variable(storage.CopyToDevice(PolymorphicDevice{ storage.CurDevice() }), std::move(quantization)));
			}
		}

		auto& Data(this auto&& self)
		{
			return self.data_;
		}

		auto& Grad(this auto&& self)
		{
			return self.grad_;
		}

		bool IsQuantized() const
		{
			return quantization_.has_value();
		}

		const std::optional<QuantizationParams>& Quantization() const
		{
			return quantization_;
		}

		void SetQuantization(std::optional<QuantizationParams> quantization)
		{
			quantization_ = std::move(quantization);
		}

	private:
		Tensor<PolymorphicDevice> data_;
		Tensor<PolymorphicDevice> grad_;
		std::optional<QuantizationParams> quantization_;

		Variable(Tensor<PolymorphicDevice> data)
		    : Variable(std::move(data), std::nullopt)
		{
		}

		Variable(Tensor<PolymorphicDevice> data, std::optional<QuantizationParams> quantization)
		    : data_(std::move(data)), grad_(data_.Shape(), data_.DType(), data_.CurDevice()),
		      quantization_(std::move(quantization))
		{
		}

		template <Device D>
		Variable(Tensor<D> data) : Variable(data.CopyToDevice(PolymorphicDevice{ data.CurDevice() }))
		{
		}

	};

	// 前向/反向共享的激活值存储槽位
	struct ActivationSlot
	{
		DataType dtype;
		std::vector<std::size_t> shape;
	};

	// 循环体的栈式激活值槽位
	struct TapeSlot
	{
		DataType dtype;
		std::vector<std::size_t> shape;
	};

	struct SubgraphParam
	{
		DataType dtype;
		std::vector<std::size_t> shape;
	};

	class Graph;

	// 一个计算块，有明确的参数和返回值
	class Subgraph
	{
	public:
		// 添加一个输入参数声明，创建 ParamRefNode，返回对应的 NodeId
		// NOTE: 可能导致已有的 GetNodeEntry/GetOutputInfo/Nodes()/Params() 返回的引用或 span 失效
		NodeId AddParam(DataType dtype, std::vector<std::size_t> shape)
		{
			const auto paramIndex = params_.size();
			params_.push_back({ dtype, std::move(shape) });

			const auto nodeId = nodes_.size();
			nodes_.push_back(
			    { ParamRefNode{ paramIndex }, { OutputInfo{ params_.back().dtype, params_.back().shape } } });
			return nodeId;
		}

		// 添加一个节点，返回 NodeId
		// NOTE: 可能导致已有的 GetNodeEntry/GetOutputInfo/Nodes() 返回的引用或 span 失效
		NodeId AddNode(NodeVariant node, std::vector<OutputInfo> outputInfos)
		{
			const auto nodeId = nodes_.size();
			nodes_.push_back({ std::move(node), std::move(outputInfos) });
			return nodeId;
		}

		// 设置子图的输出
		void SetResults(std::vector<NodeOutput> results)
		{
			results_ = std::move(results);
		}

		std::span<const SubgraphParam> Params() const
		{
			return params_;
		}

		std::span<const NodeOutput> Results() const
		{
			return results_;
		}

		std::span<const NodeEntry> Nodes() const
		{
			return nodes_;
		}

		auto& GetNodeEntry(this auto&& self, NodeId id)
		{
			return self.nodes_[id];
		}

		const OutputInfo& GetOutputInfo(NodeOutput output) const
		{
			return nodes_[output.node].outputInfos[output.port];
		}

		std::size_t NodeCount() const
		{
			return nodes_.size();
		}

	private:
		std::vector<SubgraphParam> params_;
		std::vector<NodeOutput> results_;
		std::vector<NodeEntry> nodes_; // arena, 拓扑序
	};

	// 表示 Subgraph 的集合，封闭世界，对外只暴露 forward/backward 入口
	// 因此，Graph 内部的 Subgraph 之间可以相互引用（通过 SubgraphId），但外部无法直接访问 Subgraph
	// 内部结构，从而可以进行更激进的跨过程优化
	class Graph
	{
	public:
		// 添加一个子图，返回 SubgraphId
		// 不会导致已有的 Subgraph 的引用失效
		SubgraphId AddSubgraph(Subgraph subgraph)
		{
			const auto id = subgraphs_.size();
			subgraphs_.push_back(std::move(subgraph));
			return id;
		}

		// 添加一个可训练参数，返回索引
		// NOTE: 可能导致已有的 GetVariable 返回的引用失效，如需在添加后继续使用，应重新获取
		std::size_t AddVariable(std::shared_ptr<Variable> variable)
		{
			const auto index = variables_.size();
			variables_.push_back(std::move(variable));
			return index;
		}

		// 添加一个激活值槽位，返回 slotId
		// NOTE: 可能导致已有的 GetActivationSlot 返回的引用失效，如需在添加后继续使用，应重新获取
		std::size_t AddActivationSlot(ActivationSlot slot)
		{
			const auto id = activationSlots_.size();
			activationSlots_.push_back(std::move(slot));
			return id;
		}

		void SetForward(SubgraphId id)
		{
			forward_ = id;
		}

		void SetBackward(SubgraphId id)
		{
			backward_ = id;
		}

		void SetInputNames(std::vector<std::string> names)
		{
			inputNames_ = std::move(names);
		}

		void SetVariableNames(std::vector<std::string> names)
		{
			variableNames_ = std::move(names);
		}

		void SetOutputNames(std::vector<std::string> names)
		{
			outputNames_ = std::move(names);
		}

		void SetInputName(std::size_t index, std::string name)
		{
			if (inputNames_.size() <= index)
			{
				inputNames_.resize(index + 1);
			}
			inputNames_[index] = std::move(name);
		}

		void SetVariableName(std::size_t index, std::string name)
		{
			if (variableNames_.size() <= index)
			{
				variableNames_.resize(index + 1);
			}
			variableNames_[index] = std::move(name);
		}

		void SetOutputName(std::size_t index, std::string name)
		{
			if (outputNames_.size() <= index)
			{
				outputNames_.resize(index + 1);
			}
			outputNames_[index] = std::move(name);
		}

		void SetMetadata(std::vector<ModelMetadataEntry> metadata)
		{
			metadata_ = std::move(metadata);
		}

		void SetMetadataEntry(std::string key, ModelMetadataValue value)
		{
			for (auto& entry : metadata_)
			{
				if (entry.key == key)
				{
					entry.value = std::move(value);
					return;
				}
			}
			metadata_.push_back({ std::move(key), std::move(value) });
		}

		SubgraphId Forward() const
		{
			return forward_;
		}

		std::optional<SubgraphId> Backward() const
		{
			return backward_;
		}

		auto& GetSubgraph(this auto&& self, SubgraphId id)
		{
			return self.subgraphs_[id];
		}

		std::span<const std::string> InputNames() const
		{
			return inputNames_;
		}

		std::span<const std::string> VariableNames() const
		{
			return variableNames_;
		}

		std::span<const std::string> OutputNames() const
		{
			return outputNames_;
		}

		std::string InputName(std::size_t index) const
		{
			return NameOrDefault(inputNames_, index, "input");
		}

		std::string VariableName(std::size_t index) const
		{
			return NameOrDefault(variableNames_, index, "variable");
		}

		std::string OutputName(std::size_t index) const
		{
			return NameOrDefault(outputNames_, index, "output");
		}

		std::optional<std::size_t> FindInput(std::string_view name) const
		{
			return FindName(inputNames_, name);
		}

		std::optional<std::size_t> FindVariable(std::string_view name) const
		{
			return FindName(variableNames_, name);
		}

		std::optional<std::size_t> FindOutput(std::string_view name) const
		{
			return FindName(outputNames_, name);
		}

		const ModelMetadataEntry* FindMetadata(std::string_view key) const
		{
			for (const auto& entry : metadata_)
			{
				if (entry.key == key)
				{
					return &entry;
				}
			}
			return nullptr;
		}

		std::span<const ModelMetadataEntry> Metadata() const
		{
			return metadata_;
		}

		TensorSpec InputSpec(std::size_t index) const
		{
			const auto& param = GetSubgraph(Forward()).Params()[index];
			return { param.dtype, param.shape };
		}

		TensorSpec OutputSpec(std::size_t index) const
		{
			const auto& forward = GetSubgraph(Forward());
			const auto& info = forward.GetOutputInfo(forward.Results()[index]);
			return { info.dtype, info.shape };
		}

		std::vector<NamedTensorSpec> InputSignature() const
		{
			const auto& params = GetSubgraph(Forward()).Params();
			std::vector<NamedTensorSpec> signature;
			signature.reserve(params.size());
			for (std::size_t i = 0; i < params.size(); ++i)
			{
				signature.push_back({ InputName(i), params[i].dtype, params[i].shape });
			}
			return signature;
		}

		std::vector<NamedTensorSpec> OutputSignature() const
		{
			const auto& forward = GetSubgraph(Forward());
			std::vector<NamedTensorSpec> signature;
			signature.reserve(forward.Results().size());
			for (std::size_t i = 0; i < forward.Results().size(); ++i)
			{
				const auto& info = forward.GetOutputInfo(forward.Results()[i]);
				signature.push_back({ OutputName(i), info.dtype, info.shape });
			}
			return signature;
		}

		auto Subgraphs() const
		{
			return std::views::all(subgraphs_);
		}

		const std::shared_ptr<Variable>& GetVariable(std::size_t index) const
		{
			return variables_[index];
		}

		std::span<const std::shared_ptr<Variable>> Variables() const
		{
			return variables_;
		}

		std::size_t SubgraphCount() const
		{
			return subgraphs_.size();
		}

		std::size_t VariableCount() const
		{
			return variables_.size();
		}

		const ActivationSlot& GetActivationSlot(std::size_t id) const
		{
			return activationSlots_[id];
		}

		std::size_t ActivationSlotCount() const
		{
			return activationSlots_.size();
		}

		std::size_t AddTapeSlot(TapeSlot slot)
		{
			const auto id = tapeSlots_.size();
			tapeSlots_.push_back(std::move(slot));
			return id;
		}

		const TapeSlot& GetTapeSlot(std::size_t id) const
		{
			return tapeSlots_[id];
		}

		std::size_t TapeSlotCount() const
		{
			return tapeSlots_.size();
		}

	private:
		static std::string NameOrDefault(const std::vector<std::string>& names, std::size_t index,
		                                 std::string_view prefix)
		{
			if (index < names.size() && !names[index].empty())
			{
				return names[index];
			}
			std::string fallback(prefix);
			fallback += std::to_string(index);
			return fallback;
		}

		static std::optional<std::size_t> FindName(const std::vector<std::string>& names, std::string_view name)
		{
			for (std::size_t i = 0; i < names.size(); ++i)
			{
				if (names[i] == name)
				{
					return i;
				}
			}
			return std::nullopt;
		}

		SubgraphId forward_{};
		std::optional<SubgraphId> backward_;
		std::deque<Subgraph> subgraphs_;
		std::vector<std::shared_ptr<Variable>> variables_;
		std::vector<ActivationSlot> activationSlots_;
		std::vector<TapeSlot> tapeSlots_;
		std::vector<std::string> inputNames_;
		std::vector<std::string> variableNames_;
		std::vector<std::string> outputNames_;
		std::vector<ModelMetadataEntry> metadata_;
	};
} // namespace LiteNN

#endif
