#ifndef LITENN_GRAPH_H
#define LITENN_GRAPH_H

#include "meta"
#include <LiteNN/Device.h>
#include <LiteNN/Tensor.h>
#include <deque>
#include <memory>
#include <optional>
#include <ranges>

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

		auto& Data(this auto&& self)
		{
			return self.data_;
		}

		auto& Grad(this auto&& self)
		{
			return self.grad_;
		}

	private:
		Tensor<PolymorphicDevice> data_;
		Tensor<PolymorphicDevice> grad_;

		Variable(Tensor<PolymorphicDevice> data)
		    : data_(std::move(data)), grad_(data_.Shape(), data_.DType(), data_.CurDevice())
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
		SubgraphId forward_{};
		std::optional<SubgraphId> backward_;
		std::deque<Subgraph> subgraphs_;
		std::vector<std::shared_ptr<Variable>> variables_;
		std::vector<ActivationSlot> activationSlots_;
		std::vector<TapeSlot> tapeSlots_;
	};
} // namespace LiteNN

#endif
