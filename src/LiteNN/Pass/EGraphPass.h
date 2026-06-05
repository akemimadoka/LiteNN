#include <LiteNN/Graph.h>
#include <LiteNN/Misc.h>
#include <LiteNN/Pass.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <chrono>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef LITENN_PASS_EGRAPH_H
#define LITENN_PASS_EGRAPH_H

namespace LiteNN
{
	struct EGraphOptions
	{
		std::size_t maxIterations{ 6 };
		std::size_t maxTerms{ 4096 };
		std::size_t maxEClasses{ 4096 };
		std::chrono::milliseconds timeout{ 100 };
		bool allowUnsafeFloatingRewrites{ false };
		bool enableCommutativeCanonicalization{ true };
	};

	struct EGraphRewriteEvent
	{
		SubgraphId subgraph{};
		NodeId node{};
		std::string rule;
		std::string before;
		std::string after;
	};

	struct EGraphReport
	{
		std::size_t importedTerms{};
		std::size_t eClasses{};
		std::size_t rewrites{};
		std::size_t iterations{};
		bool hitLimit{};
		std::vector<EGraphRewriteEvent> events;

		std::string Dump() const
		{
			std::ostringstream out;
			out << "LiteNN EGraph report\n";
			out << "  importedTerms: " << importedTerms << '\n';
			out << "  eClasses: " << eClasses << '\n';
			out << "  rewrites: " << rewrites << '\n';
			out << "  iterations: " << iterations << '\n';
			out << "  hitLimit: " << (hitLimit ? "true" : "false") << '\n';
			for (const auto& event : events)
			{
				out << "  sg" << event.subgraph << ":n" << event.node << " " << event.rule << ": "
				    << event.before << " -> " << event.after << '\n';
			}
			return out.str();
		}
	};

	namespace Detail
	{
		struct EGraphTerm
		{
			std::string key;
			std::vector<std::size_t> children;
			OutputInfo info;
		};

		class TinyEGraph
		{
		public:
			std::size_t AddTerm(EGraphTerm term)
			{
				if (const auto it = memo_.find(term.key); it != memo_.end())
				{
					return Find(termToClass_[it->second]);
				}
				const auto termId = terms_.size();
				const auto classId = parent_.size();
				memo_[term.key] = termId;
				terms_.push_back(std::move(term));
				termToClass_.push_back(classId);
				parent_.push_back(classId);
				return classId;
			}

			std::size_t Find(std::size_t id)
			{
				if (parent_[id] != id)
				{
					parent_[id] = Find(parent_[id]);
				}
				return parent_[id];
			}

			bool Union(std::size_t lhs, std::size_t rhs)
			{
				auto lhsRoot = Find(lhs);
				auto rhsRoot = Find(rhs);
				if (lhsRoot == rhsRoot)
				{
					return false;
				}
				if (rhsRoot < lhsRoot)
				{
					std::swap(lhsRoot, rhsRoot);
				}
				parent_[rhsRoot] = lhsRoot;
				return true;
			}

			std::size_t TermCount() const
			{
				return terms_.size();
			}

			std::size_t EClassCount()
			{
				std::set<std::size_t> roots;
				for (auto id = 0uz; id < parent_.size(); ++id)
				{
					roots.insert(Find(id));
				}
				return roots.size();
			}

		private:
			std::vector<EGraphTerm> terms_;
			std::vector<std::size_t> termToClass_;
			std::vector<std::size_t> parent_;
			std::map<std::string, std::size_t> memo_;
		};
	} // namespace Detail

	/**
	 * Equality-saturation inspired graph optimizer for conservative, pure tensor rewrites.
	 *
	 * The first tranche intentionally keeps the e-graph small: it imports pure single-output
	 * expressions, records equivalence classes for safe rewrites, and extracts a deterministic
	 * simplified graph. Stateful/runtime-only nodes remain opaque boundaries.
	 */
	class EGraphPass : public Migration::GraphMutationPass
	{
	public:
		explicit EGraphPass(EGraphOptions options = {}) : options_(std::move(options)) {}

		void Run(Graph& graph) override
		{
			Validation::ValidateGraph(graph);
			lastReport_ = {};
			const auto originalCount = graph.SubgraphCount();
			for (std::size_t sgId = 0; sgId < originalCount; ++sgId)
			{
				ProcessSubgraph(graph, sgId);
			}
			Validation::ValidateGraph(graph);
		}

		const EGraphReport& LastReport() const
		{
			return lastReport_;
		}

		std::string DumpLastReport() const
		{
			return lastReport_.Dump();
		}

	private:
		using NodeOutputKey = std::pair<NodeId, std::size_t>;

		static bool SameInfo(const OutputInfo& lhs, const OutputInfo& rhs)
		{
			return lhs.dtype == rhs.dtype && ShapeView{ lhs.shape } == ShapeView{ rhs.shape };
		}

		static bool SameInfo(const OutputInfo& lhs, const TensorSpec& rhs)
		{
			return lhs.dtype == rhs.dtype && ShapeView{ lhs.shape } == ShapeView{ rhs.shape };
		}

		static bool SameInfo(const OutputInfo& lhs, ShapeView rhsShape, DataType rhsDType)
		{
			return lhs.dtype == rhsDType && ShapeView{ lhs.shape } == rhsShape;
		}

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

		static bool IsIdentityPermutation(std::span<const std::size_t> permutation)
		{
			for (auto i = 0uz; i < permutation.size(); ++i)
			{
				if (permutation[i] != i)
				{
					return false;
				}
			}
			return true;
		}

		static bool CanBroadcastTrailing(ShapeView inputShape, ShapeView targetShape)
		{
			if (inputShape.NumDim() > targetShape.NumDim())
			{
				return false;
			}
			const auto offset = targetShape.NumDim() - inputShape.NumDim();
			for (auto dim = 0uz; dim < inputShape.NumDim(); ++dim)
			{
				const auto inputDim = inputShape[dim];
				const auto targetDim = targetShape[offset + dim];
				if (inputDim != targetDim && inputDim != 1)
				{
					return false;
				}
			}
			return true;
		}

		static std::vector<std::size_t> ComposePermutation(std::span<const std::size_t> first,
		                                                   std::span<const std::size_t> second)
		{
			std::vector<std::size_t> combined;
			combined.reserve(second.size());
			for (const auto axis : second)
			{
				combined.push_back(first[axis]);
			}
			return combined;
		}

		static std::string ShapeKey(std::span<const std::size_t> shape)
		{
			std::ostringstream out;
			out << '[';
			for (auto i = 0uz; i < shape.size(); ++i)
			{
				if (i != 0)
				{
					out << ',';
				}
				out << shape[i];
			}
			out << ']';
			return out.str();
		}

		static std::string InfoKey(const OutputInfo& info)
		{
			std::ostringstream out;
			out << EnumToString<EnumToStringStyle::Unqualified>(info.dtype) << ShapeKey(info.shape);
			return out.str();
		}

		static std::string OutputKey(NodeOutput out)
		{
			std::ostringstream text;
			text << "n" << out.node << ":" << out.port;
			return text.str();
		}

		static std::string NodeKey(NodeId node)
		{
			std::ostringstream text;
			text << "n" << node;
			return text.str();
		}

		static bool IsCommutative(BinaryOp op)
		{
			switch (op)
			{
			case BinaryOp::Add:
			case BinaryOp::Multiply:
			case BinaryOp::Max:
			case BinaryOp::Min:
			case BinaryOp::Equal:
				return true;
			default:
				return false;
			}
		}

		bool CanCanonicalizeCommutative(BinaryOp op, DataType dtype) const
		{
			if (!options_.enableCommutativeCanonicalization || !IsCommutative(op))
			{
				return false;
			}
			return options_.allowUnsafeFloatingRewrites || !IsFloatingDataType(dtype) || op == BinaryOp::Equal;
		}

		template <typename Fn>
		static void ForEachInput(const NodeVariant& node, Fn&& fn)
		{
			std::visit(
			    [&](const auto& n) {
				    using T = std::decay_t<decltype(n)>;
				    if constexpr (std::same_as<T, UnaryOpNode> || std::same_as<T, CastNode> ||
				                  std::same_as<T, QuantizeNode> || std::same_as<T, DequantizeNode> ||
				                  std::same_as<T, ReduceOpNode> || std::same_as<T, ReshapeNode> ||
				                  std::same_as<T, PermuteNode> || std::same_as<T, BroadcastToNode> ||
				                  std::same_as<T, PadNode> || std::same_as<T, ScanNode> ||
				                  std::same_as<T, Im2ColNode> ||
				                  std::same_as<T, Pool2DNode> || std::same_as<T, UpsampleNode> ||
				                  std::same_as<T, ArgsortNode> || std::same_as<T, SaveActivationNode> ||
				                  std::same_as<T, TapeSaveActivationNode>)
				    {
					    fn(n.input);
				    }
				    else if constexpr (std::same_as<T, TimestepEmbeddingNode>)
				    {
					    fn(n.timesteps);
				    }
				    else if constexpr (std::same_as<T, BinaryOpNode> || std::same_as<T, BatchMatMulNode> ||
				                       std::same_as<T, OutProdNode>)
				    {
					    fn(n.lhs);
					    fn(n.rhs);
				    }
				    else if constexpr (std::same_as<T, SolveTriNode>)
				    {
					    fn(n.a);
					    fn(n.b);
				    }
				    else if constexpr (std::same_as<T, GatherNode> || std::same_as<T, GetRowsNode>)
				    {
					    fn(n.data);
					    fn(n.indices);
				    }
				    else if constexpr (std::same_as<T, ScatterNode>)
				    {
					    fn(n.data);
					    fn(n.indices);
					    fn(n.updates);
				    }
				    else if constexpr (std::same_as<T, SSMScanNode>)
				    {
					    fn(n.state);
					    fn(n.dt);
					    fn(n.a);
					    fn(n.b);
					    fn(n.c);
					    if (n.d)
					    {
						    fn(*n.d);
					    }
				    }
				    else if constexpr (std::same_as<T, RWKVWKVNode>)
				    {
					    fn(n.key);
					    fn(n.value);
					    fn(n.receptance);
					    fn(n.timeDecay);
					    fn(n.timeFirst);
				    }
				    else if constexpr (std::same_as<T, CrossEntropyLossNode>)
				    {
					    fn(n.logits);
					    fn(n.labels);
				    }
				    else if constexpr (std::same_as<T, CrossEntropyLossBackwardNode>)
				    {
					    fn(n.grad);
					    fn(n.logits);
					    fn(n.labels);
				    }
				    else if constexpr (std::same_as<T, NormalizationNode>)
				    {
					    fn(n.input);
					    if (n.scale)
					    {
						    fn(*n.scale);
					    }
					    if (n.bias)
					    {
						    fn(*n.bias);
					    }
				    }
				    else if constexpr (std::same_as<T, SGDStepNode>)
				    {
					    fn(n.parameter);
					    fn(n.gradient);
					    if (n.velocity)
					    {
						    fn(*n.velocity);
					    }
				    }
				    else if constexpr (std::same_as<T, AdamWStepNode>)
				    {
					    fn(n.parameter);
					    fn(n.gradient);
					    fn(n.firstMoment);
					    fn(n.secondMoment);
				    }
				    else if constexpr (std::same_as<T, Conv2DNode> || std::same_as<T, ConvTranspose2DNode>)
				    {
					    fn(n.input);
					    fn(n.weight);
					    if (n.bias)
					    {
						    fn(*n.bias);
					    }
				    }
				    else if constexpr (std::same_as<T, ConcatNode>)
				    {
					    for (const auto& input : n.inputs)
					    {
						    fn(input);
					    }
				    }
				    else if constexpr (std::same_as<T, SliceNode>)
				    {
					    fn(n.input);
				    }
				    else if constexpr (std::same_as<T, MulMatIdNode>)
				    {
					    fn(n.as);
					    fn(n.b);
					    fn(n.ids);
				    }
				    else if constexpr (std::same_as<T, CallNode>)
				    {
					    for (const auto& arg : n.args)
					    {
						    fn(arg);
					    }
				    }
				    else if constexpr (std::same_as<T, CondNode>)
				    {
					    fn(n.condition);
					    for (const auto& arg : n.args)
					    {
						    fn(arg);
					    }
				    }
				    else if constexpr (std::same_as<T, WhileNode>)
				    {
					    for (const auto& arg : n.initArgs)
					    {
						    fn(arg);
					    }
				    }
				    else if constexpr (std::same_as<T, FusedOpNode>)
				    {
					    for (const auto& arg : n.args)
					    {
						    fn(arg);
					    }
				    }
				    else if constexpr (std::same_as<T, ParamRefNode> || std::same_as<T, ConstantNode> ||
				                       std::same_as<T, QuantizedConstantNode> ||
				                       std::same_as<T, VariableRefNode> ||
				                       std::same_as<T, LoadActivationNode> ||
				                       std::same_as<T, TapeLoadActivationNode>)
				    {
					    // no inputs
				    }
			    },
			    node);
		}

		template <typename RemapFn>
		static NodeVariant RemapNodeInputs(const NodeVariant& node, RemapFn&& remap)
		{
			auto remapVector = [&](const std::vector<NodeOutput>& inputs) {
				std::vector<NodeOutput> result;
				result.reserve(inputs.size());
				for (const auto& input : inputs)
				{
					result.push_back(remap(input));
				}
				return result;
			};
			return std::visit(
			    [&](const auto& n) -> NodeVariant {
				    using T = std::decay_t<decltype(n)>;
				    if constexpr (std::same_as<T, ParamRefNode> || std::same_as<T, ConstantNode> ||
				                  std::same_as<T, QuantizedConstantNode> || std::same_as<T, VariableRefNode> ||
				                  std::same_as<T, LoadActivationNode> || std::same_as<T, TapeLoadActivationNode>)
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
				    else if constexpr (std::same_as<T, CallNode>)
				    {
					    return CallNode{ n.callee, remapVector(n.args) };
				    }
				    else if constexpr (std::same_as<T, CondNode>)
				    {
					    return CondNode{ remap(n.condition), n.thenBranch, n.elseBranch, remapVector(n.args) };
				    }
				    else if constexpr (std::same_as<T, WhileNode>)
				    {
					    return WhileNode{ n.condBranch, n.bodyBranch, remapVector(n.initArgs) };
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
				    else if constexpr (std::same_as<T, CrossEntropyLossNode>)
				    {
					    return CrossEntropyLossNode{ remap(n.logits), remap(n.labels) };
				    }
				    else if constexpr (std::same_as<T, CrossEntropyLossBackwardNode>)
				    {
					    return CrossEntropyLossBackwardNode{ remap(n.grad), remap(n.logits), remap(n.labels) };
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
				    else if constexpr (std::same_as<T, OutProdNode>)
				    {
					    return OutProdNode{ remap(n.lhs), remap(n.rhs) };
				    }
				    else if constexpr (std::same_as<T, TimestepEmbeddingNode>)
				    {
					    return TimestepEmbeddingNode{ remap(n.timesteps), n.dim, n.maxPeriod };
				    }
				    else if constexpr (std::same_as<T, SolveTriNode>)
				    {
					    return SolveTriNode{ remap(n.a), remap(n.b), n.lower, n.unitDiagonal };
				    }
				    else if constexpr (std::same_as<T, SGDStepNode>)
				    {
					    return SGDStepNode{ remap(n.parameter), remap(n.gradient),
					                        n.velocity ? std::optional<NodeOutput>{ remap(*n.velocity) } : std::nullopt,
					                        n.learningRate, n.momentum, n.weightDecay, n.nesterov };
				    }
				    else if constexpr (std::same_as<T, AdamWStepNode>)
				    {
					    return AdamWStepNode{ remap(n.parameter), remap(n.gradient), remap(n.firstMoment),
					                          remap(n.secondMoment), n.learningRate, n.beta1, n.beta2,
					                          n.epsilon, n.weightDecay, n.step };
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
					    return ConvTranspose2DNode{ remap(n.input), remap(n.weight),
					                                n.bias ? std::optional<NodeOutput>{ remap(*n.bias) } : std::nullopt,
					                                n.strides, n.dilations, n.lowPads, n.highPads,
					                                n.outputPads, n.groupCount };
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
					    return ConcatNode{ remapVector(n.inputs), n.axis };
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
				    else if constexpr (std::same_as<T, TapeSaveActivationNode>)
				    {
					    return TapeSaveActivationNode{ remap(n.input), n.tapeSlotId };
				    }
				    else if constexpr (std::same_as<T, FusedOpNode>)
				    {
					    return FusedOpNode{ n.pattern, n.body, remapVector(n.args) };
				    }
				    else
				    {
					    throw std::runtime_error("EGraphPass: unsupported node type in remap");
				    }
			    },
			    node);
		}

		static NodeOutput Resolve(NodeOutput out, const std::vector<std::optional<NodeOutput>>& replacements)
		{
			std::set<NodeOutputKey> seen;
			while (out.node < replacements.size() && replacements[out.node])
			{
				if (!seen.insert({ out.node, out.port }).second)
				{
					break;
				}
				out = *replacements[out.node];
			}
			return out;
		}

		static void MarkReachable(const Subgraph& sg, NodeId nodeId, std::vector<bool>& alive,
		                          const std::vector<std::optional<NodeOutput>>& replacements,
		                          const std::vector<std::optional<NodeVariant>>& rewrites)
		{
			if (nodeId >= alive.size() || alive[nodeId])
			{
				return;
			}
			if (replacements[nodeId])
			{
				const auto resolved = Resolve(*replacements[nodeId], replacements);
				MarkReachable(sg, resolved.node, alive, replacements, rewrites);
				return;
			}
			alive[nodeId] = true;
			const auto& entry = sg.GetNodeEntry(nodeId);
			const auto& node = rewrites[nodeId] ? *rewrites[nodeId] : entry.node;
			ForEachInput(node, [&](NodeOutput input) {
				const auto resolved = Resolve(input, replacements);
				MarkReachable(sg, resolved.node, alive, replacements, rewrites);
			});
		}

		static std::string TermKey(std::string_view op, std::span<const std::size_t> children,
		                           const OutputInfo& info, std::string_view attrs = {})
		{
			std::ostringstream out;
			out << op << '(';
			for (auto i = 0uz; i < children.size(); ++i)
			{
				if (i != 0)
				{
					out << ',';
				}
				out << children[i];
			}
			out << "){" << attrs << "}:" << InfoKey(info);
			return out.str();
		}

		static std::string NodeTermName(const NodeVariant& node)
		{
			return std::visit(
			    []<typename T>(const T& n) -> std::string {
				    using NodeT = std::decay_t<T>;
				    if constexpr (std::same_as<NodeT, ParamRefNode>)
				    {
					    return "param";
				    }
				    else if constexpr (std::same_as<NodeT, ConstantNode>)
				    {
					    return "constant";
				    }
				    else if constexpr (std::same_as<NodeT, QuantizedConstantNode>)
				    {
					    return "quantized_constant";
				    }
				    else if constexpr (std::same_as<NodeT, VariableRefNode>)
				    {
					    return "variable";
				    }
				    else if constexpr (std::same_as<NodeT, UnaryOpNode>)
				    {
					    return std::string("unary.") + std::string(EnumToString<EnumToStringStyle::Unqualified>(n.op));
				    }
				    else if constexpr (std::same_as<NodeT, BinaryOpNode>)
				    {
					    return std::string("binary.") + std::string(EnumToString<EnumToStringStyle::Unqualified>(n.op));
				    }
				    else if constexpr (std::same_as<NodeT, ReshapeNode>)
				    {
					    return "reshape";
				    }
				    else if constexpr (std::same_as<NodeT, PermuteNode>)
				    {
					    return "permute";
				    }
				    else if constexpr (std::same_as<NodeT, BroadcastToNode>)
				    {
					    return "broadcast_to";
				    }
				    else if constexpr (std::same_as<NodeT, CastNode>)
				    {
					    return "cast";
				    }
				    else
				    {
					    return "opaque";
				    }
			    },
			    node);
		}

		bool ImportTerm(const Subgraph& sg, NodeId nodeId, Detail::TinyEGraph& egraph,
		                std::vector<std::optional<std::size_t>>& nodeClass)
		{
			const auto& entry = sg.GetNodeEntry(nodeId);
			if (entry.outputInfos.size() != 1)
			{
				return false;
			}
			const auto& info = entry.outputInfos[0];
			std::vector<std::size_t> children;
			bool pure = true;
			ForEachInput(entry.node, [&](NodeOutput input) {
				if (input.port != 0 || input.node >= nodeClass.size() || !nodeClass[input.node])
				{
					pure = false;
					return;
				}
				children.push_back(*nodeClass[input.node]);
			});
			if (!pure)
			{
				return false;
			}

			std::string attrs;
			if (const auto* param = std::get_if<ParamRefNode>(&entry.node))
			{
				attrs = std::to_string(param->paramIndex);
			}
			else if (const auto* variable = std::get_if<VariableRefNode>(&entry.node))
			{
				attrs = std::to_string(variable->variableIndex);
			}
			else if (const auto* reshape = std::get_if<ReshapeNode>(&entry.node))
			{
				attrs = ShapeKey(reshape->targetShape);
			}
			else if (const auto* permute = std::get_if<PermuteNode>(&entry.node))
			{
				attrs = ShapeKey(permute->permutation);
			}
			else if (const auto* broadcast = std::get_if<BroadcastToNode>(&entry.node))
			{
				attrs = ShapeKey(broadcast->targetShape);
			}
			else if (const auto* cast = std::get_if<CastNode>(&entry.node))
			{
				attrs = std::string(EnumToString<EnumToStringStyle::Unqualified>(cast->targetType));
			}
			else if (!std::holds_alternative<ParamRefNode>(entry.node) &&
			         !std::holds_alternative<ConstantNode>(entry.node) &&
			         !std::holds_alternative<QuantizedConstantNode>(entry.node) &&
			         !std::holds_alternative<VariableRefNode>(entry.node) &&
			         !std::holds_alternative<UnaryOpNode>(entry.node) &&
			         !std::holds_alternative<BinaryOpNode>(entry.node))
			{
				return false;
			}

			nodeClass[nodeId] = egraph.AddTerm({ TermKey(NodeTermName(entry.node), children, info, attrs),
			                                     children, info });
			return true;
		}

		void AddEvent(SubgraphId sgId, NodeId nodeId, std::string rule, std::string before, std::string after)
		{
			lastReport_.events.push_back({ sgId, nodeId, std::move(rule), std::move(before), std::move(after) });
			++lastReport_.rewrites;
		}

		bool SetReplacement(SubgraphId sgId, NodeId nodeId, NodeOutput target, std::string rule,
		                    std::string before, std::string after,
		                    std::vector<std::optional<NodeOutput>>& replacements)
		{
			const auto resolved = Resolve(target, replacements);
			if (resolved.node == nodeId && resolved.port == 0)
			{
				return false;
			}
			replacements[nodeId] = resolved;
			AddEvent(sgId, nodeId, std::move(rule), std::move(before), std::move(after));
			return true;
		}

		bool SetRewrite(SubgraphId sgId, NodeId nodeId, NodeVariant rewrite, std::string rule,
		                std::string before, std::string after, std::vector<std::optional<NodeVariant>>& rewrites)
		{
			rewrites[nodeId].emplace(std::move(rewrite));
			AddEvent(sgId, nodeId, std::move(rule), std::move(before), std::move(after));
			return true;
		}

		bool ApplyRules(const Subgraph& sg, SubgraphId sgId, Detail::TinyEGraph& egraph,
		                const std::vector<std::optional<std::size_t>>& nodeClass,
		                std::vector<std::optional<NodeOutput>>& replacements,
		                std::vector<std::optional<NodeVariant>>& rewrites)
		{
			bool changed = false;
			for (NodeId nodeId = 0; nodeId < sg.NodeCount(); ++nodeId)
			{
				const auto& entry = sg.GetNodeEntry(nodeId);
				if (entry.outputInfos.size() != 1)
				{
					continue;
				}
				const auto& outInfo = entry.outputInfos[0];

				if (const auto* bin = std::get_if<BinaryOpNode>(&entry.node))
				{
					const auto lhs = Resolve(bin->lhs, replacements);
					const auto rhs = Resolve(bin->rhs, replacements);
					const auto& lhsInfo = sg.GetOutputInfo(lhs);
					const auto& rhsInfo = sg.GetOutputInfo(rhs);
					const auto lhsConst = std::get_if<ConstantNode>(&sg.GetNodeEntry(lhs.node).node);
					const auto rhsConst = std::get_if<ConstantNode>(&sg.GetNodeEntry(rhs.node).node);
					std::optional<Tensor<CPU>> lhsCPUStorage;
					std::optional<Tensor<CPU>> rhsCPUStorage;
					if (lhsConst)
					{
						lhsCPUStorage.emplace(lhsConst->value.CopyToDevice(CPU{}));
					}
					if (rhsConst)
					{
						rhsCPUStorage.emplace(rhsConst->value.CopyToDevice(CPU{}));
					}
					const auto* lhsCPU = lhsCPUStorage ? &*lhsCPUStorage : nullptr;
					const auto* rhsCPU = rhsCPUStorage ? &*rhsCPUStorage : nullptr;

					if (bin->op == BinaryOp::Add)
					{
						if (rhsCPU && IsZeroTensor(*rhsCPU) && SameInfo(lhsInfo, outInfo))
						{
							if (nodeClass[nodeId] && nodeClass[lhs.node])
							{
								egraph.Union(*nodeClass[nodeId], *nodeClass[lhs.node]);
							}
							changed |= SetReplacement(sgId, nodeId, lhs, "add-zero-rhs", NodeKey(nodeId), OutputKey(lhs),
							                          replacements);
							continue;
						}
						if (lhsCPU && IsZeroTensor(*lhsCPU) && SameInfo(rhsInfo, outInfo))
						{
							if (nodeClass[nodeId] && nodeClass[rhs.node])
							{
								egraph.Union(*nodeClass[nodeId], *nodeClass[rhs.node]);
							}
							changed |= SetReplacement(sgId, nodeId, rhs, "add-zero-lhs", NodeKey(nodeId), OutputKey(rhs),
							                          replacements);
							continue;
						}
					}
					else if (bin->op == BinaryOp::Subtract)
					{
						if (rhsCPU && IsZeroTensor(*rhsCPU) && SameInfo(lhsInfo, outInfo))
						{
							if (nodeClass[nodeId] && nodeClass[lhs.node])
							{
								egraph.Union(*nodeClass[nodeId], *nodeClass[lhs.node]);
							}
							changed |= SetReplacement(sgId, nodeId, lhs, "subtract-zero-rhs", NodeKey(nodeId),
							                          OutputKey(lhs), replacements);
							continue;
						}
					}
					else if (bin->op == BinaryOp::Multiply)
					{
						if (rhsCPU && IsOneTensor(*rhsCPU) && SameInfo(lhsInfo, outInfo))
						{
							if (nodeClass[nodeId] && nodeClass[lhs.node])
							{
								egraph.Union(*nodeClass[nodeId], *nodeClass[lhs.node]);
							}
							changed |= SetReplacement(sgId, nodeId, lhs, "multiply-one-rhs", NodeKey(nodeId),
							                          OutputKey(lhs), replacements);
							continue;
						}
						if (lhsCPU && IsOneTensor(*lhsCPU) && SameInfo(rhsInfo, outInfo))
						{
							if (nodeClass[nodeId] && nodeClass[rhs.node])
							{
								egraph.Union(*nodeClass[nodeId], *nodeClass[rhs.node]);
							}
							changed |= SetReplacement(sgId, nodeId, rhs, "multiply-one-lhs", NodeKey(nodeId),
							                          OutputKey(rhs), replacements);
							continue;
						}
						if (rhsCPU && IsZeroTensor(*rhsCPU) && SameInfo(rhsInfo, outInfo))
						{
							if (nodeClass[nodeId] && nodeClass[rhs.node])
							{
								egraph.Union(*nodeClass[nodeId], *nodeClass[rhs.node]);
							}
							changed |= SetReplacement(sgId, nodeId, rhs, "multiply-zero-rhs", NodeKey(nodeId),
							                          OutputKey(rhs), replacements);
							continue;
						}
						if (lhsCPU && IsZeroTensor(*lhsCPU) && SameInfo(lhsInfo, outInfo))
						{
							if (nodeClass[nodeId] && nodeClass[lhs.node])
							{
								egraph.Union(*nodeClass[nodeId], *nodeClass[lhs.node]);
							}
							changed |= SetReplacement(sgId, nodeId, lhs, "multiply-zero-lhs", NodeKey(nodeId),
							                          OutputKey(lhs), replacements);
							continue;
						}
					}

					if (CanCanonicalizeCommutative(bin->op, outInfo.dtype) && OutputKey(rhs) < OutputKey(lhs))
					{
						changed |= SetRewrite(sgId, nodeId, BinaryOpNode{ bin->op, rhs, lhs }, "commutative-canonical",
						                      OutputKey(lhs) + "," + OutputKey(rhs),
						                      OutputKey(rhs) + "," + OutputKey(lhs), rewrites);
					}
				}
				else if (const auto* unary = std::get_if<UnaryOpNode>(&entry.node))
				{
					if (unary->op == UnaryOp::Negate)
					{
						const auto input = Resolve(unary->input, replacements);
						const auto& innerEntry = sg.GetNodeEntry(input.node);
						if (const auto* inner = std::get_if<UnaryOpNode>(&innerEntry.node);
						    inner && inner->op == UnaryOp::Negate && SameInfo(sg.GetOutputInfo(inner->input), outInfo))
						{
							const auto target = Resolve(inner->input, replacements);
							if (nodeClass[nodeId] && nodeClass[target.node])
							{
								egraph.Union(*nodeClass[nodeId], *nodeClass[target.node]);
							}
							changed |= SetReplacement(sgId, nodeId, target, "double-negate", NodeKey(nodeId),
							                          OutputKey(target), replacements);
						}
					}
				}
				else if (const auto* reshape = std::get_if<ReshapeNode>(&entry.node))
				{
					const auto input = Resolve(reshape->input, replacements);
					const auto& inputInfo = sg.GetOutputInfo(input);
					if (SameInfo(inputInfo, outInfo))
					{
						if (nodeClass[nodeId] && nodeClass[input.node])
						{
							egraph.Union(*nodeClass[nodeId], *nodeClass[input.node]);
						}
						changed |= SetReplacement(sgId, nodeId, input, "reshape-noop", NodeKey(nodeId), OutputKey(input),
						                          replacements);
						continue;
					}
					if (const auto* inner = std::get_if<ReshapeNode>(&sg.GetNodeEntry(input.node).node))
					{
						changed |= SetRewrite(sgId, nodeId, ReshapeNode{ Resolve(inner->input, replacements),
						                                                 reshape->targetShape },
						                      "reshape-compose", OutputKey(input), OutputKey(inner->input), rewrites);
					}
				}
				else if (const auto* permute = std::get_if<PermuteNode>(&entry.node))
				{
					const auto input = Resolve(permute->input, replacements);
					const auto& inputInfo = sg.GetOutputInfo(input);
					if (SameInfo(inputInfo, outInfo) && IsIdentityPermutation(permute->permutation))
					{
						if (nodeClass[nodeId] && nodeClass[input.node])
						{
							egraph.Union(*nodeClass[nodeId], *nodeClass[input.node]);
						}
						changed |= SetReplacement(sgId, nodeId, input, "permute-identity", NodeKey(nodeId),
						                          OutputKey(input), replacements);
						continue;
					}
					if (const auto* inner = std::get_if<PermuteNode>(&sg.GetNodeEntry(input.node).node))
					{
						auto combined = ComposePermutation(inner->permutation, permute->permutation);
						if (IsIdentityPermutation(combined))
						{
							const auto target = Resolve(inner->input, replacements);
							changed |= SetReplacement(sgId, nodeId, target, "permute-compose-identity",
							                          NodeKey(nodeId), OutputKey(target), replacements);
						}
						else
						{
							changed |= SetRewrite(sgId, nodeId, PermuteNode{ Resolve(inner->input, replacements),
							                                                  std::move(combined) },
							                      "permute-compose", OutputKey(input), OutputKey(inner->input), rewrites);
						}
					}
				}
				else if (const auto* broadcast = std::get_if<BroadcastToNode>(&entry.node))
				{
					const auto input = Resolve(broadcast->input, replacements);
					const auto& inputInfo = sg.GetOutputInfo(input);
					if (SameInfo(inputInfo, outInfo))
					{
						if (nodeClass[nodeId] && nodeClass[input.node])
						{
							egraph.Union(*nodeClass[nodeId], *nodeClass[input.node]);
						}
						changed |= SetReplacement(sgId, nodeId, input, "broadcast-noop", NodeKey(nodeId),
						                          OutputKey(input), replacements);
						continue;
					}
					if (const auto* inner = std::get_if<BroadcastToNode>(&sg.GetNodeEntry(input.node).node))
					{
						const auto innerInput = Resolve(inner->input, replacements);
						const auto& innerInputInfo = sg.GetOutputInfo(innerInput);
						if (CanBroadcastTrailing(innerInputInfo.shape, broadcast->targetShape))
						{
							changed |= SetRewrite(sgId, nodeId,
							                      BroadcastToNode{ innerInput, broadcast->targetShape },
							                      "broadcast-compose", OutputKey(input), OutputKey(innerInput), rewrites);
						}
					}
				}
			}
			return changed;
		}

		void RebuildSubgraph(Graph& graph, SubgraphId sgId,
		                     const std::vector<std::optional<NodeOutput>>& replacements,
		                     const std::vector<std::optional<NodeVariant>>& rewrites)
		{
			const auto& sg = graph.GetSubgraph(sgId);
			const auto nodeCount = sg.NodeCount();
			std::vector<bool> alive(nodeCount, false);
			for (const auto& result : sg.Results())
			{
				const auto resolved = Resolve(result, replacements);
				MarkReachable(sg, resolved.node, alive, replacements, rewrites);
			}

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
				const auto resolved = Resolve(out, replacements);
				return { nodeMap[resolved.node], resolved.port };
			};

			for (NodeId oldId = 0; oldId < nodeCount; ++oldId)
			{
				if (!alive[oldId] || replacements[oldId] || std::holds_alternative<ParamRefNode>(sg.GetNodeEntry(oldId).node))
				{
					continue;
				}
				const auto& entry = sg.GetNodeEntry(oldId);
				const auto& node = rewrites[oldId] ? *rewrites[oldId] : entry.node;
				auto remapped = RemapNodeInputs(node, remapOutput);
				nodeMap[oldId] =
				    newSg.AddNode(std::move(remapped), { entry.outputInfos.begin(), entry.outputInfos.end() });
			}

			std::vector<NodeOutput> results;
			results.reserve(sg.Results().size());
			for (const auto& result : sg.Results())
			{
				results.push_back(remapOutput(result));
			}
			newSg.SetResults(std::move(results));
			graph.GetSubgraph(sgId) = std::move(newSg);
		}

		bool ProcessSubgraphIteration(Graph& graph, SubgraphId sgId,
		                              std::chrono::steady_clock::time_point startTime)
		{
			const auto& sg = graph.GetSubgraph(sgId);
			if (sg.NodeCount() > options_.maxTerms)
			{
				lastReport_.hitLimit = true;
				return false;
			}

			Detail::TinyEGraph egraph;
			std::vector<std::optional<std::size_t>> nodeClass(sg.NodeCount());
			for (NodeId nodeId = 0; nodeId < sg.NodeCount(); ++nodeId)
			{
				if (std::chrono::steady_clock::now() - startTime > options_.timeout)
				{
					lastReport_.hitLimit = true;
					return false;
				}
				ImportTerm(sg, nodeId, egraph, nodeClass);
				if (egraph.TermCount() > options_.maxTerms || egraph.EClassCount() > options_.maxEClasses)
				{
					lastReport_.hitLimit = true;
					return false;
				}
			}
			lastReport_.importedTerms += egraph.TermCount();
			lastReport_.eClasses += egraph.EClassCount();

			std::vector<std::optional<NodeOutput>> replacements(sg.NodeCount());
			std::vector<std::optional<NodeVariant>> rewrites(sg.NodeCount());
			const bool changed = ApplyRules(sg, sgId, egraph, nodeClass, replacements, rewrites);
			if (changed)
			{
				RebuildSubgraph(graph, sgId, replacements, rewrites);
				Validation::ValidateGraph(graph);
			}
			return changed;
		}

		void ProcessSubgraph(Graph& graph, SubgraphId sgId)
		{
			const auto start = std::chrono::steady_clock::now();
			for (auto iteration = 0uz; iteration < options_.maxIterations; ++iteration)
			{
				if (std::chrono::steady_clock::now() - start > options_.timeout)
				{
					lastReport_.hitLimit = true;
					return;
				}
				++lastReport_.iterations;
				if (!ProcessSubgraphIteration(graph, sgId, start))
				{
					return;
				}
			}
			lastReport_.hitLimit = true;
		}

		EGraphOptions options_;
		EGraphReport lastReport_;
	};
} // namespace LiteNN

#endif
