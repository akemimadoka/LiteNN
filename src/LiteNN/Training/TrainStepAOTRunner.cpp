#include <LiteNN/Training/TrainStepAOTRunner.h>

#ifdef LITENN_ENABLE_MLIR
#include <LiteNN/Compiler/CompiledModule.h>
#endif
#include <LiteNN/Optimizer/GraphOps.h>

#include <algorithm>
#include <format>
#include <map>
#include <utility>

namespace LiteNN::Training
{
	namespace
	{
		struct SavedActivationCapture
		{
			std::size_t slot{};
			NodeOutput value;
			TensorType type;
		};

		std::vector<SavedActivationCapture> CollectForwardSavedActivations(const ExecutablePlan& plan)
		{
			if (plan.forward >= plan.subgraphs.size())
			{
				throw std::runtime_error("Trainer AOT forward runner references an invalid forward subgraph");
			}
			std::vector<SavedActivationCapture> captures;
			const auto& forwardSubgraph = plan.subgraphs[plan.forward];
			for (const auto& node : forwardSubgraph.nodes)
			{
				const auto* save = std::get_if<SaveActivationNode>(&node.node);
				if (save == nullptr)
				{
					continue;
				}
				if (save->slotId >= plan.activationSlots.size())
				{
					throw std::runtime_error("Trainer AOT forward runner references an invalid activation slot");
				}
				if (node.outputs.empty())
				{
					throw std::runtime_error("Trainer AOT forward runner cannot capture a save node without outputs");
				}
				captures.push_back({ .slot = save->slotId, .value = { node.sourceNode, 0 }, .type = node.outputs[0] });
			}
			std::ranges::sort(captures, {}, &SavedActivationCapture::slot);
			const auto duplicate =
			    std::ranges::adjacent_find(captures, {}, &SavedActivationCapture::slot) != captures.end();
			if (duplicate)
			{
				throw std::runtime_error("Trainer AOT forward runner cannot capture duplicate activation slots yet");
			}
			return captures;
		}

		ExecutablePlan BuildForwardCapturePlan(const ExecutablePlan& plan)
		{
			auto capturePlan = plan;
			const auto forwardSubgraphCopy = capturePlan.subgraphs[capturePlan.forward];
			capturePlan.subgraphs.clear();
			capturePlan.subgraphs.push_back(forwardSubgraphCopy);
			capturePlan.subgraphs[0].sourceSubgraph = 0;
			capturePlan.forward = 0;
			capturePlan.backward = std::nullopt;
			auto& forwardSubgraph = capturePlan.subgraphs[capturePlan.forward];
			for (const auto& capture : CollectForwardSavedActivations(capturePlan))
			{
				forwardSubgraph.results.push_back(capture.value);
				capturePlan.outputs.push_back({ .source = capture.value,
				                                .type = capture.type,
				                                .name = std::format("activation.{}", capture.slot) });
			}
			ValidateExecutablePlan(capturePlan);
			return capturePlan;
		}

		ExecutablePlanNode MakeParamRefPlanNode(NodeId sourceNode, std::size_t paramIndex, const TensorType& type)
		{
			ParamRefNode param{ paramIndex };
			const auto opKind = OpKindName(NodeVariant{ param });
			const auto& registry = DefaultOpSchemaRegistry();
			const auto& schema = registry.Require(opKind);
			ExecutablePlanNode node;
			node.sourceNode = sourceNode;
			node.op = BuildExecutablePlanOp(NodeVariant{ param }, schema,
			                                static_cast<std::uint32_t>(registry.IndexOf(opKind)));
			node.node = param;
			node.opKind = opKind;
			node.category = schema.category;
			node.effect = schema.effect;
			node.outputs = { type };
			return node;
		}

		void RefreshPlanNodeMetadata(ExecutablePlanNode& node)
		{
			const auto opKind = OpKindName(node.node);
			const auto& registry = DefaultOpSchemaRegistry();
			const auto& schema = registry.Require(opKind);
			node.op = BuildExecutablePlanOp(node.node, schema, static_cast<std::uint32_t>(registry.IndexOf(opKind)));
			node.opKind = opKind;
			node.category = schema.category;
			node.effect = schema.effect;
			node.inputs = NodeInputs(node.node);
		}

		template <typename RemapFn>
		NodeVariant RemapNodeInputsForTrainingAOT(const NodeVariant& node, RemapFn&& remap)
		{
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
				    else if constexpr (std::same_as<T, QuantizedMatMulNode>)
				    {
					    return QuantizedMatMulNode{ remap(n.lhs), remap(n.rhsStorage), n.params, n.transposeRhs };
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
				    else if constexpr (std::same_as<T, BatchMatMulNode>)
				    {
					    return BatchMatMulNode{ remap(n.lhs), remap(n.rhs) };
				    }
				    else if constexpr (std::same_as<T, OutProdNode>)
				    {
					    return OutProdNode{ remap(n.lhs), remap(n.rhs) };
				    }
				    else if constexpr (std::same_as<T, ActivePrefixAttentionNode>)
				    {
					    return ActivePrefixAttentionNode{ remap(n.query),           remap(n.keys), remap(n.values),
						                                  remap(n.currentPosition), n.scale,       n.kvHeadIndex };
				    }
				    else if constexpr (std::same_as<T, GroupedActivePrefixAttentionNode>)
				    {
					    return GroupedActivePrefixAttentionNode{ remap(n.queries), remap(n.keys),
						                                         remap(n.values),  remap(n.currentPosition),
						                                         n.scale,          n.queryGroupsPerKVHead };
				    }
				    else if constexpr (std::same_as<T, SaveActivationNode>)
				    {
					    return SaveActivationNode{ remap(n.input), n.slotId };
				    }
				    else
				    {
					    return n;
				    }
			    },
			    node);
		}

		void BindBackwardSavedActivations(ExecutablePlanSubgraph& subgraph,
		                                  std::span<const SavedActivationCapture> captures)
		{
			if (captures.empty())
			{
				return;
			}
			const auto oldParamCount = subgraph.params.size();
			const auto insertedParamCount = captures.size();
			std::map<std::size_t, NodeOutput> activationValueBySlot;
			for (std::size_t i = 0; i < captures.size(); ++i)
			{
				subgraph.params.push_back(captures[i].type);
				activationValueBySlot[captures[i].slot] = { oldParamCount + i, 0 };
			}

			std::map<NodeId, NodeOutput> loadReplacement;
			for (const auto& node : subgraph.nodes)
			{
				if (const auto* load = std::get_if<LoadActivationNode>(&node.node))
				{
					const auto it = activationValueBySlot.find(load->slotId);
					if (it == activationValueBySlot.end())
					{
						throw std::runtime_error(
						    std::format("Trainer AOT backward runner cannot bind activation slot {}", load->slotId));
					}
					loadReplacement[node.sourceNode] = it->second;
				}
			}

			auto remap = [&](NodeOutput output) -> NodeOutput {
				if (const auto it = loadReplacement.find(output.node); it != loadReplacement.end())
				{
					return { it->second.node, output.port };
				}
				if (output.node < oldParamCount)
				{
					return output;
				}
				return { output.node + insertedParamCount, output.port };
			};

			std::vector<ExecutablePlanNode> nodes;
			nodes.reserve(subgraph.nodes.size() + insertedParamCount);
			for (std::size_t i = 0; i < oldParamCount; ++i)
			{
				nodes.push_back(subgraph.nodes[i]);
			}
			for (std::size_t i = 0; i < captures.size(); ++i)
			{
				nodes.push_back(MakeParamRefPlanNode(nodes.size(), oldParamCount + i, captures[i].type));
			}
			for (std::size_t i = oldParamCount; i < subgraph.nodes.size(); ++i)
			{
				auto node = subgraph.nodes[i];
				node.sourceNode = nodes.size();
				if (const auto* load = std::get_if<LoadActivationNode>(&node.node))
				{
					const auto input = activationValueBySlot.at(load->slotId);
					node.node = CastNode{ input, node.outputs[0].dtype };
				}
				else
				{
					node.node = RemapNodeInputsForTrainingAOT(node.node, remap);
				}
				RefreshPlanNodeMetadata(node);
				nodes.push_back(std::move(node));
			}
			for (auto& result : subgraph.results)
			{
				result = remap(result);
			}
			subgraph.nodes = std::move(nodes);
		}

		ExecutablePlan BuildBackwardEntryPlan(const ExecutablePlan& plan)
		{
			if (!plan.backward)
			{
				throw std::runtime_error("Trainer AOT backward runner requires a backward subgraph");
			}
			if (*plan.backward >= plan.subgraphs.size())
			{
				throw std::runtime_error("Trainer AOT backward runner references an invalid backward subgraph");
			}

			ExecutablePlan backwardPlan = plan;
			const auto backwardSubgraphSource = *plan.backward;
			const auto backwardSubgraphCopy = backwardPlan.subgraphs[backwardSubgraphSource];
			backwardPlan.subgraphs.clear();
			backwardPlan.subgraphs.push_back(backwardSubgraphCopy);
			backwardPlan.subgraphs[0].sourceSubgraph = 0;
			backwardPlan.forward = 0;
			backwardPlan.backward = std::nullopt;
			const auto captures = CollectForwardSavedActivations(plan);
			BindBackwardSavedActivations(backwardPlan.subgraphs[0], captures);
			auto& backwardSubgraph = backwardPlan.subgraphs[backwardPlan.forward];

			backwardPlan.inputs.clear();
			backwardPlan.inputs.reserve(backwardSubgraph.params.size());
			for (std::size_t i = 0; i < backwardSubgraph.params.size(); ++i)
			{
				backwardPlan.inputs.push_back({ .source = { static_cast<NodeId>(i), 0 },
				                                .type = backwardSubgraph.params[i],
				                                .name = std::format("backward.input{}", i) });
			}

			backwardPlan.outputs.clear();
			backwardPlan.outputs.reserve(backwardSubgraph.results.size());
			for (std::size_t i = 0; i < backwardSubgraph.results.size(); ++i)
			{
				const auto result = backwardSubgraph.results[i];
				if (result.node >= backwardSubgraph.nodes.size() ||
				    result.port >= backwardSubgraph.nodes[result.node].outputs.size())
				{
					throw std::runtime_error("Trainer AOT backward runner references an invalid backward result");
				}
				backwardPlan.outputs.push_back({ .source = result,
				                                 .type = backwardSubgraph.nodes[result.node].outputs[result.port],
				                                 .name = std::format("backward.output{}", i) });
			}
			ValidateExecutablePlan(backwardPlan);
			return backwardPlan;
		}

		Graph BuildSGDUpdateGraph(const TensorType& parameterType, const Optimizer::SGDOptions& options)
		{
			if (parameterType.dtype != DataType::Float32 || !parameterType.IsFullyStatic())
			{
				throw std::runtime_error("Trainer AOT SGD update runner requires a static Float32 parameter type");
			}
			if (options.momentum != 0.0F)
			{
				throw std::runtime_error("Trainer AOT SGD update runner currently supports momentum-free SGD only");
			}

			Graph graph;
			Subgraph sg;
			const auto parameter = sg.AddParam(parameterType.dtype, parameterType.StaticShape());
			const auto gradient = sg.AddParam(parameterType.dtype, parameterType.StaticShape());
			const auto outputs =
			    Optimizer::AddSGDStep(sg, { parameter, 0 }, { gradient, 0 }, std::nullopt, options.learningRate,
			                          options.momentum, options.weightDecay, options.nesterov);
			sg.SetResults(outputs);
			graph.SetForward(graph.AddSubgraph(std::move(sg)));
			ValidateExecutablePlan(Detail::BuildExecutablePlanFromGraph(graph));
			return graph;
		}

		Graph BuildAdamWUpdateGraph(const TensorType& parameterType, const Optimizer::AdamWOptions& options,
		                            std::size_t step)
		{
			if (parameterType.dtype != DataType::Float32 || !parameterType.IsFullyStatic())
			{
				throw std::runtime_error("Trainer AOT AdamW update runner requires a static Float32 parameter type");
			}
			if (step == 0)
			{
				throw std::runtime_error("Trainer AOT AdamW update runner requires a positive step");
			}

			Graph graph;
			Subgraph sg;
			const auto parameter = sg.AddParam(parameterType.dtype, parameterType.StaticShape());
			const auto gradient = sg.AddParam(parameterType.dtype, parameterType.StaticShape());
			const auto firstMoment = sg.AddParam(parameterType.dtype, parameterType.StaticShape());
			const auto secondMoment = sg.AddParam(parameterType.dtype, parameterType.StaticShape());
			const auto outputs = Optimizer::AddAdamWStep(sg, { parameter, 0 }, { gradient, 0 }, { firstMoment, 0 },
			                                             { secondMoment, 0 }, options.learningRate, options.beta1,
			                                             options.beta2, options.epsilon, options.weightDecay, step);
			sg.SetResults(outputs);
			graph.SetForward(graph.AddSubgraph(std::move(sg)));
			ValidateExecutablePlan(Detail::BuildExecutablePlanFromGraph(graph));
			return graph;
		}
	} // namespace

	template <>
	CompiledForwardRunner<CPU> CreateCompiledTrainForwardRunner(const ExecutablePlan& plan, CPU)
	{
#ifndef LITENN_ENABLE_MLIR
		throw std::runtime_error("Trainer AOT forward runner requires LiteNNCompiler/MLIR support");
#else
		auto module = Compiler<CPU>::Compile(BuildForwardCapturePlan(plan));
		return [module = std::move(module)](std::span<const Tensor<CPU>> inputs) { return module.RunTensors(inputs); };
#endif
	}

	template <>
	CompiledBackwardRunner<CPU> CreateCompiledTrainBackwardRunner(const ExecutablePlan& plan, CPU)
	{
#ifndef LITENN_ENABLE_MLIR
		throw std::runtime_error("Trainer AOT backward runner requires LiteNNCompiler/MLIR support");
#else
		auto module = Compiler<CPU>::Compile(BuildBackwardEntryPlan(plan));
		return [module = std::move(module)](std::span<const Tensor<CPU>> inputs) { return module.RunTensors(inputs); };
#endif
	}

	template <>
	CompiledOptimizerUpdateRunner<CPU> CreateCompiledSGDUpdateRunner(const TensorType& parameterType,
	                                                                 Optimizer::SGDOptions options, CPU)
	{
#ifndef LITENN_ENABLE_MLIR
		throw std::runtime_error("Trainer AOT SGD update runner requires LiteNNCompiler/MLIR support");
#else
		auto module =
		    Compiler<CPU>::Compile(Detail::BuildExecutablePlanFromGraph(BuildSGDUpdateGraph(parameterType, options)));
		return [module = std::move(module)](std::span<const Tensor<CPU>> inputs) { return module.RunTensors(inputs); };
#endif
	}

	template <>
	CompiledOptimizerUpdateRunner<CPU> CreateCompiledAdamWUpdateRunner(const TensorType& parameterType,
	                                                                   Optimizer::AdamWOptions options,
	                                                                   std::size_t step, CPU)
	{
#ifndef LITENN_ENABLE_MLIR
		throw std::runtime_error("Trainer AOT AdamW update runner requires LiteNNCompiler/MLIR support");
#else
		auto module = Compiler<CPU>::Compile(
		    Detail::BuildExecutablePlanFromGraph(BuildAdamWUpdateGraph(parameterType, options, step)));
		return [module = std::move(module)](std::span<const Tensor<CPU>> inputs) { return module.RunTensors(inputs); };
#endif
	}

#ifdef LITENN_ENABLE_CUDA
	template <>
	CompiledForwardRunner<CUDA> CreateCompiledTrainForwardRunner(const ExecutablePlan& plan, CUDA device)
	{
#ifndef LITENN_ENABLE_MLIR
		throw std::runtime_error("Trainer CUDA AOT forward runner requires LiteNNCompiler/MLIR support");
#else
		auto module = Compiler<CUDA>::Compile(BuildForwardCapturePlan(plan), std::move(device));
		return [module = std::move(module)](std::span<const Tensor<CUDA>> inputs) { return module.RunTensors(inputs); };
#endif
	}

	template <>
	CompiledBackwardRunner<CUDA> CreateCompiledTrainBackwardRunner(const ExecutablePlan& plan, CUDA device)
	{
#ifndef LITENN_ENABLE_MLIR
		throw std::runtime_error("Trainer CUDA AOT backward runner requires LiteNNCompiler/MLIR support");
#else
		auto module = Compiler<CUDA>::Compile(BuildBackwardEntryPlan(plan), std::move(device));
		return [module = std::move(module)](std::span<const Tensor<CUDA>> inputs) { return module.RunTensors(inputs); };
#endif
	}
#endif
} // namespace LiteNN::Training
