#include <LiteNN/Training/TrainStepAOTRunner.h>

#ifdef LITENN_ENABLE_MLIR
#include <LiteNN/Compiler/CompiledModule.h>
#endif
#include <LiteNN/Optimizer/GraphOps.h>

#include <format>
#include <utility>

namespace LiteNN::Training
{
	namespace
	{
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
			backwardPlan.forward = *plan.backward;
			backwardPlan.backward = std::nullopt;
			const auto& backwardSubgraph = backwardPlan.subgraphs[backwardPlan.forward];

			backwardPlan.inputs.clear();
			backwardPlan.inputs.reserve(backwardSubgraph.params.size());
			for (std::size_t i = 0; i < backwardSubgraph.params.size(); ++i)
			{
				backwardPlan.inputs.push_back(
				    { .source = { static_cast<NodeId>(i), 0 },
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
				backwardPlan.outputs.push_back(
				    { .source = result,
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
			const auto outputs = Optimizer::AddSGDStep(sg, { parameter, 0 }, { gradient, 0 }, std::nullopt,
			                                           options.learningRate, options.momentum,
			                                           options.weightDecay, options.nesterov);
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
			const auto outputs =
			    Optimizer::AddAdamWStep(sg, { parameter, 0 }, { gradient, 0 }, { firstMoment, 0 },
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
		auto module = Compiler<CPU>::Compile(plan);
		return [module = std::move(module)](std::span<const Tensor<CPU>> inputs) {
			return module.RunTensors(inputs);
		};
#endif
	}

	template <>
	CompiledBackwardRunner<CPU> CreateCompiledTrainBackwardRunner(const ExecutablePlan& plan, CPU)
	{
#ifndef LITENN_ENABLE_MLIR
		throw std::runtime_error("Trainer AOT backward runner requires LiteNNCompiler/MLIR support");
#else
		auto module = Compiler<CPU>::Compile(BuildBackwardEntryPlan(plan));
		return [module = std::move(module)](std::span<const Tensor<CPU>> inputs) {
			return module.RunTensors(inputs);
		};
#endif
	}

	template <>
	CompiledOptimizerUpdateRunner<CPU> CreateCompiledSGDUpdateRunner(const TensorType& parameterType,
	                                                                 Optimizer::SGDOptions options, CPU)
	{
#ifndef LITENN_ENABLE_MLIR
		throw std::runtime_error("Trainer AOT SGD update runner requires LiteNNCompiler/MLIR support");
#else
		auto module = Compiler<CPU>::Compile(Detail::BuildExecutablePlanFromGraph(BuildSGDUpdateGraph(parameterType, options)));
		return [module = std::move(module)](std::span<const Tensor<CPU>> inputs) {
			return module.RunTensors(inputs);
		};
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
		auto module = Compiler<CPU>::Compile(Detail::BuildExecutablePlanFromGraph(BuildAdamWUpdateGraph(parameterType, options, step)));
		return [module = std::move(module)](std::span<const Tensor<CPU>> inputs) {
			return module.RunTensors(inputs);
		};
#endif
	}

#ifdef LITENN_ENABLE_CUDA
	template <>
	CompiledForwardRunner<CUDA> CreateCompiledTrainForwardRunner(const ExecutablePlan& plan, CUDA device)
	{
#ifndef LITENN_ENABLE_MLIR
		throw std::runtime_error("Trainer CUDA AOT forward runner requires LiteNNCompiler/MLIR support");
#else
		auto module = Compiler<CUDA>::Compile(plan, std::move(device));
		return [module = std::move(module)](std::span<const Tensor<CUDA>> inputs) {
			return module.RunTensors(inputs);
		};
#endif
	}

	template <>
	CompiledBackwardRunner<CUDA> CreateCompiledTrainBackwardRunner(const ExecutablePlan& plan, CUDA device)
	{
#ifndef LITENN_ENABLE_MLIR
		throw std::runtime_error("Trainer CUDA AOT backward runner requires LiteNNCompiler/MLIR support");
#else
		auto module = Compiler<CUDA>::Compile(BuildBackwardEntryPlan(plan), std::move(device));
		return [module = std::move(module)](std::span<const Tensor<CUDA>> inputs) {
			return module.RunTensors(inputs);
		};
#endif
	}
#endif
} // namespace LiteNN::Training
