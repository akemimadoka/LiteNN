#include <LiteNN/Training/TrainStepAOTRunner.h>

#ifdef LITENN_ENABLE_MLIR
#include <LiteNN/Compiler/CompiledModule.h>
#endif

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
	} // namespace

	template <>
	CompiledForwardRunner<CPU> CreateCompiledTrainForwardRunner(const ExecutablePlan& plan, CPU)
	{
#ifndef LITENN_ENABLE_MLIR
		throw std::runtime_error("Trainer AOT forward runner requires LiteNNCompiler/MLIR support");
#else
		auto module = Compiler<CPU>::Compile(plan);
		return [module = std::move(module)](std::span<const Tensor<CPU>> inputs) {
			return module.Run(inputs);
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
			return module.Run(inputs);
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
			return module.Run(inputs);
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
			return module.Run(inputs);
		};
#endif
	}
#endif
} // namespace LiteNN::Training
