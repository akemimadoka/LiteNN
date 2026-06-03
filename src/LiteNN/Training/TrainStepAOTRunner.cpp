#include <LiteNN/Training/TrainStepAOTRunner.h>

#ifdef LITENN_ENABLE_MLIR
#include <LiteNN/Compiler/CompiledModule.h>
#endif

#include <utility>

namespace LiteNN::Training
{
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
#endif
} // namespace LiteNN::Training
