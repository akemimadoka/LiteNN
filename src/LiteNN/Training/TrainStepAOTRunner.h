#ifndef LITENN_TRAINING_TRAIN_STEP_AOT_RUNNER_H
#define LITENN_TRAINING_TRAIN_STEP_AOT_RUNNER_H

#include <LiteNN/Device.h>
#ifdef LITENN_ENABLE_CUDA
#include <LiteNN/Device/CUDA.h>
#endif
#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/Optimizer/SGD.h>
#include <LiteNN/Tensor.h>

#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

namespace LiteNN::Training
{
	template <Device D>
	using CompiledForwardRunner = std::function<std::vector<Tensor<D>>(std::span<const Tensor<D>>)>;

	template <Device D>
	using CompiledBackwardRunner = std::function<std::vector<Tensor<D>>(std::span<const Tensor<D>>)>;

	template <Device D>
	using CompiledOptimizerUpdateRunner = std::function<std::vector<Tensor<D>>(std::span<const Tensor<D>>)>;

	/// Creates the compiled forward half of the vNext train-step contract.
	template <Device D>
	CompiledForwardRunner<D> CreateCompiledTrainForwardRunner(const ExecutablePlan&, D)
	{
		throw std::runtime_error("Trainer AOT forward runner is not available for this device");
	}

	/// Creates the compiled backward half of the vNext train-step contract.
	///
	/// Optimizer/loss/update execution remains owned by the named multi-entry train-step artifact ABI; this runner
	/// intentionally compiles the explicit backward graph only.
	template <Device D>
	CompiledBackwardRunner<D> CreateCompiledTrainBackwardRunner(const ExecutablePlan&, D)
	{
		throw std::runtime_error("Trainer AOT backward runner is not available for this device");
	}

	template <Device D>
	CompiledOptimizerUpdateRunner<D> CreateCompiledSGDUpdateRunner(const TensorType&, Optimizer::SGDOptions, D)
	{
		throw std::runtime_error("Trainer AOT SGD update runner is not available for this device");
	}

	template <>
	CompiledForwardRunner<CPU> CreateCompiledTrainForwardRunner(const ExecutablePlan& plan, CPU device);
	template <>
	CompiledBackwardRunner<CPU> CreateCompiledTrainBackwardRunner(const ExecutablePlan& plan, CPU device);
	template <>
	CompiledOptimizerUpdateRunner<CPU> CreateCompiledSGDUpdateRunner(const TensorType& parameterType,
	                                                                 Optimizer::SGDOptions options, CPU device);

#ifdef LITENN_ENABLE_CUDA
	template <>
	CompiledForwardRunner<CUDA> CreateCompiledTrainForwardRunner(const ExecutablePlan& plan, CUDA device);
	template <>
	CompiledBackwardRunner<CUDA> CreateCompiledTrainBackwardRunner(const ExecutablePlan& plan, CUDA device);
#endif
} // namespace LiteNN::Training

#endif
