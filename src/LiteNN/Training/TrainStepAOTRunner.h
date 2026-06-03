#ifndef LITENN_TRAINING_TRAIN_STEP_AOT_RUNNER_H
#define LITENN_TRAINING_TRAIN_STEP_AOT_RUNNER_H

#include <LiteNN/Device.h>
#ifdef LITENN_ENABLE_CUDA
#include <LiteNN/Device/CUDA.h>
#endif
#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/Tensor.h>

#include <functional>
#include <span>
#include <stdexcept>
#include <vector>

namespace LiteNN::Training
{
	template <Device D>
	using CompiledForwardRunner = std::function<std::vector<Tensor<D>>(std::span<const Tensor<D>>)>;

	/// Creates the compiled forward half of the vNext train-step contract.
	///
	/// Full compiled backward/update execution remains owned by the G13 AOT-training ABI because it needs
	/// mutable parameter bindings, saved-activation/tape bindings, and named multi-entry artifacts.
	template <Device D>
	CompiledForwardRunner<D> CreateCompiledTrainForwardRunner(const ExecutablePlan&, D)
	{
		throw std::runtime_error("Trainer AOT forward runner is not available for this device");
	}

	template <>
	CompiledForwardRunner<CPU> CreateCompiledTrainForwardRunner(const ExecutablePlan& plan, CPU device);

#ifdef LITENN_ENABLE_CUDA
	template <>
	CompiledForwardRunner<CUDA> CreateCompiledTrainForwardRunner(const ExecutablePlan& plan, CUDA device);
#endif
} // namespace LiteNN::Training

#endif
