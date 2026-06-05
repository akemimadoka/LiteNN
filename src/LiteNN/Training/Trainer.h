#include <LiteNN/Graph.h>
#include <LiteNN/Optimizer/Loss.h>
#include <LiteNN/Optimizer/OptimizerUtils.h>
#include <LiteNN/Pass/AutogradPass.h>
#include <LiteNN/Runtime/Interpreter.h>
#include <LiteNN/Training/StateDict.h>
#include <LiteNN/Training/TrainStepAOTRunner.h>
#include <LiteNN/Training/TrainStepPlan.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef LITENN_TRAINING_TRAINER_H
#define LITENN_TRAINING_TRAINER_H

namespace LiteNN::Training
{
	template <Device D>
	struct BasicTrainStepResult
	{
		std::vector<Tensor<D>> outputs;
		std::vector<Tensor<D>> backwardResults;
	};

	template <Device D>
	struct BasicLossTrainStepResult
	{
		double loss{};
		std::vector<Tensor<D>> outputs;
		std::vector<Tensor<D>> backwardResults;
	};

	using TrainStepResult = BasicTrainStepResult<CPU>;
	using LossTrainStepResult = BasicLossTrainStepResult<CPU>;

	struct TrainerOptions
	{
		bool buildBackwardIfMissing{ true };
		bool storeVariableGradients{ true };
		bool zeroVariableGradientsBeforeBackward{ true };
		TrainExecutionPolicy executionPolicy{ TrainExecutionPolicy::Auto };
		bool aotBackendAvailable{};
	};

	template <Device D, typename OptimizerT>
	class Trainer
	{
	public:
		Trainer(Graph& graph, OptimizerT optimizer, TrainerOptions options = {}, D device = D{})
		    : graph_(&graph), optimizer_(std::move(optimizer)), options_(options), device_(std::move(device))
		{
			if (options_.buildBackwardIfMissing && !graph_->Backward())
			{
				AutogradPass autograd;
				autograd.Run(*graph_);
			}
			Validation::ValidateGraph(*graph_);
			parameters_ = ParameterSet::BindGraph(*graph_);
			trainStepPlan_ = BuildTrainStepPlan(BuildExecutableModule(*graph_), options_.executionPolicy,
			                                    options_.aotBackendAvailable);
			ValidateTrainStepPlan(trainStepPlan_);
			if (trainStepPlan_.policy == TrainExecutionPolicy::AOT)
			{
				InitializeCompiledForwardRunner();
			}
		}

		Trainer(Graph& graph, OptimizerT optimizer, D device)
		    : Trainer(graph, std::move(optimizer), TrainerOptions{}, std::move(device))
		{
		}

		std::vector<Tensor<D>> Forward(std::span<const Tensor<D>> inputs)
		{
			return RunForward(inputs);
		}

		BasicTrainStepResult<D> Step(std::span<const Tensor<D>> inputs, std::span<const Tensor<D>> outputGradients)
		{
			EnsureCompiledTrainStepRunnerAvailable();
			auto outputs = RunForward(inputs);
			auto backwardResults = BackwardAndStep(inputs, outputGradients);
			return { std::move(outputs), std::move(backwardResults) };
		}

		BasicLossTrainStepResult<D> StepSoftmaxCrossEntropy(std::span<const Tensor<D>> inputs, std::size_t targetClass)
		{
			EnsureCompiledTrainStepRunnerAvailable();
			auto outputs = RunForward(inputs);
			if (outputs.size() != 1)
			{
				throw std::runtime_error("StepSoftmaxCrossEntropy requires a graph with exactly one output");
			}

			auto lossGradient =
			    LiteNN::Optimizer::SoftmaxCrossEntropyWithLogits(outputs[0].CopyToDevice(CPU{}), targetClass);
			std::vector<Tensor<D>> outputGradients;
			outputGradients.push_back(lossGradient.gradient.CopyToDevice(device_));
			auto backwardResults = BackwardAndStep(inputs, outputGradients);
			return { lossGradient.loss, std::move(outputs), std::move(backwardResults) };
		}

		BasicLossTrainStepResult<D> StepSoftmaxCrossEntropyBatch(std::span<const Tensor<D>> inputs,
		                                                         std::span<const std::size_t> targetClasses)
		{
			EnsureCompiledTrainStepRunnerAvailable();
			auto outputs = RunForward(inputs);
			if (outputs.size() != 1)
			{
				throw std::runtime_error("StepSoftmaxCrossEntropyBatch requires a graph with exactly one output");
			}

			auto lossGradient =
			    LiteNN::Optimizer::SoftmaxCrossEntropyWithLogitsBatch(outputs[0].CopyToDevice(CPU{}), targetClasses);
			std::vector<Tensor<D>> outputGradients;
			outputGradients.push_back(lossGradient.gradient.CopyToDevice(device_));
			auto backwardResults = BackwardAndStep(inputs, outputGradients);
			return { lossGradient.loss, std::move(outputs), std::move(backwardResults) };
		}

		void ZeroGradients()
		{
			LiteNN::Optimizer::ZeroGradients(parameters_);
		}

		ParameterSet& Parameters()
		{
			return parameters_;
		}

		const ParameterSet& Parameters() const
		{
			return parameters_;
		}

		StateDict SaveStateDict() const
		{
			return LiteNN::Training::SaveStateDict(parameters_);
		}

		void LoadStateDict(const StateDict& state)
		{
			LiteNN::Training::LoadStateDict(parameters_, state);
		}

		OptimizerT& Optimizer()
		{
			return optimizer_;
		}

		const OptimizerT& Optimizer() const
		{
			return optimizer_;
		}

		Runtime::Interpreter<D>& Interpreter()
		{
			return interpreter_;
		}

		const Runtime::Interpreter<D>& Interpreter() const
		{
			return interpreter_;
		}

		const TrainStepPlan& Plan() const
		{
			return trainStepPlan_;
		}

		TrainExecutionPolicy ExecutionPolicy() const
		{
			return trainStepPlan_.policy;
		}

		D& Device()
		{
			return device_;
		}

		const D& Device() const
		{
			return device_;
		}

	private:
		void InitializeCompiledForwardRunner()
		{
			compiledForward_ = CreateCompiledTrainForwardRunner(trainStepPlan_.module.plan, device_);
		}

		void EnsureCompiledTrainStepRunnerAvailable() const
		{
			if (trainStepPlan_.policy != TrainExecutionPolicy::AOT)
			{
				return;
			}
			throw std::runtime_error(
			    "Trainer AOT forward runner is available, but compiled backward/update train-step execution is not "
			    "wired yet; use Forward() for compiled inference or TrainExecutionPolicy::Interpreter for training");
		}

		std::vector<Tensor<D>> RunForward(std::span<const Tensor<D>> inputs)
		{
			if (trainStepPlan_.policy != TrainExecutionPolicy::AOT)
			{
				return interpreter_.RunForward(trainStepPlan_.module.plan, inputs, device_);
			}
			if (!compiledForward_)
			{
				throw std::runtime_error("Trainer AOT forward runner was not initialized");
			}
			return compiledForward_(inputs);
		}

		static std::vector<Tensor<CPU>> CopyToCPU(std::span<const Tensor<D>> tensors)
		{
			std::vector<Tensor<CPU>> cpuTensors;
			cpuTensors.reserve(tensors.size());
			for (const auto& tensor : tensors)
			{
				cpuTensors.push_back(tensor.CopyToDevice(CPU{}));
			}
			return cpuTensors;
		}

		std::vector<Tensor<D>> BackwardAndStep(std::span<const Tensor<D>> inputs,
		                                       std::span<const Tensor<D>> outputGradients)
		{
			const auto outputCount = graph_->GetSubgraph(graph_->Forward()).Results().size();
			if (outputGradients.size() != outputCount)
			{
				throw std::runtime_error("Trainer output gradient count does not match graph output count");
			}

			if (options_.zeroVariableGradientsBeforeBackward)
			{
				LiteNN::Optimizer::ZeroGradients(parameters_);
			}

			std::vector<Tensor<D>> backwardInputs;
			backwardInputs.reserve(inputs.size() + outputGradients.size());
			for (const auto& input : inputs)
			{
				backwardInputs.push_back(input);
			}
			for (const auto& gradient : outputGradients)
			{
				backwardInputs.push_back(gradient);
			}

			auto backwardResults = interpreter_.RunBackward(trainStepPlan_.module.plan, backwardInputs, device_);
			const auto inputGradientCount = inputs.size();
			auto cpuBackwardResults = CopyToCPU(backwardResults);
			if (options_.storeVariableGradients)
			{
				LiteNN::Optimizer::StoreVariableGradients(parameters_, cpuBackwardResults, inputGradientCount);
			}
			optimizer_.Step(parameters_, cpuBackwardResults, inputGradientCount);
			return backwardResults;
		}

		Graph* graph_;
		ParameterSet parameters_;
		OptimizerT optimizer_;
		TrainerOptions options_;
		D device_;
		Runtime::Interpreter<D> interpreter_;
		TrainStepPlan trainStepPlan_;
		CompiledForwardRunner<D> compiledForward_;
	};
} // namespace LiteNN::Training

#endif
