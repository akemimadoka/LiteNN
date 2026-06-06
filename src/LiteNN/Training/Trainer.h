#include <LiteNN/Graph.h>
#include <LiteNN/Optimizer/Loss.h>
#include <LiteNN/Optimizer/OptimizerUtils.h>
#include <LiteNN/Pass/AutogradPass.h>
#include <LiteNN/Runtime/Interpreter.h>
#include <LiteNN/Training/StateDict.h>
#include <LiteNN/Training/TrainStepAOTRunner.h>
#include <LiteNN/Training/TrainStepPlan.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <array>
#include <concepts>
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
		Trainer(ModelGraph& model, OptimizerT optimizer, TrainerOptions options = {}, D device = D{})
		    : model_(&model), graph_(&model.UnsafeMutableGraph()), optimizer_(std::move(optimizer)), options_(options),
		      device_(std::move(device))
		{
			if (options_.buildBackwardIfMissing && !graph_->Backward())
			{
				AutogradPass autograd;
				autograd.Run(*graph_);
			}
			Validation::ValidateGraph(*graph_);
			parameters_ = ParameterSet::BindGraph(*graph_);
			trainStepPlan_ = BuildTrainStepPlan(Detail::BuildExecutableModuleFromGraph(*graph_), options_.executionPolicy,
			                                    options_.aotBackendAvailable);
			ValidateTrainStepPlan(trainStepPlan_);
			if (trainStepPlan_.policy == TrainExecutionPolicy::AOT)
			{
				InitializeCompiledForwardRunner();
				InitializeCompiledOptimizerUpdateRunners();
			}
		}

		Trainer(ModelGraph& model, OptimizerT optimizer, D device)
		    : Trainer(model, std::move(optimizer), TrainerOptions{}, std::move(device))
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

		bool UsesCompiledOptimizerUpdateEntries() const noexcept
		{
			return compiledOptimizerUpdatesAvailable_;
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
			compiledBackward_ = CreateCompiledTrainBackwardRunner(trainStepPlan_.module.plan, device_);
		}

		void InitializeCompiledOptimizerUpdateRunners()
		{
			compiledOptimizerUpdates_.clear();
			compiledOptimizerUpdatesAvailable_ = false;
			if constexpr (std::same_as<D, CPU> && std::same_as<OptimizerT, Optimizer::SGD>)
			{
				if (optimizer_.Options().momentum != 0.0F)
				{
					return;
				}
				compiledOptimizerUpdates_.reserve(parameters_.Size());
				for (const auto& parameter : parameters_.Entries())
				{
					compiledOptimizerUpdates_.push_back(
					    CreateCompiledSGDUpdateRunner(parameter.type, optimizer_.Options(), device_));
				}
				compiledOptimizerUpdatesAvailable_ = compiledOptimizerUpdates_.size() == parameters_.Size();
			}
			else if constexpr (std::same_as<D, CPU> && std::same_as<OptimizerT, Optimizer::AdamW>)
			{
				optimizer_.EnsureState(parameters_);
				compiledOptimizerUpdatesAvailable_ = parameters_.Size() > 0;
			}
		}

		void EnsureCompiledTrainStepRunnerAvailable() const
		{
			if (trainStepPlan_.policy != TrainExecutionPolicy::AOT)
			{
				return;
			}
			if (!compiledForward_ || !compiledBackward_)
			{
				throw std::runtime_error("Trainer AOT train-step runners were not initialized");
			}
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

			auto backwardResults = RunBackward(backwardInputs);
			const auto inputGradientCount = inputs.size();
			auto cpuBackwardResults = CopyToCPU(backwardResults);
			if (options_.storeVariableGradients)
			{
				LiteNN::Optimizer::StoreVariableGradients(parameters_, cpuBackwardResults, inputGradientCount);
			}
			if (!RunCompiledOptimizerUpdates(cpuBackwardResults, inputGradientCount))
			{
				optimizer_.Step(parameters_, cpuBackwardResults, inputGradientCount);
			}
			if (trainStepPlan_.policy == TrainExecutionPolicy::AOT)
			{
				InitializeCompiledForwardRunner();
			}
			return backwardResults;
		}

		bool RunCompiledOptimizerUpdates(std::span<const Tensor<CPU>> backwardResults, std::size_t inputGradientCount)
		{
			if (!compiledOptimizerUpdatesAvailable_)
			{
				return false;
			}
			Optimizer::Detail::ValidateBackwardResults(parameters_, backwardResults, inputGradientCount);
			if constexpr (std::same_as<D, CPU> && std::same_as<OptimizerT, Optimizer::SGD>)
			{
				for (std::size_t parameterIndex = 0; parameterIndex < parameters_.Size(); ++parameterIndex)
				{
					auto& parameter = parameters_[parameterIndex].Parameter();
					if (!parameter.CurDevice().template Is<CPU>())
					{
						return false;
					}
					const auto parameterCPU = parameter.CopyToDevice(CPU{});
					const auto& gradient =
					    Optimizer::Detail::VariableGradient(backwardResults, inputGradientCount, parameterIndex);
					Optimizer::Detail::ValidateVariableGradient(parameter, gradient, parameterIndex);
					std::array<Tensor<CPU>, 2> updateInputs = { parameterCPU, gradient };
					auto updateOutputs = compiledOptimizerUpdates_[parameterIndex](updateInputs);
					if (updateOutputs.empty())
					{
						throw std::runtime_error("Trainer AOT optimizer update runner returned no outputs");
					}
					DeviceTraits<PolymorphicDevice>::CopyFromCPU(parameter.CurDevice(), parameter.DType(),
					                                             parameter.UnsafeRawData(), updateOutputs[0].DType(),
					                                             updateOutputs[0].UnsafeRawData(),
					                                             updateOutputs[0].NumElements());
				}
				return true;
			}
			else if constexpr (std::same_as<D, CPU> && std::same_as<OptimizerT, Optimizer::AdamW>)
			{
				optimizer_.EnsureState(parameters_);
				const auto step = optimizer_.AdvanceStep();
				for (std::size_t parameterIndex = 0; parameterIndex < parameters_.Size(); ++parameterIndex)
				{
					auto& parameter = parameters_[parameterIndex].Parameter();
					if (!parameter.CurDevice().template Is<CPU>())
					{
						return false;
					}
					const auto parameterCPU = parameter.CopyToDevice(CPU{});
					const auto& gradient =
					    Optimizer::Detail::VariableGradient(backwardResults, inputGradientCount, parameterIndex);
					Optimizer::Detail::ValidateVariableGradient(parameter, gradient, parameterIndex);
					std::array<Tensor<CPU>, 4> updateInputs = { parameterCPU, gradient,
						                                    optimizer_.FirstMoment(parameterIndex),
						                                    optimizer_.SecondMoment(parameterIndex) };
					auto updateRunner =
					    CreateCompiledAdamWUpdateRunner(parameters_[parameterIndex].type, optimizer_.Options(), step,
					                                    device_);
					auto updateOutputs = updateRunner(updateInputs);
					if (updateOutputs.size() != 3)
					{
						throw std::runtime_error("Trainer AOT AdamW update runner returned an unexpected output count");
					}
					DeviceTraits<PolymorphicDevice>::CopyFromCPU(parameter.CurDevice(), parameter.DType(),
					                                             parameter.UnsafeRawData(), updateOutputs[0].DType(),
					                                             updateOutputs[0].UnsafeRawData(),
					                                             updateOutputs[0].NumElements());
					optimizer_.FirstMoment(parameterIndex) = std::move(updateOutputs[1]);
					optimizer_.SecondMoment(parameterIndex) = std::move(updateOutputs[2]);
				}
				return true;
			}
			else
			{
				return false;
			}
		}

		std::vector<Tensor<D>> RunBackward(std::span<const Tensor<D>> inputs)
		{
			if (trainStepPlan_.policy != TrainExecutionPolicy::AOT)
			{
				return interpreter_.RunBackward(trainStepPlan_.module.plan, inputs, device_);
			}
			if (!compiledBackward_)
			{
				throw std::runtime_error("Trainer AOT backward runner was not initialized");
			}
			return compiledBackward_(inputs);
		}

		ModelGraph* model_;
		Graph* graph_;
		ParameterSet parameters_;
		OptimizerT optimizer_;
		TrainerOptions options_;
		D device_;
		Runtime::Interpreter<D> interpreter_;
		TrainStepPlan trainStepPlan_;
		CompiledForwardRunner<D> compiledForward_;
		CompiledBackwardRunner<D> compiledBackward_;
		std::vector<CompiledOptimizerUpdateRunner<CPU>> compiledOptimizerUpdates_;
		bool compiledOptimizerUpdatesAvailable_{};
	};
} // namespace LiteNN::Training

#endif
