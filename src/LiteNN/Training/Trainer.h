#include <LiteNN/Graph.h>
#include <LiteNN/Optimizer/Loss.h>
#include <LiteNN/Optimizer/OptimizerUtils.h>
#include <LiteNN/Pass/AutogradPass.h>
#include <LiteNN/Runtime/Interpreter.h>
#include <LiteNN/Validation/GraphValidator.h>

#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef LITENN_TRAINING_TRAINER_H
#define LITENN_TRAINING_TRAINER_H

namespace LiteNN::Training
{
	struct TrainStepResult
	{
		std::vector<Tensor<CPU>> outputs;
		std::vector<Tensor<CPU>> backwardResults;
	};

	struct LossTrainStepResult
	{
		double loss{};
		std::vector<Tensor<CPU>> outputs;
		std::vector<Tensor<CPU>> backwardResults;
	};

	struct TrainerOptions
	{
		bool buildBackwardIfMissing{ true };
		bool storeVariableGradients{ true };
		bool zeroVariableGradientsBeforeBackward{ true };
	};

	template <typename OptimizerT>
	class CPUTrainer
	{
	public:
		CPUTrainer(Graph& graph, OptimizerT optimizer, TrainerOptions options = {})
		    : graph_(&graph), optimizer_(std::move(optimizer)), options_(options)
		{
			if (options_.buildBackwardIfMissing && !graph_->Backward())
			{
				AutogradPass autograd;
				autograd.Run(*graph_);
			}
			Validation::ValidateGraph(*graph_);
		}

		std::vector<Tensor<CPU>> Forward(std::span<const Tensor<CPU>> inputs)
		{
			return interpreter_.RunForward(*graph_, inputs);
		}

		TrainStepResult Step(std::span<const Tensor<CPU>> inputs, std::span<const Tensor<CPU>> outputGradients)
		{
			auto outputs = interpreter_.RunForward(*graph_, inputs);
			auto backwardResults = BackwardAndStep(inputs, outputGradients);
			return { std::move(outputs), std::move(backwardResults) };
		}

		LossTrainStepResult StepSoftmaxCrossEntropy(std::span<const Tensor<CPU>> inputs, std::size_t targetClass)
		{
			auto outputs = interpreter_.RunForward(*graph_, inputs);
			if (outputs.size() != 1)
			{
				throw std::runtime_error("StepSoftmaxCrossEntropy requires a graph with exactly one output");
			}

			auto lossGradient = LiteNN::Optimizer::SoftmaxCrossEntropyWithLogits(outputs[0], targetClass);
			std::vector<Tensor<CPU>> outputGradients;
			outputGradients.push_back(std::move(lossGradient.gradient));
			auto backwardResults = BackwardAndStep(inputs, outputGradients);
			return { lossGradient.loss, std::move(outputs), std::move(backwardResults) };
		}

		LossTrainStepResult StepSoftmaxCrossEntropyBatch(std::span<const Tensor<CPU>> inputs,
		                                                 std::span<const std::size_t> targetClasses)
		{
			auto outputs = interpreter_.RunForward(*graph_, inputs);
			if (outputs.size() != 1)
			{
				throw std::runtime_error("StepSoftmaxCrossEntropyBatch requires a graph with exactly one output");
			}

			auto lossGradient = LiteNN::Optimizer::SoftmaxCrossEntropyWithLogitsBatch(outputs[0], targetClasses);
			std::vector<Tensor<CPU>> outputGradients;
			outputGradients.push_back(std::move(lossGradient.gradient));
			auto backwardResults = BackwardAndStep(inputs, outputGradients);
			return { lossGradient.loss, std::move(outputs), std::move(backwardResults) };
		}

		void ZeroGradients()
		{
			LiteNN::Optimizer::ZeroGradients(*graph_);
		}

		OptimizerT& Optimizer()
		{
			return optimizer_;
		}

		const OptimizerT& Optimizer() const
		{
			return optimizer_;
		}

		Runtime::Interpreter<CPU>& Interpreter()
		{
			return interpreter_;
		}

	private:
		std::vector<Tensor<CPU>> BackwardAndStep(std::span<const Tensor<CPU>> inputs,
		                                         std::span<const Tensor<CPU>> outputGradients)
		{
			const auto outputCount = graph_->GetSubgraph(graph_->Forward()).Results().size();
			if (outputGradients.size() != outputCount)
			{
				throw std::runtime_error("CPUTrainer output gradient count does not match graph output count");
			}

			if (options_.zeroVariableGradientsBeforeBackward)
			{
				LiteNN::Optimizer::ZeroGradients(*graph_);
			}

			std::vector<Tensor<CPU>> backwardInputs;
			backwardInputs.reserve(inputs.size() + outputGradients.size());
			for (const auto& input : inputs)
			{
				backwardInputs.push_back(input);
			}
			for (const auto& gradient : outputGradients)
			{
				backwardInputs.push_back(gradient);
			}

			auto backwardResults = interpreter_.RunBackward(*graph_, backwardInputs);
			const auto inputGradientCount = inputs.size();
			if (options_.storeVariableGradients)
			{
				LiteNN::Optimizer::StoreVariableGradients(*graph_, backwardResults, inputGradientCount);
			}
			optimizer_.Step(*graph_, backwardResults, inputGradientCount);
			return backwardResults;
		}

		Graph* graph_;
		OptimizerT optimizer_;
		TrainerOptions options_;
		Runtime::Interpreter<CPU> interpreter_;
	};
} // namespace LiteNN::Training

#endif
