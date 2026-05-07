#ifndef LITENN_OPTIMIZER_ADAM_H
#define LITENN_OPTIMIZER_ADAM_H

#include <LiteNN/Optimizer/OptimizerUtils.h>

#include <cmath>
#include <stdexcept>
#include <vector>

namespace LiteNN::Optimizer
{
	struct AdamOptions
	{
		float learningRate{ 1.0e-3f };
		float beta1{ 0.9f };
		float beta2{ 0.999f };
		float epsilon{ 1.0e-8f };
		float weightDecay{ 0.0f };
	};

	class Adam
	{
	public:
		explicit Adam(AdamOptions options = {}) : options_(options)
		{
			ValidateOptions();
		}

		void Step(Graph& graph, std::span<const Tensor<CPU>> backwardResults)
		{
			Step(graph, backwardResults, Detail::InferInputGradientCount(graph));
		}

		void Step(Graph& graph, std::span<const Tensor<CPU>> backwardResults, std::size_t inputGradientCount)
		{
			Detail::ValidateBackwardResults(graph, backwardResults, inputGradientCount);
			EnsureState(graph);
			++step_;

			const auto beta1Correction = 1.0f - std::pow(options_.beta1, static_cast<float>(step_));
			const auto beta2Correction = 1.0f - std::pow(options_.beta2, static_cast<float>(step_));

			for (std::size_t variableIndex = 0; variableIndex < graph.VariableCount(); ++variableIndex)
			{
				auto& variable = graph.GetVariable(variableIndex)->Data();
				const auto& gradient = Detail::VariableGradient(backwardResults, inputGradientCount, variableIndex);
				Detail::ValidateVariableGradient(variable, gradient, variableIndex);

				auto* data = static_cast<float*>(variable.RawData());
				const auto* grad = static_cast<const float*>(gradient.RawData());
				auto& firstMoment = firstMoment_[variableIndex];
				auto& secondMoment = secondMoment_[variableIndex];

				for (std::size_t i = 0; i < variable.NumElements(); ++i)
				{
					const auto regularizedGradient = grad[i] + options_.weightDecay * data[i];
					firstMoment[i] = options_.beta1 * firstMoment[i] + (1.0f - options_.beta1) * regularizedGradient;
					secondMoment[i] =
					    options_.beta2 * secondMoment[i] + (1.0f - options_.beta2) * regularizedGradient * regularizedGradient;

					const auto mHat = firstMoment[i] / beta1Correction;
					const auto vHat = secondMoment[i] / beta2Correction;
					data[i] -= options_.learningRate * mHat / (std::sqrt(vHat) + options_.epsilon);
				}
			}
		}

		void Reset()
		{
			step_ = 0;
			firstMoment_.clear();
			secondMoment_.clear();
		}

	private:
		void ValidateOptions() const
		{
			if (!(options_.learningRate > 0.0f))
			{
				throw std::runtime_error("Adam learningRate must be greater than zero");
			}
			if (!(options_.beta1 >= 0.0f && options_.beta1 < 1.0f && options_.beta2 >= 0.0f && options_.beta2 < 1.0f))
			{
				throw std::runtime_error("Adam beta values must be in [0, 1)");
			}
			if (!(options_.epsilon > 0.0f))
			{
				throw std::runtime_error("Adam epsilon must be greater than zero");
			}
		}

		void EnsureState(const Graph& graph)
		{
			firstMoment_.resize(graph.VariableCount());
			secondMoment_.resize(graph.VariableCount());
			for (std::size_t variableIndex = 0; variableIndex < graph.VariableCount(); ++variableIndex)
			{
				const auto elementCount = graph.GetVariable(variableIndex)->Data().NumElements();
				if (firstMoment_[variableIndex].size() != elementCount)
				{
					firstMoment_[variableIndex].assign(elementCount, 0.0f);
					secondMoment_[variableIndex].assign(elementCount, 0.0f);
				}
			}
		}

		AdamOptions options_;
		std::size_t step_{};
		std::vector<std::vector<float>> firstMoment_;
		std::vector<std::vector<float>> secondMoment_;
	};
} // namespace LiteNN::Optimizer

#endif
