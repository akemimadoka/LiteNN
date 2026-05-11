#include <LiteNN/Optimizer/OptimizerUtils.h>

#include <stdexcept>
#include <vector>

#ifndef LITENN_OPTIMIZER_SGD_H
#define LITENN_OPTIMIZER_SGD_H

namespace LiteNN::Optimizer
{
	struct SGDOptions
	{
		float learningRate{ 1.0e-3f };
		float momentum{ 0.0f };
		float weightDecay{ 0.0f };
		bool nesterov{ false };
	};

	class SGD
	{
	public:
		explicit SGD(float learningRate) : options_{ .learningRate = learningRate }
		{
			ValidateOptions();
		}

		explicit SGD(SGDOptions options) : options_(options)
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
			if (options_.momentum != 0.0f)
			{
				EnsureVelocity(graph);
			}

			for (std::size_t variableIndex = 0; variableIndex < graph.VariableCount(); ++variableIndex)
			{
				auto& variable = graph.GetVariable(variableIndex)->Data();
				const auto& gradient = Detail::VariableGradient(backwardResults, inputGradientCount, variableIndex);
				Detail::ValidateVariableGradient(variable, gradient, variableIndex);

				auto* data = static_cast<float*>(variable.RawData());
				const auto* grad = static_cast<const float*>(gradient.RawData());
				const auto elementCount = variable.NumElements();

				if (options_.momentum == 0.0f)
				{
					for (std::size_t i = 0; i < elementCount; ++i)
					{
						const auto regularizedGradient = grad[i] + options_.weightDecay * data[i];
						data[i] -= options_.learningRate * regularizedGradient;
					}
					continue;
				}

				auto& velocity = velocity_[variableIndex];
				for (std::size_t i = 0; i < elementCount; ++i)
				{
					const auto regularizedGradient = grad[i] + options_.weightDecay * data[i];
					velocity[i] = options_.momentum * velocity[i] + regularizedGradient;
					const auto update = options_.nesterov ? regularizedGradient + options_.momentum * velocity[i] : velocity[i];
					data[i] -= options_.learningRate * update;
				}
			}
		}

		void Reset()
		{
			velocity_.clear();
		}

	private:
		void ValidateOptions() const
		{
			if (!(options_.learningRate > 0.0f))
			{
				throw std::runtime_error("SGD learningRate must be greater than zero");
			}
			if (options_.momentum < 0.0f)
			{
				throw std::runtime_error("SGD momentum must be non-negative");
			}
			if (options_.nesterov && options_.momentum == 0.0f)
			{
				throw std::runtime_error("SGD nesterov requires momentum");
			}
		}

		void EnsureVelocity(const Graph& graph)
		{
			velocity_.resize(graph.VariableCount());
			for (std::size_t variableIndex = 0; variableIndex < graph.VariableCount(); ++variableIndex)
			{
				const auto elementCount = graph.GetVariable(variableIndex)->Data().NumElements();
				if (velocity_[variableIndex].size() != elementCount)
				{
					velocity_[variableIndex].assign(elementCount, 0.0f);
				}
			}
		}

		SGDOptions options_;
		std::vector<std::vector<float>> velocity_;
	};
} // namespace LiteNN::Optimizer

#endif
