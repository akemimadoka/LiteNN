#ifndef LITENN_OPTIMIZER_ADAMW_H
#define LITENN_OPTIMIZER_ADAMW_H

#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/Optimizer/OptimizerUtils.h>

#include <optional>
#include <stdexcept>
#include <vector>

namespace LiteNN::Optimizer
{
	struct AdamWOptions
	{
		float learningRate{ 1.0e-3f };
		float beta1{ 0.9f };
		float beta2{ 0.999f };
		float epsilon{ 1.0e-8f };
		float weightDecay{ 1.0e-2f };
	};

	class AdamW
	{
	public:
		explicit AdamW(AdamWOptions options = {}) : options_(options)
		{
			ValidateOptions();
		}

		const AdamWOptions& Options() const noexcept
		{
			return options_;
		}

		std::size_t StepIndex() const noexcept
		{
			return step_;
		}

		void Step(Training::ParameterSet& parameters, std::span<const Tensor<CPU>> backwardResults,
		          std::size_t inputGradientCount)
		{
			Detail::ValidateBackwardResults(parameters, backwardResults, inputGradientCount);
			EnsureState(parameters);
			AdvanceStep();

			for (std::size_t variableIndex = 0; variableIndex < parameters.Size(); ++variableIndex)
			{
				auto& variable = parameters[variableIndex].Parameter();
				const auto& gradient = Detail::VariableGradient(backwardResults, inputGradientCount, variableIndex);
				Detail::ValidateVariableGradient(variable, gradient, variableIndex);

				auto cpuResults = LiteNN::Detail::EvalAdamWStep(variable.CopyToDevice(CPU{}), gradient,
				                                                *firstMoment_[variableIndex],
				                                                *secondMoment_[variableIndex],
				                                                options_.learningRate, options_.beta1,
				                                                options_.beta2, options_.epsilon,
				                                                options_.weightDecay, step_);
				DeviceTraits<PolymorphicDevice>::CopyFromCPU(variable.CurDevice(), variable.DType(), variable.RawData(),
				                                             cpuResults[0].DType(), cpuResults[0].RawData(),
				                                             cpuResults[0].NumElements());
				firstMoment_[variableIndex] = std::move(cpuResults[1]);
				secondMoment_[variableIndex] = std::move(cpuResults[2]);
			}
		}

		void EnsureState(const Training::ParameterSet& parameters)
		{
			firstMoment_.resize(parameters.Size());
			secondMoment_.resize(parameters.Size());
			for (std::size_t variableIndex = 0; variableIndex < parameters.Size(); ++variableIndex)
			{
				const auto& parameter = parameters[variableIndex].Parameter();
				if (parameter.DType() != DataType::Float32)
				{
					throw std::runtime_error("AdamW currently supports Float32 parameters only");
				}
				if (!firstMoment_[variableIndex] || !secondMoment_[variableIndex] ||
				    firstMoment_[variableIndex]->Shape() != parameter.Shape() ||
				    secondMoment_[variableIndex]->Shape() != parameter.Shape())
				{
					firstMoment_[variableIndex].emplace(parameter.Shape(), DataType::Float32);
					secondMoment_[variableIndex].emplace(parameter.Shape(), DataType::Float32);
				}
			}
		}

		std::size_t AdvanceStep() noexcept
		{
			return ++step_;
		}

		Tensor<CPU>& FirstMoment(std::size_t parameterIndex)
		{
			return *firstMoment_.at(parameterIndex);
		}

		Tensor<CPU>& SecondMoment(std::size_t parameterIndex)
		{
			return *secondMoment_.at(parameterIndex);
		}

		const Tensor<CPU>& FirstMoment(std::size_t parameterIndex) const
		{
			return *firstMoment_.at(parameterIndex);
		}

		const Tensor<CPU>& SecondMoment(std::size_t parameterIndex) const
		{
			return *secondMoment_.at(parameterIndex);
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
				throw std::runtime_error("AdamW learningRate must be greater than zero");
			}
			if (!(options_.beta1 >= 0.0f && options_.beta1 < 1.0f && options_.beta2 >= 0.0f &&
			      options_.beta2 < 1.0f))
			{
				throw std::runtime_error("AdamW beta values must be in [0, 1)");
			}
			if (!(options_.epsilon > 0.0f))
			{
				throw std::runtime_error("AdamW epsilon must be greater than zero");
			}
			if (!(options_.weightDecay >= 0.0f))
			{
				throw std::runtime_error("AdamW weightDecay must be non-negative");
			}
		}

		AdamWOptions options_;
		std::size_t step_{};
		std::vector<std::optional<Tensor<CPU>>> firstMoment_;
		std::vector<std::optional<Tensor<CPU>>> secondMoment_;
	};
} // namespace LiteNN::Optimizer

#endif
