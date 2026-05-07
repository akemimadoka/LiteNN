#ifndef LITENN_OPTIMIZER_LOSS_H
#define LITENN_OPTIMIZER_LOSS_H

#include <LiteNN/Tensor.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

namespace LiteNN::Optimizer
{
	struct LossGradient
	{
		double loss;
		Tensor<CPU> gradient;
	};

	inline Tensor<CPU> MakeFloatTensor(std::span<const float> values, ShapeView shape)
	{
		if (values.size() != shape.NumElements())
		{
			throw std::runtime_error("Tensor initializer element count does not match shape");
		}
		Tensor<CPU> tensor(Uninitialized, shape, DataType::Float32);
		std::memcpy(tensor.RawData(), values.data(), values.size() * sizeof(float));
		return tensor;
	}

	inline LossGradient SoftmaxCrossEntropyWithLogits(const Tensor<CPU>& logits, std::size_t targetClass)
	{
		if (logits.DType() != DataType::Float32 || logits.NumElements() == 0)
		{
			throw std::runtime_error("SoftmaxCrossEntropyWithLogits expects a non-empty Float32 logits tensor");
		}
		if (targetClass >= logits.NumElements())
		{
			throw std::runtime_error("SoftmaxCrossEntropyWithLogits target class is out of range");
		}

		const auto* data = static_cast<const float*>(logits.RawData());
		const auto maxLogit = *std::max_element(data, data + logits.NumElements());

		double sumExp = 0.0;
		for (std::size_t i = 0; i < logits.NumElements(); ++i)
		{
			sumExp += std::exp(static_cast<double>(data[i] - maxLogit));
		}

		std::vector<float> gradient(logits.NumElements());
		for (std::size_t i = 0; i < logits.NumElements(); ++i)
		{
			gradient[i] = static_cast<float>(std::exp(static_cast<double>(data[i] - maxLogit)) / sumExp);
		}

		const auto targetProbability = std::max(static_cast<double>(gradient[targetClass]), 1.0e-12);
		const auto loss = -std::log(targetProbability);
		gradient[targetClass] -= 1.0f;
		return { loss, MakeFloatTensor(gradient, logits.Shape()) };
	}
} // namespace LiteNN::Optimizer

#endif
