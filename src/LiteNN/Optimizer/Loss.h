#include <LiteNN/Tensor.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <span>
#include <stdexcept>
#include <vector>

#ifndef LITENN_OPTIMIZER_LOSS_H
#define LITENN_OPTIMIZER_LOSS_H

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

	inline LossGradient SoftmaxCrossEntropyWithLogitsBatch(const Tensor<CPU>& logits,
	                                                       std::span<const std::size_t> targetClasses)
	{
		if (logits.DType() != DataType::Float32 || logits.Shape().Dims.size() != 2)
		{
			throw std::runtime_error("SoftmaxCrossEntropyWithLogitsBatch expects a Float32 [batch, classes] logits tensor");
		}

		const auto batchSize = logits.Shape().Dims[0];
		const auto classCount = logits.Shape().Dims[1];
		if (batchSize == 0 || classCount == 0)
		{
			throw std::runtime_error("SoftmaxCrossEntropyWithLogitsBatch expects non-empty batch and class dimensions");
		}
		if (targetClasses.size() != batchSize)
		{
			throw std::runtime_error("SoftmaxCrossEntropyWithLogitsBatch target count does not match batch size");
		}

		const auto* data = static_cast<const float*>(logits.RawData());
		std::vector<float> gradient(logits.NumElements());
		double totalLoss = 0.0;
		const auto invBatch = 1.0f / static_cast<float>(batchSize);

		for (std::size_t row = 0; row < batchSize; ++row)
		{
			const auto targetClass = targetClasses[row];
			if (targetClass >= classCount)
			{
				throw std::runtime_error("SoftmaxCrossEntropyWithLogitsBatch target class is out of range");
			}

			const auto rowOffset = row * classCount;
			const auto* rowData = data + rowOffset;
			const auto maxLogit = *std::max_element(rowData, rowData + classCount);

			double sumExp = 0.0;
			for (std::size_t col = 0; col < classCount; ++col)
			{
				sumExp += std::exp(static_cast<double>(rowData[col] - maxLogit));
			}

			for (std::size_t col = 0; col < classCount; ++col)
			{
				const auto probability = static_cast<float>(std::exp(static_cast<double>(rowData[col] - maxLogit)) / sumExp);
				gradient[rowOffset + col] = probability * invBatch;
			}

			const auto targetProbability =
			    std::max(static_cast<double>(gradient[rowOffset + targetClass] / invBatch), 1.0e-12);
			totalLoss += -std::log(targetProbability);
			gradient[rowOffset + targetClass] -= invBatch;
		}

		return { totalLoss / static_cast<double>(batchSize), MakeFloatTensor(gradient, logits.Shape()) };
	}
} // namespace LiteNN::Optimizer

#endif
