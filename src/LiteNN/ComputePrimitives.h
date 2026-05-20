#ifndef LITENN_COMPUTEPRIMITIVES_H
#define LITENN_COMPUTEPRIMITIVES_H

#include <LiteNN/DataMovement.h>

#include <algorithm>
#include <cmath>
#include <format>
#include <limits>
#include <optional>
#include <ranges>
#include <span>
#include <stdexcept>
#include <vector>

namespace LiteNN::Detail
{
	inline std::size_t Product(std::span<const std::size_t> shape)
	{
		auto result = 1uz;
		for (const auto dim : shape)
		{
			result *= dim;
		}
		return result;
	}

	inline std::vector<std::size_t> BroadcastShapesTrailing(std::span<const std::size_t> lhs,
	                                                       std::span<const std::size_t> rhs,
	                                                       std::string_view label)
	{
		const auto rank = std::max(lhs.size(), rhs.size());
		std::vector<std::size_t> result(rank, 1uz);
		for (auto outDim = 0uz; outDim < rank; ++outDim)
		{
			const auto lhsDim = outDim + lhs.size() >= rank ? lhs[outDim + lhs.size() - rank] : 1uz;
			const auto rhsDim = outDim + rhs.size() >= rank ? rhs[outDim + rhs.size() - rank] : 1uz;
			if (lhsDim != rhsDim && lhsDim != 1 && rhsDim != 1)
			{
				throw std::runtime_error(std::format("{} cannot broadcast shapes", label));
			}
			result[outDim] = std::max(lhsDim, rhsDim);
		}
		return result;
	}

	inline std::vector<std::size_t> BatchMatMulOutputShape(std::span<const std::size_t> lhsShape,
	                                                       std::span<const std::size_t> rhsShape)
	{
		if (lhsShape.size() < 2 || rhsShape.size() < 2)
		{
			throw std::runtime_error("BatchMatMul inputs must have rank >= 2");
		}
		const auto lhsK = lhsShape[lhsShape.size() - 1];
		const auto rhsK = rhsShape[rhsShape.size() - 2];
		if (lhsK != rhsK)
		{
			throw std::runtime_error("BatchMatMul inner dimensions do not match");
		}

		const auto lhsLead = lhsShape.subspan(0, lhsShape.size() - 2);
		const auto rhsLead = rhsShape.subspan(0, rhsShape.size() - 2);
		auto outputShape = BroadcastShapesTrailing(lhsLead, rhsLead, "BatchMatMul leading dimensions");
		outputShape.push_back(lhsShape[lhsShape.size() - 2]);
		outputShape.push_back(rhsShape[rhsShape.size() - 1]);
		return outputShape;
	}

	inline std::vector<std::size_t> OutProdOutputShape(std::span<const std::size_t> lhsShape,
	                                                   std::span<const std::size_t> rhsShape)
	{
		if (lhsShape.size() < 2 || rhsShape.size() < 2)
		{
			throw std::runtime_error("OutProd inputs must have rank >= 2");
		}
		const auto lhsK = lhsShape[lhsShape.size() - 1];
		const auto rhsK = rhsShape[rhsShape.size() - 1];
		if (lhsK != rhsK)
		{
			throw std::runtime_error("OutProd contraction dimensions do not match");
		}

		const auto lhsLead = lhsShape.subspan(0, lhsShape.size() - 2);
		const auto rhsLead = rhsShape.subspan(0, rhsShape.size() - 2);
		auto outputShape = BroadcastShapesTrailing(lhsLead, rhsLead, "OutProd leading dimensions");
		outputShape.push_back(lhsShape[lhsShape.size() - 2]);
		outputShape.push_back(rhsShape[rhsShape.size() - 2]);
		return outputShape;
	}

	inline std::vector<std::size_t> TimestepEmbeddingOutputShape(std::span<const std::size_t> timestepShape,
	                                                            std::size_t dim, std::size_t maxPeriod)
	{
		if (timestepShape.size() != 1)
		{
			throw std::runtime_error("TimestepEmbedding timesteps must be rank-1");
		}
		if (dim == 0 || maxPeriod == 0)
		{
			throw std::runtime_error("TimestepEmbedding dim and maxPeriod must be positive");
		}
		return { timestepShape[0], dim };
	}

	inline std::vector<std::size_t> SolveTriOutputShape(std::span<const std::size_t> aShape,
	                                                    std::span<const std::size_t> bShape,
	                                                    bool lower, bool unitDiagonal)
	{
		if (!lower || unitDiagonal)
		{
			throw std::runtime_error("SolveTri currently supports only lower, non-unit diagonal solves");
		}
		if (aShape.size() < 2 || bShape.size() < 2)
		{
			throw std::runtime_error("SolveTri inputs must have rank >= 2");
		}
		if (aShape.size() != bShape.size())
		{
			throw std::runtime_error("SolveTri inputs must have the same rank");
		}
		const auto rank = aShape.size();
		const auto n = aShape[rank - 1];
		if (aShape[rank - 2] != n)
		{
			throw std::runtime_error("SolveTri matrix input must be square");
		}
		if (bShape[rank - 2] != n)
		{
			throw std::runtime_error("SolveTri rhs row dimension must match matrix size");
		}
		for (auto dim = 0uz; dim + 2 < rank; ++dim)
		{
			if (aShape[dim] != bShape[dim])
			{
				throw std::runtime_error("SolveTri batch dimensions must match exactly");
			}
		}
		return std::vector<std::size_t>{ bShape.begin(), bShape.end() };
	}

	inline std::size_t SGDStepOutputCount(bool hasVelocity, double momentum)
	{
		if (momentum < 0.0)
		{
			throw std::runtime_error("SGDStep momentum must be non-negative");
		}
		if (momentum == 0.0 && hasVelocity)
		{
			throw std::runtime_error("SGDStep velocity requires momentum > 0");
		}
		return momentum > 0.0 ? 2uz : 1uz;
	}

	inline void ValidateOptimizerStepShape(std::span<const std::size_t> parameterShape,
	                                       std::span<const std::size_t> otherShape,
	                                       std::string_view label)
	{
		if (!std::ranges::equal(parameterShape, otherShape))
		{
			throw std::runtime_error(std::format("{} shape must match parameter shape", label));
		}
	}

	inline std::size_t BroadcastOffsetForLinear(const Tensor<CPU>& tensor, std::span<const std::size_t> targetShape,
	                                            std::size_t targetLinear)
	{
		const auto sourceShape = tensor.Shape().Dims;
		if (sourceShape.size() > targetShape.size())
		{
			throw std::runtime_error("Broadcast source rank is greater than target rank");
		}
		const auto targetStrides = RowMajorStrides(targetShape);
		const auto sourceStrides = RowMajorStrides(sourceShape);
		const auto rankDelta = targetShape.size() - sourceShape.size();
		auto remaining = targetLinear;
		auto offset = 0uz;

		for (auto dim = 0uz; dim < targetShape.size(); ++dim)
		{
			const auto coord = targetStrides.empty() ? 0uz : remaining / targetStrides[dim];
			if (!targetStrides.empty())
			{
				remaining %= targetStrides[dim];
			}
			if (dim >= rankDelta)
			{
				const auto sourceDim = dim - rankDelta;
				const auto sourceExtent = sourceShape[sourceDim];
				if (sourceExtent != targetShape[dim] && sourceExtent != 1)
				{
					throw std::runtime_error("Broadcast source shape is not compatible with target shape");
				}
				offset += (sourceExtent == 1 ? 0uz : coord) * sourceStrides[sourceDim];
			}
		}
		return offset;
	}

	inline double BroadcastValueAsDouble(const Tensor<CPU>& tensor, std::span<const std::size_t> targetShape,
	                                     std::size_t targetLinear)
	{
		const auto offset = BroadcastOffsetForLinear(tensor, targetShape, targetLinear);
		return EnumDispatch(tensor.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			return static_cast<double>(static_cast<const T*>(tensor.RawData())[offset]);
		});
	}

	inline Tensor<CPU> EvalBatchMatMul(const Tensor<CPU>& lhs, const Tensor<CPU>& rhs)
	{
		const auto outputShape = BatchMatMulOutputShape(lhs.Shape().Dims, rhs.Shape().Dims);
		if (lhs.DType() != rhs.DType())
		{
			throw std::runtime_error("BatchMatMul inputs must have the same dtype");
		}
		if (lhs.DType() == DataType::Bool)
		{
			throw std::runtime_error("BatchMatMul does not support Bool tensors");
		}

		CPU cpu;
		Tensor<CPU> result(Uninitialized, outputShape, lhs.DType(), cpu);

		const auto lhsRank = lhs.Shape().NumDim();
		const auto rhsRank = rhs.Shape().NumDim();
		const auto outputRank = outputShape.size();
		const auto leadRank = outputRank - 2;
		const auto m = outputShape[outputRank - 2];
		const auto n = outputShape[outputRank - 1];
		const auto k = lhs.Shape()[lhsRank - 1];
		const auto outputShapeView = std::span<const std::size_t>{ outputShape };
		const auto batchCount = Product(outputShapeView.subspan(0, leadRank));
		const auto outputLeadStrides = RowMajorStrides(outputShapeView.subspan(0, leadRank));
		const auto lhsStrides = RowMajorStrides(lhs.Shape().Dims);
		const auto rhsStrides = RowMajorStrides(rhs.Shape().Dims);
		const auto lhsLeadRank = lhsRank - 2;
		const auto rhsLeadRank = rhsRank - 2;
		const auto lhsLeadDelta = leadRank - lhsLeadRank;
		const auto rhsLeadDelta = leadRank - rhsLeadRank;

		EnumDispatch(lhs.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			const auto* lhsPtr = static_cast<const T*>(lhs.RawData());
			const auto* rhsPtr = static_cast<const T*>(rhs.RawData());
			auto* dst = static_cast<T*>(result.RawData());

			std::vector<std::size_t> coord(leadRank, 0uz);
			for (auto batch = 0uz; batch < batchCount; ++batch)
			{
				auto remaining = batch;
				for (auto dim = 0uz; dim < leadRank; ++dim)
				{
					coord[dim] = outputLeadStrides.empty() ? 0uz : remaining / outputLeadStrides[dim];
					if (!outputLeadStrides.empty())
					{
						remaining %= outputLeadStrides[dim];
					}
				}

				auto lhsBase = 0uz;
				for (auto dim = 0uz; dim < lhsLeadRank; ++dim)
				{
					const auto outCoord = coord[lhsLeadDelta + dim];
					const auto lhsCoord = lhs.Shape()[dim] == 1 ? 0uz : outCoord;
					lhsBase += lhsCoord * lhsStrides[dim];
				}
				auto rhsBase = 0uz;
				for (auto dim = 0uz; dim < rhsLeadRank; ++dim)
				{
					const auto outCoord = coord[rhsLeadDelta + dim];
					const auto rhsCoord = rhs.Shape()[dim] == 1 ? 0uz : outCoord;
					rhsBase += rhsCoord * rhsStrides[dim];
				}

				for (auto row = 0uz; row < m; ++row)
				{
					for (auto col = 0uz; col < n; ++col)
					{
						double acc = 0.0;
						for (auto kk = 0uz; kk < k; ++kk)
						{
							const auto lhsOffset = lhsBase + row * lhsStrides[lhsRank - 2] + kk * lhsStrides[lhsRank - 1];
							const auto rhsOffset = rhsBase + kk * rhsStrides[rhsRank - 2] + col * rhsStrides[rhsRank - 1];
							acc += static_cast<double>(lhsPtr[lhsOffset]) * static_cast<double>(rhsPtr[rhsOffset]);
						}
						dst[(batch * m + row) * n + col] = static_cast<T>(acc);
					}
				}
			}
		});
		return result;
	}

	inline Tensor<CPU> EvalOutProd(const Tensor<CPU>& lhs, const Tensor<CPU>& rhs)
	{
		const auto outputShape = OutProdOutputShape(lhs.Shape().Dims, rhs.Shape().Dims);
		if (lhs.DType() != rhs.DType())
		{
			throw std::runtime_error("OutProd inputs must have the same dtype");
		}
		if (!IsFloatingDataType(lhs.DType()))
		{
			throw std::runtime_error("OutProd requires floating-point tensors");
		}

		CPU cpu;
		Tensor<CPU> result(Uninitialized, outputShape, lhs.DType(), cpu);

		const auto lhsRank = lhs.Shape().NumDim();
		const auto rhsRank = rhs.Shape().NumDim();
		const auto outputRank = outputShape.size();
		const auto leadRank = outputRank - 2;
		const auto m = outputShape[outputRank - 2];
		const auto n = outputShape[outputRank - 1];
		const auto k = lhs.Shape()[lhsRank - 1];
		const auto outputShapeView = std::span<const std::size_t>{ outputShape };
		const auto batchCount = Product(outputShapeView.subspan(0, leadRank));
		const auto outputLeadStrides = RowMajorStrides(outputShapeView.subspan(0, leadRank));
		const auto lhsStrides = RowMajorStrides(lhs.Shape().Dims);
		const auto rhsStrides = RowMajorStrides(rhs.Shape().Dims);
		const auto lhsLeadRank = lhsRank - 2;
		const auto rhsLeadRank = rhsRank - 2;
		const auto lhsLeadDelta = leadRank - lhsLeadRank;
		const auto rhsLeadDelta = leadRank - rhsLeadRank;

		EnumDispatch(lhs.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			const auto* lhsPtr = static_cast<const T*>(lhs.RawData());
			const auto* rhsPtr = static_cast<const T*>(rhs.RawData());
			auto* dst = static_cast<T*>(result.RawData());

			std::vector<std::size_t> coord(leadRank, 0uz);
			for (auto batch = 0uz; batch < batchCount; ++batch)
			{
				auto remaining = batch;
				for (auto dim = 0uz; dim < leadRank; ++dim)
				{
					coord[dim] = outputLeadStrides.empty() ? 0uz : remaining / outputLeadStrides[dim];
					if (!outputLeadStrides.empty())
					{
						remaining %= outputLeadStrides[dim];
					}
				}

				auto lhsBase = 0uz;
				for (auto dim = 0uz; dim < lhsLeadRank; ++dim)
				{
					const auto outCoord = coord[lhsLeadDelta + dim];
					const auto lhsCoord = lhs.Shape()[dim] == 1 ? 0uz : outCoord;
					lhsBase += lhsCoord * lhsStrides[dim];
				}
				auto rhsBase = 0uz;
				for (auto dim = 0uz; dim < rhsLeadRank; ++dim)
				{
					const auto outCoord = coord[rhsLeadDelta + dim];
					const auto rhsCoord = rhs.Shape()[dim] == 1 ? 0uz : outCoord;
					rhsBase += rhsCoord * rhsStrides[dim];
				}

				for (auto row = 0uz; row < m; ++row)
				{
					for (auto col = 0uz; col < n; ++col)
					{
						double acc = 0.0;
						for (auto kk = 0uz; kk < k; ++kk)
						{
							const auto lhsOffset = lhsBase + row * lhsStrides[lhsRank - 2] + kk * lhsStrides[lhsRank - 1];
							const auto rhsOffset = rhsBase + col * rhsStrides[rhsRank - 2] + kk * rhsStrides[rhsRank - 1];
							acc += static_cast<double>(lhsPtr[lhsOffset]) * static_cast<double>(rhsPtr[rhsOffset]);
						}
						dst[(batch * m + row) * n + col] = static_cast<T>(acc);
					}
				}
			}
		});
		return result;
	}

	inline Tensor<CPU> EvalTimestepEmbedding(const Tensor<CPU>& timesteps, std::size_t dim, std::size_t maxPeriod)
	{
		const auto outputShape = TimestepEmbeddingOutputShape(timesteps.Shape().Dims, dim, maxPeriod);
		if (!IsFloatingDataType(timesteps.DType()))
		{
			throw std::runtime_error("TimestepEmbedding requires floating-point timesteps");
		}

		CPU cpu;
		Tensor<CPU> result(Uninitialized, outputShape, DataType::Float32, cpu);
		auto* dst = static_cast<float*>(result.RawData());
		const auto count = timesteps.Shape()[0];
		const auto half = dim / 2;

		EnumDispatch(timesteps.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			const auto* src = static_cast<const T*>(timesteps.RawData());
			for (auto i = 0uz; i < count; ++i)
			{
				const auto timestep = static_cast<float>(src[i]);
				auto* row = dst + i * dim;
				for (auto j = 0uz; j < half; ++j)
				{
					const auto freq = half == 0 ? 0.0F
					                            : std::exp(-std::log(static_cast<float>(maxPeriod)) *
					                                       static_cast<float>(j) / static_cast<float>(half));
					const auto arg = timestep * freq;
					row[j] = std::cos(arg);
					row[j + half] = std::sin(arg);
				}
				if (dim % 2 != 0)
				{
					row[2 * half] = 0.0F;
				}
			}
		});
		return result;
	}

	inline Tensor<CPU> EvalSolveTri(const Tensor<CPU>& a, const Tensor<CPU>& b, bool lower, bool unitDiagonal)
	{
		const auto outputShape = SolveTriOutputShape(a.Shape().Dims, b.Shape().Dims, lower, unitDiagonal);
		if (a.DType() != DataType::Float32 || b.DType() != DataType::Float32)
		{
			throw std::runtime_error("SolveTri currently supports Float32 tensors only");
		}

		CPU cpu;
		Tensor<CPU> result(Uninitialized, outputShape, DataType::Float32, cpu);
		const auto rank = a.Shape().NumDim();
		const auto n = a.Shape()[rank - 1];
		const auto rhsCols = b.Shape()[rank - 1];
		const auto batchCount = Product(std::span<const std::size_t>{ outputShape }.subspan(0, rank - 2));
		const auto* aPtr = static_cast<const float*>(a.RawData());
		const auto* bPtr = static_cast<const float*>(b.RawData());
		auto* dst = static_cast<float*>(result.RawData());

		for (auto batch = 0uz; batch < batchCount; ++batch)
		{
			const auto aBase = batch * n * n;
			const auto bBase = batch * n * rhsCols;
			const auto dstBase = bBase;
			for (auto col = 0uz; col < rhsCols; ++col)
			{
				for (auto row = 0uz; row < n; ++row)
				{
					float sum = 0.0F;
					for (auto prev = 0uz; prev < row; ++prev)
					{
						sum += aPtr[aBase + row * n + prev] * dst[dstBase + prev * rhsCols + col];
					}
					const auto diag = aPtr[aBase + row * n + row];
					if (diag == 0.0F)
					{
						throw std::runtime_error("SolveTri diagonal contains zero");
					}
					dst[dstBase + row * rhsCols + col] = (bPtr[bBase + row * rhsCols + col] - sum) / diag;
				}
			}
		}
		return result;
	}

	inline std::vector<Tensor<CPU>> EvalSGDStep(const Tensor<CPU>& parameter, const Tensor<CPU>& gradient,
	                                            const Tensor<CPU>* velocity, double learningRate,
	                                            double momentum, double weightDecay, bool nesterov)
	{
		if (parameter.DType() != DataType::Float32 || gradient.DType() != DataType::Float32 ||
		    (velocity && velocity->DType() != DataType::Float32))
		{
			throw std::runtime_error("SGDStep currently supports Float32 tensors only");
		}
		ValidateOptimizerStepShape(parameter.Shape().Dims, gradient.Shape().Dims, "SGDStep gradient");
		if (velocity)
		{
			ValidateOptimizerStepShape(parameter.Shape().Dims, velocity->Shape().Dims, "SGDStep velocity");
		}
		if (!std::isfinite(learningRate) || learningRate <= 0.0)
		{
			throw std::runtime_error("SGDStep learningRate must be finite and positive");
		}
		if (!std::isfinite(momentum) || momentum < 0.0)
		{
			throw std::runtime_error("SGDStep momentum must be finite and non-negative");
		}
		if (!std::isfinite(weightDecay) || weightDecay < 0.0)
		{
			throw std::runtime_error("SGDStep weightDecay must be finite and non-negative");
		}
		(void)SGDStepOutputCount(velocity != nullptr, momentum);

		CPU cpu;
		Tensor<CPU> updatedParameter(Uninitialized, parameter.Shape(), DataType::Float32, cpu);
		std::optional<Tensor<CPU>> updatedVelocity;
		if (momentum > 0.0)
		{
			updatedVelocity.emplace(Uninitialized, parameter.Shape(), DataType::Float32, cpu);
		}

		const auto* parameterPtr = static_cast<const float*>(parameter.RawData());
		const auto* gradientPtr = static_cast<const float*>(gradient.RawData());
		const auto* velocityPtr = velocity ? static_cast<const float*>(velocity->RawData()) : nullptr;
		auto* parameterOut = static_cast<float*>(updatedParameter.RawData());
		auto* velocityOut = updatedVelocity ? static_cast<float*>(updatedVelocity->RawData()) : nullptr;
		for (auto index = 0uz; index < parameter.NumElements(); ++index)
		{
			const auto decayedGradient =
			    static_cast<float>(static_cast<double>(gradientPtr[index]) + weightDecay * parameterPtr[index]);
			float update = decayedGradient;
			if (momentum > 0.0)
			{
				const auto previousVelocity = velocityPtr ? velocityPtr[index] : 0.0F;
				const auto nextVelocity =
				    static_cast<float>(momentum * previousVelocity + static_cast<double>(decayedGradient));
				velocityOut[index] = nextVelocity;
				update = nesterov ? static_cast<float>(decayedGradient + momentum * nextVelocity) : nextVelocity;
			}
			parameterOut[index] = static_cast<float>(parameterPtr[index] - learningRate * update);
		}

		std::vector<Tensor<CPU>> outputs;
		outputs.push_back(std::move(updatedParameter));
		if (updatedVelocity)
		{
			outputs.push_back(std::move(*updatedVelocity));
		}
		return outputs;
	}

	inline std::vector<Tensor<CPU>> EvalAdamWStep(const Tensor<CPU>& parameter, const Tensor<CPU>& gradient,
	                                             const Tensor<CPU>& firstMoment,
	                                             const Tensor<CPU>& secondMoment,
	                                             double learningRate, double beta1, double beta2,
	                                             double epsilon, double weightDecay, std::size_t step)
	{
		if (parameter.DType() != DataType::Float32 || gradient.DType() != DataType::Float32 ||
		    firstMoment.DType() != DataType::Float32 || secondMoment.DType() != DataType::Float32)
		{
			throw std::runtime_error("AdamWStep currently supports Float32 tensors only");
		}
		ValidateOptimizerStepShape(parameter.Shape().Dims, gradient.Shape().Dims, "AdamWStep gradient");
		ValidateOptimizerStepShape(parameter.Shape().Dims, firstMoment.Shape().Dims, "AdamWStep firstMoment");
		ValidateOptimizerStepShape(parameter.Shape().Dims, secondMoment.Shape().Dims, "AdamWStep secondMoment");
		if (!std::isfinite(learningRate) || learningRate <= 0.0)
		{
			throw std::runtime_error("AdamWStep learningRate must be finite and positive");
		}
		if (!std::isfinite(beta1) || !std::isfinite(beta2) || beta1 < 0.0 || beta1 >= 1.0 ||
		    beta2 < 0.0 || beta2 >= 1.0)
		{
			throw std::runtime_error("AdamWStep beta values must be finite and in [0, 1)");
		}
		if (!std::isfinite(epsilon) || epsilon <= 0.0)
		{
			throw std::runtime_error("AdamWStep epsilon must be finite and positive");
		}
		if (!std::isfinite(weightDecay) || weightDecay < 0.0)
		{
			throw std::runtime_error("AdamWStep weightDecay must be finite and non-negative");
		}
		if (step == 0)
		{
			throw std::runtime_error("AdamWStep step must be positive");
		}

		CPU cpu;
		Tensor<CPU> updatedParameter(Uninitialized, parameter.Shape(), DataType::Float32, cpu);
		Tensor<CPU> updatedFirstMoment(Uninitialized, parameter.Shape(), DataType::Float32, cpu);
		Tensor<CPU> updatedSecondMoment(Uninitialized, parameter.Shape(), DataType::Float32, cpu);

		const auto* parameterPtr = static_cast<const float*>(parameter.RawData());
		const auto* gradientPtr = static_cast<const float*>(gradient.RawData());
		const auto* firstPtr = static_cast<const float*>(firstMoment.RawData());
		const auto* secondPtr = static_cast<const float*>(secondMoment.RawData());
		auto* parameterOut = static_cast<float*>(updatedParameter.RawData());
		auto* firstOut = static_cast<float*>(updatedFirstMoment.RawData());
		auto* secondOut = static_cast<float*>(updatedSecondMoment.RawData());
		const auto biasCorrection1 = 1.0 - std::pow(beta1, static_cast<double>(step));
		const auto biasCorrection2 = 1.0 - std::pow(beta2, static_cast<double>(step));
		for (auto index = 0uz; index < parameter.NumElements(); ++index)
		{
			const auto grad = static_cast<double>(gradientPtr[index]);
			const auto first = beta1 * static_cast<double>(firstPtr[index]) + (1.0 - beta1) * grad;
			const auto second = beta2 * static_cast<double>(secondPtr[index]) + (1.0 - beta2) * grad * grad;
			firstOut[index] = static_cast<float>(first);
			secondOut[index] = static_cast<float>(second);
			const auto firstHat = first / biasCorrection1;
			const auto secondHat = second / biasCorrection2;
			const auto decayedParameter = static_cast<double>(parameterPtr[index]) *
			                              (1.0 - learningRate * weightDecay);
			parameterOut[index] = static_cast<float>(
			    decayedParameter - learningRate * firstHat / (std::sqrt(secondHat) + epsilon));
		}

		std::vector<Tensor<CPU>> outputs;
		outputs.push_back(std::move(updatedParameter));
		outputs.push_back(std::move(updatedFirstMoment));
		outputs.push_back(std::move(updatedSecondMoment));
		return outputs;
	}

	inline Tensor<CPU> EvalSoftmax(const Tensor<CPU>& input, std::size_t axis)
	{
		if (input.Shape().NumDim() == 0 || axis >= input.Shape().NumDim())
		{
			throw std::runtime_error("Softmax axis out of range");
		}
		if (!IsFloatingDataType(input.DType()))
		{
			throw std::runtime_error("Softmax requires a floating-point tensor");
		}

		CPU cpu;
		Tensor<CPU> result(Uninitialized, input.Shape(), input.DType(), cpu);
		const auto axisSize = input.Shape()[axis];
		auto outer = 1uz;
		for (auto dim = 0uz; dim < axis; ++dim)
		{
			outer *= input.Shape()[dim];
		}
		auto inner = 1uz;
		for (auto dim = axis + 1; dim < input.Shape().NumDim(); ++dim)
		{
			inner *= input.Shape()[dim];
		}

		EnumDispatch(input.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			const auto* src = static_cast<const T*>(input.RawData());
			auto* dst = static_cast<T*>(result.RawData());
			for (auto o = 0uz; o < outer; ++o)
			{
				for (auto i = 0uz; i < inner; ++i)
				{
					auto maxValue = -std::numeric_limits<double>::infinity();
					for (auto a = 0uz; a < axisSize; ++a)
					{
						maxValue = std::max(maxValue, static_cast<double>(src[(o * axisSize + a) * inner + i]));
					}
					double sum = 0.0;
					for (auto a = 0uz; a < axisSize; ++a)
					{
						sum += std::exp(static_cast<double>(src[(o * axisSize + a) * inner + i]) - maxValue);
					}
					for (auto a = 0uz; a < axisSize; ++a)
					{
						const auto value = std::exp(static_cast<double>(src[(o * axisSize + a) * inner + i]) - maxValue) / sum;
						dst[(o * axisSize + a) * inner + i] = static_cast<T>(value);
					}
				}
			}
		});
		return result;
	}

	inline Tensor<CPU> EvalScan(const Tensor<CPU>& input, std::size_t axis, ScanOp op)
	{
		if (input.Shape().NumDim() == 0 || axis >= input.Shape().NumDim())
		{
			throw std::runtime_error("Scan axis out of range");
		}
		if (input.DType() == DataType::Bool)
		{
			throw std::runtime_error("Scan does not support Bool tensors");
		}
		if (op == ScanOp::LogSumExp && !IsFloatingDataType(input.DType()))
		{
			throw std::runtime_error("Scan LogSumExp requires a floating-point tensor");
		}

		CPU cpu;
		Tensor<CPU> result(Uninitialized, input.Shape(), input.DType(), cpu);
		const auto axisSize = input.Shape()[axis];
		auto outer = 1uz;
		for (auto dim = 0uz; dim < axis; ++dim)
		{
			outer *= input.Shape()[dim];
		}
		auto inner = 1uz;
		for (auto dim = axis + 1; dim < input.Shape().NumDim(); ++dim)
		{
			inner *= input.Shape()[dim];
		}

		EnumDispatch(input.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			const auto* src = static_cast<const T*>(input.RawData());
			auto* dst = static_cast<T*>(result.RawData());
			for (auto o = 0uz; o < outer; ++o)
			{
				for (auto i = 0uz; i < inner; ++i)
				{
					double acc = 0.0;
					for (auto a = 0uz; a < axisSize; ++a)
					{
						const auto offset = (o * axisSize + a) * inner + i;
						const auto value = static_cast<double>(src[offset]);
						if (a == 0)
						{
							acc = value;
						}
						else
						{
							switch (op)
							{
							case ScanOp::Sum:
								acc += value;
								break;
							case ScanOp::Max:
								acc = std::max(acc, value);
								break;
							case ScanOp::Prod:
								acc *= value;
								break;
							case ScanOp::LogSumExp: {
								const auto m = std::max(acc, value);
								acc = m + std::log(std::exp(acc - m) + std::exp(value - m));
								break;
							}
							}
						}
						dst[offset] = static_cast<T>(acc);
					}
				}
			}
		});
		return result;
	}

	inline Tensor<CPU> EvalNormalization(const Tensor<CPU>& input, const Tensor<CPU>* scale,
	                                     const Tensor<CPU>* bias, NormalizationMode mode, std::size_t axis,
	                                     std::size_t groupCount, double epsilon)
	{
		if (!IsFloatingDataType(input.DType()))
		{
			throw std::runtime_error("Normalization requires a floating-point input tensor");
		}
		if (!std::isfinite(epsilon) || epsilon <= 0.0)
		{
			throw std::runtime_error("Normalization epsilon must be finite and positive");
		}
		if ((scale && !IsFloatingDataType(scale->DType())) || (bias && !IsFloatingDataType(bias->DType())))
		{
			throw std::runtime_error("Normalization scale/bias must be floating-point tensors");
		}
		if (scale)
		{
			(void)BroadcastToShape(scale->Shape().Dims, input.Shape().Dims);
		}
		if (bias)
		{
			(void)BroadcastToShape(bias->Shape().Dims, input.Shape().Dims);
		}

		CPU cpu;
		Tensor<CPU> result(Uninitialized, input.Shape(), input.DType(), cpu);
		const auto applyAffine = [&](double value, std::size_t linear) {
			if (scale)
			{
				value *= BroadcastValueAsDouble(*scale, input.Shape().Dims, linear);
			}
			if (bias)
			{
				value += BroadcastValueAsDouble(*bias, input.Shape().Dims, linear);
			}
			return value;
		};

		EnumDispatch(input.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			const auto* src = static_cast<const T*>(input.RawData());
			auto* dst = static_cast<T*>(result.RawData());
			if (mode == NormalizationMode::LayerNorm || mode == NormalizationMode::RMSNorm)
			{
				if (input.Shape().NumDim() == 0 || axis >= input.Shape().NumDim())
				{
					throw std::runtime_error("Normalization axis out of range");
				}
				const auto axisSize = input.Shape()[axis];
				auto outer = 1uz;
				for (auto dim = 0uz; dim < axis; ++dim)
				{
					outer *= input.Shape()[dim];
				}
				auto inner = 1uz;
				for (auto dim = axis + 1; dim < input.Shape().NumDim(); ++dim)
				{
					inner *= input.Shape()[dim];
				}
				for (auto o = 0uz; o < outer; ++o)
				{
					for (auto i = 0uz; i < inner; ++i)
					{
						double mean = 0.0;
						if (mode == NormalizationMode::LayerNorm)
						{
							for (auto a = 0uz; a < axisSize; ++a)
							{
								mean += static_cast<double>(src[(o * axisSize + a) * inner + i]);
							}
							mean /= static_cast<double>(axisSize);
						}

						double variance = 0.0;
						for (auto a = 0uz; a < axisSize; ++a)
						{
							const auto raw = static_cast<double>(src[(o * axisSize + a) * inner + i]);
							const auto centered = mode == NormalizationMode::LayerNorm ? raw - mean : raw;
							variance += centered * centered;
						}
						variance /= static_cast<double>(axisSize);
						const auto denom = std::sqrt(variance + epsilon);
						for (auto a = 0uz; a < axisSize; ++a)
						{
							const auto linear = (o * axisSize + a) * inner + i;
							const auto raw = static_cast<double>(src[linear]);
							const auto centered = mode == NormalizationMode::LayerNorm ? raw - mean : raw;
							dst[linear] = static_cast<T>(applyAffine(centered / denom, linear));
						}
					}
				}
				return;
			}

			if (groupCount == 0)
			{
				throw std::runtime_error("GroupNorm requires groupCount > 0");
			}
			const auto rank = input.Shape().NumDim();
			if (rank == 0 || rank > 4)
			{
				throw std::runtime_error("GroupNorm input rank must be between 1 and 4");
			}
			const auto batch = rank == 4 ? input.Shape()[3] : 1uz;
			auto groupedVolume = 1uz;
			const auto groupedRank = std::min<std::size_t>(rank, 3);
			for (auto dim = 0uz; dim < groupedRank; ++dim)
			{
				groupedVolume *= input.Shape()[dim];
			}
			if (groupedVolume % groupCount != 0)
			{
				throw std::runtime_error("GroupNorm grouped element count must be divisible by groupCount");
			}
			const auto groupSize = groupedVolume / groupCount;
			for (auto batchIndex = 0uz; batchIndex < batch; ++batchIndex)
			{
				for (auto group = 0uz; group < groupCount; ++group)
				{
					double mean = 0.0;
					for (auto member = 0uz; member < groupSize; ++member)
					{
						const auto linear = (group * groupSize + member) * batch + batchIndex;
						mean += static_cast<double>(src[linear]);
					}
					mean /= static_cast<double>(groupSize);
					double variance = 0.0;
					for (auto member = 0uz; member < groupSize; ++member)
					{
						const auto linear = (group * groupSize + member) * batch + batchIndex;
						const auto centered = static_cast<double>(src[linear]) - mean;
						variance += centered * centered;
					}
					variance /= static_cast<double>(groupSize);
					const auto denom = std::sqrt(variance + epsilon);
					for (auto member = 0uz; member < groupSize; ++member)
					{
						const auto linear = (group * groupSize + member) * batch + batchIndex;
						const auto centered = static_cast<double>(src[linear]) - mean;
						dst[linear] = static_cast<T>(applyAffine(centered / denom, linear));
					}
				}
			}
		});
		return result;
	}

	inline Tensor<CPU> EvalSSMScan(const Tensor<CPU>& state, const Tensor<CPU>& dt, const Tensor<CPU>& a,
	                               const Tensor<CPU>& b, const Tensor<CPU>& c, const Tensor<CPU>* d)
	{
		if (state.Shape().NumDim() != 2)
		{
			throw std::runtime_error("SSMScan state must be rank-2 [steps, channels]");
		}
		if (!IsFloatingDataType(state.DType()))
		{
			throw std::runtime_error("SSMScan requires floating-point tensors");
		}
		for (const Tensor<CPU>* tensor : { &dt, &a, &b, &c })
		{
			if (!IsFloatingDataType(tensor->DType()))
			{
				throw std::runtime_error("SSMScan requires floating-point tensors");
			}
			(void)BroadcastToShape(tensor->Shape().Dims, state.Shape().Dims);
		}
		if (d)
		{
			if (!IsFloatingDataType(d->DType()))
			{
				throw std::runtime_error("SSMScan requires floating-point tensors");
			}
			(void)BroadcastToShape(d->Shape().Dims, state.Shape().Dims);
		}

		CPU cpu;
		Tensor<CPU> result(Uninitialized, state.Shape(), state.DType(), cpu);
		const auto steps = state.Shape()[0];
		const auto channels = state.Shape()[1];
		EnumDispatch(state.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			const auto* statePtr = static_cast<const T*>(state.RawData());
			auto* dst = static_cast<T*>(result.RawData());
			std::vector<double> hidden(channels, 0.0);
			for (auto step = 0uz; step < steps; ++step)
			{
				for (auto channel = 0uz; channel < channels; ++channel)
				{
					const auto linear = step * channels + channel;
					const auto x = static_cast<double>(statePtr[linear]);
					const auto dtValue = BroadcastValueAsDouble(dt, state.Shape().Dims, linear);
					const auto aValue = BroadcastValueAsDouble(a, state.Shape().Dims, linear);
					const auto bValue = BroadcastValueAsDouble(b, state.Shape().Dims, linear);
					const auto cValue = BroadcastValueAsDouble(c, state.Shape().Dims, linear);
					hidden[channel] = std::exp(dtValue * aValue) * hidden[channel] + dtValue * bValue * x;
					auto y = cValue * hidden[channel];
					if (d)
					{
						y += BroadcastValueAsDouble(*d, state.Shape().Dims, linear) * x;
					}
					dst[linear] = static_cast<T>(y);
				}
			}
		});
		return result;
	}

	inline Tensor<CPU> EvalRWKVWKV(const Tensor<CPU>& key, const Tensor<CPU>& value, const Tensor<CPU>& receptance,
	                               const Tensor<CPU>& timeDecay, const Tensor<CPU>& timeFirst)
	{
		if (key.Shape().NumDim() != 2)
		{
			throw std::runtime_error("RWKVWKV key must be rank-2 [steps, channels]");
		}
		if (key.DType() != value.DType() || key.DType() != receptance.DType())
		{
			throw std::runtime_error("RWKVWKV key/value/receptance must have the same dtype");
		}
		if (!std::ranges::equal(key.Shape().Dims, value.Shape().Dims) ||
		    !std::ranges::equal(key.Shape().Dims, receptance.Shape().Dims))
		{
			throw std::runtime_error("RWKVWKV key/value/receptance shapes must match");
		}
		if (!IsFloatingDataType(key.DType()) || !IsFloatingDataType(timeDecay.DType()) ||
		    !IsFloatingDataType(timeFirst.DType()))
		{
			throw std::runtime_error("RWKVWKV requires floating-point tensors");
		}
		(void)BroadcastToShape(timeDecay.Shape().Dims, key.Shape().Dims);
		(void)BroadcastToShape(timeFirst.Shape().Dims, key.Shape().Dims);

		CPU cpu;
		Tensor<CPU> result(Uninitialized, key.Shape(), key.DType(), cpu);
		const auto steps = key.Shape()[0];
		const auto channels = key.Shape()[1];
		EnumDispatch(key.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			const auto* keyPtr = static_cast<const T*>(key.RawData());
			const auto* valuePtr = static_cast<const T*>(value.RawData());
			const auto* receptancePtr = static_cast<const T*>(receptance.RawData());
			auto* dst = static_cast<T*>(result.RawData());
			std::vector<double> aa(channels, 0.0);
			std::vector<double> bb(channels, 0.0);

			for (auto step = 0uz; step < steps; ++step)
			{
				for (auto channel = 0uz; channel < channels; ++channel)
				{
					const auto linear = step * channels + channel;
					const auto wk = std::exp(static_cast<double>(keyPtr[linear]));
					const auto v = static_cast<double>(valuePtr[linear]);
					const auto r = static_cast<double>(receptancePtr[linear]);
					const auto first = std::exp(BroadcastValueAsDouble(timeFirst, key.Shape().Dims, linear));
					const auto numerator = first * wk * v + aa[channel];
					const auto denominator = first * wk + bb[channel];
					dst[linear] = static_cast<T>(r * numerator / denominator);
					const auto decay = std::exp(BroadcastValueAsDouble(timeDecay, key.Shape().Dims, linear));
					aa[channel] = decay * aa[channel] + wk * v;
					bb[channel] = decay * bb[channel] + wk;
				}
			}
		});
		return result;
	}

	inline void ValidateSlidingWindowParams(std::span<const std::size_t> kernelShape,
	                                        std::span<const std::size_t> strides,
	                                        std::span<const std::size_t> dilations,
	                                        std::span<const std::size_t> lowPads,
	                                        std::span<const std::size_t> highPads,
	                                        std::string_view label)
	{
		const auto rank = kernelShape.size();
		if (rank == 0)
		{
			throw std::runtime_error(std::format("{} spatial rank must be >= 1", label));
		}
		if (strides.size() != rank || dilations.size() != rank || lowPads.size() != rank || highPads.size() != rank)
		{
			throw std::runtime_error(std::format("{} parameter ranks must match kernel rank", label));
		}
		for (auto dim = 0uz; dim < rank; ++dim)
		{
			if (kernelShape[dim] == 0 || strides[dim] == 0 || dilations[dim] == 0)
			{
				throw std::runtime_error(std::format("{} kernel/stride/dilation values must be > 0", label));
			}
		}
	}

	inline std::vector<std::size_t> SlidingOutputSpatialShape(std::span<const std::size_t> inputSpatial,
	                                                         std::span<const std::size_t> kernelShape,
	                                                         std::span<const std::size_t> strides,
	                                                         std::span<const std::size_t> dilations,
	                                                         std::span<const std::size_t> lowPads,
	                                                         std::span<const std::size_t> highPads,
	                                                         std::string_view label)
	{
		ValidateSlidingWindowParams(kernelShape, strides, dilations, lowPads, highPads, label);
		if (inputSpatial.size() != kernelShape.size())
		{
			throw std::runtime_error(std::format("{} input spatial rank does not match kernel rank", label));
		}

		std::vector<std::size_t> output(inputSpatial.size(), 0uz);
		for (auto dim = 0uz; dim < inputSpatial.size(); ++dim)
		{
			const auto effectiveKernel = (kernelShape[dim] - 1) * dilations[dim] + 1;
			const auto padded = inputSpatial[dim] + lowPads[dim] + highPads[dim];
			if (padded < effectiveKernel)
			{
				throw std::runtime_error(std::format("{} padded input is smaller than effective kernel", label));
			}
			output[dim] = (padded - effectiveKernel) / strides[dim] + 1;
			if (output[dim] == 0)
			{
				throw std::runtime_error(std::format("{} output dimensions must be > 0", label));
			}
		}
		return output;
	}

	inline std::vector<std::size_t> Im2ColOutputShape(std::span<const std::size_t> inputShape,
	                                                 std::span<const std::size_t> kernelShape,
	                                                 std::span<const std::size_t> strides,
	                                                 std::span<const std::size_t> dilations,
	                                                 std::span<const std::size_t> lowPads,
	                                                 std::span<const std::size_t> highPads)
	{
		const auto spatialRank = kernelShape.size();
		if (inputShape.size() != spatialRank + 2)
		{
			throw std::runtime_error("Im2Col input must have shape [batch, channels, spatial...]");
		}
		const auto inputSpatial = inputShape.subspan(2);
		const auto outputSpatial =
		    SlidingOutputSpatialShape(inputSpatial, kernelShape, strides, dilations, lowPads, highPads, "Im2Col");
		return { inputShape[0], Product(outputSpatial), inputShape[1] * Product(kernelShape) };
	}

	inline std::vector<std::size_t> Conv2DOutputShape(std::span<const std::size_t> inputShape,
	                                                 std::span<const std::size_t> weightShape,
	                                                 std::span<const std::size_t> strides,
	                                                 std::span<const std::size_t> dilations,
	                                                 std::span<const std::size_t> lowPads,
	                                                 std::span<const std::size_t> highPads,
	                                                 std::size_t groupCount)
	{
		if (inputShape.size() != 4 || weightShape.size() != 4)
		{
			throw std::runtime_error("Conv2D input must be [batch, channels, height, width] and weight [out, in/group, kh, kw]");
		}
		if (groupCount == 0)
		{
			throw std::runtime_error("Conv2D groupCount must be > 0");
		}
		if (inputShape[1] % groupCount != 0 || weightShape[0] % groupCount != 0)
		{
			throw std::runtime_error("Conv2D input/output channels must be divisible by groupCount");
		}
		if (weightShape[1] != inputShape[1] / groupCount)
		{
			throw std::runtime_error("Conv2D weight inChannelsPerGroup does not match input channels/groupCount");
		}
		const std::size_t kernelStorage[] = { weightShape[2], weightShape[3] };
		const auto outputSpatial =
		    SlidingOutputSpatialShape(inputShape.subspan(2), kernelStorage, strides, dilations, lowPads, highPads,
		                              "Conv2D");
		return { inputShape[0], weightShape[0], outputSpatial[0], outputSpatial[1] };
	}

	inline void ValidateConv2DBiasShape(std::span<const std::size_t> biasShape, std::size_t outputChannels)
	{
		const std::size_t channelBias[] = { outputChannels };
		const std::size_t nchwBias[] = { 1uz, outputChannels, 1uz, 1uz };
		if (!std::ranges::equal(biasShape, channelBias) && !std::ranges::equal(biasShape, nchwBias))
		{
			throw std::runtime_error("Conv2D bias must have shape [outChannels] or [1, outChannels, 1, 1]");
		}
	}

	inline std::vector<std::size_t> ConvTranspose2DOutputShape(std::span<const std::size_t> inputShape,
	                                                           std::span<const std::size_t> weightShape,
	                                                           std::span<const std::size_t> strides,
	                                                           std::span<const std::size_t> dilations,
	                                                           std::span<const std::size_t> lowPads,
	                                                           std::span<const std::size_t> highPads,
	                                                           std::span<const std::size_t> outputPads,
	                                                           std::size_t groupCount)
	{
		if (inputShape.size() != 4 || weightShape.size() != 4)
		{
			throw std::runtime_error(
			    "ConvTranspose2D input must be [batch, channels, height, width] and weight [in, out/group, kh, kw]");
		}
		if (groupCount == 0)
		{
			throw std::runtime_error("ConvTranspose2D groupCount must be > 0");
		}
		if (strides.size() != 2 || dilations.size() != 2 || lowPads.size() != 2 || highPads.size() != 2 ||
		    outputPads.size() != 2)
		{
			throw std::runtime_error("ConvTranspose2D parameter ranks must be 2");
		}
		if (inputShape[1] != weightShape[0])
		{
			throw std::runtime_error("ConvTranspose2D weight input channels must match input channels");
		}
		if (inputShape[1] % groupCount != 0)
		{
			throw std::runtime_error("ConvTranspose2D input channels must be divisible by groupCount");
		}
		const std::size_t kernelStorage[] = { weightShape[2], weightShape[3] };
		ValidateSlidingWindowParams(kernelStorage, strides, dilations, lowPads, highPads, "ConvTranspose2D");
		for (auto dim = 0uz; dim < 2uz; ++dim)
		{
			if (outputPads[dim] >= strides[dim])
			{
				throw std::runtime_error("ConvTranspose2D output padding must be smaller than stride");
			}
		}

		std::vector<std::size_t> output{ inputShape[0], weightShape[1] * groupCount, 0uz, 0uz };
		for (auto dim = 0uz; dim < 2uz; ++dim)
		{
			const auto inputSpatial = inputShape[dim + 2];
			if (inputSpatial == 0)
			{
				throw std::runtime_error("ConvTranspose2D input spatial dimensions must be > 0");
			}
			const auto expanded = (inputSpatial - 1) * strides[dim] + (kernelStorage[dim] - 1) * dilations[dim] +
			                      outputPads[dim] + 1;
			const auto pads = lowPads[dim] + highPads[dim];
			if (expanded <= pads)
			{
				throw std::runtime_error("ConvTranspose2D output dimensions must be > 0");
			}
			output[dim + 2] = expanded - pads;
		}
		return output;
	}

	inline std::vector<std::size_t> Pool2DOutputShape(std::span<const std::size_t> inputShape,
	                                                 std::span<const std::size_t> kernelShape,
	                                                 std::span<const std::size_t> strides,
	                                                 std::span<const std::size_t> lowPads,
	                                                 std::span<const std::size_t> highPads)
	{
		if (inputShape.size() != 4)
		{
			throw std::runtime_error("Pool2D input must have shape [batch, channels, height, width]");
		}
		const std::size_t dilationStorage[] = { 1uz, 1uz };
		const auto outputSpatial =
		    SlidingOutputSpatialShape(inputShape.subspan(2), kernelShape, strides, dilationStorage, lowPads, highPads,
		                              "Pool2D");
		return { inputShape[0], inputShape[1], outputSpatial[0], outputSpatial[1] };
	}

	inline std::vector<std::size_t> UpsampleOutputShape(std::span<const std::size_t> inputShape,
	                                                   std::span<const std::size_t> outputSpatialShape)
	{
		if (inputShape.size() != 4)
		{
			throw std::runtime_error("Upsample input must have shape [batch, channels, height, width]");
		}
		if (outputSpatialShape.size() != 2 || outputSpatialShape[0] == 0 || outputSpatialShape[1] == 0)
		{
			throw std::runtime_error("Upsample output spatial shape must be [height, width] with non-zero dimensions");
		}
		return { inputShape[0], inputShape[1], outputSpatialShape[0], outputSpatialShape[1] };
	}

	inline Tensor<CPU> EvalIm2Col(const Tensor<CPU>& input, std::span<const std::size_t> kernelShape,
	                             std::span<const std::size_t> strides,
	                             std::span<const std::size_t> dilations,
	                             std::span<const std::size_t> lowPads,
	                             std::span<const std::size_t> highPads)
	{
		const auto outputShape = Im2ColOutputShape(input.Shape().Dims, kernelShape, strides, dilations, lowPads, highPads);
		const auto spatialRank = kernelShape.size();
		const auto outputSpatial = SlidingOutputSpatialShape(input.Shape().Dims.subspan(2), kernelShape, strides,
		                                                     dilations, lowPads, highPads, "Im2Col");
		const auto inputStrides = RowMajorStrides(input.Shape().Dims);
		const auto outputSpatialStrides = RowMajorStrides(outputSpatial);
		const auto kernelStrides = RowMajorStrides(kernelShape);
		const auto batch = input.Shape()[0];
		const auto channels = input.Shape()[1];
		const auto outputPositions = outputShape[1];
		const auto kernelVolume = Product(kernelShape);
		const auto columns = outputShape[2];

		CPU cpu;
		Tensor<CPU> result(Uninitialized, outputShape, input.DType(), cpu);
		EnumDispatch(input.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			const auto* src = static_cast<const T*>(input.RawData());
			auto* dst = static_cast<T*>(result.RawData());
			std::vector<std::size_t> outCoord(spatialRank, 0uz);
			std::vector<std::size_t> kernelCoord(spatialRank, 0uz);

			for (auto n = 0uz; n < batch; ++n)
			{
				for (auto outPos = 0uz; outPos < outputPositions; ++outPos)
				{
					auto remainingOut = outPos;
					for (auto dim = 0uz; dim < spatialRank; ++dim)
					{
						outCoord[dim] = outputSpatialStrides.empty() ? 0uz : remainingOut / outputSpatialStrides[dim];
						if (!outputSpatialStrides.empty())
						{
							remainingOut %= outputSpatialStrides[dim];
						}
					}

					for (auto col = 0uz; col < columns; ++col)
					{
						const auto channel = col / kernelVolume;
						auto kernelLinear = col % kernelVolume;
						for (auto dim = 0uz; dim < spatialRank; ++dim)
						{
							kernelCoord[dim] = kernelStrides.empty() ? 0uz : kernelLinear / kernelStrides[dim];
							if (!kernelStrides.empty())
							{
								kernelLinear %= kernelStrides[dim];
							}
						}

						auto inBounds = channel < channels;
						auto inputOffset = n * inputStrides[0] + channel * inputStrides[1];
						for (auto dim = 0uz; dim < spatialRank; ++dim)
						{
							const auto paddedCoord = outCoord[dim] * strides[dim] + kernelCoord[dim] * dilations[dim];
							if (paddedCoord < lowPads[dim])
							{
								inBounds = false;
								break;
							}
							const auto inputCoord = paddedCoord - lowPads[dim];
							if (inputCoord >= input.Shape()[dim + 2])
							{
								inBounds = false;
								break;
							}
							inputOffset += inputCoord * inputStrides[dim + 2];
						}
						dst[(n * outputPositions + outPos) * columns + col] =
						    inBounds ? src[inputOffset] : static_cast<T>(0);
					}
				}
			}
		});
		return result;
	}

	inline Tensor<CPU> EvalConv2D(const Tensor<CPU>& input, const Tensor<CPU>& weight,
	                             const Tensor<CPU>* bias,
	                             std::span<const std::size_t> strides,
	                             std::span<const std::size_t> dilations,
	                             std::span<const std::size_t> lowPads,
	                             std::span<const std::size_t> highPads,
	                             std::size_t groupCount)
	{
		if (input.DType() == DataType::Bool)
		{
			throw std::runtime_error("Conv2D does not support Bool tensors");
		}
		if (input.DType() != weight.DType() || (bias != nullptr && input.DType() != bias->DType()))
		{
			throw std::runtime_error("Conv2D input, weight, and bias dtypes must match");
		}

		const auto outputShape =
		    Conv2DOutputShape(input.Shape().Dims, weight.Shape().Dims, strides, dilations, lowPads, highPads,
		                      groupCount);
		if (bias != nullptr)
		{
			ValidateConv2DBiasShape(bias->Shape().Dims, outputShape[1]);
		}

		const auto batch = input.Shape()[0];
		const auto inputChannels = input.Shape()[1];
		const auto inputHeight = input.Shape()[2];
		const auto inputWidth = input.Shape()[3];
		const auto outputChannels = weight.Shape()[0];
		const auto inputChannelsPerGroup = weight.Shape()[1];
		const auto kernelHeight = weight.Shape()[2];
		const auto kernelWidth = weight.Shape()[3];
		const auto outputHeight = outputShape[2];
		const auto outputWidth = outputShape[3];
		const auto outputChannelsPerGroup = outputChannels / groupCount;

		CPU cpu;
		Tensor<CPU> result(Uninitialized, outputShape, input.DType(), cpu);
		EnumDispatch(input.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			const auto* src = static_cast<const T*>(input.RawData());
			const auto* filter = static_cast<const T*>(weight.RawData());
			const auto* biasData = bias == nullptr ? nullptr : static_cast<const T*>(bias->RawData());
			auto* dst = static_cast<T*>(result.RawData());

			for (auto n = 0uz; n < batch; ++n)
			{
				for (auto oc = 0uz; oc < outputChannels; ++oc)
				{
					const auto group = oc / outputChannelsPerGroup;
					const auto inputChannelBegin = group * inputChannelsPerGroup;
					for (auto oh = 0uz; oh < outputHeight; ++oh)
					{
						for (auto ow = 0uz; ow < outputWidth; ++ow)
						{
							auto acc = biasData == nullptr ? 0.0 : static_cast<double>(biasData[oc]);
							for (auto icg = 0uz; icg < inputChannelsPerGroup; ++icg)
							{
								const auto ic = inputChannelBegin + icg;
								for (auto kh = 0uz; kh < kernelHeight; ++kh)
								{
									const auto paddedH = oh * strides[0] + kh * dilations[0];
									if (paddedH < lowPads[0])
									{
										continue;
									}
									const auto ih = paddedH - lowPads[0];
									if (ih >= inputHeight)
									{
										continue;
									}
									for (auto kw = 0uz; kw < kernelWidth; ++kw)
									{
										const auto paddedW = ow * strides[1] + kw * dilations[1];
										if (paddedW < lowPads[1])
										{
											continue;
										}
										const auto iw = paddedW - lowPads[1];
										if (iw >= inputWidth)
										{
											continue;
										}
										const auto inputIndex = ((n * inputChannels + ic) * inputHeight + ih) * inputWidth + iw;
										const auto weightIndex =
										    ((oc * inputChannelsPerGroup + icg) * kernelHeight + kh) * kernelWidth + kw;
										acc += static_cast<double>(src[inputIndex]) * static_cast<double>(filter[weightIndex]);
									}
								}
							}
							dst[((n * outputChannels + oc) * outputHeight + oh) * outputWidth + ow] = static_cast<T>(acc);
						}
					}
				}
			}
		});
		return result;
	}

	inline Tensor<CPU> EvalConvTranspose2D(const Tensor<CPU>& input, const Tensor<CPU>& weight,
	                                      const Tensor<CPU>* bias,
	                                      std::span<const std::size_t> strides,
	                                      std::span<const std::size_t> dilations,
	                                      std::span<const std::size_t> lowPads,
	                                      std::span<const std::size_t> highPads,
	                                      std::span<const std::size_t> outputPads,
	                                      std::size_t groupCount)
	{
		if (input.DType() == DataType::Bool)
		{
			throw std::runtime_error("ConvTranspose2D does not support Bool tensors");
		}
		if (input.DType() != weight.DType() || (bias != nullptr && input.DType() != bias->DType()))
		{
			throw std::runtime_error("ConvTranspose2D input, weight, and bias dtypes must match");
		}

		const auto outputShape = ConvTranspose2DOutputShape(input.Shape().Dims, weight.Shape().Dims, strides, dilations,
		                                                    lowPads, highPads, outputPads, groupCount);
		if (bias != nullptr)
		{
			ValidateConv2DBiasShape(bias->Shape().Dims, outputShape[1]);
		}

		const auto batch = input.Shape()[0];
		const auto inputChannels = input.Shape()[1];
		const auto inputHeight = input.Shape()[2];
		const auto inputWidth = input.Shape()[3];
		const auto outputChannels = outputShape[1];
		const auto outputHeight = outputShape[2];
		const auto outputWidth = outputShape[3];
		const auto inputChannelsPerGroup = inputChannels / groupCount;
		const auto outputChannelsPerGroup = weight.Shape()[1];
		const auto kernelHeight = weight.Shape()[2];
		const auto kernelWidth = weight.Shape()[3];

		CPU cpu;
		Tensor<CPU> result(Uninitialized, outputShape, input.DType(), cpu);
		EnumDispatch(input.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			const auto* src = static_cast<const T*>(input.RawData());
			const auto* filter = static_cast<const T*>(weight.RawData());
			const auto* biasData = bias == nullptr ? nullptr : static_cast<const T*>(bias->RawData());
			auto* dst = static_cast<T*>(result.RawData());

			std::fill(dst, dst + result.NumElements(), static_cast<T>(0));
			if (biasData != nullptr)
			{
				for (auto n = 0uz; n < batch; ++n)
				{
					for (auto oc = 0uz; oc < outputChannels; ++oc)
					{
						for (auto oh = 0uz; oh < outputHeight; ++oh)
						{
							for (auto ow = 0uz; ow < outputWidth; ++ow)
							{
								dst[((n * outputChannels + oc) * outputHeight + oh) * outputWidth + ow] = biasData[oc];
							}
						}
					}
				}
			}

			for (auto n = 0uz; n < batch; ++n)
			{
				for (auto ic = 0uz; ic < inputChannels; ++ic)
				{
					const auto group = ic / inputChannelsPerGroup;
					const auto outputChannelBegin = group * outputChannelsPerGroup;
					for (auto ih = 0uz; ih < inputHeight; ++ih)
					{
						for (auto iw = 0uz; iw < inputWidth; ++iw)
						{
							const auto inputValue = static_cast<double>(
							    src[((n * inputChannels + ic) * inputHeight + ih) * inputWidth + iw]);
							for (auto ocg = 0uz; ocg < outputChannelsPerGroup; ++ocg)
							{
								const auto oc = outputChannelBegin + ocg;
								for (auto kh = 0uz; kh < kernelHeight; ++kh)
								{
									const auto paddedH = ih * strides[0] + kh * dilations[0];
									if (paddedH < lowPads[0])
									{
										continue;
									}
									const auto oh = paddedH - lowPads[0];
									if (oh >= outputHeight)
									{
										continue;
									}
									for (auto kw = 0uz; kw < kernelWidth; ++kw)
									{
										const auto paddedW = iw * strides[1] + kw * dilations[1];
										if (paddedW < lowPads[1])
										{
											continue;
										}
										const auto ow = paddedW - lowPads[1];
										if (ow >= outputWidth)
										{
											continue;
										}
										const auto weightIndex =
										    ((ic * outputChannelsPerGroup + ocg) * kernelHeight + kh) * kernelWidth + kw;
										const auto outputIndex =
										    ((n * outputChannels + oc) * outputHeight + oh) * outputWidth + ow;
										const auto acc = static_cast<double>(dst[outputIndex]) +
										                 inputValue * static_cast<double>(filter[weightIndex]);
										dst[outputIndex] = static_cast<T>(acc);
									}
								}
							}
						}
					}
				}
			}
		});
		return result;
	}

	inline Tensor<CPU> EvalPool2D(const Tensor<CPU>& input, PoolMode mode,
	                             std::span<const std::size_t> kernelShape,
	                             std::span<const std::size_t> strides,
	                             std::span<const std::size_t> lowPads,
	                             std::span<const std::size_t> highPads,
	                             bool countIncludePad)
	{
		if (input.DType() == DataType::Bool)
		{
			throw std::runtime_error("Pool2D does not support Bool tensors");
		}
		const auto outputShape = Pool2DOutputShape(input.Shape().Dims, kernelShape, strides, lowPads, highPads);
		const auto batch = input.Shape()[0];
		const auto channels = input.Shape()[1];
		const auto height = input.Shape()[2];
		const auto width = input.Shape()[3];
		const auto outHeight = outputShape[2];
		const auto outWidth = outputShape[3];

		CPU cpu;
		Tensor<CPU> result(Uninitialized, outputShape, input.DType(), cpu);
		EnumDispatch(input.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			const auto* src = static_cast<const T*>(input.RawData());
			auto* dst = static_cast<T*>(result.RawData());
			for (auto n = 0uz; n < batch; ++n)
			{
				for (auto c = 0uz; c < channels; ++c)
				{
					for (auto oh = 0uz; oh < outHeight; ++oh)
					{
						for (auto ow = 0uz; ow < outWidth; ++ow)
						{
							double acc = mode == PoolMode::Max ? -std::numeric_limits<double>::infinity() : 0.0;
							auto count = 0uz;
							auto sawValue = false;
							for (auto kh = 0uz; kh < kernelShape[0]; ++kh)
							{
								const auto paddedH = oh * strides[0] + kh;
								const auto validH = paddedH >= lowPads[0] && paddedH - lowPads[0] < height;
								for (auto kw = 0uz; kw < kernelShape[1]; ++kw)
								{
									const auto paddedW = ow * strides[1] + kw;
									const auto validW = paddedW >= lowPads[1] && paddedW - lowPads[1] < width;
									if (!validH || !validW)
									{
										if (mode == PoolMode::Average && countIncludePad)
										{
											++count;
										}
										continue;
									}
									const auto ih = paddedH - lowPads[0];
									const auto iw = paddedW - lowPads[1];
									const auto value = static_cast<double>(src[((n * channels + c) * height + ih) * width + iw]);
									sawValue = true;
									if (mode == PoolMode::Max)
									{
										acc = std::max(acc, value);
									}
									else
									{
										acc += value;
										++count;
									}
								}
							}
							if (mode == PoolMode::Average)
							{
								acc = count == 0 ? 0.0 : acc / static_cast<double>(count);
							}
							else if (!sawValue)
							{
								acc = 0.0;
							}
							dst[((n * channels + c) * outHeight + oh) * outWidth + ow] = static_cast<T>(acc);
						}
					}
				}
			}
		});
		return result;
	}

	inline double UpsampleSourceCoordinate(std::size_t outputIndex, std::size_t inputSize, std::size_t outputSize,
	                                      bool alignCorners)
	{
		if (alignCorners)
		{
			return outputSize == 1 ? 0.0 : static_cast<double>(outputIndex) * static_cast<double>(inputSize - 1) /
			                                static_cast<double>(outputSize - 1);
		}
		const auto coordinate = (static_cast<double>(outputIndex) + 0.5) * static_cast<double>(inputSize) /
		                        static_cast<double>(outputSize) - 0.5;
		return std::max(0.0, coordinate);
	}

	inline std::size_t ClampSpatialIndex(long long index, std::size_t size)
	{
		if (index < 0)
		{
			return 0uz;
		}
		const auto asSize = static_cast<std::size_t>(index);
		return std::min(asSize, size - 1);
	}

	inline double CubicInterpolationWeight(double x)
	{
		constexpr auto alpha = -0.75;
		x = std::abs(x);
		if (x <= 1.0)
		{
			return (alpha + 2.0) * x * x * x - (alpha + 3.0) * x * x + 1.0;
		}
		if (x < 2.0)
		{
			return alpha * x * x * x - 5.0 * alpha * x * x + 8.0 * alpha * x - 4.0 * alpha;
		}
		return 0.0;
	}

	inline Tensor<CPU> EvalUpsample(const Tensor<CPU>& input, UpsampleMode mode,
	                               std::span<const std::size_t> outputSpatialShape,
	                               bool alignCorners)
	{
		const auto outputShape = UpsampleOutputShape(input.Shape().Dims, outputSpatialShape);
		const auto batch = input.Shape()[0];
		const auto channels = input.Shape()[1];
		const auto inputHeight = input.Shape()[2];
		const auto inputWidth = input.Shape()[3];
		const auto outputHeight = outputShape[2];
		const auto outputWidth = outputShape[3];

		if (input.DType() == DataType::Bool && mode != UpsampleMode::Nearest)
		{
			throw std::runtime_error("Upsample only supports Bool tensors in nearest mode");
		}

		CPU cpu;
		Tensor<CPU> result(Uninitialized, outputShape, input.DType(), cpu);
		EnumDispatch(input.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			const auto* src = static_cast<const T*>(input.RawData());
			auto* dst = static_cast<T*>(result.RawData());

			for (auto n = 0uz; n < batch; ++n)
			{
				for (auto c = 0uz; c < channels; ++c)
				{
					for (auto oh = 0uz; oh < outputHeight; ++oh)
					{
						const auto sourceY = UpsampleSourceCoordinate(oh, inputHeight, outputHeight, alignCorners);
						for (auto ow = 0uz; ow < outputWidth; ++ow)
						{
							const auto sourceX = UpsampleSourceCoordinate(ow, inputWidth, outputWidth, alignCorners);
							const auto outputIndex = ((n * channels + c) * outputHeight + oh) * outputWidth + ow;
							if (mode == UpsampleMode::Nearest)
							{
								const auto nearestY = alignCorners ? std::llround(sourceY)
								                                  : static_cast<long long>((oh * inputHeight) / outputHeight);
								const auto nearestX = alignCorners ? std::llround(sourceX)
								                                  : static_cast<long long>((ow * inputWidth) / outputWidth);
								const auto iy = ClampSpatialIndex(nearestY, inputHeight);
								const auto ix = ClampSpatialIndex(nearestX, inputWidth);
								dst[outputIndex] = src[((n * channels + c) * inputHeight + iy) * inputWidth + ix];
							}
							else if (mode == UpsampleMode::Bilinear)
							{
								const auto y0 = ClampSpatialIndex(static_cast<long long>(std::floor(sourceY)), inputHeight);
								const auto x0 = ClampSpatialIndex(static_cast<long long>(std::floor(sourceX)), inputWidth);
								const auto y1 = std::min(y0 + 1, inputHeight - 1);
								const auto x1 = std::min(x0 + 1, inputWidth - 1);
								const auto wy = sourceY - std::floor(sourceY);
								const auto wx = sourceX - std::floor(sourceX);
								const auto v00 = static_cast<double>(src[((n * channels + c) * inputHeight + y0) * inputWidth + x0]);
								const auto v01 = static_cast<double>(src[((n * channels + c) * inputHeight + y0) * inputWidth + x1]);
								const auto v10 = static_cast<double>(src[((n * channels + c) * inputHeight + y1) * inputWidth + x0]);
								const auto v11 = static_cast<double>(src[((n * channels + c) * inputHeight + y1) * inputWidth + x1]);
								const auto top = v00 * (1.0 - wx) + v01 * wx;
								const auto bottom = v10 * (1.0 - wx) + v11 * wx;
								dst[outputIndex] = static_cast<T>(top * (1.0 - wy) + bottom * wy);
							}
							else if (mode == UpsampleMode::Bicubic)
							{
								const auto yBase = static_cast<long long>(std::floor(sourceY));
								const auto xBase = static_cast<long long>(std::floor(sourceX));
								auto acc = 0.0;
								for (auto ky = -1; ky <= 2; ++ky)
								{
									const auto iy = ClampSpatialIndex(yBase + ky, inputHeight);
									const auto wy = CubicInterpolationWeight(sourceY - static_cast<double>(yBase + ky));
									for (auto kx = -1; kx <= 2; ++kx)
									{
										const auto ix = ClampSpatialIndex(xBase + kx, inputWidth);
										const auto wx = CubicInterpolationWeight(sourceX - static_cast<double>(xBase + kx));
										const auto value =
										    static_cast<double>(src[((n * channels + c) * inputHeight + iy) * inputWidth + ix]);
										acc += value * wy * wx;
									}
								}
								dst[outputIndex] = static_cast<T>(acc);
							}
							else
							{
								throw std::runtime_error("Unsupported UpsampleMode");
							}
						}
					}
				}
			}
		});
		return result;
	}
} // namespace LiteNN::Detail

#endif
