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
} // namespace LiteNN::Detail

#endif
