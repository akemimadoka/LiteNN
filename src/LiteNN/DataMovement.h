#ifndef LITENN_DATAMOVEMENT_H
#define LITENN_DATAMOVEMENT_H

#include <LiteNN/Graph.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <format>
#include <ranges>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace LiteNN::Detail
{
	inline void ValidatePositiveShape(std::span<const std::size_t> shape, std::string_view label)
	{
		for (const auto dim : shape)
		{
			if (dim == 0)
			{
				throw std::runtime_error(std::format("{} dimensions must be greater than 0", label));
			}
		}
	}

	inline std::vector<std::size_t> BroadcastToShape(std::span<const std::size_t> inputShape,
	                                                std::span<const std::size_t> targetShape)
	{
		ValidatePositiveShape(targetShape, "BroadcastTo target shape");
		if (targetShape.size() < inputShape.size())
		{
			throw std::runtime_error("BroadcastTo target rank must be >= input rank");
		}
		const auto rankDelta = targetShape.size() - inputShape.size();
		for (auto dim = 0uz; dim < inputShape.size(); ++dim)
		{
			const auto inputDim = inputShape[dim];
			const auto targetDim = targetShape[rankDelta + dim];
			if (inputDim != targetDim && inputDim != 1)
			{
				throw std::runtime_error(std::format(
				    "BroadcastTo input dim {} (size {}) cannot broadcast to target dim {} (size {})",
				    dim, inputDim, rankDelta + dim, targetDim));
			}
		}
		return { targetShape.begin(), targetShape.end() };
	}

	inline std::vector<std::size_t> PadOutputShape(std::span<const std::size_t> inputShape,
	                                              std::span<const std::size_t> lowPads,
	                                              std::span<const std::size_t> highPads)
	{
		if (inputShape.empty())
		{
			throw std::runtime_error("Pad input must have rank >= 1");
		}
		if (lowPads.size() != inputShape.size() || highPads.size() != inputShape.size())
		{
			throw std::runtime_error("Pad low/high padding ranks must match input rank");
		}
		auto outputShape = std::vector<std::size_t>(inputShape.begin(), inputShape.end());
		for (auto dim = 0uz; dim < inputShape.size(); ++dim)
		{
			outputShape[dim] += lowPads[dim] + highPads[dim];
		}
		return outputShape;
	}

	inline std::vector<std::size_t> GatherOutputShape(std::span<const std::size_t> dataShape,
	                                                 std::span<const std::size_t> indexShape,
	                                                 std::size_t axis)
	{
		if (dataShape.empty())
		{
			throw std::runtime_error("Gather data must have rank >= 1");
		}
		if (axis >= dataShape.size())
		{
			throw std::runtime_error("Gather axis out of range");
		}
		std::vector<std::size_t> outputShape;
		outputShape.insert(outputShape.end(), dataShape.begin(), dataShape.begin() + static_cast<std::ptrdiff_t>(axis));
		outputShape.insert(outputShape.end(), indexShape.begin(), indexShape.end());
		outputShape.insert(outputShape.end(), dataShape.begin() + static_cast<std::ptrdiff_t>(axis + 1),
		                   dataShape.end());
		return outputShape;
	}

	inline std::vector<std::size_t> ScatterUpdatesShape(std::span<const std::size_t> dataShape,
	                                                   std::span<const std::size_t> indexShape,
	                                                   std::size_t axis)
	{
		return GatherOutputShape(dataShape, indexShape, axis);
	}

	inline std::vector<std::size_t> RowMajorStrides(std::span<const std::size_t> shape)
	{
		std::vector<std::size_t> strides(shape.size(), 1uz);
		for (auto dim = shape.size(); dim-- > 1;)
		{
			strides[dim - 1] = strides[dim] * shape[dim];
		}
		return strides;
	}

	template <typename T>
	T CastPadValue(double value)
	{
		if constexpr (std::same_as<T, bool>)
		{
			return value != 0.0;
		}
		else
		{
			return static_cast<T>(value);
		}
	}

	inline std::int64_t ReflectIndex(std::int64_t index, std::int64_t size)
	{
		if (size < 2)
		{
			throw std::runtime_error("Reflect padding requires padded dimensions to have size >= 2");
		}
		while (index < 0 || index >= size)
		{
			index = index < 0 ? -index : 2 * size - 2 - index;
		}
		return index;
	}

	template <typename IndexT>
	std::size_t CheckedIndex(IndexT rawIndex, std::size_t upperBound, std::string_view label)
	{
		if constexpr (std::is_signed_v<IndexT>)
		{
			if (rawIndex < 0)
			{
				throw std::runtime_error(std::format("{} index out of range", label));
			}
		}
		const auto index = static_cast<std::size_t>(rawIndex);
		if (index >= upperBound)
		{
			throw std::runtime_error(std::format("{} index {} out of range for axis size {}", label, index, upperBound));
		}
		return index;
	}

	inline Tensor<CPU> EvalBroadcastTo(const Tensor<CPU>& input, std::span<const std::size_t> targetShape)
	{
		const auto outputShape = BroadcastToShape(input.Shape().Dims, targetShape);
		CPU cpu;
		Tensor<CPU> result(Uninitialized, outputShape, input.DType(), cpu);
		const auto inputRank = input.Shape().NumDim();
		const auto outputRank = outputShape.size();
		const auto rankDelta = outputRank - inputRank;
		const auto inputStrides = RowMajorStrides(input.Shape().Dims);
		const auto outputStrides = RowMajorStrides(outputShape);
		const auto outputElements = ShapeView{ outputShape }.NumElements();

		EnumDispatch(input.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			const auto* src = static_cast<const T*>(input.RawData());
			auto* dst = static_cast<T*>(result.RawData());

			for (auto linear = 0uz; linear < outputElements; ++linear)
			{
				auto remaining = linear;
				auto srcOffset = 0uz;
				for (auto dim = 0uz; dim < outputRank; ++dim)
				{
					const auto coord = outputStrides.empty() ? 0uz : remaining / outputStrides[dim];
					if (!outputStrides.empty())
					{
						remaining %= outputStrides[dim];
					}
					if (dim >= rankDelta)
					{
						const auto inputDim = dim - rankDelta;
						const auto inputCoord = input.Shape()[inputDim] == 1 ? 0uz : coord;
						srcOffset += inputCoord * inputStrides[inputDim];
					}
				}
				dst[linear] = src[srcOffset];
			}
		});
		return result;
	}

	inline Tensor<CPU> EvalPad(const Tensor<CPU>& input, std::span<const std::size_t> lowPads,
	                          std::span<const std::size_t> highPads, PadMode mode, double constantValue)
	{
		const auto outputShape = PadOutputShape(input.Shape().Dims, lowPads, highPads);
		CPU cpu;
		Tensor<CPU> result(Uninitialized, outputShape, input.DType(), cpu);
		const auto rank = input.Shape().NumDim();
		const auto inputStrides = RowMajorStrides(input.Shape().Dims);
		const auto outputStrides = RowMajorStrides(outputShape);
		const auto outputElements = ShapeView{ outputShape }.NumElements();

		EnumDispatch(input.DType(), [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			const auto* src = static_cast<const T*>(input.RawData());
			auto* dst = static_cast<T*>(result.RawData());
			const auto padValue = CastPadValue<T>(constantValue);

			for (auto linear = 0uz; linear < outputElements; ++linear)
			{
				auto remaining = linear;
				auto srcOffset = 0uz;
				bool useConstant = false;
				for (auto dim = 0uz; dim < rank; ++dim)
				{
					const auto outCoord = remaining / outputStrides[dim];
					remaining %= outputStrides[dim];
					auto srcCoord = static_cast<std::int64_t>(outCoord) - static_cast<std::int64_t>(lowPads[dim]);
					if (srcCoord < 0 || srcCoord >= static_cast<std::int64_t>(input.Shape()[dim]))
					{
						switch (mode)
						{
						case PadMode::Constant:
							useConstant = true;
							break;
						case PadMode::Reflect:
							srcCoord = ReflectIndex(srcCoord, static_cast<std::int64_t>(input.Shape()[dim]));
							break;
						case PadMode::Replicate:
							srcCoord = std::clamp<std::int64_t>(srcCoord, 0, static_cast<std::int64_t>(input.Shape()[dim]) - 1);
							break;
						}
					}
					if (!useConstant)
					{
						srcOffset += static_cast<std::size_t>(srcCoord) * inputStrides[dim];
					}
				}
				dst[linear] = useConstant ? padValue : src[srcOffset];
			}
		});
		return result;
	}

	inline Tensor<CPU> EvalGather(const Tensor<CPU>& data, const Tensor<CPU>& indices, std::size_t axis)
	{
		const auto outputShape = GatherOutputShape(data.Shape().Dims, indices.Shape().Dims, axis);
		CPU cpu;
		Tensor<CPU> result(Uninitialized, outputShape, data.DType(), cpu);
		const auto axisDim = data.Shape()[axis];
		auto outer = 1uz;
		for (auto dim = 0uz; dim < axis; ++dim)
		{
			outer *= data.Shape()[dim];
		}
		auto inner = 1uz;
		for (auto dim = axis + 1; dim < data.Shape().NumDim(); ++dim)
		{
			inner *= data.Shape()[dim];
		}
		const auto indexCount = indices.NumElements();

		EnumDispatch(data.DType(), [&]<DataType DataTypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<DataTypeValue>;
			auto run = [&]<typename IndexT>() {
				const auto* src = static_cast<const T*>(data.RawData());
				const auto* idx = static_cast<const IndexT*>(indices.RawData());
				auto* dst = static_cast<T*>(result.RawData());
				for (auto outOuter = 0uz; outOuter < outer; ++outOuter)
				{
					for (auto indexLinear = 0uz; indexLinear < indexCount; ++indexLinear)
					{
						const auto gathered = CheckedIndex(idx[indexLinear], axisDim, "Gather");
						for (auto innerIdx = 0uz; innerIdx < inner; ++innerIdx)
						{
							dst[(outOuter * indexCount + indexLinear) * inner + innerIdx] =
							    src[(outOuter * axisDim + gathered) * inner + innerIdx];
						}
					}
				}
			};
			switch (indices.DType())
			{
			case DataType::Int32:
				run.template operator()<std::int32_t>();
				break;
			case DataType::Int64:
				run.template operator()<std::int64_t>();
				break;
			default:
				throw std::runtime_error("Gather indices must have dtype Int32 or Int64");
			}
		});
		return result;
	}

	inline Tensor<CPU> EvalScatter(const Tensor<CPU>& data, const Tensor<CPU>& indices, const Tensor<CPU>& updates,
	                              std::size_t axis, ScatterMode mode)
	{
		const auto expectedUpdatesShape = ScatterUpdatesShape(data.Shape().Dims, indices.Shape().Dims, axis);
		if (updates.DType() != data.DType() || !std::ranges::equal(updates.Shape().Dims, expectedUpdatesShape))
		{
			throw std::runtime_error("Scatter updates shape or dtype does not match data/indices/axis");
		}
		if (mode == ScatterMode::Add && data.DType() == DataType::Bool)
		{
			throw std::runtime_error("Scatter Add mode does not support Bool tensors");
		}
		CPU cpu;
		Tensor<CPU> result(Uninitialized, data.Shape(), data.DType(), cpu);
		DeviceTraits<CPU>::ConvertTo(cpu, data.DType(), data.RawData(), data.NumElements(), data.DType(), result.RawData());

		const auto axisDim = data.Shape()[axis];
		auto outer = 1uz;
		for (auto dim = 0uz; dim < axis; ++dim)
		{
			outer *= data.Shape()[dim];
		}
		auto inner = 1uz;
		for (auto dim = axis + 1; dim < data.Shape().NumDim(); ++dim)
		{
			inner *= data.Shape()[dim];
		}
		const auto indexCount = indices.NumElements();

		EnumDispatch(data.DType(), [&]<DataType DataTypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<DataTypeValue>;
			auto run = [&]<typename IndexT>() {
				const auto* idx = static_cast<const IndexT*>(indices.RawData());
				const auto* updatePtr = static_cast<const T*>(updates.RawData());
				auto* dst = static_cast<T*>(result.RawData());
				for (auto outOuter = 0uz; outOuter < outer; ++outOuter)
				{
					for (auto indexLinear = 0uz; indexLinear < indexCount; ++indexLinear)
					{
						const auto scattered = CheckedIndex(idx[indexLinear], axisDim, "Scatter");
						for (auto innerIdx = 0uz; innerIdx < inner; ++innerIdx)
						{
							auto& target = dst[(outOuter * axisDim + scattered) * inner + innerIdx];
							const auto update = updatePtr[(outOuter * indexCount + indexLinear) * inner + innerIdx];
							switch (mode)
							{
							case ScatterMode::Update:
								target = update;
								break;
							case ScatterMode::Add:
								if constexpr (std::same_as<T, bool>)
								{
									throw std::runtime_error("Scatter Add mode does not support Bool tensors");
								}
								else
								{
									target += update;
								}
								break;
							}
						}
					}
				}
			};
			switch (indices.DType())
			{
			case DataType::Int32:
				run.template operator()<std::int32_t>();
				break;
			case DataType::Int64:
				run.template operator()<std::int64_t>();
				break;
			default:
				throw std::runtime_error("Scatter indices must have dtype Int32 or Int64");
			}
		});
		return result;
	}
} // namespace LiteNN::Detail

#endif
