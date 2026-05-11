#include <LiteNN/Device.h>
#include <LiteNN/Tensor.h>

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>

#ifndef LITENN_INITIALIZER_INITIALIZER_H
#define LITENN_INITIALIZER_INITIALIZER_H

namespace LiteNN::Initializer
{
	struct FanInFanOut
	{
		std::size_t fanIn{};
		std::size_t fanOut{};
	};

	namespace Detail
	{
		inline void ValidateFloatingDType(DataType dtype)
		{
			if (dtype != DataType::Float32 && dtype != DataType::Float64)
			{
				throw std::runtime_error("Random initializers require Float32 or Float64 tensors");
			}
		}

		template <typename Generator>
		inline Tensor<CPU> GenerateFloatingTensor(ShapeView shape, DataType dtype, Generator&& generator)
		{
			ValidateFloatingDType(dtype);
			Tensor<CPU> tensor(Uninitialized, shape, dtype);
			EnumDispatch(dtype, [&]<DataType TypeValue> {
				if constexpr (TypeValue == DataType::Float32 || TypeValue == DataType::Float64)
				{
					using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
					auto* data = static_cast<T*>(tensor.RawData());
					for (std::size_t i = 0; i < tensor.NumElements(); ++i)
					{
						data[i] = static_cast<T>(generator());
					}
				}
			});
			return tensor;
		}
	} // namespace Detail

	inline FanInFanOut ComputeFanInFanOut(ShapeView shape)
	{
		if (shape.NumDim() < 2)
		{
			throw std::runtime_error("Fan-in/fan-out initializers require tensors with at least 2 dimensions");
		}
		if (std::ranges::any_of(shape.Dims, [](std::size_t dim) { return dim == 0; }))
		{
			throw std::runtime_error("Initializer shape dimensions must be greater than 0");
		}

		if (shape.NumDim() == 2)
		{
			return { .fanIn = shape[0], .fanOut = shape[1] };
		}

		const auto receptiveFieldSize =
		    std::accumulate(shape.Dims.begin() + 2, shape.Dims.end(), 1uz, std::multiplies{});
		return { .fanIn = shape[1] * receptiveFieldSize, .fanOut = shape[0] * receptiveFieldSize };
	}

	inline Tensor<CPU> Constant(ShapeView shape, double value, DataType dtype = DataType::Float32)
	{
		Tensor<CPU> tensor(Uninitialized, shape, dtype);
		EnumDispatch(dtype, [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			auto* data = static_cast<T*>(tensor.RawData());
			std::fill(data, data + tensor.NumElements(), static_cast<T>(value));
		});
		return tensor;
	}

	inline Tensor<CPU> Zeros(ShapeView shape, DataType dtype = DataType::Float32)
	{
		return Tensor<CPU>(shape, dtype);
	}

	inline Tensor<CPU> Ones(ShapeView shape, DataType dtype = DataType::Float32)
	{
		return Constant(shape, 1.0, dtype);
	}

	template <typename UniformRandomBitGenerator>
	inline Tensor<CPU> Uniform(ShapeView shape, double low, double high, UniformRandomBitGenerator& rng,
	                           DataType dtype = DataType::Float32)
	{
		if (!(low <= high))
		{
			throw std::runtime_error("Uniform initializer requires low <= high");
		}

		std::uniform_real_distribution<double> distribution(low, high);
		return Detail::GenerateFloatingTensor(shape, dtype, [&] { return distribution(rng); });
	}

	template <typename UniformRandomBitGenerator>
	inline Tensor<CPU> Normal(ShapeView shape, double mean, double stddev, UniformRandomBitGenerator& rng,
	                          DataType dtype = DataType::Float32)
	{
		if (!(stddev >= 0.0))
		{
			throw std::runtime_error("Normal initializer requires non-negative stddev");
		}

		std::normal_distribution<double> distribution(mean, stddev);
		return Detail::GenerateFloatingTensor(shape, dtype, [&] { return distribution(rng); });
	}

	template <typename UniformRandomBitGenerator>
	inline Tensor<CPU> XavierUniform(ShapeView shape, UniformRandomBitGenerator& rng, double gain = 1.0,
	                                 DataType dtype = DataType::Float32)
	{
		const auto [fanIn, fanOut] = ComputeFanInFanOut(shape);
		const auto bound = gain * std::sqrt(6.0 / static_cast<double>(fanIn + fanOut));
		return Uniform(shape, -bound, bound, rng, dtype);
	}

	template <typename UniformRandomBitGenerator>
	inline Tensor<CPU> XavierNormal(ShapeView shape, UniformRandomBitGenerator& rng, double gain = 1.0,
	                                DataType dtype = DataType::Float32)
	{
		const auto [fanIn, fanOut] = ComputeFanInFanOut(shape);
		const auto stddev = gain * std::sqrt(2.0 / static_cast<double>(fanIn + fanOut));
		return Normal(shape, 0.0, stddev, rng, dtype);
	}

	template <typename UniformRandomBitGenerator>
	inline Tensor<CPU> KaimingUniform(ShapeView shape, UniformRandomBitGenerator& rng, double negativeSlope = 0.0,
	                                  DataType dtype = DataType::Float32)
	{
		const auto [fanIn, fanOut] = ComputeFanInFanOut(shape);
		(void)fanOut;
		const auto gain = std::sqrt(2.0 / (1.0 + negativeSlope * negativeSlope));
		const auto bound = gain * std::sqrt(3.0 / static_cast<double>(fanIn));
		return Uniform(shape, -bound, bound, rng, dtype);
	}

	template <typename UniformRandomBitGenerator>
	inline Tensor<CPU> KaimingNormal(ShapeView shape, UniformRandomBitGenerator& rng, double negativeSlope = 0.0,
	                                 DataType dtype = DataType::Float32)
	{
		const auto [fanIn, fanOut] = ComputeFanInFanOut(shape);
		(void)fanOut;
		const auto gain = std::sqrt(2.0 / (1.0 + negativeSlope * negativeSlope));
		const auto stddev = gain / std::sqrt(static_cast<double>(fanIn));
		return Normal(shape, 0.0, stddev, rng, dtype);
	}
} // namespace LiteNN::Initializer

#endif
