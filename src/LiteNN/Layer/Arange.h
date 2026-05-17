#include <LiteNN/Layer/LayerUtils.h>

#include <stdexcept>
#include <utility>

#ifndef LITENN_LAYER_ARANGE_H
#define LITENN_LAYER_ARANGE_H

namespace LiteNN::Layer
{
	namespace Detail
	{
		inline Tensor<CPU> MakeArangeTensor(std::size_t length, DataType dtype, double start, double step)
		{
			if (dtype == DataType::Bool)
			{
				throw std::runtime_error("Arange does not support boolean tensors");
			}

			Tensor<CPU> tensor(Uninitialized, { length }, dtype);
			EnumDispatch(dtype, [&]<DataType TypeValue> {
				using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
				auto* data = static_cast<T*>(tensor.RawData());
				for (auto index = 0uz; index < length; ++index)
				{
					data[index] = static_cast<T>(start + step * static_cast<double>(index));
				}
			});
			return tensor;
		}
	} // namespace Detail

	inline NodeOutput AddArange(Subgraph& subgraph, DataType dtype, std::size_t length, double start = 0.0,
	                           double step = 1.0)
	{
		const auto constant = Detail::MakeArangeTensor(length, dtype, start, step);
		const auto node = Detail::AddConstant(subgraph, constant);
		return { node, 0 };
	}

	inline SubgraphId BuildArange(Graph& graph, DataType dtype, std::size_t length, double start = 0.0,
	                             double step = 1.0)
	{
		Subgraph subgraph;
		const auto result = AddArange(subgraph, dtype, length, start, step);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif