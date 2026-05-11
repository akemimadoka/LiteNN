#include <LiteNN/Device.h>
#include <LiteNN/Graph.h>
#include <algorithm>

#ifndef LITENN_LAYER_LAYERUTILS_H
#define LITENN_LAYER_LAYERUTILS_H

namespace LiteNN::Layer::Detail
{
	inline Tensor<CPU> MakeFilledTensor(ShapeView shape, DataType dtype, double value)
	{
		Tensor<CPU> tensor(Uninitialized, shape, dtype);
		EnumDispatch(dtype, [&]<DataType TypeValue> {
			using T = typename DeviceTraits<CPU>::template DataTypeMapping<TypeValue>;
			auto* data = static_cast<T*>(tensor.RawData());
			std::fill(data, data + tensor.NumElements(), static_cast<T>(value));
		});
		return tensor;
	}

	inline Tensor<CPU> MakeScalarTensor(DataType dtype, double value)
	{
		return MakeFilledTensor({ 1 }, dtype, value);
	}

	inline NodeId AddConstant(Subgraph& subgraph, const Tensor<CPU>& tensor)
	{
		return subgraph.AddNode(ConstantNode{ tensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
		                        { OutputInfo{ tensor.DType(), tensor.Shape().ToOwned() } });
	}
} // namespace LiteNN::Layer::Detail

#endif
