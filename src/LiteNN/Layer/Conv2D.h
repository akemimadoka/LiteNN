#ifndef LITENN_LAYER_CONV2D_H
#define LITENN_LAYER_CONV2D_H

#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/Graph.h>

#include <optional>
#include <stdexcept>
#include <vector>

namespace LiteNN::Layer
{
	inline NodeOutput AddConv2D(Subgraph& subgraph, NodeOutput input, NodeOutput weight,
	                            std::optional<NodeOutput> bias = std::nullopt,
	                            std::vector<std::size_t> strides = { 1, 1 },
	                            std::vector<std::size_t> dilations = { 1, 1 },
	                            std::vector<std::size_t> lowPads = { 0, 0 },
	                            std::vector<std::size_t> highPads = { 0, 0 },
	                            std::size_t groupCount = 1)
	{
		const auto inputInfo = subgraph.GetOutputInfo(input);
		const auto weightInfo = subgraph.GetOutputInfo(weight);
		if (inputInfo.dtype == DataType::Bool)
		{
			throw std::runtime_error("Conv2D does not support Bool tensors");
		}
		if (inputInfo.dtype != weightInfo.dtype)
		{
			throw std::runtime_error("Conv2D input and weight dtypes must match");
		}
		const auto outputShape = ::LiteNN::Detail::Conv2DOutputShape(
		    inputInfo.shape, weightInfo.shape, strides, dilations, lowPads, highPads, groupCount);
		if (bias)
		{
			const auto biasInfo = subgraph.GetOutputInfo(*bias);
			if (biasInfo.dtype != inputInfo.dtype)
			{
				throw std::runtime_error("Conv2D bias dtype must match input dtype");
			}
			::LiteNN::Detail::ValidateConv2DBiasShape(biasInfo.shape, outputShape[1]);
		}

		const auto result = subgraph.AddNode(
		    Conv2DNode{ input, weight, bias, std::move(strides), std::move(dilations), std::move(lowPads),
		                std::move(highPads), groupCount },
		    { OutputInfo{ inputInfo.dtype, outputShape } });
		return { result, 0 };
	}

	inline SubgraphId BuildConv2D(Graph& graph, DataType dtype, ShapeView inputShape, ShapeView weightShape,
	                              std::optional<ShapeView> biasShape = std::nullopt,
	                              std::vector<std::size_t> strides = { 1, 1 },
	                              std::vector<std::size_t> dilations = { 1, 1 },
	                              std::vector<std::size_t> lowPads = { 0, 0 },
	                              std::vector<std::size_t> highPads = { 0, 0 },
	                              std::size_t groupCount = 1)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, inputShape.ToOwned());
		const auto weight = subgraph.AddParam(dtype, weightShape.ToOwned());
		std::optional<NodeOutput> bias;
		if (biasShape)
		{
			bias = NodeOutput{ subgraph.AddParam(dtype, biasShape->ToOwned()), 0 };
		}
		const auto result = AddConv2D(subgraph, { input, 0 }, { weight, 0 }, bias, std::move(strides),
		                              std::move(dilations), std::move(lowPads), std::move(highPads), groupCount);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
