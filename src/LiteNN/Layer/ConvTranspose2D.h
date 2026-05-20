#ifndef LITENN_LAYER_CONVTRANSPOSE2D_H
#define LITENN_LAYER_CONVTRANSPOSE2D_H

#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/Graph.h>

#include <optional>
#include <stdexcept>
#include <vector>

namespace LiteNN::Layer
{
	inline NodeOutput AddConvTranspose2D(Subgraph& subgraph, NodeOutput input, NodeOutput weight,
	                                     std::optional<NodeOutput> bias = std::nullopt,
	                                     std::vector<std::size_t> strides = { 1, 1 },
	                                     std::vector<std::size_t> dilations = { 1, 1 },
	                                     std::vector<std::size_t> lowPads = { 0, 0 },
	                                     std::vector<std::size_t> highPads = { 0, 0 },
	                                     std::vector<std::size_t> outputPads = { 0, 0 },
	                                     std::size_t groupCount = 1)
	{
		const auto inputInfo = subgraph.GetOutputInfo(input);
		const auto weightInfo = subgraph.GetOutputInfo(weight);
		if (inputInfo.dtype == DataType::Bool)
		{
			throw std::runtime_error("ConvTranspose2D does not support Bool tensors");
		}
		if (inputInfo.dtype != weightInfo.dtype)
		{
			throw std::runtime_error("ConvTranspose2D input and weight dtypes must match");
		}
		const auto outputShape = ::LiteNN::Detail::ConvTranspose2DOutputShape(
		    inputInfo.shape, weightInfo.shape, strides, dilations, lowPads, highPads, outputPads, groupCount);
		if (bias)
		{
			const auto biasInfo = subgraph.GetOutputInfo(*bias);
			if (biasInfo.dtype != inputInfo.dtype)
			{
				throw std::runtime_error("ConvTranspose2D bias dtype must match input dtype");
			}
			::LiteNN::Detail::ValidateConv2DBiasShape(biasInfo.shape, outputShape[1]);
		}

		const auto result = subgraph.AddNode(
		    ConvTranspose2DNode{ input, weight, bias, std::move(strides), std::move(dilations), std::move(lowPads),
		                         std::move(highPads), std::move(outputPads), groupCount },
		    { OutputInfo{ inputInfo.dtype, outputShape } });
		return { result, 0 };
	}

	inline SubgraphId BuildConvTranspose2D(Graph& graph, DataType dtype, ShapeView inputShape,
	                                       ShapeView weightShape, std::optional<ShapeView> biasShape = std::nullopt,
	                                       std::vector<std::size_t> strides = { 1, 1 },
	                                       std::vector<std::size_t> dilations = { 1, 1 },
	                                       std::vector<std::size_t> lowPads = { 0, 0 },
	                                       std::vector<std::size_t> highPads = { 0, 0 },
	                                       std::vector<std::size_t> outputPads = { 0, 0 },
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
		const auto result = AddConvTranspose2D(subgraph, { input, 0 }, { weight, 0 }, bias, std::move(strides),
		                                       std::move(dilations), std::move(lowPads), std::move(highPads),
		                                       std::move(outputPads), groupCount);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
