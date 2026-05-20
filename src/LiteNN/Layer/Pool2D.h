#ifndef LITENN_LAYER_POOL2D_H
#define LITENN_LAYER_POOL2D_H

#include <LiteNN/ComputePrimitives.h>
#include <LiteNN/Graph.h>

#include <stdexcept>
#include <vector>

namespace LiteNN::Layer
{
	inline NodeOutput AddPool2D(Subgraph& subgraph, NodeOutput input, PoolMode mode,
	                            std::vector<std::size_t> kernelShape,
	                            std::vector<std::size_t> strides,
	                            std::vector<std::size_t> lowPads = { 0, 0 },
	                            std::vector<std::size_t> highPads = { 0, 0 },
	                            bool countIncludePad = false)
	{
		const auto info = subgraph.GetOutputInfo(input);
		if (info.dtype == DataType::Bool)
		{
			throw std::runtime_error("Pool2D does not support Bool tensors");
		}
		const auto outputShape = ::LiteNN::Detail::Pool2DOutputShape(info.shape, kernelShape, strides,
		                                                             lowPads, highPads);
		const auto result = subgraph.AddNode(
		    Pool2DNode{ input, mode, std::move(kernelShape), std::move(strides), std::move(lowPads),
		                std::move(highPads), countIncludePad },
		    { OutputInfo{ info.dtype, outputShape } });
		return { result, 0 };
	}

	inline NodeOutput AddMaxPool2D(Subgraph& subgraph, NodeOutput input,
	                               std::vector<std::size_t> kernelShape,
	                               std::vector<std::size_t> strides,
	                               std::vector<std::size_t> lowPads = { 0, 0 },
	                               std::vector<std::size_t> highPads = { 0, 0 })
	{
		return AddPool2D(subgraph, input, PoolMode::Max, std::move(kernelShape), std::move(strides),
		                 std::move(lowPads), std::move(highPads));
	}

	inline NodeOutput AddAveragePool2D(Subgraph& subgraph, NodeOutput input,
	                                   std::vector<std::size_t> kernelShape,
	                                   std::vector<std::size_t> strides,
	                                   std::vector<std::size_t> lowPads = { 0, 0 },
	                                   std::vector<std::size_t> highPads = { 0, 0 },
	                                   bool countIncludePad = false)
	{
		return AddPool2D(subgraph, input, PoolMode::Average, std::move(kernelShape), std::move(strides),
		                 std::move(lowPads), std::move(highPads), countIncludePad);
	}

	inline SubgraphId BuildPool2D(Graph& graph, DataType dtype, ShapeView inputShape, PoolMode mode,
	                              std::vector<std::size_t> kernelShape,
	                              std::vector<std::size_t> strides,
	                              std::vector<std::size_t> lowPads = { 0, 0 },
	                              std::vector<std::size_t> highPads = { 0, 0 },
	                              bool countIncludePad = false)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, inputShape.ToOwned());
		const auto result = AddPool2D(subgraph, { input, 0 }, mode, std::move(kernelShape), std::move(strides),
		                              std::move(lowPads), std::move(highPads), countIncludePad);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
