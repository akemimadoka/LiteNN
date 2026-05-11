#include <LiteNN/Graph.h>

#include <functional>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef LITENN_LAYER_FLATTEN_H
#define LITENN_LAYER_FLATTEN_H

namespace LiteNN::Layer
{
	inline std::vector<std::size_t> FlattenShape(ShapeView shape, std::size_t startDim = 1)
	{
		if (startDim >= shape.NumDim())
		{
			throw std::runtime_error("Flatten startDim must be smaller than the input rank");
		}

		std::vector<std::size_t> result(shape.Dims.begin(), shape.Dims.begin() + startDim);
		const auto flatSize = std::accumulate(shape.Dims.begin() + startDim, shape.Dims.end(), 1uz, std::multiplies{});
		result.push_back(flatSize);
		return result;
	}

	inline NodeOutput AddFlatten(Subgraph& subgraph, NodeOutput input, std::size_t startDim = 1)
	{
		const auto info = subgraph.GetOutputInfo(input);
		auto outputShape = FlattenShape(info.shape, startDim);
		const auto result = subgraph.AddNode(ReshapeNode{ input, outputShape }, { OutputInfo{ info.dtype, outputShape } });
		return { result, 0 };
	}

	inline SubgraphId BuildFlatten(Graph& graph, DataType dtype, ShapeView shape, std::size_t startDim = 1)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddFlatten(subgraph, { input, 0 }, startDim);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
