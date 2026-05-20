#include <LiteNN/Graph.h>

#include <stdexcept>
#include <utility>
#include <vector>

#ifndef LITENN_LAYER_SUMROWS_H
#define LITENN_LAYER_SUMROWS_H

namespace LiteNN::Layer
{
	inline NodeOutput AddSumRows(Subgraph& subgraph, NodeOutput input)
	{
		const auto info = subgraph.GetOutputInfo(input);
		if (info.shape.empty())
		{
			throw std::runtime_error("SumRows requires at least one dimension");
		}

		std::vector<std::size_t> reducedShape;
		reducedShape.reserve(info.shape.size());
		for (auto dim = 1uz; dim < info.shape.size(); ++dim)
		{
			reducedShape.push_back(info.shape[dim]);
		}
		if (reducedShape.empty())
		{
			reducedShape.push_back(1);
		}

		auto outputShape = info.shape;
		outputShape[0] = 1;

		const auto reduced = subgraph.AddNode(ReduceOpNode{ ReduceOp::Sum, input, 0 },
		                                     { OutputInfo{ info.dtype, reducedShape } });
		const auto reshaped = subgraph.AddNode(ReshapeNode{ { reduced, 0 }, outputShape },
		                                      { OutputInfo{ info.dtype, outputShape } });
		return { reshaped, 0 };
	}

	inline SubgraphId BuildSumRows(Graph& graph, DataType dtype, ShapeView shape)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddSumRows(subgraph, { input, 0 });
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif