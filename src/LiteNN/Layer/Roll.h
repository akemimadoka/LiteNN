#include <LiteNN/Graph.h>

#include <stdexcept>
#include <utility>

#ifndef LITENN_LAYER_ROLL_H
#define LITENN_LAYER_ROLL_H

namespace LiteNN::Layer
{
	inline NodeOutput AddRoll(Subgraph& subgraph, NodeOutput input, std::size_t axis, std::ptrdiff_t shift)
	{
		const auto info = subgraph.GetOutputInfo(input);
		if (axis >= info.shape.size())
		{
			throw std::runtime_error("Roll axis out of range");
		}

		const auto axisExtent = static_cast<std::ptrdiff_t>(info.shape[axis]);
		if (axisExtent <= 0)
		{
			throw std::runtime_error("Roll axis must have positive extent");
		}

		auto normalizedShift = shift % axisExtent;
		if (normalizedShift < 0)
		{
			normalizedShift += axisExtent;
		}
		if (normalizedShift == 0)
		{
			return input;
		}

		const auto tailLength = static_cast<std::size_t>(axisExtent - normalizedShift);
		const auto headStart = tailLength;
		const auto headLength = static_cast<std::size_t>(normalizedShift);

		auto headShape = info.shape;
		headShape[axis] = headLength;
		auto tailShape = info.shape;
		tailShape[axis] = tailLength;

		const auto head = subgraph.AddNode(SliceNode{ input, axis, headStart, headLength },
		                                  { OutputInfo{ info.dtype, std::move(headShape) } });
		const auto tail = subgraph.AddNode(SliceNode{ input, axis, 0, tailLength },
		                                  { OutputInfo{ info.dtype, std::move(tailShape) } });
		const auto rolled = subgraph.AddNode(ConcatNode{ { { head, 0 }, { tail, 0 } }, axis },
		                                    { OutputInfo{ info.dtype, info.shape } });
		return { rolled, 0 };
	}

	inline SubgraphId BuildRoll(Graph& graph, DataType dtype, ShapeView shape, std::size_t axis,
	                           std::ptrdiff_t shift)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddRoll(subgraph, { input, 0 }, axis, shift);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif