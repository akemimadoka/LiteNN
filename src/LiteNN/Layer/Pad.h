#include <LiteNN/DataMovement.h>
#include <LiteNN/Graph.h>

#include <span>
#include <stdexcept>
#include <vector>

#ifndef LITENN_LAYER_PAD_H
#define LITENN_LAYER_PAD_H

namespace LiteNN::Layer
{
	inline NodeOutput AddPad(Subgraph& subgraph, NodeOutput input, std::span<const std::size_t> lowPads,
	                         std::span<const std::size_t> highPads,
	                         PadMode mode = PadMode::Constant, double constantValue = 0.0)
	{
		const auto info = subgraph.GetOutputInfo(input);
		auto low = std::vector<std::size_t>(lowPads.begin(), lowPads.end());
		auto high = std::vector<std::size_t>(highPads.begin(), highPads.end());
		auto outputShape = ::LiteNN::Detail::PadOutputShape(info.shape, low, high);
		const auto result = subgraph.AddNode(PadNode{ input, std::move(low), std::move(high), mode, constantValue },
		                                     { OutputInfo{ info.dtype, std::move(outputShape) } });
		return { result, 0 };
	}

	// Backward-compatible helper: append zero padding at the end of each axis.
	inline NodeOutput AddPad(Subgraph& subgraph, NodeOutput input, std::span<const std::size_t> paddings)
	{
		const auto info = subgraph.GetOutputInfo(input);
		std::vector<std::size_t> lowPads(info.shape.size(), 0uz);
		return AddPad(subgraph, input, lowPads, paddings, PadMode::Constant, 0.0);
	}

	inline SubgraphId BuildPad(Graph& graph, DataType dtype, ShapeView shape, std::span<const std::size_t> paddings)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddPad(subgraph, { input, 0 }, paddings);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}

	inline SubgraphId BuildPad(Graph& graph, DataType dtype, ShapeView shape, std::span<const std::size_t> lowPads,
	                           std::span<const std::size_t> highPads,
	                           PadMode mode = PadMode::Constant, double constantValue = 0.0)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddPad(subgraph, { input, 0 }, lowPads, highPads, mode, constantValue);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
