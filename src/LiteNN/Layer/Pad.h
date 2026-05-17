#include <LiteNN/Graph.h>
#include <LiteNN/Layer/LayerUtils.h>

#include <span>
#include <stdexcept>
#include <vector>

#ifndef LITENN_LAYER_PAD_H
#define LITENN_LAYER_PAD_H

namespace LiteNN::Layer
{
	// 在每个轴的末尾追加零填充，paddings.size() 必须等于输入 rank。
	inline NodeOutput AddPad(Subgraph& subgraph, NodeOutput input, std::span<const std::size_t> paddings)
	{
		const auto info = subgraph.GetOutputInfo(input);
		if (info.shape.empty())
		{
			throw std::runtime_error("Pad input must have rank >= 1");
		}
		if (paddings.size() != info.shape.size())
		{
			throw std::runtime_error("Pad paddings rank must match input rank");
		}

		auto current = input;
		auto currentShape = info.shape;
		for (std::size_t axis = 0; axis < paddings.size(); ++axis)
		{
			if (paddings[axis] == 0)
			{
				continue;
			}

			auto padShape = currentShape;
			padShape[axis] = paddings[axis];
			const auto zeroPad = Detail::AddConstant(subgraph, Detail::MakeFilledTensor(padShape, info.dtype, 0.0));

			auto outputShape = currentShape;
			outputShape[axis] += paddings[axis];
			const auto concatId = subgraph.AddNode(ConcatNode{ { current, { zeroPad, 0 } }, axis },
			                                      { OutputInfo{ info.dtype, outputShape } });
			current = { concatId, 0 };
			currentShape = std::move(outputShape);
		}

		return current;
	}

	inline SubgraphId BuildPad(Graph& graph, DataType dtype, ShapeView shape, std::span<const std::size_t> paddings)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddPad(subgraph, { input, 0 }, paddings);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif