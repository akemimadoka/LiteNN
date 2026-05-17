#include <LiteNN/Graph.h>

#include <stdexcept>
#include <vector>

#ifndef LITENN_LAYER_CUMSUM_H
#define LITENN_LAYER_CUMSUM_H

namespace LiteNN::Layer
{
	// 沿指定轴做前缀和，输出 shape 与输入相同。
	inline NodeOutput AddCumsum(Subgraph& subgraph, NodeOutput input, std::size_t axis = 0)
	{
		const auto info = subgraph.GetOutputInfo(input);
		if (info.shape.empty())
		{
			throw std::runtime_error("Cumsum input must have rank >= 1");
		}
		if (axis >= info.shape.size())
		{
			throw std::runtime_error("Cumsum axis is out of range");
		}
		if (info.dtype == DataType::Bool)
		{
			throw std::runtime_error("Cumsum does not support Bool tensors");
		}

		auto sliceShape = info.shape;
		sliceShape[axis] = 1;
		std::vector<NodeOutput> parts;
		parts.reserve(info.shape[axis]);

		NodeOutput running{};
		for (std::size_t index = 0; index < info.shape[axis]; ++index)
		{
			const auto sliceId = subgraph.AddNode(SliceNode{ input, axis, index, 1 },
			                                 { OutputInfo{ info.dtype, sliceShape } });
			const auto slice = NodeOutput{ sliceId, 0 };

			if (index == 0)
			{
				running = slice;
			}
			else
			{
				const auto addId = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, running, slice },
				                               { OutputInfo{ info.dtype, sliceShape } });
				running = { addId, 0 };
			}
			parts.push_back(running);
		}

		if (parts.size() == 1)
		{
			return parts.front();
		}

		const auto concatId = subgraph.AddNode(ConcatNode{ std::move(parts), axis }, { OutputInfo{ info.dtype, info.shape } });
		return { concatId, 0 };
	}

	inline SubgraphId BuildCumsum(Graph& graph, DataType dtype, ShapeView shape, std::size_t axis = 0)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddCumsum(subgraph, { input, 0 }, axis);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif