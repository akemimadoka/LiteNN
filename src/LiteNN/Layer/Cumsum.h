#include <LiteNN/Graph.h>
#include <LiteNN/Layer/Scan.h>

#include <stdexcept>

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
		return AddScan(subgraph, input, axis, ScanOp::Sum);
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
