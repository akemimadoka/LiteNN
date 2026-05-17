#ifndef LITENN_LAYER_SCAN_H
#define LITENN_LAYER_SCAN_H

#include <LiteNN/Graph.h>

#include <stdexcept>

namespace LiteNN::Layer
{
	inline NodeOutput AddScan(Subgraph& subgraph, NodeOutput input, std::size_t axis = 0,
	                          ScanOp op = ScanOp::Sum)
	{
		const auto info = subgraph.GetOutputInfo(input);
		if (info.shape.empty())
		{
			throw std::runtime_error("Scan input must have rank >= 1");
		}
		if (axis >= info.shape.size())
		{
			throw std::runtime_error("Scan axis is out of range");
		}
		if (info.dtype == DataType::Bool)
		{
			throw std::runtime_error("Scan does not support Bool tensors");
		}
		if (op == ScanOp::LogSumExp && !IsFloatingDataType(info.dtype))
		{
			throw std::runtime_error("Scan LogSumExp requires a floating-point input tensor");
		}
		const auto result = subgraph.AddNode(ScanNode{ input, axis, op }, { OutputInfo{ info.dtype, info.shape } });
		return { result, 0 };
	}

	inline SubgraphId BuildScan(Graph& graph, DataType dtype, ShapeView shape, std::size_t axis = 0,
	                            ScanOp op = ScanOp::Sum)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddScan(subgraph, { input, 0 }, axis, op);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
