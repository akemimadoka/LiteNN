#include <LiteNN/Graph.h>

#include <stdexcept>
#include <utility>

#ifndef LITENN_LAYER_SOFTMAX_H
#define LITENN_LAYER_SOFTMAX_H

namespace LiteNN::Layer
{
	// Softmax 沿指定轴归一化，输出 shape 与输入相同
	// 使用 max-shifted 公式保证数值稳定性
	inline NodeOutput AddSoftmax(Subgraph& subgraph, NodeOutput input, std::size_t axis = 1)
	{
		const auto info = subgraph.GetOutputInfo(input); // copy，避免 AddNode 后引用失效
		if (axis >= info.shape.size())
		{
			throw std::runtime_error("Softmax axis is out of range");
		}
		if (!IsFloatingDataType(info.dtype))
		{
			throw std::runtime_error("Softmax requires a floating-point input tensor");
		}
		const auto result = subgraph.AddNode(SoftmaxNode{ input, axis }, { OutputInfo{ info.dtype, info.shape } });
		return { result, 0 };
	}

	inline SubgraphId BuildSoftmax(Graph& graph, DataType dtype, ShapeView shape, std::size_t axis = 1)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddSoftmax(subgraph, { input, 0 }, axis);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
