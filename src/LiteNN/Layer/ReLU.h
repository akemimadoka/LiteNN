#ifndef LITENN_LAYER_RELU_H
#define LITENN_LAYER_RELU_H

#include <LiteNN/Graph.h>

namespace LiteNN::Layer
{
	// 构建 ReLU(x) = max(x, 0) 的子图，返回 SubgraphId
	inline SubgraphId BuildReLU(Graph& graph, DataType dtype, ShapeView shape)
	{
		Subgraph sg;

		// 参数: x
		const auto x = sg.AddParam(dtype, shape.ToOwned());

		// 常量: 0（与 x 同 shape）
		auto zeroTensor = Tensor<CPU>(shape, dtype);
		const auto zero = sg.AddNode(ConstantNode{ zeroTensor.CopyToDevice(PolymorphicDevice{ CPU{} }) },
		                             { OutputInfo{ dtype, shape.ToOwned() } });

		// result = max(x, 0)
		const auto result =
		    sg.AddNode(BinaryOpNode{ BinaryOp::Max, { x, 0 }, { zero, 0 } }, { OutputInfo{ dtype, shape.ToOwned() } });

		sg.SetResults({ { result, 0 } });

		return graph.AddSubgraph(std::move(sg));
	}
} // namespace LiteNN::Layer

#endif
