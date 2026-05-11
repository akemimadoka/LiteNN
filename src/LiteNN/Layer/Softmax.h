#include <LiteNN/Graph.h>

#include <stdexcept>
#include <utility>
#include <vector>

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

		// ReduceOp 去掉 axis 维度后的形状
		std::vector<std::size_t> reducedShape;
		for (std::size_t i = 0; i < info.shape.size(); ++i)
		{
			if (i != axis)
			{
				reducedShape.push_back(info.shape[i]);
			}
		}
		if (reducedShape.empty())
		{
			reducedShape.push_back(1);
		}

		// 广播用形状：在 axis 位置插入 1，与输入做元素级广播
		std::vector<std::size_t> broadcastShape = reducedShape;
		broadcastShape.insert(broadcastShape.begin() + static_cast<std::ptrdiff_t>(axis), 1);

		// max_x = ReduceMax(input, axis)  → reducedShape
		const auto maxX = subgraph.AddNode(ReduceOpNode{ ReduceOp::Max, input, axis },
		                                   { OutputInfo{ info.dtype, reducedShape } });

		// max_x_bc = Reshape(max_x, broadcastShape)  供广播减法使用
		const auto maxXBc = subgraph.AddNode(ReshapeNode{ { maxX, 0 }, broadcastShape },
		                                     { OutputInfo{ info.dtype, broadcastShape } });

		// x_shifted = input - max_x_bc  (数值稳定)
		const auto xShifted = subgraph.AddNode(BinaryOpNode{ BinaryOp::Subtract, input, { maxXBc, 0 } },
		                                       { OutputInfo{ info.dtype, info.shape } });

		// exp_x = Exp(x_shifted)
		const auto expX = subgraph.AddNode(UnaryOpNode{ UnaryOp::Exp, { xShifted, 0 } },
		                                   { OutputInfo{ info.dtype, info.shape } });

		// sum_exp = ReduceSum(exp_x, axis)  → reducedShape
		const auto sumExp = subgraph.AddNode(ReduceOpNode{ ReduceOp::Sum, { expX, 0 }, axis },
		                                     { OutputInfo{ info.dtype, reducedShape } });

		// sum_exp_bc = Reshape(sum_exp, broadcastShape)  供广播除法使用
		const auto sumExpBc = subgraph.AddNode(ReshapeNode{ { sumExp, 0 }, broadcastShape },
		                                       { OutputInfo{ info.dtype, broadcastShape } });

		// result = exp_x / sum_exp_bc
		const auto result = subgraph.AddNode(BinaryOpNode{ BinaryOp::Divide, { expX, 0 }, { sumExpBc, 0 } },
		                                     { OutputInfo{ info.dtype, info.shape } });

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
