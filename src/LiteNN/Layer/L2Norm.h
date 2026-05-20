#include <LiteNN/Graph.h>
#include <LiteNN/Layer/LayerUtils.h>

#include <stdexcept>
#include <utility>
#include <vector>

#ifndef LITENN_LAYER_L2NORM_H
#define LITENN_LAYER_L2NORM_H

namespace LiteNN::Layer
{
	inline NodeOutput AddL2Norm(Subgraph& subgraph, NodeOutput input, std::size_t axis, double eps = 0.0)
	{
		const auto info = subgraph.GetOutputInfo(input);
		if (axis >= info.shape.size())
		{
			throw std::runtime_error("L2Norm axis out of range");
		}
		if (info.dtype == DataType::Bool)
		{
			throw std::runtime_error("L2Norm does not support boolean tensors");
		}

		auto broadcastShape = info.shape;
		broadcastShape[axis] = 1;

		std::vector<std::size_t> reducedShape;
		reducedShape.reserve(info.shape.size() - 1);
		for (auto dim = 0uz; dim < info.shape.size(); ++dim)
		{
			if (dim != axis)
			{
				reducedShape.push_back(info.shape[dim]);
			}
		}
		if (reducedShape.empty())
		{
			reducedShape.push_back(1);
		}

		const auto squared =
		    subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, input, input }, { OutputInfo{ info.dtype, info.shape } });
		const auto summed = subgraph.AddNode(ReduceOpNode{ ReduceOp::Sum, { squared, 0 }, axis },
		                                     { OutputInfo{ info.dtype, reducedShape } });
		const auto summedBroadcast = subgraph.AddNode(ReshapeNode{ { summed, 0 }, broadcastShape },
		                                              { OutputInfo{ info.dtype, broadcastShape } });

		NodeOutput normSquared{ summedBroadcast, 0 };
		if (eps != 0.0)
		{
			const auto epsConst =
			    Detail::AddConstant(subgraph, Detail::MakeFilledTensor(broadcastShape, info.dtype, eps));
			const auto withEps = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, normSquared, { epsConst, 0 } },
			                                     { OutputInfo{ info.dtype, broadcastShape } });
			normSquared = { withEps, 0 };
		}

		const auto norm = subgraph.AddNode(UnaryOpNode{ UnaryOp::Sqrt, normSquared },
		                                   { OutputInfo{ info.dtype, broadcastShape } });
		const auto result = subgraph.AddNode(BinaryOpNode{ BinaryOp::Divide, input, { norm, 0 } },
		                                     { OutputInfo{ info.dtype, info.shape } });
		return { result, 0 };
	}

	inline SubgraphId BuildL2Norm(Graph& graph, DataType dtype, ShapeView shape, std::size_t axis, double eps = 0.0)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddL2Norm(subgraph, { input, 0 }, axis, eps);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif