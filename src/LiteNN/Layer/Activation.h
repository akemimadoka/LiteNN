#ifndef LITENN_LAYER_ACTIVATION_H
#define LITENN_LAYER_ACTIVATION_H

#include <LiteNN/Layer/LayerUtils.h>

#include <utility>

namespace LiteNN::Layer
{
	inline NodeOutput AddReLU(Subgraph& subgraph, NodeOutput input)
	{
		const auto info = subgraph.GetOutputInfo(input);
		auto zeroTensor = Tensor<CPU>(info.shape, info.dtype);
		const auto zero = Detail::AddConstant(subgraph, zeroTensor);
		const auto result = subgraph.AddNode(BinaryOpNode{ BinaryOp::Max, input, { zero, 0 } },
		                                     { OutputInfo{ info.dtype, info.shape } });
		return { result, 0 };
	}

	inline NodeOutput AddSigmoid(Subgraph& subgraph, NodeOutput input)
	{
		const auto info = subgraph.GetOutputInfo(input);
		const auto one = Detail::AddConstant(subgraph, Detail::MakeFilledTensor(info.shape, info.dtype, 1.0));
		const auto neg = subgraph.AddNode(UnaryOpNode{ UnaryOp::Negate, input }, { OutputInfo{ info.dtype, info.shape } });
		const auto exp =
		    subgraph.AddNode(UnaryOpNode{ UnaryOp::Exp, { neg, 0 } }, { OutputInfo{ info.dtype, info.shape } });
		const auto denom = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { one, 0 }, { exp, 0 } },
		                                    { OutputInfo{ info.dtype, info.shape } });
		const auto result = subgraph.AddNode(BinaryOpNode{ BinaryOp::Divide, { one, 0 }, { denom, 0 } },
		                                     { OutputInfo{ info.dtype, info.shape } });
		return { result, 0 };
	}

	inline NodeOutput AddTanh(Subgraph& subgraph, NodeOutput input)
	{
		const auto info = subgraph.GetOutputInfo(input);
		const auto two = Detail::AddConstant(subgraph, Detail::MakeFilledTensor(info.shape, info.dtype, 2.0));
		const auto one = Detail::AddConstant(subgraph, Detail::MakeFilledTensor(info.shape, info.dtype, 1.0));
		const auto scaled = subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, input, { two, 0 } },
		                                     { OutputInfo{ info.dtype, info.shape } });
		const auto exp =
		    subgraph.AddNode(UnaryOpNode{ UnaryOp::Exp, { scaled, 0 } }, { OutputInfo{ info.dtype, info.shape } });
		const auto numerator = subgraph.AddNode(BinaryOpNode{ BinaryOp::Subtract, { exp, 0 }, { one, 0 } },
		                                        { OutputInfo{ info.dtype, info.shape } });
		const auto denominator = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { exp, 0 }, { one, 0 } },
		                                          { OutputInfo{ info.dtype, info.shape } });
		const auto result = subgraph.AddNode(BinaryOpNode{ BinaryOp::Divide, { numerator, 0 }, { denominator, 0 } },
		                                     { OutputInfo{ info.dtype, info.shape } });
		return { result, 0 };
	}

	inline SubgraphId BuildReLU(Graph& graph, DataType dtype, ShapeView shape)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddReLU(subgraph, { input, 0 });
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}

	inline SubgraphId BuildSigmoid(Graph& graph, DataType dtype, ShapeView shape)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddSigmoid(subgraph, { input, 0 });
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}

	inline SubgraphId BuildTanh(Graph& graph, DataType dtype, ShapeView shape)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddTanh(subgraph, { input, 0 });
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
