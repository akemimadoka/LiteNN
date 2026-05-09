#ifndef LITENN_LAYER_ACTIVATION_H
#define LITENN_LAYER_ACTIVATION_H

#include <LiteNN/Layer/LayerUtils.h>

#include <utility>

namespace LiteNN::Layer
{
	inline NodeOutput AddReLU(Subgraph& subgraph, NodeOutput input)
	{
		const auto info = subgraph.GetOutputInfo(input);
		const auto zero = Detail::AddConstant(subgraph, Detail::MakeScalarTensor(info.dtype, 0.0));
		const auto result = subgraph.AddNode(BinaryOpNode{ BinaryOp::Max, input, { zero, 0 } },
		                                     { OutputInfo{ info.dtype, info.shape } });
		return { result, 0 };
	}

	inline NodeOutput AddSigmoid(Subgraph& subgraph, NodeOutput input)
	{
		const auto info = subgraph.GetOutputInfo(input);
		const auto one = Detail::AddConstant(subgraph, Detail::MakeScalarTensor(info.dtype, 1.0));
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
		const auto two = Detail::AddConstant(subgraph, Detail::MakeScalarTensor(info.dtype, 2.0));
		const auto one = Detail::AddConstant(subgraph, Detail::MakeScalarTensor(info.dtype, 1.0));
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

	// GELU 近似：x * 0.5 * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
	// 参考 Hendrycks & Gimpel (2016)，与 PyTorch tanh 近似实现一致
	inline NodeOutput AddGELU(Subgraph& subgraph, NodeOutput input)
	{
		const auto info = subgraph.GetOutputInfo(input); // copy
		const auto half = Detail::AddConstant(subgraph, Detail::MakeScalarTensor(info.dtype, 0.5));
		const auto one = Detail::AddConstant(subgraph, Detail::MakeScalarTensor(info.dtype, 1.0));
		// sqrt(2/π) ≈ 0.7978845608028654
		const auto kappa = Detail::AddConstant(subgraph, Detail::MakeScalarTensor(info.dtype, 0.7978845608028654));
		const auto coeff = Detail::AddConstant(subgraph, Detail::MakeScalarTensor(info.dtype, 0.044715));
		const auto three = Detail::AddConstant(subgraph, Detail::MakeScalarTensor(info.dtype, 3.0));

		// x³
		const auto xCubed = subgraph.AddNode(BinaryOpNode{ BinaryOp::Pow, input, { three, 0 } },
		                                     { OutputInfo{ info.dtype, info.shape } });
		// 0.044715 * x³
		const auto scaledX3 = subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, { coeff, 0 }, { xCubed, 0 } },
		                                       { OutputInfo{ info.dtype, info.shape } });
		// x + 0.044715 * x³
		const auto inner = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, input, { scaledX3, 0 } },
		                                    { OutputInfo{ info.dtype, info.shape } });
		// sqrt(2/π) * (x + 0.044715 * x³)
		const auto preAct = subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, { kappa, 0 }, { inner, 0 } },
		                                     { OutputInfo{ info.dtype, info.shape } });
		// tanh(...)
		const auto tanhOut = AddTanh(subgraph, { preAct, 0 });
		// 1 + tanh(...)
		const auto onePlus = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { one, 0 }, tanhOut },
		                                      { OutputInfo{ info.dtype, info.shape } });
		// 0.5 * (1 + tanh(...))
		const auto halfOnePlus =
		    subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, { half, 0 }, { onePlus, 0 } },
		                     { OutputInfo{ info.dtype, info.shape } });
		// x * 0.5 * (1 + tanh(...))
		const auto result = subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, input, { halfOnePlus, 0 } },
		                                     { OutputInfo{ info.dtype, info.shape } });
		return { result, 0 };
	}

	inline SubgraphId BuildGELU(Graph& graph, DataType dtype, ShapeView shape)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddGELU(subgraph, { input, 0 });
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}

	// ELU：x if x > 0, else alpha * (exp(x) - 1)，默认 alpha=1.0
	inline NodeOutput AddELU(Subgraph& subgraph, NodeOutput input, double alpha = 1.0)
	{
		const auto info = subgraph.GetOutputInfo(input); // copy
		const auto zero = Detail::AddConstant(subgraph, Detail::MakeScalarTensor(info.dtype, 0.0));
		const auto one = Detail::AddConstant(subgraph, Detail::MakeScalarTensor(info.dtype, 1.0));
		const auto alphaC = Detail::AddConstant(subgraph, Detail::MakeScalarTensor(info.dtype, alpha));

		// exp(x)
		const auto expX =
		    subgraph.AddNode(UnaryOpNode{ UnaryOp::Exp, input }, { OutputInfo{ info.dtype, info.shape } });
		// exp(x) - 1
		const auto expM1 = subgraph.AddNode(BinaryOpNode{ BinaryOp::Subtract, { expX, 0 }, { one, 0 } },
		                                    { OutputInfo{ info.dtype, info.shape } });
		// alpha * (exp(x) - 1)
		const auto negBranch = subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, { alphaC, 0 }, { expM1, 0 } },
		                                        { OutputInfo{ info.dtype, info.shape } });
		// max(x, 0) ... positive part
		const auto posPart = subgraph.AddNode(BinaryOpNode{ BinaryOp::Max, input, { zero, 0 } },
		                                      { OutputInfo{ info.dtype, info.shape } });
		// min(negBranch, 0) ... negative part (ELU contribution when x < 0)
		const auto negPart = subgraph.AddNode(BinaryOpNode{ BinaryOp::Min, { negBranch, 0 }, { zero, 0 } },
		                                      { OutputInfo{ info.dtype, info.shape } });
		// ELU(x) = max(x, 0) + min(alpha*(exp(x)-1), 0)
		const auto result = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { posPart, 0 }, { negPart, 0 } },
		                                     { OutputInfo{ info.dtype, info.shape } });
		return { result, 0 };
	}

	inline SubgraphId BuildELU(Graph& graph, DataType dtype, ShapeView shape, double alpha = 1.0)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(dtype, shape.ToOwned());
		const auto result = AddELU(subgraph, { input, 0 }, alpha);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
