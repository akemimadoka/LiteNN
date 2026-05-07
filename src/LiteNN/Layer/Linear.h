#ifndef LITENN_LAYER_LINEAR_H
#define LITENN_LAYER_LINEAR_H

#include <LiteNN/Graph.h>

#include <format>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace LiteNN::Layer
{
	struct LinearLayer
	{
		std::size_t weightVariable{};
		std::optional<std::size_t> biasVariable;
		std::size_t inFeatures{};
		std::size_t outFeatures{};
		DataType dtype{ DataType::Float32 };
	};

	inline void ValidateLinearWeight(const Tensor<CPU>& weight)
	{
		if (weight.Shape().NumDim() != 2)
		{
			throw std::runtime_error("Linear weight must be a 2D tensor");
		}
	}

	inline void ValidateLinearBias(const Tensor<CPU>& bias, std::size_t outFeatures, DataType dtype)
	{
		if (bias.DType() != dtype || bias.Shape().NumDim() != 2 || bias.Shape()[0] != 1 || bias.Shape()[1] != outFeatures)
		{
			throw std::runtime_error("Linear bias must have shape [1, outFeatures] and the same dtype as weight");
		}
	}

	inline LinearLayer CreateLinear(Graph& graph, Tensor<CPU> weight)
	{
		ValidateLinearWeight(weight);
		LinearLayer layer;
		layer.inFeatures = weight.Shape()[0];
		layer.outFeatures = weight.Shape()[1];
		layer.dtype = weight.DType();
		layer.weightVariable = graph.AddVariable(Variable::Create(std::move(weight)));
		return layer;
	}

	inline LinearLayer CreateLinear(Graph& graph, Tensor<CPU> weight, Tensor<CPU> bias)
	{
		ValidateLinearWeight(weight);
		ValidateLinearBias(bias, weight.Shape()[1], weight.DType());

		LinearLayer layer;
		layer.inFeatures = weight.Shape()[0];
		layer.outFeatures = weight.Shape()[1];
		layer.dtype = weight.DType();
		layer.weightVariable = graph.AddVariable(Variable::Create(std::move(weight)));
		layer.biasVariable = graph.AddVariable(Variable::Create(std::move(bias)));
		return layer;
	}

	inline NodeOutput AddLinear(Subgraph& subgraph, const LinearLayer& layer, NodeOutput input)
	{
		const auto inputInfo = subgraph.GetOutputInfo(input);
		if (inputInfo.dtype != layer.dtype || inputInfo.shape.size() != 2 || inputInfo.shape[1] != layer.inFeatures)
		{
			throw std::runtime_error(std::format("Linear input must have shape [batch, {}] and matching dtype",
			                                    layer.inFeatures));
		}

		const std::vector<std::size_t> weightShape{ layer.inFeatures, layer.outFeatures };
		const std::vector<std::size_t> outputShape{ inputInfo.shape[0], layer.outFeatures };
		const auto weight = subgraph.AddNode(VariableRefNode{ layer.weightVariable },
		                                     { OutputInfo{ layer.dtype, weightShape } });
		const auto matmul = subgraph.AddNode(BinaryOpNode{ BinaryOp::MatMul, input, { weight, 0 } },
		                                     { OutputInfo{ layer.dtype, outputShape } });
		if (!layer.biasVariable)
		{
			return { matmul, 0 };
		}

		const std::vector<std::size_t> biasShape{ 1, layer.outFeatures };
		const auto bias =
		    subgraph.AddNode(VariableRefNode{ *layer.biasVariable }, { OutputInfo{ layer.dtype, biasShape } });
		const auto result = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { bias, 0 } },
		                                     { OutputInfo{ layer.dtype, outputShape } });
		return { result, 0 };
	}

	inline SubgraphId BuildLinear(Graph& graph, const LinearLayer& layer, std::size_t batchSize = 1)
	{
		Subgraph subgraph;
		const auto input = subgraph.AddParam(layer.dtype, { batchSize, layer.inFeatures });
		const auto result = AddLinear(subgraph, layer, { input, 0 });
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}
} // namespace LiteNN::Layer

#endif
