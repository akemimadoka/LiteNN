#include <LiteNN/Graph.h>
#include <LiteNN/ModelBuilder.h>

#include <format>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

#ifndef LITENN_LAYER_LINEAR_H
#define LITENN_LAYER_LINEAR_H

namespace LiteNN::Layer
{
	struct LinearLayer
	{
		std::size_t weightVariable{};
		std::optional<std::size_t> biasVariable;
		std::vector<std::size_t> biasShape;
		std::size_t inFeatures{};
		std::size_t outFeatures{};
		DataType dtype{ DataType::Float32 };
		std::optional<QuantizationParams> weightQuantization;
		std::vector<std::size_t> weightStorageShape;
		bool transposeWeight{};
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
		const auto vectorBias = bias.Shape().NumDim() == 1 && bias.Shape()[0] == outFeatures;
		const auto rowBias = bias.Shape().NumDim() == 2 && bias.Shape()[0] == 1 && bias.Shape()[1] == outFeatures;
		if (bias.DType() != dtype || (!vectorBias && !rowBias))
		{
			throw std::runtime_error(
			    "Linear bias must have shape [outFeatures] or [1, outFeatures] and the same dtype as weight");
		}
	}

	namespace Detail
	{
		inline LinearLayer MakeLinearLayerImpl(Graph& graph, Tensor<CPU> weight)
		{
			ValidateLinearWeight(weight);
			LinearLayer layer;
			layer.inFeatures = weight.Shape()[0];
			layer.outFeatures = weight.Shape()[1];
			layer.dtype = weight.DType();
			layer.weightVariable = graph.AddVariable(Variable::Create(std::move(weight)));
			return layer;
		}

		inline LinearLayer MakeLinearLayerImpl(Graph& graph, Tensor<CPU> weight, Tensor<CPU> bias)
		{
			ValidateLinearWeight(weight);
			ValidateLinearBias(bias, weight.Shape()[1], weight.DType());

			LinearLayer layer;
			layer.inFeatures = weight.Shape()[0];
			layer.outFeatures = weight.Shape()[1];
			layer.dtype = weight.DType();
			layer.weightVariable = graph.AddVariable(Variable::Create(std::move(weight)));
			layer.biasShape = bias.Shape().ToOwned();
			layer.biasVariable = graph.AddVariable(Variable::Create(std::move(bias)));
			return layer;
		}

	} // namespace Detail

	inline LinearLayer CreateLinear(ModelBuilder& builder, Tensor<CPU> weight)
	{
		return Detail::MakeLinearLayerImpl(builder.UnsafeMutableGraph(), std::move(weight));
	}

	inline LinearLayer CreateLinear(ModelBuilder& builder, Tensor<CPU> weight, Tensor<CPU> bias)
	{
		return Detail::MakeLinearLayerImpl(builder.UnsafeMutableGraph(), std::move(weight), std::move(bias));
	}

	inline NodeOutput AddLinear(Subgraph& subgraph, const LinearLayer& layer, NodeOutput input)
	{
		const auto inputInfo = subgraph.GetOutputInfo(input);
		if (inputInfo.dtype != layer.dtype || inputInfo.shape.size() != 2 || inputInfo.shape[1] != layer.inFeatures)
		{
			throw std::runtime_error(
			    std::format("Linear input must have shape [batch, {}] and matching dtype", layer.inFeatures));
		}

		const std::vector<std::size_t> weightShape{ layer.inFeatures, layer.outFeatures };
		const std::vector<std::size_t> outputShape{ inputInfo.shape[0], layer.outFeatures };
		NodeOutput weight;
		if (layer.weightQuantization)
		{
			const auto& params = *layer.weightQuantization;
			if (params.expressedType != layer.dtype || layer.weightStorageShape.empty())
			{
				throw std::runtime_error(
				    "Quantized Linear weight metadata is incompatible with layer dtype or storage");
			}
			const auto expectedExpressedShape =
			    layer.transposeWeight ? std::vector<std::size_t>{ layer.outFeatures, layer.inFeatures } : weightShape;
			if (params.expressedShape != expectedExpressedShape)
			{
				throw std::runtime_error(
				    "Quantized Linear expressed weight shape is incompatible with layer dimensions");
			}
			const auto storage = subgraph.AddNode(VariableRefNode{ layer.weightVariable },
			                                      { OutputInfo{ params.storageType, layer.weightStorageShape } });
			const auto dequantized = subgraph.AddNode(DequantizeNode{ { storage, 0 }, params, layer.dtype },
			                                          { OutputInfo{ layer.dtype, params.expressedShape } });
			if (layer.transposeWeight)
			{
				const auto transposed = subgraph.AddNode(UnaryOpNode{ UnaryOp::Transpose, { dequantized, 0 } },
				                                         { OutputInfo{ layer.dtype, weightShape } });
				weight = { transposed, 0 };
			}
			else
			{
				weight = { dequantized, 0 };
			}
		}
		else
		{
			const auto storedWeightShape =
			    layer.transposeWeight ? std::vector<std::size_t>{ layer.outFeatures, layer.inFeatures } : weightShape;
			const auto plain = subgraph.AddNode(VariableRefNode{ layer.weightVariable },
			                                    { OutputInfo{ layer.dtype, storedWeightShape } });
			if (layer.transposeWeight)
			{
				const auto transposed = subgraph.AddNode(UnaryOpNode{ UnaryOp::Transpose, { plain, 0 } },
				                                         { OutputInfo{ layer.dtype, weightShape } });
				weight = { transposed, 0 };
			}
			else
			{
				weight = { plain, 0 };
			}
		}
		const auto matmul = subgraph.AddNode(BinaryOpNode{ BinaryOp::MatMul, input, weight },
		                                     { OutputInfo{ layer.dtype, outputShape } });
		if (!layer.biasVariable)
		{
			return { matmul, 0 };
		}

		const auto biasShape =
		    layer.biasShape.empty() ? std::vector<std::size_t>{ 1, layer.outFeatures } : layer.biasShape;
		if (biasShape != std::vector<std::size_t>{ layer.outFeatures } &&
		    biasShape != std::vector<std::size_t>{ 1, layer.outFeatures })
		{
			throw std::runtime_error("Linear bias metadata must have shape [outFeatures] or [1, outFeatures]");
		}
		const auto bias =
		    subgraph.AddNode(VariableRefNode{ *layer.biasVariable }, { OutputInfo{ layer.dtype, biasShape } });
		const auto result = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { matmul, 0 }, { bias, 0 } },
		                                     { OutputInfo{ layer.dtype, outputShape } });
		return { result, 0 };
	}

	namespace Detail
	{
		inline SubgraphId BuildLinearImpl(Graph& graph, const LinearLayer& layer, std::size_t batchSize)
		{
			Subgraph subgraph;
			const auto input = subgraph.AddParam(layer.dtype, { batchSize, layer.inFeatures });
			const auto result = AddLinear(subgraph, layer, { input, 0 });
			subgraph.SetResults({ result });
			return graph.AddSubgraph(std::move(subgraph));
		}
	} // namespace Detail

	inline SubgraphId BuildLinear(ModelBuilder& builder, const LinearLayer& layer, std::size_t batchSize = 1)
	{
		return Detail::BuildLinearImpl(builder.UnsafeMutableGraph(), layer, batchSize);
	}
} // namespace LiteNN::Layer

#endif
