#include <LiteNN/Graph.h>
#include <LiteNN/ModelBuilder.h>

#include <format>
#include <numeric>
#include <optional>
#include <span>
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
		NodeOutput matmul;
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
			const auto quantizedMatMul =
			    subgraph.AddNode(QuantizedMatMulNode{ input, { storage, 0 }, params, layer.transposeWeight },
			                     { OutputInfo{ layer.dtype, outputShape } });
			matmul = { quantizedMatMul, 0 };
		}
		else
		{
			NodeOutput weight;
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
			const auto plainMatMul = subgraph.AddNode(BinaryOpNode{ BinaryOp::MatMul, input, weight },
			                                          { OutputInfo{ layer.dtype, outputShape } });
			matmul = { plainMatMul, 0 };
		}
		if (!layer.biasVariable)
		{
			return matmul;
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
		const auto result = subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, matmul, { bias, 0 } },
		                                     { OutputInfo{ layer.dtype, outputShape } });
		return { result, 0 };
	}

	inline bool CanUseGroupedQuantizedLinearProjection(std::span<const LinearLayer> layers, const OutputInfo& inputInfo)
	{
		if (layers.size() < 2 || layers.size() > 3)
		{
			return false;
		}
		const auto& first = layers.front();
		if (inputInfo.dtype != first.dtype || inputInfo.shape.size() != 2 || inputInfo.shape[1] != first.inFeatures ||
		    !first.weightQuantization || first.biasVariable || !first.transposeWeight ||
		    first.weightStorageShape.size() != 1)
		{
			return false;
		}
		const auto& firstParams = *first.weightQuantization;
		if (firstParams.scheme != QuantizationScheme::Block || !IsGGMLQuantizedBlockFormat(firstParams.blockFormat) ||
		    firstParams.storageType != DataType::UInt8 || firstParams.expressedType != first.dtype ||
		    firstParams.expressedShape != std::vector<std::size_t>{ first.outFeatures, first.inFeatures })
		{
			return false;
		}
		for (const auto& layer : layers.subspan(1))
		{
			if (layer.dtype != first.dtype || layer.inFeatures != first.inFeatures || layer.biasVariable ||
			    !layer.weightQuantization || !layer.transposeWeight || layer.weightStorageShape.size() != 1)
			{
				return false;
			}
			const auto& params = *layer.weightQuantization;
			if (params.scheme != firstParams.scheme || params.blockFormat != firstParams.blockFormat ||
			    params.storageType != firstParams.storageType || params.expressedType != firstParams.expressedType ||
			    params.granularity != firstParams.granularity ||
			    params.expressedShape != std::vector<std::size_t>{ layer.outFeatures, layer.inFeatures })
			{
				return false;
			}
		}
		return true;
	}

	inline std::vector<NodeOutput> AddLinearProjectionGroup(Subgraph& subgraph, std::span<const LinearLayer> layers,
	                                                        NodeOutput input)
	{
		const auto inputInfo = subgraph.GetOutputInfo(input);
		if (!CanUseGroupedQuantizedLinearProjection(layers, inputInfo))
		{
			std::vector<NodeOutput> outputs;
			outputs.reserve(layers.size());
			for (const auto& layer : layers)
			{
				outputs.push_back(AddLinear(subgraph, layer, input));
			}
			return outputs;
		}

		std::vector<NodeOutput> rhsStorages;
		std::vector<QuantizationParams> projectionParams;
		std::vector<std::size_t> outputWidths;
		rhsStorages.reserve(layers.size());
		projectionParams.reserve(layers.size());
		outputWidths.reserve(layers.size());
		for (const auto& layer : layers)
		{
			const auto& params = *layer.weightQuantization;
			const auto storage = subgraph.AddNode(VariableRefNode{ layer.weightVariable },
			                                      { OutputInfo{ params.storageType, layer.weightStorageShape } });
			rhsStorages.push_back({ storage, 0 });
			projectionParams.push_back(params);
			outputWidths.push_back(layer.outFeatures);
		}

		const auto totalOutputWidth = std::accumulate(outputWidths.begin(), outputWidths.end(), std::size_t{ 0 });
		const auto grouped = NodeOutput{
			subgraph.AddNode(GroupedQuantizedMatMulNode{ input, rhsStorages, projectionParams, outputWidths, true },
			                 { OutputInfo{ layers.front().dtype, { inputInfo.shape[0], totalOutputWidth } } }),
			0,
		};

		std::vector<NodeOutput> outputs;
		outputs.reserve(layers.size());
		std::size_t offset = 0;
		for (std::size_t i = 0; i < layers.size(); ++i)
		{
			outputs.push_back(
			    { subgraph.AddNode(SliceNode{ grouped, 1, offset, outputWidths[i] },
			                       { OutputInfo{ layers[i].dtype, { inputInfo.shape[0], outputWidths[i] } } }),
			      0 });
			offset += outputWidths[i];
		}
		return outputs;
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
