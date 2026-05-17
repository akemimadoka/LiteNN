#include "LLaMABuilder.h"

#include <LiteNN/Layer/LayerUtils.h>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <ggml.h>

namespace LiteNN::GGUF
{
	namespace
	{
		std::optional<ggml_type> TryMapGGMLQuantizedType(QuantizedBlockFormat format)
		{
			switch (format)
			{
			case QuantizedBlockFormat::GGML_Q4_0:
				return GGML_TYPE_Q4_0;
			case QuantizedBlockFormat::GGML_Q4_1:
				return GGML_TYPE_Q4_1;
			case QuantizedBlockFormat::GGML_Q5_0:
				return GGML_TYPE_Q5_0;
			case QuantizedBlockFormat::GGML_Q5_1:
				return GGML_TYPE_Q5_1;
			case QuantizedBlockFormat::GGML_Q8_0:
				return GGML_TYPE_Q8_0;
			case QuantizedBlockFormat::GGML_Q8_1:
				return GGML_TYPE_Q8_1;
			case QuantizedBlockFormat::GGML_Q2_K:
				return GGML_TYPE_Q2_K;
			case QuantizedBlockFormat::GGML_Q3_K:
				return GGML_TYPE_Q3_K;
			case QuantizedBlockFormat::GGML_Q4_K:
				return GGML_TYPE_Q4_K;
			case QuantizedBlockFormat::GGML_Q5_K:
				return GGML_TYPE_Q5_K;
			case QuantizedBlockFormat::GGML_Q6_K:
				return GGML_TYPE_Q6_K;
			case QuantizedBlockFormat::GGML_Q8_K:
				return GGML_TYPE_Q8_K;
			case QuantizedBlockFormat::GGML_IQ2_XXS:
				return GGML_TYPE_IQ2_XXS;
			case QuantizedBlockFormat::GGML_IQ2_XS:
				return GGML_TYPE_IQ2_XS;
			case QuantizedBlockFormat::GGML_IQ3_XXS:
				return GGML_TYPE_IQ3_XXS;
			case QuantizedBlockFormat::GGML_IQ1_S:
				return GGML_TYPE_IQ1_S;
			case QuantizedBlockFormat::GGML_IQ4_NL:
				return GGML_TYPE_IQ4_NL;
			case QuantizedBlockFormat::GGML_IQ3_S:
				return GGML_TYPE_IQ3_S;
			case QuantizedBlockFormat::GGML_IQ2_S:
				return GGML_TYPE_IQ2_S;
			case QuantizedBlockFormat::GGML_IQ4_XS:
				return GGML_TYPE_IQ4_XS;
			default:
				return std::nullopt;
			}
		}

		Tensor<CPU> DequantizeGGMLVariable(const Variable& variable, std::string_view name)
		{
			if (!variable.IsQuantized())
			{
				throw std::runtime_error(std::format("GGUF tensor '{}' is not quantized", name));
			}

			const auto& params = *variable.Quantization();
			if (params.scheme != QuantizationScheme::Block)
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' uses unsupported quantization scheme {} for current LLaMA lowering",
				    name, QuantizationSchemeName(params.scheme)));
			}
			if (!IsFloatingDataType(params.expressedType))
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' must dequantize to a floating-point type, got {}", name,
				    DataTypeName(params.expressedType)));
			}

			const auto ggmlType = TryMapGGMLQuantizedType(params.blockFormat);
			if (!ggmlType)
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' uses unsupported block quantization format {}", name,
				    QuantizedBlockFormatName(params.blockFormat)));
			}

			const auto* traits = ggml_get_type_traits(*ggmlType);
			if (!traits || !traits->to_float)
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' uses block format {} which does not expose a ggml dequantizer",
				    name, QuantizedBlockFormatName(params.blockFormat)));
			}

			if (params.expressedShape.empty())
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' is missing expressedShape for block quantization", name));
			}

			std::size_t totalElements = 1;
			for (const auto dim : params.expressedShape)
			{
				totalElements *= dim;
			}

			const auto rowSize = params.expressedShape.back();
			if (rowSize == 0 || totalElements % rowSize != 0)
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' has invalid expressed shape for row-wise ggml dequantization", name));
			}
			if ((rowSize % static_cast<std::size_t>(traits->blck_size)) != 0)
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' row width {} is incompatible with ggml block size {} for format {}",
				    name, rowSize, traits->blck_size, QuantizedBlockFormatName(params.blockFormat)));
			}

			const auto rowCount = totalElements / rowSize;
			const auto rowBytes = (rowSize / static_cast<std::size_t>(traits->blck_size)) * traits->type_size;
			const auto storage = variable.Data().CopyToDevice(CPU{});
			if (storage.DType() != DataType::UInt8)
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' block payload must be stored as UInt8 bytes, got {}", name,
				    DataTypeName(storage.DType())));
			}
			if (storage.NumElements() != rowCount * rowBytes)
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' payload byte count {} does not match expressed shape {}x{} for format {}",
				    name, storage.NumElements(), rowCount, rowSize, QuantizedBlockFormatName(params.blockFormat)));
			}

			Tensor<CPU> dequantizedF32(Uninitialized, params.expressedShape, DataType::Float32);
			const auto* src = static_cast<const std::uint8_t*>(storage.RawData());
			auto* dst = static_cast<float*>(dequantizedF32.RawData());
			for (std::size_t row = 0; row < rowCount; ++row)
			{
				traits->to_float(src + row * rowBytes, dst + row * rowSize, static_cast<int64_t>(rowSize));
			}

			if (params.expressedType == DataType::Float32)
			{
				return dequantizedF32;
			}

			CPU cpu;
			Tensor<CPU> converted(Uninitialized, params.expressedShape, params.expressedType, cpu);
			DeviceTraits<CPU>::ConvertTo(cpu, DataType::Float32, dequantizedF32.RawData(), dequantizedF32.NumElements(),
			                             params.expressedType, converted.RawData());
			return converted;
		}

		std::shared_ptr<Variable> MaterializeArchiveVariable(const Graph& archive, std::size_t variableIndex,
		                                                 std::string_view name)
		{
			const auto& source = archive.GetVariable(variableIndex);
			if (!source->IsQuantized())
			{
				return source;
			}
			return Variable::Create(DequantizeGGMLVariable(*source, name));
		}

		std::string BlockTensorName(std::size_t blockIndex, std::string_view suffix)
		{
			return std::format("blk.{}.{}", blockIndex, suffix);
		}

		std::size_t ImportNamedVariable(Graph& target, const Graph& archive, std::string_view name)
		{
			const auto sourceIndex = archive.FindVariable(name);
			if (!sourceIndex)
			{
				throw std::runtime_error(std::format("Missing GGUF tensor '{}'", name));
			}
			const auto targetIndex = target.AddVariable(MaterializeArchiveVariable(archive, *sourceIndex, name));
			target.SetVariableName(targetIndex, std::string(name));
			return targetIndex;
		}

		const Variable& RequirePlainFloatingVariable(const Graph& graph, std::size_t variableIndex, std::string_view name)
		{
			const auto& variable = *graph.GetVariable(variableIndex);
			if (variable.IsQuantized())
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' is quantized; current LLaMA block lowering requires plain floating-point weights",
				    name));
			}
			if (!IsFloatingDataType(variable.Data().DType()))
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' must be floating-point for current LLaMA block lowering", name));
			}
			return variable;
		}

		Layer::LinearLayer MakeLinearFromArchive(Graph& target, const Graph& archive, std::string_view name,
		                                       std::size_t inFeatures, std::size_t outFeatures)
		{
			const auto sourceIndex = archive.FindVariable(name);
			if (!sourceIndex)
			{
				throw std::runtime_error(std::format("Missing GGUF tensor '{}'", name));
			}

			auto materialized = MaterializeArchiveVariable(archive, *sourceIndex, name);
			if (materialized->IsQuantized() || !IsFloatingDataType(materialized->Data().DType()))
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' must be floating-point for current LLaMA block lowering", name));
			}

			auto data = materialized->Data().CopyToDevice(CPU{});
			if (data.Shape().NumDim() == 2 && data.Shape()[0] == outFeatures && data.Shape()[1] == inFeatures)
			{
				data = data.Transpose();
				materialized = Variable::Create(std::move(data));
			}
			else if (data.Shape().NumDim() != 2 || data.Shape()[0] != inFeatures || data.Shape()[1] != outFeatures)
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' must have LiteNN shape [{}, {}] or imported GGUF shape [{}, {}] for current LLaMA block lowering",
				    name, inFeatures, outFeatures, outFeatures, inFeatures));
			}

			const auto variableIndex = target.AddVariable(std::move(materialized));
			target.SetVariableName(variableIndex, std::string(name));
			return {
				.weightVariable = variableIndex,
				.biasVariable = std::nullopt,
				.inFeatures = inFeatures,
				.outFeatures = outFeatures,
				.dtype = target.GetVariable(variableIndex)->Data().DType(),
			};
		}

		Layer::RMSNormLayer MakeRMSNormFromArchive(Graph& target, const Graph& archive, std::string_view name,
		                                         std::size_t featureSize, double eps)
		{
			const auto sourceIndex = archive.FindVariable(name);
			if (!sourceIndex)
			{
				throw std::runtime_error(std::format("Missing GGUF tensor '{}'", name));
			}

			auto materialized = MaterializeArchiveVariable(archive, *sourceIndex, name);
			if (materialized->IsQuantized() || !IsFloatingDataType(materialized->Data().DType()))
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' must be floating-point for current LLaMA block lowering", name));
			}

			auto data = materialized->Data().CopyToDevice(CPU{});
			const auto shape = data.Shape();
			if (shape.NumDim() == 1 && shape[0] == featureSize)
			{
				data.Reshape({ 1, featureSize });
				materialized = Variable::Create(std::move(data));
			}
			else if (!(shape.NumDim() == 2 && shape[0] == 1 && shape[1] == featureSize))
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' must have shape [{}] or [1, {}] for current LLaMA block lowering", name,
				    featureSize, featureSize));
			}

			const auto variableIndex = target.AddVariable(std::move(materialized));
			target.SetVariableName(variableIndex, std::string(name));
			return {
				.weightVariable = variableIndex,
				.featureSize = featureSize,
				.dtype = target.GetVariable(variableIndex)->Data().DType(),
				.eps = eps,
			};
		}

		NodeOutput AddTranspose(Subgraph& subgraph, NodeOutput input)
		{
			const auto info = subgraph.GetOutputInfo(input);
			if (info.shape.size() != 2)
			{
				throw std::runtime_error("Transpose helper expects a 2D tensor");
			}
			return { subgraph.AddNode(UnaryOpNode{ UnaryOp::Transpose, input },
			                         { OutputInfo{ info.dtype, { info.shape[1], info.shape[0] } } }),
			         0 };
		}

		void ValidateSupportedRoPE(const LLaMAHyperparameters& hyperparameters, std::string_view context)
		{
			if (hyperparameters.ropeScalingType != "none" && hyperparameters.ropeScalingType != "linear")
			{
				throw std::runtime_error(std::format(
				    "Current LLaMA {} lowering preserves rope.scaling.* metadata but only executes none/linear scaling, got '{}'",
				    context, hyperparameters.ropeScalingType));
			}
		}

		NodeOutput AddLLaMARoPE(Subgraph& subgraph, NodeOutput input, const LLaMAHyperparameters& hyperparameters,
		                       std::size_t positionOffset)
		{
			const auto info = subgraph.GetOutputInfo(input);
			if (info.shape.size() != 2)
			{
				throw std::runtime_error("LLaMA RoPE helper expects a 2D tensor");
			}
			if (hyperparameters.ropeDimensionCount > info.shape[1] || (hyperparameters.ropeDimensionCount % 2) != 0)
			{
				throw std::runtime_error("LLaMA rope.dimension_count must be even and no larger than headDim");
			}

			const auto rotatedPrefix = [&]() {
				if (hyperparameters.ropeDimensionCount == info.shape[1])
				{
					return input;
				}
				return NodeOutput{ subgraph.AddNode(
				                       SliceNode{ input, 1, 0, hyperparameters.ropeDimensionCount },
				                       { OutputInfo{ info.dtype, { info.shape[0], hyperparameters.ropeDimensionCount } } }),
				                   0 };
			}();
			const auto rotated = Layer::AddRoPE(subgraph, rotatedPrefix, hyperparameters.ropeFrequencyBase,
			                                   positionOffset, hyperparameters.ropeFrequencyScale);
			if (hyperparameters.ropeDimensionCount == info.shape[1])
			{
				return rotated;
			}

			const auto tailWidth = info.shape[1] - hyperparameters.ropeDimensionCount;
			const auto tail = NodeOutput{
				subgraph.AddNode(SliceNode{ input, 1, hyperparameters.ropeDimensionCount, tailWidth },
			                     { OutputInfo{ info.dtype, { info.shape[0], tailWidth } } }),
			    0
			};
			return { subgraph.AddNode(ConcatNode{ { rotated, tail }, 1 }, { info }), 0 };
		}

		NodeOutput AddSingleHeadAttention(Subgraph& subgraph, NodeOutput queries, NodeOutput keys, NodeOutput values,
		                                 const LLaMAHyperparameters& hyperparameters, std::size_t positionOffset)
		{
			ValidateSupportedRoPE(hyperparameters, "prefill");
			const auto queryInfo = subgraph.GetOutputInfo(queries);
			if (queryInfo.shape.size() != 2 || queryInfo.shape != subgraph.GetOutputInfo(keys).shape)
			{
				throw std::runtime_error("Single-head attention expects 2D query/key tensors with matching shape");
			}
			const auto valueInfo = subgraph.GetOutputInfo(values);
			if (valueInfo.shape.size() != 2 || valueInfo.shape[0] != queryInfo.shape[0])
			{
				throw std::runtime_error("Single-head attention expects value tensor shape [sequence, headDim]");
			}

			const auto rotatedQueries = AddLLaMARoPE(subgraph, queries, hyperparameters, positionOffset);
			const auto rotatedKeys = AddLLaMARoPE(subgraph, keys, hyperparameters, positionOffset);

			Layer::FlashAttnExtOptions options;
			options.scale = 1.0 / std::sqrt(static_cast<double>(queryInfo.shape[1]));
			options.causal = true;
			options.keyPositionOffset = positionOffset;
			options.queryPositionOffset = positionOffset;
			return Layer::AddFlashAttnExt(subgraph, rotatedQueries, rotatedKeys, values, options);
		}

		NodeOutput AddSingleHeadAttentionWithRotatedKV(Subgraph& subgraph, NodeOutput queries, NodeOutput rotatedKeys,
		                                              NodeOutput values,
		                                              const LLaMAHyperparameters& hyperparameters,
		                                              std::size_t queryPositionOffset)
		{
			ValidateSupportedRoPE(hyperparameters, "decode");
			const auto queryInfo = subgraph.GetOutputInfo(queries);
			if (queryInfo.shape.size() != 2)
			{
				throw std::runtime_error("Single-head decode attention expects 2D query tensor");
			}
			const auto keyInfo = subgraph.GetOutputInfo(rotatedKeys);
			const auto valueInfo = subgraph.GetOutputInfo(values);
			if (keyInfo.shape.size() != 2 || keyInfo.shape[1] != queryInfo.shape[1])
			{
				throw std::runtime_error("Single-head decode attention expects key tensor shape [keyLength, headDim]");
			}
			if (valueInfo.shape.size() != 2 || valueInfo.shape[0] != keyInfo.shape[0])
			{
				throw std::runtime_error("Single-head decode attention expects value tensor shape [keyLength, headDim]");
			}
			const auto rotatedQueries = AddLLaMARoPE(subgraph, queries, hyperparameters, queryPositionOffset);
			Layer::FlashAttnExtOptions options;
			options.scale = 1.0 / std::sqrt(static_cast<double>(queryInfo.shape[1]));
			options.causal = true;
			options.keyPositionOffset = 0;
			options.queryPositionOffset = queryPositionOffset;
			return Layer::AddFlashAttnExt(subgraph, rotatedQueries, rotatedKeys, values, options);
		}

		NodeOutput Reshape2D(Subgraph& subgraph, NodeOutput input, std::size_t rows, std::size_t cols)
		{
			const auto info = subgraph.GetOutputInfo(input);
			return { subgraph.AddNode(ReshapeNode{ input, { rows, cols } },
			                         { OutputInfo{ info.dtype, { rows, cols } } }),
			         0 };
		}

		NodeOutput Reshape3D(Subgraph& subgraph, NodeOutput input, std::size_t dim0, std::size_t dim1,
		                     std::size_t dim2)
		{
			const auto info = subgraph.GetOutputInfo(input);
			return { subgraph.AddNode(ReshapeNode{ input, { dim0, dim1, dim2 } },
			                         { OutputInfo{ info.dtype, { dim0, dim1, dim2 } } }),
			         0 };
		}

		std::vector<ModelMetadataEntry> CopyMetadata(const Graph& graph)
		{
			return { graph.Metadata().begin(), graph.Metadata().end() };
		}
	} // namespace

	LLaMAParityTolerance GetLLaMAParityTolerance(DataType dtype, std::optional<QuantizedBlockFormat> blockFormat)
	{
		if (blockFormat && *blockFormat != QuantizedBlockFormat::Scalar)
		{
			switch (*blockFormat)
			{
			case QuantizedBlockFormat::GGML_Q8_0:
			case QuantizedBlockFormat::GGML_Q8_1:
			case QuantizedBlockFormat::GGML_Q8_K:
				return { 2.0e-2, 2.0e-2 };
			case QuantizedBlockFormat::GGML_Q6_K:
				return { 5.0e-2, 5.0e-2 };
			default:
				return { 1.0e-1, 1.0e-1 };
			}
		}

		switch (dtype)
		{
		case DataType::Float64:
			return { 1.0e-8, 1.0e-8 };
		case DataType::Float32:
			return { 1.0e-5, 1.0e-5 };
		case DataType::Float16:
		case DataType::BFloat16:
			return { 5.0e-3, 5.0e-3 };
		case DataType::Float8E4M3:
		case DataType::Float8E5M2:
			return { 5.0e-2, 5.0e-2 };
		default:
			throw std::runtime_error(
			    std::format("LLaMA parity tolerance is only defined for floating-point logits, got {}",
			                DataTypeName(dtype)));
		}
	}

	LLaMADecoderBlock CreateLLaMADecoderBlock(Graph& graph, const Graph& archive,
	                                         const LLaMAHyperparameters& hyperparameters,
	                                         std::size_t blockIndex)
	{
		const auto headDim = hyperparameters.HeadDimension();
		const auto kvWidth = hyperparameters.attentionHeadCountKV * headDim;

		return {
			.attentionNorm = MakeRMSNormFromArchive(graph, archive, BlockTensorName(blockIndex, "attn_norm.weight"),
			                                      hyperparameters.embeddingLength,
			                                      hyperparameters.rmsNormEpsilon),
			.queryProjection = MakeLinearFromArchive(graph, archive, BlockTensorName(blockIndex, "attn_q.weight"),
			                                        hyperparameters.embeddingLength, hyperparameters.embeddingLength),
			.keyProjection = MakeLinearFromArchive(graph, archive, BlockTensorName(blockIndex, "attn_k.weight"),
			                                      hyperparameters.embeddingLength, kvWidth),
			.valueProjection = MakeLinearFromArchive(graph, archive, BlockTensorName(blockIndex, "attn_v.weight"),
			                                        hyperparameters.embeddingLength, kvWidth),
			.outputProjection = MakeLinearFromArchive(graph, archive,
			                                         BlockTensorName(blockIndex, "attn_output.weight"),
			                                         hyperparameters.embeddingLength, hyperparameters.embeddingLength),
			.feedForwardNorm = MakeRMSNormFromArchive(graph, archive, BlockTensorName(blockIndex, "ffn_norm.weight"),
			                                        hyperparameters.embeddingLength,
			                                        hyperparameters.rmsNormEpsilon),
			.mlp = {
				.gateProjection = MakeLinearFromArchive(graph, archive, BlockTensorName(blockIndex, "ffn_gate.weight"),
				                                      hyperparameters.embeddingLength,
				                                      hyperparameters.feedForwardLength),
				.upProjection = MakeLinearFromArchive(graph, archive, BlockTensorName(blockIndex, "ffn_up.weight"),
				                                    hyperparameters.embeddingLength,
				                                    hyperparameters.feedForwardLength),
				.downProjection = MakeLinearFromArchive(graph, archive, BlockTensorName(blockIndex, "ffn_down.weight"),
				                                      hyperparameters.feedForwardLength,
				                                      hyperparameters.embeddingLength),
			},
		};
	}

	NodeOutput AddLLaMADecoderBlock(Subgraph& subgraph, const LLaMADecoderBlock& block,
	                               const LLaMAHyperparameters& hyperparameters, NodeOutput hiddenState,
	                               std::size_t positionOffset)
	{
		const auto hiddenInfo = subgraph.GetOutputInfo(hiddenState);
		if (hiddenInfo.dtype != block.attentionNorm.dtype || hiddenInfo.shape.size() != 2 ||
		    hiddenInfo.shape[1] != hyperparameters.embeddingLength)
		{
			throw std::runtime_error("LLaMA decoder block input must be 2D [sequence, embeddingLength]");
		}

		const auto headDim = hyperparameters.HeadDimension();
		const auto queryGroupsPerKVHead = hyperparameters.QueryGroupsPerKVHead();
		const auto normalizedAttentionInput = Layer::AddRMSNorm(subgraph, block.attentionNorm, hiddenState);
		const auto queries = Layer::AddLinear(subgraph, block.queryProjection, normalizedAttentionInput);
		const auto keys = Layer::AddLinear(subgraph, block.keyProjection, normalizedAttentionInput);
		const auto values = Layer::AddLinear(subgraph, block.valueProjection, normalizedAttentionInput);

		std::vector<NodeOutput> headContexts;
		headContexts.reserve(hyperparameters.attentionHeadCount);
		for (std::size_t headIndex = 0; headIndex < hyperparameters.attentionHeadCount; ++headIndex)
		{
			const auto kvHeadIndex = headIndex / queryGroupsPerKVHead;
			const auto queryHead = NodeOutput{ subgraph.AddNode(
			                                    SliceNode{ queries, 1, headIndex * headDim, headDim },
			                                    { OutputInfo{ hiddenInfo.dtype, { hiddenInfo.shape[0], headDim } } }),
			                                0 };
			const auto keyHead = NodeOutput{ subgraph.AddNode(
			                                  SliceNode{ keys, 1, kvHeadIndex * headDim, headDim },
			                                  { OutputInfo{ hiddenInfo.dtype, { hiddenInfo.shape[0], headDim } } }),
			                              0 };
			const auto valueHead = NodeOutput{ subgraph.AddNode(
			                                    SliceNode{ values, 1, kvHeadIndex * headDim, headDim },
			                                    { OutputInfo{ hiddenInfo.dtype, { hiddenInfo.shape[0], headDim } } }),
			                                0 };
			headContexts.push_back(
			    AddSingleHeadAttention(subgraph, queryHead, keyHead, valueHead, hyperparameters, positionOffset));
		}

		NodeOutput mergedContext = headContexts.front();
		if (headContexts.size() > 1)
		{
			mergedContext = { subgraph.AddNode(
			                      ConcatNode{ headContexts, 1 },
			                      { OutputInfo{ hiddenInfo.dtype,
			                                    { hiddenInfo.shape[0], hyperparameters.embeddingLength } } }),
			                  0 };
		}

		const auto attentionOutput = Layer::AddLinear(subgraph, block.outputProjection, mergedContext);
		const auto attentionResidual =
		    NodeOutput{ subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, hiddenState, attentionOutput }, { hiddenInfo }), 0 };
		const auto normalizedFeedForwardInput = Layer::AddRMSNorm(subgraph, block.feedForwardNorm, attentionResidual);
		const auto feedForwardOutput = Layer::AddSwiGLUMLP(subgraph, block.mlp, normalizedFeedForwardInput);
		return { subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, attentionResidual, feedForwardOutput }, { hiddenInfo }), 0 };
	}

	namespace
	{
		struct BlockDecodeResult
		{
			NodeOutput hiddenState;
			Layer::KVCachePair updatedCache;
		};
	}

	BlockDecodeResult AddLLaMADecoderBlockDecode(Subgraph& subgraph, const LLaMADecoderBlock& block,
	                                             const LLaMAHyperparameters& hyperparameters,
	                                             NodeOutput hiddenState, Layer::KVCachePair pastCache,
	                                             std::size_t positionOffset)
	{
		const auto hiddenInfo = subgraph.GetOutputInfo(hiddenState);
		if (hiddenInfo.dtype != block.attentionNorm.dtype || hiddenInfo.shape.size() != 2 ||
		    hiddenInfo.shape[1] != hyperparameters.embeddingLength)
		{
			throw std::runtime_error("LLaMA decoder decode block input must be 2D [sequence, embeddingLength]");
		}

		const auto pastKeyInfo = subgraph.GetOutputInfo(pastCache.keys);
		const auto pastValueInfo = subgraph.GetOutputInfo(pastCache.values);
		const auto headDim = hyperparameters.HeadDimension();
		const auto queryGroupsPerKVHead = hyperparameters.QueryGroupsPerKVHead();
		const std::vector<std::size_t> cacheShape{ positionOffset, hyperparameters.attentionHeadCountKV, headDim };
		if (pastKeyInfo.dtype != hiddenInfo.dtype || pastValueInfo.dtype != hiddenInfo.dtype ||
		    pastKeyInfo.shape != cacheShape || pastValueInfo.shape != cacheShape)
		{
			throw std::runtime_error("LLaMA decode cache tensors must have shape [pastLength, kvHeadCount, headDim]");
		}

		const auto normalizedAttentionInput = Layer::AddRMSNorm(subgraph, block.attentionNorm, hiddenState);
		const auto queries = Layer::AddLinear(subgraph, block.queryProjection, normalizedAttentionInput);
		const auto keys = Layer::AddLinear(subgraph, block.keyProjection, normalizedAttentionInput);
		const auto values = Layer::AddLinear(subgraph, block.valueProjection, normalizedAttentionInput);
		const auto sequenceLength = hiddenInfo.shape[0];
		const auto keys3D = Reshape3D(subgraph, keys, sequenceLength, hyperparameters.attentionHeadCountKV, headDim);
		const auto values3D = Reshape3D(subgraph, values, sequenceLength, hyperparameters.attentionHeadCountKV, headDim);

		std::vector<NodeOutput> rotatedKeyHeads;
		rotatedKeyHeads.reserve(hyperparameters.attentionHeadCountKV);
		for (std::size_t kvHeadIndex = 0; kvHeadIndex < hyperparameters.attentionHeadCountKV; ++kvHeadIndex)
		{
			const auto keyHead3D = NodeOutput{ subgraph.AddNode(
			                                    SliceNode{ keys3D, 1, kvHeadIndex, 1 },
			                                    { OutputInfo{ hiddenInfo.dtype, { sequenceLength, 1, headDim } } }),
			                                0 };
			const auto keyHead2D = Reshape2D(subgraph, keyHead3D, sequenceLength, headDim);
			const auto rotatedKeyHead = AddLLaMARoPE(subgraph, keyHead2D, hyperparameters, positionOffset);
			rotatedKeyHeads.push_back(Reshape3D(subgraph, rotatedKeyHead, sequenceLength, 1, headDim));
		}

		NodeOutput rotatedKeys3D = rotatedKeyHeads.front();
		if (rotatedKeyHeads.size() > 1)
		{
			rotatedKeys3D = { subgraph.AddNode(
			                      ConcatNode{ rotatedKeyHeads, 1 },
			                      { OutputInfo{ hiddenInfo.dtype,
			                                    { sequenceLength, hyperparameters.attentionHeadCountKV, headDim } } }),
			                  0 };
		}
		const auto updatedCache = Layer::AddKVCacheAppend(subgraph, pastCache, { rotatedKeys3D, values3D }, 0);
		const auto totalKeyLength = positionOffset + sequenceLength;

		std::vector<NodeOutput> headContexts;
		headContexts.reserve(hyperparameters.attentionHeadCount);
		for (std::size_t headIndex = 0; headIndex < hyperparameters.attentionHeadCount; ++headIndex)
		{
			const auto kvHeadIndex = headIndex / queryGroupsPerKVHead;
			const auto queryHead = NodeOutput{ subgraph.AddNode(
			                                    SliceNode{ queries, 1, headIndex * headDim, headDim },
			                                    { OutputInfo{ hiddenInfo.dtype, { sequenceLength, headDim } } }),
			                                0 };
			const auto keyHead3D = NodeOutput{ subgraph.AddNode(
			                                    SliceNode{ updatedCache.keys, 1, kvHeadIndex, 1 },
			                                    { OutputInfo{ hiddenInfo.dtype, { totalKeyLength, 1, headDim } } }),
			                                0 };
			const auto valueHead3D = NodeOutput{ subgraph.AddNode(
			                                      SliceNode{ updatedCache.values, 1, kvHeadIndex, 1 },
			                                      { OutputInfo{ hiddenInfo.dtype, { totalKeyLength, 1, headDim } } }),
			                                  0 };
			const auto keyHead = Reshape2D(subgraph, keyHead3D, totalKeyLength, headDim);
			const auto valueHead = Reshape2D(subgraph, valueHead3D, totalKeyLength, headDim);
			headContexts.push_back(AddSingleHeadAttentionWithRotatedKV(subgraph, queryHead, keyHead, valueHead,
			                                                           hyperparameters, positionOffset));
		}

		NodeOutput mergedContext = headContexts.front();
		if (headContexts.size() > 1)
		{
			mergedContext = { subgraph.AddNode(
			                      ConcatNode{ headContexts, 1 },
			                      { OutputInfo{ hiddenInfo.dtype,
			                                    { sequenceLength, hyperparameters.embeddingLength } } }),
			                  0 };
		}

		const auto attentionOutput = Layer::AddLinear(subgraph, block.outputProjection, mergedContext);
		const auto attentionResidual =
		    NodeOutput{ subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, hiddenState, attentionOutput }, { hiddenInfo }), 0 };
		const auto normalizedFeedForwardInput = Layer::AddRMSNorm(subgraph, block.feedForwardNorm, attentionResidual);
		const auto feedForwardOutput = Layer::AddSwiGLUMLP(subgraph, block.mlp, normalizedFeedForwardInput);
		return {
			.hiddenState = { subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, attentionResidual, feedForwardOutput },
			                                  { hiddenInfo }),
			                 0 },
			.updatedCache = updatedCache,
		};
	}

	SubgraphId BuildLLaMADecoderBlock(Graph& graph, const LLaMADecoderBlock& block,
	                                const LLaMAHyperparameters& hyperparameters,
	                                std::size_t sequenceLength, std::size_t positionOffset)
	{
		Subgraph subgraph;
		const auto hiddenState = subgraph.AddParam(block.attentionNorm.dtype,
		                                          { sequenceLength, hyperparameters.embeddingLength });
		const auto result = AddLLaMADecoderBlock(subgraph, block, hyperparameters, { hiddenState, 0 }, positionOffset);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}

	LLaMACausalLM CreateLLaMACausalLM(Graph& graph, const Graph& archive,
	                                 const LLaMAHyperparameters& hyperparameters)
	{
		const auto tokenEmbeddingVariable = ImportNamedVariable(graph, archive, "token_embd.weight");
		const auto& tokenEmbedding = RequirePlainFloatingVariable(graph, tokenEmbeddingVariable, "token_embd.weight");
		if (tokenEmbedding.Data().Shape().NumDim() != 2)
		{
			throw std::runtime_error("GGUF tensor 'token_embd.weight' must be 2D for current LLaMA lowering");
		}
		const auto tokenEmbeddingShape = tokenEmbedding.Data().Shape();
		const auto vocabMajor = tokenEmbeddingShape[1] == hyperparameters.embeddingLength;
		const auto featureMajor = tokenEmbeddingShape[0] == hyperparameters.embeddingLength;
		if (!vocabMajor && !featureMajor)
		{
			throw std::runtime_error(std::format(
			    "GGUF tensor 'token_embd.weight' must have LiteNN shape [vocab, {}] or legacy shape [{}, vocab] for current LLaMA lowering",
			    hyperparameters.embeddingLength, hyperparameters.embeddingLength));
		}
		const auto vocabSize = vocabMajor ? tokenEmbeddingShape[0] : tokenEmbeddingShape[1];
		if (vocabSize == 0)
		{
			throw std::runtime_error("GGUF tensor 'token_embd.weight' must have a non-zero vocabulary dimension");
		}

		LLaMACausalLM model;
		model.tokenEmbeddingVariable = tokenEmbeddingVariable;
		model.vocabSize = vocabSize;
		model.tokenEmbeddingIsVocabMajor = vocabMajor;
		model.dtype = tokenEmbedding.Data().DType();
		model.blocks.reserve(hyperparameters.blockCount);
		for (std::size_t blockIndex = 0; blockIndex < hyperparameters.blockCount; ++blockIndex)
		{
			model.blocks.push_back(CreateLLaMADecoderBlock(graph, archive, hyperparameters, blockIndex));
		}
		model.outputNorm = MakeRMSNormFromArchive(graph, archive, "output_norm.weight", hyperparameters.embeddingLength,
		                                        hyperparameters.rmsNormEpsilon);

		if (archive.FindVariable("output.weight"))
		{
			model.lmHead = MakeLinearFromArchive(graph, archive, "output.weight", hyperparameters.embeddingLength,
			                                    vocabSize);
		}
		else
		{
			model.lmHead = {
				.weightVariable = tokenEmbeddingVariable,
				.biasVariable = std::nullopt,
				.inFeatures = hyperparameters.embeddingLength,
				.outFeatures = vocabSize,
				.dtype = model.dtype,
			};
		}

		return model;
	}

	NodeOutput AddLLaMATokenEmbedding(Subgraph& subgraph, const LLaMACausalLM& model, NodeOutput tokenIds)
	{
		const auto info = subgraph.GetOutputInfo(tokenIds);
		if ((info.dtype != DataType::Int32 && info.dtype != DataType::Int64) || info.shape.size() != 1)
		{
			throw std::runtime_error("LLaMA token id input must be 1D [sequence] with Int32 or Int64 dtype");
		}

		const std::vector<std::size_t> tokenEmbeddingShape = model.tokenEmbeddingIsVocabMajor
		                                                       ? std::vector<std::size_t>{ model.vocabSize,
		                                                                                   model.outputNorm.featureSize }
		                                                       : std::vector<std::size_t>{ model.outputNorm.featureSize,
		                                                                                   model.vocabSize };
		const auto tokenEmbedding = subgraph.AddNode(VariableRefNode{ model.tokenEmbeddingVariable },
		                                           { OutputInfo{ model.dtype, tokenEmbeddingShape } });
		const auto tokenEmbeddingRows =
		    model.tokenEmbeddingIsVocabMajor ? NodeOutput{ tokenEmbedding, 0 } : AddTranspose(subgraph, { tokenEmbedding, 0 });
		const auto hiddenState = subgraph.AddNode(
		    GetRowsNode{ tokenEmbeddingRows, tokenIds },
		    { OutputInfo{ model.dtype, { info.shape[0], model.outputNorm.featureSize } } });
		return { hiddenState, 0 };
	}

	NodeOutput AddLLaMACausalLM(Subgraph& subgraph, const LLaMACausalLM& model,
	                           const LLaMAHyperparameters& hyperparameters, NodeOutput tokenIds,
	                           std::size_t positionOffset)
	{
		auto hiddenState = AddLLaMATokenEmbedding(subgraph, model, tokenIds);
		for (const auto& block : model.blocks)
		{
			hiddenState = AddLLaMADecoderBlock(subgraph, block, hyperparameters, hiddenState, positionOffset);
		}
		const auto normalized = Layer::AddRMSNorm(subgraph, model.outputNorm, hiddenState);
		return Layer::AddLinear(subgraph, model.lmHead, normalized);
	}

	LLaMADecodeResult AddLLaMACausalLMDecode(Subgraph& subgraph, const LLaMACausalLM& model,
	                                         const LLaMAHyperparameters& hyperparameters, NodeOutput tokenIds,
	                                         std::span<const Layer::KVCachePair> pastCaches,
	                                         std::size_t positionOffset)
	{
		if (pastCaches.size() != model.blocks.size())
		{
			throw std::runtime_error("LLaMA decode requires one KV cache pair per decoder block");
		}

		auto hiddenState = AddLLaMATokenEmbedding(subgraph, model, tokenIds);
		std::vector<Layer::KVCachePair> updatedCaches;
		updatedCaches.reserve(model.blocks.size());
		for (std::size_t blockIndex = 0; blockIndex < model.blocks.size(); ++blockIndex)
		{
			auto blockResult = AddLLaMADecoderBlockDecode(subgraph, model.blocks[blockIndex], hyperparameters,
			                                             hiddenState, pastCaches[blockIndex], positionOffset);
			hiddenState = blockResult.hiddenState;
			updatedCaches.push_back(blockResult.updatedCache);
		}

		const auto normalized = Layer::AddRMSNorm(subgraph, model.outputNorm, hiddenState);
		return {
			.hiddenState = Layer::AddLinear(subgraph, model.lmHead, normalized),
			.updatedCaches = std::move(updatedCaches),
		};
	}

	SubgraphId BuildLLaMACausalLM(Graph& graph, const LLaMACausalLM& model,
	                            const LLaMAHyperparameters& hyperparameters,
	                            std::size_t sequenceLength, std::size_t positionOffset)
	{
		Subgraph subgraph;
		const auto tokenIds = subgraph.AddParam(DataType::Int32, { sequenceLength });
		const auto logits = AddLLaMACausalLM(subgraph, model, hyperparameters, { tokenIds, 0 }, positionOffset);
		subgraph.SetResults({ logits });
		return graph.AddSubgraph(std::move(subgraph));
	}

	Graph LowerLLaMACausalLM(const Graph& archive, std::size_t sequenceLength, std::size_t positionOffset)
	{
		auto graph = Graph{};
		graph.SetMetadata(CopyMetadata(archive));
		const auto hyperparameters = ParseLLaMAHyperparameters(archive);
		const auto model = CreateLLaMACausalLM(graph, archive, hyperparameters);
		const auto forward = BuildLLaMACausalLM(graph, model, hyperparameters, sequenceLength, positionOffset);
		graph.SetForward(forward);
		graph.SetInputNames({ "token_ids" });
		graph.SetOutputNames({ "logits" });
		return graph;
	}

	Graph LowerLLaMACausalLMDecode(const Graph& archive, std::size_t sequenceLength, std::size_t pastLength,
	                               std::size_t positionOffset)
	{
		auto graph = Graph{};
		graph.SetMetadata(CopyMetadata(archive));
		const auto hyperparameters = ParseLLaMAHyperparameters(archive);
		if (positionOffset != pastLength)
		{
			throw std::runtime_error("Current LLaMA decode lowering requires positionOffset == pastLength");
		}

		const auto model = CreateLLaMACausalLM(graph, archive, hyperparameters);
		const auto headDim = hyperparameters.HeadDimension();

		Subgraph subgraph;
		const auto tokenIds = subgraph.AddParam(DataType::Int32, { sequenceLength });
		std::vector<Layer::KVCachePair> pastCaches;
		pastCaches.reserve(model.blocks.size());
		std::vector<std::string> inputNames{ "token_ids" };
		for (std::size_t blockIndex = 0; blockIndex < model.blocks.size(); ++blockIndex)
		{
			const std::vector<std::size_t> cacheShape{ pastLength, hyperparameters.attentionHeadCountKV, headDim };
			const auto keys = subgraph.AddParam(model.dtype, cacheShape);
			const auto values = subgraph.AddParam(model.dtype, cacheShape);
			pastCaches.push_back({ { keys, 0 }, { values, 0 } });
			inputNames.push_back(std::format("past_key_{}", blockIndex));
			inputNames.push_back(std::format("past_value_{}", blockIndex));
		}

		const auto result = AddLLaMACausalLMDecode(subgraph, model, hyperparameters, { tokenIds, 0 }, pastCaches,
		                                          positionOffset);
		std::vector<NodeOutput> outputs{ result.hiddenState };
		std::vector<std::string> outputNames{ "logits" };
		for (std::size_t blockIndex = 0; blockIndex < result.updatedCaches.size(); ++blockIndex)
		{
			outputs.push_back(result.updatedCaches[blockIndex].keys);
			outputs.push_back(result.updatedCaches[blockIndex].values);
			outputNames.push_back(std::format("updated_key_{}", blockIndex));
			outputNames.push_back(std::format("updated_value_{}", blockIndex));
		}
		subgraph.SetResults(std::move(outputs));
		const auto forward = graph.AddSubgraph(std::move(subgraph));
		graph.SetForward(forward);
		graph.SetInputNames(std::move(inputNames));
		graph.SetOutputNames(std::move(outputNames));
		return graph;
	}
} // namespace LiteNN::GGUF
