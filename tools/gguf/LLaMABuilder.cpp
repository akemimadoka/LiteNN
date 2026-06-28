#include "LLaMABuilder.h"
#include "GGMLQuantizedKernels.h"

#include <LiteNN/Layer/LayerUtils.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace LiteNN::GGUF
{
	namespace
	{
		std::shared_ptr<Variable> MaterializeArchiveVariable(const Graph& archive, std::size_t variableIndex,
		                                                     std::string_view name)
		{
			const auto& source = archive.GetVariable(variableIndex);
			if (!source->IsQuantized())
			{
				return source;
			}
			return Variable::Create(DequantizeGGMLBlockVariable(*source, name));
		}

		std::string BlockTensorName(std::size_t blockIndex, std::string_view suffix)
		{
			return std::format("blk.{}.{}", blockIndex, suffix);
		}

		std::size_t ImportNamedVariable(Graph& target, const Graph& archive, std::string_view name,
		                                bool preserveQuantized = false)
		{
			const auto sourceIndex = archive.FindVariable(name);
			if (!sourceIndex)
			{
				throw std::runtime_error(std::format("Missing GGUF tensor '{}'", name));
			}
			const auto& source = archive.GetVariable(*sourceIndex);
			const auto targetIndex = target.AddVariable(preserveQuantized && source->IsQuantized()
			                                                ? source
			                                                : MaterializeArchiveVariable(archive, *sourceIndex, name));
			target.SetVariableName(targetIndex, std::string(name));
			return targetIndex;
		}

		struct ImportedLinearBias
		{
			std::size_t variable{};
			std::vector<std::size_t> shape;
		};

		std::optional<ImportedLinearBias> ImportOptionalLinearBias(Graph& target, const Graph& archive,
		                                                           std::string_view weightName, std::size_t outFeatures,
		                                                           DataType dtype)
		{
			if (!weightName.ends_with(".weight"))
			{
				return std::nullopt;
			}
			auto biasName = std::string(weightName.substr(0, weightName.size() - std::string_view("weight").size()));
			biasName += "bias";
			const auto sourceIndex = archive.FindVariable(biasName);
			if (!sourceIndex)
			{
				return std::nullopt;
			}

			auto materialized = MaterializeArchiveVariable(archive, *sourceIndex, biasName);
			if (materialized->IsQuantized() || materialized->Data().DType() != dtype)
			{
				throw std::runtime_error(
				    std::format("GGUF Linear bias '{}' must use the projection's expressed dtype {}", biasName,
				                DataTypeName(dtype)));
			}
			const auto shape = materialized->Data().Shape().ToOwned();
			if (shape != std::vector<std::size_t>{ outFeatures } && shape != std::vector<std::size_t>{ 1, outFeatures })
			{
				throw std::runtime_error(std::format("GGUF Linear bias '{}' must have shape [{}] or [1, {}]", biasName,
				                                     outFeatures, outFeatures));
			}
			if (shape == std::vector<std::size_t>{ outFeatures })
			{
				auto data = materialized->Data().CopyToDevice(CPU{});
				data.Reshape({ 1, outFeatures });
				materialized = Variable::Create(std::move(data));
			}
			const auto variable = target.AddVariable(std::move(materialized));
			target.SetVariableName(variable, biasName);
			return ImportedLinearBias{ variable, { 1, outFeatures } };
		}

		const Variable& RequirePlainFloatingVariable(const Graph& graph, std::size_t variableIndex,
		                                             std::string_view name)
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
				throw std::runtime_error(
				    std::format("GGUF tensor '{}' must be floating-point for current LLaMA block lowering", name));
			}
			return variable;
		}

		Layer::LinearLayer MakeLinearFromArchive(Graph& target, const Graph& archive, std::string_view name,
		                                         std::size_t inFeatures, std::size_t outFeatures,
		                                         bool preserveQuantized = false)
		{
			const auto sourceIndex = archive.FindVariable(name);
			if (!sourceIndex)
			{
				throw std::runtime_error(std::format("Missing GGUF tensor '{}'", name));
			}

			const auto& source = archive.GetVariable(*sourceIndex);
			if (preserveQuantized && source->IsQuantized())
			{
				const auto& params = *source->Quantization();
				const auto& shape = params.expressedShape;
				const auto transpose = shape == std::vector<std::size_t>{ outFeatures, inFeatures };
				if (!transpose && shape != std::vector<std::size_t>{ inFeatures, outFeatures })
				{
					throw std::runtime_error(
					    std::format("GGUF quantized tensor '{}' must have expressed shape [{}, {}] or [{}, {}]", name,
					                inFeatures, outFeatures, outFeatures, inFeatures));
				}
				const auto variableIndex = target.AddVariable(source);
				target.SetVariableName(variableIndex, std::string(name));
				const auto bias = ImportOptionalLinearBias(target, archive, name, outFeatures, params.expressedType);
				return {
					.weightVariable = variableIndex,
					.biasVariable = bias ? std::optional{ bias->variable } : std::nullopt,
					.biasShape = bias ? bias->shape : std::vector<std::size_t>{},
					.inFeatures = inFeatures,
					.outFeatures = outFeatures,
					.dtype = params.expressedType,
					.weightQuantization = params,
					.weightStorageShape = source->Data().Shape().ToOwned(),
					.transposeWeight = transpose,
				};
			}

			auto materialized = MaterializeArchiveVariable(archive, *sourceIndex, name);
			if (materialized->IsQuantized() || !IsFloatingDataType(materialized->Data().DType()))
			{
				throw std::runtime_error(
				    std::format("GGUF tensor '{}' must be floating-point for current LLaMA block lowering", name));
			}

			auto data = materialized->Data().CopyToDevice(CPU{});
			if (data.Shape().NumDim() == 2 && data.Shape()[0] == outFeatures && data.Shape()[1] == inFeatures)
			{
				data = data.Transpose();
				materialized = Variable::Create(std::move(data));
			}
			else if (data.Shape().NumDim() != 2 || data.Shape()[0] != inFeatures || data.Shape()[1] != outFeatures)
			{
				throw std::runtime_error(std::format("GGUF tensor '{}' must have LiteNN shape [{}, {}] or imported "
				                                     "GGUF shape [{}, {}] for current LLaMA block lowering",
				                                     name, inFeatures, outFeatures, outFeatures, inFeatures));
			}

			const auto variableIndex = target.AddVariable(std::move(materialized));
			target.SetVariableName(variableIndex, std::string(name));
			const auto bias = ImportOptionalLinearBias(target, archive, name, outFeatures,
			                                           target.GetVariable(variableIndex)->Data().DType());
			return {
				.weightVariable = variableIndex,
				.biasVariable = bias ? std::optional{ bias->variable } : std::nullopt,
				.biasShape = bias ? bias->shape : std::vector<std::size_t>{},
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
				throw std::runtime_error(
				    std::format("GGUF tensor '{}' must be floating-point for current LLaMA block lowering", name));
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
				throw std::runtime_error(
				    std::format("GGUF tensor '{}' must have shape [{}] or [1, {}] for current LLaMA block lowering",
				                name, featureSize, featureSize));
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
				throw std::runtime_error(std::format("Current LLaMA {} lowering preserves rope.scaling.* metadata but "
				                                     "only executes none/linear scaling, got '{}'",
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
				return NodeOutput{
					subgraph.AddNode(
					    SliceNode{ input, 1, 0, hyperparameters.ropeDimensionCount },
					    { OutputInfo{ info.dtype, { info.shape[0], hyperparameters.ropeDimensionCount } } }),
					0
				};
			}();
			const auto rotated = Layer::AddRoPE(subgraph, rotatedPrefix, hyperparameters.ropeFrequencyBase,
			                                    positionOffset, hyperparameters.ropeFrequencyScale);
			if (hyperparameters.ropeDimensionCount == info.shape[1])
			{
				return rotated;
			}

			const auto tailWidth = info.shape[1] - hyperparameters.ropeDimensionCount;
			const auto tail =
			    NodeOutput{ subgraph.AddNode(SliceNode{ input, 1, hyperparameters.ropeDimensionCount, tailWidth },
				                             { OutputInfo{ info.dtype, { info.shape[0], tailWidth } } }),
				            0 };
			return { subgraph.AddNode(ConcatNode{ { rotated, tail }, 1 }, { info }), 0 };
		}

		NodeOutput AddLLaMARoPEAtPositions(Subgraph& subgraph, NodeOutput input, NodeOutput positions,
		                                   const LLaMAHyperparameters& hyperparameters)
		{
			const auto info = subgraph.GetOutputInfo(input);
			if (info.shape.size() != 2 || hyperparameters.ropeDimensionCount > info.shape[1] ||
			    (hyperparameters.ropeDimensionCount % 2) != 0)
			{
				throw std::runtime_error(
				    "Dynamic LLaMA RoPE requires a 2D input and a valid even rope.dimension_count");
			}
			const auto rotatedPrefix =
			    hyperparameters.ropeDimensionCount == info.shape[1]
			        ? input
			        : NodeOutput{
				          subgraph.AddNode(
				              SliceNode{ input, 1, 0, hyperparameters.ropeDimensionCount },
				              { OutputInfo{ info.dtype, { info.shape[0], hyperparameters.ropeDimensionCount } } }),
				          0
			          };
			const auto rotated =
			    Layer::AddRoPEAtPositions(subgraph, rotatedPrefix, positions, hyperparameters.ropeFrequencyBase,
			                              hyperparameters.ropeFrequencyScale);
			if (hyperparameters.ropeDimensionCount == info.shape[1])
			{
				return rotated;
			}
			const auto tailWidth = info.shape[1] - hyperparameters.ropeDimensionCount;
			const auto tail = NodeOutput{
				subgraph.AddNode(SliceNode{ input, 1, hyperparameters.ropeDimensionCount, tailWidth },
				                 { OutputInfo{ info.dtype, { info.shape[0], tailWidth } } }),
				0,
			};
			return { subgraph.AddNode(ConcatNode{ { rotated, tail }, 1 }, { info }), 0 };
		}

		NodeOutput AddActiveCacheMask(Subgraph& subgraph, NodeOutput currentPosition, std::size_t maxCacheLength,
		                              DataType dtype)
		{
			std::vector<double> positions(maxCacheLength);
			std::iota(positions.begin(), positions.end(), 0.0);
			const auto positionTable = Layer::Detail::AddConstant(
			    subgraph, Tensor<CPU>(std::span<const double>(positions), { 1, maxCacheLength }, DataType::Int64));
			const auto inactive =
			    subgraph.AddNode(BinaryOpNode{ BinaryOp::Greater, { positionTable, 0 }, currentPosition },
			                     { OutputInfo{ DataType::Bool, { 1, maxCacheLength } } });
			const auto typedMask =
			    subgraph.AddNode(CastNode{ { inactive, 0 }, dtype }, { OutputInfo{ dtype, { 1, maxCacheLength } } });
			const auto negative = Layer::Detail::AddConstant(subgraph, Layer::Detail::MakeScalarTensor(dtype, -1.0e9));
			return { subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, { typedMask, 0 }, { negative, 0 } },
				                      { OutputInfo{ dtype, { 1, maxCacheLength } } }),
				     0 };
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
		                                               NodeOutput values, const LLaMAHyperparameters& hyperparameters,
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
				throw std::runtime_error(
				    "Single-head decode attention expects value tensor shape [keyLength, headDim]");
			}
			const auto rotatedQueries = AddLLaMARoPE(subgraph, queries, hyperparameters, queryPositionOffset);
			Layer::FlashAttnExtOptions options;
			options.scale = 1.0 / std::sqrt(static_cast<double>(queryInfo.shape[1]));
			options.causal = true;
			options.keyPositionOffset = 0;
			options.queryPositionOffset = queryPositionOffset;
			return Layer::AddFlashAttnExt(subgraph, rotatedQueries, rotatedKeys, values, options);
		}

		NodeOutput AddSingleHeadAttentionAtPosition(Subgraph& subgraph, NodeOutput queries, NodeOutput rotatedKeys,
		                                            NodeOutput values, NodeOutput currentPosition,
		                                            NodeOutput activeMask, const LLaMAHyperparameters& hyperparameters)
		{
			ValidateSupportedRoPE(hyperparameters, "dynamic decode");
			const auto queryInfo = subgraph.GetOutputInfo(queries);
			const auto keyInfo = subgraph.GetOutputInfo(rotatedKeys);
			const auto valueInfo = subgraph.GetOutputInfo(values);
			if (queryInfo.shape.size() != 2 || queryInfo.shape[0] != 1 || keyInfo.shape.size() != 2 ||
			    keyInfo.shape[1] != queryInfo.shape[1] || valueInfo.shape.size() != 2 ||
			    valueInfo.shape[0] != keyInfo.shape[0])
			{
				throw std::runtime_error("Dynamic decode attention received incompatible query/key/value shapes");
			}
			const auto rotatedQueries = AddLLaMARoPEAtPositions(subgraph, queries, currentPosition, hyperparameters);
			Layer::FlashAttnExtOptions options;
			options.scale = 1.0 / std::sqrt(static_cast<double>(queryInfo.shape[1]));
			options.mask = activeMask;
			return Layer::AddFlashAttnExt(subgraph, rotatedQueries, rotatedKeys, values, options);
		}

		NodeOutput Reshape2D(Subgraph& subgraph, NodeOutput input, std::size_t rows, std::size_t cols)
		{
			const auto info = subgraph.GetOutputInfo(input);
			return {
				subgraph.AddNode(ReshapeNode{ input, { rows, cols } }, { OutputInfo{ info.dtype, { rows, cols } } }), 0
			};
		}

		NodeOutput Reshape3D(Subgraph& subgraph, NodeOutput input, std::size_t dim0, std::size_t dim1, std::size_t dim2)
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
			throw std::runtime_error(std::format(
			    "LLaMA parity tolerance is only defined for floating-point logits, got {}", DataTypeName(dtype)));
		}
	}

	LLaMAArtifactPlan PlanLLaMAArtifacts(const Graph& archive, const LLaMAArtifactPlanningOptions& options)
	{
		if (options.prefillSequenceLength == 0)
		{
			throw std::runtime_error("LLaMA artifact plan requires prefillSequenceLength > 0");
		}
		const auto requiredCacheLength = options.decodePastLength + 1;
		const auto maxCacheLength = options.maxCacheLength == 0 ? requiredCacheLength : options.maxCacheLength;
		if (maxCacheLength < requiredCacheLength)
		{
			throw std::runtime_error("LLaMA artifact plan requires maxCacheLength >= decodePastLength + 1");
		}
		const auto hyperparameters = ParseLLaMAHyperparameters(archive);
		const auto tokenEmbeddingIndex = archive.FindVariable("token_embd.weight");
		if (!tokenEmbeddingIndex)
		{
			throw std::runtime_error("Missing GGUF tensor 'token_embd.weight'");
		}

		const auto& tokenEmbedding = *archive.GetVariable(*tokenEmbeddingIndex);
		const auto dtype =
		    tokenEmbedding.IsQuantized() ? tokenEmbedding.Quantization()->expressedType : tokenEmbedding.Data().DType();
		std::vector<std::size_t> embeddingShape;
		if (tokenEmbedding.IsQuantized())
		{
			embeddingShape = tokenEmbedding.Quantization()->expressedShape;
		}
		else
		{
			const auto shape = tokenEmbedding.Data().Shape();
			embeddingShape.reserve(shape.NumDim());
			for (std::size_t i = 0; i < shape.NumDim(); ++i)
			{
				embeddingShape.push_back(shape[i]);
			}
		}
		if (embeddingShape.size() != 2)
		{
			throw std::runtime_error("GGUF tensor 'token_embd.weight' must be 2D for LLaMA artifact planning");
		}
		const auto vocabMajor = embeddingShape[1] == hyperparameters.embeddingLength;
		const auto featureMajor = embeddingShape[0] == hyperparameters.embeddingLength;
		if (!vocabMajor && !featureMajor)
		{
			throw std::runtime_error("GGUF tensor 'token_embd.weight' is incompatible with LLaMA embedding_length");
		}
		const auto vocabSize = vocabMajor ? embeddingShape[0] : embeddingShape[1];
		const auto headDim = hyperparameters.HeadDimension();
		const std::vector<std::size_t> cacheShape{
			options.dynamicDecodePosition ? maxCacheLength : options.decodePastLength,
			hyperparameters.attentionHeadCountKV,
			headDim,
		};
		const auto cacheType = TensorType::Dense(dtype, ShapeView{ cacheShape });
		const std::vector<std::size_t> stateShape{ 2, maxCacheLength, hyperparameters.attentionHeadCountKV, headDim };
		const auto stateType = TensorType::Dense(dtype, ShapeView{ stateShape });
		const auto cacheCapacityPerPlaneBytes =
		    maxCacheLength * hyperparameters.attentionHeadCountKV * headDim * ElementByteSize(dtype);
		const auto stateByteSize = stateType.ByteSize().value_or(0);
		const auto tokenByteStride = hyperparameters.attentionHeadCountKV * headDim * ElementByteSize(dtype);

		LLaMAArtifactEntry prefill{
			.kind = LLaMAArtifactKind::Prefill,
			.name = "prefill",
			.sequenceLength = options.prefillSequenceLength,
			.pastLength = 0,
			.maxCacheLength = maxCacheLength,
			.positionOffset = 0,
			.dynamicPosition = false,
			.inputNames = { "token_ids" },
			.outputNames = { "logits" },
			.kvCaches = {},
		};

		LLaMAArtifactEntry decode{
			.kind = LLaMAArtifactKind::DecodeStep,
			.name = "decode_step",
			.sequenceLength = 1,
			.pastLength = options.decodePastLength,
			.maxCacheLength = maxCacheLength,
			.positionOffset = options.decodePastLength,
			.dynamicPosition = options.dynamicDecodePosition,
			.inputNames = options.dynamicDecodePosition ? std::vector<std::string>{ "token_ids", "current_position" }
			                                            : std::vector<std::string>{ "token_ids" },
			.outputNames = options.dynamicDecodePosition ? std::vector<std::string>{ "logits", "next_position" }
			                                             : std::vector<std::string>{ "logits" },
			.kvCaches = {},
		};
		decode.kvCaches.reserve(hyperparameters.blockCount);
		for (std::size_t blockIndex = 0; blockIndex < hyperparameters.blockCount; ++blockIndex)
		{
			auto pastKey = std::format("past_key_{}", blockIndex);
			auto pastValue = std::format("past_value_{}", blockIndex);
			auto updatedKey = std::format("updated_key_{}", blockIndex);
			auto updatedValue = std::format("updated_value_{}", blockIndex);
			decode.inputNames.push_back(pastKey);
			decode.inputNames.push_back(pastValue);
			decode.outputNames.push_back(updatedKey);
			decode.outputNames.push_back(updatedValue);
			auto stateBinding = Runtime::MakeRuntimeStateBinding(
			    std::format("kv.layer{}", blockIndex), Runtime::RuntimeStateKind::KVCache, "kv-cache",
			    TensorType{ stateType }, BufferMutability::Mutable, { "read", "write", "append", "view" });
			decode.kvCaches.push_back({
			    .blockIndex = blockIndex,
			    .pastKeyInput = std::move(pastKey),
			    .pastValueInput = std::move(pastValue),
			    .updatedKeyOutput = std::move(updatedKey),
			    .updatedValueOutput = std::move(updatedValue),
			    .cacheType = cacheType,
			    .stateType = stateType,
			    .stateBinding = stateBinding,
			    .keyByteOffset = 0,
			    .valueByteOffset = cacheCapacityPerPlaneBytes,
			    .layerByteStride = stateByteSize,
			    .tokenByteStride = tokenByteStride,
			});
			const auto stateName = decode.kvCaches.back().stateBinding.name;
			const auto valueBase = options.dynamicDecodePosition ? 2uz : 1uz;
			const auto keyInput = valueBase + blockIndex * 2;
			const auto valueInput = keyInput + 1;
			const auto keyOutput = valueBase + blockIndex * 2;
			const auto valueOutput = keyOutput + 1;
			decode.stateValueBindings.push_back(
			    { stateName, 0, Runtime::RuntimeStateValueKind::FunctionInput, keyInput, 0 });
			decode.stateValueBindings.push_back({ stateName, 0, Runtime::RuntimeStateValueKind::FunctionInput,
			                                      valueInput, cacheCapacityPerPlaneBytes });
			decode.stateValueBindings.push_back(
			    { stateName, 0, Runtime::RuntimeStateValueKind::FunctionOutput, keyOutput, 0 });
			decode.stateValueBindings.push_back({ stateName, 0, Runtime::RuntimeStateValueKind::FunctionOutput,
			                                      valueOutput, cacheCapacityPerPlaneBytes });
		}

		Runtime::LLMDecodeStateABI decodeStateABI;
		decodeStateABI.kvCaches.reserve(decode.kvCaches.size());
		for (const auto& cache : decode.kvCaches)
		{
			decodeStateABI.kvCaches.push_back(cache.stateBinding);
		}
		decodeStateABI.currentPosition = Runtime::MakeRuntimeStateBinding(
		    "decode.position", Runtime::RuntimeStateKind::KVCache, "current-position",
		    TensorType::Dense(DataType::Int64, ShapeView{ std::vector<std::size_t>{ 1 } }), BufferMutability::Mutable,
		    { "read", "write", "increment" });
		if (options.dynamicDecodePosition)
		{
			decode.stateValueBindings.push_back(
			    { decodeStateABI.currentPosition->name, 0, Runtime::RuntimeStateValueKind::FunctionInput, 1, 0 });
			decode.stateValueBindings.push_back(
			    { decodeStateABI.currentPosition->name, 0, Runtime::RuntimeStateValueKind::FunctionOutput, 1, 0 });
		}

		std::vector<LLaMATensorLayoutRecord> tensorLayouts{
			{
			    .name = "gguf.imported_weight",
			    .domain = "imported-gguf",
			    .axes = { "gguf_dim0", "gguf_dim1" },
			    .layout = "source-order",
			    .note = "GGUF tensor payloads are preserved in archive order; LLaMA lowering validates and transposes "
			            "linear "
			            "weights into LiteNN semantic layout when needed.",
			},
			{
			    .name = "litenn.hidden_state",
			    .domain = "litenn-semantic",
			    .axes = { "sequence", "embedding" },
			    .layout = "row-major-2d",
			    .note = "Token embeddings, decoder block inputs, residuals, and logits use sequence-major 2D tensors.",
			},
			{
			    .name = "runtime.functional_kv_cache",
			    .domain = "functional-decode",
			    .axes = { "past", "kv_head", "head_dim" },
			    .layout = "row-major-3d",
			    .note =
			        "Current decode graph accepts key/value cache tensors as explicit functional inputs and outputs.",
			},
			{
			    .name = "runtime.mutable_kv_state",
			    .domain = "runtime-state",
			    .axes = { "key_value", "capacity", "kv_head", "head_dim" },
			    .layout = "row-major-4d",
			    .note = "Artifact ABI exposes a mutable state buffer with key plane at offset 0 and value plane at the "
			            "per-plane capacity offset.",
			},
		};

		return {
			.hyperparameters = hyperparameters,
			.dtype = dtype,
			.vocabSize = vocabSize,
			.prefill = std::move(prefill),
			.decodeStep = std::move(decode),
			.decodeStateABI = std::move(decodeStateABI),
			.tensorLayouts = std::move(tensorLayouts),
		};
	}

	LLaMAArtifactPlan PlanLLaMAArtifacts(const Graph& archive, std::size_t prefillSequenceLength,
	                                     std::size_t decodePastLength)
	{
		return PlanLLaMAArtifacts(archive, { .prefillSequenceLength = prefillSequenceLength,
		                                     .decodePastLength = decodePastLength,
		                                     .maxCacheLength = decodePastLength + 1 });
	}

	LLaMADecoderBlock CreateLLaMADecoderBlock(Graph& graph, const Graph& archive,
	                                          const LLaMAHyperparameters& hyperparameters, std::size_t blockIndex,
	                                          const LLaMALoweringOptions& options)
	{
		const auto headDim = hyperparameters.HeadDimension();
		const auto kvWidth = hyperparameters.attentionHeadCountKV * headDim;

		return {
			.attentionNorm = MakeRMSNormFromArchive(graph, archive, BlockTensorName(blockIndex, "attn_norm.weight"),
			                                      hyperparameters.embeddingLength,
			                                      hyperparameters.rmsNormEpsilon),
			.queryProjection = MakeLinearFromArchive(graph, archive, BlockTensorName(blockIndex, "attn_q.weight"),
			                                        hyperparameters.embeddingLength, hyperparameters.embeddingLength,
			                                        options.preserveQuantizedWeights),
			.keyProjection = MakeLinearFromArchive(graph, archive, BlockTensorName(blockIndex, "attn_k.weight"),
			                                      hyperparameters.embeddingLength, kvWidth,
			                                      options.preserveQuantizedWeights),
			.valueProjection = MakeLinearFromArchive(graph, archive, BlockTensorName(blockIndex, "attn_v.weight"),
			                                        hyperparameters.embeddingLength, kvWidth,
			                                        options.preserveQuantizedWeights),
			.outputProjection = MakeLinearFromArchive(graph, archive,
			                                         BlockTensorName(blockIndex, "attn_output.weight"),
			                                         hyperparameters.embeddingLength, hyperparameters.embeddingLength,
			                                         options.preserveQuantizedWeights),
			.feedForwardNorm = MakeRMSNormFromArchive(graph, archive, BlockTensorName(blockIndex, "ffn_norm.weight"),
			                                        hyperparameters.embeddingLength,
			                                        hyperparameters.rmsNormEpsilon),
			.mlp = {
				.gateProjection = MakeLinearFromArchive(graph, archive, BlockTensorName(blockIndex, "ffn_gate.weight"),
				                                      hyperparameters.embeddingLength,
				                                      hyperparameters.feedForwardLength,
				                                      options.preserveQuantizedWeights),
				.upProjection = MakeLinearFromArchive(graph, archive, BlockTensorName(blockIndex, "ffn_up.weight"),
				                                    hyperparameters.embeddingLength,
				                                    hyperparameters.feedForwardLength,
				                                    options.preserveQuantizedWeights),
				.downProjection = MakeLinearFromArchive(graph, archive, BlockTensorName(blockIndex, "ffn_down.weight"),
				                                      hyperparameters.feedForwardLength,
				                                      hyperparameters.embeddingLength,
				                                      options.preserveQuantizedWeights),
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
			const auto queryHead =
			    NodeOutput{ subgraph.AddNode(SliceNode{ queries, 1, headIndex * headDim, headDim },
				                             { OutputInfo{ hiddenInfo.dtype, { hiddenInfo.shape[0], headDim } } }),
				            0 };
			const auto keyHead =
			    NodeOutput{ subgraph.AddNode(SliceNode{ keys, 1, kvHeadIndex * headDim, headDim },
				                             { OutputInfo{ hiddenInfo.dtype, { hiddenInfo.shape[0], headDim } } }),
				            0 };
			const auto valueHead =
			    NodeOutput{ subgraph.AddNode(SliceNode{ values, 1, kvHeadIndex * headDim, headDim },
				                             { OutputInfo{ hiddenInfo.dtype, { hiddenInfo.shape[0], headDim } } }),
				            0 };
			headContexts.push_back(
			    AddSingleHeadAttention(subgraph, queryHead, keyHead, valueHead, hyperparameters, positionOffset));
		}

		NodeOutput mergedContext = headContexts.front();
		if (headContexts.size() > 1)
		{
			mergedContext = {
				subgraph.AddNode(
				    ConcatNode{ headContexts, 1 },
				    { OutputInfo{ hiddenInfo.dtype, { hiddenInfo.shape[0], hyperparameters.embeddingLength } } }),
				0
			};
		}

		const auto attentionOutput = Layer::AddLinear(subgraph, block.outputProjection, mergedContext);
		const auto attentionResidual =
		    NodeOutput{ subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, hiddenState, attentionOutput }, { hiddenInfo }),
			            0 };
		const auto normalizedFeedForwardInput = Layer::AddRMSNorm(subgraph, block.feedForwardNorm, attentionResidual);
		const auto feedForwardOutput = Layer::AddSwiGLUMLP(subgraph, block.mlp, normalizedFeedForwardInput);
		return { subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, attentionResidual, feedForwardOutput }, { hiddenInfo }),
			     0 };
	}

	namespace
	{
		struct BlockDecodeResult
		{
			NodeOutput hiddenState;
			Layer::KVCachePair updatedCache;
		};
	} // namespace

	BlockDecodeResult AddLLaMADecoderBlockDecode(Subgraph& subgraph, const LLaMADecoderBlock& block,
	                                             const LLaMAHyperparameters& hyperparameters, NodeOutput hiddenState,
	                                             Layer::KVCachePair pastCache, std::size_t positionOffset)
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
		const auto values3D =
		    Reshape3D(subgraph, values, sequenceLength, hyperparameters.attentionHeadCountKV, headDim);

		std::vector<NodeOutput> rotatedKeyHeads;
		rotatedKeyHeads.reserve(hyperparameters.attentionHeadCountKV);
		for (std::size_t kvHeadIndex = 0; kvHeadIndex < hyperparameters.attentionHeadCountKV; ++kvHeadIndex)
		{
			const auto keyHead3D =
			    NodeOutput{ subgraph.AddNode(SliceNode{ keys3D, 1, kvHeadIndex, 1 },
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
			const auto queryHead =
			    NodeOutput{ subgraph.AddNode(SliceNode{ queries, 1, headIndex * headDim, headDim },
				                             { OutputInfo{ hiddenInfo.dtype, { sequenceLength, headDim } } }),
				            0 };
			const auto keyHead3D =
			    NodeOutput{ subgraph.AddNode(SliceNode{ updatedCache.keys, 1, kvHeadIndex, 1 },
				                             { OutputInfo{ hiddenInfo.dtype, { totalKeyLength, 1, headDim } } }),
				            0 };
			const auto valueHead3D =
			    NodeOutput{ subgraph.AddNode(SliceNode{ updatedCache.values, 1, kvHeadIndex, 1 },
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
			mergedContext = { subgraph.AddNode(ConcatNode{ headContexts, 1 },
				                               { OutputInfo{ hiddenInfo.dtype,
				                                             { sequenceLength, hyperparameters.embeddingLength } } }),
				              0 };
		}

		const auto attentionOutput = Layer::AddLinear(subgraph, block.outputProjection, mergedContext);
		const auto attentionResidual =
		    NodeOutput{ subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, hiddenState, attentionOutput }, { hiddenInfo }),
			            0 };
		const auto normalizedFeedForwardInput = Layer::AddRMSNorm(subgraph, block.feedForwardNorm, attentionResidual);
		const auto feedForwardOutput = Layer::AddSwiGLUMLP(subgraph, block.mlp, normalizedFeedForwardInput);
		return {
			.hiddenState = { subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, attentionResidual, feedForwardOutput },
			                                  { hiddenInfo }),
			                 0 },
			.updatedCache = updatedCache,
		};
	}

	BlockDecodeResult AddLLaMADecoderBlockDecodeCapacity(Subgraph& subgraph, const LLaMADecoderBlock& block,
	                                                     const LLaMAHyperparameters& hyperparameters,
	                                                     NodeOutput hiddenState, Layer::KVCachePair cache,
	                                                     NodeOutput currentPosition, std::size_t maxCacheLength)
	{
		const auto hiddenInfo = subgraph.GetOutputInfo(hiddenState);
		const auto positionInfo = subgraph.GetOutputInfo(currentPosition);
		const auto headDim = hyperparameters.HeadDimension();
		const std::vector<std::size_t> cacheShape{ maxCacheLength, hyperparameters.attentionHeadCountKV, headDim };
		if (hiddenInfo.shape != std::vector<std::size_t>{ 1, hyperparameters.embeddingLength } ||
		    positionInfo.dtype != DataType::Int64 || positionInfo.shape != std::vector<std::size_t>{ 1 } ||
		    subgraph.GetOutputInfo(cache.keys).shape != cacheShape ||
		    subgraph.GetOutputInfo(cache.values).shape != cacheShape)
		{
			throw std::runtime_error(
			    "Capacity decode requires one token, Int64 position[1], and full-capacity KV tensors");
		}

		const auto normalizedAttentionInput = Layer::AddRMSNorm(subgraph, block.attentionNorm, hiddenState);
		const auto queries = Layer::AddLinear(subgraph, block.queryProjection, normalizedAttentionInput);
		const auto keys = Layer::AddLinear(subgraph, block.keyProjection, normalizedAttentionInput);
		const auto values = Layer::AddLinear(subgraph, block.valueProjection, normalizedAttentionInput);
		const auto keys3D = Reshape3D(subgraph, keys, 1, hyperparameters.attentionHeadCountKV, headDim);
		const auto values3D = Reshape3D(subgraph, values, 1, hyperparameters.attentionHeadCountKV, headDim);

		std::vector<NodeOutput> rotatedKeyHeads;
		rotatedKeyHeads.reserve(hyperparameters.attentionHeadCountKV);
		for (std::size_t kvHeadIndex = 0; kvHeadIndex < hyperparameters.attentionHeadCountKV; ++kvHeadIndex)
		{
			const auto keyHead3D = NodeOutput{
				subgraph.AddNode(SliceNode{ keys3D, 1, kvHeadIndex, 1 },
				                 { OutputInfo{ hiddenInfo.dtype, { 1, 1, headDim } } }),
				0,
			};
			const auto keyHead2D = Reshape2D(subgraph, keyHead3D, 1, headDim);
			const auto rotatedKey = AddLLaMARoPEAtPositions(subgraph, keyHead2D, currentPosition, hyperparameters);
			rotatedKeyHeads.push_back(Reshape3D(subgraph, rotatedKey, 1, 1, headDim));
		}
		NodeOutput rotatedKeys3D = rotatedKeyHeads.front();
		if (rotatedKeyHeads.size() > 1)
		{
			rotatedKeys3D = { subgraph.AddNode(ConcatNode{ rotatedKeyHeads, 1 },
				                               { OutputInfo{ hiddenInfo.dtype,
				                                             { 1, hyperparameters.attentionHeadCountKV, headDim } } }),
				              0 };
		}
		const Layer::KVCachePair updatedCache{
			Layer::AddScatter(subgraph, cache.keys, currentPosition, rotatedKeys3D, 0),
			Layer::AddScatter(subgraph, cache.values, currentPosition, values3D, 0),
		};
		const auto activeMask = AddActiveCacheMask(subgraph, currentPosition, maxCacheLength, hiddenInfo.dtype);

		std::vector<NodeOutput> headContexts;
		headContexts.reserve(hyperparameters.attentionHeadCount);
		const auto queryGroupsPerKVHead = hyperparameters.QueryGroupsPerKVHead();
		for (std::size_t headIndex = 0; headIndex < hyperparameters.attentionHeadCount; ++headIndex)
		{
			const auto kvHeadIndex = headIndex / queryGroupsPerKVHead;
			const auto queryHead = NodeOutput{
				subgraph.AddNode(SliceNode{ queries, 1, headIndex * headDim, headDim },
				                 { OutputInfo{ hiddenInfo.dtype, { 1, headDim } } }),
				0,
			};
			const auto keyHead3D = NodeOutput{
				subgraph.AddNode(SliceNode{ updatedCache.keys, 1, kvHeadIndex, 1 },
				                 { OutputInfo{ hiddenInfo.dtype, { maxCacheLength, 1, headDim } } }),
				0,
			};
			const auto valueHead3D = NodeOutput{
				subgraph.AddNode(SliceNode{ updatedCache.values, 1, kvHeadIndex, 1 },
				                 { OutputInfo{ hiddenInfo.dtype, { maxCacheLength, 1, headDim } } }),
				0,
			};
			headContexts.push_back(AddSingleHeadAttentionAtPosition(
			    subgraph, queryHead, Reshape2D(subgraph, keyHead3D, maxCacheLength, headDim),
			    Reshape2D(subgraph, valueHead3D, maxCacheLength, headDim), currentPosition, activeMask,
			    hyperparameters));
		}

		NodeOutput mergedContext = headContexts.front();
		if (headContexts.size() > 1)
		{
			mergedContext = { subgraph.AddNode(
				                  ConcatNode{ headContexts, 1 },
				                  { OutputInfo{ hiddenInfo.dtype, { 1, hyperparameters.embeddingLength } } }),
				              0 };
		}
		const auto attentionOutput = Layer::AddLinear(subgraph, block.outputProjection, mergedContext);
		const auto attentionResidual = NodeOutput{
			subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, hiddenState, attentionOutput }, { hiddenInfo }),
			0,
		};
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
	                                  const LLaMAHyperparameters& hyperparameters, std::size_t sequenceLength,
	                                  std::size_t positionOffset)
	{
		Subgraph subgraph;
		const auto hiddenState =
		    subgraph.AddParam(block.attentionNorm.dtype, { sequenceLength, hyperparameters.embeddingLength });
		const auto result = AddLLaMADecoderBlock(subgraph, block, hyperparameters, { hiddenState, 0 }, positionOffset);
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}

	LLaMACausalLM CreateLLaMACausalLM(Graph& graph, const Graph& archive, const LLaMAHyperparameters& hyperparameters,
	                                  const LLaMALoweringOptions& options)
	{
		const auto tokenEmbeddingVariable =
		    ImportNamedVariable(graph, archive, "token_embd.weight", options.preserveQuantizedWeights);
		const auto& tokenEmbedding = *graph.GetVariable(tokenEmbeddingVariable);
		std::vector<std::size_t> tokenEmbeddingShape;
		std::optional<QuantizationParams> tokenEmbeddingQuantization;
		std::vector<std::size_t> tokenEmbeddingStorageShape;
		DataType tokenEmbeddingType{};
		if (tokenEmbedding.IsQuantized())
		{
			tokenEmbeddingQuantization = *tokenEmbedding.Quantization();
			tokenEmbeddingShape = tokenEmbeddingQuantization->expressedShape;
			tokenEmbeddingStorageShape = tokenEmbedding.Data().Shape().ToOwned();
			tokenEmbeddingType = tokenEmbeddingQuantization->expressedType;
		}
		else
		{
			const auto& plain = RequirePlainFloatingVariable(graph, tokenEmbeddingVariable, "token_embd.weight");
			tokenEmbeddingShape = plain.Data().Shape().ToOwned();
			tokenEmbeddingType = plain.Data().DType();
		}
		if (tokenEmbeddingShape.size() != 2)
		{
			throw std::runtime_error("GGUF tensor 'token_embd.weight' must be 2D for current LLaMA lowering");
		}
		const auto vocabMajor = tokenEmbeddingShape[1] == hyperparameters.embeddingLength;
		const auto featureMajor = tokenEmbeddingShape[0] == hyperparameters.embeddingLength;
		if (!vocabMajor && !featureMajor)
		{
			throw std::runtime_error(std::format("GGUF tensor 'token_embd.weight' must have LiteNN shape [vocab, {}] "
			                                     "or legacy shape [{}, vocab] for current LLaMA lowering",
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
		model.dtype = tokenEmbeddingType;
		model.tokenEmbeddingQuantization = tokenEmbeddingQuantization;
		model.tokenEmbeddingStorageShape = std::move(tokenEmbeddingStorageShape);
		model.blocks.reserve(hyperparameters.blockCount);
		for (std::size_t blockIndex = 0; blockIndex < hyperparameters.blockCount; ++blockIndex)
		{
			model.blocks.push_back(CreateLLaMADecoderBlock(graph, archive, hyperparameters, blockIndex, options));
		}
		model.outputNorm = MakeRMSNormFromArchive(graph, archive, "output_norm.weight", hyperparameters.embeddingLength,
		                                          hyperparameters.rmsNormEpsilon);

		if (archive.FindVariable("output.weight"))
		{
			model.lmHead = MakeLinearFromArchive(graph, archive, "output.weight", hyperparameters.embeddingLength,
			                                     vocabSize, options.preserveQuantizedWeights);
		}
		else
		{
			model.lmHead = {
				.weightVariable = tokenEmbeddingVariable,
				.biasVariable = std::nullopt,
				.inFeatures = hyperparameters.embeddingLength,
				.outFeatures = vocabSize,
				.dtype = model.dtype,
				.weightQuantization = model.tokenEmbeddingQuantization,
				.weightStorageShape = model.tokenEmbeddingStorageShape,
				.transposeWeight = model.tokenEmbeddingIsVocabMajor,
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

		const std::vector<std::size_t> tokenEmbeddingShape =
		    model.tokenEmbeddingIsVocabMajor
		        ? std::vector<std::size_t>{ model.vocabSize, model.outputNorm.featureSize }
		        : std::vector<std::size_t>{ model.outputNorm.featureSize, model.vocabSize };
		NodeOutput tokenEmbedding;
		if (model.tokenEmbeddingQuantization)
		{
			const auto& params = *model.tokenEmbeddingQuantization;
			if (model.tokenEmbeddingStorageShape.empty() || params.expressedShape != tokenEmbeddingShape)
			{
				throw std::runtime_error("Quantized LLaMA token embedding metadata is inconsistent");
			}
			if (!model.tokenEmbeddingIsVocabMajor)
			{
				throw std::runtime_error("Quantized LLaMA token embedding currently requires vocab-major storage");
			}
			const auto storage =
			    subgraph.AddNode(VariableRefNode{ model.tokenEmbeddingVariable },
			                     { OutputInfo{ params.storageType, model.tokenEmbeddingStorageShape } });
			const auto rows =
			    subgraph.AddNode(QuantizedGetRowsNode{ { storage, 0 }, tokenIds, params },
			                     { OutputInfo{ model.dtype, { info.shape[0], model.outputNorm.featureSize } } });
			return { rows, 0 };
		}
		else
		{
			const auto plain = subgraph.AddNode(VariableRefNode{ model.tokenEmbeddingVariable },
			                                    { OutputInfo{ model.dtype, tokenEmbeddingShape } });
			tokenEmbedding = { plain, 0 };
		}
		const auto tokenEmbeddingRows =
		    model.tokenEmbeddingIsVocabMajor ? tokenEmbedding : AddTranspose(subgraph, tokenEmbedding);
		const auto hiddenState =
		    subgraph.AddNode(GetRowsNode{ tokenEmbeddingRows, tokenIds },
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
	                                         std::span<const Layer::KVCachePair> pastCaches, std::size_t positionOffset)
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

	LLaMADecodeResult AddLLaMACausalLMDecodeCapacity(Subgraph& subgraph, const LLaMACausalLM& model,
	                                                 const LLaMAHyperparameters& hyperparameters, NodeOutput tokenIds,
	                                                 NodeOutput currentPosition,
	                                                 std::span<const Layer::KVCachePair> caches,
	                                                 std::size_t maxCacheLength)
	{
		if (caches.size() != model.blocks.size())
		{
			throw std::runtime_error("Capacity decode requires one full KV cache pair per decoder block");
		}
		auto hiddenState = AddLLaMATokenEmbedding(subgraph, model, tokenIds);
		std::vector<Layer::KVCachePair> updatedCaches;
		updatedCaches.reserve(model.blocks.size());
		for (std::size_t blockIndex = 0; blockIndex < model.blocks.size(); ++blockIndex)
		{
			auto blockResult =
			    AddLLaMADecoderBlockDecodeCapacity(subgraph, model.blocks[blockIndex], hyperparameters, hiddenState,
			                                       caches[blockIndex], currentPosition, maxCacheLength);
			hiddenState = blockResult.hiddenState;
			updatedCaches.push_back(blockResult.updatedCache);
		}
		const auto normalized = Layer::AddRMSNorm(subgraph, model.outputNorm, hiddenState);
		return {
			.hiddenState = Layer::AddLinear(subgraph, model.lmHead, normalized),
			.updatedCaches = std::move(updatedCaches),
		};
	}

	SubgraphId BuildLLaMACausalLM(Graph& graph, const LLaMACausalLM& model, const LLaMAHyperparameters& hyperparameters,
	                              std::size_t sequenceLength, std::size_t positionOffset)
	{
		Subgraph subgraph;
		const auto tokenIds = subgraph.AddParam(DataType::Int32, { sequenceLength });
		const auto logits = AddLLaMACausalLM(subgraph, model, hyperparameters, { tokenIds, 0 }, positionOffset);
		subgraph.SetResults({ logits });
		return graph.AddSubgraph(std::move(subgraph));
	}

	Graph LowerLLaMACausalLM(const Graph& archive, std::size_t sequenceLength, std::size_t positionOffset,
	                         const LLaMALoweringOptions& options)
	{
		auto graph = Graph{};
		graph.SetMetadata(CopyMetadata(archive));
		const auto hyperparameters = ParseLLaMAHyperparameters(archive);
		const auto model = CreateLLaMACausalLM(graph, archive, hyperparameters, options);
		const auto forward = BuildLLaMACausalLM(graph, model, hyperparameters, sequenceLength, positionOffset);
		graph.SetForward(forward);
		graph.SetInputNames({ "token_ids" });
		graph.SetOutputNames({ "logits" });
		return graph;
	}

	Graph LowerLLaMACausalLMPrefillCapacity(const Graph& archive, std::size_t maxSequenceLength,
	                                        const LLaMALoweringOptions& options)
	{
		if (maxSequenceLength == 0)
		{
			throw std::runtime_error("LLaMA capacity prefill requires maxSequenceLength > 0");
		}
		auto graph = Graph{};
		graph.SetMetadata(CopyMetadata(archive));
		const auto hyperparameters = ParseLLaMAHyperparameters(archive);
		const auto model = CreateLLaMACausalLM(graph, archive, hyperparameters, options);

		Subgraph subgraph;
		const auto tokenIds = subgraph.AddParam(DataType::Int32, { maxSequenceLength });
		const auto logits = AddLLaMACausalLM(subgraph, model, hyperparameters, { tokenIds, 0 }, 0);
		subgraph.SetResults({ logits });
		const auto forward = graph.AddSubgraph(std::move(subgraph));
		graph.SetForward(forward);
		graph.SetInputNames({ "token_ids" });
		graph.SetOutputNames({ "logits" });
		return graph;
	}

	Graph LowerLLaMACausalLMDecode(const Graph& archive, std::size_t sequenceLength, std::size_t pastLength,
	                               std::size_t positionOffset, const LLaMALoweringOptions& options)
	{
		auto graph = Graph{};
		graph.SetMetadata(CopyMetadata(archive));
		const auto hyperparameters = ParseLLaMAHyperparameters(archive);
		if (positionOffset != pastLength)
		{
			throw std::runtime_error("Current LLaMA decode lowering requires positionOffset == pastLength");
		}

		const auto model = CreateLLaMACausalLM(graph, archive, hyperparameters, options);
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

		const auto result =
		    AddLLaMACausalLMDecode(subgraph, model, hyperparameters, { tokenIds, 0 }, pastCaches, positionOffset);
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

	Graph LowerLLaMACausalLMDecodeCapacity(const Graph& archive, std::size_t maxCacheLength,
	                                       const LLaMALoweringOptions& options)
	{
		if (maxCacheLength == 0)
		{
			throw std::runtime_error("Capacity LLaMA decode requires maxCacheLength > 0");
		}
		auto graph = Graph{};
		graph.SetMetadata(CopyMetadata(archive));
		const auto hyperparameters = ParseLLaMAHyperparameters(archive);
		const auto model = CreateLLaMACausalLM(graph, archive, hyperparameters, options);
		const auto headDim = hyperparameters.HeadDimension();
		const std::vector<std::size_t> cacheShape{ maxCacheLength, hyperparameters.attentionHeadCountKV, headDim };

		std::vector<SubgraphId> blockSubgraphs;
		blockSubgraphs.reserve(model.blocks.size());
		for (const auto& block : model.blocks)
		{
			Subgraph blockSubgraph;
			const auto hiddenState = blockSubgraph.AddParam(model.dtype, { 1, hyperparameters.embeddingLength });
			const auto cacheKeys = blockSubgraph.AddParam(model.dtype, cacheShape);
			const auto cacheValues = blockSubgraph.AddParam(model.dtype, cacheShape);
			const auto currentPosition = blockSubgraph.AddParam(DataType::Int64, { 1 });
			const auto result = AddLLaMADecoderBlockDecodeCapacity(
			    blockSubgraph, block, hyperparameters, { hiddenState, 0 },
			    Layer::KVCachePair{ { cacheKeys, 0 }, { cacheValues, 0 } }, { currentPosition, 0 }, maxCacheLength);
			blockSubgraph.SetResults(
			    std::vector<NodeOutput>{ result.hiddenState, result.updatedCache.keys, result.updatedCache.values });
			blockSubgraphs.push_back(graph.AddSubgraph(std::move(blockSubgraph)));
		}

		Subgraph subgraph;
		const auto tokenIds = subgraph.AddParam(DataType::Int32, { 1 });
		const auto currentPosition = subgraph.AddParam(DataType::Int64, { 1 });
		std::vector<Layer::KVCachePair> caches;
		caches.reserve(model.blocks.size());
		std::vector<std::string> inputNames{ "token_ids", "current_position" };
		for (std::size_t blockIndex = 0; blockIndex < model.blocks.size(); ++blockIndex)
		{
			const auto keys = subgraph.AddParam(model.dtype, cacheShape);
			const auto values = subgraph.AddParam(model.dtype, cacheShape);
			caches.push_back(Layer::KVCachePair{ { keys, 0 }, { values, 0 } });
			inputNames.push_back(std::format("past_key_{}", blockIndex));
			inputNames.push_back(std::format("past_value_{}", blockIndex));
		}

		auto hiddenState = AddLLaMATokenEmbedding(subgraph, model, { tokenIds, 0 });
		std::vector<Layer::KVCachePair> updatedCaches;
		updatedCaches.reserve(model.blocks.size());
		for (std::size_t blockIndex = 0; blockIndex < model.blocks.size(); ++blockIndex)
		{
			std::vector<NodeOutput> args{
				hiddenState,
				caches[blockIndex].keys,
				caches[blockIndex].values,
				{ currentPosition, 0 },
			};
			const auto call =
			    subgraph.AddNode(CallNode{ blockSubgraphs[blockIndex], std::move(args) },
			                     { OutputInfo{ model.dtype, { 1, hyperparameters.embeddingLength } },
			                       OutputInfo{ model.dtype, cacheShape }, OutputInfo{ model.dtype, cacheShape } });
			hiddenState = NodeOutput{ call, 0 };
			updatedCaches.push_back(Layer::KVCachePair{ { call, 1 }, { call, 2 } });
		}
		const auto normalized = Layer::AddRMSNorm(subgraph, model.outputNorm, hiddenState);
		const LLaMADecodeResult result{
			.hiddenState = Layer::AddLinear(subgraph, model.lmHead, normalized),
			.updatedCaches = std::move(updatedCaches),
		};
		const std::array<double, 1> one{ 1.0 };
		const auto oneValue =
		    Layer::Detail::AddConstant(subgraph, Tensor<CPU>(std::span<const double>(one), { 1 }, DataType::Int64));
		const auto nextPosition =
		    subgraph.AddNode(BinaryOpNode{ BinaryOp::Add, { currentPosition, 0 }, { oneValue, 0 } },
		                     { OutputInfo{ DataType::Int64, { 1 } } });
		std::vector<NodeOutput> outputs{ result.hiddenState, NodeOutput{ nextPosition, 0 } };
		std::vector<std::string> outputNames{ "logits", "next_position" };
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

	Runtime::RuntimeSchedule BuildLLaMADecodeRuntimeSchedule(const Graph& archive,
	                                                         const LLaMAArtifactPlanningOptions& options)
	{
		const auto artifacts = PlanLLaMAArtifacts(archive, options);
		auto graph =
		    options.dynamicDecodePosition
		        ? LowerLLaMACausalLMDecodeCapacity(archive, artifacts.decodeStep.maxCacheLength,
		                                           { .preserveQuantizedWeights = options.preserveQuantizedWeights })
		        : LowerLLaMACausalLMDecode(archive, 1, options.decodePastLength, options.decodePastLength,
		                                   { .preserveQuantizedWeights = options.preserveQuantizedWeights });
		auto module = Detail::BuildExecutableModuleFromGraph(graph);
		auto states = artifacts.decodeStateABI.kvCaches;
		if (artifacts.decodeStateABI.currentPosition)
		{
			states.push_back(*artifacts.decodeStateABI.currentPosition);
		}
		auto stateValueBindings = artifacts.decodeStep.stateValueBindings;
		for (auto& binding : stateValueBindings)
		{
			binding.function = module.plan.forward;
		}
		return Runtime::BuildRuntimeSchedule(std::move(module), std::move(states), std::move(stateValueBindings));
	}

} // namespace LiteNN::GGUF
