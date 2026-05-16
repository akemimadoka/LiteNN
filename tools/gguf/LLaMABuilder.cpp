#include "LLaMABuilder.h"

#include <LiteNN/Layer/LayerUtils.h>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace LiteNN::GGUF
{
	namespace
	{
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
			const auto targetIndex = target.AddVariable(archive.GetVariable(*sourceIndex));
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
			const auto variableIndex = ImportNamedVariable(target, archive, name);
			const auto& variable = RequirePlainFloatingVariable(target, variableIndex, name);
			if (variable.Data().Shape().NumDim() != 2 || variable.Data().Shape()[0] != inFeatures ||
			    variable.Data().Shape()[1] != outFeatures)
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' must have shape [{}, {}] for current LLaMA block lowering", name, inFeatures,
				    outFeatures));
			}

			return {
				.weightVariable = variableIndex,
				.biasVariable = std::nullopt,
				.inFeatures = inFeatures,
				.outFeatures = outFeatures,
				.dtype = variable.Data().DType(),
			};
		}

		Layer::RMSNormLayer MakeRMSNormFromArchive(Graph& target, const Graph& archive, std::string_view name,
		                                         std::size_t featureSize, double eps)
		{
			const auto variableIndex = ImportNamedVariable(target, archive, name);
			const auto& variable = RequirePlainFloatingVariable(target, variableIndex, name);
			if (variable.Data().Shape().NumDim() != 2 || variable.Data().Shape()[0] != 1 ||
			    variable.Data().Shape()[1] != featureSize)
			{
				throw std::runtime_error(std::format(
				    "GGUF tensor '{}' must have shape [1, {}] for current LLaMA block lowering", name, featureSize));
			}

			return {
				.weightVariable = variableIndex,
				.featureSize = featureSize,
				.dtype = variable.Data().DType(),
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

		NodeOutput AddScale(Subgraph& subgraph, NodeOutput input, double scale)
		{
			const auto info = subgraph.GetOutputInfo(input);
			const auto scaleTensor = Tensor<CPU>({ scale }, { 1 }, info.dtype);
			const auto constant =
			    Layer::Detail::AddConstant(subgraph, scaleTensor);
			return { subgraph.AddNode(BinaryOpNode{ BinaryOp::Multiply, input, constant }, { info }), 0 };
		}

		NodeOutput AddSingleHeadAttention(Subgraph& subgraph, NodeOutput queries, NodeOutput keys, NodeOutput values,
		                                 double ropeFrequencyBase)
		{
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

			const auto rotatedQueries = Layer::AddRoPE(subgraph, queries, ropeFrequencyBase);
			const auto rotatedKeys = Layer::AddRoPE(subgraph, keys, ropeFrequencyBase);
			const auto transposedKeys = AddTranspose(subgraph, rotatedKeys);
			const auto scores = subgraph.AddNode(
			    BinaryOpNode{ BinaryOp::MatMul, rotatedQueries, transposedKeys },
			    { OutputInfo{ queryInfo.dtype, { queryInfo.shape[0], queryInfo.shape[0] } } });
			const auto scaledScores = AddScale(subgraph, { scores, 0 }, 1.0 / std::sqrt(static_cast<double>(queryInfo.shape[1])));
			const auto maskedScores = Layer::AddCausalMask(subgraph, scaledScores);
			const auto probabilities = Layer::AddSoftmax(subgraph, maskedScores, 1);
			return { subgraph.AddNode(BinaryOpNode{ BinaryOp::MatMul, probabilities, values },
			                         { OutputInfo{ valueInfo.dtype, valueInfo.shape } }),
			         0 };
		}

		std::vector<ModelMetadataEntry> CopyMetadata(const Graph& graph)
		{
			return { graph.Metadata().begin(), graph.Metadata().end() };
		}
	} // namespace

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
	                               const LLaMAHyperparameters& hyperparameters, NodeOutput hiddenState)
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
			    AddSingleHeadAttention(subgraph, queryHead, keyHead, valueHead, hyperparameters.ropeFrequencyBase));
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

	SubgraphId BuildLLaMADecoderBlock(Graph& graph, const LLaMADecoderBlock& block,
	                                const LLaMAHyperparameters& hyperparameters,
	                                std::size_t sequenceLength)
	{
		Subgraph subgraph;
		const auto hiddenState = subgraph.AddParam(block.attentionNorm.dtype,
		                                          { sequenceLength, hyperparameters.embeddingLength });
		const auto result = AddLLaMADecoderBlock(subgraph, block, hyperparameters, { hiddenState, 0 });
		subgraph.SetResults({ result });
		return graph.AddSubgraph(std::move(subgraph));
	}

	LLaMACausalLM CreateLLaMACausalLM(Graph& graph, const Graph& archive,
	                                 const LLaMAHyperparameters& hyperparameters)
	{
		const auto tokenEmbeddingVariable = ImportNamedVariable(graph, archive, "token_embd.weight");
		const auto& tokenEmbedding = RequirePlainFloatingVariable(graph, tokenEmbeddingVariable, "token_embd.weight");
		if (tokenEmbedding.Data().Shape().NumDim() != 2 || tokenEmbedding.Data().Shape()[0] != hyperparameters.embeddingLength)
		{
			throw std::runtime_error(std::format(
			    "GGUF tensor 'token_embd.weight' must have shape [{}, vocab] for current LLaMA lowering",
			    hyperparameters.embeddingLength));
		}
		const auto vocabSize = tokenEmbedding.Data().Shape()[1];
		if (vocabSize == 0)
		{
			throw std::runtime_error("GGUF tensor 'token_embd.weight' must have a non-zero vocabulary dimension");
		}

		LLaMACausalLM model;
		model.tokenEmbeddingVariable = tokenEmbeddingVariable;
		model.vocabSize = vocabSize;
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

	NodeOutput AddLLaMATokenEmbedding(Subgraph& subgraph, const LLaMACausalLM& model, NodeOutput tokenPlane)
	{
		const auto info = subgraph.GetOutputInfo(tokenPlane);
		if (info.dtype != model.dtype || info.shape.size() != 2 || info.shape[1] != model.vocabSize)
		{
			throw std::runtime_error(std::format(
			    "LLaMA token plane input must be 2D [sequence, {}] with matching dtype", model.vocabSize));
		}

		const std::vector<std::size_t> tokenEmbeddingShape{ model.outputNorm.featureSize, model.vocabSize };
		const auto tokenEmbedding = subgraph.AddNode(VariableRefNode{ model.tokenEmbeddingVariable },
		                                           { OutputInfo{ model.dtype, tokenEmbeddingShape } });
		const auto tokenEmbeddingT = AddTranspose(subgraph, { tokenEmbedding, 0 });
		const auto hiddenState = subgraph.AddNode(BinaryOpNode{ BinaryOp::MatMul, tokenPlane, tokenEmbeddingT },
		                                        { OutputInfo{ model.dtype, { info.shape[0], model.outputNorm.featureSize } } });
		return { hiddenState, 0 };
	}

	NodeOutput AddLLaMACausalLM(Subgraph& subgraph, const LLaMACausalLM& model,
	                           const LLaMAHyperparameters& hyperparameters, NodeOutput tokenPlane)
	{
		auto hiddenState = AddLLaMATokenEmbedding(subgraph, model, tokenPlane);
		for (const auto& block : model.blocks)
		{
			hiddenState = AddLLaMADecoderBlock(subgraph, block, hyperparameters, hiddenState);
		}
		const auto normalized = Layer::AddRMSNorm(subgraph, model.outputNorm, hiddenState);
		return Layer::AddLinear(subgraph, model.lmHead, normalized);
	}

	SubgraphId BuildLLaMACausalLM(Graph& graph, const LLaMACausalLM& model,
	                            const LLaMAHyperparameters& hyperparameters,
	                            std::size_t sequenceLength)
	{
		Subgraph subgraph;
		const auto tokenPlane = subgraph.AddParam(model.dtype, { sequenceLength, model.vocabSize });
		const auto logits = AddLLaMACausalLM(subgraph, model, hyperparameters, { tokenPlane, 0 });
		subgraph.SetResults({ logits });
		return graph.AddSubgraph(std::move(subgraph));
	}

	Graph LowerLLaMACausalLM(const Graph& archive, std::size_t sequenceLength)
	{
		auto graph = Graph{};
		graph.SetMetadata(CopyMetadata(archive));
		const auto hyperparameters = ParseLLaMAHyperparameters(archive);
		const auto model = CreateLLaMACausalLM(graph, archive, hyperparameters);
		const auto forward = BuildLLaMACausalLM(graph, model, hyperparameters, sequenceLength);
		graph.SetForward(forward);
		graph.SetInputNames({ "token_plane" });
		graph.SetOutputNames({ "logits" });
		return graph;
	}
} // namespace LiteNN::GGUF