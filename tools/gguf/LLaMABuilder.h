#include "GGUFImporter.h"

#include <LiteNN/Layer/Layer.h>

#include <cstddef>
#include <optional>
#include <span>
#include <vector>

#ifndef LITENN_LLAMABUILDER_H
#define LITENN_LLAMABUILDER_H

namespace LiteNN::GGUF
{
	struct LLaMADecoderBlock
	{
		Layer::RMSNormLayer attentionNorm;
		Layer::LinearLayer queryProjection;
		Layer::LinearLayer keyProjection;
		Layer::LinearLayer valueProjection;
		Layer::LinearLayer outputProjection;
		Layer::RMSNormLayer feedForwardNorm;
		Layer::SwiGLUMLPLayer mlp;
	};

	struct LLaMACausalLM
	{
		std::size_t tokenEmbeddingVariable{};
		std::size_t vocabSize{};
		bool tokenEmbeddingIsVocabMajor = true;
		DataType dtype{ DataType::Float32 };
		std::vector<LLaMADecoderBlock> blocks;
		Layer::RMSNormLayer outputNorm;
		Layer::LinearLayer lmHead;
	};

	struct LLaMADecodeResult
	{
		NodeOutput hiddenState;
		std::vector<Layer::KVCachePair> updatedCaches;
	};

	struct LLaMAParityTolerance
	{
		double absolute;
		double relative;
	};

	LLaMAParityTolerance GetLLaMAParityTolerance(DataType dtype,
	                                             std::optional<QuantizedBlockFormat> blockFormat = std::nullopt);
	LLaMADecoderBlock CreateLLaMADecoderBlock(Graph& graph, const Graph& archive,
	                                         const LLaMAHyperparameters& hyperparameters,
	                                         std::size_t blockIndex);
	NodeOutput AddLLaMADecoderBlock(Subgraph& subgraph, const LLaMADecoderBlock& block,
	                               const LLaMAHyperparameters& hyperparameters, NodeOutput hiddenState,
	                               std::size_t positionOffset = 0);
	SubgraphId BuildLLaMADecoderBlock(Graph& graph, const LLaMADecoderBlock& block,
	                                const LLaMAHyperparameters& hyperparameters,
	                                std::size_t sequenceLength, std::size_t positionOffset = 0);
	LLaMACausalLM CreateLLaMACausalLM(Graph& graph, const Graph& archive,
	                                 const LLaMAHyperparameters& hyperparameters);
	NodeOutput AddLLaMATokenEmbedding(Subgraph& subgraph, const LLaMACausalLM& model, NodeOutput tokenIds);
	NodeOutput AddLLaMACausalLM(Subgraph& subgraph, const LLaMACausalLM& model,
	                           const LLaMAHyperparameters& hyperparameters, NodeOutput tokenIds,
	                           std::size_t positionOffset = 0);
	LLaMADecodeResult AddLLaMACausalLMDecode(Subgraph& subgraph, const LLaMACausalLM& model,
	                                         const LLaMAHyperparameters& hyperparameters, NodeOutput tokenIds,
	                                         std::span<const Layer::KVCachePair> pastCaches,
	                                         std::size_t positionOffset);
	SubgraphId BuildLLaMACausalLM(Graph& graph, const LLaMACausalLM& model,
	                            const LLaMAHyperparameters& hyperparameters,
	                            std::size_t sequenceLength, std::size_t positionOffset = 0);
	Graph LowerLLaMACausalLM(const Graph& archive, std::size_t sequenceLength, std::size_t positionOffset = 0);
	Graph LowerLLaMACausalLMDecode(const Graph& archive, std::size_t sequenceLength, std::size_t pastLength,
	                               std::size_t positionOffset);
} // namespace LiteNN::GGUF

#endif
