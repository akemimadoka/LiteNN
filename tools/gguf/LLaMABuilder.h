#include "GGUFImporter.h"

#include <LiteNN/Layer/Layer.h>

#include <cstddef>
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
		DataType dtype{ DataType::Float32 };
		std::vector<LLaMADecoderBlock> blocks;
		Layer::RMSNormLayer outputNorm;
		Layer::LinearLayer lmHead;
	};

	LLaMADecoderBlock CreateLLaMADecoderBlock(Graph& graph, const Graph& archive,
	                                         const LLaMAHyperparameters& hyperparameters,
	                                         std::size_t blockIndex);
	NodeOutput AddLLaMADecoderBlock(Subgraph& subgraph, const LLaMADecoderBlock& block,
	                               const LLaMAHyperparameters& hyperparameters, NodeOutput hiddenState);
	SubgraphId BuildLLaMADecoderBlock(Graph& graph, const LLaMADecoderBlock& block,
	                                const LLaMAHyperparameters& hyperparameters,
	                                std::size_t sequenceLength);
	LLaMACausalLM CreateLLaMACausalLM(Graph& graph, const Graph& archive,
	                                 const LLaMAHyperparameters& hyperparameters);
	NodeOutput AddLLaMATokenEmbedding(Subgraph& subgraph, const LLaMACausalLM& model, NodeOutput tokenPlane);
	NodeOutput AddLLaMACausalLM(Subgraph& subgraph, const LLaMACausalLM& model,
	                           const LLaMAHyperparameters& hyperparameters, NodeOutput tokenPlane);
	SubgraphId BuildLLaMACausalLM(Graph& graph, const LLaMACausalLM& model,
	                            const LLaMAHyperparameters& hyperparameters,
	                            std::size_t sequenceLength);
	Graph LowerLLaMACausalLM(const Graph& archive, std::size_t sequenceLength);
} // namespace LiteNN::GGUF

#endif