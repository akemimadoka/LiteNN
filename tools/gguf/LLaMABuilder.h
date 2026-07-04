#include "GGUFImporter.h"

#include <LiteNN/Layer/Layer.h>
#include <LiteNN/Runtime/Scheduler.h>

#include <cstddef>
#include <optional>
#include <span>
#include <string>
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
		std::optional<QuantizationParams> tokenEmbeddingQuantization;
		std::vector<std::size_t> tokenEmbeddingStorageShape;
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

	enum class LLaMAArtifactKind
	{
		Prefill,
		DecodeStep
	};

	struct LLaMAKVCacheBinding
	{
		std::size_t blockIndex{};
		std::string pastKeyInput;
		std::string pastValueInput;
		std::string updatedKeyOutput;
		std::string updatedValueOutput;
		TensorType cacheType;
		TensorType stateType;
		Runtime::RuntimeStateBinding stateBinding;
		std::optional<Runtime::RuntimeStateBinding> pageTableStateBinding;
		std::optional<Runtime::RuntimeStateBinding> pageDescriptorStateBinding;
		std::optional<Runtime::RuntimeStateBinding> activeLengthStateBinding;
		std::size_t keyByteOffset{};
		std::size_t valueByteOffset{};
		std::size_t layerByteStride{};
		std::size_t tokenByteStride{};
	};

	struct LLaMAArtifactEntry
	{
		LLaMAArtifactKind kind{ LLaMAArtifactKind::Prefill };
		std::string name;
		std::size_t sequenceLength{};
		std::size_t pastLength{};
		std::size_t maxCacheLength{};
		std::size_t positionOffset{};
		bool dynamicPosition{};
		std::vector<std::string> inputNames;
		std::vector<std::string> outputNames;
		std::vector<LLaMAKVCacheBinding> kvCaches;
		std::vector<Runtime::RuntimeStateValueBinding> stateValueBindings;
	};

	struct LLaMATensorLayoutRecord
	{
		std::string name;
		std::string domain;
		std::vector<std::string> axes;
		std::string layout;
		std::string note;
	};

	enum class LLaMAAttentionExecutionMode
	{
		ActivePrefix,
		PagedAttention
	};

	struct LLaMAAttentionExecutionPlan
	{
		std::string name;
		LLaMAAttentionExecutionMode mode{ LLaMAAttentionExecutionMode::ActivePrefix };
		std::string backend;
		std::size_t maxContextLength{};
		std::size_t pageSizeTokens{};
		bool usesPagedKV{};
		bool requiresPageTable{};
		bool materializesFullMask{};
		bool streamingDecode{};
		std::string status;
		std::vector<std::string> requiredRuntimeStates;
	};

	struct LLaMAArtifactPlan
	{
		LLaMAHyperparameters hyperparameters;
		DataType dtype{ DataType::Float32 };
		std::size_t vocabSize{};
		LLaMAArtifactEntry prefill;
		LLaMAArtifactEntry decodeStep;
		Runtime::LLMDecodeStateABI decodeStateABI;
		std::vector<LLaMATensorLayoutRecord> tensorLayouts;
		std::vector<LLaMAAttentionExecutionPlan> attentionExecutionPlans;
	};

	struct LLaMAArtifactPlanningOptions
	{
		std::size_t prefillSequenceLength{};
		std::size_t decodePastLength{};
		std::size_t maxCacheLength{};
		bool preserveQuantizedWeights{};
		/// Build a max-capacity prefill graph; the caller tracks prompt length and selects the final row.
		bool dynamicPrefillLength{};
		/// Build a max-capacity decode graph whose current position is a runtime state value.
		bool dynamicDecodePosition{};
	};

	struct LLaMALoweringOptions
	{
		bool preserveQuantizedWeights{};
	};

	LLaMAParityTolerance GetLLaMAParityTolerance(DataType dtype,
	                                             std::optional<QuantizedBlockFormat> blockFormat = std::nullopt);
	LLaMAArtifactPlan PlanLLaMAArtifacts(const Graph& archive, const LLaMAArtifactPlanningOptions& options);
	LLaMAArtifactPlan PlanLLaMAArtifacts(const Graph& archive, std::size_t prefillSequenceLength,
	                                     std::size_t decodePastLength);
	Runtime::RuntimeSchedule BuildLLaMADecodeRuntimeSchedule(const Graph& archive,
	                                                         const LLaMAArtifactPlanningOptions& options);
	LLaMADecoderBlock CreateLLaMADecoderBlock(Graph& graph, const Graph& archive,
	                                          const LLaMAHyperparameters& hyperparameters, std::size_t blockIndex,
	                                          const LLaMALoweringOptions& options = {});
	NodeOutput AddLLaMADecoderBlock(Subgraph& subgraph, const LLaMADecoderBlock& block,
	                                const LLaMAHyperparameters& hyperparameters, NodeOutput hiddenState,
	                                std::size_t positionOffset = 0);
	SubgraphId BuildLLaMADecoderBlock(Graph& graph, const LLaMADecoderBlock& block,
	                                  const LLaMAHyperparameters& hyperparameters, std::size_t sequenceLength,
	                                  std::size_t positionOffset = 0);
	LLaMACausalLM CreateLLaMACausalLM(Graph& graph, const Graph& archive, const LLaMAHyperparameters& hyperparameters,
	                                  const LLaMALoweringOptions& options = {});
	NodeOutput AddLLaMATokenEmbedding(Subgraph& subgraph, const LLaMACausalLM& model, NodeOutput tokenIds);
	NodeOutput AddLLaMACausalLM(Subgraph& subgraph, const LLaMACausalLM& model,
	                            const LLaMAHyperparameters& hyperparameters, NodeOutput tokenIds,
	                            std::size_t positionOffset = 0);
	LLaMADecodeResult AddLLaMACausalLMDecode(Subgraph& subgraph, const LLaMACausalLM& model,
	                                         const LLaMAHyperparameters& hyperparameters, NodeOutput tokenIds,
	                                         std::span<const Layer::KVCachePair> pastCaches,
	                                         std::size_t positionOffset);
	SubgraphId BuildLLaMACausalLM(Graph& graph, const LLaMACausalLM& model, const LLaMAHyperparameters& hyperparameters,
	                              std::size_t sequenceLength, std::size_t positionOffset = 0);
	Graph LowerLLaMACausalLM(const Graph& archive, std::size_t sequenceLength, std::size_t positionOffset = 0,
	                         const LLaMALoweringOptions& options = {});
	/// Lowers a max-capacity prefill graph; callers select the final valid logits row from the returned full logits.
	Graph LowerLLaMACausalLMPrefillCapacity(const Graph& archive, std::size_t maxSequenceLength,
	                                        const LLaMALoweringOptions& options = {});
	Graph LowerLLaMACausalLMDecode(const Graph& archive, std::size_t sequenceLength, std::size_t pastLength,
	                               std::size_t positionOffset, const LLaMALoweringOptions& options = {});
	/// Lowers one-token decode with full-capacity caches and a runtime Int64 position input.
	Graph LowerLLaMACausalLMDecodeCapacity(const Graph& archive, std::size_t maxCacheLength,
	                                       const LLaMALoweringOptions& options = {});
	/// Lowers one-token logits-only decode that reads paged KV state through the CPU reference attention node.
	Graph LowerLLaMACausalLMDecodePagedReference(const Graph& archive, std::size_t maxCacheLength,
	                                             const LLaMALoweringOptions& options = {});
} // namespace LiteNN::GGUF

#endif
