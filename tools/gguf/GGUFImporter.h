#include <LiteNN.h>

#include <cstddef>
#include <filesystem>
#include <optional>

#ifndef LITENN_GGUFIMPORTER_H
#define LITENN_GGUFIMPORTER_H

namespace LiteNN::GGUF
{
	struct LLaMAHyperparameters
	{
		std::string architecture;
		std::size_t contextLength{};
		std::size_t embeddingLength{};
		std::size_t blockCount{};
		std::size_t feedForwardLength{};
		std::size_t attentionHeadCount{};
		std::size_t attentionHeadCountKV{};
		double rmsNormEpsilon{};
		double ropeFrequencyBase{ 10000.0 };
		double ropeFrequencyScale{ 1.0 };
		std::size_t ropeDimensionCount{};
		std::string ropeScalingType{ "none" };
		std::optional<double> ropeScalingFactor;
		std::optional<double> ropeScalingAlpha;
		std::optional<double> ropeScalingAttentionFactor;
		std::optional<std::size_t> ropeScalingOriginalContextLength;
		std::optional<bool> ropeScalingFinetuned;
		std::optional<double> ropeScalingYarnLogMultiplier;
		std::optional<double> ropeScalingYarnExtFactor;
		std::optional<double> ropeScalingYarnAttentionFactor;
		std::optional<double> ropeScalingYarnBetaFast;
		std::optional<double> ropeScalingYarnBetaSlow;

		std::size_t HeadDimension() const;
		std::size_t QueryGroupsPerKVHead() const;
	};

	struct ImportSummary
	{
		std::size_t tensorCount{};
		std::size_t metadataCount{};
	};

	struct ImportResult
	{
		Graph graph;
		ImportSummary summary;
	};

	LLaMAHyperparameters ParseLLaMAHyperparameters(const Graph& graph);
	ImportResult ImportGGUFArchive(const std::filesystem::path& inputPath);
	ImportSummary ConvertGGUFArchive(const std::filesystem::path& inputPath,
	                                const std::filesystem::path& outputPath);
} // namespace LiteNN::GGUF

#endif
