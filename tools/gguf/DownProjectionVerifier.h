#ifndef LITENN_GGUF_DOWN_PROJECTION_VERIFIER_H
#define LITENN_GGUF_DOWN_PROJECTION_VERIFIER_H

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

namespace LiteNN::GGUF::Tooling
{
	struct DownProjectionVerificationOptions
	{
		std::filesystem::path modelPath;
		std::filesystem::path checkpointDirectory;
		std::filesystem::path outputPath;
		std::size_t generatedIndex{};
		std::vector<std::size_t> blockIndices;
		std::size_t threadCount{ 8 };
	};

	struct DownProjectionVerificationSummary
	{
		std::size_t blockCount{};
		std::string closestCandidateCounts;
		double maximumProductionVersusCapturedNRMSE{};
		double maximumProductionVersusCapturedAbsoluteError{};
	};

	DownProjectionVerificationSummary
	VerifyLLaMADownProjectionCheckpoints(const DownProjectionVerificationOptions& options);

	struct FFNActivationVerificationSummary
	{
		std::size_t blockCount{};
		double maximumProductionGateVersusCapturedNRMSE{};
		double maximumProductionUpVersusCapturedNRMSE{};
		double maximumProductionSwiGLUVersusCapturedNRMSE{};
		double maximumCapturedInputSwiGLUVersusCapturedNRMSE{};
	};

	FFNActivationVerificationSummary
	VerifyLLaMAFFNActivationCheckpoints(const DownProjectionVerificationOptions& options);
} // namespace LiteNN::GGUF::Tooling

#endif
