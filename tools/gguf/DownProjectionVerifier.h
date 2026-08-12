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

	struct FinalLogitVerificationOptions
	{
		std::filesystem::path modelPath;
		std::filesystem::path candidateCheckpointDirectory;
		std::filesystem::path referenceCheckpointDirectory;
		std::filesystem::path candidateLogitsPath;
		std::filesystem::path referenceLogitsPath;
		std::filesystem::path outputPath;
		std::size_t generatedIndex{};
		std::size_t threadCount{ 8 };
		std::size_t topK{ 10 };
	};

	struct FinalLogitVerificationSummary
	{
		std::size_t candidateTop1{};
		std::size_t referenceTop1{};
		double candidateMargin{};
		double referenceMargin{};
		double candidateReferenceMinusCandidateMargin{};
		double referenceReferenceMinusCandidateMargin{};
		double pairMarginShift{};
		double candidateReconstructionNRMSE{};
		double referenceReconstructionNRMSE{};
	};

	FinalLogitVerificationSummary VerifyLLaMAFinalLogits(const FinalLogitVerificationOptions& options);
} // namespace LiteNN::GGUF::Tooling

#endif
