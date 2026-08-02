#include "DecodeAOTCache.h"

#include <atomic>
#include <chrono>
#include <random>
#include <stdexcept>
#include <system_error>

namespace LiteNN::GGUF::Tooling
{
	namespace
	{
		bool IsCompleteSharedWeightsPayload(const std::filesystem::path& weightsPath, std::size_t expectedBytes)
		{
			std::error_code ec;
			const auto size = std::filesystem::file_size(weightsPath, ec);
			return !ec && size == expectedBytes && std::filesystem::exists(weightsPath.parent_path() / "complete");
		}

		std::filesystem::path CreateUniqueStagingDirectory(const std::filesystem::path& payloadDirectory)
		{
			static std::atomic<std::uint64_t> counter;
			std::random_device random;
			for (std::uint32_t attempt = 0; attempt < 32; ++attempt)
			{
				const auto now =
				    static_cast<std::uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
				const auto nonce = now ^ (static_cast<std::uint64_t>(random()) << 32) ^ counter.fetch_add(1);
				auto staging = payloadDirectory.parent_path() /
				               std::format("{}.tmp.{:016x}", payloadDirectory.filename().string(), nonce);
				std::error_code ec;
				if (std::filesystem::create_directory(staging, ec))
				{
					return staging;
				}
				if (ec && ec != std::errc::file_exists)
				{
					throw std::runtime_error("Failed to create shared weight staging directory: " + ec.message());
				}
			}
			throw std::runtime_error("Failed to allocate a unique shared weight staging directory");
		}
	} // namespace

	SharedWeightsPublishResult PublishDecodeAOTSharedWeightsAtomically(const std::filesystem::path& weightsPath,
	                                                                   std::size_t expectedBytes,
	                                                                   const SharedWeightsStagingWriter& stagingWriter)
	{
		if (IsCompleteSharedWeightsPayload(weightsPath, expectedBytes))
		{
			return SharedWeightsPublishResult::Reused;
		}

		const auto payloadDirectory = weightsPath.parent_path();
		std::filesystem::create_directories(payloadDirectory.parent_path());
		if (std::filesystem::exists(payloadDirectory))
		{
			if (IsCompleteSharedWeightsPayload(weightsPath, expectedBytes))
			{
				return SharedWeightsPublishResult::Reused;
			}
			throw std::runtime_error("Shared weight payload directory exists but is incomplete: " +
			                         payloadDirectory.string());
		}

		const auto stagingDirectory = CreateUniqueStagingDirectory(payloadDirectory);
		try
		{
			const auto stagingWeights = stagingDirectory / weightsPath.filename();
			const auto stagingComplete = stagingDirectory / "complete";
			stagingWriter(stagingWeights, stagingComplete);
			if (!IsCompleteSharedWeightsPayload(stagingWeights, expectedBytes))
			{
				throw std::runtime_error("Shared weight staging payload is incomplete");
			}

			std::error_code publishError;
			std::filesystem::rename(stagingDirectory, payloadDirectory, publishError);
			if (!publishError)
			{
				return SharedWeightsPublishResult::Published;
			}
			if (IsCompleteSharedWeightsPayload(weightsPath, expectedBytes))
			{
				std::filesystem::remove_all(stagingDirectory);
				return SharedWeightsPublishResult::Reused;
			}
			throw std::runtime_error("Failed to publish shared weight payload: " + publishError.message());
		}
		catch (...)
		{
			std::error_code ignored;
			std::filesystem::remove_all(stagingDirectory, ignored);
			throw;
		}
	}
} // namespace LiteNN::GGUF::Tooling
