#include <LiteNN/DType.h>
#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/Serialization/Safetensors.h>

#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifndef LITENN_SERIALIZATION_TORCHMANIFEST_H
#define LITENN_SERIALIZATION_TORCHMANIFEST_H

namespace LiteNN::Serialization
{
	struct TorchManifestImportOptions
	{
		bool failOnUnusedWeights{ true };
		bool trainableVariables{ false };
	};

	struct TorchManifestOpMapping
	{
		std::string_view torchOp;
		std::string_view liteNNLowering;
		std::string_view notes;
	};

	struct TorchManifestReport
	{
		std::vector<std::string> importedTensors;
		std::vector<std::string> loweredOps;
		std::vector<std::string> foldedConstants;
		std::vector<std::string> unsupportedOps;
		std::vector<std::string> fallbacks;
		std::vector<std::string> diagnostics;
	};

	struct TorchManifestImportResult
	{
		ModelGraph model;
		TorchManifestReport report;
	};

	std::span<const TorchManifestOpMapping> SupportedTorchManifestOpMappings();

	std::optional<DataType> TryMapTorchManifestDataType(std::string_view dtype);
	DataType MapTorchManifestDataType(std::string_view dtype);

	TorchManifestImportResult ImportTorchManifest(std::string_view manifestJson,
	                                              const SafetensorsArchive& archive,
	                                              const TorchManifestImportOptions& options = {});
	TorchManifestImportResult LoadTorchManifest(const std::filesystem::path& manifestPath,
	                                            const std::filesystem::path& safetensorsPath,
	                                            const TorchManifestImportOptions& options = {});
} // namespace LiteNN::Serialization

#endif
