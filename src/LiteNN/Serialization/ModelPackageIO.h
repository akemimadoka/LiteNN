#ifndef LITENN_SERIALIZATION_MODELPACKAGEIO_H
#define LITENN_SERIALIZATION_MODELPACKAGEIO_H

#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/Graph.h>
#include <LiteNN/Serialization/ExternalWeights.h>
#include <LiteNN/VNextPackage.h>

#include <cstddef>
#include <filesystem>
#include <string_view>
#include <vector>

namespace LiteNN::Serialization
{
	struct VNextModelPackage
	{
		VNextPackageManifest manifest;
		ExecutablePlan plan;
		std::filesystem::path sourcePath;
	};

	struct VNextLoadedArtifactRegion
	{
		VNextArtifactRegionRef ref;
		std::vector<std::byte> bytes;
	};

	struct VNextLoadedArtifactRegions
	{
		VNextArtifactRef artifact;
		std::vector<VNextLoadedArtifactRegion> regions;

		const VNextLoadedArtifactRegion* FindRegion(std::string_view name) const;
	};

	void SaveVNextModelPackage(const ExecutableModule& module, const std::filesystem::path& path,
	                           std::vector<VNextArtifactRef> artifacts = {}, VNextPackageLayout layout = {},
	                           std::vector<VNextAdapterRef> adapters = {},
	                           std::vector<Runtime::RuntimeStateBinding> runtimeStates = {});
	void SaveVNextModelPackage(const Runtime::RuntimeSchedule& schedule, const std::filesystem::path& path,
	                           std::vector<VNextArtifactRef> artifacts = {}, VNextPackageLayout layout = {},
	                           std::vector<VNextAdapterRef> adapters = {});
	void SaveVNextModelPackageExternalWeights(const Graph& graph, const std::filesystem::path& path,
	                                          const std::filesystem::path& externalWeightsPath,
	                                          const ExternalWeightSaveOptions& externalOptions = {});

	VNextModelPackage LoadVNextModelPackage(const std::filesystem::path& path);
	VNextLoadedArtifactRegions LoadVNextArtifactRegions(const VNextModelPackage& package,
	                                                    std::string_view artifactName);
	VNextLoadedArtifactRegions LoadVNextArtifactRegions(const VNextModelPackage& package,
	                                                    const std::filesystem::path& baseDirectory,
	                                                    std::string_view artifactName);
} // namespace LiteNN::Serialization

#endif
