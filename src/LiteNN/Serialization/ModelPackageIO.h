#ifndef LITENN_SERIALIZATION_MODELPACKAGEIO_H
#define LITENN_SERIALIZATION_MODELPACKAGEIO_H

#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/Graph.h>
#include <LiteNN/Serialization/ExternalWeights.h>
#include <LiteNN/VNextPackage.h>

#include <filesystem>
#include <vector>

namespace LiteNN::Serialization
{
	struct VNextModelPackage
	{
		VNextPackageManifest manifest;
		ExecutablePlan plan;
	};

	void SaveVNextModelPackage(const ExecutableModule& module, const std::filesystem::path& path,
	                           std::vector<VNextArtifactRef> artifacts = {},
	                           VNextPackageLayout layout = {}, std::vector<VNextAdapterRef> adapters = {});
	void SaveVNextModelPackageExternalWeights(const Graph& graph, const std::filesystem::path& path,
	                                          const std::filesystem::path& externalWeightsPath,
	                                          const ExternalWeightSaveOptions& externalOptions = {});

	VNextModelPackage LoadVNextModelPackage(const std::filesystem::path& path);
} // namespace LiteNN::Serialization

#endif
