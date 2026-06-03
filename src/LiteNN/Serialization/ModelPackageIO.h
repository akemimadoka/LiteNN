#ifndef LITENN_SERIALIZATION_MODELPACKAGEIO_H
#define LITENN_SERIALIZATION_MODELPACKAGEIO_H

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
	                           VNextPackageLayout layout = {});

	VNextModelPackage LoadVNextModelPackage(const std::filesystem::path& path);
} // namespace LiteNN::Serialization

#endif
