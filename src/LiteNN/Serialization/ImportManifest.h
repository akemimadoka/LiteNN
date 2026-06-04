#ifndef LITENN_SERIALIZATION_IMPORT_MANIFEST_H
#define LITENN_SERIALIZATION_IMPORT_MANIFEST_H

#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/Runtime/Placement.h>

#include <cstddef>
#include <format>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace LiteNN::Serialization
{
	enum class ImportDiagnosticKind
	{
		UnsupportedOp,
		UnsupportedDType,
		UnsupportedLayout,
		MissingMetadata,
		UnsupportedStateABI,
		UnsupportedBackendCapability,
		CompatibilityOp
	};

	struct ImportDiagnostic
	{
		ImportDiagnosticKind kind{ ImportDiagnosticKind::MissingMetadata };
		std::string subject;
		std::string message;
	};

	struct ImportedWeightMapping
	{
		std::string sourceName;
		std::string graphName;
		TensorType sourceType;
		TensorType graphType;
		std::string layoutConversion;
		std::string quantizationMapping;
		std::string loraBinding;
	};

	struct ImporterOwnedManifest
	{
		std::string sourceFormat;
		ModelGraph model;
		std::vector<ImportedWeightMapping> weights;
		std::vector<ModelMetadataEntry> tokenizerMetadata;
		std::vector<ModelMetadataEntry> configMetadata;
		std::vector<ImportDiagnostic> diagnostics;
		std::vector<std::string> moduleNames;
	};

	inline ImportDiagnostic MakeImportDiagnostic(ImportDiagnosticKind kind, std::string subject, std::string message)
	{
		return { .kind = kind, .subject = std::move(subject), .message = std::move(message) };
	}

	inline ImporterOwnedManifest BuildImporterOwnedManifest(std::string sourceFormat, Graph graph)
	{
		ImporterOwnedManifest manifest;
		manifest.sourceFormat = std::move(sourceFormat);
		manifest.model = ModelGraph(std::move(graph));
		return manifest;
	}

	inline void AddImportBackendDiagnostics(ImporterOwnedManifest& manifest,
	                                        std::span<const std::string_view> backends =
	                                            std::span<const std::string_view>{ DefaultBackendNames },
	                                        const OpSchemaRegistry& registry = DefaultOpSchemaRegistry())
	{
		const auto plan = BuildExecutablePlan(manifest.model, registry);
		for (const auto backend : backends)
		{
			for (const auto& issue : CollectExecutablePlanBackendIssues(plan, backend, registry, false))
			{
				manifest.diagnostics.push_back(MakeImportDiagnostic(
				    ImportDiagnosticKind::UnsupportedBackendCapability,
				    std::format("{}:{}:{}", issue.subgraph, issue.node, issue.opKind),
				    std::format("backend '{}' cannot lower op '{}'", backend, issue.opKind)));
			}
		}
	}

	inline void AddImportCompatibilityDiagnostics(ImporterOwnedManifest& manifest,
	                                              const OpSchemaRegistry& registry = DefaultOpSchemaRegistry())
	{
		const auto plan = BuildExecutablePlan(manifest.model, registry);
		for (const auto& diagnostic : CollectExecutablePlanCompatibilityDiagnostics(plan, registry))
		{
			manifest.diagnostics.push_back(MakeImportDiagnostic(
			    ImportDiagnosticKind::CompatibilityOp,
			    std::format("{}:{}:{}", diagnostic.subgraph, diagnostic.node, diagnostic.opKind),
			    diagnostic.message));
		}
	}

	inline void ValidateImporterOwnedManifest(const ImporterOwnedManifest& manifest)
	{
		if (manifest.sourceFormat.empty())
		{
			throw std::runtime_error("Importer manifest source format cannot be empty");
		}
		ValidateExecutablePlan(BuildExecutablePlan(manifest.model));
		for (const auto& weight : manifest.weights)
		{
			if (weight.sourceName.empty() || weight.graphName.empty())
			{
				throw std::runtime_error("Importer manifest weight mappings require source and graph names");
			}
			ValidateExecutableTensorType(weight.sourceType, "import source weight");
			ValidateExecutableTensorType(weight.graphType, "import graph weight");
		}
		for (const auto& diagnostic : manifest.diagnostics)
		{
			if (diagnostic.subject.empty() || diagnostic.message.empty())
			{
				throw std::runtime_error("Importer manifest diagnostics require subject and message");
			}
		}
	}
} // namespace LiteNN::Serialization

#endif
