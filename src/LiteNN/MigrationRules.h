#ifndef LITENN_MIGRATION_RULES_H
#define LITENN_MIGRATION_RULES_H

#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/VNextPackage.h>

#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace LiteNN
{
	enum class MigrationRuleSeverity
	{
		Info,
		Warning,
		Error
	};

	struct MigrationRule
	{
		std::string id;
		MigrationRuleSeverity severity{ MigrationRuleSeverity::Warning };
		std::string message;
	};

	inline std::vector<MigrationRule> VNextMigrationRules()
	{
		return {
			{ "multi-output-core", MigrationRuleSeverity::Error,
			  "multi-output nodes are a core invariant and must not be flattened by importers or backends" },
			{ "artifact-storage-abi", MigrationRuleSeverity::Error,
			  "rodata, instructions, and external weights must be represented through storage/artifact ABI metadata" },
			{ "interpreter-reference", MigrationRuleSeverity::Warning,
			  "Interpreter remains the reference/debug path; production execution should consume executable plans" },
			{ "graph-entrypoints-migration", MigrationRuleSeverity::Warning,
			  "direct Graph runtime/compiler entry points are migration conveniences and should lower to ExecutablePlan" },
			{ "schema-serialization", MigrationRuleSeverity::Warning,
			  "serializer knowledge of raw NodeVariant layout is temporary until schema-driven op serialization lands" },
			{ "backend-shortcuts", MigrationRuleSeverity::Warning,
			  "CPU/CUDA shortcuts should move into capability, cost, layout, or artifact metadata" },
			{ "builder-helper-migration", MigrationRuleSeverity::Warning,
			  "graph-builder helpers that bypass TensorType, schema validation, or external storage are migration-only" },
		};
	}

	inline void ValidateVNextMigrationInvariants(const ExecutablePlan& plan,
	                                             const VNextPackageManifest* manifest = nullptr)
	{
		ValidateExecutablePlan(plan);
		for (const auto& subgraph : plan.subgraphs)
		{
			bool hasMultiOutput = false;
			for (const auto& node : subgraph.nodes)
			{
				if (node.outputs.size() > 1)
				{
					hasMultiOutput = true;
					break;
				}
			}
			(void)hasMultiOutput;
		}
		if (manifest)
		{
			ValidateVNextPackageManifest(*manifest);
			for (const auto& artifact : manifest->artifacts)
			{
				if (artifact.regions.empty())
				{
					throw std::runtime_error("vNext migration invariant failed: artifact must expose regions");
				}
			}
		}
	}
} // namespace LiteNN

#endif
