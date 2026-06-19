#ifndef LITENN_VNEXT_RULES_H
#define LITENN_VNEXT_RULES_H

#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/VNextPackage.h>

#include <stdexcept>
#include <string>
#include <vector>

namespace LiteNN
{
	enum class VNextRuleSeverity
	{
		Info,
		Warning,
		Error
	};

	struct VNextRule
	{
		std::string id;
		VNextRuleSeverity severity{ VNextRuleSeverity::Warning };
		std::string message;
	};

	inline std::vector<VNextRule> VNextRules()
	{
		return {
			{ "multi-output-core", VNextRuleSeverity::Error,
			  "multi-output nodes are a core invariant and must not be flattened by importers or backends" },
			{ "artifact-storage-abi", VNextRuleSeverity::Error,
			  "rodata, instructions, and external weights must be represented through storage/artifact ABI metadata" },
			{ "interpreter-reference", VNextRuleSeverity::Warning,
			  "Interpreter remains the reference/debug path; production execution should consume executable plans" },
			{ "graph-entrypoints", VNextRuleSeverity::Warning,
			  "runtime/compiler entry points should consume ModelGraph, ExecutablePlan, or ExecutableModule "
			  "contracts" },
			{ "schema-serialization", VNextRuleSeverity::Warning,
			  "serializer knowledge of raw NodeVariant layout is temporary until schema-driven op serialization "
			  "lands" },
			{ "backend-shortcuts", VNextRuleSeverity::Warning,
			  "CPU/CUDA shortcuts should move into capability, cost, layout, or artifact metadata" },
			{ "builder-helper-contract", VNextRuleSeverity::Warning,
			  "builder helpers should preserve TensorType, schema validation, and external storage metadata" },
		};
	}

	inline void ValidateVNextInvariants(const ExecutablePlan& plan, const VNextPackageManifest* manifest = nullptr)
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
			(void) hasMultiOutput;
		}
		if (manifest)
		{
			ValidateVNextPackageManifest(*manifest);
			for (const auto& artifact : manifest->artifacts)
			{
				if (artifact.regions.empty())
				{
					throw std::runtime_error("vNext invariant failed: artifact must expose regions");
				}
			}
		}
	}
} // namespace LiteNN

#endif
