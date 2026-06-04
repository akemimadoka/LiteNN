#include <LiteNN/Serialization/ModelIO.h>
#include <LiteNN/Serialization/Safetensors.h>
#include <LiteNN/Serialization/TorchManifest.h>

#include <exception>
#include <filesystem>
#include <iostream>
#include <map>
#include <set>
#include <string>
#include <string_view>

namespace
{
	void PrintUsage(const char* argv0)
	{
		std::cerr << "Usage:\n"
		          << "  " << argv0 << " <input.safetensors> <output.ltnn> [--rename from=to] [--transpose name]\n\n"
		          << "  " << argv0 << " --manifest <manifest.json> <input.safetensors> <output.ltnn>"
		          << " [--allow-extra-tensors]\n\n"
		          << "Options:\n"
		          << "  --rename from=to   Rename an imported tensor variable. May be repeated.\n"
		          << "  --transpose name   Transpose a rank-2 tensor while importing. May be repeated.\n"
		          << "  --manifest path    Import a LiteNN Torch manifest plus safetensors weights.\n"
		          << "  --allow-extra-tensors\n"
		          << "                     Allow safetensors weights not referenced by the manifest.\n";
	}

	std::pair<std::string, std::string> ParseRename(std::string_view value)
	{
		const auto pos = value.find('=');
		if (pos == std::string_view::npos || pos == 0 || pos + 1 == value.size())
		{
			throw std::runtime_error("--rename expects from=to");
		}
		return { std::string(value.substr(0, pos)), std::string(value.substr(pos + 1)) };
	}

	void PrintReport(const LiteNN::Serialization::TorchManifestReport& report)
	{
		const auto printGroup = [](std::string_view title, const std::vector<std::string>& entries) {
			if (entries.empty())
			{
				return;
			}
			std::cout << title << ":\n";
			for (const auto& entry : entries)
			{
				std::cout << "  - " << entry << '\n';
			}
		};
		printGroup("Imported tensors", report.importedTensors);
		printGroup("Lowered ops", report.loweredOps);
		printGroup("Folded constants", report.foldedConstants);
		printGroup("Fallbacks", report.fallbacks);
		printGroup("Diagnostics", report.diagnostics);
	}
} // namespace

int main(int argc, char** argv)
{
	try
	{
		if (argc == 2 && (std::string_view(argv[1]) == "--help" || std::string_view(argv[1]) == "-h"))
		{
			PrintUsage(argv[0]);
			return 0;
		}
		if (argc < 3)
		{
			PrintUsage(argv[0]);
			return 1;
		}

		if (std::string_view(argv[1]) == "--manifest")
		{
			if (argc < 5)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			std::filesystem::path manifestPath = argv[2];
			std::filesystem::path inputPath = argv[3];
			std::filesystem::path outputPath = argv[4];
			LiteNN::Serialization::TorchManifestImportOptions options;
			for (int i = 5; i < argc; ++i)
			{
				const std::string_view arg = argv[i];
				if (arg == "--allow-extra-tensors")
				{
					options.failOnUnusedWeights = false;
				}
				else if (arg == "--help" || arg == "-h")
				{
					PrintUsage(argv[0]);
					return 0;
				}
				else
				{
					throw std::runtime_error("Unknown manifest argument: " + std::string(arg));
				}
			}

			auto result = LiteNN::Serialization::LoadTorchManifest(manifestPath, inputPath, options);
			LiteNN::Serialization::Migration::SaveGraphArchive(result.graph, outputPath);
			std::cout << "Imported Torch manifest graph with " << result.graph.VariableCount() << " variable(s), "
			          << result.graph.GetSubgraph(result.graph.Forward()).NodeCount() << " node(s) into "
			          << outputPath.string() << '\n';
			PrintReport(result.report);
			return 0;
		}

		std::filesystem::path inputPath = argv[1];
		std::filesystem::path outputPath = argv[2];
		std::map<std::string, std::string, std::less<>> renames;
		std::set<std::string, std::less<>> transposes;

		for (int i = 3; i < argc; ++i)
		{
			const std::string_view arg = argv[i];
			if (arg == "--rename")
			{
				if (++i >= argc)
				{
					throw std::runtime_error("--rename requires a from=to value");
				}
				auto [from, to] = ParseRename(argv[i]);
				renames.insert_or_assign(std::move(from), std::move(to));
			}
			else if (arg == "--transpose")
			{
				if (++i >= argc)
				{
					throw std::runtime_error("--transpose requires a tensor name");
				}
				transposes.insert(argv[i]);
			}
			else if (arg == "--help" || arg == "-h")
			{
				PrintUsage(argv[0]);
				return 0;
			}
			else
			{
				throw std::runtime_error("Unknown argument: " + std::string(arg));
			}
		}

		auto archive = LiteNN::Serialization::SafetensorsArchive::LoadFile(inputPath);
		LiteNN::Serialization::SafetensorsImportOptions options;
		options.renameTensor = [&renames](std::string_view name) {
			if (const auto it = renames.find(name); it != renames.end())
			{
				return it->second;
			}
			return std::string(name);
		};
		options.transpose2D = [&transposes](std::string_view name) { return transposes.contains(name); };

		auto graph = LiteNN::Serialization::ImportSafetensorsVariables(archive, options);
		LiteNN::Serialization::Migration::SaveGraphArchive(graph, outputPath);

		std::cout << "Imported " << graph.VariableCount() << " safetensors tensors into " << outputPath.string()
		          << '\n';
		return 0;
	}
	catch (const std::exception& ex)
	{
		std::cerr << "litenn_safetensors_convert: " << ex.what() << '\n';
		return 1;
	}
}
