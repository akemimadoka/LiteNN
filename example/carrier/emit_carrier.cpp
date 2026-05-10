#include "graph_common.h"

#include <filesystem>
#include <format>
#include <iostream>

namespace
{
	using namespace LiteNN;
	using namespace LiteNN::Examples::Carrier;

	int Run(int argc, char** argv)
	{
		if (argc != 3)
		{
			std::cerr << "Usage: litenn_carrier_emit <output-object> <symbol-prefix>\n";
			return 1;
		}

		const std::filesystem::path outputPath = argv[1];
		const std::string_view symbolPrefix = argv[2];
		std::filesystem::create_directories(outputPath.parent_path());

		auto artifact = Compiler<CPU>::CompileArtifact(BuildCarrierExampleGraph());
		artifact.WriteObjectFile(outputPath, symbolPrefix);

		std::cout << std::format(
		    "Wrote carrier object {} (rodata={} bytes, instructions={} bytes, prefix={})\n",
		    outputPath.string(), artifact.Rodata().size(), artifact.Instructions().size(), symbolPrefix);
		return 0;
	}
} // namespace

int main(int argc, char** argv)
{
	try
	{
		return Run(argc, argv);
	}
	catch (const std::exception& ex)
	{
		std::cerr << "error: " << ex.what() << '\n';
		return 1;
	}
}