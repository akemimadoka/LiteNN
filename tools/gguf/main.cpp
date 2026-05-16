#include "GGUFImporter.h"

#include <iostream>
#include <string_view>

namespace
{
	void PrintUsage(std::string_view executable)
	{
		std::cerr << "Usage: " << executable << " <input.gguf> <output.ltnn>\n";
	}
} // namespace

int main(int argc, char** argv)
{
	try
	{
		if (argc == 2)
		{
			const std::string_view arg = argv[1];
			if (arg == "-h" || arg == "--help")
			{
				PrintUsage(argv[0]);
				return 0;
			}
		}

		if (argc != 3)
		{
			PrintUsage(argv[0]);
			return 1;
		}

		const auto summary = LiteNN::GGUF::ConvertGGUFArchive(argv[1], argv[2]);
		std::cout << "Converted " << summary.tensorCount << " tensors and " << summary.metadataCount
		          << " metadata entries\n";
		return 0;
	}
	catch (const std::exception& ex)
	{
		std::cerr << "litenn_gguf_convert: " << ex.what() << '\n';
		return 1;
	}
}