#include "GGUFImporter.h"
#include "LLaMABuilder.h"

#include <LiteNN/Serialization/ModelIO.h>

#include <charconv>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace
{
	void PrintUsage(std::string_view executable)
	{
		std::cerr << "Usage:\n"
		          << "  " << executable << " --import <input.gguf> <output.ltnn>\n"
		          << "  " << executable << " --lower-llama <input.gguf> <output.ltnn> <sequence-length> [position-offset]\n"
		          << "  " << executable << " <input.gguf> <output.ltnn>  (alias for --import)\n";
	}

	std::size_t ParseSize(std::string_view text, std::string_view label, bool allowZero = false)
	{
		std::size_t value{};
		const auto* first = text.data();
		const auto* last = text.data() + text.size();
		const auto result = std::from_chars(first, last, value);
		if (result.ec != std::errc{} || result.ptr != last || (!allowZero && value == 0))
		{
			throw std::runtime_error(std::string(label) + (allowZero ? " must be a non-negative integer"
			                                                        : " must be a positive integer"));
		}
		return value;
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

		if (argc == 3)
		{
			const auto summary = LiteNN::GGUF::ConvertGGUFArchive(argv[1], argv[2]);
			std::cout << "Imported archive with " << summary.tensorCount << " tensors and " << summary.metadataCount
			          << " metadata entries\n";
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--import")
		{
			if (argc != 4)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			const auto summary = LiteNN::GGUF::ConvertGGUFArchive(argv[2], argv[3]);
			std::cout << "Imported archive with " << summary.tensorCount << " tensors and " << summary.metadataCount
			          << " metadata entries\n";
			return 0;
		}

		if (argc >= 2 && std::string_view(argv[1]) == "--lower-llama")
		{
			if (argc != 5 && argc != 6)
			{
				PrintUsage(argv[0]);
				return 1;
			}
			const auto imported = LiteNN::GGUF::ImportGGUFArchive(argv[2]);
			const auto sequenceLength = ParseSize(argv[4], "sequence-length");
			const auto positionOffset = argc == 6 ? ParseSize(argv[5], "position-offset", true) : 0uz;
			auto lowered = LiteNN::GGUF::LowerLLaMACausalLM(imported.graph, sequenceLength, positionOffset);
			LiteNN::Serialization::SaveModel(lowered, argv[3]);
			std::cout << "Lowered LLaMA graph from " << imported.summary.tensorCount << " tensors and "
			          << imported.summary.metadataCount << " metadata entries\n";
			return 0;
		}

		PrintUsage(argv[0]);
		return 1;
	}
	catch (const std::exception& ex)
	{
		std::cerr << "litenn_gguf_convert: " << ex.what() << '\n';
		return 1;
	}
}
