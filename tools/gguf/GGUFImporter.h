#include <LiteNN.h>

#include <cstddef>
#include <filesystem>

#ifndef LITENN_GGUFIMPORTER_H
#define LITENN_GGUFIMPORTER_H

namespace LiteNN::GGUF
{
	struct ImportSummary
	{
		std::size_t tensorCount{};
		std::size_t metadataCount{};
	};

	struct ImportResult
	{
		Graph graph;
		ImportSummary summary;
	};

	ImportResult ImportGGUFArchive(const std::filesystem::path& inputPath);
	ImportSummary ConvertGGUFArchive(const std::filesystem::path& inputPath,
	                                const std::filesystem::path& outputPath);
} // namespace LiteNN::GGUF

#endif