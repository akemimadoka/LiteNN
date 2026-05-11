#include <LiteNN/Graph.h>

#include <cstddef>
#include <string>

#ifndef LITENN_DEBUG_DUMP_H
#define LITENN_DEBUG_DUMP_H

namespace LiteNN::Debug
{
	struct GraphDumpOptions
	{
		bool includeConstantValues{ true };
		std::size_t maxConstantElements{ 16 };
	};

	std::string DumpGraph(const Graph& graph, const GraphDumpOptions& options = {});
} // namespace LiteNN::Debug

#endif