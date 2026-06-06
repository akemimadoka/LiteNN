#ifndef LITENN_SERIALIZATION_EXTERNALWEIGHTS_H
#define LITENN_SERIALIZATION_EXTERNALWEIGHTS_H

#include <cstdint>

namespace LiteNN::Serialization
{
	/// Controls when variable tensor payloads are written to a sibling external weight file.
	struct ExternalWeightSaveOptions
	{
		std::uint64_t minVariableBytes{ 0 };
		std::uint64_t alignment{ 64 };
	};
} // namespace LiteNN::Serialization

#endif
