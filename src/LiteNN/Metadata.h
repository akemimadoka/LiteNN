#include <cstdint>
#include <string>
#include <variant>
#include <vector>

#ifndef LITENN_METADATA_H
#define LITENN_METADATA_H

namespace LiteNN
{
	using ModelMetadataValue = std::variant<std::int64_t, std::uint64_t, double, bool, std::string,
	                                       std::vector<std::int64_t>, std::vector<std::uint64_t>,
	                                       std::vector<double>, std::vector<bool>,
	                                       std::vector<std::string>>;

	struct ModelMetadataEntry
	{
		std::string key;
		ModelMetadataValue value;

		auto operator<=>(const ModelMetadataEntry&) const = default;
	};
} // namespace LiteNN

#endif