#pragma once

#include <cstdint>

namespace LiteNN::Detail
{
	struct GGMLQ8KActivationBlock
	{
		float d{};
		std::int8_t qs[256]{};
		std::int16_t bsums[16]{};
	};

	struct GGMLQ4KFieldInterleaved8Block
	{
		std::uint16_t d[8]{};
		std::uint16_t dmin[8]{};
		std::uint8_t scales[8][8]{};
		std::uint8_t minimums[8][8]{};
		std::uint8_t qs[1024]{};
	};

	struct GGMLQ6KFieldInterleaved8Block
	{
		std::uint16_t d[8]{};
		std::int8_t scales[16][8]{};
		std::uint8_t ql[1024]{};
		std::uint8_t qh[512]{};
	};

	static_assert(sizeof(GGMLQ8KActivationBlock) == 292);
	static_assert(sizeof(GGMLQ4KFieldInterleaved8Block) == 1184);
	static_assert(sizeof(GGMLQ6KFieldInterleaved8Block) == 1680);

	void AccumulateGGMLQ4KFieldInterleavedV4BlockQ8Kx8AVX2(const GGMLQ4KFieldInterleaved8Block& block,
	                                                       const GGMLQ8KActivationBlock& lhs, float acc[8]);
	void AccumulateGGMLQ6KFieldInterleavedV4BlockQ8Kx8AVX2(const GGMLQ6KFieldInterleaved8Block& block,
	                                                       const GGMLQ8KActivationBlock& lhs, float acc[8]);
	void AccumulateGGMLQ4KFieldInterleavedV4BlockQ8Kx16AVX2(const GGMLQ4KFieldInterleaved8Block& block0,
	                                                        const GGMLQ4KFieldInterleaved8Block& block1,
	                                                        const GGMLQ8KActivationBlock& lhs, float acc[16]);
	void AccumulateGGMLQ6KFieldInterleavedV4BlockQ8Kx16AVX2(const GGMLQ6KFieldInterleaved8Block& block0,
	                                                        const GGMLQ6KFieldInterleaved8Block& block1,
	                                                        const GGMLQ8KActivationBlock& lhs, float acc[16]);
} // namespace LiteNN::Detail
