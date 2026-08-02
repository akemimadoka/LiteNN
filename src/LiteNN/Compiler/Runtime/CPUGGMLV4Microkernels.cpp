#include "CPUGGMLV4Microkernels.h"

#include <cstdint>
#include <cstring>

namespace LiteNN::Detail
{
	bool CPUHasGGMLV4AVX512F16C()
	{
#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) &&                               \
    (defined(__GNUC__) || defined(__clang__))
		static const bool supported = [] {
			__builtin_cpu_init();
			return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
			       __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("f16c");
		}();
		return supported;
#else
		return false;
#endif
	}
} // namespace LiteNN::Detail

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) &&                               \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>

#define LITENN_TARGET_AVX2_F16C __attribute__((target("avx2,f16c")))
#define LITENN_TARGET_AVX512_F16C __attribute__((target("avx512f,avx512bw,avx512vl,f16c")))

namespace LiteNN::Detail
{
	namespace
	{
		LITENN_TARGET_AVX2_F16C __m256 LoadGGMLFieldInterleavedF16x8(const std::uint16_t values[8])
		{
			return _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(values)));
		}
	} // namespace

	LITENN_TARGET_AVX2_F16C void
	AccumulateGGMLQ4KFieldInterleavedV4BlockQ8Kx8AVX2(const GGMLQ4KFieldInterleaved8Block& block,
	                                                  const GGMLQ8KActivationBlock& lhs, float acc[8])
	{
		const auto d = LoadGGMLFieldInterleavedF16x8(block.d);
		const auto dmin = LoadGGMLFieldInterleavedF16x8(block.dmin);
		const auto lhsD = _mm256_set1_ps(lhs.d);
		auto accumulators = _mm256_loadu_ps(acc);
		auto scaledQuantSum = _mm256_setzero_si256();
		auto scaledMinimumSum = _mm256_setzero_si256();
		const auto nibbleMask = _mm256_set1_epi8(15);
		const auto pairOnes = _mm256_set1_epi16(1);
		for (std::uint64_t subblock = 0; subblock < 8; ++subblock)
		{
			auto quantSum = _mm256_setzero_si256();
			for (std::uint64_t chunk = 0; chunk < 8; ++chunk)
			{
				const auto sourceOffset = (subblock / 2) * 32 + chunk * 4;
				auto quantBytes =
				    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block.qs + (sourceOffset / 4) * 32));
				if (subblock % 2 != 0)
				{
					quantBytes = _mm256_srli_epi16(quantBytes, 4);
				}
				quantBytes = _mm256_and_si256(quantBytes, nibbleMask);
				std::uint32_t q8Word = 0;
				std::memcpy(&q8Word, lhs.qs + subblock * 32 + chunk * 4, sizeof(q8Word));
				const auto q8Bytes = _mm256_set1_epi32(static_cast<int>(q8Word));
				quantSum =
				    _mm256_add_epi32(quantSum, _mm256_madd_epi16(_mm256_maddubs_epi16(quantBytes, q8Bytes), pairOnes));
			}
			const auto scale =
			    _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(block.scales[subblock])));
			const auto minimum =
			    _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(block.minimums[subblock])));
			const auto lhsSum = static_cast<std::int32_t>(lhs.bsums[subblock * 2]) +
			                    static_cast<std::int32_t>(lhs.bsums[subblock * 2 + 1]);
			scaledQuantSum = _mm256_add_epi32(scaledQuantSum, _mm256_mullo_epi32(quantSum, scale));
			scaledMinimumSum =
			    _mm256_add_epi32(scaledMinimumSum, _mm256_mullo_epi32(minimum, _mm256_set1_epi32(lhsSum)));
		}
		const auto blockContribution = _mm256_sub_ps(_mm256_mul_ps(d, _mm256_cvtepi32_ps(scaledQuantSum)),
		                                             _mm256_mul_ps(dmin, _mm256_cvtepi32_ps(scaledMinimumSum)));
		accumulators = _mm256_add_ps(accumulators, _mm256_mul_ps(lhsD, blockContribution));
		_mm256_storeu_ps(acc, accumulators);
	}

	LITENN_TARGET_AVX2_F16C void
	AccumulateGGMLQ6KFieldInterleavedV4BlockQ8Kx8AVX2(const GGMLQ6KFieldInterleaved8Block& block,
	                                                  const GGMLQ8KActivationBlock& lhs, float acc[8])
	{
		const auto d = LoadGGMLFieldInterleavedF16x8(block.d);
		const auto lhsD = _mm256_set1_ps(lhs.d);
		auto accumulators = _mm256_loadu_ps(acc);
		auto scaledQuantSum = _mm256_setzero_si256();
		const auto lowFourMask = _mm256_set1_epi8(15);
		const auto highTwoMask = _mm256_set1_epi8(3);
		const auto pairOnes = _mm256_set1_epi16(1);
		for (std::uint64_t halfBlock = 0; halfBlock < 2; ++halfBlock)
		{
			for (std::uint64_t segment = 0; segment < 4; ++segment)
			{
				for (std::uint64_t group = 0; group < 2; ++group)
				{
					const auto scaleOffset = halfBlock * 8 + group + segment * 2;
					const auto q8Offset = halfBlock * 128 + segment * 32 + group * 16;
					auto quantSum = _mm256_setzero_si256();
					for (std::uint64_t chunk = 0; chunk < 4; ++chunk)
					{
						const auto qlOffset = halfBlock * 64 + group * 16 + chunk * 4 + (segment % 2) * 32;
						const auto qhOffset = halfBlock * 32 + group * 16 + chunk * 4;
						auto lowFour =
						    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block.ql + (qlOffset / 4) * 32));
						if (segment >= 2)
						{
							lowFour = _mm256_srli_epi16(lowFour, 4);
						}
						lowFour = _mm256_and_si256(lowFour, lowFourMask);
						auto highTwo =
						    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block.qh + (qhOffset / 4) * 32));
						switch (segment)
						{
						case 1:
							highTwo = _mm256_srli_epi16(highTwo, 2);
							break;
						case 2:
							highTwo = _mm256_srli_epi16(highTwo, 4);
							break;
						case 3:
							highTwo = _mm256_srli_epi16(highTwo, 6);
							break;
						default:
							break;
						}
						highTwo = _mm256_and_si256(highTwo, highTwoMask);
						const auto quantBytes = _mm256_or_si256(lowFour, _mm256_slli_epi16(highTwo, 4));
						std::uint32_t q8Word = 0;
						std::memcpy(&q8Word, lhs.qs + q8Offset + chunk * 4, sizeof(q8Word));
						const auto q8Bytes = _mm256_set1_epi32(static_cast<int>(q8Word));
						quantSum = _mm256_add_epi32(
						    quantSum, _mm256_madd_epi16(_mm256_maddubs_epi16(quantBytes, q8Bytes), pairOnes));
					}
					quantSum = _mm256_sub_epi32(
					    quantSum, _mm256_set1_epi32(32 * static_cast<std::int32_t>(lhs.bsums[q8Offset / 16])));
					const auto scale = _mm256_cvtepi8_epi32(
					    _mm_loadl_epi64(reinterpret_cast<const __m128i*>(block.scales[scaleOffset])));
					scaledQuantSum = _mm256_add_epi32(scaledQuantSum, _mm256_mullo_epi32(quantSum, scale));
				}
			}
		}
		accumulators =
		    _mm256_add_ps(accumulators, _mm256_mul_ps(_mm256_mul_ps(lhsD, d), _mm256_cvtepi32_ps(scaledQuantSum)));
		_mm256_storeu_ps(acc, accumulators);
	}

	LITENN_TARGET_AVX2_F16C void
	AccumulateGGMLQ4KFieldInterleavedV4BlockQ8Kx16AVX2(const GGMLQ4KFieldInterleaved8Block& block0,
	                                                   const GGMLQ4KFieldInterleaved8Block& block1,
	                                                   const GGMLQ8KActivationBlock& lhs, float acc[16])
	{
		const auto d0 = LoadGGMLFieldInterleavedF16x8(block0.d);
		const auto d1 = LoadGGMLFieldInterleavedF16x8(block1.d);
		const auto dmin0 = LoadGGMLFieldInterleavedF16x8(block0.dmin);
		const auto dmin1 = LoadGGMLFieldInterleavedF16x8(block1.dmin);
		const auto lhsD = _mm256_set1_ps(lhs.d);
		auto accumulators0 = _mm256_loadu_ps(acc);
		auto accumulators1 = _mm256_loadu_ps(acc + 8);
		auto scaledQuantSum0 = _mm256_setzero_si256();
		auto scaledQuantSum1 = _mm256_setzero_si256();
		auto scaledMinimumSum0 = _mm256_setzero_si256();
		auto scaledMinimumSum1 = _mm256_setzero_si256();
		const auto nibbleMask = _mm256_set1_epi8(15);
		const auto pairOnes = _mm256_set1_epi16(1);
		for (std::uint64_t subblock = 0; subblock < 8; ++subblock)
		{
			auto quantSum0 = _mm256_setzero_si256();
			auto quantSum1 = _mm256_setzero_si256();
			for (std::uint64_t chunk = 0; chunk < 8; ++chunk)
			{
				const auto sourceOffset = (subblock / 2) * 32 + chunk * 4;
				auto quantBytes0 =
				    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block0.qs + (sourceOffset / 4) * 32));
				auto quantBytes1 =
				    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block1.qs + (sourceOffset / 4) * 32));
				if (subblock % 2 != 0)
				{
					quantBytes0 = _mm256_srli_epi16(quantBytes0, 4);
					quantBytes1 = _mm256_srli_epi16(quantBytes1, 4);
				}
				quantBytes0 = _mm256_and_si256(quantBytes0, nibbleMask);
				quantBytes1 = _mm256_and_si256(quantBytes1, nibbleMask);
				std::uint32_t q8Word = 0;
				std::memcpy(&q8Word, lhs.qs + subblock * 32 + chunk * 4, sizeof(q8Word));
				const auto q8Bytes = _mm256_set1_epi32(static_cast<int>(q8Word));
				quantSum0 = _mm256_add_epi32(quantSum0,
				                             _mm256_madd_epi16(_mm256_maddubs_epi16(quantBytes0, q8Bytes), pairOnes));
				quantSum1 = _mm256_add_epi32(quantSum1,
				                             _mm256_madd_epi16(_mm256_maddubs_epi16(quantBytes1, q8Bytes), pairOnes));
			}
			const auto scale0 =
			    _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(block0.scales[subblock])));
			const auto scale1 =
			    _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(block1.scales[subblock])));
			const auto minimum0 =
			    _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(block0.minimums[subblock])));
			const auto minimum1 =
			    _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(block1.minimums[subblock])));
			const auto lhsSum = static_cast<std::int32_t>(lhs.bsums[subblock * 2]) +
			                    static_cast<std::int32_t>(lhs.bsums[subblock * 2 + 1]);
			const auto lhsSumVector = _mm256_set1_epi32(lhsSum);
			scaledQuantSum0 = _mm256_add_epi32(scaledQuantSum0, _mm256_mullo_epi32(quantSum0, scale0));
			scaledQuantSum1 = _mm256_add_epi32(scaledQuantSum1, _mm256_mullo_epi32(quantSum1, scale1));
			scaledMinimumSum0 = _mm256_add_epi32(scaledMinimumSum0, _mm256_mullo_epi32(minimum0, lhsSumVector));
			scaledMinimumSum1 = _mm256_add_epi32(scaledMinimumSum1, _mm256_mullo_epi32(minimum1, lhsSumVector));
		}
		const auto blockContribution0 = _mm256_sub_ps(_mm256_mul_ps(d0, _mm256_cvtepi32_ps(scaledQuantSum0)),
		                                              _mm256_mul_ps(dmin0, _mm256_cvtepi32_ps(scaledMinimumSum0)));
		const auto blockContribution1 = _mm256_sub_ps(_mm256_mul_ps(d1, _mm256_cvtepi32_ps(scaledQuantSum1)),
		                                              _mm256_mul_ps(dmin1, _mm256_cvtepi32_ps(scaledMinimumSum1)));
		accumulators0 = _mm256_add_ps(accumulators0, _mm256_mul_ps(lhsD, blockContribution0));
		accumulators1 = _mm256_add_ps(accumulators1, _mm256_mul_ps(lhsD, blockContribution1));
		_mm256_storeu_ps(acc, accumulators0);
		_mm256_storeu_ps(acc + 8, accumulators1);
	}

	LITENN_TARGET_AVX2_F16C void
	AccumulateGGMLQ6KFieldInterleavedV4BlockQ8Kx16AVX2(const GGMLQ6KFieldInterleaved8Block& block0,
	                                                   const GGMLQ6KFieldInterleaved8Block& block1,
	                                                   const GGMLQ8KActivationBlock& lhs, float acc[16])
	{
		const auto d0 = LoadGGMLFieldInterleavedF16x8(block0.d);
		const auto d1 = LoadGGMLFieldInterleavedF16x8(block1.d);
		const auto lhsD = _mm256_set1_ps(lhs.d);
		auto accumulators0 = _mm256_loadu_ps(acc);
		auto accumulators1 = _mm256_loadu_ps(acc + 8);
		auto scaledQuantSum0 = _mm256_setzero_si256();
		auto scaledQuantSum1 = _mm256_setzero_si256();
		const auto lowFourMask = _mm256_set1_epi8(15);
		const auto highTwoMask = _mm256_set1_epi8(3);
		const auto pairOnes = _mm256_set1_epi16(1);
		for (std::uint64_t halfBlock = 0; halfBlock < 2; ++halfBlock)
		{
			for (std::uint64_t segment = 0; segment < 4; ++segment)
			{
				for (std::uint64_t group = 0; group < 2; ++group)
				{
					const auto scaleOffset = halfBlock * 8 + group + segment * 2;
					const auto q8Offset = halfBlock * 128 + segment * 32 + group * 16;
					auto quantSum0 = _mm256_setzero_si256();
					auto quantSum1 = _mm256_setzero_si256();
					for (std::uint64_t chunk = 0; chunk < 4; ++chunk)
					{
						const auto qlOffset = halfBlock * 64 + group * 16 + chunk * 4 + (segment % 2) * 32;
						const auto qhOffset = halfBlock * 32 + group * 16 + chunk * 4;
						auto lowFour0 =
						    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block0.ql + (qlOffset / 4) * 32));
						auto lowFour1 =
						    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block1.ql + (qlOffset / 4) * 32));
						if (segment >= 2)
						{
							lowFour0 = _mm256_srli_epi16(lowFour0, 4);
							lowFour1 = _mm256_srli_epi16(lowFour1, 4);
						}
						lowFour0 = _mm256_and_si256(lowFour0, lowFourMask);
						lowFour1 = _mm256_and_si256(lowFour1, lowFourMask);
						auto highTwo0 =
						    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block0.qh + (qhOffset / 4) * 32));
						auto highTwo1 =
						    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block1.qh + (qhOffset / 4) * 32));
						switch (segment)
						{
						case 1:
							highTwo0 = _mm256_srli_epi16(highTwo0, 2);
							highTwo1 = _mm256_srli_epi16(highTwo1, 2);
							break;
						case 2:
							highTwo0 = _mm256_srli_epi16(highTwo0, 4);
							highTwo1 = _mm256_srli_epi16(highTwo1, 4);
							break;
						case 3:
							highTwo0 = _mm256_srli_epi16(highTwo0, 6);
							highTwo1 = _mm256_srli_epi16(highTwo1, 6);
							break;
						default:
							break;
						}
						highTwo0 = _mm256_and_si256(highTwo0, highTwoMask);
						highTwo1 = _mm256_and_si256(highTwo1, highTwoMask);
						const auto quantBytes0 = _mm256_or_si256(lowFour0, _mm256_slli_epi16(highTwo0, 4));
						const auto quantBytes1 = _mm256_or_si256(lowFour1, _mm256_slli_epi16(highTwo1, 4));
						std::uint32_t q8Word = 0;
						std::memcpy(&q8Word, lhs.qs + q8Offset + chunk * 4, sizeof(q8Word));
						const auto q8Bytes = _mm256_set1_epi32(static_cast<int>(q8Word));
						quantSum0 = _mm256_add_epi32(
						    quantSum0, _mm256_madd_epi16(_mm256_maddubs_epi16(quantBytes0, q8Bytes), pairOnes));
						quantSum1 = _mm256_add_epi32(
						    quantSum1, _mm256_madd_epi16(_mm256_maddubs_epi16(quantBytes1, q8Bytes), pairOnes));
					}
					const auto correction = _mm256_set1_epi32(32 * static_cast<std::int32_t>(lhs.bsums[q8Offset / 16]));
					quantSum0 = _mm256_sub_epi32(quantSum0, correction);
					quantSum1 = _mm256_sub_epi32(quantSum1, correction);
					const auto scale0 = _mm256_cvtepi8_epi32(
					    _mm_loadl_epi64(reinterpret_cast<const __m128i*>(block0.scales[scaleOffset])));
					const auto scale1 = _mm256_cvtepi8_epi32(
					    _mm_loadl_epi64(reinterpret_cast<const __m128i*>(block1.scales[scaleOffset])));
					scaledQuantSum0 = _mm256_add_epi32(scaledQuantSum0, _mm256_mullo_epi32(quantSum0, scale0));
					scaledQuantSum1 = _mm256_add_epi32(scaledQuantSum1, _mm256_mullo_epi32(quantSum1, scale1));
				}
			}
		}
		accumulators0 =
		    _mm256_add_ps(accumulators0, _mm256_mul_ps(_mm256_mul_ps(lhsD, d0), _mm256_cvtepi32_ps(scaledQuantSum0)));
		accumulators1 =
		    _mm256_add_ps(accumulators1, _mm256_mul_ps(_mm256_mul_ps(lhsD, d1), _mm256_cvtepi32_ps(scaledQuantSum1)));
		_mm256_storeu_ps(acc, accumulators0);
		_mm256_storeu_ps(acc + 8, accumulators1);
	}

	LITENN_TARGET_AVX512_F16C void
	AccumulateGGMLQ6KFieldInterleavedV4BlockQ8Kx16AVX512(const GGMLQ6KFieldInterleaved8Block& block0,
	                                                     const GGMLQ6KFieldInterleaved8Block& block1,
	                                                     const GGMLQ8KActivationBlock& lhs, float acc[16])
	{
		const auto dF16 = _mm256_set_m128i(_mm_loadu_si128(reinterpret_cast<const __m128i*>(block1.d)),
		                                   _mm_loadu_si128(reinterpret_cast<const __m128i*>(block0.d)));
		const auto d = _mm512_cvtph_ps(dF16);
		const auto lhsD = _mm512_set1_ps(lhs.d);
		auto accumulators = _mm512_loadu_ps(acc);
		auto scaledQuantSum = _mm512_setzero_si512();
		const auto lowFourMask = _mm512_set1_epi8(15);
		const auto highTwoMask = _mm512_set1_epi8(3);
		const auto pairOnes = _mm512_set1_epi16(1);
		for (std::uint64_t halfBlock = 0; halfBlock < 2; ++halfBlock)
		{
			for (std::uint64_t segment = 0; segment < 4; ++segment)
			{
				for (std::uint64_t group = 0; group < 2; ++group)
				{
					const auto scaleOffset = halfBlock * 8 + group + segment * 2;
					const auto q8Offset = halfBlock * 128 + segment * 32 + group * 16;
					auto quantSum = _mm512_setzero_si512();
					for (std::uint64_t chunk = 0; chunk < 4; ++chunk)
					{
						const auto qlOffset = halfBlock * 64 + group * 16 + chunk * 4 + (segment % 2) * 32;
						const auto qhOffset = halfBlock * 32 + group * 16 + chunk * 4;
						const auto lowFour0 =
						    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block0.ql + (qlOffset / 4) * 32));
						const auto lowFour1 =
						    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block1.ql + (qlOffset / 4) * 32));
						auto lowFour = _mm512_inserti64x4(_mm512_castsi256_si512(lowFour0), lowFour1, 1);
						if (segment >= 2)
						{
							lowFour = _mm512_srli_epi16(lowFour, 4);
						}
						lowFour = _mm512_and_si512(lowFour, lowFourMask);
						const auto highTwo0 =
						    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block0.qh + (qhOffset / 4) * 32));
						const auto highTwo1 =
						    _mm256_loadu_si256(reinterpret_cast<const __m256i*>(block1.qh + (qhOffset / 4) * 32));
						auto highTwo = _mm512_inserti64x4(_mm512_castsi256_si512(highTwo0), highTwo1, 1);
						switch (segment)
						{
						case 1:
							highTwo = _mm512_srli_epi16(highTwo, 2);
							break;
						case 2:
							highTwo = _mm512_srli_epi16(highTwo, 4);
							break;
						case 3:
							highTwo = _mm512_srli_epi16(highTwo, 6);
							break;
						default:
							break;
						}
						highTwo = _mm512_and_si512(highTwo, highTwoMask);
						const auto quantBytes = _mm512_or_si512(lowFour, _mm512_slli_epi16(highTwo, 4));
						std::uint32_t q8Word = 0;
						std::memcpy(&q8Word, lhs.qs + q8Offset + chunk * 4, sizeof(q8Word));
						const auto q8Bytes = _mm512_set1_epi32(static_cast<int>(q8Word));
						quantSum = _mm512_add_epi32(
						    quantSum, _mm512_madd_epi16(_mm512_maddubs_epi16(quantBytes, q8Bytes), pairOnes));
					}
					quantSum = _mm512_sub_epi32(
					    quantSum, _mm512_set1_epi32(32 * static_cast<std::int32_t>(lhs.bsums[q8Offset / 16])));
					const auto scales = _mm_unpacklo_epi64(
					    _mm_loadl_epi64(reinterpret_cast<const __m128i*>(block0.scales[scaleOffset])),
					    _mm_loadl_epi64(reinterpret_cast<const __m128i*>(block1.scales[scaleOffset])));
					const auto scale = _mm512_cvtepi8_epi32(scales);
					scaledQuantSum = _mm512_add_epi32(scaledQuantSum, _mm512_mullo_epi32(quantSum, scale));
				}
			}
		}
		accumulators =
		    _mm512_add_ps(accumulators, _mm512_mul_ps(_mm512_mul_ps(lhsD, d), _mm512_cvtepi32_ps(scaledQuantSum)));
		_mm512_storeu_ps(acc, accumulators);
	}
} // namespace LiteNN::Detail

#endif
