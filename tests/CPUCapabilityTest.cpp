#include <gtest/gtest.h>

#include <LiteNN/Device.h>
#include <LiteNN/Tensor.h>

#include <memory>

using namespace LiteNN;

TEST(CPUCapabilityTest, ReportsCompileTimeSIMDFeatures)
{
	const auto caps = QueryCPUCapabilities();

#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
	EXPECT_TRUE(caps.x86SSE2);
#else
	EXPECT_FALSE(caps.x86SSE2);
#endif

#if defined(__ARM_NEON) || defined(__ARM_NEON__) || defined(_M_ARM64)
	EXPECT_TRUE(caps.armNEON);
#else
	EXPECT_FALSE(caps.armNEON);
#endif
}

TEST(CPUAllocatorTest, ArenaBacksTensorStorage)
{
	auto arena = std::make_shared<CPULinearArena>(128);
	CPU device{ .allocator = arena };

	Tensor<CPU> first(Uninitialized, { 4 }, DataType::Float32, device);
	const auto usedAfterFirst = arena->Used();
	EXPECT_GE(usedAfterFirst, 4u * sizeof(float));

	Tensor<CPU> second(Uninitialized, { 4 }, DataType::Float32, device);
	EXPECT_GT(arena->Used(), usedAfterFirst);

	arena->Reset();
	EXPECT_EQ(arena->Used(), 0u);
}

TEST(CPUAllocatorTest, ArenaReportsCapacityExhaustion)
{
	auto arena = std::make_shared<CPULinearArena>(8);
	CPU device{ .allocator = arena };

	EXPECT_THROW((Tensor<CPU>(Uninitialized, { 3 }, DataType::Float32, device)), std::bad_alloc);
}

TEST(CPUAllocatorTest, CPUEqualityTracksAllocatorIdentity)
{
	CPU defaultA;
	CPU defaultB;
	EXPECT_EQ(defaultA, defaultB);

	auto arena = std::make_shared<CPULinearArena>(64);
	CPU arenaA{ .allocator = arena };
	CPU arenaB{ .allocator = arena };
	CPU differentArena{ .allocator = std::make_shared<CPULinearArena>(64) };

	EXPECT_EQ(arenaA, arenaB);
	EXPECT_NE(arenaA, defaultA);
	EXPECT_NE(arenaA, differentArena);
}
