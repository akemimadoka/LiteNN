#include <gtest/gtest.h>

#include <LiteNN/Device.h>

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
