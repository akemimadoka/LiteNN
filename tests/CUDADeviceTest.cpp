#include <gtest/gtest.h>

#include <LiteNN.h>

#ifdef LITENN_ENABLE_CUDA

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

using namespace LiteNN;

static float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
{
	return static_cast<const float*>(tensor.RawData())[index];
}

TEST(CUDADevice, RoundTripAndElementwiseFallback)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	Tensor<CPU> cpuTensor({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	auto cudaTensor = cpuTensor.CopyToDevice(CUDA{});
	auto cudaClone = cudaTensor.CopyToDevice(CUDA{});
	auto sum = cudaTensor + cudaTensor;
	auto backToCpu = sum.CopyToDevice(CPU{});
	auto cloneBackToCpu = cudaClone.CopyToDevice(CPU{});

	for (auto i = 0uz; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(backToCpu, i), ReadFloat(cpuTensor, i) * 2.0F);
		EXPECT_FLOAT_EQ(ReadFloat(cloneBackToCpu, i), ReadFloat(cpuTensor, i));
	}
}

TEST(CUDADevice, MatMulMatchesCPU)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	Tensor<CPU> cpuTensor({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	const auto cpuResult = cpuTensor.MatMul(cpuTensor.Transpose());

	auto cudaTensor = cpuTensor.CopyToDevice(CUDA{});
	auto cudaResult = cudaTensor.MatMul(cudaTensor.Transpose()).CopyToDevice(CPU{});

	for (auto i = 0uz; i < cpuResult.NumElements(); ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(cudaResult, i), ReadFloat(cpuResult, i));
	}
}

TEST(CUDADevice, PolymorphicRoundTrip)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	Tensor<CPU> cpuTensor({ 1, 2, 3, 4 }, { 2, 2 });
	auto polyTensor = cpuTensor.CopyToDevice(PolymorphicDevice{ CUDA{} });
	auto polyClone = polyTensor.CopyToDevice(PolymorphicDevice{ CUDA{} });
	auto backToCpu = polyClone.CopyToDevice(CPU{});

	for (auto i = 0uz; i < cpuTensor.NumElements(); ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(backToCpu, i), ReadFloat(cpuTensor, i));
	}
}

TEST(CUDADevice, CrossDeviceCopyUsesFallbackWhenAvailable)
{
	if (CUDADeviceCount() < 2)
	{
		GTEST_SKIP() << "At least two CUDA devices are required";
	}

	Tensor<CPU> cpuTensor({ 1, 2, 3, 4 }, { 2, 2 });
	auto cudaTensor = cpuTensor.CopyToDevice(CUDA{ .deviceIndex = 0 });
	auto secondDeviceTensor = cudaTensor.CopyToDevice(CUDA{ .deviceIndex = 1 });
	auto backToCpu = secondDeviceTensor.CopyToDevice(CPU{});

	for (auto i = 0uz; i < cpuTensor.NumElements(); ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(backToCpu, i), ReadFloat(cpuTensor, i));
	}
}

TEST(CUDADevice, DriverModuleLaunchesPTXKernel)
{
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	constexpr std::string_view kAddOnePTX = R"ptx(
.version 6.4
.target sm_30
.address_size 64

.visible .entry litenn_add_one(
	.param .u64 out_ptr,
	.param .u64 in_ptr,
	.param .u32 count
)
{
	.reg .pred %p<2>;
	.reg .b32 %r<4>;
	.reg .b64 %rd<7>;
	.reg .f32 %f<4>;

	ld.param.u64 %rd1, [out_ptr];
	ld.param.u64 %rd2, [in_ptr];
	ld.param.u32 %r1, [count];
	mov.u32 %r2, %tid.x;
	setp.ge.u32 %p1, %r2, %r1;
	@%p1 bra DONE;

	mul.wide.u32 %rd3, %r2, 4;
	add.s64 %rd4, %rd2, %rd3;
	ld.global.f32 %f1, [%rd4];
	mov.f32 %f2, 0f3f800000;
	add.rn.f32 %f3, %f1, %f2;
	add.s64 %rd5, %rd1, %rd3;
	st.global.f32 [%rd5], %f3;

DONE:
	ret;
}
)ptx";

	const auto ptxBytes = std::as_bytes(std::span(kAddOnePTX.data(), kAddOnePTX.size()));
	CUDADriverModule module(CUDA{}, ptxBytes);

	Tensor<CPU> input({ 1, 2, 3, 4 }, { 4 }, DataType::Float32);
	auto cudaInput = input.CopyToDevice(CUDA{});
	Tensor<CUDA> cudaOutput(Uninitialized, { 4 }, DataType::Float32, CUDA{});

	void* outPtr = cudaOutput.RawData();
	void* inPtr = cudaInput.RawData();
	std::uint32_t count = 4;
	std::array<void*, 3> args = {
		&outPtr,
		&inPtr,
		&count,
	};

	module.Launch("litenn_add_one",
	              {
	                  .grid = { .x = 1, .y = 1, .z = 1 },
	                  .block = { .x = 4, .y = 1, .z = 1 },
	                  .synchronize = true,
	              },
	              args);

	auto result = cudaOutput.CopyToDevice(CPU{});
	for (auto i = 0uz; i < input.NumElements(); ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(result, i), ReadFloat(input, i) + 1.0F);
	}
}

#endif
