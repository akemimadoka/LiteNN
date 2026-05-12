#include <gtest/gtest.h>

#include <LiteNN.h>

#ifdef LITENN_ENABLE_CUDA

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

#endif
