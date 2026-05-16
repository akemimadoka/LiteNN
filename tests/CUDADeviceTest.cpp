#include <gtest/gtest.h>

#include <LiteNN.h>

#ifdef LITENN_ENABLE_CUDA

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace LiteNN;

static float ReadFloat(const Tensor<CPU>& tensor, std::size_t index)
{
	return static_cast<const float*>(tensor.RawData())[index];
}

static std::vector<float> ReadAsFloat32(const Tensor<CPU>& tensor)
{
	std::vector<float> values(tensor.NumElements());
	CPU cpu;
	DeviceTraits<CPU>::ConvertTo(cpu, tensor.DType(), tensor.RawData(), tensor.NumElements(), DataType::Float32,
	                             values.data());
	return values;
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

TEST(CUDADevice, LowPrecisionCapabilitiesAreReported)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	const auto capabilities = GetCUDALowPrecisionCapabilities();
	EXPECT_EQ(capabilities.deviceIndex, 0);
	EXPECT_GE(capabilities.computeCapabilityMajor, 0);
	EXPECT_GE(capabilities.computeCapabilityMinor, 0);
	EXPECT_TRUE(capabilities.supportsFloat16Storage);
	EXPECT_TRUE(capabilities.supportsBFloat16Storage);
	EXPECT_TRUE(capabilities.supportsFloat8Storage);
	EXPECT_TRUE(capabilities.supportsInt8Storage);
	EXPECT_TRUE(CUDASupportsLowPrecisionStorage(DataType::Float16));
	EXPECT_TRUE(CUDASupportsNativeMatMul(DataType::Float32));

	const auto summary = FormatCUDALowPrecisionCapabilities(capabilities);
	EXPECT_NE(summary.find("cc"), std::string::npos);
	EXPECT_NE(summary.find("fp16"), std::string::npos);
	EXPECT_NE(summary.find("bf16"), std::string::npos);
}

TEST(CUDADevice, LowPrecisionConversionRoundTripUsesCPUBridge)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	const std::array dataTypes{
		DataType::Float16,
		DataType::BFloat16,
		DataType::Float8E4M3,
		DataType::Float8E5M2,
		DataType::Int8,
		DataType::UInt8,
	};
	for (const auto dataType : dataTypes)
	{
		Tensor<CPU> cpuTensor({ 0, 1, 2, 3, 4 }, { 5 }, dataType);
		const auto expected = ReadAsFloat32(cpuTensor);

		auto cudaTensor = cpuTensor.CopyToDevice(CUDA{});
		auto backToCpu = cudaTensor.CopyToDevice(CPU{});
		const auto actual = ReadAsFloat32(backToCpu);

		ASSERT_EQ(actual.size(), expected.size());
		for (auto i = 0uz; i < actual.size(); ++i)
		{
			EXPECT_FLOAT_EQ(actual[i], expected[i]) << "dtype=" << DataTypeName(dataType) << ", index=" << i;
		}
	}
}

TEST(CUDADevice, Float16MatMulMatchesCPU)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}
	if (!CUDASupportsNativeMatMul(DataType::Float16))
	{
		GTEST_SKIP() << "CUDA device does not report native Float16 MatMul support";
	}

	Tensor<CPU> lhs({ 1, 2, 3, 4, 5, 6 }, { 2, 3 }, DataType::Float16);
	Tensor<CPU> rhs({ 7, 8, 9, 10, 11, 12 }, { 3, 2 }, DataType::Float16);
	const auto expected = ReadAsFloat32(lhs.MatMul(rhs));

	auto cudaLhs = lhs.CopyToDevice(CUDA{});
	auto cudaRhs = rhs.CopyToDevice(CUDA{});
	auto cudaResult = cudaLhs.MatMul(cudaRhs).CopyToDevice(CPU{});
	const auto actual = ReadAsFloat32(cudaResult);

	ASSERT_EQ(actual.size(), expected.size());
	for (auto i = 0uz; i < actual.size(); ++i)
	{
		EXPECT_FLOAT_EQ(actual[i], expected[i]);
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

TEST(CUDADevice, ReportsInvalidDeviceIndex)
{
	const int invalidDevice = CUDADeviceCount();
	try
	{
		Tensor<CUDA> tensor(Uninitialized, { 1 }, DataType::Float32, CUDA{ .deviceIndex = invalidDevice });
		(void)tensor;
		FAIL() << "expected invalid CUDA device allocation to throw";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_TRUE(message.find("cudaGetDevice") != std::string::npos ||
		            message.find("cudaSetDevice") != std::string::npos ||
		            message.find("cudaMalloc") != std::string::npos)
		    << message;
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

TEST(CUDADevice, DriverModuleReportsInvalidImageDiagnostics)
{
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	constexpr std::string_view kInvalidImage = "this is not PTX";
	const auto bytes = std::as_bytes(std::span(kInvalidImage.data(), kInvalidImage.size()));
	try
	{
		CUDADriverModule module(CUDA{}, bytes);
		(void)module;
		FAIL() << "expected invalid CUDA module image to throw";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("cuModuleLoadDataEx"), std::string::npos);
	}
}

TEST(CUDADevice, DriverModuleReportsUnsupportedTargetDiagnostics)
{
	if (!IsCUDADriverAvailable())
	{
		GTEST_SKIP() << "CUDA driver is not available";
	}

	constexpr std::string_view kUnsupportedTargetPTX = R"ptx(
.version 6.4
.target sm_999
.address_size 64

.visible .entry litenn_noop()
{
	ret;
}
)ptx";
	const auto bytes = std::as_bytes(std::span(kUnsupportedTargetPTX.data(), kUnsupportedTargetPTX.size()));
	try
	{
		CUDADriverModule module(CUDA{}, bytes);
		(void)module;
		FAIL() << "expected unsupported CUDA module target to throw";
	}
	catch (const std::runtime_error& ex)
	{
		const std::string message = ex.what();
		EXPECT_NE(message.find("cuModuleLoadDataEx"), std::string::npos);
	}
}

#endif
