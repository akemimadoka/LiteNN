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

static std::vector<float> QuantizeAsFloat32(std::span<const float> values, DataType dataType)
{
	Tensor<CPU> quantized(Uninitialized, { values.size() }, dataType);
	CPU cpu;
	DeviceTraits<CPU>::CopyFromCPU(cpu, dataType, quantized.RawData(), DataType::Float32, values.data(), values.size());
	return ReadAsFloat32(quantized);
}

static std::vector<float> RoundTripFloat32ThroughCUDA(std::span<const float> values, DataType dataType)
{
	CUDA device{};
	Tensor<CUDA> quantized(Uninitialized, { values.size() }, dataType, device);
	DeviceTraits<CUDA>::CopyFromCPU(device, dataType, quantized.RawData(), DataType::Float32, values.data(),
	                                values.size());
	std::vector<float> result(values.size());
	DeviceTraits<CUDA>::CopyToCPU(device, dataType, quantized.RawData(), values.size(), DataType::Float32,
	                              result.data());
	return result;
}

static std::vector<float> ConvertCUDADeviceToFloat32(std::span<const float> values, DataType dataType)
{
	CUDA device{};
	Tensor<CUDA> quantized(Uninitialized, { values.size() }, dataType, device);
	Tensor<CUDA> converted(Uninitialized, { values.size() }, DataType::Float32, device);
	DeviceTraits<CUDA>::CopyFromCPU(device, dataType, quantized.RawData(), DataType::Float32, values.data(),
	                                values.size());
	DeviceTraits<CUDA>::ConvertTo(device, dataType, quantized.RawData(), values.size(), DataType::Float32,
	                              converted.RawData());
	std::vector<float> result(values.size());
	DeviceTraits<CUDA>::CopyToCPU(device, DataType::Float32, converted.RawData(), values.size(), DataType::Float32,
	                              result.data());
	return result;
}

static std::vector<float> HighPrecisionMatMulReference(const Tensor<CPU>& lhs, const Tensor<CPU>& rhs,
	                                                    std::size_t m, std::size_t k, std::size_t n)
{
	const auto lhsValues = ReadAsFloat32(lhs);
	const auto rhsValues = ReadAsFloat32(rhs);
	std::vector<float> accumulated(m * n, 0.0F);
	for (auto row = 0uz; row < m; ++row)
	{
		for (auto depth = 0uz; depth < k; ++depth)
		{
			const auto lhsValue = lhsValues[row * k + depth];
			for (auto col = 0uz; col < n; ++col)
			{
				accumulated[row * n + col] += lhsValue * rhsValues[depth * n + col];
			}
		}
	}
	return QuantizeAsFloat32(accumulated, lhs.DType());
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
	EXPECT_NE(summary.find("nativeConvert"), std::string::npos);

	const std::array lowPrecisionTypes{
		DataType::Float16,
		DataType::BFloat16,
		DataType::Float8E4M3,
		DataType::Float8E5M2,
		DataType::Int8,
		DataType::UInt8,
	};
	const auto nativeConversionAvailable = CUDASupportsNativeConversion(DataType::Float32, DataType::Float16);
	for (const auto dataType : lowPrecisionTypes)
	{
		EXPECT_EQ(CUDASupportsNativeConversion(DataType::Float32, dataType), nativeConversionAvailable)
		    << DataTypeName(dataType);
		EXPECT_EQ(CUDASupportsNativeConversion(dataType, DataType::Float32), nativeConversionAvailable)
		    << DataTypeName(dataType);
	}
}

TEST(CUDADevice, LowPrecisionConversionPathsMatchCPUReference)
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
		const std::vector<float> sourceValues = dataType == DataType::UInt8
		                                          ? std::vector<float>{ 0.0F, 1.0F, 2.0F, 3.0F, 7.0F }
		                                          : dataType == DataType::Int8
		                                              ? std::vector<float>{ -5.0F, -1.0F, 0.0F, 3.0F, 7.0F }
		                                              : std::vector<float>{ -2.25F, -1.0F, 0.0F, 1.5F, 3.75F };
		const auto expected = QuantizeAsFloat32(sourceValues, dataType);
		const auto roundTrip = RoundTripFloat32ThroughCUDA(sourceValues, dataType);
		const auto converted = ConvertCUDADeviceToFloat32(sourceValues, dataType);

		ASSERT_EQ(roundTrip.size(), expected.size());
		ASSERT_EQ(converted.size(), expected.size());
		for (auto i = 0uz; i < expected.size(); ++i)
		{
			EXPECT_FLOAT_EQ(roundTrip[i], expected[i]) << "roundTrip dtype=" << DataTypeName(dataType)
			                                         << ", index=" << i;
			EXPECT_FLOAT_EQ(converted[i], expected[i]) << "deviceConvert dtype=" << DataTypeName(dataType)
			                                          << ", index=" << i;
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

TEST(CUDADevice, Float8MatMulMatchesHighPrecisionQuantizedReference)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	constexpr auto m = 2uz;
	constexpr auto k = 3uz;
	constexpr auto n = 2uz;
	const std::vector<double> lhsValues{ 0.5, 1.5, -0.75, 2.25, 1.25, -1.0 };
	const std::vector<double> rhsValues{ 1.0, -0.5, 0.25, 2.0, -1.5, 1.25 };
	const std::array dataTypes{ DataType::Float8E4M3, DataType::Float8E5M2 };

	std::size_t executedCases = 0;
	for (const auto dataType : dataTypes)
	{
		if (!CUDASupportsNativeMatMul(dataType))
		{
			continue;
		}
		Tensor<CPU> lhs(std::span<const double>(lhsValues.data(), lhsValues.size()), { m, k }, dataType);
		Tensor<CPU> rhs(std::span<const double>(rhsValues.data(), rhsValues.size()), { k, n }, dataType);
		const auto expected = HighPrecisionMatMulReference(lhs, rhs, m, k, n);

		auto cudaResult = lhs.CopyToDevice(CUDA{}).MatMul(rhs.CopyToDevice(CUDA{})).CopyToDevice(CPU{});
		const auto actual = ReadAsFloat32(cudaResult);

		ASSERT_EQ(actual.size(), expected.size());
		for (auto i = 0uz; i < actual.size(); ++i)
		{
			EXPECT_FLOAT_EQ(actual[i], expected[i]) << "dtype=" << DataTypeName(dataType) << ", index=" << i;
		}
		++executedCases;
	}

	if (executedCases == 0)
	{
		GTEST_SKIP() << "CUDA device does not report native Float8 MatMul support";
	}
}

TEST(CUDADevice, Int8MatMulMatchesHighPrecisionQuantizedReference)
{
	if (!IsCUDADeviceAvailable())
	{
		GTEST_SKIP() << "CUDA device is not available";
	}

	constexpr auto m = 2uz;
	constexpr auto k = 3uz;
	constexpr auto n = 2uz;
	const std::array testCases{
		std::pair{ DataType::Int8, std::pair{ std::vector<double>{ 1, 2, -3, 4, -1, 2 },
		                                     std::vector<double>{ 2, -1, 1, 3, -2, 1 } } },
		std::pair{ DataType::UInt8, std::pair{ std::vector<double>{ 1, 2, 3, 4, 1, 2 },
		                                      std::vector<double>{ 2, 1, 1, 3, 2, 1 } } },
	};

	std::size_t executedCases = 0;
	for (const auto& [dataType, inputs] : testCases)
	{
		if (!CUDASupportsNativeMatMul(dataType))
		{
			continue;
		}
		const auto& [lhsValues, rhsValues] = inputs;
		Tensor<CPU> lhs(std::span<const double>(lhsValues.data(), lhsValues.size()), { m, k }, dataType);
		Tensor<CPU> rhs(std::span<const double>(rhsValues.data(), rhsValues.size()), { k, n }, dataType);
		const auto expected = HighPrecisionMatMulReference(lhs, rhs, m, k, n);

		auto cudaResult = lhs.CopyToDevice(CUDA{}).MatMul(rhs.CopyToDevice(CUDA{})).CopyToDevice(CPU{});
		const auto actual = ReadAsFloat32(cudaResult);

		ASSERT_EQ(actual.size(), expected.size());
		for (auto i = 0uz; i < actual.size(); ++i)
		{
			EXPECT_FLOAT_EQ(actual[i], expected[i]) << "dtype=" << DataTypeName(dataType) << ", index=" << i;
		}
		++executedCases;
	}

	if (executedCases == 0)
	{
		GTEST_SKIP() << "CUDA device does not report native Int8 MatMul support";
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
