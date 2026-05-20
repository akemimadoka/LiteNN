#include <gtest/gtest.h>

#include <LiteNN.h>
#include <initializer_list>

using namespace LiteNN;

// 辅助函数: 读取 CPU Tensor 的第 i 个 float 元素
static float ReadFloat(const Tensor<CPU>& t, std::size_t i)
{
	return static_cast<const float*>(t.RawData())[i];
}

static bool ReadBool(const Tensor<CPU>& t, std::size_t i)
{
	return static_cast<const bool*>(t.RawData())[i];
}

static float ReadAsFloat(const Tensor<CPU>& t, std::size_t i)
{
	Tensor<CPU> converted(Uninitialized, t.Shape(), DataType::Float32);
	CPU cpu;
	DeviceTraits<CPU>::ConvertTo(cpu, t.DType(), t.RawData(), t.NumElements(), DataType::Float32,
	                             converted.RawData());
	return ReadFloat(converted, i);
}

TEST(DataType, LowPrecisionMetadata)
{
	EXPECT_TRUE(IsValidDataTypeValue(DataType::Float16));
	EXPECT_TRUE(IsValidDataTypeValue(DataType::BFloat16));
	EXPECT_TRUE(IsValidDataTypeValue(DataType::Float8E4M3));
	EXPECT_TRUE(IsValidDataTypeValue(DataType::Float8E5M2));
	EXPECT_TRUE(IsValidDataTypeValue(DataType::Int8));
	EXPECT_TRUE(IsValidDataTypeValue(DataType::UInt8));
	EXPECT_TRUE(IsFloatingDataType(DataType::Float16));
	EXPECT_TRUE(IsFloatingDataType(DataType::BFloat16));
	EXPECT_TRUE(IsFloatingDataType(DataType::Float8E4M3));
	EXPECT_TRUE(IsFloatingDataType(DataType::Float8E5M2));
	EXPECT_FALSE(IsFloatingDataType(DataType::Int8));
	EXPECT_FALSE(IsFloatingDataType(DataType::UInt8));
	EXPECT_EQ(ElementByteSize(DataType::Float16), 2);
	EXPECT_EQ(ElementByteSize(DataType::BFloat16), 2);
	EXPECT_EQ(ElementByteSize(DataType::Float8E4M3), 1);
	EXPECT_EQ(ElementByteSize(DataType::Float8E5M2), 1);
	EXPECT_EQ(ElementByteSize(DataType::Int8), 1);
	EXPECT_EQ(ElementByteSize(DataType::UInt8), 1);
	EXPECT_EQ(DataTypeName(DataType::Float16), "Float16");
	EXPECT_EQ(DataTypeName(DataType::BFloat16), "BFloat16");
	EXPECT_EQ(DataTypeName(DataType::Float8E4M3), "Float8E4M3");
	EXPECT_EQ(DataTypeName(DataType::Float8E5M2), "Float8E5M2");
}

TEST(Tensor, LowPrecisionCPUConversionRoundTrip)
{
	const Tensor<CPU> f16({ 1.0, -2.5, 0.5 }, { 3 }, DataType::Float16);
	EXPECT_EQ(f16.DType(), DataType::Float16);
	EXPECT_NEAR(ReadAsFloat(f16, 0), 1.0F, 1e-3F);
	EXPECT_NEAR(ReadAsFloat(f16, 1), -2.5F, 1e-3F);
	EXPECT_NEAR(ReadAsFloat(f16, 2), 0.5F, 1e-3F);

	const Tensor<CPU> bf16({ 1.0, -2.5, 0.5 }, { 3 }, DataType::BFloat16);
	EXPECT_EQ(bf16.DType(), DataType::BFloat16);
	EXPECT_NEAR(ReadAsFloat(bf16, 0), 1.0F, 1e-2F);
	EXPECT_NEAR(ReadAsFloat(bf16, 1), -2.5F, 1e-2F);
	EXPECT_NEAR(ReadAsFloat(bf16, 2), 0.5F, 1e-2F);

	const Tensor<CPU> e4m3({ 1.0, -2.0, 0.5 }, { 3 }, DataType::Float8E4M3);
	EXPECT_EQ(e4m3.DType(), DataType::Float8E4M3);
	EXPECT_NEAR(ReadAsFloat(e4m3, 0), 1.0F, 0.25F);
	EXPECT_NEAR(ReadAsFloat(e4m3, 1), -2.0F, 0.25F);
	EXPECT_NEAR(ReadAsFloat(e4m3, 2), 0.5F, 0.25F);

	const Tensor<CPU> e5m2({ 1.0, -2.0, 0.5 }, { 3 }, DataType::Float8E5M2);
	EXPECT_EQ(e5m2.DType(), DataType::Float8E5M2);
	EXPECT_NEAR(ReadAsFloat(e5m2, 0), 1.0F, 0.25F);
	EXPECT_NEAR(ReadAsFloat(e5m2, 1), -2.0F, 0.25F);
	EXPECT_NEAR(ReadAsFloat(e5m2, 2), 0.5F, 0.25F);

	const Tensor<CPU> i8({ -2.0, 0.0, 3.0 }, { 3 }, DataType::Int8);
	EXPECT_FLOAT_EQ(ReadAsFloat(i8, 0), -2.0F);
	EXPECT_FLOAT_EQ(ReadAsFloat(i8, 1), 0.0F);
	EXPECT_FLOAT_EQ(ReadAsFloat(i8, 2), 3.0F);

	const Tensor<CPU> u8({ 2.0, 0.0, 7.0 }, { 3 }, DataType::UInt8);
	EXPECT_FLOAT_EQ(ReadAsFloat(u8, 0), 2.0F);
	EXPECT_FLOAT_EQ(ReadAsFloat(u8, 1), 0.0F);
	EXPECT_FLOAT_EQ(ReadAsFloat(u8, 2), 7.0F);
}

TEST(Tensor, LowPrecisionArithmeticUsesCPUReference)
{
	const Tensor<CPU> lhs({ 1.0, 2.0, 3.0 }, { 3 }, DataType::Float16);
	const Tensor<CPU> rhs({ 0.5, 1.0, 1.5 }, { 3 }, DataType::Float16);
	const auto sum = lhs + rhs;
	EXPECT_EQ(sum.DType(), DataType::Float16);
	EXPECT_NEAR(ReadAsFloat(sum, 0), 1.5F, 1e-3F);
	EXPECT_NEAR(ReadAsFloat(sum, 1), 3.0F, 1e-3F);
	EXPECT_NEAR(ReadAsFloat(sum, 2), 4.5F, 1e-3F);
}

TEST(Initializer, LowPrecisionFloatingInitializers)
{
	std::mt19937 rng(1234);
	const auto f16 = Initializer::Uniform({ 4 }, -1.0, 1.0, rng, DataType::Float16);
	EXPECT_EQ(f16.DType(), DataType::Float16);
	const auto bf16 = Initializer::Ones({ 2 }, DataType::BFloat16);
	EXPECT_EQ(bf16.DType(), DataType::BFloat16);
	EXPECT_NEAR(ReadAsFloat(bf16, 0), 1.0F, 1e-2F);
	EXPECT_THROW((void)Initializer::Uniform({ 4 }, -1.0, 1.0, rng, DataType::Int8), std::runtime_error);
}

// tensor = [[1,2,3],[4,5,6]]
TEST(Tensor, BasicOperations)
{
	Tensor<CPU> tensor({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });

	// view = tensor[0] = [1,2,3]
	const auto view = tensor[0];
	EXPECT_FLOAT_EQ(ReadFloat(view, 0), 1);
	EXPECT_FLOAT_EQ(ReadFloat(view, 1), 2);
	EXPECT_FLOAT_EQ(ReadFloat(view, 2), 3);

	// addResult = tensor + tensor = [[2,4,6],[8,10,12]]
	const auto addResult = tensor + tensor;
	EXPECT_FLOAT_EQ(ReadFloat(addResult, 0), 2);
	EXPECT_FLOAT_EQ(ReadFloat(addResult, 1), 4);
	EXPECT_FLOAT_EQ(ReadFloat(addResult, 2), 6);
	EXPECT_FLOAT_EQ(ReadFloat(addResult, 3), 8);
	EXPECT_FLOAT_EQ(ReadFloat(addResult, 4), 10);
	EXPECT_FLOAT_EQ(ReadFloat(addResult, 5), 12);

	// transposeResult = [[1,4],[2,5],[3,6]]
	const auto transposeResult = tensor.Transpose();
	EXPECT_FLOAT_EQ(ReadFloat(transposeResult, 0), 1);
	EXPECT_FLOAT_EQ(ReadFloat(transposeResult, 1), 4);
	EXPECT_FLOAT_EQ(ReadFloat(transposeResult, 2), 2);
	EXPECT_FLOAT_EQ(ReadFloat(transposeResult, 3), 5);
	EXPECT_FLOAT_EQ(ReadFloat(transposeResult, 4), 3);
	EXPECT_FLOAT_EQ(ReadFloat(transposeResult, 5), 6);

	// matmulResult = tensor @ transposeResult = [[1,2,3],[4,5,6]] @ [[1,4],[2,5],[3,6]]
	// = [[14,32],[32,77]]
	const auto matmulResult = tensor.MatMul(transposeResult);
	EXPECT_FLOAT_EQ(ReadFloat(matmulResult, 0), 14);
	EXPECT_FLOAT_EQ(ReadFloat(matmulResult, 1), 32);
	EXPECT_FLOAT_EQ(ReadFloat(matmulResult, 2), 32);
	EXPECT_FLOAT_EQ(ReadFloat(matmulResult, 3), 77);
}

TEST(Tensor, ComparisonAndLogical)
{
	Tensor<CPU> tensor({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	const auto addResult = tensor + tensor; // [[2,4,6],[8,10,12]]

	// tensor < addResult: 全部 true (每个元素 x < 2x)
	const auto comparisonResult = tensor < addResult;
	for (auto i = 0uz; i < 6; ++i)
	{
		EXPECT_TRUE(ReadBool(comparisonResult, i));
	}

	// !comparisonResult: 全部 false
	const auto logicalNegationResult = !comparisonResult;
	for (auto i = 0uz; i < 6; ++i)
	{
		EXPECT_FALSE(ReadBool(logicalNegationResult, i));
	}

	// tensor == addResult: 全部 false (x != 2x, x>0)
	const auto equalityResult = tensor == addResult;
	for (auto i = 0uz; i < 6; ++i)
	{
		EXPECT_FALSE(ReadBool(equalityResult, i));
	}
}

TEST(Tensor, Broadcast)
{
	Tensor<CPU> tensor({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	Tensor<CPU> tensor2({ 10, 20 }, { 2, 1 });

	// tensor + tensor2: [[1+10,2+10,3+10],[4+20,5+20,6+20]] = [[11,12,13],[24,25,26]]
	const auto addResult2 = tensor + tensor2;
	EXPECT_FLOAT_EQ(ReadFloat(addResult2, 0), 11);
	EXPECT_FLOAT_EQ(ReadFloat(addResult2, 1), 12);
	EXPECT_FLOAT_EQ(ReadFloat(addResult2, 2), 13);
	EXPECT_FLOAT_EQ(ReadFloat(addResult2, 3), 24);
	EXPECT_FLOAT_EQ(ReadFloat(addResult2, 4), 25);
	EXPECT_FLOAT_EQ(ReadFloat(addResult2, 5), 26);
}

TEST(Tensor, CopyToDevice)
{
	Tensor<CPU> tensor({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });

	const auto polyTensor = tensor.CopyToDevice(PolymorphicDevice{ CPU{} });
	// 复制回 CPU 验证数据一致
	const auto backToCpu = polyTensor.CopyToDevice(CPU{});
	for (auto i = 0uz; i < 6; ++i)
	{
		EXPECT_FLOAT_EQ(ReadFloat(backToCpu, i), ReadFloat(tensor, i));
	}
}

TEST(Device, SameDeviceInstance)
{
	PolymorphicDevice dev1{ CPU{} };
	PolymorphicDevice dev2{ CPU{} };
	EXPECT_TRUE(dev1.IsSameDevice(dev2));
}

TEST(Device, SameDeviceType)
{
	PolymorphicDevice dev1{ CPU{} };
	PolymorphicDevice dev2{ CPU{} };
	EXPECT_TRUE(dev1.IsSameDeviceType(dev2));
}

TEST(Tensor, Strides)
{
	Tensor<CPU> tensor({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });

	auto strides = tensor.Strides();
	ASSERT_EQ(strides.size(), 2);
	EXPECT_EQ(strides[0], 3);
	EXPECT_EQ(strides[1], 1);
	EXPECT_TRUE(tensor.IsContiguous());
}

TEST(Tensor, StridesAfterReshape)
{
	Tensor<CPU> tensor({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	tensor.Reshape({ 3, 2 });

	auto strides = tensor.Strides();
	ASSERT_EQ(strides.size(), 2);
	EXPECT_EQ(strides[0], 2);
	EXPECT_EQ(strides[1], 1);
	EXPECT_TRUE(tensor.IsContiguous());
}

TEST(Tensor, ViewStrides)
{
	Tensor<CPU> tensor({ 1, 2, 3, 4, 5, 6 }, { 2, 3 });
	auto view = tensor[0]; // shape {3}

	auto strides = view.Strides();
	ASSERT_EQ(strides.size(), 1);
	EXPECT_EQ(strides[0], 1);
	EXPECT_TRUE(view.IsContiguous());
}
