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
