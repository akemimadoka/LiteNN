#include <gtest/gtest.h>

#include <LiteNN.h>

#include <array>

using namespace LiteNN;

TEST(VulkanDeviceTest, ReportsAvailability)
{
	const auto count = VulkanDeviceCount();
	EXPECT_EQ(IsVulkanDeviceAvailable(count), false);
	if (count != 0)
	{
		EXPECT_TRUE(IsVulkanDeviceAvailable(0));
	}
}

TEST(VulkanDeviceTest, CopiesHostVisibleTensorData)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	Vulkan device;
	Tensor<Vulkan> tensor({ 1.0, 2.5, -3.0, 4.0 }, { 4 }, DataType::Float32, device);
	Tensor<CPU> cpu(Uninitialized, { 4 }, DataType::Float32, CPU{});
	DeviceTraits<Vulkan>::CopyToCPU(tensor.CurDevice(), tensor.DType(), tensor.UnsafeRawData(), tensor.NumElements(),
	                                cpu.DType(), cpu.UnsafeRawData());

	const auto* values = static_cast<const float*>(cpu.UnsafeRawData());
	const std::array expected{ 1.0f, 2.5f, -3.0f, 4.0f };
	for (std::size_t i = 0; i < expected.size(); ++i)
	{
		EXPECT_FLOAT_EQ(values[i], expected[i]);
	}
}

TEST(VulkanDeviceTest, DeviceLocalTensorRejectsHostMapUntilStagingIsImplemented)
{
	if (!IsVulkanDeviceAvailable())
	{
		GTEST_SKIP() << "No Vulkan compute device is available";
	}

	Vulkan device;
	device.bufferResidency = VulkanBufferResidency::DeviceLocal;
	Tensor<Vulkan> tensor(Uninitialized, { 4 }, DataType::Float32, device);
	Tensor<CPU> cpu(Uninitialized, { 4 }, DataType::Float32, CPU{});

	EXPECT_THROW(DeviceTraits<Vulkan>::CopyToCPU(tensor.CurDevice(), tensor.DType(), tensor.UnsafeRawData(),
	                                             tensor.NumElements(), cpu.DType(), cpu.UnsafeRawData()),
	             std::runtime_error);
}
