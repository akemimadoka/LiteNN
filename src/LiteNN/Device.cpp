#include "Device.h"

namespace LiteNN
{
	PolymorphicDevice::Interface::~Interface() = default;

	std::string_view DeviceTraits<PolymorphicDevice>::Name()
	{
		return "PolymorphicDevice";
	}

	std::string_view DeviceTraits<PolymorphicDevice>::Info(const PolymorphicDevice& device)
	{
		device.infoCache_.clear();
		std::format_to(std::back_inserter(device.infoCache_), "PolymorphicDevice wrapping [{}: {}]",
		               device.impl_->Name(), device.impl_->Info());
		return device.infoCache_;
	}

	void* DeviceTraits<PolymorphicDevice>::Allocate(PolymorphicDevice& device, DataType type, std::size_t size)
	{
		return device.impl_->Allocate(type, size);
	}

	void DeviceTraits<PolymorphicDevice>::Deallocate(PolymorphicDevice& device, void* ptr, DataType type,
	                                                 std::size_t size)
	{
		device.impl_->Deallocate(ptr, type, size);
	}

	void DeviceTraits<PolymorphicDevice>::ZeroFill(PolymorphicDevice& device, void* ptr, DataType type,
	                                               std::size_t size)
	{
		device.impl_->ZeroFill(ptr, type, size);
	}

	void DeviceTraits<PolymorphicDevice>::CopyToCPU(PolymorphicDevice& device, DataType srcType, const void* src,
	                                                std::size_t size, DataType dstType, void* dst)
	{
		device.impl_->CopyToCPU(srcType, src, size, dstType, dst);
	}

	void DeviceTraits<PolymorphicDevice>::CopyFromCPU(PolymorphicDevice& device, DataType dstType, void* dst,
	                                                  DataType srcType, const void* src, std::size_t size)
	{
		device.impl_->CopyFromCPU(dstType, dst, srcType, src, size);
	}

	void DeviceTraits<PolymorphicDevice>::ConvertTo(PolymorphicDevice& device, DataType srcType, const void* src,
	                                                std::size_t size, DataType dstType, void* dst)
	{
		device.impl_->ConvertTo(srcType, src, size, dstType, dst);
	}

	void DeviceTraits<PolymorphicDevice>::DoUnaryOp(PolymorphicDevice& device, UnaryOp unaryOp, void* dst,
	                                                DataType type, ShapeView shape, const void* src)
	{
		device.impl_->DoUnaryOp(unaryOp, dst, type, shape, src);
	}

	void DeviceTraits<PolymorphicDevice>::DoBinaryOp(PolymorphicDevice& device, BinaryOp binaryOp, void* dst,
	                                                 DataType type1, ShapeView shape1, const void* src1, DataType type2,
	                                                 ShapeView shape2, const void* src2)
	{
		device.impl_->DoBinaryOp(binaryOp, dst, type1, shape1, src1, type2, shape2, src2);
	}

	void DeviceTraits<PolymorphicDevice>::DoReduceOp(PolymorphicDevice& device, ReduceOp reduceOp, void* dst,
	                                                 DataType type, ShapeView shape, const void* src, std::size_t axis)
	{
		device.impl_->DoReduceOp(reduceOp, dst, type, shape, src, axis);
	}

	void DeviceTraits<PolymorphicDevice>::DoConcatOp(PolymorphicDevice& device, void* dst, DataType type,
	                                                 const void* const* srcPtrs, const ShapeView* srcShapes,
	                                                 std::size_t inputCount, std::size_t axis)
	{
		device.impl_->DoConcatOp(dst, type, srcPtrs, srcShapes, inputCount, axis);
	}

	void DeviceTraits<PolymorphicDevice>::DoSliceOp(PolymorphicDevice& device, void* dst, DataType type,
	                                                ShapeView srcShape, const void* src, std::size_t axis,
	                                                std::size_t start, std::size_t length)
	{
		device.impl_->DoSliceOp(dst, type, srcShape, src, axis, start, length);
	}
} // namespace LiteNN
