#ifndef LITENN_MODULE_IMPL
#include "CUDA.h"
#endif

#ifdef LITENN_ENABLE_CUDA

#include <cuda_runtime_api.h>

#include <cstring>
#include <format>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace LiteNN
{
	namespace
	{
		std::size_t ElementSize(DataType type)
		{
			switch (type)
			{
			case DataType::Float32:
				return sizeof(float);
			case DataType::Float64:
				return sizeof(double);
			case DataType::Int32:
				return sizeof(int32_t);
			case DataType::Int64:
				return sizeof(int64_t);
			case DataType::Bool:
				return sizeof(bool);
			}
			throw std::runtime_error("Invalid DataType");
		}

		std::vector<std::byte> MakeHostBuffer(DataType type, std::size_t size)
		{
			return std::vector<std::byte>(ElementSize(type) * size);
		}

		void CheckCUDA(cudaError_t status, std::string_view action)
		{
			if (status != cudaSuccess)
			{
				throw std::runtime_error(std::format("{} failed: {}", action, cudaGetErrorString(status)));
			}
		}

		void ClearCUDAError() noexcept
		{
			(void)cudaGetLastError();
		}

		class CUDADeviceGuard
		{
		public:
			explicit CUDADeviceGuard(int deviceIndex) : previousDevice_(0), changed_(false)
			{
				CheckCUDA(cudaGetDevice(&previousDevice_), "cudaGetDevice");
				if (previousDevice_ != deviceIndex)
				{
					CheckCUDA(cudaSetDevice(deviceIndex), "cudaSetDevice");
					changed_ = true;
				}
			}

			~CUDADeviceGuard()
			{
				if (changed_)
				{
					(void)cudaSetDevice(previousDevice_);
				}
			}

			CUDADeviceGuard(const CUDADeviceGuard&) = delete;
			CUDADeviceGuard& operator=(const CUDADeviceGuard&) = delete;

		private:
			int previousDevice_;
			bool changed_;
		};

		DataType ResolveUnaryResultType(UnaryOp op, DataType inputType)
		{
			return EnumDispatch(op, [&]<UnaryOp OpValue> {
				const auto resultType = UnaryOpTraits<OpValue>::ResultType(inputType);
				if (!resultType)
				{
					throw std::runtime_error(resultType.error());
				}
				return *resultType;
			});
		}

		std::vector<std::size_t> ResolveUnaryResultShape(UnaryOp op, ShapeView inputShape)
		{
			return EnumDispatch(op, [&]<UnaryOp OpValue> {
				auto resultShape = UnaryOpTraits<OpValue>::ResultShape(inputShape);
				if (!resultShape)
				{
					throw std::runtime_error(resultShape.error());
				}
				return *resultShape;
			});
		}

		DataType ResolveBinaryResultType(BinaryOp op, DataType type1, DataType type2)
		{
			return EnumDispatch(op, [&]<BinaryOp OpValue> {
				const auto resultType = BinaryOpTraits<OpValue>::ResultType(type1, type2);
				if (!resultType)
				{
					throw std::runtime_error(resultType.error());
				}
				return *resultType;
			});
		}

		std::vector<std::size_t> ResolveBinaryResultShape(BinaryOp op, ShapeView shape1, ShapeView shape2)
		{
			return EnumDispatch(op, [&]<BinaryOp OpValue> {
				auto resultShape = BinaryOpTraits<OpValue>::ResultShape(shape1, shape2);
				if (!resultShape)
				{
					throw std::runtime_error(resultShape.error());
				}
				return *resultShape;
			});
		}

		DataType ResolveReduceResultType(ReduceOp op, DataType inputType)
		{
			return EnumDispatch(op, [&]<ReduceOp OpValue> {
				const auto resultType = ReduceOpTraits<OpValue>::ResultType(inputType);
				if (!resultType)
				{
					throw std::runtime_error(resultType.error());
				}
				return *resultType;
			});
		}

		std::vector<std::size_t> ResolveReduceResultShape(ReduceOp op, ShapeView inputShape, std::size_t axis)
		{
			return EnumDispatch(op, [&]<ReduceOp OpValue> {
				auto resultShape = ReduceOpTraits<OpValue>::ResultShape(inputShape, axis);
				if (!resultShape)
				{
					throw std::runtime_error(resultShape.error());
				}
				return *resultShape;
			});
		}

		std::vector<std::size_t> ResolveConcatResultShape(const ShapeView* srcShapes, std::size_t inputCount,
		                                                  std::size_t axis)
		{
			if (inputCount == 0)
			{
				throw std::runtime_error("Concat requires at least one input");
			}
			if (axis >= srcShapes[0].NumDim())
			{
				throw std::runtime_error("Concat axis out of range");
			}

			auto resultShape = srcShapes[0].ToOwned();
			resultShape[axis] = 0;
			for (auto i = 0uz; i < inputCount; ++i)
			{
				if (srcShapes[i].NumDim() != srcShapes[0].NumDim())
				{
					throw std::runtime_error("Concat requires all inputs to have the same rank");
				}
				for (auto dim = 0uz; dim < srcShapes[0].NumDim(); ++dim)
				{
					if (dim != axis && srcShapes[i][dim] != srcShapes[0][dim])
					{
						throw std::runtime_error("Concat requires non-axis dimensions to match");
					}
				}
				resultShape[axis] += srcShapes[i][axis];
			}
			return resultShape;
		}

		std::vector<std::size_t> ResolveSliceResultShape(ShapeView srcShape, std::size_t axis, std::size_t start,
		                                                 std::size_t length)
		{
			if (axis >= srcShape.NumDim())
			{
				throw std::runtime_error("Slice axis out of range");
			}
			if (start + length > srcShape[axis])
			{
				throw std::runtime_error("Slice range out of bounds");
			}
			auto resultShape = srcShape.ToOwned();
			resultShape[axis] = length;
			return resultShape;
		}
	} // namespace

	int CUDADeviceCount() noexcept
	{
		int count = 0;
		const auto status = cudaGetDeviceCount(&count);
		if (status != cudaSuccess)
		{
			ClearCUDAError();
			return 0;
		}
		return count;
	}

	bool IsCUDADeviceAvailable(int deviceIndex) noexcept
	{
		const auto count = CUDADeviceCount();
		return deviceIndex >= 0 && deviceIndex < count;
	}

	std::string_view DeviceTraits<CUDA>::Name()
	{
		return "CUDA";
	}

	std::string_view DeviceTraits<CUDA>::Info(const CUDA& device)
	{
		device.infoCache.clear();
		cudaDeviceProp properties{};
		const auto status = cudaGetDeviceProperties(&properties, device.deviceIndex);
		if (status != cudaSuccess)
		{
			ClearCUDAError();
			std::format_to(std::back_inserter(device.infoCache), "Unavailable CUDA Device {}", device.deviceIndex);
			return device.infoCache;
		}

		std::format_to(std::back_inserter(device.infoCache), "{} (device {}, cc {}.{}, {} MiB global memory)",
		               properties.name, device.deviceIndex, properties.major, properties.minor,
		               static_cast<unsigned long long>(properties.totalGlobalMem / (1024 * 1024)));
		return device.infoCache;
	}

	void* DeviceTraits<CUDA>::Allocate(CUDA& device, DataType type, std::size_t size)
	{
		CUDADeviceGuard guard(device.deviceIndex);
		void* ptr = nullptr;
		CheckCUDA(cudaMalloc(&ptr, ElementSize(type) * size), "cudaMalloc");
		return ptr;
	}

	void DeviceTraits<CUDA>::Deallocate(CUDA& device, void* ptr, DataType type, std::size_t size)
	{
		(void)type;
		(void)size;
		if (ptr == nullptr)
		{
			return;
		}
		CUDADeviceGuard guard(device.deviceIndex);
		CheckCUDA(cudaFree(ptr), "cudaFree");
	}

	void DeviceTraits<CUDA>::ZeroFill(CUDA& device, void* ptr, DataType type, std::size_t size)
	{
		CUDADeviceGuard guard(device.deviceIndex);
		CheckCUDA(cudaMemset(ptr, 0, ElementSize(type) * size), "cudaMemset");
	}

	void DeviceTraits<CUDA>::CopyToCPU(CUDA& device, DataType srcType, const void* src, std::size_t size,
	                                   DataType dstType, void* dst)
	{
		CUDADeviceGuard guard(device.deviceIndex);
		if (srcType == dstType)
		{
			CheckCUDA(cudaMemcpy(dst, src, ElementSize(srcType) * size, cudaMemcpyDeviceToHost), "cudaMemcpy D2H");
			return;
		}

		auto hostSrc = MakeHostBuffer(srcType, size);
		CheckCUDA(cudaMemcpy(hostSrc.data(), src, ElementSize(srcType) * size, cudaMemcpyDeviceToHost),
		          "cudaMemcpy D2H");
		CPU cpu;
		DeviceTraits<CPU>::ConvertTo(cpu, srcType, hostSrc.data(), size, dstType, dst);
	}

	void DeviceTraits<CUDA>::CopyFromCPU(CUDA& device, DataType dstType, void* dst, DataType srcType,
	                                     const void* src, std::size_t size)
	{
		CUDADeviceGuard guard(device.deviceIndex);
		if (srcType == dstType)
		{
			CheckCUDA(cudaMemcpy(dst, src, ElementSize(dstType) * size, cudaMemcpyHostToDevice), "cudaMemcpy H2D");
			return;
		}

		auto hostDst = MakeHostBuffer(dstType, size);
		CPU cpu;
		DeviceTraits<CPU>::ConvertTo(cpu, srcType, src, size, dstType, hostDst.data());
		CheckCUDA(cudaMemcpy(dst, hostDst.data(), ElementSize(dstType) * size, cudaMemcpyHostToDevice),
		          "cudaMemcpy H2D");
	}

	void DeviceTraits<CUDA>::ConvertTo(CUDA& device, DataType srcType, const void* src, std::size_t size,
	                                   DataType dstType, void* dst)
	{
		CUDADeviceGuard guard(device.deviceIndex);
		if (srcType == dstType)
		{
			CheckCUDA(cudaMemcpy(dst, src, ElementSize(srcType) * size, cudaMemcpyDeviceToDevice),
			          "cudaMemcpy D2D");
			return;
		}

		auto hostSrc = MakeHostBuffer(srcType, size);
		auto hostDst = MakeHostBuffer(dstType, size);
		CheckCUDA(cudaMemcpy(hostSrc.data(), src, ElementSize(srcType) * size, cudaMemcpyDeviceToHost),
		          "cudaMemcpy D2H");
		CPU cpu;
		DeviceTraits<CPU>::ConvertTo(cpu, srcType, hostSrc.data(), size, dstType, hostDst.data());
		CheckCUDA(cudaMemcpy(dst, hostDst.data(), ElementSize(dstType) * size, cudaMemcpyHostToDevice),
		          "cudaMemcpy H2D");
	}

	void DeviceTraits<CUDA>::DoUnaryOp(CUDA& device, UnaryOp unaryOp, void* dst, DataType type, ShapeView shape,
	                                   const void* src)
	{
		const auto resultType = ResolveUnaryResultType(unaryOp, type);
		const auto resultShape = ResolveUnaryResultShape(unaryOp, shape);
		auto hostSrc = MakeHostBuffer(type, shape.NumElements());
		auto hostDst = MakeHostBuffer(resultType, ShapeView{ resultShape }.NumElements());
		CopyToCPU(device, type, src, shape.NumElements(), type, hostSrc.data());

		CPU cpu;
		DeviceTraits<CPU>::DoUnaryOp(cpu, unaryOp, hostDst.data(), type, shape, hostSrc.data());
		CopyFromCPU(device, resultType, dst, resultType, hostDst.data(), ShapeView{ resultShape }.NumElements());
	}

	void DeviceTraits<CUDA>::DoBinaryOp(CUDA& device, BinaryOp binaryOp, void* dst, DataType type1, ShapeView shape1,
	                                    const void* src1, DataType type2, ShapeView shape2, const void* src2)
	{
		const auto resultType = ResolveBinaryResultType(binaryOp, type1, type2);
		const auto resultShape = ResolveBinaryResultShape(binaryOp, shape1, shape2);
		auto hostSrc1 = MakeHostBuffer(type1, shape1.NumElements());
		auto hostSrc2 = MakeHostBuffer(type2, shape2.NumElements());
		auto hostDst = MakeHostBuffer(resultType, ShapeView{ resultShape }.NumElements());
		CopyToCPU(device, type1, src1, shape1.NumElements(), type1, hostSrc1.data());
		CopyToCPU(device, type2, src2, shape2.NumElements(), type2, hostSrc2.data());

		CPU cpu;
		DeviceTraits<CPU>::DoBinaryOp(cpu, binaryOp, hostDst.data(), type1, shape1, hostSrc1.data(), type2, shape2,
		                              hostSrc2.data());
		CopyFromCPU(device, resultType, dst, resultType, hostDst.data(), ShapeView{ resultShape }.NumElements());
	}

	void DeviceTraits<CUDA>::DoReduceOp(CUDA& device, ReduceOp reduceOp, void* dst, DataType type, ShapeView shape,
	                                    const void* src, std::size_t axis)
	{
		const auto resultType = ResolveReduceResultType(reduceOp, type);
		const auto resultShape = ResolveReduceResultShape(reduceOp, shape, axis);
		auto hostSrc = MakeHostBuffer(type, shape.NumElements());
		auto hostDst = MakeHostBuffer(resultType, ShapeView{ resultShape }.NumElements());
		CopyToCPU(device, type, src, shape.NumElements(), type, hostSrc.data());

		CPU cpu;
		DeviceTraits<CPU>::DoReduceOp(cpu, reduceOp, hostDst.data(), type, shape, hostSrc.data(), axis);
		CopyFromCPU(device, resultType, dst, resultType, hostDst.data(), ShapeView{ resultShape }.NumElements());
	}

	void DeviceTraits<CUDA>::DoConcatOp(CUDA& device, void* dst, DataType type, const void* const* srcPtrs,
	                                    const ShapeView* srcShapes, std::size_t inputCount, std::size_t axis)
	{
		const auto resultShape = ResolveConcatResultShape(srcShapes, inputCount, axis);
		std::vector<std::vector<std::byte>> hostInputs;
		std::vector<const void*> hostPtrs;
		hostInputs.reserve(inputCount);
		hostPtrs.reserve(inputCount);
		for (auto i = 0uz; i < inputCount; ++i)
		{
			auto& hostInput = hostInputs.emplace_back(MakeHostBuffer(type, srcShapes[i].NumElements()));
			CopyToCPU(device, type, srcPtrs[i], srcShapes[i].NumElements(), type, hostInput.data());
			hostPtrs.push_back(hostInput.data());
		}
		auto hostDst = MakeHostBuffer(type, ShapeView{ resultShape }.NumElements());

		CPU cpu;
		DeviceTraits<CPU>::DoConcatOp(cpu, hostDst.data(), type, hostPtrs.data(), srcShapes, inputCount, axis);
		CopyFromCPU(device, type, dst, type, hostDst.data(), ShapeView{ resultShape }.NumElements());
	}

	void DeviceTraits<CUDA>::DoSliceOp(CUDA& device, void* dst, DataType type, ShapeView srcShape, const void* src,
	                                   std::size_t axis, std::size_t start, std::size_t length)
	{
		const auto resultShape = ResolveSliceResultShape(srcShape, axis, start, length);
		auto hostSrc = MakeHostBuffer(type, srcShape.NumElements());
		auto hostDst = MakeHostBuffer(type, ShapeView{ resultShape }.NumElements());
		CopyToCPU(device, type, src, srcShape.NumElements(), type, hostSrc.data());

		CPU cpu;
		DeviceTraits<CPU>::DoSliceOp(cpu, hostDst.data(), type, srcShape, hostSrc.data(), axis, start, length);
		CopyFromCPU(device, type, dst, type, hostDst.data(), ShapeView{ resultShape }.NumElements());
	}
} // namespace LiteNN

#endif
