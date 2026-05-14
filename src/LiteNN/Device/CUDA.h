#ifndef LITENN_DEVICE_CUDA_H
#define LITENN_DEVICE_CUDA_H

#include <cstdint>
#include <LiteNN/Device.h>
#include <memory>
#include <span>
#include <string>
#include <string_view>

#ifdef LITENN_ENABLE_CUDA

namespace LiteNN
{
	struct CUDA
	{
		int deviceIndex = 0;
		mutable std::string infoCache;

		bool operator==(const CUDA& other) const
		{
			return deviceIndex == other.deviceIndex;
		}
	};

	int CUDADeviceCount() noexcept;
	bool IsCUDADeviceAvailable(int deviceIndex = 0) noexcept;
	bool IsCUDADriverAvailable(int deviceIndex = 0) noexcept;

	struct CUDADriverLaunchDim
	{
		unsigned int x{ 1 };
		unsigned int y{ 1 };
		unsigned int z{ 1 };
	};

	struct CUDADriverLaunchOptions
	{
		CUDADriverLaunchDim grid;
		CUDADriverLaunchDim block;
		std::size_t sharedMemoryBytes{};
		void* stream{};
		bool synchronize{ true };
	};

	struct CUDAExecutionOptions
	{
		void* stream{};
		bool synchronize{ true };
	};

	class CUDADriverModule
	{
	public:
		CUDADriverModule();
		CUDADriverModule(CUDA device, std::span<const std::byte> image);
		CUDADriverModule(const CUDADriverModule&) = delete;
		CUDADriverModule& operator=(const CUDADriverModule&) = delete;
		CUDADriverModule(CUDADriverModule&&) noexcept;
		CUDADriverModule& operator=(CUDADriverModule&&) noexcept;
		~CUDADriverModule();

		bool Empty() const noexcept;
		void Launch(std::string_view functionName, const CUDADriverLaunchOptions& options,
		            std::span<void*> arguments = {}) const;

	private:
		struct Impl;

		std::unique_ptr<Impl> impl_;
	};

	template <>
	struct DeviceTraits<CUDA>
	{
		static consteval std::meta::info DataTypeMappingFunc(DataType dataType)
		{
			switch (dataType)
			{
			case DataType::Float32:
				return ^^float;
			case DataType::Float64:
				return ^^double;
			case DataType::Int32:
				return ^^int32_t;
			case DataType::Int64:
				return ^^int64_t;
			case DataType::Bool:
				return ^^bool;
			}
		}

		template <DataType DT>
		using DataTypeMapping = [:DataTypeMappingFunc(DT):];

		static std::string_view Name();
		static std::string_view Info(const CUDA& device);
		static void* Allocate(CUDA& device, DataType type, std::size_t size);
		static void Deallocate(CUDA& device, void* ptr, DataType type, std::size_t size);
		static void ZeroFill(CUDA& device, void* ptr, DataType type, std::size_t size);
		static void CopyToCPU(CUDA& device, DataType srcType, const void* src, std::size_t size, DataType dstType,
		                      void* dst);
		static void CopyToCPU(CUDA& device, DataType srcType, const void* src, std::size_t size, DataType dstType,
		                      void* dst, CUDAExecutionOptions options);
		static void CopyFromCPU(CUDA& device, DataType dstType, void* dst, DataType srcType, const void* src,
		                        std::size_t size);
		static void CopyFromCPU(CUDA& device, DataType dstType, void* dst, DataType srcType, const void* src,
		                        std::size_t size, CUDAExecutionOptions options);
		static void ConvertTo(CUDA& device, DataType srcType, const void* src, std::size_t size, DataType dstType,
		                      void* dst);
		static void DoUnaryOp(CUDA& device, UnaryOp unaryOp, void* dst, DataType type, ShapeView shape,
		                      const void* src);
		static void DoBinaryOp(CUDA& device, BinaryOp binaryOp, void* dst, DataType type1, ShapeView shape1,
		                       const void* src1, DataType type2, ShapeView shape2, const void* src2);
		static void DoBinaryOp(CUDA& device, BinaryOp binaryOp, void* dst, DataType type1, ShapeView shape1,
		                       const void* src1, DataType type2, ShapeView shape2, const void* src2,
		                       CUDAExecutionOptions options);
		static void DoReduceOp(CUDA& device, ReduceOp reduceOp, void* dst, DataType type, ShapeView shape,
		                       const void* src, std::size_t axis);
		static void DoConcatOp(CUDA& device, void* dst, DataType type, const void* const* srcPtrs,
		                       const ShapeView* srcShapes, std::size_t inputCount, std::size_t axis);
		static void DoSliceOp(CUDA& device, void* dst, DataType type, ShapeView srcShape, const void* src,
		                      std::size_t axis, std::size_t start, std::size_t length);
	};
} // namespace LiteNN

#endif

#endif
