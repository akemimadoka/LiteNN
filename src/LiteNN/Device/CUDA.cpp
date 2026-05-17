#ifndef LITENN_MODULE_IMPL
#include "CUDA.h"
#endif

#ifdef LITENN_ENABLE_CUDA

#include <cublas_v2.h>
#ifdef LITENN_ENABLE_CUBLASLT
#include <cublasLt.h>
#endif
#include <cuda_runtime_api.h>
#ifdef LITENN_ENABLE_CUDA_DRIVER
#include <cuda.h>
#endif

#include <array>
#include <cstdlib>
#include <cstring>
#include <format>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace LiteNN
{
	namespace
	{
		std::size_t ElementSize(DataType type)
		{
			return ElementByteSize(type);
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

		std::string_view CUBLASStatusName(cublasStatus_t status)
		{
			switch (status)
			{
			case CUBLAS_STATUS_SUCCESS:
				return "CUBLAS_STATUS_SUCCESS";
			case CUBLAS_STATUS_NOT_INITIALIZED:
				return "CUBLAS_STATUS_NOT_INITIALIZED";
			case CUBLAS_STATUS_ALLOC_FAILED:
				return "CUBLAS_STATUS_ALLOC_FAILED";
			case CUBLAS_STATUS_INVALID_VALUE:
				return "CUBLAS_STATUS_INVALID_VALUE";
			case CUBLAS_STATUS_ARCH_MISMATCH:
				return "CUBLAS_STATUS_ARCH_MISMATCH";
			case CUBLAS_STATUS_MAPPING_ERROR:
				return "CUBLAS_STATUS_MAPPING_ERROR";
			case CUBLAS_STATUS_EXECUTION_FAILED:
				return "CUBLAS_STATUS_EXECUTION_FAILED";
			case CUBLAS_STATUS_INTERNAL_ERROR:
				return "CUBLAS_STATUS_INTERNAL_ERROR";
			case CUBLAS_STATUS_NOT_SUPPORTED:
				return "CUBLAS_STATUS_NOT_SUPPORTED";
			case CUBLAS_STATUS_LICENSE_ERROR:
				return "CUBLAS_STATUS_LICENSE_ERROR";
			}
			return "CUBLAS_STATUS_UNKNOWN";
		}

		void CheckCUBLAS(cublasStatus_t status, std::string_view action)
		{
			if (status != CUBLAS_STATUS_SUCCESS)
			{
				throw std::runtime_error(std::format("{} failed: {}", action, CUBLASStatusName(status)));
			}
		}

#ifdef LITENN_ENABLE_CUDA_DRIVER
		std::string CUDADriverStatusMessage(CUresult status)
		{
			const char* name = nullptr;
			const char* message = nullptr;
			(void)cuGetErrorName(status, &name);
			(void)cuGetErrorString(status, &message);
			if (name != nullptr && message != nullptr)
			{
				return std::format("{} ({})", name, message);
			}
			if (name != nullptr)
			{
				return name;
			}
			return std::format("CUresult({})", static_cast<int>(status));
		}

		void CheckCUDADriver(CUresult status, std::string_view action)
		{
			if (status != CUDA_SUCCESS)
			{
				throw std::runtime_error(std::format("{} failed: {}", action, CUDADriverStatusMessage(status)));
			}
		}

		void CheckCUDADriver(CUresult status, std::string_view action, std::string_view detail)
		{
			if (status == CUDA_SUCCESS)
			{
				return;
			}
			auto message = std::format("{} failed: {}", action, CUDADriverStatusMessage(status));
			if (!detail.empty())
			{
				message += std::format("\n{}", detail);
			}
			throw std::runtime_error(message);
		}

		void InitializeCUDADriver()
		{
			CheckCUDADriver(cuInit(0), "cuInit");
		}

		class CUDADriverContextScope
		{
		public:
			explicit CUDADriverContextScope(CUcontext context) : context_(context)
			{
				CheckCUDADriver(cuCtxPushCurrent(context_), "cuCtxPushCurrent");
				pushed_ = true;
			}

			~CUDADriverContextScope()
			{
				if (pushed_)
				{
					CUcontext popped{};
					(void)cuCtxPopCurrent(&popped);
				}
			}

			CUDADriverContextScope(const CUDADriverContextScope&) = delete;
			CUDADriverContextScope& operator=(const CUDADriverContextScope&) = delete;

		private:
			CUcontext context_{};
			bool pushed_{};
		};
#endif

		void ClearCUDAError() noexcept
		{
			(void)cudaGetLastError();
		}

		constexpr bool BuildHasCUBLASLt() noexcept
		{
#ifdef LITENN_ENABLE_CUBLASLT
			return true;
#else
			return false;
#endif
		}

		constexpr bool BuildHasCUBLASBFloat16() noexcept
		{
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
			return true;
#else
			return false;
#endif
		}

		int ComputeCapabilityScore(int major, int minor) noexcept
		{
			return major * 10 + minor;
		}

		CUDALowPrecisionCapabilities MakeLowPrecisionCapabilities(int deviceIndex,
		                                                           const cudaDeviceProp& properties) noexcept
		{
			const auto cc = ComputeCapabilityScore(properties.major, properties.minor);
			return {
				.deviceIndex = deviceIndex,
				.computeCapabilityMajor = properties.major,
				.computeCapabilityMinor = properties.minor,
				.hasCUBLASLt = BuildHasCUBLASLt(),
				.supportsFloat16Storage = true,
				.supportsFloat16MatMul = cc >= 53,
				.supportsBFloat16Storage = true,
				.supportsBFloat16MatMul = cc >= 80 && BuildHasCUBLASBFloat16(),
				.supportsFloat8Storage = true,
				.supportsFloat8MatMul = false,
				.supportsInt8Storage = true,
				.supportsInt8TensorCores = cc >= 75,
			};
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

		class CUBLASHandle
		{
		public:
			explicit CUBLASHandle(int deviceIndex) : deviceIndex_(deviceIndex)
			{
				CUDADeviceGuard guard(deviceIndex_);
				CheckCUBLAS(cublasCreate(&handle_), "cublasCreate");
			}

			~CUBLASHandle()
			{
				if (handle_ != nullptr)
				{
					(void)cublasDestroy(handle_);
				}
			}

			CUBLASHandle(const CUBLASHandle&) = delete;
			CUBLASHandle& operator=(const CUBLASHandle&) = delete;

			int DeviceIndex() const noexcept
			{
				return deviceIndex_;
			}

			void SetStream(void* stream)
			{
				if (stream_ == stream)
				{
					return;
				}
				CheckCUBLAS(cublasSetStream(handle_, reinterpret_cast<cudaStream_t>(stream)), "cublasSetStream");
				stream_ = stream;
			}

			cublasHandle_t get() const
			{
				return handle_;
			}

		private:
			int deviceIndex_{};
			cublasHandle_t handle_{};
			void* stream_{};
		};

		CUBLASHandle& ThreadLocalCUBLASHandle(int deviceIndex)
		{
			static thread_local std::vector<std::unique_ptr<CUBLASHandle>> handles;
			for (auto& handle : handles)
			{
				if (handle->DeviceIndex() == deviceIndex)
				{
					return *handle;
				}
			}
			auto handle = std::make_unique<CUBLASHandle>(deviceIndex);
			auto& ref = *handle;
			handles.push_back(std::move(handle));
			return ref;
		}

#ifdef LITENN_ENABLE_CUBLASLT
		class CUBLASLtHandle
		{
		public:
			explicit CUBLASLtHandle(int deviceIndex) : deviceIndex_(deviceIndex)
			{
				CUDADeviceGuard guard(deviceIndex_);
				CheckCUBLAS(cublasLtCreate(&handle_), "cublasLtCreate");
			}

			~CUBLASLtHandle()
			{
				if (handle_ != nullptr)
				{
					(void)cublasLtDestroy(handle_);
				}
			}

			CUBLASLtHandle(const CUBLASLtHandle&) = delete;
			CUBLASLtHandle& operator=(const CUBLASLtHandle&) = delete;

			int DeviceIndex() const noexcept
			{
				return deviceIndex_;
			}

			cublasLtHandle_t get() const noexcept
			{
				return handle_;
			}

			struct AlgoKey
			{
				DataType type{};
				int m{};
				int k{};
				int n{};

				bool operator==(const AlgoKey& other) const noexcept
				{
					return type == other.type && m == other.m && k == other.k && n == other.n;
				}
			};

			const cublasLtMatmulAlgo_t* CachedAlgo(const AlgoKey& key) const
			{
				for (const auto& entry : algoCache_)
				{
					if (entry.key == key)
					{
						return &entry.algo;
					}
				}
				return nullptr;
			}

			void CacheAlgo(const AlgoKey& key, const cublasLtMatmulAlgo_t& algo)
			{
				algoCache_.push_back(AlgoCacheEntry{ key, algo });
			}

		private:
			struct AlgoCacheEntry
			{
				AlgoKey key;
				cublasLtMatmulAlgo_t algo;
			};

			int deviceIndex_{};
			cublasLtHandle_t handle_{};
			std::vector<AlgoCacheEntry> algoCache_;
		};

		CUBLASLtHandle& ThreadLocalCUBLASLtHandle(int deviceIndex)
		{
			static thread_local std::vector<std::unique_ptr<CUBLASLtHandle>> handles;
			for (auto& handle : handles)
			{
				if (handle->DeviceIndex() == deviceIndex)
				{
					return *handle;
				}
			}
			auto handle = std::make_unique<CUBLASLtHandle>(deviceIndex);
			auto& ref = *handle;
			handles.push_back(std::move(handle));
			return ref;
		}

		template <typename Descriptor, cublasStatus_t (*DestroyFn)(Descriptor)>
		class CUBLASLtDescriptor
		{
		public:
			CUBLASLtDescriptor() = default;
			explicit CUBLASLtDescriptor(Descriptor descriptor) : descriptor_(descriptor) {}
			~CUBLASLtDescriptor()
			{
				if (descriptor_ != nullptr)
				{
					(void)DestroyFn(descriptor_);
				}
			}

			CUBLASLtDescriptor(const CUBLASLtDescriptor&) = delete;
			CUBLASLtDescriptor& operator=(const CUBLASLtDescriptor&) = delete;

			Descriptor get() const noexcept
			{
				return descriptor_;
			}

		private:
			Descriptor descriptor_{};
		};

		using CUBLASLtMatmulDescOwner =
		    CUBLASLtDescriptor<cublasLtMatmulDesc_t, cublasLtMatmulDescDestroy>;
		using CUBLASLtMatrixLayoutOwner =
		    CUBLASLtDescriptor<cublasLtMatrixLayout_t, cublasLtMatrixLayoutDestroy>;
		using CUBLASLtPreferenceOwner =
		    CUBLASLtDescriptor<cublasLtMatmulPreference_t, cublasLtMatmulPreferenceDestroy>;

		bool ShouldTryCUBLASLtMatMul(int m, int k, int n)
		{
			const char* enable = std::getenv("LITENN_CUDA_ENABLE_CUBLASLT");
			if (enable == nullptr || (std::string_view(enable) != "1" && std::string_view(enable) != "true" &&
			                          std::string_view(enable) != "TRUE" && std::string_view(enable) != "on" &&
			                          std::string_view(enable) != "ON"))
			{
				return false;
			}
			const std::uint64_t flops =
			    static_cast<std::uint64_t>(m) * static_cast<std::uint64_t>(k) * static_cast<std::uint64_t>(n) * 2;
			return flops >= (1ull << 27);
		}

		bool TryCUBLASLtMatMul(CUDA& device, void* dst, DataType type1, ShapeView shape1, const void* src1,
		                       DataType type2, ShapeView shape2, const void* src2, CUDAExecutionOptions options)
		{
			if (shape1.NumDim() != 2 || shape2.NumDim() != 2 || shape1[1] != shape2[0] || type1 != type2 ||
			    type1 != DataType::Float32)
			{
				return false;
			}
			if (shape1[0] > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
			    shape1[1] > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
			    shape2[1] > static_cast<std::size_t>(std::numeric_limits<int>::max()))
			{
				throw std::runtime_error("CUDA MatMul dimensions exceed cuBLASLt int range");
			}

			const auto m = static_cast<int>(shape1[0]);
			const auto k = static_cast<int>(shape1[1]);
			const auto n = static_cast<int>(shape2[1]);
			if (!ShouldTryCUBLASLtMatMul(m, k, n))
			{
				return false;
			}

			CUDADeviceGuard guard(device.deviceIndex);
			auto& handle = ThreadLocalCUBLASLtHandle(device.deviceIndex);

			cublasLtMatmulDesc_t operation{};
			if (cublasLtMatmulDescCreate(&operation, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS)
			{
				return false;
			}
			CUBLASLtMatmulDescOwner operationOwner(operation);
			const cublasOperation_t opN = CUBLAS_OP_N;
			(void)cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
			(void)cublasLtMatmulDescSetAttribute(operation, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

			const cublasLtOrder_t rowMajor = CUBLASLT_ORDER_ROW;
			const auto makeLayout = [&](std::uint64_t rows, std::uint64_t cols, std::int64_t ld)
			    -> CUBLASLtMatrixLayoutOwner {
				cublasLtMatrixLayout_t layout{};
				if (cublasLtMatrixLayoutCreate(&layout, CUDA_R_32F, rows, cols, ld) != CUBLAS_STATUS_SUCCESS)
				{
					return {};
				}
				if (cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_ORDER, &rowMajor,
				                                     sizeof(rowMajor)) != CUBLAS_STATUS_SUCCESS)
				{
					(void)cublasLtMatrixLayoutDestroy(layout);
					return {};
				}
				return CUBLASLtMatrixLayoutOwner(layout);
			};
			auto aLayout = makeLayout(static_cast<std::uint64_t>(m), static_cast<std::uint64_t>(k), k);
			auto bLayout = makeLayout(static_cast<std::uint64_t>(k), static_cast<std::uint64_t>(n), n);
			auto cLayout = makeLayout(static_cast<std::uint64_t>(m), static_cast<std::uint64_t>(n), n);
			if (!aLayout.get() || !bLayout.get() || !cLayout.get())
			{
				return false;
			}

			CUBLASLtHandle::AlgoKey key{ .type = type1, .m = m, .k = k, .n = n };
			const cublasLtMatmulAlgo_t* algo = handle.CachedAlgo(key);
			if (algo == nullptr)
			{
				cublasLtMatmulPreference_t preference{};
				if (cublasLtMatmulPreferenceCreate(&preference) != CUBLAS_STATUS_SUCCESS)
				{
					return false;
				}
				CUBLASLtPreferenceOwner preferenceOwner(preference);
				std::uint64_t workspaceBytes = 0;
				(void)cublasLtMatmulPreferenceSetAttribute(preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
				                                           &workspaceBytes, sizeof(workspaceBytes));
				cublasLtMatmulHeuristicResult_t heuristic{};
				int returned = 0;
				if (cublasLtMatmulAlgoGetHeuristic(handle.get(), operation, aLayout.get(), bLayout.get(), cLayout.get(),
				                                   cLayout.get(), preference, 1, &heuristic, &returned) !=
				        CUBLAS_STATUS_SUCCESS ||
				    returned <= 0 || heuristic.state != CUBLAS_STATUS_SUCCESS)
				{
					return false;
				}
				handle.CacheAlgo(key, heuristic.algo);
				algo = handle.CachedAlgo(key);
			}

			const float alpha = 1.0F;
			const float beta = 0.0F;
			const auto status = cublasLtMatmul(handle.get(), operation, &alpha, src1, aLayout.get(), src2,
			                                   bLayout.get(), &beta, dst, cLayout.get(), dst, cLayout.get(), algo,
			                                   nullptr, 0, reinterpret_cast<cudaStream_t>(options.stream));
			if (status != CUBLAS_STATUS_SUCCESS)
			{
				return false;
			}
			if (options.synchronize)
			{
				CheckCUDA(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(options.stream)),
				          "cudaStreamSynchronize after cuBLASLt MatMul");
			}
			return true;
		}
#endif

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

		std::vector<std::size_t> ResolveGetRowsResultShape(ShapeView dataShape, ShapeView indexShape)
		{
			if (dataShape.NumDim() == 0)
			{
				throw std::runtime_error("GetRows requires a rank >= 1 data tensor");
			}

			auto resultShape = indexShape.ToOwned();
			for (auto dim = 1uz; dim < dataShape.NumDim(); ++dim)
			{
				resultShape.push_back(dataShape[dim]);
			}
			return resultShape;
		}

		void SynchronizeCUDAStream(void* stream, std::string_view action)
		{
			CheckCUDA(cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream)), action);
		}

		bool TryCUBLASMatMul(CUDA& device, void* dst, DataType type1, ShapeView shape1, const void* src1,
		                     DataType type2, ShapeView shape2, const void* src2,
		                     CUDAExecutionOptions options = {})
		{
			if (shape1.NumDim() != 2 || shape2.NumDim() != 2 || shape1[1] != shape2[0] || type1 != type2)
			{
				return false;
			}
			if (!CUDASupportsNativeMatMul(type1, device.deviceIndex))
			{
				return false;
			}
			if (shape1[0] > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
			    shape1[1] > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
			    shape2[1] > static_cast<std::size_t>(std::numeric_limits<int>::max()))
			{
				throw std::runtime_error("CUDA MatMul dimensions exceed cuBLAS int range");
			}

			CUDADeviceGuard guard(device.deviceIndex);
			auto& handle = ThreadLocalCUBLASHandle(device.deviceIndex);
			handle.SetStream(options.stream);
			const auto m = static_cast<int>(shape1[0]);
			const auto k = static_cast<int>(shape1[1]);
			const auto n = static_cast<int>(shape2[1]);

#ifdef LITENN_ENABLE_CUBLASLT
			if (TryCUBLASLtMatMul(device, dst, type1, shape1, src1, type2, shape2, src2, options))
			{
				return true;
			}
#endif

			if (type1 == DataType::Float32)
			{
				const float alpha = 1.0F;
				const float beta = 0.0F;
				CheckCUBLAS(cublasSgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
				                        static_cast<const float*>(src2), n, static_cast<const float*>(src1), k,
				                        &beta, static_cast<float*>(dst), n),
				            "cublasSgemm");
			}
			else if (type1 == DataType::Float64)
			{
				const double alpha = 1.0;
				const double beta = 0.0;
				CheckCUBLAS(cublasDgemm(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha,
				                        static_cast<const double*>(src2), n, static_cast<const double*>(src1), k,
				                        &beta, static_cast<double*>(dst), n),
				            "cublasDgemm");
			}
			else if (type1 == DataType::Float16)
			{
				const float alpha = 1.0F;
				const float beta = 0.0F;
				const auto status = cublasGemmEx(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, src2,
				                                  CUDA_R_16F, n, src1, CUDA_R_16F, k, &beta, dst, CUDA_R_16F, n,
				                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
				if (status != CUBLAS_STATUS_SUCCESS)
				{
					return false;
				}
			}
#if defined(CUDART_VERSION) && CUDART_VERSION >= 11000
			else if (type1 == DataType::BFloat16)
			{
				const float alpha = 1.0F;
				const float beta = 0.0F;
				const auto status = cublasGemmEx(handle.get(), CUBLAS_OP_N, CUBLAS_OP_N, n, m, k, &alpha, src2,
				                                  CUDA_R_16BF, n, src1, CUDA_R_16BF, k, &beta, dst, CUDA_R_16BF, n,
				                                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
				if (status != CUBLAS_STATUS_SUCCESS)
				{
					return false;
				}
			}
#endif
			else
			{
				return false;
			}
			if (options.synchronize)
			{
				SynchronizeCUDAStream(options.stream, "cudaStreamSynchronize after cuBLAS MatMul");
			}
			return true;
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

	std::optional<CUDALowPrecisionCapabilities> TryGetCUDALowPrecisionCapabilities(int deviceIndex) noexcept
	{
		if (!IsCUDADeviceAvailable(deviceIndex))
		{
			return std::nullopt;
		}
		cudaDeviceProp properties{};
		const auto status = cudaGetDeviceProperties(&properties, deviceIndex);
		if (status != cudaSuccess)
		{
			ClearCUDAError();
			return std::nullopt;
		}
		return MakeLowPrecisionCapabilities(deviceIndex, properties);
	}

	CUDALowPrecisionCapabilities GetCUDALowPrecisionCapabilities(int deviceIndex)
	{
		if (auto capabilities = TryGetCUDALowPrecisionCapabilities(deviceIndex))
		{
			return *capabilities;
		}
		throw std::runtime_error(std::format("CUDA device {} is not available", deviceIndex));
	}

	bool CUDASupportsLowPrecisionStorage(DataType dtype, int deviceIndex) noexcept
	{
		const auto capabilities = TryGetCUDALowPrecisionCapabilities(deviceIndex);
		if (!capabilities)
		{
			return false;
		}
		switch (dtype)
		{
		case DataType::Float16:
			return capabilities->supportsFloat16Storage;
		case DataType::BFloat16:
			return capabilities->supportsBFloat16Storage;
		case DataType::Float8E4M3:
		case DataType::Float8E5M2:
			return capabilities->supportsFloat8Storage;
		case DataType::Int8:
		case DataType::UInt8:
			return capabilities->supportsInt8Storage;
		case DataType::Float32:
		case DataType::Float64:
		case DataType::Int32:
		case DataType::Int64:
		case DataType::Bool:
			return false;
		}
		return false;
	}

	bool CUDASupportsNativeMatMul(DataType dtype, int deviceIndex) noexcept
	{
		if (!IsCUDADeviceAvailable(deviceIndex))
		{
			return false;
		}
		if (dtype == DataType::Float32 || dtype == DataType::Float64)
		{
			return true;
		}
		const auto capabilities = TryGetCUDALowPrecisionCapabilities(deviceIndex);
		if (!capabilities)
		{
			return false;
		}
		switch (dtype)
		{
		case DataType::Float16:
			return capabilities->supportsFloat16MatMul;
		case DataType::BFloat16:
			return capabilities->supportsBFloat16MatMul;
		case DataType::Float8E4M3:
		case DataType::Float8E5M2:
			return capabilities->supportsFloat8MatMul;
		case DataType::Float32:
		case DataType::Float64:
		case DataType::Int32:
		case DataType::Int64:
		case DataType::Int8:
		case DataType::UInt8:
		case DataType::Bool:
			return false;
		}
		return false;
	}

	std::string FormatCUDALowPrecisionCapabilities(const CUDALowPrecisionCapabilities& capabilities)
	{
		return std::format(
		    "CUDA device {} cc {}.{}: cuBLASLt={}, fp16(storage={}, matmul={}), "
		    "bf16(storage={}, matmul={}), fp8(storage={}, matmul={}), int8(storage={}, tensorCores={})",
		    capabilities.deviceIndex, capabilities.computeCapabilityMajor, capabilities.computeCapabilityMinor,
		    capabilities.hasCUBLASLt, capabilities.supportsFloat16Storage, capabilities.supportsFloat16MatMul,
		    capabilities.supportsBFloat16Storage, capabilities.supportsBFloat16MatMul,
		    capabilities.supportsFloat8Storage, capabilities.supportsFloat8MatMul, capabilities.supportsInt8Storage,
		    capabilities.supportsInt8TensorCores);
	}

#ifdef LITENN_ENABLE_CUDA_DRIVER
	bool IsCUDADriverAvailable(int deviceIndex) noexcept
	{
		if (deviceIndex < 0 || cuInit(0) != CUDA_SUCCESS)
		{
			return false;
		}
		int count = 0;
		if (cuDeviceGetCount(&count) != CUDA_SUCCESS || deviceIndex >= count)
		{
			return false;
		}
		CUdevice device{};
		return cuDeviceGet(&device, deviceIndex) == CUDA_SUCCESS;
	}

	struct CUDADriverModule::Impl
	{
		CUDA device;
		CUdevice driverDevice{};
		CUcontext context{};
		CUmodule module{};
		mutable std::mutex functionCacheMutex;
		mutable std::vector<std::pair<std::string, CUfunction>> functionCache;

		Impl(CUDA deviceValue, std::span<const std::byte> image) : device(std::move(deviceValue))
		{
			if (image.empty())
			{
				throw std::runtime_error("CUDA driver module image must not be empty");
			}

			InitializeCUDADriver();
			CheckCUDADriver(cuDeviceGet(&driverDevice, device.deviceIndex), "cuDeviceGet");
			CheckCUDADriver(cuDevicePrimaryCtxRetain(&context, driverDevice), "cuDevicePrimaryCtxRetain");
			try
			{
				CUDADriverContextScope scope(context);
				std::array<char, 8192> errorLog{};
				std::array<char, 8192> infoLog{};
				std::array<CUjit_option, 4> options{
					CU_JIT_ERROR_LOG_BUFFER,
					CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
					CU_JIT_INFO_LOG_BUFFER,
					CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
				};
				std::array<void*, 4> values{
					errorLog.data(),
					reinterpret_cast<void*>(errorLog.size()),
					infoLog.data(),
					reinterpret_cast<void*>(infoLog.size()),
				};
				const auto status = cuModuleLoadDataEx(&module, image.data(), static_cast<unsigned int>(options.size()),
				                                       options.data(), values.data());
				std::string detail;
				if (errorLog[0] != '\0')
				{
					detail += "CUDA JIT error log:\n";
					detail += errorLog.data();
				}
				if (infoLog[0] != '\0')
				{
					if (!detail.empty())
					{
						detail += "\n";
					}
					detail += "CUDA JIT info log:\n";
					detail += infoLog.data();
				}
				CheckCUDADriver(status, "cuModuleLoadDataEx", detail);
			}
			catch (...)
			{
				(void)cuDevicePrimaryCtxRelease(driverDevice);
				context = nullptr;
				throw;
			}
		}

		~Impl()
		{
			if (module != nullptr && context != nullptr)
			{
				if (cuCtxPushCurrent(context) == CUDA_SUCCESS)
				{
					(void)cuModuleUnload(module);
					CUcontext popped{};
					(void)cuCtxPopCurrent(&popped);
				}
			}
			if (context != nullptr)
			{
				(void)cuDevicePrimaryCtxRelease(driverDevice);
			}
		}

		CUfunction GetFunction(std::string_view functionName) const
		{
			if (functionName.empty())
			{
				throw std::runtime_error("CUDA driver kernel function name must not be empty");
			}
			std::lock_guard lock(functionCacheMutex);
			for (const auto& entry : functionCache)
			{
				if (entry.first == functionName)
				{
					return entry.second;
				}
			}
			CUDADriverContextScope scope(context);
			CUfunction function{};
			CheckCUDADriver(cuModuleGetFunction(&function, module, std::string(functionName).c_str()),
			                "cuModuleGetFunction");
			functionCache.emplace_back(functionName, function);
			return function;
		}
	};
#else
	bool IsCUDADriverAvailable(int) noexcept
	{
		return false;
	}

	struct CUDADriverModule::Impl
	{
	};
#endif

	CUDADriverModule::CUDADriverModule() = default;
	CUDADriverModule::CUDADriverModule(CUDADriverModule&&) noexcept = default;
	CUDADriverModule& CUDADriverModule::operator=(CUDADriverModule&&) noexcept = default;
	CUDADriverModule::~CUDADriverModule() = default;

	CUDADriverModule::CUDADriverModule(CUDA device, std::span<const std::byte> image)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		impl_ = std::make_unique<Impl>(std::move(device), image);
#else
		(void)device;
		(void)image;
		throw std::runtime_error("CUDA driver runtime is not enabled in this LiteNN build");
#endif
	}

	bool CUDADriverModule::Empty() const noexcept
	{
		return !impl_;
	}

	void CUDADriverModule::CacheFunction(std::string_view functionName) const
	{
		if (!impl_)
		{
			throw std::runtime_error("CUDA driver module is empty");
		}
#ifdef LITENN_ENABLE_CUDA_DRIVER
		(void)impl_->GetFunction(functionName);
#else
		(void)functionName;
		throw std::runtime_error("CUDA driver runtime is not enabled in this LiteNN build");
#endif
	}

	void CUDADriverModule::Launch(std::string_view functionName, const CUDADriverLaunchOptions& options,
	                              std::span<void*> arguments) const
	{
		if (!impl_)
		{
			throw std::runtime_error("CUDA driver module is empty");
		}
		if (options.grid.x == 0 || options.grid.y == 0 || options.grid.z == 0 ||
		    options.block.x == 0 || options.block.y == 0 || options.block.z == 0)
		{
			throw std::runtime_error("CUDA driver launch dimensions must be non-zero");
		}
		if (options.sharedMemoryBytes > static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()))
		{
			throw std::runtime_error("CUDA driver launch shared memory exceeds unsigned int range");
		}
#ifdef LITENN_ENABLE_CUDA_DRIVER
		const auto function = impl_->GetFunction(functionName);
		CUDADriverContextScope scope(impl_->context);
		CheckCUDADriver(cuLaunchKernel(function, options.grid.x, options.grid.y, options.grid.z,
		                               options.block.x, options.block.y, options.block.z,
		                               static_cast<unsigned int>(options.sharedMemoryBytes),
		                               reinterpret_cast<CUstream>(options.stream),
		                               arguments.empty() ? nullptr : arguments.data(), nullptr),
		                "cuLaunchKernel");
		if (options.synchronize)
		{
			CheckCUDADriver(cuStreamSynchronize(reinterpret_cast<CUstream>(options.stream)), "cuStreamSynchronize");
		}
#else
		(void)functionName;
		(void)options;
		(void)arguments;
		throw std::runtime_error("CUDA driver runtime is not enabled in this LiteNN build");
#endif
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
		CopyToCPU(device, srcType, src, size, dstType, dst, CUDAExecutionOptions{});
	}

	void DeviceTraits<CUDA>::CopyToCPU(CUDA& device, DataType srcType, const void* src, std::size_t size,
	                                   DataType dstType, void* dst, CUDAExecutionOptions options)
	{
		CUDADeviceGuard guard(device.deviceIndex);
		const auto stream = reinterpret_cast<cudaStream_t>(options.stream);
		if (srcType == dstType)
		{
			CheckCUDA(cudaMemcpyAsync(dst, src, ElementSize(srcType) * size, cudaMemcpyDeviceToHost, stream),
			          "cudaMemcpyAsync D2H");
			if (options.synchronize)
			{
				SynchronizeCUDAStream(options.stream, "cudaStreamSynchronize after cudaMemcpyAsync D2H");
			}
			return;
		}
		if (!options.synchronize)
		{
			throw std::runtime_error("Asynchronous CUDA D2H copy with data type conversion is not supported");
		}

		auto hostSrc = MakeHostBuffer(srcType, size);
		CheckCUDA(cudaMemcpyAsync(hostSrc.data(), src, ElementSize(srcType) * size, cudaMemcpyDeviceToHost, stream),
		          "cudaMemcpyAsync D2H");
		SynchronizeCUDAStream(options.stream, "cudaStreamSynchronize after cudaMemcpyAsync D2H");
		CPU cpu;
		DeviceTraits<CPU>::ConvertTo(cpu, srcType, hostSrc.data(), size, dstType, dst);
	}

	void DeviceTraits<CUDA>::CopyFromCPU(CUDA& device, DataType dstType, void* dst, DataType srcType,
	                                     const void* src, std::size_t size)
	{
		CopyFromCPU(device, dstType, dst, srcType, src, size, CUDAExecutionOptions{});
	}

	void DeviceTraits<CUDA>::CopyFromCPU(CUDA& device, DataType dstType, void* dst, DataType srcType,
	                                     const void* src, std::size_t size, CUDAExecutionOptions options)
	{
		CUDADeviceGuard guard(device.deviceIndex);
		const auto stream = reinterpret_cast<cudaStream_t>(options.stream);
		if (srcType == dstType)
		{
			CheckCUDA(cudaMemcpyAsync(dst, src, ElementSize(dstType) * size, cudaMemcpyHostToDevice, stream),
			          "cudaMemcpyAsync H2D");
			if (options.synchronize)
			{
				SynchronizeCUDAStream(options.stream, "cudaStreamSynchronize after cudaMemcpyAsync H2D");
			}
			return;
		}
		if (!options.synchronize)
		{
			throw std::runtime_error("Asynchronous CUDA H2D copy with data type conversion is not supported");
		}

		auto hostDst = MakeHostBuffer(dstType, size);
		CPU cpu;
		DeviceTraits<CPU>::ConvertTo(cpu, srcType, src, size, dstType, hostDst.data());
		CheckCUDA(cudaMemcpyAsync(dst, hostDst.data(), ElementSize(dstType) * size, cudaMemcpyHostToDevice, stream),
		          "cudaMemcpyAsync H2D");
		SynchronizeCUDAStream(options.stream, "cudaStreamSynchronize after cudaMemcpyAsync H2D");
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
		DoBinaryOp(device, binaryOp, dst, type1, shape1, src1, type2, shape2, src2, CUDAExecutionOptions{});
	}

	void DeviceTraits<CUDA>::DoBinaryOp(CUDA& device, BinaryOp binaryOp, void* dst, DataType type1, ShapeView shape1,
	                                    const void* src1, DataType type2, ShapeView shape2, const void* src2,
	                                    CUDAExecutionOptions options)
	{
		if (binaryOp == BinaryOp::MatMul && TryCUBLASMatMul(device, dst, type1, shape1, src1, type2, shape2, src2,
		                                                     options))
		{
			return;
		}
		if (options.stream != nullptr || !options.synchronize)
		{
			throw std::runtime_error("Stream-aware asynchronous CUDA binary fallback is not supported");
		}

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

	void DeviceTraits<CUDA>::DoGetRowsOp(CUDA& device, void* dst, DataType dataType, ShapeView dataShape,
	                                    const void* data, DataType indexType, ShapeView indexShape,
	                                    const void* indices)
	{
		const auto resultShape = ResolveGetRowsResultShape(dataShape, indexShape);
		auto hostData = MakeHostBuffer(dataType, dataShape.NumElements());
		auto hostIndices = MakeHostBuffer(indexType, indexShape.NumElements());
		auto hostDst = MakeHostBuffer(dataType, ShapeView{ resultShape }.NumElements());

		CopyToCPU(device, dataType, data, dataShape.NumElements(), dataType, hostData.data());
		CopyToCPU(device, indexType, indices, indexShape.NumElements(), indexType, hostIndices.data());

		CPU cpu;
		DeviceTraits<CPU>::DoGetRowsOp(cpu, hostDst.data(), dataType, dataShape, hostData.data(), indexType,
		                              indexShape, hostIndices.data());
		CopyFromCPU(device, dataType, dst, dataType, hostDst.data(), ShapeView{ resultShape }.NumElements());
	}

	void DeviceTraits<CUDA>::DoPermuteOp(CUDA& device, void* dst, DataType type, ShapeView srcShape,
	                                    const void* src, ShapeView permutation)
	{
		const auto numElements = srcShape.NumElements();
		auto hostSrc = MakeHostBuffer(type, numElements);
		auto hostDst = MakeHostBuffer(type, numElements);
		CopyToCPU(device, type, src, numElements, type, hostSrc.data());

		CPU cpu;
		DeviceTraits<CPU>::DoPermuteOp(cpu, hostDst.data(), type, srcShape, hostSrc.data(), permutation);
		CopyFromCPU(device, type, dst, type, hostDst.data(), numElements);
	}
} // namespace LiteNN

#endif
