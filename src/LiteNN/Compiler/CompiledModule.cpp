#include "CompiledModule.h"

#include "CUDANativeCodegen.h"
#include "CUDANativePayload.h"
#include "Dialect/LiteNNDialect.h"
#include "Dialect/LiteNNOps.h"
#include "Pass/BufferizationPipeline.h"
#include "Pass/LLVMCodegenPipeline.h"
#include "Pass/LowerLiteNNPass.h"
#include "Translation/GraphToMLIR.h"
#ifdef LITENN_ENABLE_VULKAN
#include "VulkanNativeCodegen.h"
#include "VulkanNativePayload.h"
#endif

#include <LiteNN/Misc.h>
#include <LiteNN/Pass/FusionPass.h>
#include <LiteNN/Validation/GraphValidator.h>

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/Analysis/LoopAnalysisManager.h"
#include "llvm/ExecutionEngine/ExecutionEngine.h"
#include "llvm/ExecutionEngine/MCJIT.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Verifier.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/Support/DynamicLibrary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Host.h"
#include "llvm/TargetParser/SubtargetFeature.h"
#include "llvm/TargetParser/Triple.h"

#ifdef LITENN_ENABLE_CUDA
#include <cuda_runtime_api.h>
#endif

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

using namespace LiteNN;

namespace
{
#if defined(__GNUC__) || defined(__clang__)
#define LITENN_RESTRICT __restrict__
#define LITENN_GCC_IVDEP _Pragma("GCC ivdep")
#else
#define LITENN_RESTRICT
#define LITENN_GCC_IVDEP
#endif

	bool IsCPUExternalRegionsEnabled(const CompilerOptions& options)
	{
		return options.enableCPUAOTExternalRegions;
	}

	void LogCompileDiagnostic(const CompilerOptions& options, std::string_view message)
	{
		if (options.enableCompileDiagnostics)
		{
			std::cerr << "[LiteNN compile] " << message << '\n' << std::flush;
		}
	}

	template <typename F>
	auto TimedCompileDiagnostic(const CompilerOptions& options, std::string_view label, F&& f)
	{
		LogCompileDiagnostic(options, std::string(label) + "...");
		const auto begin = std::chrono::steady_clock::now();
		if constexpr (std::is_void_v<std::invoke_result_t<F>>)
		{
			std::forward<F>(f)();
			const auto elapsed =
			    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
			LogCompileDiagnostic(options, std::format("{}: ok {:.3f} ms", label, elapsed));
		}
		else
		{
			auto result = std::forward<F>(f)();
			const auto elapsed =
			    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
			LogCompileDiagnostic(options, std::format("{}: ok {:.3f} ms", label, elapsed));
			return result;
		}
	}

	thread_local const void* tCPUExternalConstants = nullptr;
	thread_local const void* tCPUExternalWeights = nullptr;

	extern "C" const void* litenn_cpu_external_constants()
	{
		return tCPUExternalConstants;
	}

	extern "C" const void* litenn_cpu_external_weights()
	{
		return tCPUExternalWeights;
	}

	class ScopedCPUExternalRegions
	{
	public:
		ScopedCPUExternalRegions(const void* constants, const void* weights)
		    : previousConstants_(tCPUExternalConstants), previousWeights_(tCPUExternalWeights)
		{
			tCPUExternalConstants = constants;
			tCPUExternalWeights = weights;
		}

		~ScopedCPUExternalRegions()
		{
			tCPUExternalConstants = previousConstants_;
			tCPUExternalWeights = previousWeights_;
		}

	private:
		const void* previousConstants_{};
		const void* previousWeights_{};
	};

	using LiteNNCPUParallelForBody = void (*)(std::uint64_t begin, std::uint64_t end, void* userData);

	std::size_t LiteNNCPUHardwareThreadCount()
	{
		const auto hardware = std::thread::hardware_concurrency();
		return hardware == 0 ? 1 : static_cast<std::size_t>(hardware);
	}

	std::size_t ResolveCPUAOTThreadCount(const CompilerOptions& options)
	{
		return options.cpuAOTThreadCount == 0 ? LiteNNCPUHardwareThreadCount() : options.cpuAOTThreadCount;
	}

	std::size_t LiteNNCPUMaxThreadCount()
	{
		return LiteNNCPUHardwareThreadCount();
	}

	class LiteNNCPUThreadAffinityState
	{
	public:
		LiteNNCPUThreadAffinityState() = default;

		void Apply(CPUAOTAffinityPolicy policy, std::size_t workerSlot)
		{
			if (policy == activePolicy_)
			{
				return;
			}
			Restore();
			if (policy != CPUAOTAffinityPolicy::Compact)
			{
				return;
			}
			const auto hardware = LiteNNCPUHardwareThreadCount();
			if (hardware == 0 || workerSlot >= hardware)
			{
				return;
			}
#ifdef _WIN32
			if (workerSlot >= sizeof(DWORD_PTR) * 8)
			{
				return;
			}
			const DWORD_PTR mask = static_cast<DWORD_PTR>(1) << workerSlot;
			const auto previous = SetThreadAffinityMask(GetCurrentThread(), mask);
			if (previous != 0)
			{
				previousMask_ = previous;
				activePolicy_ = policy;
			}
#elif defined(__linux__)
			cpu_set_t currentSet;
			CPU_ZERO(&currentSet);
			if (pthread_getaffinity_np(pthread_self(), sizeof(currentSet), &currentSet) != 0)
			{
				return;
			}
			cpu_set_t targetSet;
			CPU_ZERO(&targetSet);
			CPU_SET(workerSlot, &targetSet);
			if (pthread_setaffinity_np(pthread_self(), sizeof(targetSet), &targetSet) == 0)
			{
				previousSet_ = currentSet;
				activePolicy_ = policy;
			}
#endif
		}

		~LiteNNCPUThreadAffinityState()
		{
			Restore();
		}

		LiteNNCPUThreadAffinityState(const LiteNNCPUThreadAffinityState&) = delete;
		LiteNNCPUThreadAffinityState& operator=(const LiteNNCPUThreadAffinityState&) = delete;

	private:
		void Restore()
		{
			if (activePolicy_ == CPUAOTAffinityPolicy::None)
			{
				return;
			}
#ifdef _WIN32
			SetThreadAffinityMask(GetCurrentThread(), previousMask_);
#elif defined(__linux__)
			pthread_setaffinity_np(pthread_self(), sizeof(previousSet_), &previousSet_);
#endif
			activePolicy_ = CPUAOTAffinityPolicy::None;
		}

		CPUAOTAffinityPolicy activePolicy_{ CPUAOTAffinityPolicy::None };
#ifdef _WIN32
		DWORD_PTR previousMask_{};
#elif defined(__linux__)
		cpu_set_t previousSet_{};
#endif
	};

	class LiteNNCPUThreadPool
	{
	public:
		explicit LiteNNCPUThreadPool(std::size_t threadCount)
		{
			const auto workerCount = threadCount > 1 ? threadCount - 1 : 0;
			workers_.reserve(workerCount);
			for (std::size_t i = 0; i < workerCount; ++i)
			{
				workers_.emplace_back([this, i] { WorkerLoop(i); });
			}
		}

		~LiteNNCPUThreadPool()
		{
			{
				std::lock_guard lock(mutex_);
				stopping_ = true;
				++generation_;
			}
			start_.notify_all();
			for (auto& worker : workers_)
			{
				if (worker.joinable())
				{
					worker.join();
				}
			}
		}

		LiteNNCPUThreadPool(const LiteNNCPUThreadPool&) = delete;
		LiteNNCPUThreadPool& operator=(const LiteNNCPUThreadPool&) = delete;

		void ParallelFor(std::uint64_t begin, std::uint64_t end, std::uint64_t grain, LiteNNCPUParallelForBody body,
		                 void* userData, std::size_t requestedThreads, CPUAOTAffinityPolicy affinityPolicy)
		{
			if (begin >= end)
			{
				return;
			}
			grain = std::max<std::uint64_t>(1, grain);
			const auto taskCount = (end - begin + grain - 1) / grain;
			const auto participantCount = std::min<std::uint64_t>(
			    std::max<std::uint64_t>(1, static_cast<std::uint64_t>(requestedThreads)), taskCount);
			if (participantCount <= 1 || workers_.empty())
			{
				body(begin, end, userData);
				return;
			}

			std::unique_lock runLock(runMutex_);
			const auto desiredWorkers =
			    std::min<std::size_t>(static_cast<std::size_t>(participantCount - 1), workers_.size());
			{
				std::lock_guard lock(mutex_);
				begin_ = begin;
				end_ = end;
				grain_ = grain;
				body_ = body;
				userData_ = userData;
				affinityPolicy_ = affinityPolicy;
				next_.store(begin, std::memory_order_relaxed);
				workersDone_ = 0;
				desiredWorkers_ = desiredWorkers;
				++generation_;
			}
			start_.notify_all();
			RunTasks();

			std::unique_lock lock(mutex_);
			done_.wait(lock, [&] { return workersDone_ == desiredWorkers; });
		}

	private:
		void RunTasks()
		{
			while (true)
			{
				const auto taskBegin = next_.fetch_add(grain_, std::memory_order_relaxed);
				if (taskBegin >= end_)
				{
					break;
				}
				const auto taskEnd = std::min<std::uint64_t>(taskBegin + grain_, end_);
				body_(taskBegin, taskEnd, userData_);
			}
		}

		void WorkerLoop(std::size_t workerIndex)
		{
			LiteNNCPUThreadAffinityState affinity;
			std::uint64_t seenGeneration = 0;
			while (true)
			{
				bool participate = false;
				CPUAOTAffinityPolicy affinityPolicy = CPUAOTAffinityPolicy::None;
				{
					std::unique_lock lock(mutex_);
					start_.wait(lock, [&] { return stopping_ || generation_ != seenGeneration; });
					if (stopping_)
					{
						return;
					}
					seenGeneration = generation_;
					participate = workerIndex < desiredWorkers_;
					affinityPolicy = affinityPolicy_;
				}

				if (participate)
				{
					affinity.Apply(affinityPolicy, workerIndex + 1);
					RunTasks();

					std::lock_guard lock(mutex_);
					++workersDone_;
					if (workersDone_ == desiredWorkers_)
					{
						done_.notify_one();
					}
				}
				else
				{
					affinity.Apply(CPUAOTAffinityPolicy::None, workerIndex);
				}
			}
		}

		std::vector<std::thread> workers_;
		std::mutex runMutex_;
		std::mutex mutex_;
		std::condition_variable start_;
		std::condition_variable done_;
		std::atomic<std::uint64_t> next_{ 0 };
		std::uint64_t begin_{};
		std::uint64_t end_{};
		std::uint64_t grain_{ 1 };
		LiteNNCPUParallelForBody body_{};
		void* userData_{};
		CPUAOTAffinityPolicy affinityPolicy_{ CPUAOTAffinityPolicy::None };
		std::size_t desiredWorkers_{};
		std::size_t workersDone_{};
		std::uint64_t generation_{};
		bool stopping_{};
	};

	LiteNNCPUThreadPool& GetLiteNNCPUThreadPool()
	{
		static LiteNNCPUThreadPool pool(LiteNNCPUMaxThreadCount());
		return pool;
	}

	void LiteNNCPUParallelFor(std::uint64_t begin, std::uint64_t end, std::uint64_t grain,
	                          LiteNNCPUParallelForBody body, void* userData, std::uint64_t threadCount,
	                          CPUAOTAffinityPolicy affinityPolicy)
	{
		GetLiteNNCPUThreadPool().ParallelFor(begin, end, grain, body, userData, static_cast<std::size_t>(threadCount),
		                                     affinityPolicy);
	}

	void LiteNNCPUMatMulBiasReLURange(const float* LITENN_RESTRICT lhs, const float* LITENN_RESTRICT rhs,
	                                  const float* LITENN_RESTRICT bias, float* LITENN_RESTRICT out,
	                                  std::uint64_t rowBegin, std::uint64_t rowEnd, std::uint64_t k, std::uint64_t n,
	                                  std::uint64_t biasRows, bool relu)
	{
		constexpr std::uint64_t kRowBlock = 4;
		auto copyBias = [&](std::uint64_t row, float* outRow) {
			const float* biasRow = bias + (biasRows == 1 ? 0 : row) * n;
			std::memcpy(outRow, biasRow, static_cast<std::size_t>(n) * sizeof(float));
		};
		auto applyRelu = [&](float* outRow) {
			if (!relu)
			{
				return;
			}
			LITENN_GCC_IVDEP
			for (std::uint64_t col = 0; col < n; ++col)
			{
				if (outRow[col] < 0.0f)
				{
					outRow[col] = 0.0f;
				}
			}
		};

		std::uint64_t row = rowBegin;
		for (; row + kRowBlock <= rowEnd; row += kRowBlock)
		{
			float* out0 = out + (row + 0) * n;
			float* out1 = out + (row + 1) * n;
			float* out2 = out + (row + 2) * n;
			float* out3 = out + (row + 3) * n;
			copyBias(row + 0, out0);
			copyBias(row + 1, out1);
			copyBias(row + 2, out2);
			copyBias(row + 3, out3);

			for (std::uint64_t kk = 0; kk < k; ++kk)
			{
				const auto lhsOffset = kk;
				const float a0 = lhs[(row + 0) * k + lhsOffset];
				const float a1 = lhs[(row + 1) * k + lhsOffset];
				const float a2 = lhs[(row + 2) * k + lhsOffset];
				const float a3 = lhs[(row + 3) * k + lhsOffset];
				const float* rhsRow = rhs + kk * n;
				LITENN_GCC_IVDEP
				for (std::uint64_t col = 0; col < n; ++col)
				{
					const float b = rhsRow[col];
					out0[col] += a0 * b;
					out1[col] += a1 * b;
					out2[col] += a2 * b;
					out3[col] += a3 * b;
				}
			}

			applyRelu(out0);
			applyRelu(out1);
			applyRelu(out2);
			applyRelu(out3);
		}

		for (; row < rowEnd; ++row)
		{
			float* outRow = out + row * n;
			copyBias(row, outRow);

			for (std::uint64_t kk = 0; kk < k; ++kk)
			{
				const float a = lhs[row * k + kk];
				const float* rhsRow = rhs + kk * n;
				LITENN_GCC_IVDEP
				for (std::uint64_t col = 0; col < n; ++col)
				{
					outRow[col] += a * rhsRow[col];
				}
			}

			applyRelu(outRow);
		}
	}

	void LiteNNCPUMatMulBiasReLUParallel(const float* LITENN_RESTRICT lhs, const float* LITENN_RESTRICT rhs,
	                                     const float* LITENN_RESTRICT bias, float* LITENN_RESTRICT out, std::uint64_t m,
	                                     std::uint64_t k, std::uint64_t n, std::uint64_t biasRows,
	                                     std::uint64_t requestedThreadCount, bool relu,
	                                     CPUAOTAffinityPolicy affinityPolicy)
	{
		const auto flops = m * k * n * 2;
		const auto threadCount = std::min<std::uint64_t>(
		    requestedThreadCount == 0 ? LiteNNCPUHardwareThreadCount() : requestedThreadCount, m);
		if (threadCount <= 1 || flops < (1ull << 20))
		{
			LiteNNCPUMatMulBiasReLURange(lhs, rhs, bias, out, 0, m, k, n, biasRows, relu);
			return;
		}

		struct Context
		{
			const float* lhs{};
			const float* rhs{};
			const float* bias{};
			float* out{};
			std::uint64_t k{};
			std::uint64_t n{};
			std::uint64_t biasRows{};
			bool relu{};
		};
		Context context{ lhs, rhs, bias, out, k, n, biasRows, relu };
		const auto body = [](std::uint64_t begin, std::uint64_t end, void* userData) {
			const auto& ctx = *static_cast<const Context*>(userData);
			LiteNNCPUMatMulBiasReLURange(ctx.lhs, ctx.rhs, ctx.bias, ctx.out, begin, end, ctx.k, ctx.n, ctx.biasRows,
			                             ctx.relu);
		};

		const auto grain = std::max<std::uint64_t>(1, (m + threadCount * 4 - 1) / (threadCount * 4));
		LiteNNCPUParallelFor(0, m, grain, body, &context, threadCount, affinityPolicy);
	}

	bool ShouldUseCPUSidecarLinearLayer(std::uint64_t m, std::uint64_t k, std::uint64_t n, std::uint64_t flops)
	{
		constexpr std::uint64_t kMinLayerFlops = 1ull << 26;
		constexpr std::uint64_t kMaxRowsBeforePackedMLIR = 256;
		constexpr std::uint64_t kMinOutputColumns = 64;
		return flops >= kMinLayerFlops && m <= kMaxRowsBeforePackedMLIR && k >= 64 && n >= kMinOutputColumns;
	}

	extern "C" void litenn_cpu_matmul_bias_relu_parallel_f32(const float* lhs, const float* rhs, const float* bias,
	                                                         float* out, std::uint64_t m, std::uint64_t k,
	                                                         std::uint64_t n, std::uint64_t biasRows,
	                                                         std::uint64_t threadCount, std::uint64_t affinityPolicy,
	                                                         bool relu)
	{
		const auto policy = affinityPolicy == static_cast<std::uint64_t>(CPUAOTAffinityPolicy::Compact)
		                        ? CPUAOTAffinityPolicy::Compact
		                        : CPUAOTAffinityPolicy::None;
		LiteNNCPUMatMulBiasReLUParallel(lhs, rhs, bias, out, m, k, n, biasRows, threadCount, relu, policy);
	}

	extern "C" void litenn_cpu_ggml_block_matmul_f32(
	    const float*, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows, std::int64_t lhsColumns,
	    std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t*, const std::uint8_t* rhsAligned,
	    std::int64_t rhsOffset, std::int64_t rhsBytes, std::int64_t rhsStride, float*, float* outAligned,
	    std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride,
	    std::int64_t outColumnStride, std::uint64_t formatValue)
	{
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		const auto layout = GetQuantizedBlockLayout(format);
		if (!layout || lhsRows < 0 || lhsColumns < 0 || outRows < 0 || outColumns < 0 || rhsBytes < 0 ||
		    lhsRows != outRows || lhsColumns == 0 || outColumns == 0 ||
		    static_cast<std::uint64_t>(lhsColumns) % layout->elementsPerBlock != 0)
		{
			return;
		}

		const auto rowBytes =
		    (static_cast<std::uint64_t>(lhsColumns) / layout->elementsPerBlock) * layout->bytesPerBlock;
		if (static_cast<std::uint64_t>(rhsBytes) < static_cast<std::uint64_t>(outColumns) * rowBytes)
		{
			return;
		}

		for (std::int64_t row = 0; row < lhsRows; ++row)
		{
			for (std::int64_t column = 0; column < outColumns; ++column)
			{
				const auto* weightRow =
				    rhsAligned + rhsOffset + column * static_cast<std::int64_t>(rowBytes) * rhsStride;
				double acc = 0.0;
				for (std::int64_t kk = 0; kk < lhsColumns; ++kk)
				{
					const auto blockIndex = static_cast<std::uint64_t>(kk) / layout->elementsPerBlock;
					const auto lane = static_cast<std::uint64_t>(kk) % layout->elementsPerBlock;
					const auto* block = weightRow + blockIndex * layout->bytesPerBlock * rhsStride;
					const auto lhsValue = lhsAligned[lhsOffset + row * lhsRowStride + kk * lhsColumnStride];
					acc += static_cast<double>(lhsValue) *
					       static_cast<double>(QuantizationDetail::DecodeGGMLBlockElement(block, format, lane));
				}
				outAligned[outOffset + row * outRowStride + column * outColumnStride] = static_cast<float>(acc);
			}
		}
	}

	constexpr std::string_view kEntrySymbol = "litenn_forward";
	constexpr std::array<std::byte, 8> kRodataMagic = {
		std::byte{ 'L' }, std::byte{ 'T' }, std::byte{ 'N' }, std::byte{ 'N' },
		std::byte{ 'C' }, std::byte{ 'M' }, std::byte{ '0' }, std::byte{ 0 },
	};
	constexpr std::array<std::byte, 8> kSeparatedMetadataMagic = {
		std::byte{ 'L' }, std::byte{ 'T' }, std::byte{ 'N' }, std::byte{ 'N' },
		std::byte{ 'S' }, std::byte{ 'E' }, std::byte{ 'P' }, std::byte{ 0 },
	};
	constexpr std::uint32_t kRodataVersion = 4;
	constexpr std::uint32_t kSeparatedMetadataVersion = 2;
	constexpr std::uint32_t kRodataLittleEndian = 1;
	constexpr std::uint32_t kRodataBigEndian = 2;
	constexpr std::string_view kMetadataRegionName = "metadata";
	constexpr std::string_view kConstantsRegionName = "constants";
	constexpr std::string_view kWeightsRegionName = "weights";
	constexpr std::string_view kInstructionsRegionName = "instructions";

	using EntryFn = void (*)(void**, void**);

	struct NativeTargetConfig
	{
		std::string triple;
		std::string cpu;
		std::string features;
		std::unique_ptr<llvm::TargetMachine> targetMachine;
	};

	struct RodataMetadata
	{
		CompiledModuleBackend backend{ CompiledModuleBackend::CPUNative };
		std::vector<CompiledTensorSpec> inputSpecs;
		std::vector<CompiledTensorSpec> outputSpecs;
	};

	struct SeparatedMetadata
	{
		RodataMetadata legacyMetadata;
		std::vector<std::byte> legacyRodata;
		std::vector<CompiledModuleRegionInfo> regions;
		std::vector<CompiledModuleExternalTensorInfo> externalTensorInfos;
	};

	std::string ToString(llvm::Error error)
	{
		std::string message;
		llvm::raw_string_ostream os(message);
		llvm::logAllUnhandledErrors(std::move(error), os);
		return os.str();
	}

	template <typename T>
	T TakeExpected(llvm::Expected<T> expected, std::string_view what)
	{
		if (!expected)
		{
			throw std::runtime_error(std::string(what) + ": " + ToString(expected.takeError()));
		}
		return std::move(*expected);
	}

	void CheckError(llvm::Error error, std::string_view what)
	{
		if (error)
		{
			throw std::runtime_error(std::string(what) + ": " + ToString(std::move(error)));
		}
	}

	void InitializeNativeLLVM()
	{
		static const bool initialized = [] {
			llvm::InitializeNativeTarget();
			llvm::InitializeNativeTargetAsmPrinter();
			llvm::InitializeNativeTargetAsmParser();
			return true;
		}();
		(void) initialized;
	}

	NativeTargetConfig CreateNativeTargetMachine()
	{
		InitializeNativeLLVM();

		auto triple = llvm::sys::getDefaultTargetTriple();
		std::string error;
		const auto* target = llvm::TargetRegistry::lookupTarget(llvm::Triple(triple), error);
		if (!target)
		{
			throw std::runtime_error("Failed to lookup native LLVM target: " + error);
		}

		llvm::TargetOptions options;
		auto relocModel = std::optional<llvm::Reloc::Model>(llvm::Reloc::PIC_);
		auto codeModel = std::optional<llvm::CodeModel::Model>();
		const llvm::Triple targetTriple(triple);
		if ((targetTriple.isOSDarwin() && targetTriple.getArch() == llvm::Triple::aarch64) ||
		    (targetTriple.isOSWindows() && targetTriple.getArch() == llvm::Triple::x86_64))
		{
			// MCJIT may allocate code and data sections far apart from runtime symbols or
			// from each other. Use address materialization instead of short relative
			// relocations for hosts where the default model has known reach limits.
			codeModel = llvm::CodeModel::Large;
		}
		auto cpuName = llvm::sys::getHostCPUName();
		std::string cpu = cpuName.empty() ? std::string("generic") : cpuName.str();

		llvm::SubtargetFeatures hostFeatureSet;
		const auto hostFeatures = llvm::sys::getHostCPUFeatures();
		for (const auto& feature : hostFeatures)
		{
			hostFeatureSet.AddFeature(feature.getKey(), feature.getValue());
		}
		std::string features = hostFeatureSet.getString();

		auto targetMachine = std::unique_ptr<llvm::TargetMachine>(target->createTargetMachine(
		    targetTriple, cpu, features, options, relocModel, codeModel, llvm::CodeGenOptLevel::Aggressive));
		if (!targetMachine)
		{
			throw std::runtime_error("Failed to create native LLVM target machine");
		}

		return { std::move(triple), std::move(cpu), std::move(features), std::move(targetMachine) };
	}

	void ConfigureForNativeObject(llvm::Module& module, const NativeTargetConfig& config)
	{
		module.setTargetTriple(llvm::Triple(config.triple));
		module.setDataLayout(config.targetMachine->createDataLayout());
	}

	void ConfigureForJITObjectRelocations(llvm::Module& module)
	{
		const llvm::Triple targetTriple(module.getTargetTriple());
		if (!targetTriple.isOSWindows() || targetTriple.getArch() != llvm::Triple::x86_64)
		{
			return;
		}

		// Windows x64 unwind tables introduce IMAGE_REL_AMD64_ADDR32NB relocations in
		// COFF .pdata/.xdata. MCJIT/RuntimeDyld requires a linker-style ordered
		// section layout for those relocations and can abort with a fatal LLVM error
		// when loading in-memory AOT objects. LiteNN AOT kernels do not throw across
		// the compiled boundary, so keep generated objects unwind-table free.
		module.setUwtable(llvm::UWTableKind::None);
		for (auto& function : module)
		{
			if (function.isDeclaration())
			{
				continue;
			}
			function.setDoesNotThrow();
			function.setUWTableKind(llvm::UWTableKind::None);
		}
	}

	llvm::OptimizationLevel ToLLVMOptimizationLevel(std::uint8_t level)
	{
		switch (std::min<std::uint8_t>(level, 3))
		{
		case 0:
			return llvm::OptimizationLevel::O0;
		case 1:
			return llvm::OptimizationLevel::O1;
		case 2:
			return llvm::OptimizationLevel::O2;
		default:
			return llvm::OptimizationLevel::O3;
		}
	}

	void OptimizeLLVMModule(llvm::Module& module, llvm::TargetMachine& targetMachine, std::uint8_t optLevel)
	{
		if (optLevel == 0)
		{
			return;
		}

		llvm::LoopAnalysisManager loopAnalysisManager;
		llvm::FunctionAnalysisManager functionAnalysisManager;
		llvm::CGSCCAnalysisManager cgsccAnalysisManager;
		llvm::ModuleAnalysisManager moduleAnalysisManager;

		llvm::PassBuilder passBuilder(&targetMachine);
		passBuilder.registerModuleAnalyses(moduleAnalysisManager);
		passBuilder.registerCGSCCAnalyses(cgsccAnalysisManager);
		passBuilder.registerFunctionAnalyses(functionAnalysisManager);
		passBuilder.registerLoopAnalyses(loopAnalysisManager);
		passBuilder.crossRegisterProxies(loopAnalysisManager, functionAnalysisManager, cgsccAnalysisManager,
		                                 moduleAnalysisManager);

		auto modulePipeline = passBuilder.buildPerModuleDefaultPipeline(ToLLVMOptimizationLevel(optLevel));
		modulePipeline.run(module, moduleAnalysisManager);
	}

	std::vector<std::byte> EmitObjectFile(llvm::Module& module)
	{
		auto config = CreateNativeTargetMachine();
		ConfigureForNativeObject(module, config);
		ConfigureForJITObjectRelocations(module);

		if (llvm::verifyModule(module, &llvm::errs()))
		{
			throw std::runtime_error("LLVM module verification failed before object emission");
		}

		llvm::SmallVector<char, 0> buffer;
		llvm::raw_svector_ostream stream(buffer);
		llvm::legacy::PassManager passManager;
		if (config.targetMachine->addPassesToEmitFile(passManager, stream, nullptr, llvm::CodeGenFileType::ObjectFile))
		{
			throw std::runtime_error("Native target cannot emit object files");
		}
		passManager.run(module);

		std::vector<std::byte> bytes(buffer.size());
		std::memcpy(bytes.data(), buffer.data(), buffer.size());
		return bytes;
	}

	void AppendU32(std::vector<std::byte>& out, std::uint32_t value)
	{
		for (int i = 0; i < 4; ++i)
		{
			out.push_back(static_cast<std::byte>((value >> (i * 8)) & 0xffu));
		}
	}

	void AppendU64(std::vector<std::byte>& out, std::uint64_t value)
	{
		for (int i = 0; i < 8; ++i)
		{
			out.push_back(static_cast<std::byte>((value >> (i * 8)) & 0xffu));
		}
	}

	void AppendI64(std::vector<std::byte>& out, std::int64_t value)
	{
		AppendU64(out, std::bit_cast<std::uint64_t>(value));
	}

	void AppendI32(std::vector<std::byte>& out, std::int32_t value)
	{
		AppendU32(out, std::bit_cast<std::uint32_t>(value));
	}

	void AppendF32(std::vector<std::byte>& out, float value)
	{
		AppendU32(out, std::bit_cast<std::uint32_t>(value));
	}

	void AppendString(std::vector<std::byte>& out, std::string_view value)
	{
		AppendU64(out, static_cast<std::uint64_t>(value.size()));
		const auto* data = reinterpret_cast<const std::byte*>(value.data());
		out.insert(out.end(), data, data + value.size());
	}

	void AppendBytes(std::vector<std::byte>& out, std::span<const std::byte> value)
	{
		AppendU64(out, static_cast<std::uint64_t>(value.size()));
		out.insert(out.end(), value.begin(), value.end());
	}

	std::uint32_t ReadU32(std::span<const std::byte> bytes, std::size_t& offset)
	{
		if (offset + 4 > bytes.size())
		{
			throw std::runtime_error("Compiled module rodata is truncated");
		}
		std::uint32_t value = 0;
		for (int i = 0; i < 4; ++i)
		{
			value |= std::to_integer<std::uint32_t>(bytes[offset + i]) << (i * 8);
		}
		offset += 4;
		return value;
	}

	std::uint64_t ReadU64(std::span<const std::byte> bytes, std::size_t& offset)
	{
		if (offset + 8 > bytes.size())
		{
			throw std::runtime_error("Compiled module rodata is truncated");
		}
		std::uint64_t value = 0;
		for (int i = 0; i < 8; ++i)
		{
			value |= std::to_integer<std::uint64_t>(bytes[offset + i]) << (i * 8);
		}
		offset += 8;
		return value;
	}

	std::int64_t ReadI64(std::span<const std::byte> bytes, std::size_t& offset)
	{
		return std::bit_cast<std::int64_t>(ReadU64(bytes, offset));
	}

	std::int32_t ReadI32(std::span<const std::byte> bytes, std::size_t& offset)
	{
		return std::bit_cast<std::int32_t>(ReadU32(bytes, offset));
	}

	float ReadF32(std::span<const std::byte> bytes, std::size_t& offset)
	{
		return std::bit_cast<float>(ReadU32(bytes, offset));
	}

	std::string ReadString(std::span<const std::byte> bytes, std::size_t& offset)
	{
		const auto size = ReadU64(bytes, offset);
		if (size > std::numeric_limits<std::size_t>::max() || static_cast<std::size_t>(size) > bytes.size() - offset)
		{
			throw std::runtime_error("Compiled module rodata string is truncated");
		}
		const auto stringSize = static_cast<std::size_t>(size);
		std::string value(reinterpret_cast<const char*>(bytes.data() + offset), stringSize);
		offset += stringSize;
		return value;
	}

	std::vector<std::byte> ReadBytes(std::span<const std::byte> bytes, std::size_t& offset, std::string_view label)
	{
		const auto size = ReadU64(bytes, offset);
		if (size > std::numeric_limits<std::size_t>::max() || static_cast<std::size_t>(size) > bytes.size() - offset)
		{
			throw std::runtime_error(std::format("Compiled module {} region is truncated", label));
		}
		const auto byteSize = static_cast<std::size_t>(size);
		std::vector<std::byte> value(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
		                             bytes.begin() + static_cast<std::ptrdiff_t>(offset + byteSize));
		offset += byteSize;
		return value;
	}

	std::uint64_t ChecksumBytes(std::span<const std::byte> bytes)
	{
		std::uint64_t hash = 1469598103934665603ull;
		for (const auto byte : bytes)
		{
			hash ^= std::to_integer<std::uint8_t>(byte);
			hash *= 1099511628211ull;
		}
		return hash;
	}

	CompiledModuleRegionInfo MakeRegionInfo(std::string_view name, std::span<const std::byte> bytes,
	                                        std::uint64_t alignment = 1)
	{
		return {
			.name = std::string(name),
			.size = static_cast<std::uint64_t>(bytes.size()),
			.alignment = alignment,
			.checksum = ChecksumBytes(bytes),
		};
	}

	void AppendQuantizationParams(std::vector<std::byte>& out, const QuantizationParams& params)
	{
		AppendU32(out, static_cast<std::uint32_t>(params.scheme));
		AppendU32(out, static_cast<std::uint32_t>(params.granularity));
		AppendU32(out, static_cast<std::uint32_t>(params.blockFormat));
		AppendU32(out, static_cast<std::uint32_t>(params.packedFormat));
		AppendU32(out, static_cast<std::uint32_t>(params.packedOrder));
		AppendU32(out, static_cast<std::uint32_t>(params.blockScaleLayout));
		AppendU32(out, static_cast<std::uint32_t>(params.storageType));
		AppendU32(out, static_cast<std::uint32_t>(params.expressedType));
		AppendI64(out, params.axis);
		AppendU64(out, static_cast<std::uint64_t>(params.groupSize));

		AppendU64(out, static_cast<std::uint64_t>(params.scales.size()));
		for (const auto scale : params.scales)
		{
			AppendF32(out, scale);
		}

		AppendU64(out, static_cast<std::uint64_t>(params.zeroPoints.size()));
		for (const auto zeroPoint : params.zeroPoints)
		{
			AppendI32(out, zeroPoint);
		}

		AppendU64(out, static_cast<std::uint64_t>(params.expressedShape.size()));
		for (const auto dim : params.expressedShape)
		{
			AppendU64(out, static_cast<std::uint64_t>(dim));
		}
	}

	QuantizationParams ReadQuantizationParams(std::span<const std::byte> bytes, std::size_t& offset)
	{
		QuantizationParams params;
		params.scheme = static_cast<QuantizationScheme>(ReadU32(bytes, offset));
		params.granularity = static_cast<QuantizationGranularity>(ReadU32(bytes, offset));
		params.blockFormat = static_cast<QuantizedBlockFormat>(ReadU32(bytes, offset));
		params.packedFormat = static_cast<PackedNibbleFormat>(ReadU32(bytes, offset));
		params.packedOrder = static_cast<PackedNibbleOrder>(ReadU32(bytes, offset));
		params.blockScaleLayout = static_cast<BlockScaleLayout>(ReadU32(bytes, offset));
		params.storageType = static_cast<DataType>(ReadU32(bytes, offset));
		params.expressedType = static_cast<DataType>(ReadU32(bytes, offset));
		params.axis = ReadI64(bytes, offset);
		params.groupSize = static_cast<std::size_t>(ReadU64(bytes, offset));

		const auto scaleCount = ReadU64(bytes, offset);
		if (scaleCount > std::numeric_limits<std::size_t>::max())
		{
			throw std::runtime_error("Compiled module rodata quantization scale count is too large");
		}
		params.scales.reserve(static_cast<std::size_t>(scaleCount));
		for (std::uint64_t i = 0; i < scaleCount; ++i)
		{
			params.scales.push_back(ReadF32(bytes, offset));
		}

		const auto zeroPointCount = ReadU64(bytes, offset);
		if (zeroPointCount > std::numeric_limits<std::size_t>::max())
		{
			throw std::runtime_error("Compiled module rodata quantization zero-point count is too large");
		}
		params.zeroPoints.reserve(static_cast<std::size_t>(zeroPointCount));
		for (std::uint64_t i = 0; i < zeroPointCount; ++i)
		{
			params.zeroPoints.push_back(ReadI32(bytes, offset));
		}

		const auto expressedRank = ReadU64(bytes, offset);
		if (expressedRank > std::numeric_limits<std::size_t>::max())
		{
			throw std::runtime_error("Compiled module rodata quantization expressed rank is too large");
		}
		params.expressedShape.reserve(static_cast<std::size_t>(expressedRank));
		for (std::uint64_t i = 0; i < expressedRank; ++i)
		{
			const auto dim = ReadU64(bytes, offset);
			if (dim > std::numeric_limits<std::size_t>::max())
			{
				throw std::runtime_error("Compiled module rodata quantization expressed dimension is too large");
			}
			params.expressedShape.push_back(static_cast<std::size_t>(dim));
		}
		return params;
	}

	std::uint32_t NativeEndianTag()
	{
		if constexpr (std::endian::native == std::endian::little)
		{
			return kRodataLittleEndian;
		}
		else if constexpr (std::endian::native == std::endian::big)
		{
			return kRodataBigEndian;
		}
		else
		{
			throw std::runtime_error("Unsupported mixed-endian target");
		}
	}

	CompiledModuleBackend DecodeBackend(std::uint32_t value)
	{
		switch (value)
		{
		case static_cast<std::uint32_t>(CompiledModuleBackend::CPUNative):
			return CompiledModuleBackend::CPUNative;
		case static_cast<std::uint32_t>(CompiledModuleBackend::CUDANative):
			return CompiledModuleBackend::CUDANative;
		case static_cast<std::uint32_t>(CompiledModuleBackend::VulkanNative):
			return CompiledModuleBackend::VulkanNative;
		default:
			throw std::runtime_error("Compiled module rodata contains an invalid backend");
		}
	}

	std::vector<std::byte> SerializeRodata(std::span<const CompiledTensorSpec> inputs,
	                                       std::span<const CompiledTensorSpec> outputs, std::string_view targetTriple,
	                                       CompiledModuleBackend backend)
	{
		std::vector<std::byte> rodata;
		rodata.insert(rodata.end(), kRodataMagic.begin(), kRodataMagic.end());
		AppendU32(rodata, kRodataVersion);
		AppendU32(rodata, static_cast<std::uint32_t>(sizeof(void*)));
		AppendU32(rodata, NativeEndianTag());
		AppendString(rodata, targetTriple);
		AppendU32(rodata, static_cast<std::uint32_t>(backend));
		AppendU32(rodata, static_cast<std::uint32_t>(inputs.size()));
		AppendU32(rodata, static_cast<std::uint32_t>(outputs.size()));

		const auto appendSpec = [&](const CompiledTensorSpec& spec) {
			const auto shape = spec.type.StaticShape();
			AppendU32(rodata, static_cast<std::uint32_t>(spec.type.dtype));
			AppendU32(rodata, static_cast<std::uint32_t>(shape.size()));
			for (auto dim : shape)
			{
				AppendU64(rodata, static_cast<std::uint64_t>(dim));
			}
			AppendString(rodata, spec.name);
			AppendU32(rodata, spec.quantization ? 1u : 0u);
			if (spec.quantization)
			{
				AppendQuantizationParams(rodata, *spec.quantization);
			}
		};

		for (const auto& spec : inputs)
		{
			appendSpec(spec);
		}
		for (const auto& spec : outputs)
		{
			appendSpec(spec);
		}
		return rodata;
	}

	RodataMetadata DeserializeRodata(std::span<const std::byte> rodata)
	{
		if (rodata.size() < kRodataMagic.size() ||
		    !std::equal(kRodataMagic.begin(), kRodataMagic.end(), rodata.begin()))
		{
			throw std::runtime_error("Compiled module rodata has an invalid magic header");
		}

		std::size_t offset = kRodataMagic.size();
		const auto version = ReadU32(rodata, offset);
		if (version == 0 || version > kRodataVersion)
		{
			throw std::runtime_error("Unsupported compiled module rodata version");
		}

		if (version >= 2)
		{
			const auto pointerSize = ReadU32(rodata, offset);
			if (pointerSize != sizeof(void*))
			{
				throw std::runtime_error("Compiled module rodata pointer size does not match this process");
			}
			const auto endianTag = ReadU32(rodata, offset);
			if (endianTag != NativeEndianTag())
			{
				throw std::runtime_error("Compiled module rodata endianness does not match this process");
			}
			const auto targetTriple = ReadString(rodata, offset);
			if (targetTriple != llvm::sys::getDefaultTargetTriple())
			{
				throw std::runtime_error("Compiled module rodata target triple does not match this process");
			}
		}
		CompiledModuleBackend backend = CompiledModuleBackend::CPUNative;
		if (version >= 3)
		{
			backend = DecodeBackend(ReadU32(rodata, offset));
		}

		const auto inputCount = ReadU32(rodata, offset);
		const auto outputCount = ReadU32(rodata, offset);

		const auto readSpec = [&]() {
			CompiledTensorSpec spec;
			const auto dtypeValue = ReadU32(rodata, offset);
			if (dtypeValue > static_cast<std::uint32_t>(LastDataType) ||
			    !IsValidDataTypeValue(static_cast<DataType>(dtypeValue)))
			{
				throw std::runtime_error("Compiled module rodata contains an invalid data type");
			}
			const auto dtype = static_cast<DataType>(dtypeValue);

			const auto rank = ReadU32(rodata, offset);
			std::vector<std::size_t> shape;
			shape.reserve(rank);
			for (std::uint32_t i = 0; i < rank; ++i)
			{
				const auto dim = ReadU64(rodata, offset);
				if (dim > std::numeric_limits<std::size_t>::max())
				{
					throw std::runtime_error("Compiled module rodata shape dimension is too large");
				}
				shape.push_back(static_cast<std::size_t>(dim));
			}
			spec.type = TensorType::Dense(dtype, ShapeView{ shape });
			if (version >= 2)
			{
				spec.name = ReadString(rodata, offset);
			}
			if (version >= 4 && ReadU32(rodata, offset) != 0)
			{
				spec.quantization = ReadQuantizationParams(rodata, offset);
				try
				{
					ValidateQuantizationParams(*spec.quantization, ShapeView{ shape }, dtype);
				}
				catch (const std::exception& ex)
				{
					throw std::runtime_error(
					    std::format("Compiled module rodata quantization metadata is invalid: {}", ex.what()));
				}
			}
			return spec;
		};

		std::vector<CompiledTensorSpec> inputs;
		inputs.reserve(inputCount);
		for (std::uint32_t i = 0; i < inputCount; ++i)
		{
			inputs.push_back(readSpec());
		}

		std::vector<CompiledTensorSpec> outputs;
		outputs.reserve(outputCount);
		for (std::uint32_t i = 0; i < outputCount; ++i)
		{
			outputs.push_back(readSpec());
		}

		if (offset != rodata.size())
		{
			throw std::runtime_error("Compiled module rodata contains trailing bytes");
		}

		return {
			.backend = backend,
			.inputSpecs = std::move(inputs),
			.outputSpecs = std::move(outputs),
		};
	}

	void AppendRegionInfo(std::vector<std::byte>& out, const CompiledModuleRegionInfo& info)
	{
		AppendString(out, info.name);
		AppendU64(out, info.size);
		AppendU64(out, info.alignment);
		AppendU64(out, info.checksum);
	}

	CompiledModuleRegionInfo ReadRegionInfo(std::span<const std::byte> bytes, std::size_t& offset)
	{
		auto info = CompiledModuleRegionInfo{
			.name = ReadString(bytes, offset),
			.size = ReadU64(bytes, offset),
			.alignment = ReadU64(bytes, offset),
			.checksum = ReadU64(bytes, offset),
		};
		if (info.name.empty())
		{
			throw std::runtime_error("Compiled module separated metadata contains an empty region name");
		}
		if (info.alignment == 0)
		{
			throw std::runtime_error(
			    std::format("Compiled module separated metadata region '{}' has zero alignment", info.name));
		}
		return info;
	}

	CompiledModuleExternalTensorRebindPolicy DecodeExternalTensorRebindPolicy(std::uint32_t value)
	{
		switch (value)
		{
		case static_cast<std::uint32_t>(CompiledModuleExternalTensorRebindPolicy::ExactChecksum):
			return CompiledModuleExternalTensorRebindPolicy::ExactChecksum;
		default:
			throw std::runtime_error(
			    "Compiled module separated metadata contains an invalid external tensor rebind policy");
		}
	}

	void AppendExternalTensorInfo(std::vector<std::byte>& out, const CompiledModuleExternalTensorInfo& info)
	{
		AppendString(out, info.name);
		AppendString(out, info.region);
		AppendU32(out, static_cast<std::uint32_t>(info.type.dtype));
		const auto shape = info.type.StaticShape();
		AppendU32(out, static_cast<std::uint32_t>(shape.size()));
		for (const auto dim : shape)
		{
			AppendU64(out, static_cast<std::uint64_t>(dim));
		}
		AppendU64(out, info.byteOffset);
		AppendU64(out, info.byteSize);
		AppendU64(out, info.alignment);
		AppendU64(out, info.checksum);
		AppendU32(out, static_cast<std::uint32_t>(info.rebindPolicy));
	}

	CompiledModuleExternalTensorInfo ReadExternalTensorInfo(std::span<const std::byte> bytes, std::size_t& offset)
	{
		CompiledModuleExternalTensorInfo info;
		info.name = ReadString(bytes, offset);
		info.region = ReadString(bytes, offset);
		const auto dtypeValue = ReadU32(bytes, offset);
		if (dtypeValue > static_cast<std::uint32_t>(LastDataType) ||
		    !IsValidDataTypeValue(static_cast<DataType>(dtypeValue)))
		{
			throw std::runtime_error(
			    "Compiled module separated metadata contains an invalid external tensor data type");
		}
		const auto dtype = static_cast<DataType>(dtypeValue);
		const auto rank = ReadU32(bytes, offset);
		std::vector<std::size_t> shape;
		shape.reserve(rank);
		for (std::uint32_t i = 0; i < rank; ++i)
		{
			const auto dim = ReadU64(bytes, offset);
			if (dim > std::numeric_limits<std::size_t>::max())
			{
				throw std::runtime_error(
				    "Compiled module separated metadata external tensor shape dimension is too large");
			}
			shape.push_back(static_cast<std::size_t>(dim));
		}
		info.type = TensorType::Dense(dtype, ShapeView{ shape });
		info.byteOffset = ReadU64(bytes, offset);
		info.byteSize = ReadU64(bytes, offset);
		info.alignment = ReadU64(bytes, offset);
		info.checksum = ReadU64(bytes, offset);
		info.rebindPolicy = DecodeExternalTensorRebindPolicy(ReadU32(bytes, offset));
		if (info.name.empty())
		{
			throw std::runtime_error("Compiled module separated metadata contains an empty external tensor name");
		}
		if (info.region.empty())
		{
			throw std::runtime_error("Compiled module separated metadata contains an empty external tensor region");
		}
		if (info.byteSize == 0)
		{
			throw std::runtime_error(
			    std::format("Compiled module separated metadata external tensor '{}' has zero byte size", info.name));
		}
		if (info.alignment == 0)
		{
			throw std::runtime_error(
			    std::format("Compiled module separated metadata external tensor '{}' has zero alignment", info.name));
		}
		return info;
	}

	std::vector<CompiledModuleRegionInfo> MakeSeparatedRegionInfos(std::span<const std::byte> constants,
	                                                               std::span<const std::byte> weights,
	                                                               std::span<const std::byte> instructions)
	{
		return {
			MakeRegionInfo(kConstantsRegionName, constants),
			MakeRegionInfo(kWeightsRegionName, weights),
			MakeRegionInfo(kInstructionsRegionName, instructions),
		};
	}

	std::vector<std::byte>
	SerializeSeparatedMetadata(std::span<const std::byte> legacyRodata, std::span<const std::byte> constants,
	                           std::span<const std::byte> weights, std::span<const std::byte> instructions,
	                           std::span<const CompiledModuleExternalTensorInfo> externalTensorInfos)
	{
		std::vector<std::byte> metadata;
		metadata.insert(metadata.end(), kSeparatedMetadataMagic.begin(), kSeparatedMetadataMagic.end());
		AppendU32(metadata, kSeparatedMetadataVersion);
		AppendU32(metadata, static_cast<std::uint32_t>(sizeof(void*)));
		AppendU32(metadata, NativeEndianTag());
		AppendBytes(metadata, legacyRodata);

		const auto regions = MakeSeparatedRegionInfos(constants, weights, instructions);
		AppendU32(metadata, static_cast<std::uint32_t>(regions.size()));
		for (const auto& region : regions)
		{
			AppendRegionInfo(metadata, region);
		}

		AppendU32(metadata, static_cast<std::uint32_t>(externalTensorInfos.size()));
		for (const auto& info : externalTensorInfos)
		{
			AppendExternalTensorInfo(metadata, info);
		}
		return metadata;
	}

	const CompiledModuleRegionInfo& FindRegionInfo(std::span<const CompiledModuleRegionInfo> regions,
	                                               std::string_view name)
	{
		for (const auto& region : regions)
		{
			if (region.name == name)
			{
				return region;
			}
		}
		throw std::runtime_error(std::format("Compiled module separated metadata is missing '{}' region", name));
	}

	void ValidateExternalTensorInfoRanges(std::span<const CompiledModuleExternalTensorInfo> infos,
	                                      std::span<const CompiledModuleRegionInfo> regions)
	{
		for (const auto& info : infos)
		{
			if (info.region != kConstantsRegionName && info.region != kWeightsRegionName)
			{
				throw std::runtime_error(std::format(
				    "Compiled module separated metadata external tensor '{}' references unsupported '{}' region",
				    info.name, info.region));
			}
			const auto& region = FindRegionInfo(regions, info.region);
			if (info.byteOffset > region.size || info.byteSize > region.size - info.byteOffset)
			{
				throw std::runtime_error(std::format(
				    "Compiled module separated metadata external tensor '{}' byte range is out of bounds", info.name));
			}
			if (info.byteOffset % info.alignment != 0)
			{
				throw std::runtime_error(std::format(
				    "Compiled module separated metadata external tensor '{}' offset is not aligned to {} bytes",
				    info.name, info.alignment));
			}
		}
	}

	SeparatedMetadata DeserializeSeparatedMetadata(std::span<const std::byte> metadata)
	{
		if (metadata.size() < kSeparatedMetadataMagic.size() ||
		    !std::equal(kSeparatedMetadataMagic.begin(), kSeparatedMetadataMagic.end(), metadata.begin()))
		{
			throw std::runtime_error("Compiled module separated metadata has an invalid magic header");
		}

		std::size_t offset = kSeparatedMetadataMagic.size();
		const auto version = ReadU32(metadata, offset);
		if (version == 0 || version > kSeparatedMetadataVersion)
		{
			throw std::runtime_error("Unsupported compiled module separated metadata version");
		}
		const auto pointerSize = ReadU32(metadata, offset);
		if (pointerSize != sizeof(void*))
		{
			throw std::runtime_error("Compiled module separated metadata pointer size does not match this process");
		}
		const auto endianTag = ReadU32(metadata, offset);
		if (endianTag != NativeEndianTag())
		{
			throw std::runtime_error("Compiled module separated metadata endianness does not match this process");
		}

		auto legacyRodata = ReadBytes(metadata, offset, "legacy metadata");
		auto legacyMetadata = DeserializeRodata(legacyRodata);
		const auto regionCount = ReadU32(metadata, offset);
		std::vector<CompiledModuleRegionInfo> regions;
		regions.reserve(regionCount);
		for (std::uint32_t i = 0; i < regionCount; ++i)
		{
			auto info = ReadRegionInfo(metadata, offset);
			for (const auto& existing : regions)
			{
				if (existing.name == info.name)
				{
					throw std::runtime_error(
					    std::format("Compiled module separated metadata contains duplicate '{}' regions", info.name));
				}
			}
			regions.push_back(std::move(info));
		}
		std::vector<CompiledModuleExternalTensorInfo> externalTensorInfos;
		if (version >= 2)
		{
			const auto externalTensorCount = ReadU32(metadata, offset);
			externalTensorInfos.reserve(externalTensorCount);
			for (std::uint32_t i = 0; i < externalTensorCount; ++i)
			{
				externalTensorInfos.push_back(ReadExternalTensorInfo(metadata, offset));
			}
		}
		if (offset != metadata.size())
		{
			throw std::runtime_error("Compiled module separated metadata contains trailing bytes");
		}

		(void) FindRegionInfo(regions, kConstantsRegionName);
		(void) FindRegionInfo(regions, kWeightsRegionName);
		(void) FindRegionInfo(regions, kInstructionsRegionName);
		ValidateExternalTensorInfoRanges(externalTensorInfos, regions);
		return {
			.legacyMetadata = std::move(legacyMetadata),
			.legacyRodata = std::move(legacyRodata),
			.regions = std::move(regions),
			.externalTensorInfos = std::move(externalTensorInfos),
		};
	}

	std::span<const std::byte> RegionBytes(CompiledModuleRegion region, std::string_view name)
	{
		if (region.size != 0 && region.data == nullptr)
		{
			throw std::runtime_error(
			    std::format("Compiled module separated '{}' region has a null data pointer", name));
		}
		return { static_cast<const std::byte*>(region.data), region.size };
	}

	void ValidateSeparatedRegion(CompiledModuleRegion region, const CompiledModuleRegionInfo& expected)
	{
		const auto bytes = RegionBytes(region, expected.name);
		if (bytes.size() != expected.size)
		{
			throw std::runtime_error(
			    std::format("Compiled module separated '{}' region size mismatch: expected {}, got {}", expected.name,
			                expected.size, bytes.size()));
		}
		if (expected.alignment > 1 && region.size != 0 &&
		    reinterpret_cast<std::uintptr_t>(region.data) % expected.alignment != 0)
		{
			throw std::runtime_error(std::format("Compiled module separated '{}' region is not aligned to {} bytes",
			                                     expected.name, expected.alignment));
		}
		const auto checksum = ChecksumBytes(bytes);
		if (checksum != expected.checksum)
		{
			throw std::runtime_error(
			    std::format("Compiled module separated '{}' region checksum mismatch", expected.name));
		}
	}

	std::span<const std::byte> SeparatedImageRegionBytes(CompiledModuleSeparatedImage image, std::string_view name)
	{
		if (name == kConstantsRegionName)
		{
			return RegionBytes(image.constants, kConstantsRegionName);
		}
		if (name == kWeightsRegionName)
		{
			return RegionBytes(image.weights, kWeightsRegionName);
		}
		throw std::runtime_error(
		    std::format("Compiled module separated external tensor references invalid '{}' region", name));
	}

	void ValidateExternalTensorChecksums(std::span<const CompiledModuleExternalTensorInfo> infos,
	                                     CompiledModuleSeparatedImage image)
	{
		for (const auto& info : infos)
		{
			const auto region = SeparatedImageRegionBytes(image, info.region);
			if (info.byteOffset > region.size() || info.byteSize > region.size() - info.byteOffset)
			{
				throw std::runtime_error(std::format(
				    "Compiled module separated external tensor '{}' byte range is out of bounds", info.name));
			}
			const auto tensorBytes =
			    std::span<const std::byte>{ region.data() + static_cast<std::ptrdiff_t>(info.byteOffset),
				                            static_cast<std::size_t>(info.byteSize) };
			if (ChecksumBytes(tensorBytes) != info.checksum)
			{
				throw std::runtime_error(
				    std::format("Compiled module separated external tensor '{}' checksum mismatch", info.name));
			}
		}
	}

	SeparatedMetadata ValidateSeparatedImage(CompiledModuleSeparatedImage image)
	{
		const auto metadataBytes = RegionBytes(image.metadata, kMetadataRegionName);
		auto metadata = DeserializeSeparatedMetadata(metadataBytes);
		ValidateSeparatedRegion(image.constants, FindRegionInfo(metadata.regions, kConstantsRegionName));
		ValidateSeparatedRegion(image.weights, FindRegionInfo(metadata.regions, kWeightsRegionName));
		ValidateSeparatedRegion(image.instructions, FindRegionInfo(metadata.regions, kInstructionsRegionName));
		ValidateExternalTensorChecksums(metadata.externalTensorInfos, image);
		return metadata;
	}

	std::vector<CompiledTensorSpec> BuildInputSpecs(const Graph& graph)
	{
		const auto signature = graph.InputTypeSignature();
		std::vector<CompiledTensorSpec> specs;
		specs.reserve(signature.size());
		for (const auto& input : signature)
		{
			specs.push_back(CompiledTensorSpec::FromType(input.name, input.type));
		}
		return specs;
	}

	std::optional<QuantizationParams> InferOutputQuantization(const Graph& graph, const Subgraph& subgraph,
	                                                          NodeOutput output)
	{
		const auto& entry = subgraph.GetNodeEntry(output.node);
		return std::visit(
		    [&](const auto& node) -> std::optional<QuantizationParams> {
			    using T = std::decay_t<decltype(node)>;
			    if constexpr (std::same_as<T, QuantizedConstantNode>)
			    {
				    return node.params;
			    }
			    else if constexpr (std::same_as<T, QuantizeNode>)
			    {
				    return node.params;
			    }
			    else if constexpr (std::same_as<T, VariableRefNode>)
			    {
				    return graph.GetVariable(node.variableIndex)->Quantization();
			    }
			    else
			    {
				    return std::nullopt;
			    }
		    },
		    entry.node);
	}

	std::vector<CompiledTensorSpec> BuildOutputSpecs(const Graph& graph)
	{
		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		const auto signature = graph.OutputTypeSignature();
		std::vector<CompiledTensorSpec> specs;
		specs.reserve(signature.size());
		for (std::size_t i = 0; i < signature.size(); ++i)
		{
			const auto output = subgraph.Results()[i];
			specs.push_back(CompiledTensorSpec::FromType(signature[i].name, signature[i].type,
			                                             InferOutputQuantization(graph, subgraph, output)));
		}
		return specs;
	}

	struct CompiledArtifactParts
	{
		std::vector<std::byte> rodata;
		std::vector<std::byte> instructions;
		std::vector<std::byte> constants;
		std::vector<std::byte> weights;
		std::vector<CompiledModuleExternalTensorInfo> externalTensorInfos;
		std::vector<CompiledTensorSpec> inputSpecs;
		std::vector<CompiledTensorSpec> outputSpecs;
		CompiledModuleBackend backend{ CompiledModuleBackend::CPUNative };
	};

	CompiledArtifactParts
	MakeCompiledArtifactParts(std::vector<std::byte> rodata, std::vector<std::byte> instructions,
	                          std::vector<CompiledTensorSpec> inputSpecs, std::vector<CompiledTensorSpec> outputSpecs,
	                          CompiledModuleBackend backend, std::vector<std::byte> constants = {},
	                          std::vector<std::byte> weights = {},
	                          std::vector<CompiledModuleExternalTensorInfo> externalTensorInfos = {})
	{
		CompiledArtifactParts parts;
		parts.rodata = std::move(rodata);
		parts.instructions = std::move(instructions);
		parts.constants = std::move(constants);
		parts.weights = std::move(weights);
		parts.externalTensorInfos = std::move(externalTensorInfos);
		parts.inputSpecs = std::move(inputSpecs);
		parts.outputSpecs = std::move(outputSpecs);
		parts.backend = backend;
		return parts;
	}

	std::uint64_t SaturatedMulU64(std::uint64_t lhs, std::uint64_t rhs)
	{
		if (lhs == 0 || rhs == 0)
		{
			return 0;
		}
		if (lhs > std::numeric_limits<std::uint64_t>::max() / rhs)
		{
			return std::numeric_limits<std::uint64_t>::max();
		}
		return lhs * rhs;
	}

	std::uint64_t SaturatedAddU64(std::uint64_t lhs, std::uint64_t rhs)
	{
		if (lhs > std::numeric_limits<std::uint64_t>::max() - rhs)
		{
			return std::numeric_limits<std::uint64_t>::max();
		}
		return lhs + rhs;
	}

	std::optional<std::uint64_t> ShapeNumElementsU64(std::span<const std::size_t> shape)
	{
		std::uint64_t count = 1;
		for (const auto dim : shape)
		{
			if (dim == 0)
			{
				return std::nullopt;
			}
			if (count > std::numeric_limits<std::uint64_t>::max() / static_cast<std::uint64_t>(dim))
			{
				return std::nullopt;
			}
			count *= static_cast<std::uint64_t>(dim);
		}
		return count;
	}

	std::uint64_t TensorByteSizeForShape(DataType dtype, std::span<const std::size_t> shape)
	{
		const auto elements = ShapeNumElementsU64(shape);
		if (!elements)
		{
			throw std::runtime_error("Compiled tensor shape is too large");
		}
		return *elements * LiteNN::ElementByteSize(dtype);
	}

	bool IsSameRankBroadcastCompatibleShape(std::span<const std::size_t> lhs, std::span<const std::size_t> rhs,
	                                        std::span<const std::size_t> output)
	{
		if (lhs.size() != output.size() || rhs.size() != output.size())
		{
			return false;
		}
		for (std::size_t i = 0; i < output.size(); ++i)
		{
			if ((lhs[i] != output[i] && lhs[i] != 1) || (rhs[i] != output[i] && rhs[i] != 1) ||
			    output[i] != std::max(lhs[i], rhs[i]))
			{
				return false;
			}
		}
		return true;
	}

	std::optional<std::vector<float>> CopyF32TensorData(const Tensor<PolymorphicDevice>& tensor,
	                                                    std::span<const std::size_t> expectedShape)
	{
		const auto cpuTensor = tensor.CopyToDevice(CPU{});
		if (cpuTensor.DType() != DataType::Float32 || cpuTensor.Shape() != expectedShape)
		{
			return std::nullopt;
		}
		std::vector<float> values(cpuTensor.NumElements());
		std::memcpy(values.data(), cpuTensor.UnsafeRawData(), values.size() * sizeof(float));
		return values;
	}

	llvm::Value* AddF32ConstantGlobal(llvm::Module& module, llvm::IRBuilder<>& builder, std::string_view name,
	                                  std::span<const float> values)
	{
		auto& ctx = module.getContext();
		auto* arrayType = llvm::ArrayType::get(llvm::Type::getFloatTy(ctx), values.size());
		auto* init = llvm::ConstantDataArray::get(ctx, llvm::ArrayRef(values.data(), values.size()));
		auto* global = new llvm::GlobalVariable(module, arrayType, true, llvm::GlobalValue::PrivateLinkage, init,
		                                        std::string(name));
		global->setAlignment(llvm::Align(64));
		auto* zero = builder.getInt64(0);
		return builder.CreateInBoundsGEP(arrayType, global, { zero, zero });
	}

	std::uint64_t AlignUpU64(std::uint64_t value, std::uint64_t alignment)
	{
		return ((value + alignment - 1) / alignment) * alignment;
	}

	std::uint64_t AppendExternalF32Region(std::vector<std::byte>& bytes, std::span<const float> values)
	{
		const auto offset = AlignUpU64(static_cast<std::uint64_t>(bytes.size()), 64);
		if (bytes.size() < offset)
		{
			bytes.resize(static_cast<std::size_t>(offset));
		}
		const auto byteSize = values.size_bytes();
		const auto oldSize = bytes.size();
		bytes.resize(oldSize + byteSize);
		std::memcpy(bytes.data() + oldSize, values.data(), byteSize);
		return offset;
	}

	std::uint64_t AppendExternalRegionBytes(std::vector<std::byte>& bytes, std::span<const std::byte> payload,
	                                        std::uint64_t alignment = 64)
	{
		const auto offset = AlignUpU64(static_cast<std::uint64_t>(bytes.size()), alignment);
		if (bytes.size() < offset)
		{
			bytes.resize(static_cast<std::size_t>(offset));
		}
		const auto oldSize = bytes.size();
		bytes.resize(oldSize + payload.size());
		std::memcpy(bytes.data() + oldSize, payload.data(), payload.size());
		return offset;
	}

	llvm::Value* AddExternalRegionPointer(llvm::Module& module, llvm::IRBuilder<>& builder, std::string_view symbol,
	                                      std::uint64_t offset)
	{
		auto& ctx = module.getContext();
		auto* ptrTy = llvm::PointerType::get(ctx, 0);
		auto* i8Ty = llvm::Type::getInt8Ty(ctx);
		auto externalRegionFn =
		    module.getOrInsertFunction(std::string(symbol), llvm::FunctionType::get(ptrTy, {}, false));
		auto* base = builder.CreateCall(externalRegionFn);
		return builder.CreateInBoundsGEP(i8Ty, base, builder.getInt64(offset));
	}

	CompiledModuleExternalTensorInfo MakeExternalF32TensorInfo(std::string name, std::string_view regionName,
	                                                           std::span<const std::byte> regionBytes,
	                                                           std::span<const std::size_t> shape,
	                                                           std::uint64_t byteOffset, std::uint64_t byteSize,
	                                                           std::uint64_t alignment)
	{
		const auto bytes = std::span<const std::byte>{
			regionBytes.data() + static_cast<std::ptrdiff_t>(byteOffset),
			static_cast<std::size_t>(byteSize),
		};
		return {
			.name = std::move(name),
			.region = std::string(regionName),
			.type = TensorType::Dense(DataType::Float32, ShapeView{ shape }),
			.byteOffset = byteOffset,
			.byteSize = byteSize,
			.alignment = alignment,
			.checksum = ChecksumBytes(bytes),
			.rebindPolicy = CompiledModuleExternalTensorRebindPolicy::ExactChecksum,
		};
	}

	CompiledModuleExternalTensorInfo MakeExternalTensorInfo(std::string name, std::string_view regionName,
	                                                        DataType dtype, std::span<const std::byte> regionBytes,
	                                                        std::span<const std::size_t> shape,
	                                                        std::uint64_t byteOffset, std::uint64_t byteSize,
	                                                        std::uint64_t alignment)
	{
		const auto bytes = std::span<const std::byte>{
			regionBytes.data() + static_cast<std::ptrdiff_t>(byteOffset),
			static_cast<std::size_t>(byteSize),
		};
		return {
			.name = std::move(name),
			.region = std::string(regionName),
			.type = TensorType::Dense(dtype, ShapeView{ shape }),
			.byteOffset = byteOffset,
			.byteSize = byteSize,
			.alignment = alignment,
			.checksum = ChecksumBytes(bytes),
			.rebindPolicy = CompiledModuleExternalTensorRebindPolicy::ExactChecksum,
		};
	}

	bool CanExternalizeCPUTensorInMLIR(DataType dtype)
	{
		// MLIR lowers memref<i1> with bit-level element semantics, while LiteNN CPU
		// tensors store Bool as byte-addressable host values.
		return dtype != DataType::Bool;
	}

	std::optional<std::uint64_t> AppendTensorPayloadBytes(std::vector<std::byte>& bytes,
	                                                      const Tensor<PolymorphicDevice>& tensor,
	                                                      DataType expectedDType,
	                                                      std::span<const std::size_t> expectedShape,
	                                                      std::uint64_t alignment)
	{
		if (!CanExternalizeCPUTensorInMLIR(expectedDType))
		{
			return std::nullopt;
		}
		if (tensor.DType() != expectedDType || tensor.Shape() != expectedShape)
		{
			return std::nullopt;
		}

		const auto byteSize = tensor.NumElements() * LiteNN::ElementByteSize(expectedDType);
		const auto offset = AlignUpU64(static_cast<std::uint64_t>(bytes.size()), alignment);
		if (bytes.size() < offset)
		{
			bytes.resize(static_cast<std::size_t>(offset));
		}
		const auto oldSize = bytes.size();
		bytes.resize(oldSize + byteSize);
		if (tensor.CurDevice().template As<CPU>() != nullptr)
		{
			std::memcpy(bytes.data() + oldSize, tensor.UnsafeRawData(), byteSize);
		}
		else
		{
			const auto cpuTensor = tensor.CopyToDevice(CPU{});
			std::memcpy(bytes.data() + oldSize, cpuTensor.UnsafeRawData(), byteSize);
		}
		return offset;
	}

	struct CPUMLIRExternalizedGraph
	{
		Graph graph;
		std::vector<std::byte> constants;
		std::vector<std::byte> weights;
		std::vector<CompiledModuleExternalTensorInfo> externalTensorInfos;
		std::vector<CompiledModuleExternalTensorInfo> entryExternalTensorInfos;
	};

	void AppendUniqueExternalId(std::vector<std::size_t>& ids, std::size_t id)
	{
		if (!std::ranges::contains(ids, id))
		{
			ids.push_back(id);
		}
	}

	bool AppendUniqueExternalIds(std::vector<std::size_t>& ids, std::span<const std::size_t> extras)
	{
		bool changed = false;
		for (const auto id : extras)
		{
			if (!std::ranges::contains(ids, id))
			{
				ids.push_back(id);
				changed = true;
			}
		}
		return changed;
	}

	std::vector<std::size_t> UnionExternalIds(std::span<const std::size_t> lhs, std::span<const std::size_t> rhs)
	{
		std::vector<std::size_t> result(lhs.begin(), lhs.end());
		AppendUniqueExternalIds(result, rhs);
		return result;
	}

	template <class Remap>
	NodeVariant RemapNodeInputs(const NodeVariant& node, Remap&& remap)
	{
		return std::visit(
		    [&](const auto& typedNode) -> NodeVariant {
			    using T = std::decay_t<decltype(typedNode)>;
			    auto copy = typedNode;
			    if constexpr (std::same_as<T, ParamRefNode> || std::same_as<T, ConstantNode> ||
			                  std::same_as<T, QuantizedConstantNode> || std::same_as<T, VariableRefNode> ||
			                  std::same_as<T, LoadActivationNode> || std::same_as<T, TapeLoadActivationNode>)
			    {
				    return copy;
			    }
			    else if constexpr (std::same_as<T, UnaryOpNode>)
			    {
				    copy.input = remap(copy.input);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, BinaryOpNode>)
			    {
				    copy.lhs = remap(copy.lhs);
				    copy.rhs = remap(copy.rhs);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, QuantizedMatMulNode>)
			    {
				    copy.lhs = remap(copy.lhs);
				    copy.rhsStorage = remap(copy.rhsStorage);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, CallNode>)
			    {
				    for (auto& arg : copy.args)
				    {
					    arg = remap(arg);
				    }
				    return copy;
			    }
			    else if constexpr (std::same_as<T, CastNode> || std::same_as<T, QuantizeNode> ||
			                       std::same_as<T, DequantizeNode> || std::same_as<T, SaveActivationNode> ||
			                       std::same_as<T, TapeSaveActivationNode> || std::same_as<T, ReduceOpNode> ||
			                       std::same_as<T, ReshapeNode> || std::same_as<T, PermuteNode> ||
			                       std::same_as<T, BroadcastToNode> || std::same_as<T, PadNode> ||
			                       std::same_as<T, ScanNode> || std::same_as<T, SoftmaxNode> ||
			                       std::same_as<T, Im2ColNode> || std::same_as<T, Pool2DNode> ||
			                       std::same_as<T, UpsampleNode> || std::same_as<T, SliceNode> ||
			                       std::same_as<T, ArgsortNode>)
			    {
				    copy.input = remap(copy.input);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, TimestepEmbeddingNode>)
			    {
				    copy.timesteps = remap(copy.timesteps);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, CondNode>)
			    {
				    copy.condition = remap(copy.condition);
				    for (auto& arg : copy.args)
				    {
					    arg = remap(arg);
				    }
				    return copy;
			    }
			    else if constexpr (std::same_as<T, WhileNode>)
			    {
				    for (auto& arg : copy.initArgs)
				    {
					    arg = remap(arg);
				    }
				    return copy;
			    }
			    else if constexpr (std::same_as<T, GatherNode> || std::same_as<T, GetRowsNode>)
			    {
				    copy.data = remap(copy.data);
				    copy.indices = remap(copy.indices);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, QuantizedGetRowsNode>)
			    {
				    copy.storage = remap(copy.storage);
				    copy.indices = remap(copy.indices);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, ScatterNode>)
			    {
				    copy.data = remap(copy.data);
				    copy.indices = remap(copy.indices);
				    copy.updates = remap(copy.updates);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, SSMScanNode>)
			    {
				    copy.state = remap(copy.state);
				    copy.dt = remap(copy.dt);
				    copy.a = remap(copy.a);
				    copy.b = remap(copy.b);
				    copy.c = remap(copy.c);
				    if (copy.d)
				    {
					    copy.d = remap(*copy.d);
				    }
				    return copy;
			    }
			    else if constexpr (std::same_as<T, RWKVWKVNode>)
			    {
				    copy.key = remap(copy.key);
				    copy.value = remap(copy.value);
				    copy.receptance = remap(copy.receptance);
				    copy.timeDecay = remap(copy.timeDecay);
				    copy.timeFirst = remap(copy.timeFirst);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, CrossEntropyLossNode>)
			    {
				    copy.logits = remap(copy.logits);
				    copy.labels = remap(copy.labels);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, CrossEntropyLossBackwardNode>)
			    {
				    copy.grad = remap(copy.grad);
				    copy.logits = remap(copy.logits);
				    copy.labels = remap(copy.labels);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, NormalizationNode>)
			    {
				    copy.input = remap(copy.input);
				    if (copy.scale)
				    {
					    copy.scale = remap(*copy.scale);
				    }
				    if (copy.bias)
				    {
					    copy.bias = remap(*copy.bias);
				    }
				    return copy;
			    }
			    else if constexpr (std::same_as<T, RoPENode>)
			    {
				    copy.input = remap(copy.input);
				    if (copy.positions)
				    {
					    copy.positions = remap(*copy.positions);
				    }
				    return copy;
			    }
			    else if constexpr (std::same_as<T, BatchMatMulNode> || std::same_as<T, OutProdNode>)
			    {
				    copy.lhs = remap(copy.lhs);
				    copy.rhs = remap(copy.rhs);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, SolveTriNode>)
			    {
				    copy.a = remap(copy.a);
				    copy.b = remap(copy.b);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, SGDStepNode>)
			    {
				    copy.parameter = remap(copy.parameter);
				    copy.gradient = remap(copy.gradient);
				    if (copy.velocity)
				    {
					    copy.velocity = remap(*copy.velocity);
				    }
				    return copy;
			    }
			    else if constexpr (std::same_as<T, AdamWStepNode>)
			    {
				    copy.parameter = remap(copy.parameter);
				    copy.gradient = remap(copy.gradient);
				    copy.firstMoment = remap(copy.firstMoment);
				    copy.secondMoment = remap(copy.secondMoment);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, Conv2DNode> || std::same_as<T, ConvTranspose2DNode>)
			    {
				    copy.input = remap(copy.input);
				    copy.weight = remap(copy.weight);
				    if (copy.bias)
				    {
					    copy.bias = remap(*copy.bias);
				    }
				    return copy;
			    }
			    else if constexpr (std::same_as<T, ConcatNode>)
			    {
				    for (auto& input : copy.inputs)
				    {
					    input = remap(input);
				    }
				    return copy;
			    }
			    else if constexpr (std::same_as<T, MulMatIdNode>)
			    {
				    copy.as = remap(copy.as);
				    copy.b = remap(copy.b);
				    copy.ids = remap(copy.ids);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, FusedOpNode>)
			    {
				    for (auto& arg : copy.args)
				    {
					    arg = remap(arg);
				    }
				    return copy;
			    }
			    else
			    {
				    static_assert(sizeof(T) == 0, "Unhandled LiteNN node input remap case");
			    }
		    },
		    node);
	}

	std::optional<CPUMLIRExternalizedGraph> BuildCPUMLIRExternalizedGraph(const Graph& graph,
	                                                                      const CompilerOptions& options)
	{
		if (graph.Backward().has_value() || graph.ActivationSlotCount() != 0 || graph.TapeSlotCount() != 0 ||
		    graph.SubgraphCount() == 0)
		{
			return std::nullopt;
		}

		CPUMLIRExternalizedGraph result;
		std::uint64_t projectedWeightBytes = 0;
		for (std::size_t variableIndex = 0; variableIndex < graph.VariableCount(); ++variableIndex)
		{
			const auto& data = graph.GetVariable(variableIndex)->Data();
			projectedWeightBytes = AlignUpU64(projectedWeightBytes, 64);
			projectedWeightBytes += static_cast<std::uint64_t>(data.NumElements()) * ElementByteSize(data.DType());
		}
		if (projectedWeightBytes > std::numeric_limits<std::size_t>::max())
		{
			throw std::runtime_error("CPU MLIR external weight region exceeds the host address space");
		}
		result.weights.reserve(static_cast<std::size_t>(projectedWeightBytes));
		std::unordered_map<std::size_t, std::size_t> variableExternalIdMap;
		std::vector<std::vector<std::optional<std::size_t>>> directExternalByNode(graph.SubgraphCount());
		std::vector<std::vector<std::size_t>> externalDepsBySubgraph(graph.SubgraphCount());

		for (SubgraphId subgraphId = 0; subgraphId < graph.SubgraphCount(); ++subgraphId)
		{
			const auto& subgraph = graph.GetSubgraph(subgraphId);
			directExternalByNode[subgraphId].resize(subgraph.NodeCount());
			for (NodeId nodeId = 0; nodeId < subgraph.NodeCount(); ++nodeId)
			{
				const auto& entry = subgraph.GetNodeEntry(nodeId);
				if (const auto* variable = std::get_if<VariableRefNode>(&entry.node))
				{
					if (entry.outputInfos.size() != 1)
					{
						return std::nullopt;
					}
					if (variable->variableIndex >= graph.VariableCount())
					{
						return std::nullopt;
					}
					const auto output = entry.outputInfos[0];
					auto [it, inserted] = variableExternalIdMap.emplace(variable->variableIndex, 0);
					if (inserted)
					{
						constexpr std::uint64_t kAlignment = 64;
						const auto offset =
						    AppendTensorPayloadBytes(result.weights, graph.GetVariable(variable->variableIndex)->Data(),
						                             output.dtype, output.shape, kAlignment);
						if (!offset)
						{
							return std::nullopt;
						}
						const auto name = graph.VariableName(variable->variableIndex);
						it->second = result.externalTensorInfos.size();
						result.externalTensorInfos.push_back(MakeExternalTensorInfo(
						    name, kWeightsRegionName, output.dtype, result.weights, output.shape, *offset,
						    TensorByteSizeForShape(output.dtype, output.shape), kAlignment));
					}
					directExternalByNode[subgraphId][nodeId] = it->second;
					AppendUniqueExternalId(externalDepsBySubgraph[subgraphId], it->second);
					continue;
				}

				if (const auto* constant = std::get_if<ConstantNode>(&entry.node))
				{
					if (entry.outputInfos.size() != 1)
					{
						return std::nullopt;
					}
					const auto output = entry.outputInfos[0];
					if (!CanExternalizeCPUTensorInMLIR(output.dtype) || constant->value.DType() != output.dtype ||
					    constant->value.Shape() != output.shape)
					{
						continue;
					}
					const auto byteSize = TensorByteSizeForShape(output.dtype, output.shape);
					if (byteSize < options.cpuAOTExternalConstantMinBytes)
					{
						continue;
					}

					constexpr std::uint64_t kAlignment = 64;
					const auto offset = AppendTensorPayloadBytes(result.constants, constant->value, output.dtype,
					                                             output.shape, kAlignment);
					if (!offset)
					{
						continue;
					}
					const auto externalId = result.externalTensorInfos.size();
					const auto name = std::format("constant_{}_{}", subgraphId, nodeId);
					result.externalTensorInfos.push_back(
					    MakeExternalTensorInfo(name, kConstantsRegionName, output.dtype, result.constants, output.shape,
					                           *offset, byteSize, kAlignment));
					directExternalByNode[subgraphId][nodeId] = externalId;
					AppendUniqueExternalId(externalDepsBySubgraph[subgraphId], externalId);
				}
			}
		}

		if (result.externalTensorInfos.empty())
		{
			return std::nullopt;
		}

		bool changed = true;
		while (changed)
		{
			changed = false;
			for (SubgraphId subgraphId = 0; subgraphId < graph.SubgraphCount(); ++subgraphId)
			{
				const auto& subgraph = graph.GetSubgraph(subgraphId);
				for (const auto& entry : subgraph.Nodes())
				{
					if (const auto* call = std::get_if<CallNode>(&entry.node))
					{
						if (call->callee >= graph.SubgraphCount())
						{
							return std::nullopt;
						}
						changed |= AppendUniqueExternalIds(externalDepsBySubgraph[subgraphId],
						                                   externalDepsBySubgraph[call->callee]);
					}
					else if (const auto* cond = std::get_if<CondNode>(&entry.node))
					{
						if (cond->thenBranch >= graph.SubgraphCount() || cond->elseBranch >= graph.SubgraphCount())
						{
							return std::nullopt;
						}
						const auto branchDeps = UnionExternalIds(externalDepsBySubgraph[cond->thenBranch],
						                                         externalDepsBySubgraph[cond->elseBranch]);
						if (externalDepsBySubgraph[cond->thenBranch] != branchDeps)
						{
							externalDepsBySubgraph[cond->thenBranch] = branchDeps;
							changed = true;
						}
						if (externalDepsBySubgraph[cond->elseBranch] != branchDeps)
						{
							externalDepsBySubgraph[cond->elseBranch] = branchDeps;
							changed = true;
						}
						changed |= AppendUniqueExternalIds(externalDepsBySubgraph[subgraphId], branchDeps);
					}
				}
			}
		}

		if (externalDepsBySubgraph[graph.Forward()].empty())
		{
			return std::nullopt;
		}

		for (SubgraphId subgraphId = 0; subgraphId < graph.SubgraphCount(); ++subgraphId)
		{
			const auto& subgraph = graph.GetSubgraph(subgraphId);
			for (const auto& entry : subgraph.Nodes())
			{
				if (const auto* whileNode = std::get_if<WhileNode>(&entry.node))
				{
					if (whileNode->condBranch >= graph.SubgraphCount() ||
					    whileNode->bodyBranch >= graph.SubgraphCount())
					{
						return std::nullopt;
					}
					if (!externalDepsBySubgraph[whileNode->condBranch].empty() ||
					    !externalDepsBySubgraph[whileNode->bodyBranch].empty())
					{
						return std::nullopt;
					}
				}
				else if (const auto* fused = std::get_if<FusedOpNode>(&entry.node))
				{
					if (fused->body >= graph.SubgraphCount())
					{
						return std::nullopt;
					}
					if (!externalDepsBySubgraph[fused->body].empty())
					{
						return std::nullopt;
					}
				}
			}
		}

		const auto hiddenOutputFor = [&](const std::vector<NodeId>& hiddenNodesByExternalId, std::size_t externalId) {
			if (externalId >= hiddenNodesByExternalId.size() ||
			    hiddenNodesByExternalId[externalId] == std::numeric_limits<NodeId>::max())
			{
				throw std::runtime_error("CPU MLIR externalization planned a missing hidden parameter");
			}
			return NodeOutput{ hiddenNodesByExternalId[externalId], 0 };
		};

		for (SubgraphId subgraphId = 0; subgraphId < graph.SubgraphCount(); ++subgraphId)
		{
			const auto& original = graph.GetSubgraph(subgraphId);
			Subgraph rebuilt;
			std::vector<NodeId> publicParamNodes;
			publicParamNodes.reserve(original.Params().size());
			for (const auto& param : original.Params())
			{
				publicParamNodes.push_back(
				    rebuilt.AddParam(param.dtype, std::vector<std::size_t>(param.shape.begin(), param.shape.end())));
			}

			std::vector<NodeId> hiddenNodesByExternalId(result.externalTensorInfos.size(),
			                                            std::numeric_limits<NodeId>::max());
			for (const auto externalId : externalDepsBySubgraph[subgraphId])
			{
				const auto& external = result.externalTensorInfos[externalId];
				hiddenNodesByExternalId[externalId] =
				    rebuilt.AddParam(external.type.dtype, external.type.StaticShape());
			}

			std::vector<std::vector<NodeOutput>> outputMap(original.NodeCount());
			const auto remapOutput = [&](NodeOutput output) {
				if (output.node >= outputMap.size() || output.port >= outputMap[output.node].size())
				{
					throw std::runtime_error("CPU MLIR externalization encountered a non-topological node reference");
				}
				return outputMap[output.node][output.port];
			};

			for (NodeId nodeId = 0; nodeId < original.NodeCount(); ++nodeId)
			{
				const auto& entry = original.GetNodeEntry(nodeId);
				if (const auto* param = std::get_if<ParamRefNode>(&entry.node))
				{
					if (param->paramIndex >= publicParamNodes.size())
					{
						return std::nullopt;
					}
					outputMap[nodeId] = { NodeOutput{ publicParamNodes[param->paramIndex], 0 } };
					continue;
				}

				if (const auto externalId = directExternalByNode[subgraphId][nodeId])
				{
					outputMap[nodeId] = { hiddenOutputFor(hiddenNodesByExternalId, *externalId) };
					continue;
				}

				auto remappedNode = RemapNodeInputs(entry.node, remapOutput);
				if (auto* call = std::get_if<CallNode>(&remappedNode))
				{
					for (const auto externalId : externalDepsBySubgraph[call->callee])
					{
						call->args.push_back(hiddenOutputFor(hiddenNodesByExternalId, externalId));
					}
				}
				else if (auto* cond = std::get_if<CondNode>(&remappedNode))
				{
					if (externalDepsBySubgraph[cond->thenBranch] != externalDepsBySubgraph[cond->elseBranch])
					{
						return std::nullopt;
					}
					for (const auto externalId : externalDepsBySubgraph[cond->thenBranch])
					{
						cond->args.push_back(hiddenOutputFor(hiddenNodesByExternalId, externalId));
					}
				}

				auto outputInfos = std::vector<OutputInfo>(entry.outputInfos.begin(), entry.outputInfos.end());
				const auto newNode = rebuilt.AddNode(std::move(remappedNode), std::move(outputInfos));
				outputMap[nodeId].reserve(entry.outputInfos.size());
				for (std::size_t port = 0; port < entry.outputInfos.size(); ++port)
				{
					outputMap[nodeId].push_back({ newNode, port });
				}
			}

			std::vector<NodeOutput> remappedResults;
			remappedResults.reserve(original.Results().size());
			for (const auto resultOutput : original.Results())
			{
				remappedResults.push_back(remapOutput(resultOutput));
			}
			rebuilt.SetResults(std::move(remappedResults));
			result.graph.AddSubgraph(std::move(rebuilt));
		}

		result.graph.SetForward(graph.Forward());
		const auto& forward = graph.GetSubgraph(graph.Forward());
		std::vector<std::string> inputNames;
		inputNames.reserve(forward.Params().size() + externalDepsBySubgraph[graph.Forward()].size());
		for (std::size_t i = 0; i < forward.Params().size(); ++i)
		{
			inputNames.push_back(graph.InputName(i));
		}
		for (const auto externalId : externalDepsBySubgraph[graph.Forward()])
		{
			const auto& external = result.externalTensorInfos[externalId];
			inputNames.push_back(std::format("__litenn_external_{}_{}", inputNames.size(), external.name));
			result.entryExternalTensorInfos.push_back(external);
		}
		result.graph.SetInputNames(std::move(inputNames));
		std::vector<std::string> outputNames;
		outputNames.reserve(forward.Results().size());
		for (std::size_t i = 0; i < forward.Results().size(); ++i)
		{
			outputNames.push_back(graph.OutputName(i));
		}
		result.graph.SetOutputNames(std::move(outputNames));
		result.graph.SetMetadata(std::vector<ModelMetadataEntry>(graph.Metadata().begin(), graph.Metadata().end()));
		return result;
	}

	std::optional<CompiledArtifactParts> TryCompileCPUParallelLinearChainF32(const Graph& graph,
	                                                                         const CompilerOptions& options)
	{
		const auto threadCount = ResolveCPUAOTThreadCount(options);
		if (threadCount <= 1)
		{
			return std::nullopt;
		}

		if (graph.Backward().has_value() || graph.ActivationSlotCount() != 0 || graph.TapeSlotCount() != 0 ||
		    graph.SubgraphCount() == 0)
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Results().size() != 1)
		{
			return std::nullopt;
		}
		const auto finalResult = subgraph.Results()[0];
		if (finalResult.port != 0 || finalResult.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		struct ValueRef
		{
			llvm::Value* ptr{};
			DataType dtype{ DataType::Float32 };
			std::vector<std::size_t> shape;
		};

		llvm::LLVMContext ctx;
		auto module = std::make_unique<llvm::Module>("litenn_cpu_parallel_linear_chain", ctx);
		auto* voidTy = llvm::Type::getVoidTy(ctx);
		auto* ptrTy = llvm::PointerType::get(ctx, 0);
		auto* i64Ty = llvm::Type::getInt64Ty(ctx);
		auto* i1Ty = llvm::Type::getInt1Ty(ctx);
		auto* entryType = llvm::FunctionType::get(voidTy, { ptrTy, ptrTy }, false);
		auto* entry =
		    llvm::Function::Create(entryType, llvm::GlobalValue::ExternalLinkage, std::string(kEntrySymbol), *module);
		auto* block = llvm::BasicBlock::Create(ctx, "entry", entry);
		llvm::IRBuilder<> builder(block);

		auto argIt = entry->arg_begin();
		llvm::Value* inputArray = &*argIt++;
		llvm::Value* outputArray = &*argIt;
		auto mallocFn = module->getOrInsertFunction("malloc", llvm::FunctionType::get(ptrTy, { i64Ty }, false));
		auto freeFn = module->getOrInsertFunction("free", llvm::FunctionType::get(voidTy, { ptrTy }, false));
		auto kernelFn = module->getOrInsertFunction(
		    "litenn_cpu_matmul_bias_relu_parallel_f32",
		    llvm::FunctionType::get(
		        voidTy, { ptrTy, ptrTy, ptrTy, ptrTy, i64Ty, i64Ty, i64Ty, i64Ty, i64Ty, i64Ty, i1Ty }, false));

		const bool useExternalRegions = IsCPUExternalRegionsEnabled(options);
		std::vector<std::byte> externalConstants;
		std::vector<std::byte> externalWeights;
		std::vector<CompiledModuleExternalTensorInfo> externalTensorInfos;
		std::vector<std::optional<ValueRef>> values(subgraph.NodeCount());
		std::vector<llvm::Value*> heapAllocations;
		std::size_t fusedLayerCount = 0;
		bool hasParallelEligibleLayer = false;
		const bool forceSidecarShapeGate = options.cpuAOTParallelMinFlops <= 1;
		std::uint64_t totalFlops = 0;

		const auto loadArrayPointer = [&](llvm::Value* array, std::size_t index) {
			auto* slot = builder.CreateGEP(ptrTy, array, builder.getInt64(index));
			return builder.CreateLoad(ptrTy, slot);
		};
		const auto requireValue = [&](NodeOutput output) -> std::optional<ValueRef> {
			if (output.port != 0 || output.node >= values.size() || !values[output.node])
			{
				return std::nullopt;
			}
			return *values[output.node];
		};
		const auto tensorBytes = [&](const OutputInfo& info) { return TensorByteSizeForShape(info.dtype, info.shape); };

		for (NodeId nodeId = 0; nodeId < subgraph.NodeCount(); ++nodeId)
		{
			const auto& entryNode = subgraph.GetNodeEntry(nodeId);
			if (entryNode.outputInfos.size() != 1)
			{
				return std::nullopt;
			}
			const auto& output = entryNode.outputInfos[0];
			if (const auto* param = std::get_if<ParamRefNode>(&entryNode.node))
			{
				values[nodeId] = ValueRef{
					.ptr = loadArrayPointer(inputArray, param->paramIndex),
					.dtype = output.dtype,
					.shape = output.shape,
				};
				continue;
			}
			if (const auto* variable = std::get_if<VariableRefNode>(&entryNode.node))
			{
				if (variable->variableIndex >= graph.VariableCount())
				{
					return std::nullopt;
				}
				auto constantData = CopyF32TensorData(graph.GetVariable(variable->variableIndex)->Data(), output.shape);
				if (!constantData)
				{
					return std::nullopt;
				}
				llvm::Value* ptr = nullptr;
				if (useExternalRegions)
				{
					constexpr std::uint64_t kAlignment = 64;
					const auto offset = AppendExternalF32Region(externalWeights, *constantData);
					ptr = AddExternalRegionPointer(*module, builder, "litenn_cpu_external_weights", offset);
					externalTensorInfos.push_back(MakeExternalF32TensorInfo(
					    graph.VariableName(variable->variableIndex), kWeightsRegionName, externalWeights, output.shape,
					    offset, static_cast<std::uint64_t>(constantData->size() * sizeof(float)), kAlignment));
				}
				else
				{
					ptr = AddF32ConstantGlobal(*module, builder, std::format("litenn_cpu_const_{}", nodeId),
					                           *constantData);
				}
				values[nodeId] = ValueRef{
					.ptr = ptr,
					.dtype = output.dtype,
					.shape = output.shape,
				};
				continue;
			}
			if (const auto* constant = std::get_if<ConstantNode>(&entryNode.node))
			{
				auto constantData = CopyF32TensorData(constant->value, output.shape);
				if (!constantData)
				{
					return std::nullopt;
				}
				llvm::Value* ptr = nullptr;
				if (useExternalRegions)
				{
					constexpr std::uint64_t kAlignment = 64;
					const auto offset = AppendExternalF32Region(externalConstants, *constantData);
					ptr = AddExternalRegionPointer(*module, builder, "litenn_cpu_external_constants", offset);
					externalTensorInfos.push_back(MakeExternalF32TensorInfo(
					    std::format("constant_{}", nodeId), kConstantsRegionName, externalConstants, output.shape,
					    offset, static_cast<std::uint64_t>(constantData->size() * sizeof(float)), kAlignment));
				}
				else
				{
					ptr = AddF32ConstantGlobal(*module, builder, std::format("litenn_cpu_const_{}", nodeId),
					                           *constantData);
				}
				values[nodeId] = ValueRef{
					.ptr = ptr,
					.dtype = output.dtype,
					.shape = output.shape,
				};
				continue;
			}

			const auto* fused = std::get_if<FusedOpNode>(&entryNode.node);
			if (!fused ||
			    (fused->pattern != FusionPattern::MatMulBiasAdd &&
			     fused->pattern != FusionPattern::MatMulBiasAddReLU) ||
			    fused->args.size() < 3)
			{
				return std::nullopt;
			}
			auto lhs = requireValue(fused->args[0]);
			auto rhs = requireValue(fused->args[1]);
			auto bias = requireValue(fused->args[2]);
			if (!lhs || !rhs || !bias || lhs->dtype != DataType::Float32 || rhs->dtype != DataType::Float32 ||
			    bias->dtype != DataType::Float32 || output.dtype != DataType::Float32 || lhs->shape.size() != 2 ||
			    rhs->shape.size() != 2 || output.shape.size() != 2 || bias->shape.size() != output.shape.size() ||
			    lhs->shape[1] != rhs->shape[0] || output.shape[0] != lhs->shape[0] ||
			    output.shape[1] != rhs->shape[1] ||
			    !IsSameRankBroadcastCompatibleShape(output.shape, bias->shape, output.shape))
			{
				return std::nullopt;
			}

			llvm::Value* outPtr = nullptr;
			if (nodeId == finalResult.node)
			{
				outPtr = loadArrayPointer(outputArray, 0);
			}
			else
			{
				outPtr = builder.CreateCall(mallocFn, { builder.getInt64(tensorBytes(output)) });
				heapAllocations.push_back(outPtr);
			}

			const auto m = static_cast<std::uint64_t>(output.shape[0]);
			const auto k = static_cast<std::uint64_t>(lhs->shape[1]);
			const auto n = static_cast<std::uint64_t>(output.shape[1]);
			const auto layerFlops = SaturatedMulU64(SaturatedMulU64(SaturatedMulU64(m, k), n), 2);
			if (!forceSidecarShapeGate && m > 256)
			{
				return std::nullopt;
			}
			const auto layerThreadCount = (forceSidecarShapeGate || ShouldUseCPUSidecarLinearLayer(m, k, n, layerFlops))
			                                  ? static_cast<std::uint64_t>(threadCount)
			                                  : 1;
			hasParallelEligibleLayer |= layerThreadCount > 1;
			totalFlops = SaturatedAddU64(totalFlops, layerFlops);
			builder.CreateCall(kernelFn,
			                   { lhs->ptr, rhs->ptr, bias->ptr, outPtr, builder.getInt64(m), builder.getInt64(k),
			                     builder.getInt64(n), builder.getInt64(static_cast<std::uint64_t>(bias->shape[0])),
			                     builder.getInt64(layerThreadCount),
			                     builder.getInt64(static_cast<std::uint64_t>(options.cpuAOTAffinityPolicy)),
			                     builder.getInt1(fused->pattern == FusionPattern::MatMulBiasAddReLU) });
			values[nodeId] = ValueRef{ .ptr = outPtr, .dtype = output.dtype, .shape = output.shape };
			++fusedLayerCount;
		}

		if (fusedLayerCount == 0 || !hasParallelEligibleLayer || !values[finalResult.node] ||
		    totalFlops < options.cpuAOTParallelMinFlops)
		{
			return std::nullopt;
		}
		for (auto it = heapAllocations.rbegin(); it != heapAllocations.rend(); ++it)
		{
			builder.CreateCall(freeFn, { *it });
		}
		builder.CreateRetVoid();

		const auto inputSpecs = BuildInputSpecs(graph);
		const auto outputSpecs = BuildOutputSpecs(graph);
		auto config = CreateNativeTargetMachine();
		ConfigureForNativeObject(*module, config);
		OptimizeLLVMModule(*module, *config.targetMachine, options.cpuAOTLLVMOptLevel);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, config.triple, CompiledModuleBackend::CPUNative);
		auto instructions = EmitObjectFile(*module);
		return CompiledArtifactParts{ std::move(rodata),
			                          std::move(instructions),
			                          std::move(externalConstants),
			                          std::move(externalWeights),
			                          std::move(externalTensorInfos),
			                          inputSpecs,
			                          outputSpecs };
	}

	std::optional<CompiledArtifactParts>
	TryCompileCPUParallelLinearChainF32WithExternalRegionFusion(const Graph& graph, const CompilerOptions& options)
	{
		if (auto parts = TryCompileCPUParallelLinearChainF32(graph, options))
		{
			return parts;
		}
		if (!IsCPUExternalRegionsEnabled(options) || !options.enableCPUAOTExternalRegionFusion)
		{
			return std::nullopt;
		}

		auto optimized = graph;
		FusionPass{}.Run(optimized);
		return TryCompileCPUParallelLinearChainF32(optimized, options);
	}

	std::uint64_t NumElements(const CompiledTensorSpec& spec)
	{
		std::uint64_t n = 1;
		for (const auto dim : spec.type.StaticShape())
		{
			n *= static_cast<std::uint64_t>(dim);
		}
		return n;
	}

	std::vector<std::uint64_t> ContiguousStrides(const CompiledTensorSpec& spec)
	{
		const auto shape = spec.type.StaticShape();
		std::vector<std::uint64_t> strides(shape.size());
		if (!strides.empty())
		{
			strides.back() = 1;
			for (std::size_t i = strides.size() - 1; i > 0; --i)
			{
				strides[i - 1] = strides[i] * static_cast<std::uint64_t>(shape[i]);
			}
		}
		return strides;
	}

	llvm::Type* GetElementType(llvm::LLVMContext& ctx, DataType dtype)
	{
		switch (dtype)
		{
		case DataType::Float32:
			return llvm::Type::getFloatTy(ctx);
		case DataType::Float64:
			return llvm::Type::getDoubleTy(ctx);
		case DataType::Float16:
			return llvm::Type::getHalfTy(ctx);
		case DataType::BFloat16:
			return llvm::Type::getBFloatTy(ctx);
		case DataType::Float8E4M3:
		case DataType::Float8E5M2:
			return llvm::Type::getInt8Ty(ctx);
		case DataType::Int32:
			return llvm::Type::getInt32Ty(ctx);
		case DataType::Int64:
			return llvm::Type::getInt64Ty(ctx);
		case DataType::Int8:
		case DataType::UInt8:
			return llvm::Type::getInt8Ty(ctx);
		case DataType::Bool:
			return llvm::Type::getInt1Ty(ctx);
		}
		throw std::runtime_error("Invalid data type");
	}

	std::string LLVMTypeToString(llvm::Type* type)
	{
		std::string text;
		llvm::raw_string_ostream os(text);
		type->print(os);
		return os.str();
	}

	llvm::StructType* GetMemRefDescriptorType(llvm::LLVMContext& ctx, std::size_t rank)
	{
		auto* ptrTy = llvm::PointerType::get(ctx, 0);
		auto* i64Ty = llvm::Type::getInt64Ty(ctx);
		auto* shapeArrayTy = llvm::ArrayType::get(i64Ty, rank);
		return llvm::StructType::get(ctx, { ptrTy, ptrTy, i64Ty, shapeArrayTy, shapeArrayTy });
	}

	bool IsMemRefDescriptorType(llvm::LLVMContext& ctx, llvm::Type* type, std::size_t rank)
	{
		auto* descTy = llvm::dyn_cast<llvm::StructType>(type);
		if (!descTy)
		{
			return false;
		}

		auto* ptrTy = llvm::PointerType::get(ctx, 0);
		auto* i64Ty = llvm::Type::getInt64Ty(ctx);
		auto* shapeArrayTy = llvm::ArrayType::get(i64Ty, rank);
		if (descTy->getNumElements() == 5)
		{
			return descTy->getElementType(0) == ptrTy && descTy->getElementType(1) == ptrTy &&
			       descTy->getElementType(2) == i64Ty && descTy->getElementType(3) == shapeArrayTy &&
			       descTy->getElementType(4) == shapeArrayTy;
		}
		if (descTy->getNumElements() == 4)
		{
			return descTy->getElementType(0) == ptrTy && descTy->getElementType(1) == i64Ty &&
			       descTy->getElementType(2) == shapeArrayTy && descTy->getElementType(3) == shapeArrayTy;
		}
		return false;
	}

	llvm::Value* BuildI64Array(llvm::IRBuilder<>& builder, std::span<const std::uint64_t> values)
	{
		auto& ctx = builder.getContext();
		auto* i64Ty = llvm::Type::getInt64Ty(ctx);
		auto* arrayTy = llvm::ArrayType::get(i64Ty, values.size());
		llvm::Value* array = llvm::PoisonValue::get(arrayTy);
		for (std::size_t i = 0; i < values.size(); ++i)
		{
			array = builder.CreateInsertValue(array, builder.getInt64(values[i]), { static_cast<unsigned>(i) });
		}
		return array;
	}

	llvm::Value* BuildMemRefDescriptor(llvm::IRBuilder<>& builder, llvm::Value* data, const CompiledTensorSpec& spec)
	{
		auto& ctx = builder.getContext();
		const auto shape = spec.type.StaticShape();
		auto* descTy = GetMemRefDescriptorType(ctx, shape.size());
		llvm::Value* desc = llvm::PoisonValue::get(descTy);
		desc = builder.CreateInsertValue(desc, data, { 0 });
		desc = builder.CreateInsertValue(desc, data, { 1 });
		desc = builder.CreateInsertValue(desc, builder.getInt64(0), { 2 });

		std::vector<std::uint64_t> sizes(shape.begin(), shape.end());
		desc = builder.CreateInsertValue(desc, BuildI64Array(builder, sizes), { 3 });

		const auto strides = ContiguousStrides(spec);
		desc = builder.CreateInsertValue(desc, BuildI64Array(builder, strides), { 4 });
		return desc;
	}

	void AppendDescriptorCallArgument(llvm::IRBuilder<>& builder, llvm::FunctionType* calleeType,
	                                  std::size_t& paramIndex, llvm::Value* descriptor, std::vector<llvm::Value*>& args)
	{
		if (paramIndex >= calleeType->getNumParams())
		{
			throw std::runtime_error("Compiled subgraph function has fewer arguments than expected");
		}

		auto* expectedTy = calleeType->getParamType(paramIndex);
		if (expectedTy == descriptor->getType())
		{
			args.push_back(descriptor);
			++paramIndex;
			return;
		}

		auto* descTy = llvm::cast<llvm::StructType>(descriptor->getType());
		const auto appendValue = [&](llvm::Value* value) -> bool {
			if (paramIndex >= calleeType->getNumParams() || calleeType->getParamType(paramIndex) != value->getType())
			{
				return false;
			}
			args.push_back(value);
			++paramIndex;
			return true;
		};

		const auto tryAppendPattern = [&](const auto& appendPattern) {
			const auto savedParamIndex = paramIndex;
			const auto savedArgCount = args.size();
			if (appendPattern())
			{
				return true;
			}
			paramIndex = savedParamIndex;
			args.resize(savedArgCount);
			return false;
		};

		const auto appendWholeField = [&](unsigned index) -> bool {
			return appendValue(builder.CreateExtractValue(descriptor, { index }));
		};

		const auto* sizesTy = llvm::cast<llvm::ArrayType>(descTy->getElementType(3));
		const auto* stridesTy = llvm::cast<llvm::ArrayType>(descTy->getElementType(4));
		const auto appendArrayScalars = [&](unsigned index, const llvm::ArrayType* arrayTy) -> bool {
			for (unsigned i = 0; i < arrayTy->getNumElements(); ++i)
			{
				if (!appendValue(builder.CreateExtractValue(descriptor, { index, i })))
				{
					return false;
				}
			}
			return true;
		};

		if (tryAppendPattern([&] {
			    return appendWholeField(0) && appendWholeField(1) && appendWholeField(2) && appendWholeField(3) &&
			           appendWholeField(4);
		    }) ||
		    tryAppendPattern([&] {
			    return appendWholeField(1) && appendWholeField(2) && appendWholeField(3) && appendWholeField(4);
		    }) ||
		    tryAppendPattern([&] {
			    return appendWholeField(1) && appendWholeField(2) && appendArrayScalars(3, sizesTy) &&
			           appendArrayScalars(4, stridesTy);
		    }) ||
		    tryAppendPattern([&] {
			    return appendWholeField(0) && appendWholeField(1) && appendWholeField(2) &&
			           appendArrayScalars(3, sizesTy) && appendArrayScalars(4, stridesTy);
		    }))
		{
			return;
		}

		if (expectedTy->isPointerTy())
		{
			auto* alloca = builder.CreateAlloca(descriptor->getType());
			builder.CreateStore(descriptor, alloca);
			args.push_back(alloca);
			++paramIndex;
			return;
		}

		std::string message =
		    "Compiled subgraph function has an unsupported memref ABI at parameter " + std::to_string(paramIndex);
		if (paramIndex < calleeType->getNumParams())
		{
			message += ": expected " + LLVMTypeToString(calleeType->getParamType(paramIndex));
		}
		message += ", got " + LLVMTypeToString(descriptor->getType());
		throw std::runtime_error(message);
	}

	void CopyDescriptorToOutput(llvm::IRBuilder<>& builder, llvm::Value* descriptor, llvm::Value* outputArray,
	                            std::size_t outputIndex, const CompiledTensorSpec& spec)
	{
		auto& ctx = builder.getContext();
		auto* ptrTy = llvm::PointerType::get(ctx, 0);
		auto* outputSlot = builder.CreateGEP(ptrTy, outputArray, builder.getInt64(outputIndex));
		auto* outputData = builder.CreateLoad(ptrTy, outputSlot);
		auto* descTy = llvm::cast<llvm::StructType>(descriptor->getType());
		const unsigned dataField = descTy->getNumElements() == 5 ? 1 : 0;
		auto* sourceData = builder.CreateExtractValue(descriptor, { dataField });
		const auto byteCount = NumElements(spec) * LiteNN::ElementByteSize(spec.type.dtype);
		builder.CreateMemCpy(outputData, llvm::Align(1), sourceData, llvm::Align(1), builder.getInt64(byteCount));
	}

	std::optional<std::size_t> FindStructRetParamIndex(const llvm::Function& function)
	{
		std::size_t index = 0;
		for (const auto& arg : function.args())
		{
			if (arg.hasStructRetAttr())
			{
				return index;
			}
			++index;
		}
		return std::nullopt;
	}

	CompiledTensorSpec ExternalTensorAsSpec(const CompiledModuleExternalTensorInfo& info)
	{
		return {
			.type = info.type,
			.name = info.name,
		};
	}

	void AddUniformEntryWrapper(llvm::Module& module, std::string_view calleeName,
	                            std::span<const CompiledTensorSpec> inputs, std::span<const CompiledTensorSpec> outputs,
	                            std::span<const CompiledModuleExternalTensorInfo> externalInputs = {})
	{
		auto* callee = module.getFunction(calleeName);
		if (!callee)
		{
			throw std::runtime_error("Compiled subgraph function was not found in LLVM module");
		}

		auto& ctx = module.getContext();
		auto* voidTy = llvm::Type::getVoidTy(ctx);
		auto* ptrTy = llvm::PointerType::get(ctx, 0);
		auto* entryType = llvm::FunctionType::get(voidTy, { ptrTy, ptrTy }, false);
		auto* entry =
		    llvm::Function::Create(entryType, llvm::GlobalValue::ExternalLinkage, std::string(kEntrySymbol), module);

		auto* block = llvm::BasicBlock::Create(ctx, "entry", entry);
		llvm::IRBuilder<> builder(block);
		auto argIt = entry->arg_begin();
		llvm::Value* inputArray = &*argIt++;
		llvm::Value* outputArray = &*argIt;

		std::vector<llvm::Value*> descriptors;
		descriptors.reserve(inputs.size() + externalInputs.size());
		for (std::size_t i = 0; i < inputs.size(); ++i)
		{
			auto* inputSlot = builder.CreateGEP(ptrTy, inputArray, builder.getInt64(i));
			auto* inputData = builder.CreateLoad(ptrTy, inputSlot);
			descriptors.push_back(BuildMemRefDescriptor(builder, inputData, inputs[i]));
		}
		for (const auto& external : externalInputs)
		{
			const auto symbol =
			    external.region == kWeightsRegionName ? "litenn_cpu_external_weights" : "litenn_cpu_external_constants";
			auto* data = AddExternalRegionPointer(module, builder, symbol, external.byteOffset);
			descriptors.push_back(BuildMemRefDescriptor(builder, data, ExternalTensorAsSpec(external)));
		}

		auto* calleeType = callee->getFunctionType();
		std::vector<llvm::Value*> callArgs;
		std::size_t paramIndex = 0;
		const auto sretParamIndex = FindStructRetParamIndex(*callee);
		llvm::AllocaInst* sretStorage = nullptr;
		llvm::Type* sretType = nullptr;

		if (sretParamIndex)
		{
			sretType = callee->getParamStructRetType(static_cast<unsigned>(*sretParamIndex));
			if (!sretType)
			{
				throw std::runtime_error("Compiled subgraph function has an invalid sret ABI");
			}
			sretStorage = builder.CreateAlloca(sretType);
		}

		const auto appendStructRetIfNeeded = [&]() {
			if (sretStorage && paramIndex == *sretParamIndex)
			{
				callArgs.push_back(sretStorage);
				++paramIndex;
				return true;
			}
			return false;
		};

		appendStructRetIfNeeded();
		for (auto* descriptor : descriptors)
		{
			appendStructRetIfNeeded();
			AppendDescriptorCallArgument(builder, calleeType, paramIndex, descriptor, callArgs);
		}

		std::vector<llvm::Value*> outputDescriptors;
		outputDescriptors.reserve(outputs.size());
		for (std::size_t i = 0; i < outputs.size(); ++i)
		{
			auto* outputSlot = builder.CreateGEP(ptrTy, outputArray, builder.getInt64(i));
			auto* outputData = builder.CreateLoad(ptrTy, outputSlot);
			outputDescriptors.push_back(BuildMemRefDescriptor(builder, outputData, outputs[i]));
		}
		for (auto* descriptor : outputDescriptors)
		{
			appendStructRetIfNeeded();
			if (paramIndex >= calleeType->getNumParams())
			{
				break;
			}
			AppendDescriptorCallArgument(builder, calleeType, paramIndex, descriptor, callArgs);
		}
		appendStructRetIfNeeded();

		if (paramIndex != calleeType->getNumParams())
		{
			throw std::runtime_error("Compiled subgraph function has more arguments than expected");
		}

		auto* call = builder.CreateCall(callee, callArgs);
		call->setCallingConv(callee->getCallingConv());
		call->setAttributes(callee->getAttributes());
		auto* retTy = calleeType->getReturnType();

		if (sretStorage)
		{
			if (outputs.size() == 1 && IsMemRefDescriptorType(ctx, sretType, outputs[0].type.Rank()))
			{
				auto* descriptor = builder.CreateLoad(sretType, sretStorage);
				CopyDescriptorToOutput(builder, descriptor, outputArray, 0, outputs[0]);
				builder.CreateRetVoid();
				return;
			}

			auto* resultTupleTy = llvm::dyn_cast<llvm::StructType>(sretType);
			if (!resultTupleTy || resultTupleTy->getNumElements() != outputs.size())
			{
				throw std::runtime_error("Compiled subgraph function has an unsupported sret ABI");
			}

			auto* result = builder.CreateLoad(sretType, sretStorage);
			for (std::size_t i = 0; i < outputs.size(); ++i)
			{
				auto* descriptor = builder.CreateExtractValue(result, { static_cast<unsigned>(i) });
				CopyDescriptorToOutput(builder, descriptor, outputArray, i, outputs[i]);
			}
			builder.CreateRetVoid();
			return;
		}

		if (retTy->isVoidTy())
		{
			builder.CreateRetVoid();
			return;
		}

		if (outputs.size() == 1)
		{
			auto* expectedDescTy = GetMemRefDescriptorType(ctx, outputs[0].type.Rank());
			if (retTy == expectedDescTy)
			{
				CopyDescriptorToOutput(builder, call, outputArray, 0, outputs[0]);
				builder.CreateRetVoid();
				return;
			}
		}

		auto* resultTupleTy = llvm::dyn_cast<llvm::StructType>(retTy);
		if (!resultTupleTy || resultTupleTy->getNumElements() != outputs.size())
		{
			throw std::runtime_error("Compiled subgraph function has an unsupported result ABI");
		}

		for (std::size_t i = 0; i < outputs.size(); ++i)
		{
			auto* descriptor = builder.CreateExtractValue(call, { static_cast<unsigned>(i) });
			CopyDescriptorToOutput(builder, descriptor, outputArray, i, outputs[i]);
		}
		builder.CreateRetVoid();
	}

	std::vector<std::byte> ToByteVector(const void* data, std::size_t size)
	{
		if (size != 0 && data == nullptr)
		{
			throw std::runtime_error("Compiled module image has a null data pointer");
		}
		std::vector<std::byte> bytes(size);
		if (size != 0)
		{
			std::memcpy(bytes.data(), data, size);
		}
		return bytes;
	}

	std::vector<std::byte> ToByteVector(CompiledModuleRegion region, std::string_view name)
	{
		const auto bytes = RegionBytes(region, name);
		return std::vector<std::byte>(bytes.begin(), bytes.end());
	}

	std::vector<std::byte> RestoreLegacyInstructionsFromSeparated(CompiledModuleBackend backend,
	                                                              std::span<const std::byte> instructions,
	                                                              std::span<const std::byte> constants)
	{
		std::vector<std::byte> restored(instructions.begin(), instructions.end());
		if (backend != CompiledModuleBackend::CUDANative)
		{
			return restored;
		}

		auto payload = DeserializeCUDANativeInstructionPayload(restored);
		if (!constants.empty())
		{
			payload.constantData.assign(constants.begin(), constants.end());
			restored = SerializeCUDANativeInstructionPayload(payload);
		}
		return restored;
	}

	std::size_t ReadExportedSymbolSize(const void* symbol, std::string_view label)
	{
		if (symbol == nullptr)
		{
			throw std::runtime_error(std::format("Compiled module exported symbol '{}' is null", label));
		}

		std::uint64_t rawSize = 0;
		std::memcpy(&rawSize, symbol, sizeof(rawSize));
		if (rawSize > std::numeric_limits<std::size_t>::max())
		{
			throw std::runtime_error(
			    std::format("Compiled module exported symbol '{}' does not fit in size_t on this host", label));
		}
		return static_cast<std::size_t>(rawSize);
	}

	struct LoadedJIT
	{
		std::unique_ptr<llvm::LLVMContext> context;
		std::unique_ptr<llvm::ExecutionEngine> engine;
		EntryFn entry{};
	};

	struct MLIRUnrankedMemRefDescriptor
	{
		std::int64_t rank{};
		void* descriptor{};
	};

	struct MLIRMemRefView
	{
		std::byte* aligned{};
		std::int64_t offset{};
		std::vector<std::int64_t> sizes;
		std::vector<std::int64_t> strides;
	};

	template <typename T>
	T ReadMLIRDescriptorField(const std::byte* base, std::size_t offset)
	{
		T value{};
		std::memcpy(&value, base + offset, sizeof(T));
		return value;
	}

	MLIRMemRefView ReadMLIRMemRefView(const MLIRUnrankedMemRefDescriptor& descriptor)
	{
		MLIRMemRefView view;
		if (descriptor.descriptor == nullptr || descriptor.rank < 0)
		{
			return view;
		}

		const auto rank = static_cast<std::size_t>(descriptor.rank);
		const auto* raw = static_cast<const std::byte*>(descriptor.descriptor);
		constexpr std::size_t pointerSize = sizeof(void*);
		constexpr std::size_t indexSize = sizeof(std::int64_t);
		view.aligned = static_cast<std::byte*>(ReadMLIRDescriptorField<void*>(raw, pointerSize));
		view.offset = ReadMLIRDescriptorField<std::int64_t>(raw, pointerSize * 2);
		const auto sizesOffset = pointerSize * 2 + indexSize;
		const auto stridesOffset = sizesOffset + rank * indexSize;
		view.sizes.resize(rank);
		view.strides.resize(rank);
		for (std::size_t i = 0; i < rank; ++i)
		{
			view.sizes[i] = ReadMLIRDescriptorField<std::int64_t>(raw, sizesOffset + i * indexSize);
			view.strides[i] = ReadMLIRDescriptorField<std::int64_t>(raw, stridesOffset + i * indexSize);
		}
		return view;
	}

	void CopyMLIRMemRefRecursive(const MLIRMemRefView& source, const MLIRMemRefView& target,
	                             std::size_t elementSizeBytes, std::size_t dimension, std::int64_t sourceElementOffset,
	                             std::int64_t targetElementOffset)
	{
		if (dimension == source.sizes.size())
		{
			std::memcpy(target.aligned + targetElementOffset * static_cast<std::int64_t>(elementSizeBytes),
			            source.aligned + sourceElementOffset * static_cast<std::int64_t>(elementSizeBytes),
			            elementSizeBytes);
			return;
		}

		for (std::int64_t i = 0; i < source.sizes[dimension]; ++i)
		{
			CopyMLIRMemRefRecursive(source, target, elementSizeBytes, dimension + 1,
			                        sourceElementOffset + i * source.strides[dimension],
			                        targetElementOffset + i * target.strides[dimension]);
		}
	}

	extern "C" void LiteNNRuntimeMemRefCopy(std::int64_t elementSizeBytes, void* rawSource, void* rawTarget)
	{
		if (elementSizeBytes <= 0 || rawSource == nullptr || rawTarget == nullptr)
		{
			return;
		}

		const auto* sourceDescriptor = static_cast<const MLIRUnrankedMemRefDescriptor*>(rawSource);
		const auto* targetDescriptor = static_cast<const MLIRUnrankedMemRefDescriptor*>(rawTarget);
		auto source = ReadMLIRMemRefView(*sourceDescriptor);
		auto target = ReadMLIRMemRefView(*targetDescriptor);
		if (source.aligned == nullptr || target.aligned == nullptr || source.sizes.size() != target.sizes.size())
		{
			return;
		}
		for (std::size_t i = 0; i < source.sizes.size(); ++i)
		{
			if (source.sizes[i] < 0 || target.sizes[i] != source.sizes[i])
			{
				return;
			}
		}

		CopyMLIRMemRefRecursive(source, target, static_cast<std::size_t>(elementSizeBytes), 0, source.offset,
		                        target.offset);
	}

	void RegisterJITRuntimeSymbol(std::string_view name, void* address)
	{
		const auto symbolName = std::string(name);
		llvm::sys::DynamicLibrary::AddSymbol(symbolName, address);
		llvm::sys::DynamicLibrary::AddSymbol("_" + symbolName, address);
	}

	float LiteNNRuntimeExpF(float value)
	{
		return std::exp(value);
	}

	float LiteNNRuntimeLogF(float value)
	{
		return std::log(value);
	}

	float LiteNNRuntimeSqrtF(float value)
	{
		return std::sqrt(value);
	}

	float LiteNNRuntimeSinF(float value)
	{
		return std::sin(value);
	}

	float LiteNNRuntimeCosF(float value)
	{
		return std::cos(value);
	}

	float LiteNNRuntimeTanF(float value)
	{
		return std::tan(value);
	}

	float LiteNNRuntimeAsinF(float value)
	{
		return std::asin(value);
	}

	float LiteNNRuntimeAcosF(float value)
	{
		return std::acos(value);
	}

	float LiteNNRuntimeAtanF(float value)
	{
		return std::atan(value);
	}

	float LiteNNRuntimeErfF(float value)
	{
		return std::erf(value);
	}

	void LiteNNRuntimeSinCosF(float value, float* sinOut, float* cosOut)
	{
		*sinOut = std::sin(value);
		*cosOut = std::cos(value);
	}

	LoadedJIT LoadJIT(std::span<const std::byte> instructions)
	{
		InitializeNativeLLVM();
		RegisterJITRuntimeSymbol("malloc", reinterpret_cast<void*>(static_cast<void* (*) (std::size_t)>(&std::malloc)));
		RegisterJITRuntimeSymbol("free", reinterpret_cast<void*>(static_cast<void (*)(void*)>(&std::free)));
		RegisterJITRuntimeSymbol(
		    "memcpy", reinterpret_cast<void*>(static_cast<void* (*) (void*, const void*, std::size_t)>(&std::memcpy)));
		RegisterJITRuntimeSymbol(
		    "memset", reinterpret_cast<void*>(static_cast<void* (*) (void*, int, std::size_t)>(&std::memset)));
		RegisterJITRuntimeSymbol("memrefCopy", reinterpret_cast<void*>(&LiteNNRuntimeMemRefCopy));
		RegisterJITRuntimeSymbol("expf", reinterpret_cast<void*>(&LiteNNRuntimeExpF));
		RegisterJITRuntimeSymbol("logf", reinterpret_cast<void*>(&LiteNNRuntimeLogF));
		RegisterJITRuntimeSymbol("sqrtf", reinterpret_cast<void*>(&LiteNNRuntimeSqrtF));
		RegisterJITRuntimeSymbol("sinf", reinterpret_cast<void*>(&LiteNNRuntimeSinF));
		RegisterJITRuntimeSymbol("cosf", reinterpret_cast<void*>(&LiteNNRuntimeCosF));
		RegisterJITRuntimeSymbol("tanf", reinterpret_cast<void*>(&LiteNNRuntimeTanF));
		RegisterJITRuntimeSymbol("asinf", reinterpret_cast<void*>(&LiteNNRuntimeAsinF));
		RegisterJITRuntimeSymbol("acosf", reinterpret_cast<void*>(&LiteNNRuntimeAcosF));
		RegisterJITRuntimeSymbol("atanf", reinterpret_cast<void*>(&LiteNNRuntimeAtanF));
		RegisterJITRuntimeSymbol("erff", reinterpret_cast<void*>(&LiteNNRuntimeErfF));
		RegisterJITRuntimeSymbol("sincosf", reinterpret_cast<void*>(&LiteNNRuntimeSinCosF));
		RegisterJITRuntimeSymbol("litenn_cpu_matmul_bias_relu_parallel_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_matmul_bias_relu_parallel_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_matmul_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_matmul_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_external_constants",
		                         reinterpret_cast<void*>(&litenn_cpu_external_constants));
		RegisterJITRuntimeSymbol("litenn_cpu_external_weights", reinterpret_cast<void*>(&litenn_cpu_external_weights));

		LoadedJIT loaded;
		loaded.context = std::make_unique<llvm::LLVMContext>();
		auto module = std::make_unique<llvm::Module>("litenn_jit_loader", *loaded.context);

		std::string error;
		llvm::EngineBuilder builder(std::move(module));
		builder.setErrorStr(&error);
		builder.setEngineKind(llvm::EngineKind::JIT);
		loaded.engine.reset(builder.create());
		if (!loaded.engine)
		{
			throw std::runtime_error("Failed to create LiteNN JIT: " + error);
		}

		auto buffer = llvm::MemoryBuffer::getMemBufferCopy(
		    llvm::StringRef(reinterpret_cast<const char*>(instructions.data()), instructions.size()),
		    "litenn-compiled-module.o");
		auto object = TakeExpected(llvm::object::ObjectFile::createObjectFile(buffer->getMemBufferRef()),
		                           "Failed to parse LiteNN object image");
		// NOTE: Under Linux/WSL sanitizers, LLVM MCJIT/RuntimeDyld currently reports
		// one fixed 80-byte leak per addObjectFile/finalizeObject cycle on this path.
		// LiteNN copies and owns rodata/instruction bytes above this boundary; local
		// experiments disabling EH-frame registration and the GDB JIT listener did not
		// eliminate the leak, so treat that specific LSan report as external loader
		// behavior rather than a LiteNN-owned buffer lifetime bug.
		loaded.engine->addObjectFile(
		    llvm::object::OwningBinary<llvm::object::ObjectFile>(std::move(object), std::move(buffer)));
		loaded.engine->finalizeObject();

		const auto mangledEntrySymbol = "_" + std::string(kEntrySymbol);
		auto address = loaded.engine->getFunctionAddress(mangledEntrySymbol);
		if (address == 0)
		{
			address = loaded.engine->getFunctionAddress(std::string(kEntrySymbol));
		}
		if (address == 0)
		{
			throw std::runtime_error("Failed to lookup LiteNN entry symbol");
		}
		loaded.entry = reinterpret_cast<EntryFn>(address);
		return loaded;
	}

	llvm::Constant* ByteArrayConstant(llvm::LLVMContext& ctx, std::span<const std::byte> bytes)
	{
		return llvm::ConstantDataArray::getString(
		    ctx, llvm::StringRef(reinterpret_cast<const char*>(bytes.data()), bytes.size()),
		    /*AddNull=*/false);
	}

	void AddByteArraySymbol(llvm::Module& module, std::string_view name, std::span<const std::byte> bytes,
	                        std::string_view section = {})
	{
		auto& ctx = module.getContext();
		auto* init = ByteArrayConstant(ctx, bytes);
		auto* global = new llvm::GlobalVariable(module, init->getType(), true, llvm::GlobalValue::ExternalLinkage, init,
		                                        std::string(name));
		global->setAlignment(llvm::Align(1));
		if (!section.empty())
		{
			global->setSection(std::string(section));
		}
	}

	void AddSizeSymbol(llvm::Module& module, std::string_view name, std::size_t size, std::string_view section = {})
	{
		auto& ctx = module.getContext();
		auto* i64Ty = llvm::Type::getInt64Ty(ctx);
		auto* init = llvm::ConstantInt::get(i64Ty, static_cast<std::uint64_t>(size));
		auto* global =
		    new llvm::GlobalVariable(module, i64Ty, true, llvm::GlobalValue::ExternalLinkage, init, std::string(name));
		if (!section.empty())
		{
			global->setSection(std::string(section));
		}
	}

	std::vector<std::byte> EmitCarrierObject(std::span<const std::byte> rodata, std::span<const std::byte> instructions,
	                                         std::string_view symbolPrefix)
	{
		llvm::LLVMContext ctx;
		llvm::Module module("litenn_compiled_module_carrier", ctx);

		const auto prefix = std::string(symbolPrefix);
		AddByteArraySymbol(module, prefix + "_rodata", rodata);
		AddSizeSymbol(module, prefix + "_rodata_size", rodata.size());
		AddByteArraySymbol(module, prefix + "_instructions", instructions);
		AddSizeSymbol(module, prefix + "_instructions_size", instructions.size());

		return EmitObjectFile(module);
	}

	std::vector<std::byte> EmitSingleRegionCarrierObject(std::span<const std::byte> region,
	                                                     std::string_view symbolPrefix, std::string_view regionName,
	                                                     std::string_view sectionName)
	{
		llvm::LLVMContext ctx;
		llvm::Module module("litenn_compiled_module_region_carrier", ctx);

		const auto prefix = std::string(symbolPrefix);
		const auto name = std::string(regionName);
		AddByteArraySymbol(module, prefix + "_" + name, region, sectionName);
		AddSizeSymbol(module, prefix + "_" + name + "_size", region.size(), sectionName);
		return EmitObjectFile(module);
	}

	std::vector<std::byte> EmitSeparatedCarrierObject(std::span<const std::byte> metadata,
	                                                  std::span<const std::byte> constants,
	                                                  std::span<const std::byte> weights,
	                                                  std::span<const std::byte> instructions,
	                                                  std::string_view symbolPrefix)
	{
		llvm::LLVMContext ctx;
		llvm::Module module("litenn_compiled_module_separated_carrier", ctx);

		const auto prefix = std::string(symbolPrefix);
		AddByteArraySymbol(module, prefix + "_metadata", metadata, ".litenn_metadata");
		AddSizeSymbol(module, prefix + "_metadata_size", metadata.size(), ".litenn_metadata");
		AddByteArraySymbol(module, prefix + "_constants", constants, ".litenn_constants");
		AddSizeSymbol(module, prefix + "_constants_size", constants.size(), ".litenn_constants");
		AddByteArraySymbol(module, prefix + "_weights", weights, ".litenn_weights");
		AddSizeSymbol(module, prefix + "_weights_size", weights.size(), ".litenn_weights");
		AddByteArraySymbol(module, prefix + "_instructions", instructions, ".litenn_instructions");
		AddSizeSymbol(module, prefix + "_instructions_size", instructions.size(), ".litenn_instructions");
		return EmitObjectFile(module);
	}

	void WriteAllBytes(const std::filesystem::path& path, std::span<const std::byte> bytes)
	{
		std::ofstream out(path, std::ios::binary);
		if (!out)
		{
			throw std::runtime_error("Failed to open output object file");
		}
		constexpr std::size_t kChunkBytes = 64ull * 1024ull * 1024ull;
		std::size_t offset = 0;
		while (offset < bytes.size())
		{
			const auto remaining = bytes.size() - offset;
			const auto chunk = std::min(remaining, kChunkBytes);
			out.write(reinterpret_cast<const char*>(bytes.data() + static_cast<std::ptrdiff_t>(offset)),
			          static_cast<std::streamsize>(chunk));
			if (!out)
			{
				throw std::runtime_error("Failed to write output object file");
			}
			offset += chunk;
		}
	}

	std::optional<std::size_t> FindSpecIndex(std::span<const CompiledTensorSpec> specs, std::string_view name)
	{
		for (std::size_t i = 0; i < specs.size(); ++i)
		{
			if (specs[i].name == name)
			{
				return i;
			}
		}
		return std::nullopt;
	}

	template <Device D>
	void ValidateTensorAgainstSpec(const Tensor<D>& tensor, const CompiledTensorSpec& spec, std::size_t inputIndex)
	{
		const auto shape = spec.type.StaticShape();
		if (tensor.DType() != spec.type.dtype || tensor.Shape() != ShapeView{ shape })
		{
			const auto label =
			    spec.name.empty() ? std::to_string(inputIndex) : std::format("{} ('{}')", inputIndex, spec.name);
			throw std::runtime_error(std::format("CompiledModule input {} mismatch: expected {}, got {}", label,
			                                     Validation::FormatInfo(spec.type.dtype, shape),
			                                     Validation::FormatInfo(tensor.DType(), tensor.Shape().Dims)));
		}
	}

	template <Device D>
	void ValidateOutputTensorAgainstSpec(const Tensor<D>& tensor, const CompiledTensorSpec& spec,
	                                     std::size_t outputIndex)
	{
		const auto shape = spec.type.StaticShape();
		if (tensor.DType() != spec.type.dtype || tensor.Shape() != ShapeView{ shape })
		{
			const auto label =
			    spec.name.empty() ? std::to_string(outputIndex) : std::format("{} ('{}')", outputIndex, spec.name);
			throw std::runtime_error(std::format("CompiledModule output {} mismatch: expected {}, got {}", label,
			                                     Validation::FormatInfo(spec.type.dtype, shape),
			                                     Validation::FormatInfo(tensor.DType(), tensor.Shape().Dims)));
		}
	}

	void ValidateBindingAgainstSpec(const CompiledTensorBinding& binding, const CompiledTensorSpec& spec,
	                                std::size_t index, std::string_view role)
	{
		if (binding.data == nullptr)
		{
			const auto label = spec.name.empty() ? std::to_string(index) : std::format("{} ('{}')", index, spec.name);
			throw std::runtime_error(std::format("CompiledModule {} {} has null data", role, label));
		}
		const auto shape = spec.type.StaticShape();
		if (binding.type.dtype != spec.type.dtype || ShapeView{ binding.type.StaticShape() } != ShapeView{ shape })
		{
			const auto label = spec.name.empty() ? std::to_string(index) : std::format("{} ('{}')", index, spec.name);
			throw std::runtime_error(
			    std::format("CompiledModule {} {} mismatch: expected {}, got {}", role, label,
			                Validation::FormatInfo(spec.type.dtype, shape),
			                Validation::FormatInfo(binding.type.dtype, binding.type.StaticShape())));
		}
		if (!binding.name.empty() && !spec.name.empty() && binding.name != spec.name)
		{
			const auto label = std::format("{} ('{}')", index, spec.name);
			throw std::runtime_error(
			    std::format("CompiledModule {} {} name mismatch: got '{}'", role, label, binding.name));
		}
	}

	template <Device D>
	CompiledTensorBinding MakeBindingFromTensor(Tensor<D>& tensor, std::string name = {},
	                                            std::optional<QuantizationParams> quantization = std::nullopt)
	{
		return { .data = tensor.UnsafeRawData(),
			     .type = MakeTensorType(tensor.DType(), tensor.Shape().Dims),
			     .name = std::move(name),
			     .quantization = std::move(quantization) };
	}

	template <Device D>
	CompiledTensorBinding MakeBindingFromTensor(const Tensor<D>& tensor, std::string name = {},
	                                            std::optional<QuantizationParams> quantization = std::nullopt)
	{
		return { .data = const_cast<void*>(tensor.UnsafeRawData()),
			     .type = MakeTensorType(tensor.DType(), tensor.Shape().Dims),
			     .name = std::move(name),
			     .quantization = std::move(quantization) };
	}

	std::size_t NormalizeThreadCount(std::size_t requested, std::size_t workCount)
	{
		if (workCount == 0)
		{
			return 0;
		}
		if (requested == 0)
		{
			requested = std::thread::hardware_concurrency();
			if (requested == 0)
			{
				requested = 1;
			}
		}
		return std::clamp<std::size_t>(requested, 1, workCount);
	}

#ifdef LITENN_ENABLE_CUDA
	struct CUDANativeTensorRef
	{
		CUDANativeArgumentKind kind{ CUDANativeArgumentKind::InputTensor };
		std::uint32_t index{};
		std::uint64_t byteOffset{};
		std::uint64_t byteSize{};
		DataType dtype{ DataType::Float32 };
		std::vector<std::size_t> shape;
	};

	struct CUDANativeBinaryPlan
	{
		BinaryOp op{ BinaryOp::Add };
		CUDANativeTensorRef lhs;
		CUDANativeTensorRef rhs;
		std::uint32_t elementCount{};
		bool requiresBroadcast{};
		std::vector<std::size_t> outputShape;
	};

	struct CUDANativeUnaryPlan
	{
		UnaryOp op{ UnaryOp::Negate };
		std::uint32_t inputIndex{};
		std::uint32_t elementCount{};
	};

	struct CUDANativeSiLUPlan
	{
		std::uint32_t inputIndex{};
		std::uint32_t elementCount{};
	};

	struct CUDANativeCastPlan
	{
		std::uint32_t inputIndex{};
		std::uint32_t elementCount{};
		DataType srcType{ DataType::Float32 };
		DataType dstType{ DataType::Float32 };
	};

	struct CUDANativeGetRowsPlan
	{
		CUDANativeTensorRef table;
		std::uint32_t indicesInputIndex{};
		DataType indexType{ DataType::Int32 };
		std::uint32_t rowSize{};
		std::uint32_t indexCount{};
		std::uint32_t outputElementCount{};
	};

	struct CUDANativeRMSNormPlan
	{
		std::uint32_t inputIndex{};
		std::optional<CUDANativeTensorRef> scale;
		std::uint32_t rowSize{};
		std::uint32_t elementCount{};
		float epsilon{};
	};

	struct CUDANativeRoPEPlan
	{
		std::uint32_t inputIndex{};
		std::optional<std::uint32_t> positionsInputIndex;
		std::optional<DataType> positionType;
		std::uint32_t featureSize{};
		std::uint32_t elementCount{};
		std::uint32_t positionOffset{};
		double base{};
		double frequencyScale{};
	};

	struct CUDANativeBatchMatMulPlan
	{
		std::uint32_t lhsInputIndex{};
		std::uint32_t rhsInputIndex{};
		std::uint32_t lhsElementCount{};
		std::uint32_t rhsElementCount{};
		std::uint32_t outputElementCount{};
		std::vector<std::size_t> lhsShape;
		std::vector<std::size_t> rhsShape;
		std::vector<std::size_t> outputShape;
	};

	struct CUDANativeScatterUpdatePlan
	{
		std::uint32_t dataInputIndex{};
		std::uint32_t indicesInputIndex{};
		std::uint32_t updatesInputIndex{};
		DataType indexType{ DataType::Int64 };
		std::uint32_t rowSize{};
		std::uint32_t dataElementCount{};
		std::uint32_t indexElementCount{};
		std::uint32_t updateElementCount{};
	};

	struct CUDANativeMatMulPlan
	{
		std::uint32_t lhsInputIndex{};
		std::uint32_t rhsInputIndex{};
		std::uint32_t m{};
		std::uint32_t k{};
		std::uint32_t n{};
		std::uint32_t lhsElementCount{};
		std::uint32_t rhsElementCount{};
		std::uint32_t outputElementCount{};
	};

	struct CUDANativeGGMLBlockMatMulPlan
	{
		QuantizedBlockFormat format{ QuantizedBlockFormat::GGML_Q8_0 };
		std::uint32_t lhsInputIndex{};
		CUDANativeTensorRef rhsStorage;
		std::uint32_t m{};
		std::uint32_t k{};
		std::uint32_t n{};
		std::uint32_t lhsElementCount{};
		std::uint32_t rhsStorageElementCount{};
		std::uint32_t outputElementCount{};
	};

	struct CUDANativeReducePlan
	{
		ReduceOp op{ ReduceOp::Sum };
		std::uint32_t inputIndex{};
		std::uint32_t inputElementCount{};
		std::uint32_t outputElementCount{};
		std::size_t axis{};
		std::vector<std::size_t> inputShape;
		std::vector<std::size_t> outputShape;
	};

	struct CUDANativeSoftmaxPlan
	{
		std::uint32_t inputIndex{};
		std::uint32_t elementCount{};
		std::size_t axis{};
		std::vector<std::size_t> inputShape;
	};

	struct CUDANativeConcatPlan
	{
		std::uint32_t outputElementCount{};
		std::vector<std::uint32_t> inputElementCounts;
		std::vector<std::uint32_t> inputIndices;
		std::vector<std::vector<std::size_t>> inputShapes;
		std::vector<std::size_t> outputShape;
		std::size_t axis{};
	};

	struct CUDANativeSlicePlan
	{
		std::uint32_t inputIndex{};
		std::uint32_t inputElementCount{};
		std::uint32_t outputElementCount{};
		std::size_t axis{};
		std::size_t start{};
		std::vector<std::size_t> inputShape;
		std::vector<std::size_t> outputShape;
	};

	struct CUDANativeMatMulBiasPlan
	{
		DataType dtype{ DataType::Float32 };
		std::uint32_t lhsInputIndex{};
		std::uint32_t rhsInputIndex{};
		std::uint32_t biasInputIndex{};
		std::uint32_t m{};
		std::uint32_t k{};
		std::uint32_t n{};
		std::uint32_t lhsElementCount{};
		std::uint32_t rhsElementCount{};
		std::uint32_t biasElementCount{};
		std::uint32_t outputElementCount{};
		std::vector<std::size_t> outputShape;
		std::vector<std::size_t> biasShape;
		bool relu{};
	};

	struct CUDANativeLinearChainPlan
	{
		std::vector<CUDANativeMatMulBiasEpilogueCodegenSpec> epilogues;
		CUDANativeInstructionPayload payload;
	};

	struct CUDANativeArtifactParts
	{
		std::vector<std::byte> rodata;
		std::vector<std::byte> instructions;
		std::vector<CompiledTensorSpec> inputSpecs;
		std::vector<CompiledTensorSpec> outputSpecs;
	};

	std::optional<std::uint32_t> GetParamIndex(const Subgraph& subgraph, NodeOutput output)
	{
		if (output.port != 0 || output.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto* param = std::get_if<ParamRefNode>(&subgraph.GetNodeEntry(output.node).node);
		if (!param || param->paramIndex >= subgraph.Params().size() ||
		    param->paramIndex > std::numeric_limits<std::uint32_t>::max())
		{
			return std::nullopt;
		}
		return static_cast<std::uint32_t>(param->paramIndex);
	}

	bool IsSameRankBroadcastCompatible(std::span<const std::size_t> lhs, std::span<const std::size_t> rhs,
	                                   std::span<const std::size_t> output)
	{
		if (lhs.size() != output.size() || rhs.size() != output.size())
		{
			return false;
		}

		for (std::size_t i = 0; i < output.size(); ++i)
		{
			if ((lhs[i] != output[i] && lhs[i] != 1) || (rhs[i] != output[i] && rhs[i] != 1) ||
			    output[i] != std::max(lhs[i], rhs[i]))
			{
				return false;
			}
		}
		return true;
	}

	bool IsCUDANativeSingleForwardGraph(const Graph& graph)
	{
		return graph.SubgraphCount() == 1 && !graph.Backward().has_value() && graph.VariableCount() == 0 &&
		       graph.ActivationSlotCount() == 0 && graph.TapeSlotCount() == 0;
	}

	std::optional<std::uint32_t> ShapeNumElementsU32(std::span<const std::size_t> shape)
	{
		std::uint64_t count = 1;
		for (const auto dim : shape)
		{
			if (dim == 0)
			{
				return std::nullopt;
			}
			count *= static_cast<std::uint64_t>(dim);
			if (count > std::numeric_limits<std::uint32_t>::max())
			{
				return std::nullopt;
			}
		}
		return static_cast<std::uint32_t>(count);
	}

	std::uint64_t AlignUp(std::uint64_t value, std::uint64_t alignment)
	{
		return ((value + alignment - 1) / alignment) * alignment;
	}

	std::uint64_t TensorByteSize(DataType dtype, std::span<const std::size_t> shape)
	{
		const auto elements = ShapeNumElementsU32(shape);
		if (!elements)
		{
			throw std::runtime_error("CUDA native tensor shape is too large");
		}
		return static_cast<std::uint64_t>(*elements) * LiteNN::ElementByteSize(dtype);
	}

	CUDANativeArgumentSpec ToCUDANativeArgument(const CUDANativeTensorRef& ref)
	{
		return {
			.kind = ref.kind,
			.index = ref.index,
			.byteOffset = ref.byteOffset,
			.byteSize = ref.byteSize,
		};
	}

	std::uint64_t AppendCUDANativeConstantTensor(CUDANativeInstructionPayload& payload,
	                                             const Tensor<PolymorphicDevice>& tensor)
	{
		const auto cpuTensor = tensor.CopyToDevice(CPU{});
		const auto byteSize = TensorByteSize(cpuTensor.DType(), cpuTensor.Shape().Dims);
		const auto offset = AlignUp(static_cast<std::uint64_t>(payload.constantData.size()), 16);
		if (payload.constantData.size() < offset)
		{
			payload.constantData.resize(static_cast<std::size_t>(offset));
		}
		const auto* begin = reinterpret_cast<const std::byte*>(cpuTensor.UnsafeRawData());
		payload.constantData.insert(payload.constantData.end(), begin, begin + byteSize);
		return offset;
	}

	std::optional<CUDANativeTensorRef> ResolveCUDANativeInputOrConstantTensorRef(const Graph& graph,
	                                                                             const Subgraph& subgraph,
	                                                                             NodeOutput output,
	                                                                             CUDANativeInstructionPayload& payload)
	{
		if (output.port != 0 || output.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& entry = subgraph.GetNodeEntry(output.node);
		if (entry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}
		const auto& info = entry.outputInfos[0];
		const auto byteSize = TensorByteSize(info.dtype, info.shape);
		if (const auto inputIndex = GetParamIndex(subgraph, output))
		{
			return CUDANativeTensorRef{ .kind = CUDANativeArgumentKind::InputTensor,
				                        .index = *inputIndex,
				                        .byteOffset = 0,
				                        .byteSize = byteSize,
				                        .dtype = info.dtype,
				                        .shape = info.shape };
		}
		if (const auto* constant = std::get_if<ConstantNode>(&entry.node))
		{
			if (constant->value.DType() != info.dtype || constant->value.Shape() != info.shape)
			{
				return std::nullopt;
			}
			return CUDANativeTensorRef{ .kind = CUDANativeArgumentKind::ConstantTensor,
				                        .index = 0,
				                        .byteOffset = AppendCUDANativeConstantTensor(payload, constant->value),
				                        .byteSize = byteSize,
				                        .dtype = info.dtype,
				                        .shape = info.shape };
		}
		if (const auto* variable = std::get_if<VariableRefNode>(&entry.node))
		{
			if (variable->variableIndex >= graph.VariableCount())
			{
				return std::nullopt;
			}
			const auto& graphVariable = graph.GetVariable(variable->variableIndex);
			const auto& tensor = graphVariable->Data();
			if (graphVariable->HasGradStorage() || tensor.DType() != info.dtype || tensor.Shape() != info.shape)
			{
				return std::nullopt;
			}
			return CUDANativeTensorRef{ .kind = CUDANativeArgumentKind::ConstantTensor,
				                        .index = 0,
				                        .byteOffset = AppendCUDANativeConstantTensor(payload, tensor),
				                        .byteSize = byteSize,
				                        .dtype = info.dtype,
				                        .shape = info.shape };
		}
		return std::nullopt;
	}

	std::uint64_t AllocateCUDANativeWorkspaceTensor(CUDANativeInstructionPayload& payload, std::uint64_t byteSize)
	{
		const auto offset = AlignUp(payload.workspaceBytes, 16);
		payload.workspaceBytes = offset + byteSize;
		return offset;
	}

	CUDANativeArgumentSpec AppendU32ScalarArgument(CUDANativeInstructionPayload& payload, std::uint32_t value)
	{
		const auto offset = payload.scalarData.size();
		AppendU32(payload.scalarData, value);
		return {
			.kind = CUDANativeArgumentKind::Scalar,
			.index = 0,
			.byteOffset = static_cast<std::uint64_t>(offset),
			.byteSize = sizeof(std::uint32_t),
		};
	}

	bool IsSupportedCUDANativeReduceF32Op(ReduceOp op)
	{
		switch (op)
		{
		case ReduceOp::Sum:
		case ReduceOp::Mean:
		case ReduceOp::Max:
		case ReduceOp::Min:
			return true;
		}
		return false;
	}

	bool IsSupportedCUDANativeBinaryF32Op(BinaryOp op)
	{
		switch (op)
		{
		case BinaryOp::Add:
		case BinaryOp::Subtract:
		case BinaryOp::Multiply:
		case BinaryOp::Divide:
		case BinaryOp::Max:
		case BinaryOp::Min:
			return true;
		default:
			return false;
		}
	}

	bool IsSupportedCUDANativeUnaryF32Op(UnaryOp op)
	{
		switch (op)
		{
		case UnaryOp::Negate:
		case UnaryOp::Abs:
		case UnaryOp::Sqrt:
		case UnaryOp::Exp:
		case UnaryOp::Log:
		case UnaryOp::Sin:
		case UnaryOp::Cos:
			return true;
		default:
			return false;
		}
	}

	std::optional<CUDANativeUnaryPlan> MatchCUDANativeUnaryF32(const Graph& graph)
	{
		if (!IsCUDANativeSingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 1 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 2)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* unary = std::get_if<UnaryOpNode>(&resultEntry.node);
		if (!unary || !IsSupportedCUDANativeUnaryF32Op(unary->op))
		{
			return std::nullopt;
		}

		const auto inputIndex = GetParamIndex(subgraph, unary->input);
		if (!inputIndex)
		{
			return std::nullopt;
		}

		const auto& param = subgraph.Params()[*inputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (param.dtype != DataType::Float32 || output.dtype != DataType::Float32 || param.shape != output.shape)
		{
			return std::nullopt;
		}

		const auto elementCount = ShapeView{ output.shape }.NumElements();
		if (elementCount == 0 || elementCount > std::numeric_limits<std::uint32_t>::max())
		{
			return std::nullopt;
		}

		return CUDANativeUnaryPlan{
			.op = unary->op,
			.inputIndex = *inputIndex,
			.elementCount = static_cast<std::uint32_t>(elementCount),
		};
	}

	bool IsCUDANativeScalarConstant(const Subgraph& subgraph, NodeOutput output, double expected)
	{
		if (output.port != 0 || output.node >= subgraph.NodeCount())
		{
			return false;
		}
		const auto& entry = subgraph.GetNodeEntry(output.node);
		const auto* constant = std::get_if<ConstantNode>(&entry.node);
		if (!constant || entry.outputInfos.size() != 1 || entry.outputInfos[0].shape != std::vector<std::size_t>{ 1 })
		{
			return false;
		}
		const auto cpuTensor = constant->value.CopyToDevice(CPU{});
		if (cpuTensor.NumElements() != 1)
		{
			return false;
		}
		double value = 0.0;
		CPU cpu;
		DeviceTraits<CPU>::ConvertTo(cpu, cpuTensor.DType(), cpuTensor.UnsafeRawData(), 1, DataType::Float64, &value);
		return value == expected;
	}

	std::optional<CUDANativeSiLUPlan> MatchCUDANativeSiLUF32(const Graph& graph)
	{
		if (!IsCUDANativeSingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 1 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 7)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		const auto* multiply = std::get_if<BinaryOpNode>(&resultEntry.node);
		if (!multiply || multiply->op != BinaryOp::Multiply || resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto tryMatchSigmoid = [&](NodeOutput maybeInput,
		                                 NodeOutput maybeSigmoid) -> std::optional<std::uint32_t> {
			const auto inputIndex = GetParamIndex(subgraph, maybeInput);
			if (!inputIndex || maybeSigmoid.port != 0 || maybeSigmoid.node >= subgraph.NodeCount())
			{
				return std::nullopt;
			}

			const auto& divideEntry = subgraph.GetNodeEntry(maybeSigmoid.node);
			const auto* divide = std::get_if<BinaryOpNode>(&divideEntry.node);
			if (!divide || divide->op != BinaryOp::Divide || divideEntry.outputInfos.size() != 1 ||
			    !IsCUDANativeScalarConstant(subgraph, divide->lhs, 1.0))
			{
				return std::nullopt;
			}

			const auto denom = divide->rhs;
			if (denom.port != 0 || denom.node >= subgraph.NodeCount())
			{
				return std::nullopt;
			}
			const auto& denomEntry = subgraph.GetNodeEntry(denom.node);
			const auto* add = std::get_if<BinaryOpNode>(&denomEntry.node);
			if (!add || add->op != BinaryOp::Add || denomEntry.outputInfos.size() != 1)
			{
				return std::nullopt;
			}

			NodeOutput expOutput{};
			if (IsCUDANativeScalarConstant(subgraph, add->lhs, 1.0))
			{
				expOutput = add->rhs;
			}
			else if (IsCUDANativeScalarConstant(subgraph, add->rhs, 1.0))
			{
				expOutput = add->lhs;
			}
			else
			{
				return std::nullopt;
			}

			if (expOutput.port != 0 || expOutput.node >= subgraph.NodeCount())
			{
				return std::nullopt;
			}
			const auto& expEntry = subgraph.GetNodeEntry(expOutput.node);
			const auto* exp = std::get_if<UnaryOpNode>(&expEntry.node);
			if (!exp || exp->op != UnaryOp::Exp || expEntry.outputInfos.size() != 1 || exp->input.port != 0 ||
			    exp->input.node >= subgraph.NodeCount())
			{
				return std::nullopt;
			}

			const auto& negateEntry = subgraph.GetNodeEntry(exp->input.node);
			const auto* negate = std::get_if<UnaryOpNode>(&negateEntry.node);
			if (!negate || negate->op != UnaryOp::Negate || negateEntry.outputInfos.size() != 1 ||
			    negate->input != maybeInput)
			{
				return std::nullopt;
			}
			return inputIndex;
		};

		auto inputIndex = tryMatchSigmoid(multiply->lhs, multiply->rhs);
		if (!inputIndex)
		{
			inputIndex = tryMatchSigmoid(multiply->rhs, multiply->lhs);
		}
		if (!inputIndex)
		{
			return std::nullopt;
		}

		const auto& input = subgraph.Params()[*inputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32 || input.shape != output.shape)
		{
			return std::nullopt;
		}
		const auto elementCount = ShapeNumElementsU32(output.shape);
		if (!elementCount)
		{
			return std::nullopt;
		}

		return CUDANativeSiLUPlan{ .inputIndex = *inputIndex, .elementCount = *elementCount };
	}

	std::optional<CUDANativeBinaryPlan> MatchCUDANativeBinaryF32(const Graph& graph,
	                                                             CUDANativeInstructionPayload& payload)
	{
		if (!IsCUDANativeSingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if ((subgraph.Params().size() < 1 || subgraph.Params().size() > 2) || subgraph.Results().size() != 1 ||
		    subgraph.NodeCount() != 3)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* binary = std::get_if<BinaryOpNode>(&resultEntry.node);
		if (!binary || !IsSupportedCUDANativeBinaryF32Op(binary->op))
		{
			return std::nullopt;
		}

		const auto lhs = ResolveCUDANativeInputOrConstantTensorRef(graph, subgraph, binary->lhs, payload);
		const auto rhs = ResolveCUDANativeInputOrConstantTensorRef(graph, subgraph, binary->rhs, payload);
		if (!lhs || !rhs)
		{
			return std::nullopt;
		}

		const auto& output = resultEntry.outputInfos[0];
		if (output.dtype != DataType::Float32)
		{
			return std::nullopt;
		}
		if (lhs->dtype != DataType::Float32 || rhs->dtype != DataType::Float32 ||
		    !IsSameRankBroadcastCompatible(lhs->shape, rhs->shape, output.shape))
		{
			return std::nullopt;
		}

		const auto elementCount = ShapeView{ output.shape }.NumElements();
		if (elementCount == 0 || elementCount > std::numeric_limits<std::uint32_t>::max())
		{
			return std::nullopt;
		}

		return CUDANativeBinaryPlan{
			.op = binary->op,
			.lhs = *lhs,
			.rhs = *rhs,
			.elementCount = static_cast<std::uint32_t>(elementCount),
			.requiresBroadcast = lhs->shape != output.shape || rhs->shape != output.shape,
			.outputShape = output.shape,
		};
	}

	std::optional<CUDANativeMatMulPlan> MatchCUDANativeMatMulF32(const Graph& graph)
	{
		if (!IsCUDANativeSingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 2 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 3)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* binary = std::get_if<BinaryOpNode>(&resultEntry.node);
		if (!binary || binary->op != BinaryOp::MatMul)
		{
			return std::nullopt;
		}

		const auto lhsInputIndex = GetParamIndex(subgraph, binary->lhs);
		const auto rhsInputIndex = GetParamIndex(subgraph, binary->rhs);
		if (!lhsInputIndex || !rhsInputIndex)
		{
			return std::nullopt;
		}

		const auto& lhsParam = subgraph.Params()[*lhsInputIndex];
		const auto& rhsParam = subgraph.Params()[*rhsInputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (lhsParam.dtype != DataType::Float32 || rhsParam.dtype != DataType::Float32 ||
		    output.dtype != DataType::Float32 || lhsParam.shape.size() != 2 || rhsParam.shape.size() != 2 ||
		    output.shape.size() != 2 || lhsParam.shape[1] != rhsParam.shape[0] ||
		    output.shape[0] != lhsParam.shape[0] || output.shape[1] != rhsParam.shape[1])
		{
			return std::nullopt;
		}

		const auto m = lhsParam.shape[0];
		const auto k = lhsParam.shape[1];
		const auto n = rhsParam.shape[1];
		const auto maxInt = static_cast<std::size_t>(std::numeric_limits<int>::max());
		if (m == 0 || k == 0 || n == 0 || m > maxInt || k > maxInt || n > maxInt)
		{
			return std::nullopt;
		}

		const auto lhsElementCount = ShapeView{ lhsParam.shape }.NumElements();
		const auto rhsElementCount = ShapeView{ rhsParam.shape }.NumElements();
		const auto outputElementCount = ShapeView{ output.shape }.NumElements();
		if (lhsElementCount > std::numeric_limits<std::uint32_t>::max() ||
		    rhsElementCount > std::numeric_limits<std::uint32_t>::max() ||
		    outputElementCount > std::numeric_limits<std::uint32_t>::max())
		{
			return std::nullopt;
		}

		return CUDANativeMatMulPlan{
			.lhsInputIndex = *lhsInputIndex,
			.rhsInputIndex = *rhsInputIndex,
			.m = static_cast<std::uint32_t>(m),
			.k = static_cast<std::uint32_t>(k),
			.n = static_cast<std::uint32_t>(n),
			.lhsElementCount = static_cast<std::uint32_t>(lhsElementCount),
			.rhsElementCount = static_cast<std::uint32_t>(rhsElementCount),
			.outputElementCount = static_cast<std::uint32_t>(outputElementCount),
		};
	}

	std::optional<CUDANativeBatchMatMulPlan> MatchCUDANativeBatchMatMulF32(const Graph& graph)
	{
		if (!IsCUDANativeSingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 2 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 3)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		const auto* batchMatMul = std::get_if<BatchMatMulNode>(&resultEntry.node);
		if (!batchMatMul || resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto lhsInputIndex = GetParamIndex(subgraph, batchMatMul->lhs);
		const auto rhsInputIndex = GetParamIndex(subgraph, batchMatMul->rhs);
		if (!lhsInputIndex || !rhsInputIndex)
		{
			return std::nullopt;
		}

		const auto& lhsParam = subgraph.Params()[*lhsInputIndex];
		const auto& rhsParam = subgraph.Params()[*rhsInputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (lhsParam.dtype != DataType::Float32 || rhsParam.dtype != DataType::Float32 ||
		    output.dtype != DataType::Float32 || lhsParam.shape.size() < 3 || rhsParam.shape.size() < 3 ||
		    output.shape.size() < 3)
		{
			return std::nullopt;
		}
		if (Detail::BatchMatMulOutputShape(lhsParam.shape, rhsParam.shape) != output.shape)
		{
			return std::nullopt;
		}

		const auto lhsElementCount = ShapeNumElementsU32(lhsParam.shape);
		const auto rhsElementCount = ShapeNumElementsU32(rhsParam.shape);
		const auto outputElementCount = ShapeNumElementsU32(output.shape);
		if (!lhsElementCount || !rhsElementCount || !outputElementCount)
		{
			return std::nullopt;
		}

		return CUDANativeBatchMatMulPlan{
			.lhsInputIndex = *lhsInputIndex,
			.rhsInputIndex = *rhsInputIndex,
			.lhsElementCount = *lhsElementCount,
			.rhsElementCount = *rhsElementCount,
			.outputElementCount = *outputElementCount,
			.lhsShape = lhsParam.shape,
			.rhsShape = rhsParam.shape,
			.outputShape = output.shape,
		};
	}

	std::optional<CUDANativeScatterUpdatePlan> MatchCUDANativeScatterUpdateF32(const Graph& graph)
	{
		if (!IsCUDANativeSingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 3 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 4)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		const auto* scatter = std::get_if<ScatterNode>(&resultEntry.node);
		if (!scatter || resultEntry.outputInfos.size() != 1 || scatter->axis != 0 ||
		    scatter->mode != ScatterMode::Update)
		{
			return std::nullopt;
		}

		const auto dataInputIndex = GetParamIndex(subgraph, scatter->data);
		const auto indicesInputIndex = GetParamIndex(subgraph, scatter->indices);
		const auto updatesInputIndex = GetParamIndex(subgraph, scatter->updates);
		if (!dataInputIndex || !indicesInputIndex || !updatesInputIndex)
		{
			return std::nullopt;
		}

		const auto& data = subgraph.Params()[*dataInputIndex];
		const auto& indices = subgraph.Params()[*indicesInputIndex];
		const auto& updates = subgraph.Params()[*updatesInputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (data.dtype != DataType::Float32 || updates.dtype != DataType::Float32 ||
		    output.dtype != DataType::Float32 || data.shape.empty() || data.shape != output.shape ||
		    (indices.dtype != DataType::Int32 && indices.dtype != DataType::Int64) ||
		    indices.shape != std::vector<std::size_t>{ 1 } || updates.shape.size() != data.shape.size())
		{
			return std::nullopt;
		}

		auto expectedUpdatesShape = data.shape;
		expectedUpdatesShape[0] = 1;
		if (updates.shape != expectedUpdatesShape)
		{
			return std::nullopt;
		}

		const auto dataElementCount = ShapeNumElementsU32(data.shape);
		const auto indexElementCount = ShapeNumElementsU32(indices.shape);
		const auto updateElementCount = ShapeNumElementsU32(updates.shape);
		if (!dataElementCount || !indexElementCount || !updateElementCount || data.shape[0] == 0 ||
		    *dataElementCount % data.shape[0] != 0)
		{
			return std::nullopt;
		}

		const auto rowSize64 = static_cast<std::uint64_t>(*dataElementCount) / data.shape[0];
		if (rowSize64 == 0 || rowSize64 > std::numeric_limits<std::uint32_t>::max())
		{
			return std::nullopt;
		}

		return CUDANativeScatterUpdatePlan{
			.dataInputIndex = *dataInputIndex,
			.indicesInputIndex = *indicesInputIndex,
			.updatesInputIndex = *updatesInputIndex,
			.indexType = indices.dtype,
			.rowSize = static_cast<std::uint32_t>(rowSize64),
			.dataElementCount = *dataElementCount,
			.indexElementCount = *indexElementCount,
			.updateElementCount = *updateElementCount,
		};
	}

	bool IsSupportedCUDANativeLowPrecisionMatMulType(DataType dtype)
	{
		switch (dtype)
		{
		case DataType::Float16:
		case DataType::BFloat16:
		case DataType::Float8E4M3:
		case DataType::Float8E5M2:
		case DataType::Int8:
			return true;
		default:
			return false;
		}
	}

	bool IsSupportedCUDANativeMatMulBiasType(DataType dtype)
	{
		return dtype == DataType::Float32 || dtype == DataType::Float16 || dtype == DataType::BFloat16 ||
		       dtype == DataType::Int8;
	}

	void AddCUDANativeMatMulFeatureFlag(CUDANativeInstructionPayload& payload, DataType dtype)
	{
		payload.featureSet.AddFeature(dtype == DataType::Float32 ? CUDANativeFeature::MatMulCUBLASF32
		                                                         : CUDANativeFeature::MatMulCUBLASLowPrecision);
	}

	void AddCUDANativeMatMulBiasFeatureFlags(CUDANativeInstructionPayload& payload, DataType dtype, bool relu)
	{
		if (dtype == DataType::Float32)
		{
			payload.featureSet.AddFeature(relu ? CUDANativeFeature::MatMulBiasAddReLUF32
			                                   : CUDANativeFeature::MatMulBiasAddF32);
			return;
		}
		payload.featureSet.AddFeature(relu ? CUDANativeFeature::MatMulBiasAddReLULowPrecision
		                                   : CUDANativeFeature::MatMulBiasAddLowPrecision);
	}

	std::string_view CUDANativeMatMulLibraryCallKernelName(DataType dtype)
	{
		switch (dtype)
		{
		case DataType::Float32:
			return "litenn_cublas_matmul_f32";
		case DataType::Float16:
			return "litenn_cublas_matmul_f16";
		case DataType::BFloat16:
			return "litenn_cublas_matmul_bf16";
		case DataType::Float8E4M3:
			return "litenn_cublas_matmul_f8e4m3";
		case DataType::Float8E5M2:
			return "litenn_cublas_matmul_f8e5m2";
		case DataType::Int8:
			return "litenn_cublas_matmul_i8";
		case DataType::UInt8:
			return "litenn_cublas_matmul_u8";
		default:
			throw std::runtime_error("Unsupported CUDA native MatMul library-call dtype");
		}
	}

	std::optional<CUDANativeMatMulPlan> MatchCUDANativeMatMulLowPrecision(const Graph& graph)
	{
		if (!IsCUDANativeSingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 2 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 3)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* binary = std::get_if<BinaryOpNode>(&resultEntry.node);
		if (!binary || binary->op != BinaryOp::MatMul)
		{
			return std::nullopt;
		}

		const auto lhsInputIndex = GetParamIndex(subgraph, binary->lhs);
		const auto rhsInputIndex = GetParamIndex(subgraph, binary->rhs);
		if (!lhsInputIndex || !rhsInputIndex)
		{
			return std::nullopt;
		}

		const auto& lhsParam = subgraph.Params()[*lhsInputIndex];
		const auto& rhsParam = subgraph.Params()[*rhsInputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (!IsSupportedCUDANativeLowPrecisionMatMulType(lhsParam.dtype) || lhsParam.dtype != rhsParam.dtype ||
		    lhsParam.dtype != output.dtype || lhsParam.shape.size() != 2 || rhsParam.shape.size() != 2 ||
		    output.shape.size() != 2 || lhsParam.shape[1] != rhsParam.shape[0] ||
		    output.shape[0] != lhsParam.shape[0] || output.shape[1] != rhsParam.shape[1])
		{
			return std::nullopt;
		}

		const auto m = lhsParam.shape[0];
		const auto k = lhsParam.shape[1];
		const auto n = rhsParam.shape[1];
		const auto maxInt = static_cast<std::size_t>(std::numeric_limits<int>::max());
		if (m == 0 || k == 0 || n == 0 || m > maxInt || k > maxInt || n > maxInt)
		{
			return std::nullopt;
		}

		const auto lhsElementCount = ShapeView{ lhsParam.shape }.NumElements();
		const auto rhsElementCount = ShapeView{ rhsParam.shape }.NumElements();
		const auto outputElementCount = ShapeView{ output.shape }.NumElements();
		if (lhsElementCount > std::numeric_limits<std::uint32_t>::max() ||
		    rhsElementCount > std::numeric_limits<std::uint32_t>::max() ||
		    outputElementCount > std::numeric_limits<std::uint32_t>::max())
		{
			return std::nullopt;
		}

		return CUDANativeMatMulPlan{
			.lhsInputIndex = *lhsInputIndex,
			.rhsInputIndex = *rhsInputIndex,
			.m = static_cast<std::uint32_t>(m),
			.k = static_cast<std::uint32_t>(k),
			.n = static_cast<std::uint32_t>(n),
			.lhsElementCount = static_cast<std::uint32_t>(lhsElementCount),
			.rhsElementCount = static_cast<std::uint32_t>(rhsElementCount),
			.outputElementCount = static_cast<std::uint32_t>(outputElementCount),
		};
	}

	std::optional<CUDANativeTensorRef> GetCUDANativeQuantizedRhsStorageRef(const Graph& graph, const Subgraph& subgraph,
	                                                                       NodeOutput output,
	                                                                       CUDANativeInstructionPayload& payload)
	{
		if (output.port != 0 || output.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& entry = subgraph.GetNodeEntry(output.node);
		if (entry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}
		const auto& info = entry.outputInfos[0];
		if (info.dtype != DataType::UInt8)
		{
			return std::nullopt;
		}
		if (const auto inputIndex = GetParamIndex(subgraph, output))
		{
			return CUDANativeTensorRef{
				.kind = CUDANativeArgumentKind::InputTensor,
				.index = *inputIndex,
				.byteOffset = 0,
				.byteSize = TensorByteSize(info.dtype, info.shape),
				.dtype = info.dtype,
				.shape = info.shape,
			};
		}
		if (const auto* variable = std::get_if<VariableRefNode>(&entry.node))
		{
			if (variable->variableIndex >= graph.VariableCount())
			{
				return std::nullopt;
			}
			const auto& graphVariable = graph.GetVariable(variable->variableIndex);
			const auto& storage = graphVariable->Data();
			if (storage.DType() != DataType::UInt8 || storage.Shape() != ShapeView{ info.shape })
			{
				return std::nullopt;
			}
			const auto offset = AppendCUDANativeConstantTensor(payload, storage);
			return CUDANativeTensorRef{
				.kind = CUDANativeArgumentKind::ConstantTensor,
				.index = 0,
				.byteOffset = offset,
				.byteSize = TensorByteSize(info.dtype, info.shape),
				.dtype = info.dtype,
				.shape = info.shape,
			};
		}
		return std::nullopt;
	}

	bool IsSupportedCUDANativeGGMLBlockMatMulFormat(QuantizedBlockFormat format)
	{
		return format == QuantizedBlockFormat::GGML_Q8_0 || format == QuantizedBlockFormat::GGML_Q4_K ||
		       format == QuantizedBlockFormat::GGML_Q5_K || format == QuantizedBlockFormat::GGML_Q6_K;
	}

	CUDANativeFeature CUDANativeGGMLBlockMatMulFeature(QuantizedBlockFormat format)
	{
		switch (format)
		{
		case QuantizedBlockFormat::GGML_Q8_0:
			return CUDANativeFeature::GGMLQ8_0MatMulF32;
		case QuantizedBlockFormat::GGML_Q4_K:
			return CUDANativeFeature::GGMLQ4_KMatMulF32;
		case QuantizedBlockFormat::GGML_Q5_K:
			return CUDANativeFeature::GGMLQ5_KMatMulF32;
		case QuantizedBlockFormat::GGML_Q6_K:
			return CUDANativeFeature::GGMLQ6_KMatMulF32;
		default:
			throw std::runtime_error("Unsupported CUDA native GGML block MatMul format");
		}
	}

	std::optional<CUDANativeGGMLBlockMatMulPlan>
	MatchCUDANativeGGMLBlockQuantizedMatMul(const Graph& graph, CUDANativeInstructionPayload& payload)
	{
		if (graph.SubgraphCount() != 1 || graph.Backward().has_value() || graph.ActivationSlotCount() != 0 ||
		    graph.TapeSlotCount() != 0)
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Results().size() != 1 || subgraph.NodeCount() != 3)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}
		const auto* matmul = std::get_if<QuantizedMatMulNode>(&resultEntry.node);
		if (!matmul || !matmul->transposeRhs || matmul->params.scheme != QuantizationScheme::Block ||
		    !IsSupportedCUDANativeGGMLBlockMatMulFormat(matmul->params.blockFormat) ||
		    matmul->params.storageType != DataType::UInt8 || matmul->params.expressedType != DataType::Float32)
		{
			return std::nullopt;
		}

		const auto lhsInputIndex = GetParamIndex(subgraph, matmul->lhs);
		if (!lhsInputIndex)
		{
			return std::nullopt;
		}
		const auto& lhsParam = subgraph.Params()[*lhsInputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (lhsParam.dtype != DataType::Float32 || output.dtype != DataType::Float32 || lhsParam.shape.size() != 2 ||
		    output.shape.size() != 2 || matmul->params.expressedShape.size() != 2)
		{
			return std::nullopt;
		}

		const auto m = lhsParam.shape[0];
		const auto k = lhsParam.shape[1];
		const auto n = output.shape[1];
		const auto logicalK = matmul->params.expressedShape[1];
		const auto logicalN = matmul->params.expressedShape[0];
		const auto layout = GetQuantizedBlockLayout(matmul->params.blockFormat);
		if (!layout || m == 0 || k == 0 || n == 0 || k % layout->elementsPerBlock != 0 || logicalK != k ||
		    logicalN != n || output.shape[0] != m)
		{
			return std::nullopt;
		}

		const auto rhsStorage = GetCUDANativeQuantizedRhsStorageRef(graph, subgraph, matmul->rhsStorage, payload);
		if (!rhsStorage || rhsStorage->shape.size() != 2 || rhsStorage->shape[0] != n ||
		    rhsStorage->shape[1] != (k / layout->elementsPerBlock) * layout->bytesPerBlock)
		{
			return std::nullopt;
		}

		const auto lhsElementCount = ShapeView{ lhsParam.shape }.NumElements();
		const auto rhsElementCount = ShapeView{ rhsStorage->shape }.NumElements();
		const auto outputElementCount = ShapeView{ output.shape }.NumElements();
		if (m > std::numeric_limits<std::uint32_t>::max() || k > std::numeric_limits<std::uint32_t>::max() ||
		    n > std::numeric_limits<std::uint32_t>::max() ||
		    lhsElementCount > std::numeric_limits<std::uint32_t>::max() ||
		    rhsElementCount > std::numeric_limits<std::uint32_t>::max() ||
		    outputElementCount > std::numeric_limits<std::uint32_t>::max())
		{
			return std::nullopt;
		}

		return CUDANativeGGMLBlockMatMulPlan{
			.format = matmul->params.blockFormat,
			.lhsInputIndex = *lhsInputIndex,
			.rhsStorage = *rhsStorage,
			.m = static_cast<std::uint32_t>(m),
			.k = static_cast<std::uint32_t>(k),
			.n = static_cast<std::uint32_t>(n),
			.lhsElementCount = static_cast<std::uint32_t>(lhsElementCount),
			.rhsStorageElementCount = static_cast<std::uint32_t>(rhsElementCount),
			.outputElementCount = static_cast<std::uint32_t>(outputElementCount),
		};
	}

	std::optional<CUDANativeMatMulBiasPlan>
	MakeCUDANativeMatMulBiasPlan(const Subgraph& subgraph, std::uint32_t lhsInputIndex, std::uint32_t rhsInputIndex,
	                             std::uint32_t biasInputIndex, const OutputInfo& output, bool relu)
	{
		const auto& lhsParam = subgraph.Params()[lhsInputIndex];
		const auto& rhsParam = subgraph.Params()[rhsInputIndex];
		const auto& biasParam = subgraph.Params()[biasInputIndex];
		if (lhsParam.dtype != rhsParam.dtype || rhsParam.dtype != biasParam.dtype || biasParam.dtype != output.dtype ||
		    !IsSupportedCUDANativeMatMulBiasType(output.dtype) || lhsParam.shape.size() != 2 ||
		    rhsParam.shape.size() != 2 || output.shape.size() != 2 || biasParam.shape.size() != output.shape.size() ||
		    lhsParam.shape[1] != rhsParam.shape[0] || output.shape[0] != lhsParam.shape[0] ||
		    output.shape[1] != rhsParam.shape[1] ||
		    !IsSameRankBroadcastCompatible(output.shape, biasParam.shape, output.shape))
		{
			return std::nullopt;
		}

		const auto m = lhsParam.shape[0];
		const auto k = lhsParam.shape[1];
		const auto n = rhsParam.shape[1];
		const auto maxInt = static_cast<std::size_t>(std::numeric_limits<int>::max());
		if (m == 0 || k == 0 || n == 0 || m > maxInt || k > maxInt || n > maxInt)
		{
			return std::nullopt;
		}

		const auto lhsElementCount = ShapeNumElementsU32(lhsParam.shape);
		const auto rhsElementCount = ShapeNumElementsU32(rhsParam.shape);
		const auto biasElementCount = ShapeNumElementsU32(biasParam.shape);
		const auto outputElementCount = ShapeNumElementsU32(output.shape);
		if (!lhsElementCount || !rhsElementCount || !biasElementCount || !outputElementCount)
		{
			return std::nullopt;
		}

		return CUDANativeMatMulBiasPlan{
			.dtype = output.dtype,
			.lhsInputIndex = lhsInputIndex,
			.rhsInputIndex = rhsInputIndex,
			.biasInputIndex = biasInputIndex,
			.m = static_cast<std::uint32_t>(m),
			.k = static_cast<std::uint32_t>(k),
			.n = static_cast<std::uint32_t>(n),
			.lhsElementCount = *lhsElementCount,
			.rhsElementCount = *rhsElementCount,
			.biasElementCount = *biasElementCount,
			.outputElementCount = *outputElementCount,
			.outputShape = output.shape,
			.biasShape = biasParam.shape,
			.relu = relu,
		};
	}

	bool IsZeroConstant(const Subgraph& subgraph, NodeOutput output)
	{
		if (output.port != 0 || output.node >= subgraph.NodeCount())
		{
			return false;
		}
		const auto* constant = std::get_if<ConstantNode>(&subgraph.GetNodeEntry(output.node).node);
		if (!constant)
		{
			return false;
		}
		const auto cpuTensor = constant->value.CopyToDevice(CPU{});
		std::vector<double> values(cpuTensor.NumElements());
		CPU cpu;
		DeviceTraits<CPU>::ConvertTo(cpu, cpuTensor.DType(), cpuTensor.UnsafeRawData(), cpuTensor.NumElements(),
		                             DataType::Float64, values.data());
		for (double value : values)
		{
			if (value != 0.0)
			{
				return false;
			}
		}
		return true;
	}

	std::optional<CUDANativeMatMulBiasPlan> MatchCUDANativeMatMulBias(const Graph& graph)
	{
		if (graph.Backward().has_value() || graph.VariableCount() != 0 || graph.ActivationSlotCount() != 0 ||
		    graph.TapeSlotCount() != 0)
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Results().size() != 1)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		if (const auto* fused = std::get_if<FusedOpNode>(&resultEntry.node))
		{
			if ((fused->pattern != FusionPattern::MatMulBiasAdd &&
			     fused->pattern != FusionPattern::MatMulBiasAddReLU) ||
			    fused->args.size() < 3)
			{
				return std::nullopt;
			}
			const auto lhsInputIndex = GetParamIndex(subgraph, fused->args[0]);
			const auto rhsInputIndex = GetParamIndex(subgraph, fused->args[1]);
			const auto biasInputIndex = GetParamIndex(subgraph, fused->args[2]);
			if (!lhsInputIndex || !rhsInputIndex || !biasInputIndex)
			{
				return std::nullopt;
			}
			return MakeCUDANativeMatMulBiasPlan(subgraph, *lhsInputIndex, *rhsInputIndex, *biasInputIndex,
			                                    resultEntry.outputInfos[0],
			                                    fused->pattern == FusionPattern::MatMulBiasAddReLU);
		}

		bool relu = false;
		NodeOutput addOutput = result;
		if (const auto* maxNode = std::get_if<BinaryOpNode>(&resultEntry.node); maxNode && maxNode->op == BinaryOp::Max)
		{
			if (IsZeroConstant(subgraph, maxNode->lhs))
			{
				addOutput = maxNode->rhs;
			}
			else if (IsZeroConstant(subgraph, maxNode->rhs))
			{
				addOutput = maxNode->lhs;
			}
			else
			{
				return std::nullopt;
			}
			relu = true;
		}

		if (addOutput.port != 0 || addOutput.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& addEntry = subgraph.GetNodeEntry(addOutput.node);
		const auto* addNode = std::get_if<BinaryOpNode>(&addEntry.node);
		if (!addNode || addNode->op != BinaryOp::Add)
		{
			return std::nullopt;
		}

		NodeOutput matmulOutput{};
		NodeOutput biasOutput{};
		if (addNode->lhs.port == 0 && addNode->lhs.node < subgraph.NodeCount())
		{
			if (const auto* lhsBinary = std::get_if<BinaryOpNode>(&subgraph.GetNodeEntry(addNode->lhs.node).node);
			    lhsBinary && lhsBinary->op == BinaryOp::MatMul)
			{
				matmulOutput = addNode->lhs;
				biasOutput = addNode->rhs;
			}
		}
		if (biasOutput.node == 0 && biasOutput.port == 0 && addNode->rhs.port == 0 &&
		    addNode->rhs.node < subgraph.NodeCount())
		{
			if (const auto* rhsBinary = std::get_if<BinaryOpNode>(&subgraph.GetNodeEntry(addNode->rhs.node).node);
			    rhsBinary && rhsBinary->op == BinaryOp::MatMul)
			{
				matmulOutput = addNode->rhs;
				biasOutput = addNode->lhs;
			}
		}
		if (matmulOutput.port != 0 || matmulOutput.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto* matmul = std::get_if<BinaryOpNode>(&subgraph.GetNodeEntry(matmulOutput.node).node);
		if (!matmul || matmul->op != BinaryOp::MatMul)
		{
			return std::nullopt;
		}
		const auto lhsInputIndex = GetParamIndex(subgraph, matmul->lhs);
		const auto rhsInputIndex = GetParamIndex(subgraph, matmul->rhs);
		const auto biasInputIndex = GetParamIndex(subgraph, biasOutput);
		if (!lhsInputIndex || !rhsInputIndex || !biasInputIndex)
		{
			return std::nullopt;
		}
		return MakeCUDANativeMatMulBiasPlan(subgraph, *lhsInputIndex, *rhsInputIndex, *biasInputIndex,
		                                    resultEntry.outputInfos[0], relu);
	}

	std::optional<CUDANativeReducePlan> MatchCUDANativeReduceF32(const Graph& graph)
	{
		if (!IsCUDANativeSingleForwardGraph(graph))
		{
			return std::nullopt;
		}
		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 1 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 2)
		{
			return std::nullopt;
		}
		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		const auto* reduce = std::get_if<ReduceOpNode>(&resultEntry.node);
		if (!reduce || !IsSupportedCUDANativeReduceF32Op(reduce->op) || resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}
		const auto inputIndex = GetParamIndex(subgraph, reduce->input);
		if (!inputIndex)
		{
			return std::nullopt;
		}
		const auto& input = subgraph.Params()[*inputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32 || reduce->axis >= input.shape.size())
		{
			return std::nullopt;
		}
		const auto inputElementCount = ShapeNumElementsU32(input.shape);
		const auto outputElementCount = ShapeNumElementsU32(output.shape);
		if (!inputElementCount || !outputElementCount)
		{
			return std::nullopt;
		}
		return CUDANativeReducePlan{ reduce->op,   *inputIndex, *inputElementCount, *outputElementCount,
			                         reduce->axis, input.shape, output.shape };
	}

	std::optional<CUDANativeSoftmaxPlan> MatchCUDANativeSoftmaxF32(const Graph& graph)
	{
		if (!IsCUDANativeSingleForwardGraph(graph))
		{
			return std::nullopt;
		}
		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 1 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 2)
		{
			return std::nullopt;
		}
		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		const auto* softmax = std::get_if<SoftmaxNode>(&resultEntry.node);
		if (!softmax || resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}
		const auto inputIndex = GetParamIndex(subgraph, softmax->input);
		if (!inputIndex)
		{
			return std::nullopt;
		}
		const auto& input = subgraph.Params()[*inputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32 || input.shape != output.shape ||
		    softmax->axis >= input.shape.size())
		{
			return std::nullopt;
		}
		const auto elementCount = ShapeNumElementsU32(input.shape);
		if (!elementCount)
		{
			return std::nullopt;
		}
		return CUDANativeSoftmaxPlan{ *inputIndex, *elementCount, softmax->axis, input.shape };
	}

	std::optional<CUDANativeGetRowsPlan> MatchCUDANativeGetRowsF32(const Graph& graph,
	                                                               CUDANativeInstructionPayload& payload)
	{
		if (graph.SubgraphCount() != 1 || graph.Backward().has_value() || graph.ActivationSlotCount() != 0 ||
		    graph.TapeSlotCount() != 0)
		{
			return std::nullopt;
		}
		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Results().size() != 1 || subgraph.NodeCount() != 3)
		{
			return std::nullopt;
		}
		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		const auto* getRows = std::get_if<GetRowsNode>(&resultEntry.node);
		if (!getRows || resultEntry.outputInfos.size() != 1 || getRows->data.port != 0 ||
		    getRows->data.node >= subgraph.NodeCount() || getRows->indices.port != 0 ||
		    getRows->indices.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& tableEntry = subgraph.GetNodeEntry(getRows->data.node);
		const auto& indicesEntry = subgraph.GetNodeEntry(getRows->indices.node);
		if (tableEntry.outputInfos.size() != 1 || indicesEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}
		const auto& tableInfo = tableEntry.outputInfos[0];
		const auto& indicesInfo = indicesEntry.outputInfos[0];
		const auto& outputInfo = resultEntry.outputInfos[0];
		if (tableInfo.dtype != DataType::Float32 || outputInfo.dtype != DataType::Float32 || tableInfo.shape.empty() ||
		    (indicesInfo.dtype != DataType::Int32 && indicesInfo.dtype != DataType::Int64))
		{
			return std::nullopt;
		}
		auto expectedShape = indicesInfo.shape;
		expectedShape.insert(expectedShape.end(), tableInfo.shape.begin() + 1, tableInfo.shape.end());
		if (expectedShape != outputInfo.shape)
		{
			return std::nullopt;
		}
		const auto tableElementCount = ShapeNumElementsU32(tableInfo.shape);
		const auto indexCount = ShapeNumElementsU32(indicesInfo.shape);
		const auto outputElementCount = ShapeNumElementsU32(outputInfo.shape);
		if (!tableElementCount || !indexCount || !outputElementCount || tableInfo.shape[0] == 0 ||
		    *tableElementCount % tableInfo.shape[0] != 0)
		{
			return std::nullopt;
		}
		const auto rowSize64 = static_cast<std::uint64_t>(*tableElementCount) / tableInfo.shape[0];
		if (rowSize64 == 0 || rowSize64 > std::numeric_limits<std::uint32_t>::max())
		{
			return std::nullopt;
		}

		CUDANativeTensorRef tableRef;
		if (const auto tableInputIndex = GetParamIndex(subgraph, getRows->data))
		{
			tableRef = { .kind = CUDANativeArgumentKind::InputTensor,
				         .index = *tableInputIndex,
				         .byteOffset = 0,
				         .byteSize = TensorByteSize(tableInfo.dtype, tableInfo.shape),
				         .dtype = tableInfo.dtype,
				         .shape = tableInfo.shape };
		}
		else if (const auto* variable = std::get_if<VariableRefNode>(&tableEntry.node))
		{
			if (variable->variableIndex >= graph.VariableCount())
			{
				return std::nullopt;
			}
			const auto& graphVariable = graph.GetVariable(variable->variableIndex);
			const auto& tensor = graphVariable->Data();
			if (graphVariable->HasGradStorage() || tensor.DType() != tableInfo.dtype ||
			    tensor.Shape() != tableInfo.shape)
			{
				return std::nullopt;
			}
			tableRef = { .kind = CUDANativeArgumentKind::ConstantTensor,
				         .index = 0,
				         .byteOffset = AppendCUDANativeConstantTensor(payload, tensor),
				         .byteSize = TensorByteSize(tableInfo.dtype, tableInfo.shape),
				         .dtype = tableInfo.dtype,
				         .shape = tableInfo.shape };
		}
		else if (const auto* constant = std::get_if<ConstantNode>(&tableEntry.node))
		{
			if (constant->value.DType() != tableInfo.dtype || constant->value.Shape() != tableInfo.shape)
			{
				return std::nullopt;
			}
			tableRef = { .kind = CUDANativeArgumentKind::ConstantTensor,
				         .index = 0,
				         .byteOffset = AppendCUDANativeConstantTensor(payload, constant->value),
				         .byteSize = TensorByteSize(tableInfo.dtype, tableInfo.shape),
				         .dtype = tableInfo.dtype,
				         .shape = tableInfo.shape };
		}
		else
		{
			return std::nullopt;
		}

		const auto indicesInputIndex = GetParamIndex(subgraph, getRows->indices);
		if (!indicesInputIndex)
		{
			return std::nullopt;
		}
		return CUDANativeGetRowsPlan{ .table = std::move(tableRef),
			                          .indicesInputIndex = *indicesInputIndex,
			                          .indexType = indicesInfo.dtype,
			                          .rowSize = static_cast<std::uint32_t>(rowSize64),
			                          .indexCount = *indexCount,
			                          .outputElementCount = *outputElementCount };
	}

	std::optional<CUDANativeRMSNormPlan> MatchCUDANativeRMSNormF32(const Graph& graph,
	                                                               CUDANativeInstructionPayload& payload)
	{
		if (graph.SubgraphCount() != 1 || graph.Backward().has_value() || graph.ActivationSlotCount() != 0 ||
		    graph.TapeSlotCount() != 0)
		{
			return std::nullopt;
		}
		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Results().size() != 1 || subgraph.NodeCount() < 2 || subgraph.NodeCount() > 3)
		{
			return std::nullopt;
		}
		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		const auto* norm = std::get_if<NormalizationNode>(&resultEntry.node);
		if (!norm || norm->mode != NormalizationMode::RMSNorm || norm->bias || norm->groupCount != 1 ||
		    resultEntry.outputInfos.size() != 1 || !std::isfinite(norm->epsilon) || norm->epsilon <= 0.0)
		{
			return std::nullopt;
		}
		const auto inputIndex = GetParamIndex(subgraph, norm->input);
		if (!inputIndex)
		{
			return std::nullopt;
		}
		const auto& input = subgraph.Params()[*inputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32 || input.shape != output.shape ||
		    input.shape.empty() || norm->axis != input.shape.size() - 1 || input.shape.back() == 0 ||
		    input.shape.back() > std::numeric_limits<std::uint32_t>::max())
		{
			return std::nullopt;
		}
		const auto elementCount = ShapeNumElementsU32(input.shape);
		if (!elementCount)
		{
			return std::nullopt;
		}

		std::optional<CUDANativeTensorRef> scaleRef;
		if (norm->scale)
		{
			if (norm->scale->port != 0 || norm->scale->node >= subgraph.NodeCount())
			{
				return std::nullopt;
			}
			const auto& scaleEntry = subgraph.GetNodeEntry(norm->scale->node);
			if (scaleEntry.outputInfos.size() != 1)
			{
				return std::nullopt;
			}
			const auto& scaleInfo = scaleEntry.outputInfos[0];
			const auto scaleElementCount = ShapeNumElementsU32(scaleInfo.shape);
			if (scaleInfo.dtype != DataType::Float32 || !scaleElementCount || *scaleElementCount != input.shape.back())
			{
				return std::nullopt;
			}
			if (const auto scaleInputIndex = GetParamIndex(subgraph, *norm->scale))
			{
				scaleRef = CUDANativeTensorRef{ .kind = CUDANativeArgumentKind::InputTensor,
					                            .index = *scaleInputIndex,
					                            .byteOffset = 0,
					                            .byteSize = TensorByteSize(scaleInfo.dtype, scaleInfo.shape),
					                            .dtype = scaleInfo.dtype,
					                            .shape = scaleInfo.shape };
			}
			else if (const auto* variable = std::get_if<VariableRefNode>(&scaleEntry.node))
			{
				if (variable->variableIndex >= graph.VariableCount())
				{
					return std::nullopt;
				}
				const auto& graphVariable = graph.GetVariable(variable->variableIndex);
				const auto& tensor = graphVariable->Data();
				if (graphVariable->HasGradStorage() || tensor.DType() != scaleInfo.dtype ||
				    tensor.Shape() != scaleInfo.shape)
				{
					return std::nullopt;
				}
				scaleRef = CUDANativeTensorRef{ .kind = CUDANativeArgumentKind::ConstantTensor,
					                            .index = 0,
					                            .byteOffset = AppendCUDANativeConstantTensor(payload, tensor),
					                            .byteSize = TensorByteSize(scaleInfo.dtype, scaleInfo.shape),
					                            .dtype = scaleInfo.dtype,
					                            .shape = scaleInfo.shape };
			}
			else if (const auto* constant = std::get_if<ConstantNode>(&scaleEntry.node))
			{
				if (constant->value.DType() != scaleInfo.dtype || constant->value.Shape() != scaleInfo.shape)
				{
					return std::nullopt;
				}
				scaleRef = CUDANativeTensorRef{ .kind = CUDANativeArgumentKind::ConstantTensor,
					                            .index = 0,
					                            .byteOffset = AppendCUDANativeConstantTensor(payload, constant->value),
					                            .byteSize = TensorByteSize(scaleInfo.dtype, scaleInfo.shape),
					                            .dtype = scaleInfo.dtype,
					                            .shape = scaleInfo.shape };
			}
			else
			{
				return std::nullopt;
			}
		}

		return CUDANativeRMSNormPlan{ .inputIndex = *inputIndex,
			                          .scale = std::move(scaleRef),
			                          .rowSize = static_cast<std::uint32_t>(input.shape.back()),
			                          .elementCount = *elementCount,
			                          .epsilon = static_cast<float>(norm->epsilon) };
	}

	std::optional<CUDANativeRoPEPlan> MatchCUDANativeRoPEF32(const Graph& graph)
	{
		if (!IsCUDANativeSingleForwardGraph(graph))
		{
			return std::nullopt;
		}
		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Results().size() != 1 || subgraph.NodeCount() < 2 || subgraph.NodeCount() > 3)
		{
			return std::nullopt;
		}
		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		const auto* rope = std::get_if<RoPENode>(&resultEntry.node);
		if (!rope || resultEntry.outputInfos.size() != 1 || !std::isfinite(rope->base) || rope->base <= 0.0 ||
		    !std::isfinite(rope->frequencyScale) || rope->frequencyScale <= 0.0 ||
		    rope->positionOffset > std::numeric_limits<std::uint32_t>::max())
		{
			return std::nullopt;
		}
		const auto inputIndex = GetParamIndex(subgraph, rope->input);
		if (!inputIndex)
		{
			return std::nullopt;
		}
		const auto& input = subgraph.Params()[*inputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32 || input.shape != output.shape ||
		    input.shape.size() != 2 || input.shape[1] == 0 || (input.shape[1] % 2) != 0 ||
		    input.shape[1] > std::numeric_limits<std::uint32_t>::max())
		{
			return std::nullopt;
		}
		const auto elementCount = ShapeNumElementsU32(input.shape);
		if (!elementCount)
		{
			return std::nullopt;
		}

		std::optional<std::uint32_t> positionsInputIndex;
		std::optional<DataType> positionType;
		if (rope->positions)
		{
			positionsInputIndex = GetParamIndex(subgraph, *rope->positions);
			if (!positionsInputIndex)
			{
				return std::nullopt;
			}
			const auto& positions = subgraph.Params()[*positionsInputIndex];
			if ((positions.dtype != DataType::Int32 && positions.dtype != DataType::Int64) ||
			    positions.shape != std::vector<std::size_t>{ input.shape[0] })
			{
				return std::nullopt;
			}
			positionType = positions.dtype;
		}
		return CUDANativeRoPEPlan{ .inputIndex = *inputIndex,
			                       .positionsInputIndex = positionsInputIndex,
			                       .positionType = positionType,
			                       .featureSize = static_cast<std::uint32_t>(input.shape[1]),
			                       .elementCount = *elementCount,
			                       .positionOffset = static_cast<std::uint32_t>(rope->positionOffset),
			                       .base = rope->base,
			                       .frequencyScale = rope->frequencyScale };
	}

	std::optional<CUDANativeCastPlan> MatchCUDANativeCast(const Graph& graph)
	{
		if (!IsCUDANativeSingleForwardGraph(graph))
		{
			return std::nullopt;
		}
		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 1 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 2)
		{
			return std::nullopt;
		}
		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		const auto* castNode = std::get_if<CastNode>(&resultEntry.node);
		if (!castNode || resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}
		const auto inputIndex = GetParamIndex(subgraph, castNode->input);
		if (!inputIndex)
		{
			return std::nullopt;
		}
		const auto& input = subgraph.Params()[*inputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (output.dtype != castNode->targetType || input.shape != output.shape ||
		    !CUDANativeSupportsCast(input.dtype, output.dtype))
		{
			return std::nullopt;
		}
		const auto elementCount = ShapeNumElementsU32(output.shape);
		if (!elementCount)
		{
			return std::nullopt;
		}
		return CUDANativeCastPlan{
			.inputIndex = *inputIndex, .elementCount = *elementCount, .srcType = input.dtype, .dstType = output.dtype
		};
	}

	std::optional<CUDANativeConcatPlan> MatchCUDANativeConcatF32(const Graph& graph)
	{
		if (!IsCUDANativeSingleForwardGraph(graph))
		{
			return std::nullopt;
		}
		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().empty() || subgraph.Results().size() != 1)
		{
			return std::nullopt;
		}
		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		const auto* concat = std::get_if<ConcatNode>(&resultEntry.node);
		if (!concat || resultEntry.outputInfos.size() != 1 || concat->inputs.empty())
		{
			return std::nullopt;
		}
		const auto& output = resultEntry.outputInfos[0];
		if (output.dtype != DataType::Float32 || concat->axis >= output.shape.size())
		{
			return std::nullopt;
		}

		const auto outputElementCount = ShapeNumElementsU32(output.shape);
		if (!outputElementCount)
		{
			return std::nullopt;
		}

		CUDANativeConcatPlan plan;
		plan.outputElementCount = *outputElementCount;
		plan.outputShape = output.shape;
		plan.axis = concat->axis;
		for (const auto& inputOutput : concat->inputs)
		{
			const auto inputIndex = GetParamIndex(subgraph, inputOutput);
			if (!inputIndex)
			{
				return std::nullopt;
			}
			const auto& input = subgraph.Params()[*inputIndex];
			if (input.dtype != DataType::Float32 || input.shape.size() != output.shape.size())
			{
				return std::nullopt;
			}
			const auto inputElementCount = ShapeNumElementsU32(input.shape);
			if (!inputElementCount)
			{
				return std::nullopt;
			}
			plan.inputIndices.push_back(*inputIndex);
			plan.inputElementCounts.push_back(*inputElementCount);
			plan.inputShapes.push_back(input.shape);
		}
		return plan;
	}

	std::optional<CUDANativeSlicePlan> MatchCUDANativeSliceF32(const Graph& graph)
	{
		if (!IsCUDANativeSingleForwardGraph(graph))
		{
			return std::nullopt;
		}
		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 1 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 2)
		{
			return std::nullopt;
		}
		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		const auto* slice = std::get_if<SliceNode>(&resultEntry.node);
		if (!slice || resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}
		const auto inputIndex = GetParamIndex(subgraph, slice->input);
		if (!inputIndex)
		{
			return std::nullopt;
		}
		const auto& input = subgraph.Params()[*inputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32 ||
		    input.shape.size() != output.shape.size() || slice->axis >= input.shape.size())
		{
			return std::nullopt;
		}
		const auto inputElementCount = ShapeNumElementsU32(input.shape);
		const auto outputElementCount = ShapeNumElementsU32(output.shape);
		if (!inputElementCount || !outputElementCount)
		{
			return std::nullopt;
		}
		return CUDANativeSlicePlan{ *inputIndex,  *inputElementCount, *outputElementCount, slice->axis,
			                        slice->start, input.shape,        output.shape };
	}

	CUDANativeFeature CUDANativeBinaryF32FeatureFlag(BinaryOp op)
	{
		switch (op)
		{
		case BinaryOp::Add:
			return CUDANativeFeature::ElementwiseAddF32;
		case BinaryOp::Subtract:
			return CUDANativeFeature::ElementwiseSubtractF32;
		case BinaryOp::Multiply:
			return CUDANativeFeature::ElementwiseMultiplyF32;
		case BinaryOp::Divide:
			return CUDANativeFeature::ElementwiseDivideF32;
		case BinaryOp::Max:
			return CUDANativeFeature::ElementwiseMaxF32;
		case BinaryOp::Min:
			return CUDANativeFeature::ElementwiseMinF32;
		default:
			throw std::runtime_error("Unsupported CUDA native binary op");
		}
	}

	CUDANativeFeature CUDANativeUnaryF32FeatureFlag(UnaryOp op)
	{
		switch (op)
		{
		case UnaryOp::Negate:
			return CUDANativeFeature::ElementwiseNegateF32;
		case UnaryOp::Abs:
			return CUDANativeFeature::ElementwiseAbsF32;
		case UnaryOp::Sqrt:
			return CUDANativeFeature::ElementwiseSqrtF32;
		case UnaryOp::Exp:
			return CUDANativeFeature::ElementwiseExpF32;
		case UnaryOp::Log:
			return CUDANativeFeature::ElementwiseLogF32;
		case UnaryOp::Sin:
			return CUDANativeFeature::ElementwiseSinF32;
		case UnaryOp::Cos:
			return CUDANativeFeature::ElementwiseCosF32;
		default:
			throw std::runtime_error("Unsupported CUDA native unary op");
		}
	}

	std::optional<CUDANativeLinearChainPlan> BuildCUDANativeLinearChainPlan(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		if (graph.Backward().has_value() || graph.ActivationSlotCount() != 0 || graph.TapeSlotCount() != 0 ||
		    graph.SubgraphCount() == 0)
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Results().size() != 1)
		{
			return std::nullopt;
		}
		const auto finalResult = subgraph.Results()[0];
		if (finalResult.port != 0 || finalResult.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		CUDANativeLinearChainPlan plan;
		auto& payload = plan.payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeFeature::MultiKernelLaunch);
		payload.target = CUDANativeNVPTXTargetChip();

		std::vector<std::optional<CUDANativeTensorRef>> values(subgraph.NodeCount());
		std::size_t fusedLayerCount = 0;
		bool hasChainSpecificStorage = false;

		const auto tensorRefForOutput = [&](NodeId nodeId, const OutputInfo& output) {
			const auto byteSize = TensorByteSize(output.dtype, output.shape);
			if (finalResult.node == nodeId && finalResult.port == 0)
			{
				return CUDANativeTensorRef{
					.kind = CUDANativeArgumentKind::OutputTensor,
					.index = 0,
					.byteOffset = 0,
					.byteSize = byteSize,
					.dtype = output.dtype,
					.shape = output.shape,
				};
			}
			const auto offset = AllocateCUDANativeWorkspaceTensor(payload, byteSize);
			return CUDANativeTensorRef{
				.kind = CUDANativeArgumentKind::Workspace,
				.index = 0,
				.byteOffset = offset,
				.byteSize = byteSize,
				.dtype = output.dtype,
				.shape = output.shape,
			};
		};

		const auto requireValue = [&](NodeOutput output) -> std::optional<CUDANativeTensorRef> {
			if (output.port != 0 || output.node >= values.size() || !values[output.node])
			{
				return std::nullopt;
			}
			return *values[output.node];
		};

		for (NodeId nodeId = 0; nodeId < subgraph.NodeCount(); ++nodeId)
		{
			const auto& entry = subgraph.GetNodeEntry(nodeId);
			if (entry.outputInfos.size() != 1)
			{
				return std::nullopt;
			}
			const auto& output = entry.outputInfos[0];

			if (const auto* param = std::get_if<ParamRefNode>(&entry.node))
			{
				if (param->paramIndex > std::numeric_limits<std::uint32_t>::max())
				{
					return std::nullopt;
				}
				values[nodeId] = CUDANativeTensorRef{
					.kind = CUDANativeArgumentKind::InputTensor,
					.index = static_cast<std::uint32_t>(param->paramIndex),
					.byteOffset = 0,
					.byteSize = TensorByteSize(output.dtype, output.shape),
					.dtype = output.dtype,
					.shape = output.shape,
				};
				continue;
			}

			if (const auto* variable = std::get_if<VariableRefNode>(&entry.node))
			{
				if (variable->variableIndex >= graph.VariableCount())
				{
					return std::nullopt;
				}
				const auto& variableTensor = graph.GetVariable(variable->variableIndex)->Data();
				if (variableTensor.DType() != output.dtype || variableTensor.Shape() != output.shape)
				{
					return std::nullopt;
				}
				const auto offset = AppendCUDANativeConstantTensor(payload, variableTensor);
				values[nodeId] = CUDANativeTensorRef{
					.kind = CUDANativeArgumentKind::ConstantTensor,
					.index = 0,
					.byteOffset = offset,
					.byteSize = TensorByteSize(output.dtype, output.shape),
					.dtype = output.dtype,
					.shape = output.shape,
				};
				continue;
			}

			if (const auto* constant = std::get_if<ConstantNode>(&entry.node))
			{
				const auto offset = AppendCUDANativeConstantTensor(payload, constant->value);
				values[nodeId] = CUDANativeTensorRef{
					.kind = CUDANativeArgumentKind::ConstantTensor,
					.index = 0,
					.byteOffset = offset,
					.byteSize = TensorByteSize(output.dtype, output.shape),
					.dtype = output.dtype,
					.shape = output.shape,
				};
				continue;
			}

			const auto* fused = std::get_if<FusedOpNode>(&entry.node);
			if (!fused ||
			    (fused->pattern != FusionPattern::MatMulBiasAdd &&
			     fused->pattern != FusionPattern::MatMulBiasAddReLU) ||
			    fused->args.size() < 3)
			{
				return std::nullopt;
			}

			auto lhs = requireValue(fused->args[0]);
			auto rhs = requireValue(fused->args[1]);
			auto bias = requireValue(fused->args[2]);
			if (!lhs || !rhs || !bias || lhs->dtype != rhs->dtype || rhs->dtype != bias->dtype ||
			    bias->dtype != output.dtype || !IsSupportedCUDANativeMatMulBiasType(output.dtype) ||
			    lhs->shape.size() != 2 || rhs->shape.size() != 2 || output.shape.size() != 2 ||
			    bias->shape.size() != output.shape.size() || lhs->shape[1] != rhs->shape[0] ||
			    output.shape[0] != lhs->shape[0] || output.shape[1] != rhs->shape[1] ||
			    !IsSameRankBroadcastCompatible(output.shape, bias->shape, output.shape))
			{
				return std::nullopt;
			}

			const auto m = static_cast<std::uint32_t>(lhs->shape[0]);
			const auto k = static_cast<std::uint32_t>(lhs->shape[1]);
			const auto n = static_cast<std::uint32_t>(rhs->shape[1]);
			if (m == 0 || k == 0 || n == 0)
			{
				return std::nullopt;
			}

			auto target = tensorRefForOutput(nodeId, output);
			const auto outputElementCount = ShapeNumElementsU32(output.shape);
			if (!outputElementCount)
			{
				return std::nullopt;
			}
			hasChainSpecificStorage = hasChainSpecificStorage || target.kind == CUDANativeArgumentKind::Workspace ||
			                          lhs->kind == CUDANativeArgumentKind::Workspace ||
			                          rhs->kind == CUDANativeArgumentKind::Workspace ||
			                          bias->kind == CUDANativeArgumentKind::Workspace ||
			                          lhs->kind == CUDANativeArgumentKind::ConstantTensor ||
			                          rhs->kind == CUDANativeArgumentKind::ConstantTensor ||
			                          bias->kind == CUDANativeArgumentKind::ConstantTensor;
			const auto mArg = AppendU32ScalarArgument(payload, m);
			const auto kArg = AppendU32ScalarArgument(payload, k);
			const auto nArg = AppendU32ScalarArgument(payload, n);
			AddCUDANativeMatMulFeatureFlag(payload, output.dtype);
			payload.kernels.push_back({
			    .name = std::string(CUDANativeMatMulLibraryCallKernelName(output.dtype)),
			    .grid = { .x = 1, .y = 1, .z = 1 },
			    .block = { .x = 1, .y = 1, .z = 1 },
			    .arguments = {
			        ToCUDANativeArgument(target),
			        ToCUDANativeArgument(*lhs),
			        ToCUDANativeArgument(*rhs),
			        mArg,
			        kArg,
			        nArg,
			    },
			});

			const bool relu = fused->pattern == FusionPattern::MatMulBiasAddReLU;
			const auto epilogueName =
			    std::format("{}_{}", CUDANativeMatMulBiasEpilogueKernelName(output.dtype, relu), fusedLayerCount);
			const auto countArg = AppendU32ScalarArgument(payload, *outputElementCount);
			const auto blockSize = std::min<std::uint32_t>(*outputElementCount, 256);
			const auto gridSize = (*outputElementCount + blockSize - 1) / blockSize;
			payload.kernels.push_back({
			    .name = epilogueName,
			    .grid = { .x = gridSize, .y = 1, .z = 1 },
			    .block = { .x = blockSize, .y = 1, .z = 1 },
			    .arguments = {
			        ToCUDANativeArgument(target),
			        ToCUDANativeArgument(*bias),
			        countArg,
			    },
			});
			plan.epilogues.push_back({
			    .kernelName = epilogueName,
			    .dtype = output.dtype,
			    .outputShape = output.shape,
			    .biasShape = bias->shape,
			    .relu = relu,
			});
			AddCUDANativeMatMulBiasFeatureFlags(payload, output.dtype, relu);
			values[nodeId] = std::move(target);
			++fusedLayerCount;
		}

		if (fusedLayerCount == 0 || !values[finalResult.node])
		{
			return std::nullopt;
		}
		if (fusedLayerCount == 1 && !hasChainSpecificStorage)
		{
			return std::nullopt;
		}
		if (payload.workspaceBytes != 0)
		{
			payload.featureSet.AddFeature(CUDANativeFeature::Workspace);
		}
		if (!payload.constantData.empty())
		{
			payload.featureSet.AddFeature(CUDANativeFeature::ConstantTensor);
		}

		const auto ptx = TryCUDANativeMatMulBiasEpiloguesPTXFromMLIRNVPTX(plan.epilogues);
		if (!ptx)
		{
			return std::nullopt;
		}
		payload.binary = CUDANativeTextBytes(*ptx);
		return plan;
#else
		(void) graph;
		return std::nullopt;
#endif
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeLinearChain(const Graph& graph)
	{
		auto plan = BuildCUDANativeLinearChainPlan(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(plan->payload);
		return CUDANativeArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeUnaryF32(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		const auto plan = MatchCUDANativeUnaryF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeUnaryF32FeatureFlag(plan->op));
		payload.target = CUDANativeNVPTXTargetChip();
		const auto mlirPtx = TryCUDANativeUnaryF32PTXFromMLIRNVPTX(plan->op);
		if (!mlirPtx)
		{
			return std::nullopt;
		}
		payload.binary = CUDANativeTextBytes(*mlirPtx);
		AppendU32(payload.scalarData, plan->elementCount);

		const auto blockSize = std::min<std::uint32_t>(plan->elementCount, 256);
		const auto gridSize = (plan->elementCount + blockSize - 1) / blockSize;
		const auto tensorByteSize = static_cast<std::uint64_t>(plan->elementCount) * sizeof(float);
		payload.kernels.push_back({
		    .name = std::string(CUDANativeUnaryF32KernelName(plan->op)),
		    .grid = { .x = gridSize, .y = 1, .z = 1 },
		    .block = { .x = blockSize, .y = 1, .z = 1 },
		    .sharedMemoryBytes = 0,
		    .workspaceBytes = 0,
		    .arguments = {
		        { .kind = CUDANativeArgumentKind::OutputTensor, .index = 0, .byteOffset = 0, .byteSize = tensorByteSize },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->inputIndex,
		          .byteOffset = 0,
		          .byteSize = tensorByteSize },
		        { .kind = CUDANativeArgumentKind::Scalar, .index = 0, .byteOffset = 0, .byteSize = sizeof(std::uint32_t) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
#else
		(void) graph;
		return std::nullopt;
#endif
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeSiLUF32(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		const auto plan = MatchCUDANativeSiLUF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}
		const auto ptx = TryCUDANativeSiLUF32PTXFromMLIRNVPTX();
		if (!ptx)
		{
			return std::nullopt;
		}

		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeFeature::ElementwiseSiLUF32);
		payload.target = CUDANativeNVPTXTargetChip();
		payload.binary = CUDANativeTextBytes(*ptx);
		AppendU32(payload.scalarData, plan->elementCount);

		const auto blockSize = std::min<std::uint32_t>(plan->elementCount, 256);
		const auto gridSize = (plan->elementCount + blockSize - 1) / blockSize;
		payload.kernels.push_back({
		    .name = std::string(CUDANativeSiLUF32KernelName()),
		    .grid = { .x = gridSize, .y = 1, .z = 1 },
		    .block = { .x = blockSize, .y = 1, .z = 1 },
		    .arguments = {
		        { .kind = CUDANativeArgumentKind::OutputTensor,
		          .index = 0,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->elementCount) * sizeof(float) },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->inputIndex,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->elementCount) * sizeof(float) },
		        { .kind = CUDANativeArgumentKind::Scalar, .index = 0, .byteOffset = 0, .byteSize = sizeof(std::uint32_t) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{ std::move(rodata), std::move(instructions), std::move(inputSpecs),
			                            std::move(outputSpecs) };
#else
		(void) graph;
		return std::nullopt;
#endif
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeCast(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		const auto plan = MatchCUDANativeCast(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeFeature::Cast);
		payload.target = CUDANativeNVPTXTargetChip();
		const auto mlirPtx = TryCUDANativeCastPTXFromMLIRNVPTX(
		    CUDANativeCastCodegenSpec{ .srcType = plan->srcType, .dstType = plan->dstType });
		if (!mlirPtx)
		{
			return std::nullopt;
		}
		payload.binary = CUDANativeTextBytes(*mlirPtx);
		AppendU32(payload.scalarData, plan->elementCount);

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		const auto blockSize = std::min<std::uint32_t>(plan->elementCount, 256);
		const auto gridSize = (plan->elementCount + blockSize - 1) / blockSize;
		payload.kernels.push_back({
		    .name = CUDANativeCastKernelName(plan->srcType, plan->dstType),
		    .grid = { .x = gridSize, .y = 1, .z = 1 },
		    .block = { .x = blockSize, .y = 1, .z = 1 },
		    .sharedMemoryBytes = 0,
		    .workspaceBytes = 0,
		    .arguments = {
		        { .kind = CUDANativeArgumentKind::OutputTensor,
		          .index = 0,
		          .byteOffset = 0,
		          .byteSize = TensorByteSize(outputSpecs[0].type.dtype, outputSpecs[0].type.StaticShape()) },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->inputIndex,
		          .byteOffset = 0,
		          .byteSize = TensorByteSize(inputSpecs[plan->inputIndex].type.dtype,
		                                     inputSpecs[plan->inputIndex].type.StaticShape()) },
		        { .kind = CUDANativeArgumentKind::Scalar,
		          .index = 0,
		          .byteOffset = 0,
		          .byteSize = sizeof(std::uint32_t) },
		    },
		});

		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
#else
		(void) graph;
		return std::nullopt;
#endif
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeBinaryF32(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.target = CUDANativeNVPTXTargetChip();
		const auto plan = MatchCUDANativeBinaryF32(graph, payload);
		if (!plan)
		{
			return std::nullopt;
		}

		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeBinaryF32FeatureFlag(plan->op));
		if (plan->requiresBroadcast)
		{
			payload.featureSet.AddFeature(CUDANativeFeature::ElementwiseBroadcastF32);
		}
		if (!payload.constantData.empty())
		{
			payload.featureSet.AddFeature(CUDANativeFeature::ConstantTensor);
		}
		std::string ptx;
		if (plan->requiresBroadcast)
		{
			const auto spec = CUDANativeBroadcastBinaryF32CodegenSpec{
				.op = plan->op,
				.outputShape = plan->outputShape,
				.lhsShape = plan->lhs.shape,
				.rhsShape = plan->rhs.shape,
			};
			const auto mlirPtx = TryCUDANativeBinaryBroadcastF32PTXFromMLIRNVPTX(spec);
			if (!mlirPtx)
			{
				return std::nullopt;
			}
			ptx = *mlirPtx;
		}
		else
		{
			const auto mlirPtx = TryCUDANativeBinaryF32PTXFromMLIRNVPTX(plan->op);
			if (!mlirPtx)
			{
				return std::nullopt;
			}
			ptx = *mlirPtx;
		}
		payload.binary = CUDANativeTextBytes(ptx);
		AppendU32(payload.scalarData, plan->elementCount);

		const auto blockSize = std::min<std::uint32_t>(plan->elementCount, 256);
		const auto gridSize = (plan->elementCount + blockSize - 1) / blockSize;
		const auto outputByteSize = static_cast<std::uint64_t>(plan->elementCount) * sizeof(float);
		payload.kernels.push_back({
		    .name = std::string(CUDANativeBinaryF32KernelName(plan->op, plan->requiresBroadcast)),
		    .grid = { .x = gridSize, .y = 1, .z = 1 },
		    .block = { .x = blockSize, .y = 1, .z = 1 },
		    .sharedMemoryBytes = 0,
		    .workspaceBytes = 0,
		    .arguments = {
		        { .kind = CUDANativeArgumentKind::OutputTensor, .index = 0, .byteOffset = 0, .byteSize = outputByteSize },
		        ToCUDANativeArgument(plan->lhs),
		        ToCUDANativeArgument(plan->rhs),
		        { .kind = CUDANativeArgumentKind::Scalar, .index = 0, .byteOffset = 0, .byteSize = sizeof(std::uint32_t) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
#else
		(void) graph;
		return std::nullopt;
#endif
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeMatMulF32(const Graph& graph)
	{
		const auto plan = MatchCUDANativeMatMulF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::LibraryCall;
		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeFeature::MatMulCUBLASF32);
		payload.target = "cublas";
		AppendU32(payload.scalarData, plan->m);
		AppendU32(payload.scalarData, plan->k);
		AppendU32(payload.scalarData, plan->n);

		const auto outputByteSize = static_cast<std::uint64_t>(plan->outputElementCount) * sizeof(float);
		const auto lhsByteSize = static_cast<std::uint64_t>(plan->lhsElementCount) * sizeof(float);
		const auto rhsByteSize = static_cast<std::uint64_t>(plan->rhsElementCount) * sizeof(float);
		payload.kernels.push_back({
		    .name = std::string(CUDANativeMatMulLibraryCallKernelName(DataType::Float32)),
		    .grid = { .x = 1, .y = 1, .z = 1 },
		    .block = { .x = 1, .y = 1, .z = 1 },
		    .sharedMemoryBytes = 0,
		    .workspaceBytes = 0,
		    .arguments = {
		        { .kind = CUDANativeArgumentKind::OutputTensor, .index = 0, .byteOffset = 0, .byteSize = outputByteSize },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->lhsInputIndex,
		          .byteOffset = 0,
		          .byteSize = lhsByteSize },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->rhsInputIndex,
		          .byteOffset = 0,
		          .byteSize = rhsByteSize },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeBatchMatMulF32(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		const auto plan = MatchCUDANativeBatchMatMulF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}
		const auto ptx = TryCUDANativeBatchMatMulF32PTXFromMLIRNVPTX({
		    .lhsShape = plan->lhsShape,
		    .rhsShape = plan->rhsShape,
		    .outputShape = plan->outputShape,
		});
		if (!ptx)
		{
			return std::nullopt;
		}

		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeFeature::BatchMatMulF32);
		payload.target = CUDANativeNVPTXTargetChip();
		payload.binary = CUDANativeTextBytes(*ptx);
		AppendU32(payload.scalarData, plan->outputElementCount);

		const auto blockSize = std::min<std::uint32_t>(plan->outputElementCount, 256);
		const auto gridSize = (plan->outputElementCount + blockSize - 1) / blockSize;
		payload.kernels.push_back({
		    .name = std::string(CUDANativeBatchMatMulF32KernelName()),
		    .grid = { .x = gridSize, .y = 1, .z = 1 },
		    .block = { .x = blockSize, .y = 1, .z = 1 },
		    .arguments = {
		        { .kind = CUDANativeArgumentKind::OutputTensor,
		          .index = 0,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->outputElementCount) * sizeof(float) },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->lhsInputIndex,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->lhsElementCount) * sizeof(float) },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->rhsInputIndex,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->rhsElementCount) * sizeof(float) },
		        { .kind = CUDANativeArgumentKind::Scalar, .index = 0, .byteOffset = 0, .byteSize = sizeof(std::uint32_t) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
#else
		(void) graph;
		return std::nullopt;
#endif
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeScatterUpdateF32(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		const auto plan = MatchCUDANativeScatterUpdateF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}
		const auto ptx = TryCUDANativeScatterUpdateF32PTXFromMLIRNVPTX({
		    .indexType = plan->indexType,
		    .rowSize = plan->rowSize,
		});
		if (!ptx)
		{
			return std::nullopt;
		}

		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeFeature::ScatterUpdateF32);
		payload.target = CUDANativeNVPTXTargetChip();
		payload.binary = CUDANativeTextBytes(*ptx);
		AppendU32(payload.scalarData, plan->dataElementCount);

		const auto blockSize = std::min<std::uint32_t>(plan->dataElementCount, 256);
		const auto gridSize = (plan->dataElementCount + blockSize - 1) / blockSize;
		payload.kernels.push_back({
		    .name = CUDANativeScatterUpdateF32KernelName(plan->indexType),
		    .grid = { .x = gridSize, .y = 1, .z = 1 },
		    .block = { .x = blockSize, .y = 1, .z = 1 },
		    .arguments = {
		        { .kind = CUDANativeArgumentKind::OutputTensor,
		          .index = 0,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->dataElementCount) * sizeof(float) },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->dataInputIndex,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->dataElementCount) * sizeof(float) },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->indicesInputIndex,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->indexElementCount) *
		                      (plan->indexType == DataType::Int32 ? sizeof(std::int32_t) : sizeof(std::int64_t)) },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->updatesInputIndex,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->updateElementCount) * sizeof(float) },
		        { .kind = CUDANativeArgumentKind::Scalar, .index = 0, .byteOffset = 0, .byteSize = sizeof(std::uint32_t) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
#else
		(void) graph;
		return std::nullopt;
#endif
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeMatMulLowPrecision(const Graph& graph)
	{
		const auto plan = MatchCUDANativeMatMulLowPrecision(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		const auto dtype = outputSpecs[0].type.dtype;

		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::LibraryCall;
		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeFeature::MatMulCUBLASLowPrecision);
		payload.target = "cublas";
		AppendU32(payload.scalarData, plan->m);
		AppendU32(payload.scalarData, plan->k);
		AppendU32(payload.scalarData, plan->n);

		const auto elementByteSize = static_cast<std::uint64_t>(ElementByteSize(dtype));
		const auto outputByteSize = static_cast<std::uint64_t>(plan->outputElementCount) * elementByteSize;
		const auto lhsByteSize = static_cast<std::uint64_t>(plan->lhsElementCount) * elementByteSize;
		const auto rhsByteSize = static_cast<std::uint64_t>(plan->rhsElementCount) * elementByteSize;
		payload.kernels.push_back({
		    .name = std::string(CUDANativeMatMulLibraryCallKernelName(dtype)),
		    .grid = { .x = 1, .y = 1, .z = 1 },
		    .block = { .x = 1, .y = 1, .z = 1 },
		    .sharedMemoryBytes = 0,
		    .workspaceBytes = 0,
		    .arguments = {
		        { .kind = CUDANativeArgumentKind::OutputTensor, .index = 0, .byteOffset = 0, .byteSize = outputByteSize },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->lhsInputIndex,
		          .byteOffset = 0,
		          .byteSize = lhsByteSize },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->rhsInputIndex,
		          .byteOffset = 0,
		          .byteSize = rhsByteSize },
		    },
		});

		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeGGMLBlockQuantizedMatMul(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph);
		payload.target = CUDANativeNVPTXTargetChip();

		const auto plan = MatchCUDANativeGGMLBlockQuantizedMatMul(graph, payload);
		if (!plan)
		{
			return std::nullopt;
		}
		payload.featureSet.AddFeature(CUDANativeGGMLBlockMatMulFeature(plan->format));
		if (!payload.constantData.empty())
		{
			payload.featureSet.AddFeature(CUDANativeFeature::ConstantTensor);
		}

		const auto ptx = TryCUDANativeGGMLBlockMatMulF32PTXFromMLIRNVPTX({
		    .format = plan->format,
		    .m = plan->m,
		    .k = plan->k,
		    .n = plan->n,
		});
		if (!ptx)
		{
			return std::nullopt;
		}
		payload.binary = CUDANativeTextBytes(*ptx);
		AppendU32(payload.scalarData, plan->outputElementCount);

		const auto blockSize = std::min<std::uint32_t>(plan->outputElementCount, 256);
		const auto gridSize = (plan->outputElementCount + blockSize - 1) / blockSize;
		payload.kernels.push_back({
		    .name = std::string(CUDANativeGGMLBlockMatMulF32KernelName(plan->format)),
		    .grid = { .x = gridSize, .y = 1, .z = 1 },
		    .block = { .x = blockSize, .y = 1, .z = 1 },
		    .arguments = {
		        { .kind = CUDANativeArgumentKind::OutputTensor,
		          .index = 0,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->outputElementCount) * sizeof(float) },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->lhsInputIndex,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->lhsElementCount) * sizeof(float) },
		        ToCUDANativeArgument(plan->rhsStorage),
		        { .kind = CUDANativeArgumentKind::Scalar, .index = 0, .byteOffset = 0, .byteSize = sizeof(std::uint32_t) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
#else
		(void) graph;
		return std::nullopt;
#endif
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeMatMulBias(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		const auto plan = MatchCUDANativeMatMulBias(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		const auto epiloguePtx =
		    TryCUDANativeMatMulBiasEpiloguePTXFromMLIRNVPTX(CUDANativeMatMulBiasEpilogueCodegenSpec{
		        .dtype = plan->dtype,
		        .outputShape = plan->outputShape,
		        .biasShape = plan->biasShape,
		        .relu = plan->relu,
		    });
		if (!epiloguePtx)
		{
			return std::nullopt;
		}

		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeFeature::MultiKernelLaunch);
		AddCUDANativeMatMulFeatureFlag(payload, plan->dtype);
		AddCUDANativeMatMulBiasFeatureFlags(payload, plan->dtype, plan->relu);
		payload.target = CUDANativeNVPTXTargetChip();
		payload.binary = CUDANativeTextBytes(*epiloguePtx);
		AppendU32(payload.scalarData, plan->m);
		AppendU32(payload.scalarData, plan->k);
		AppendU32(payload.scalarData, plan->n);
		const auto epilogueCountOffset = payload.scalarData.size();
		AppendU32(payload.scalarData, plan->outputElementCount);

		const auto elementByteSize = static_cast<std::uint64_t>(ElementByteSize(plan->dtype));
		const auto outputByteSize = static_cast<std::uint64_t>(plan->outputElementCount) * elementByteSize;
		const auto lhsByteSize = static_cast<std::uint64_t>(plan->lhsElementCount) * elementByteSize;
		const auto rhsByteSize = static_cast<std::uint64_t>(plan->rhsElementCount) * elementByteSize;
		const auto biasByteSize = static_cast<std::uint64_t>(plan->biasElementCount) * elementByteSize;
		payload.kernels.push_back({
		    .name = std::string(CUDANativeMatMulLibraryCallKernelName(plan->dtype)),
		    .grid = { .x = 1, .y = 1, .z = 1 },
		    .block = { .x = 1, .y = 1, .z = 1 },
		    .arguments = {
		        { .kind = CUDANativeArgumentKind::OutputTensor, .index = 0, .byteOffset = 0, .byteSize = outputByteSize },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->lhsInputIndex,
		          .byteOffset = 0,
		          .byteSize = lhsByteSize },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->rhsInputIndex,
		          .byteOffset = 0,
		          .byteSize = rhsByteSize },
		    },
		});

		const auto blockSize = std::min<std::uint32_t>(plan->outputElementCount, 256);
		const auto gridSize = (plan->outputElementCount + blockSize - 1) / blockSize;
		payload.kernels.push_back({
		    .name = CUDANativeMatMulBiasEpilogueKernelName(plan->dtype, plan->relu),
		    .grid = { .x = gridSize, .y = 1, .z = 1 },
		    .block = { .x = blockSize, .y = 1, .z = 1 },
		    .arguments = {
		        { .kind = CUDANativeArgumentKind::OutputTensor, .index = 0, .byteOffset = 0, .byteSize = outputByteSize },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->biasInputIndex,
		          .byteOffset = 0,
		          .byteSize = biasByteSize },
		        { .kind = CUDANativeArgumentKind::Scalar,
		          .index = 0,
		          .byteOffset = epilogueCountOffset,
		          .byteSize = sizeof(std::uint32_t) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{ std::move(rodata), std::move(instructions), std::move(inputSpecs),
			                            std::move(outputSpecs) };
#else
		(void) graph;
		return std::nullopt;
#endif
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeReduceF32(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		const auto plan = MatchCUDANativeReduceF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}
		const auto ptx = TryCUDANativeReduceF32PTXFromMLIRNVPTX(CUDANativeReduceF32CodegenSpec{
		    .op = plan->op,
		    .inputShape = plan->inputShape,
		    .axis = plan->axis,
		});
		if (!ptx)
		{
			return std::nullopt;
		}

		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeFeature::ReduceF32);
		payload.target = CUDANativeNVPTXTargetChip();
		payload.binary = CUDANativeTextBytes(*ptx);
		AppendU32(payload.scalarData, plan->outputElementCount);

		const auto blockSize = std::min<std::uint32_t>(plan->outputElementCount, 256);
		const auto gridSize = (plan->outputElementCount + blockSize - 1) / blockSize;
		payload.kernels.push_back({
		    .name = std::string(CUDANativeReduceF32KernelName(plan->op)),
		    .grid = { .x = gridSize, .y = 1, .z = 1 },
		    .block = { .x = blockSize, .y = 1, .z = 1 },
		    .arguments = {
		        { .kind = CUDANativeArgumentKind::OutputTensor,
		          .index = 0,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->outputElementCount) * sizeof(float) },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->inputIndex,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->inputElementCount) * sizeof(float) },
		        { .kind = CUDANativeArgumentKind::Scalar, .index = 0, .byteOffset = 0, .byteSize = sizeof(std::uint32_t) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{ std::move(rodata), std::move(instructions), std::move(inputSpecs),
			                            std::move(outputSpecs) };
#else
		(void) graph;
		return std::nullopt;
#endif
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeSoftmaxF32(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		const auto plan = MatchCUDANativeSoftmaxF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}
		const auto ptx = TryCUDANativeSoftmaxF32PTXFromMLIRNVPTX({
		    .inputShape = plan->inputShape,
		    .axis = plan->axis,
		});
		if (!ptx)
		{
			return std::nullopt;
		}

		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeFeature::SoftmaxF32);
		payload.target = CUDANativeNVPTXTargetChip();
		payload.binary = CUDANativeTextBytes(*ptx);
		AppendU32(payload.scalarData, plan->elementCount);

		const auto blockSize = std::min<std::uint32_t>(plan->elementCount, 256);
		const auto gridSize = (plan->elementCount + blockSize - 1) / blockSize;
		payload.kernels.push_back({
		    .name = std::string(CUDANativeSoftmaxF32KernelName()),
		    .grid = { .x = gridSize, .y = 1, .z = 1 },
		    .block = { .x = blockSize, .y = 1, .z = 1 },
		    .arguments = {
		        { .kind = CUDANativeArgumentKind::OutputTensor,
		          .index = 0,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->elementCount) * sizeof(float) },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->inputIndex,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->elementCount) * sizeof(float) },
		        { .kind = CUDANativeArgumentKind::Scalar, .index = 0, .byteOffset = 0, .byteSize = sizeof(std::uint32_t) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{ std::move(rodata), std::move(instructions), std::move(inputSpecs),
			                            std::move(outputSpecs) };
#else
		(void) graph;
		return std::nullopt;
#endif
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeGetRowsF32(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.target = CUDANativeNVPTXTargetChip();
		const auto plan = MatchCUDANativeGetRowsF32(graph, payload);
		if (!plan)
		{
			return std::nullopt;
		}
		const auto ptx = TryCUDANativeGetRowsF32PTXFromMLIRNVPTX({
		    .indexType = plan->indexType,
		    .rowSize = plan->rowSize,
		});
		if (!ptx)
		{
			return std::nullopt;
		}

		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeFeature::GetRowsF32);
		if (!payload.constantData.empty())
		{
			payload.featureSet.AddFeature(CUDANativeFeature::ConstantTensor);
		}
		payload.binary = CUDANativeTextBytes(*ptx);
		AppendU32(payload.scalarData, plan->outputElementCount);
		const auto blockSize = std::min<std::uint32_t>(plan->outputElementCount, 256);
		const auto gridSize = (plan->outputElementCount + blockSize - 1) / blockSize;
		payload.kernels.push_back({
		    .name = CUDANativeGetRowsF32KernelName(plan->indexType),
		    .grid = { .x = gridSize, .y = 1, .z = 1 },
		    .block = { .x = blockSize, .y = 1, .z = 1 },
		    .arguments = {
		        { .kind = CUDANativeArgumentKind::OutputTensor,
		          .index = 0,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->outputElementCount) * sizeof(float) },
		        ToCUDANativeArgument(plan->table),
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->indicesInputIndex,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->indexCount) * ElementByteSize(plan->indexType) },
		        { .kind = CUDANativeArgumentKind::Scalar, .index = 0, .byteOffset = 0, .byteSize = sizeof(std::uint32_t) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{ std::move(rodata), std::move(instructions), std::move(inputSpecs),
			                            std::move(outputSpecs) };
#else
		(void) graph;
		return std::nullopt;
#endif
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeRMSNormF32(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.target = CUDANativeNVPTXTargetChip();
		const auto plan = MatchCUDANativeRMSNormF32(graph, payload);
		if (!plan)
		{
			return std::nullopt;
		}
		const auto ptx = TryCUDANativeRMSNormF32PTXFromMLIRNVPTX({
		    .rowSize = plan->rowSize,
		    .epsilon = plan->epsilon,
		    .hasScale = plan->scale.has_value(),
		});
		if (!ptx)
		{
			return std::nullopt;
		}

		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeFeature::RMSNormF32);
		if (!payload.constantData.empty())
		{
			payload.featureSet.AddFeature(CUDANativeFeature::ConstantTensor);
		}
		payload.binary = CUDANativeTextBytes(*ptx);
		AppendU32(payload.scalarData, plan->elementCount);
		const auto blockSize = std::min<std::uint32_t>(plan->elementCount, 256);
		const auto gridSize = (plan->elementCount + blockSize - 1) / blockSize;
		std::vector<CUDANativeArgumentSpec> arguments{
			{ .kind = CUDANativeArgumentKind::OutputTensor,
			  .index = 0,
			  .byteOffset = 0,
			  .byteSize = static_cast<std::uint64_t>(plan->elementCount) * sizeof(float) },
			{ .kind = CUDANativeArgumentKind::InputTensor,
			  .index = plan->inputIndex,
			  .byteOffset = 0,
			  .byteSize = static_cast<std::uint64_t>(plan->elementCount) * sizeof(float) },
		};
		if (plan->scale)
		{
			arguments.push_back(ToCUDANativeArgument(*plan->scale));
		}
		arguments.push_back(
		    { .kind = CUDANativeArgumentKind::Scalar, .index = 0, .byteOffset = 0, .byteSize = sizeof(std::uint32_t) });
		payload.kernels.push_back({
		    .name = std::string(CUDANativeRMSNormF32KernelName(plan->scale.has_value())),
		    .grid = { .x = gridSize, .y = 1, .z = 1 },
		    .block = { .x = blockSize, .y = 1, .z = 1 },
		    .arguments = std::move(arguments),
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{ std::move(rodata), std::move(instructions), std::move(inputSpecs),
			                            std::move(outputSpecs) };
#else
		(void) graph;
		return std::nullopt;
#endif
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeRoPEF32(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		const auto plan = MatchCUDANativeRoPEF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}
		const auto ptx = TryCUDANativeRoPEF32PTXFromMLIRNVPTX({
		    .featureSize = plan->featureSize,
		    .positionOffset = plan->positionOffset,
		    .positionType = plan->positionType,
		});
		if (!ptx)
		{
			return std::nullopt;
		}

		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeFeature::ConstantTensor, CUDANativeFeature::RoPEF32);
		payload.target = CUDANativeNVPTXTargetChip();
		payload.binary = CUDANativeTextBytes(*ptx);
		std::vector<double> frequencyValues(plan->featureSize / 2);
		for (std::uint32_t pair = 0; pair < plan->featureSize / 2; ++pair)
		{
			frequencyValues[pair] =
			    std::pow(plan->base, -2.0 * static_cast<double>(pair) / static_cast<double>(plan->featureSize)) *
			    plan->frequencyScale;
		}
		const Tensor<CPU> frequencyTensor(std::span<const double>(frequencyValues), { plan->featureSize / 2 },
		                                  DataType::Float32);
		const auto polymorphicFrequency = frequencyTensor.CopyToDevice(PolymorphicDevice{ CPU{} });
		const auto frequencyOffset = AppendCUDANativeConstantTensor(payload, polymorphicFrequency);
		AppendU32(payload.scalarData, plan->elementCount);

		std::vector<CUDANativeArgumentSpec> arguments{
			{ .kind = CUDANativeArgumentKind::OutputTensor,
			  .index = 0,
			  .byteOffset = 0,
			  .byteSize = static_cast<std::uint64_t>(plan->elementCount) * sizeof(float) },
			{ .kind = CUDANativeArgumentKind::InputTensor,
			  .index = plan->inputIndex,
			  .byteOffset = 0,
			  .byteSize = static_cast<std::uint64_t>(plan->elementCount) * sizeof(float) },
			{ .kind = CUDANativeArgumentKind::ConstantTensor,
			  .index = 0,
			  .byteOffset = frequencyOffset,
			  .byteSize = static_cast<std::uint64_t>(plan->featureSize / 2) * sizeof(float) },
		};
		if (plan->positionsInputIndex)
		{
			const auto sequenceLength = plan->elementCount / plan->featureSize;
			arguments.push_back(
			    { .kind = CUDANativeArgumentKind::InputTensor,
			      .index = *plan->positionsInputIndex,
			      .byteOffset = 0,
			      .byteSize = static_cast<std::uint64_t>(sequenceLength) * ElementByteSize(*plan->positionType) });
		}
		arguments.push_back(
		    { .kind = CUDANativeArgumentKind::Scalar, .index = 0, .byteOffset = 0, .byteSize = sizeof(std::uint32_t) });
		const auto blockSize = std::min<std::uint32_t>(plan->elementCount, 256);
		payload.kernels.push_back({
		    .name = CUDANativeRoPEF32KernelName(plan->positionType),
		    .grid = { .x = (plan->elementCount + blockSize - 1) / blockSize, .y = 1, .z = 1 },
		    .block = { .x = blockSize, .y = 1, .z = 1 },
		    .arguments = std::move(arguments),
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{ std::move(rodata), std::move(instructions), std::move(inputSpecs),
			                            std::move(outputSpecs) };
#else
		(void) graph;
		return std::nullopt;
#endif
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeConcatF32(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		const auto plan = MatchCUDANativeConcatF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}
		const auto ptx = TryCUDANativeConcatF32PTXFromMLIRNVPTX(CUDANativeConcatF32CodegenSpec{
		    .outputShape = plan->outputShape,
		    .inputShapes = plan->inputShapes,
		    .axis = plan->axis,
		});
		if (!ptx)
		{
			return std::nullopt;
		}

		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeFeature::ConcatF32, CUDANativeFeature::MultiKernelLaunch);
		payload.target = CUDANativeNVPTXTargetChip();
		payload.binary = CUDANativeTextBytes(*ptx);

		for (std::size_t i = 0; i < plan->inputElementCounts.size(); ++i)
		{
			const auto scalarOffset = payload.scalarData.size();
			AppendU32(payload.scalarData, plan->inputElementCounts[i]);
			const auto blockSize = std::min<std::uint32_t>(plan->inputElementCounts[i], 256);
			const auto gridSize = (plan->inputElementCounts[i] + blockSize - 1) / blockSize;
			payload.kernels.push_back({
			    .name = CUDANativeConcatF32KernelName(i),
			    .grid = { .x = gridSize, .y = 1, .z = 1 },
			    .block = { .x = blockSize, .y = 1, .z = 1 },
			    .arguments = {
			        { .kind = CUDANativeArgumentKind::OutputTensor,
			          .index = 0,
			          .byteOffset = 0,
			          .byteSize = static_cast<std::uint64_t>(plan->outputElementCount) * sizeof(float) },
			        { .kind = CUDANativeArgumentKind::InputTensor,
			          .index = plan->inputIndices[i],
			          .byteOffset = 0,
			          .byteSize = static_cast<std::uint64_t>(plan->inputElementCounts[i]) * sizeof(float) },
			        { .kind = CUDANativeArgumentKind::Scalar,
			          .index = 0,
			          .byteOffset = scalarOffset,
			          .byteSize = sizeof(std::uint32_t) },
			    },
			});
		}

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{ std::move(rodata), std::move(instructions), std::move(inputSpecs),
			                            std::move(outputSpecs) };
#else
		(void) graph;
		return std::nullopt;
#endif
	}

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeSliceF32(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		const auto plan = MatchCUDANativeSliceF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}
		const auto ptx = TryCUDANativeSliceF32PTXFromMLIRNVPTX(CUDANativeSliceF32CodegenSpec{
		    .inputShape = plan->inputShape,
		    .outputShape = plan->outputShape,
		    .axis = plan->axis,
		    .start = plan->start,
		});
		if (!ptx)
		{
			return std::nullopt;
		}

		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.featureSet.AddFeature(CUDANativeFeature::StaticShape, CUDANativeFeature::SingleSubgraph,
		                              CUDANativeFeature::SliceF32);
		payload.target = CUDANativeNVPTXTargetChip();
		payload.binary = CUDANativeTextBytes(*ptx);
		AppendU32(payload.scalarData, plan->outputElementCount);

		const auto blockSize = std::min<std::uint32_t>(plan->outputElementCount, 256);
		const auto gridSize = (plan->outputElementCount + blockSize - 1) / blockSize;
		payload.kernels.push_back({
		    .name = std::string(CUDANativeSliceF32KernelName()),
		    .grid = { .x = gridSize, .y = 1, .z = 1 },
		    .block = { .x = blockSize, .y = 1, .z = 1 },
		    .arguments = {
		        { .kind = CUDANativeArgumentKind::OutputTensor,
		          .index = 0,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->outputElementCount) * sizeof(float) },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->inputIndex,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->inputElementCount) * sizeof(float) },
		        { .kind = CUDANativeArgumentKind::Scalar, .index = 0, .byteOffset = 0, .byteSize = sizeof(std::uint32_t) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::CUDANative);
		auto instructions = SerializeCUDANativeInstructionPayload(payload);
		return CUDANativeArtifactParts{ std::move(rodata), std::move(instructions), std::move(inputSpecs),
			                            std::move(outputSpecs) };
#else
		(void) graph;
		return std::nullopt;
#endif
	}

	std::uint64_t TensorByteSize(const Tensor<CUDA>& tensor)
	{
		return static_cast<std::uint64_t>(tensor.NumElements()) * LiteNN::ElementByteSize(tensor.DType());
	}

	class CUDANativeWorkspaceBuffer
	{
	public:
		CUDANativeWorkspaceBuffer(CUDA& device, std::uint64_t byteSize) : device_(&device), byteSize_(byteSize)
		{
			if (byteSize_ == 0)
			{
				return;
			}
			wordCount_ = static_cast<std::size_t>((byteSize_ + sizeof(std::uint32_t) - 1) / sizeof(std::uint32_t));
			data_ = DeviceTraits<CUDA>::Allocate(device, DataType::Int32, wordCount_);
		}

		CUDANativeWorkspaceBuffer(const CUDANativeWorkspaceBuffer&) = delete;
		CUDANativeWorkspaceBuffer& operator=(const CUDANativeWorkspaceBuffer&) = delete;

		~CUDANativeWorkspaceBuffer()
		{
			if (data_ != nullptr)
			{
				DeviceTraits<CUDA>::Deallocate(*device_, data_, DataType::Int32, wordCount_);
			}
		}

		void* Pointer(const CUDANativeArgumentSpec& argument) const
		{
			if (data_ == nullptr)
			{
				throw std::runtime_error("CUDA native workspace argument requires a workspace allocation");
			}
			if (argument.byteOffset > byteSize_ ||
			    (argument.byteSize != 0 && argument.byteSize > byteSize_ - argument.byteOffset))
			{
				throw std::runtime_error("CUDA native workspace argument byte range is out of bounds");
			}
			return static_cast<std::byte*>(data_) + argument.byteOffset;
		}

		std::uint64_t ByteSize() const noexcept
		{
			return byteSize_;
		}

	private:
		CUDA* device_{};
		void* data_{};
		std::uint64_t byteSize_{};
		std::size_t wordCount_{};
	};

	class CUDANativeConstantBuffer
	{
	public:
		CUDANativeConstantBuffer(CUDA& device, std::span<const std::byte> bytes)
		    : device_(&device), byteSize_(bytes.size())
		{
			if (byteSize_ == 0)
			{
				return;
			}
			wordCount_ = (byteSize_ + sizeof(std::uint32_t) - 1) / sizeof(std::uint32_t);
			std::vector<std::uint32_t> padded(wordCount_);
			std::memcpy(padded.data(), bytes.data(), bytes.size());
			data_ = DeviceTraits<CUDA>::Allocate(device, DataType::Int32, wordCount_);
			DeviceTraits<CUDA>::CopyFromCPU(device, DataType::Int32, data_, DataType::Int32, padded.data(), wordCount_);
		}

		CUDANativeConstantBuffer(const CUDANativeConstantBuffer&) = delete;
		CUDANativeConstantBuffer& operator=(const CUDANativeConstantBuffer&) = delete;

		~CUDANativeConstantBuffer()
		{
			if (data_ != nullptr)
			{
				DeviceTraits<CUDA>::Deallocate(*device_, data_, DataType::Int32, wordCount_);
			}
		}

		void* Pointer(const CUDANativeArgumentSpec& argument) const
		{
			if (data_ == nullptr)
			{
				throw std::runtime_error("CUDA native constant argument requires a constant allocation");
			}
			if (argument.byteOffset > byteSize_ ||
			    (argument.byteSize != 0 && argument.byteSize > byteSize_ - argument.byteOffset))
			{
				throw std::runtime_error("CUDA native constant argument byte range is out of bounds");
			}
			return static_cast<std::byte*>(data_) + argument.byteOffset;
		}

		std::uint64_t ByteSize() const noexcept
		{
			return byteSize_;
		}

	private:
		CUDA* device_{};
		void* data_{};
		std::uint64_t byteSize_{};
		std::size_t wordCount_{};
	};

	std::uint64_t CUDANativeWorkspaceByteSize(const CUDANativeInstructionPayload& payload)
	{
		std::uint64_t workspaceBytes = payload.workspaceBytes;
		for (const auto& kernel : payload.kernels)
		{
			workspaceBytes = std::max(workspaceBytes, kernel.workspaceBytes);
		}
		return workspaceBytes;
	}

	bool IsCUDANativeLibraryCallKernel(std::string_view name)
	{
		return name == CUDANativeMatMulLibraryCallKernelName(DataType::Float32) ||
		       name == CUDANativeMatMulLibraryCallKernelName(DataType::Float16) ||
		       name == CUDANativeMatMulLibraryCallKernelName(DataType::BFloat16) ||
		       name == CUDANativeMatMulLibraryCallKernelName(DataType::Float8E4M3) ||
		       name == CUDANativeMatMulLibraryCallKernelName(DataType::Float8E5M2) ||
		       name == CUDANativeMatMulLibraryCallKernelName(DataType::Int8) ||
		       name == CUDANativeMatMulLibraryCallKernelName(DataType::UInt8);
	}

	std::optional<DataType> CUDANativeLibraryCallKernelDataType(std::string_view name)
	{
		for (const auto dtype : { DataType::Float32, DataType::Float16, DataType::BFloat16, DataType::Float8E4M3,
		                          DataType::Float8E5M2, DataType::Int8, DataType::UInt8 })
		{
			if (name == CUDANativeMatMulLibraryCallKernelName(dtype))
			{
				return dtype;
			}
		}
		return std::nullopt;
	}

	CUDAExecutionOptions ToCUDAExecutionOptions(CompiledModuleCUDARunOptions options)
	{
		return CUDAExecutionOptions{ .stream = options.stream,
			                         .synchronize = options.synchronize,
			                         .enableCUBLASLt = options.enableCUBLASLt };
	}

	bool RequestsCUDAGraphReplay(CompiledModuleCUDARunOptions options)
	{
		return options.graphReplay == CUDAGraphReplayMode::Enabled;
	}

	bool CanUseCUDAGraphReplay(CompiledModuleCUDARunOptions options)
	{
		return RequestsCUDAGraphReplay(options) && options.synchronize && options.stream == nullptr;
	}

	void CheckCUDARuntime(cudaError_t status, std::string_view action)
	{
		if (status != cudaSuccess)
		{
			throw std::runtime_error(std::format("{} failed: {}", action, cudaGetErrorString(status)));
		}
	}

	struct CUDAGraphBindingKey
	{
		std::vector<std::uintptr_t> pointers;

		bool operator==(const CUDAGraphBindingKey& other) const noexcept
		{
			return pointers == other.pointers;
		}
	};

	struct CUDAGraphBindingKeyHash
	{
		std::size_t operator()(const CUDAGraphBindingKey& key) const noexcept
		{
			std::size_t seed = key.pointers.size();
			for (const auto value : key.pointers)
			{
				seed ^= value + 0x9e3779b97f4a7c15ull + (seed << 6) + (seed >> 2);
			}
			return seed;
		}
	};

	CUDAGraphBindingKey MakeCUDAGraphBindingKey(std::span<const Tensor<CUDA>> inputs, std::span<Tensor<CUDA>> outputs)
	{
		CUDAGraphBindingKey key;
		key.pointers.reserve(inputs.size() + outputs.size());
		for (const auto& input : inputs)
		{
			key.pointers.push_back(reinterpret_cast<std::uintptr_t>(input.UnsafeRawData()));
		}
		for (auto& output : outputs)
		{
			key.pointers.push_back(reinterpret_cast<std::uintptr_t>(output.UnsafeRawData()));
		}
		return key;
	}

	class CUDAGraphCaptureStream
	{
	public:
		CUDAGraphCaptureStream()
		{
			CheckCUDARuntime(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
			                 "cudaStreamCreateWithFlags for CUDA graph capture");
		}

		CUDAGraphCaptureStream(const CUDAGraphCaptureStream&) = delete;
		CUDAGraphCaptureStream& operator=(const CUDAGraphCaptureStream&) = delete;

		~CUDAGraphCaptureStream()
		{
			if (stream_ != nullptr)
			{
				(void) cudaStreamDestroy(stream_);
			}
		}

		cudaStream_t Get() const noexcept
		{
			return stream_;
		}

	private:
		cudaStream_t stream_{};
	};

	class CUDAGraphExecInstance
	{
	public:
		CUDAGraphExecInstance() = default;
		explicit CUDAGraphExecInstance(cudaGraphExec_t exec) : exec_(exec)
		{
		}

		CUDAGraphExecInstance(const CUDAGraphExecInstance&) = delete;
		CUDAGraphExecInstance& operator=(const CUDAGraphExecInstance&) = delete;

		CUDAGraphExecInstance(CUDAGraphExecInstance&& other) noexcept : exec_(std::exchange(other.exec_, nullptr))
		{
		}

		CUDAGraphExecInstance& operator=(CUDAGraphExecInstance&& other) noexcept
		{
			if (this != &other)
			{
				Reset();
				exec_ = std::exchange(other.exec_, nullptr);
			}
			return *this;
		}

		~CUDAGraphExecInstance()
		{
			Reset();
		}

		void Launch(cudaStream_t stream, bool synchronize) const
		{
			if (exec_ == nullptr)
			{
				throw std::runtime_error("CUDA graph executable is empty");
			}
			CheckCUDARuntime(cudaGraphLaunch(exec_, stream), "cudaGraphLaunch");
			if (synchronize)
			{
				CheckCUDARuntime(cudaStreamSynchronize(stream), "cudaStreamSynchronize after cudaGraphLaunch");
			}
		}

	private:
		void Reset() noexcept
		{
			if (exec_ != nullptr)
			{
				(void) cudaGraphExecDestroy(exec_);
				exec_ = nullptr;
			}
		}

		cudaGraphExec_t exec_{};
	};

	using CUDAGraphReplayCache =
	    std::unordered_map<CUDAGraphBindingKey, CUDAGraphExecInstance, CUDAGraphBindingKeyHash>;

	std::uint32_t ReadScalarU32(const CUDANativeInstructionPayload& payload, const CUDANativeArgumentSpec& argument)
	{
		if (argument.kind != CUDANativeArgumentKind::Scalar || argument.byteSize != sizeof(std::uint32_t) ||
		    argument.byteOffset > payload.scalarData.size() ||
		    argument.byteSize > payload.scalarData.size() - argument.byteOffset)
		{
			throw std::runtime_error("CUDA native scalar argument is not a valid u32");
		}
		std::uint32_t value = 0;
		std::memcpy(&value, payload.scalarData.data() + argument.byteOffset, sizeof(value));
		return value;
	}

	void* TensorArgumentPointer(const CUDANativeArgumentSpec& argument, Tensor<CUDA>& tensor, std::string_view label)
	{
		const auto tensorSize = TensorByteSize(tensor);
		if (argument.byteOffset > tensorSize ||
		    (argument.byteSize != 0 && argument.byteSize > tensorSize - argument.byteOffset))
		{
			throw std::runtime_error(std::format("CUDA native {} argument byte range is out of bounds", label));
		}
		auto* base = static_cast<std::byte*>(tensor.UnsafeRawData());
		return base + argument.byteOffset;
	}

	void* ConstTensorArgumentPointer(const CUDANativeArgumentSpec& argument, const Tensor<CUDA>& tensor,
	                                 std::string_view label)
	{
		const auto tensorSize = TensorByteSize(tensor);
		if (argument.byteOffset > tensorSize ||
		    (argument.byteSize != 0 && argument.byteSize > tensorSize - argument.byteOffset))
		{
			throw std::runtime_error(std::format("CUDA native {} argument byte range is out of bounds", label));
		}
		auto* base = reinterpret_cast<const std::byte*>(tensor.UnsafeRawData());
		return const_cast<std::byte*>(base + argument.byteOffset);
	}

	void* CUDANativeDevicePointer(const CUDANativeArgumentSpec& argument, std::span<const Tensor<CUDA>> inputs,
	                              std::span<Tensor<CUDA>> outputs, CUDANativeWorkspaceBuffer& workspace,
	                              CUDANativeConstantBuffer& constants, std::string_view label)
	{
		switch (argument.kind)
		{
		case CUDANativeArgumentKind::InputTensor:
			if (argument.index >= inputs.size())
			{
				throw std::runtime_error("CUDA native input argument index is out of bounds");
			}
			return ConstTensorArgumentPointer(argument, inputs[argument.index], label);
		case CUDANativeArgumentKind::OutputTensor:
			if (argument.index >= outputs.size())
			{
				throw std::runtime_error("CUDA native output argument index is out of bounds");
			}
			return TensorArgumentPointer(argument, outputs[argument.index], label);
		case CUDANativeArgumentKind::Workspace:
			return workspace.Pointer(argument);
		case CUDANativeArgumentKind::ConstantTensor:
			return constants.Pointer(argument);
		case CUDANativeArgumentKind::Scalar:
			break;
		}
		throw std::runtime_error("CUDA native expected a device pointer argument");
	}

	void RunCUDANativeLibraryCall(CUDA& device, const CUDANativeKernelSpec& kernel,
	                              const CUDANativeInstructionPayload& payload, CUDANativeWorkspaceBuffer& workspace,
	                              CUDANativeConstantBuffer& constants, std::span<const Tensor<CUDA>> inputs,
	                              std::span<Tensor<CUDA>> outputs, CompiledModuleCUDARunOptions options)
	{
		const auto dtype = CUDANativeLibraryCallKernelDataType(kernel.name);
		if (!dtype)
		{
			throw std::runtime_error(std::format("Unsupported CUDA native library call '{}'", kernel.name));
		}
		if (kernel.arguments.size() != 3 && kernel.arguments.size() != 6)
		{
			throw std::runtime_error("CUDA native cuBLAS MatMul expects 3 pointer args and optional m/k/n scalar args");
		}

		if (kernel.arguments.size() == 3)
		{
			const auto& outputArg = kernel.arguments[0];
			const auto& lhsArg = kernel.arguments[1];
			const auto& rhsArg = kernel.arguments[2];
			if (outputArg.kind != CUDANativeArgumentKind::OutputTensor ||
			    lhsArg.kind != CUDANativeArgumentKind::InputTensor ||
			    rhsArg.kind != CUDANativeArgumentKind::InputTensor)
			{
				throw std::runtime_error("CUDA native legacy cuBLAS MatMul expects output, lhs input, rhs input");
			}
			if (outputArg.index >= outputs.size() || lhsArg.index >= inputs.size() || rhsArg.index >= inputs.size())
			{
				throw std::runtime_error("CUDA native cuBLAS MatMul argument index is out of bounds");
			}
			if (outputArg.byteOffset != 0 || lhsArg.byteOffset != 0 || rhsArg.byteOffset != 0)
			{
				throw std::runtime_error("CUDA native legacy cuBLAS MatMul does not support tensor byte offsets");
			}

			auto& output = outputs[outputArg.index];
			const auto& lhs = inputs[lhsArg.index];
			const auto& rhs = inputs[rhsArg.index];
			if (output.DType() != *dtype || lhs.DType() != *dtype || rhs.DType() != *dtype)
			{
				throw std::runtime_error("CUDA native MatMul library call tensor dtypes do not match payload kernel");
			}

			(void) TensorArgumentPointer(outputArg, output, "output");
			(void) ConstTensorArgumentPointer(lhsArg, lhs, "input");
			(void) ConstTensorArgumentPointer(rhsArg, rhs, "input");
			DeviceTraits<CUDA>::DoBinaryOp(device, BinaryOp::MatMul, output.UnsafeRawData(), lhs.DType(), lhs.Shape(),
			                               lhs.UnsafeRawData(), rhs.DType(), rhs.Shape(), rhs.UnsafeRawData(),
			                               ToCUDAExecutionOptions(options));
			return;
		}

		const auto m = static_cast<std::size_t>(ReadScalarU32(payload, kernel.arguments[3]));
		const auto k = static_cast<std::size_t>(ReadScalarU32(payload, kernel.arguments[4]));
		const auto n = static_cast<std::size_t>(ReadScalarU32(payload, kernel.arguments[5]));
		void* outputPtr = CUDANativeDevicePointer(kernel.arguments[0], inputs, outputs, workspace, constants, "output");
		void* lhsPtr = CUDANativeDevicePointer(kernel.arguments[1], inputs, outputs, workspace, constants, "lhs");
		void* rhsPtr = CUDANativeDevicePointer(kernel.arguments[2], inputs, outputs, workspace, constants, "rhs");
		auto outputView = Tensor<CUDA>::UnsafeBorrowed(outputPtr, { m, n }, *dtype, device);
		auto lhsView = Tensor<CUDA>::UnsafeBorrowed(lhsPtr, { m, k }, *dtype, device);
		auto rhsView = Tensor<CUDA>::UnsafeBorrowed(rhsPtr, { k, n }, *dtype, device);
		DeviceTraits<CUDA>::DoBinaryOp(device, BinaryOp::MatMul, outputView.UnsafeRawData(), lhsView.DType(),
		                               lhsView.Shape(), lhsView.UnsafeRawData(), rhsView.DType(), rhsView.Shape(),
		                               rhsView.UnsafeRawData(), ToCUDAExecutionOptions(options));
	}

	void RunCUDANativePayload(CUDA& device, const CUDANativeInstructionPayload& payload, const CUDADriverModule& module,
	                          CUDANativeWorkspaceBuffer& workspace, CUDANativeConstantBuffer& constants,
	                          std::span<const Tensor<CUDA>> inputs, std::span<Tensor<CUDA>> outputs,
	                          CompiledModuleCUDARunOptions options)
	{
		const bool libraryCallPayload = payload.binaryKind == CUDANativeBinaryKind::LibraryCall;
		if (!libraryCallPayload && module.Empty())
		{
			throw std::runtime_error("CUDA native compiled module is empty");
		}

		for (const auto& kernel : payload.kernels)
		{
			if (IsCUDANativeLibraryCallKernel(kernel.name))
			{
				RunCUDANativeLibraryCall(device, kernel, payload, workspace, constants, inputs, outputs, options);
				continue;
			}
			if (libraryCallPayload)
			{
				throw std::runtime_error(std::format("Unsupported CUDA native library call '{}'", kernel.name));
			}

			std::vector<void*> pointerValues;
			std::vector<std::vector<std::byte>> scalarStorage;
			std::vector<void*> argumentPointers;
			pointerValues.reserve(kernel.arguments.size());
			scalarStorage.reserve(kernel.arguments.size());
			argumentPointers.reserve(kernel.arguments.size());

			for (const auto& argument : kernel.arguments)
			{
				switch (argument.kind)
				{
				case CUDANativeArgumentKind::InputTensor:
					if (argument.index >= inputs.size())
					{
						throw std::runtime_error("CUDA native input argument index is out of bounds");
					}
					pointerValues.push_back(ConstTensorArgumentPointer(argument, inputs[argument.index], "input"));
					argumentPointers.push_back(&pointerValues.back());
					break;
				case CUDANativeArgumentKind::OutputTensor:
					if (argument.index >= outputs.size())
					{
						throw std::runtime_error("CUDA native output argument index is out of bounds");
					}
					pointerValues.push_back(TensorArgumentPointer(argument, outputs[argument.index], "output"));
					argumentPointers.push_back(&pointerValues.back());
					break;
				case CUDANativeArgumentKind::Scalar:
					if (argument.byteOffset > payload.scalarData.size() ||
					    argument.byteSize > payload.scalarData.size() - argument.byteOffset)
					{
						throw std::runtime_error("CUDA native scalar argument byte range is out of bounds");
					}
					scalarStorage.emplace_back(
					    payload.scalarData.begin() + static_cast<std::ptrdiff_t>(argument.byteOffset),
					    payload.scalarData.begin() +
					        static_cast<std::ptrdiff_t>(argument.byteOffset + argument.byteSize));
					argumentPointers.push_back(scalarStorage.back().data());
					break;
				case CUDANativeArgumentKind::Workspace:
					pointerValues.push_back(workspace.Pointer(argument));
					argumentPointers.push_back(&pointerValues.back());
					break;
				case CUDANativeArgumentKind::ConstantTensor:
					pointerValues.push_back(constants.Pointer(argument));
					argumentPointers.push_back(&pointerValues.back());
					break;
				}
			}

			module.Launch(kernel.name,
			              {
			                  .grid = { .x = kernel.grid.x, .y = kernel.grid.y, .z = kernel.grid.z },
			                  .block = { .x = kernel.block.x, .y = kernel.block.y, .z = kernel.block.z },
			                  .sharedMemoryBytes = kernel.sharedMemoryBytes,
			                  .stream = options.stream,
			                  .synchronize = options.synchronize,
			              },
			              argumentPointers);
		}
	}

	CUDAGraphExecInstance CaptureCUDANativeGraph(CUDA& device, const CUDANativeInstructionPayload& payload,
	                                             const CUDADriverModule& module, CUDANativeWorkspaceBuffer& workspace,
	                                             CUDANativeConstantBuffer& constants,
	                                             std::span<const Tensor<CUDA>> inputs, std::span<Tensor<CUDA>> outputs)
	{
		CUDAGraphCaptureStream captureStream;
		cudaGraph_t graph{};
		bool capturing = false;

		try
		{
			RunCUDANativePayload(device, payload, module, workspace, constants, inputs, outputs,
			                     CompiledModuleCUDARunOptions{
			                         .stream = captureStream.Get(),
			                         .synchronize = true,
			                         .enableCUBLASLt = false,
			                     });

			CheckCUDARuntime(cudaStreamBeginCapture(captureStream.Get(), cudaStreamCaptureModeThreadLocal),
			                 "cudaStreamBeginCapture");
			capturing = true;

			RunCUDANativePayload(device, payload, module, workspace, constants, inputs, outputs,
			                     CompiledModuleCUDARunOptions{
			                         .stream = captureStream.Get(),
			                         .synchronize = false,
			                         .enableCUBLASLt = false,
			                     });

			capturing = false;
			CheckCUDARuntime(cudaStreamEndCapture(captureStream.Get(), &graph), "cudaStreamEndCapture");
			cudaGraphExec_t exec{};
			const auto instantiateStatus = cudaGraphInstantiate(&exec, graph, 0);
			const auto destroyStatus = cudaGraphDestroy(graph);
			graph = nullptr;
			CheckCUDARuntime(instantiateStatus, "cudaGraphInstantiate");
			CheckCUDARuntime(destroyStatus, "cudaGraphDestroy after instantiate");
			return CUDAGraphExecInstance(exec);
		}
		catch (...)
		{
			if (capturing)
			{
				cudaGraph_t discardedGraph{};
				(void) cudaStreamEndCapture(captureStream.Get(), &discardedGraph);
				if (discardedGraph != nullptr)
				{
					(void) cudaGraphDestroy(discardedGraph);
				}
			}
			if (graph != nullptr)
			{
				(void) cudaGraphDestroy(graph);
			}
			throw;
		}
	}

	void RunCUDANativePayloadWithGraphReplay(CUDAGraphReplayCache& cache, CUDA& device,
	                                         const CUDANativeInstructionPayload& payload,
	                                         const CUDADriverModule& module, CUDANativeWorkspaceBuffer& workspace,
	                                         CUDANativeConstantBuffer& constants, std::span<const Tensor<CUDA>> inputs,
	                                         std::span<Tensor<CUDA>> outputs, CompiledModuleCUDARunOptions options)
	{
		auto key = MakeCUDAGraphBindingKey(inputs, outputs);
		auto it = cache.find(key);
		if (it == cache.end())
		{
			auto instance = CaptureCUDANativeGraph(device, payload, module, workspace, constants, inputs, outputs);
			it = cache.emplace(std::move(key), std::move(instance)).first;
		}
		it->second.Launch(reinterpret_cast<cudaStream_t>(options.stream), options.synchronize);
	}
#endif

#ifdef LITENN_ENABLE_VULKAN
	struct VulkanP0BinaryPlan
	{
		BinaryOp op{ BinaryOp::Add };
		std::uint32_t lhsInputIndex{};
		std::uint32_t rhsInputIndex{};
		DataType dtype{ DataType::Float32 };
		std::uint32_t elementCount{};
	};

	struct VulkanP0BinaryChainOperand
	{
		bool accumulator{};
		std::uint32_t inputIndex{};
	};

	struct VulkanP0BinaryChainKernelPlan
	{
		BinaryOp op{ BinaryOp::Add };
		VulkanP0BinaryChainOperand lhs;
		VulkanP0BinaryChainOperand rhs;
	};

	struct VulkanP0BinaryChainPlan
	{
		std::uint32_t elementCount{};
		std::vector<VulkanP0BinaryChainKernelPlan> kernels;
	};

	enum class VulkanP0BinaryDAGOperandKind
	{
		Input,
		Intermediate,
	};

	struct VulkanP0BinaryDAGOperand
	{
		VulkanP0BinaryDAGOperandKind kind{ VulkanP0BinaryDAGOperandKind::Input };
		std::uint32_t index{};
	};

	struct VulkanP0BinaryDAGKernelPlan
	{
		BinaryOp op{ BinaryOp::Add };
		VulkanP0BinaryDAGOperand lhs;
		VulkanP0BinaryDAGOperand rhs;
	};

	struct VulkanP0BinaryDAGPlan
	{
		std::uint32_t elementCount{};
		std::vector<VulkanP0BinaryDAGKernelPlan> kernels;
		std::uint32_t outputKernelIndex{};
	};

	using VulkanP0ElementwiseDAGOperandKind = VulkanP0BinaryDAGOperandKind;
	using VulkanP0ElementwiseDAGOperand = VulkanP0BinaryDAGOperand;

	struct VulkanP0ElementwiseDAGKernelPlan
	{
		VulkanNativeElementwiseF32KernelKind kind{ VulkanNativeElementwiseF32KernelKind::Binary };
		UnaryOp unaryOp{ UnaryOp::Abs };
		BinaryOp binaryOp{ BinaryOp::Add };
		VulkanP0ElementwiseDAGOperand input;
		VulkanP0ElementwiseDAGOperand lhs;
		VulkanP0ElementwiseDAGOperand rhs;
	};

	struct VulkanP0ElementwiseDAGPlan
	{
		std::uint32_t elementCount{};
		std::vector<VulkanP0ElementwiseDAGKernelPlan> kernels;
		std::uint32_t outputKernelIndex{};
	};

	class VulkanP0WorkspacePlanner
	{
	public:
		std::uint32_t Allocate(std::uint64_t byteSize, std::uint64_t alignment, std::size_t firstKernel,
		                       std::size_t lastKernel)
		{
			if (firstKernel > lastKernel)
			{
				throw std::runtime_error("Vulkan native workspace lifetime is invalid");
			}
			for (std::uint32_t i = 0; i < slots_.size(); ++i)
			{
				auto& slot = slots_[i];
				if (slot.spec.byteSize >= byteSize && slot.spec.alignment >= alignment &&
				    !Overlaps(slot.firstKernel, slot.lastKernel, firstKernel, lastKernel))
				{
					slot.firstKernel = std::min(slot.firstKernel, firstKernel);
					slot.lastKernel = std::max(slot.lastKernel, lastKernel);
					return i;
				}
			}
			if (slots_.size() >= std::numeric_limits<std::uint32_t>::max())
			{
				throw std::runtime_error("Vulkan native workspace tensor count overflows uint32_t");
			}
			const auto index = static_cast<std::uint32_t>(slots_.size());
			slots_.push_back({
			    .spec = {
			        .byteSize = byteSize,
			        .alignment = alignment,
			    },
			    .firstKernel = firstKernel,
			    .lastKernel = lastKernel,
			});
			return index;
		}

		VulkanNativeArgumentSpec Argument(std::uint32_t index, std::uint32_t binding, std::uint64_t byteSize) const
		{
			if (index >= slots_.size())
			{
				throw std::runtime_error("Vulkan native workspace planner argument index is out of bounds");
			}
			return VulkanNativeArgumentSpec{
				.kind = VulkanNativeArgumentKind::WorkspaceTensor,
				.index = index,
				.binding = binding,
				.byteOffset = 0,
				.byteSize = byteSize,
			};
		}

		std::vector<VulkanNativeWorkspaceSpec> TakeWorkspaceTensors()
		{
			std::vector<VulkanNativeWorkspaceSpec> specs;
			specs.reserve(slots_.size());
			for (const auto& slot : slots_)
			{
				specs.push_back(slot.spec);
			}
			return specs;
		}

	private:
		struct Slot
		{
			VulkanNativeWorkspaceSpec spec;
			std::size_t firstKernel{};
			std::size_t lastKernel{};
		};

		static bool Overlaps(std::size_t lhsFirst, std::size_t lhsLast, std::size_t rhsFirst, std::size_t rhsLast)
		{
			return lhsFirst <= rhsLast && rhsFirst <= lhsLast;
		}

		std::vector<Slot> slots_;
	};

	struct VulkanP0ScheduleWorkspaceAllocation
	{
		std::vector<std::uint32_t> workspaceByKernel;
		std::vector<VulkanNativeWorkspaceSpec> workspaceTensors;
	};

	template <typename VisitKernelIntermediateUses>
	std::vector<std::size_t> BuildVulkanP0ScheduleLastUses(std::size_t kernelCount,
	                                                       VisitKernelIntermediateUses&& visitKernelIntermediateUses)
	{
		std::vector<std::size_t> lastUse(kernelCount, 0);
		for (std::size_t kernelIndex = 0; kernelIndex < kernelCount; ++kernelIndex)
		{
			lastUse[kernelIndex] = kernelIndex;
			const auto markIntermediateUse = [&](std::uint32_t producerKernelIndex) {
				if (producerKernelIndex >= lastUse.size())
				{
					throw std::runtime_error("Vulkan native schedule intermediate operand index is out of bounds");
				}
				lastUse[producerKernelIndex] = std::max(lastUse[producerKernelIndex], kernelIndex);
			};
			visitKernelIntermediateUses(kernelIndex, markIntermediateUse);
		}
		return lastUse;
	}

	VulkanP0ScheduleWorkspaceAllocation AllocateVulkanP0ScheduleWorkspaces(std::size_t kernelCount,
	                                                                       std::uint32_t outputKernelIndex,
	                                                                       std::span<const std::size_t> lastUse,
	                                                                       std::uint64_t byteSize,
	                                                                       std::uint64_t alignment)
	{
		if (outputKernelIndex >= kernelCount || lastUse.size() != kernelCount)
		{
			throw std::runtime_error("Vulkan native schedule workspace plan is inconsistent");
		}

		VulkanP0WorkspacePlanner workspacePlanner;
		VulkanP0ScheduleWorkspaceAllocation allocation{
			.workspaceByKernel = std::vector<std::uint32_t>(kernelCount, 0),
		};
		for (std::size_t kernelIndex = 0; kernelIndex < kernelCount; ++kernelIndex)
		{
			if (kernelIndex == outputKernelIndex)
			{
				continue;
			}
			allocation.workspaceByKernel[kernelIndex] =
			    workspacePlanner.Allocate(byteSize, alignment, kernelIndex, lastUse[kernelIndex]);
		}
		allocation.workspaceTensors = workspacePlanner.TakeWorkspaceTensors();
		return allocation;
	}

	VulkanNativeArgumentSpec VulkanP0ScheduleWorkspaceArgument(std::uint32_t producerKernelIndex, std::uint32_t binding,
	                                                           std::uint64_t byteSize,
	                                                           std::span<const std::uint32_t> workspaceByKernel)
	{
		if (producerKernelIndex >= workspaceByKernel.size())
		{
			throw std::runtime_error("Vulkan native schedule workspace operand index is out of bounds");
		}
		return VulkanNativeArgumentSpec{
			.kind = VulkanNativeArgumentKind::WorkspaceTensor,
			.index = workspaceByKernel[producerKernelIndex],
			.binding = binding,
			.byteOffset = 0,
			.byteSize = byteSize,
		};
	}

	VulkanNativeArgumentSpec VulkanP0ScheduleOutputArgument(std::size_t kernelIndex, std::uint32_t outputKernelIndex,
	                                                        std::uint32_t binding, std::uint64_t byteSize,
	                                                        std::span<const std::uint32_t> workspaceByKernel)
	{
		if (kernelIndex == outputKernelIndex)
		{
			return VulkanNativeArgumentSpec{
				.kind = VulkanNativeArgumentKind::OutputTensor,
				.index = 0,
				.binding = binding,
				.byteOffset = 0,
				.byteSize = byteSize,
			};
		}
		if (kernelIndex > std::numeric_limits<std::uint32_t>::max())
		{
			throw std::runtime_error("Vulkan native schedule kernel index overflows uint32_t");
		}
		return VulkanP0ScheduleWorkspaceArgument(static_cast<std::uint32_t>(kernelIndex), binding, byteSize,
		                                         workspaceByKernel);
	}

	struct VulkanP0UnaryPlan
	{
		UnaryOp op{ UnaryOp::Negate };
		std::uint32_t inputIndex{};
		DataType dtype{ DataType::Float32 };
		std::uint32_t elementCount{};
	};

	struct VulkanP0CastPlan
	{
		std::uint32_t inputIndex{};
		std::uint32_t elementCount{};
		DataType srcType{ DataType::Float32 };
		DataType dstType{ DataType::Int32 };
	};

	struct VulkanP0ReducePlan
	{
		ReduceOp op{ ReduceOp::Sum };
		std::uint32_t inputIndex{};
		std::uint32_t inputElementCount{};
		std::uint32_t outputElementCount{};
		std::size_t axis{};
		std::vector<std::size_t> inputShape;
		std::vector<std::size_t> outputShape;
	};

	struct VulkanP0SoftmaxPlan
	{
		std::uint32_t inputIndex{};
		std::uint32_t elementCount{};
		std::size_t axis{};
		std::vector<std::size_t> inputShape;
	};

	struct VulkanP0Pool2DPlan
	{
		PoolMode mode{ PoolMode::Max };
		std::uint32_t inputIndex{};
		std::uint32_t inputElementCount{};
		std::uint32_t outputElementCount{};
		std::vector<std::size_t> inputShape;
		std::vector<std::size_t> outputShape;
		std::vector<std::size_t> kernelShape;
		std::vector<std::size_t> strides;
		std::vector<std::size_t> lowPads;
		std::vector<std::size_t> highPads;
		bool countIncludePad{};
	};

	struct VulkanP0TensorRef
	{
		VulkanNativeArgumentKind argumentKind{ VulkanNativeArgumentKind::InputTensor };
		std::uint32_t argumentIndex{};
		DataType dtype{ DataType::Float32 };
		std::vector<std::size_t> shape;
		std::uint32_t elementCount{};
	};

	struct VulkanP0Conv2DPlan
	{
		VulkanP0TensorRef input;
		VulkanP0TensorRef weight;
		std::optional<VulkanP0TensorRef> bias;
		std::uint32_t outputElementCount{};
		std::vector<std::size_t> inputShape;
		std::vector<std::size_t> weightShape;
		std::vector<std::size_t> outputShape;
		std::vector<std::size_t> strides;
		std::vector<std::size_t> dilations;
		std::vector<std::size_t> lowPads;
		std::vector<std::size_t> highPads;
		std::size_t groupCount{ 1 };
	};

	struct VulkanP0ConvTranspose2DPlan
	{
		VulkanP0TensorRef input;
		VulkanP0TensorRef weight;
		std::optional<VulkanP0TensorRef> bias;
		std::uint32_t outputElementCount{};
		std::vector<std::size_t> inputShape;
		std::vector<std::size_t> weightShape;
		std::vector<std::size_t> outputShape;
		std::vector<std::size_t> strides;
		std::vector<std::size_t> dilations;
		std::vector<std::size_t> lowPads;
		std::vector<std::size_t> highPads;
		std::vector<std::size_t> outputPads;
		std::size_t groupCount{ 1 };
	};

	struct VulkanP0UpsamplePlan
	{
		std::uint32_t inputIndex{};
		std::uint32_t inputElementCount{};
		std::uint32_t outputElementCount{};
		std::vector<std::size_t> inputShape;
		std::vector<std::size_t> outputShape;
	};

	struct VulkanP0SlicePlan
	{
		std::uint32_t inputIndex{};
		std::uint32_t inputElementCount{};
		std::uint32_t outputElementCount{};
		std::vector<std::size_t> inputShape;
		std::vector<std::size_t> outputShape;
		std::size_t axis{};
		std::size_t start{};
		std::size_t length{};
	};

	struct VulkanP0ConcatPlan
	{
		std::uint32_t lhsIndex{};
		std::uint32_t rhsIndex{};
		std::uint32_t lhsElementCount{};
		std::uint32_t rhsElementCount{};
		std::uint32_t outputElementCount{};
		std::vector<std::size_t> lhsShape;
		std::vector<std::size_t> rhsShape;
		std::vector<std::size_t> outputShape;
		std::size_t axis{};
	};

	struct VulkanP0NormalizationPlan
	{
		NormalizationMode mode{ NormalizationMode::LayerNorm };
		std::uint32_t inputIndex{};
		std::optional<VulkanP0TensorRef> scale;
		std::optional<VulkanP0TensorRef> bias;
		std::uint32_t elementCount{};
		std::size_t axis{};
		std::size_t groupCount{ 1 };
		double epsilon{ 1e-5 };
		std::vector<std::size_t> inputShape;
	};

	struct VulkanP0MatMulPlan
	{
		std::uint32_t lhsInputIndex{};
		std::uint32_t rhsInputIndex{};
		std::uint32_t m{};
		std::uint32_t k{};
		std::uint32_t n{};
		std::uint32_t outputElementCount{};
	};

	struct VulkanP0MatMulBiasPlan
	{
		VulkanP0TensorRef lhs;
		VulkanP0TensorRef rhs;
		VulkanP0TensorRef bias;
		std::uint32_t m{};
		std::uint32_t k{};
		std::uint32_t n{};
		std::uint32_t biasRows{};
		std::uint32_t outputElementCount{};
		bool relu{};
	};

	struct VulkanP0LinearChainKernelPlan
	{
		VulkanP0TensorRef lhs;
		VulkanP0TensorRef rhs;
		VulkanP0TensorRef bias;
		VulkanP0TensorRef output;
		std::uint32_t m{};
		std::uint32_t k{};
		std::uint32_t n{};
		std::uint32_t biasRows{};
		std::uint32_t outputElementCount{};
		bool relu{};
	};

	struct VulkanP0LinearChainPlan
	{
		std::vector<VulkanP0LinearChainKernelPlan> kernels;
		std::vector<VulkanNativeWorkspaceSpec> workspaceTensors;
	};

	struct VulkanP0ExternalTensorBuilder
	{
		std::vector<std::byte> constants;
		std::vector<std::byte> weights;
		std::vector<CompiledModuleExternalTensorInfo> externalTensorInfos;
		std::unordered_map<std::size_t, std::uint32_t> variableExternalIds;
		std::unordered_map<NodeId, std::uint32_t> constantExternalIds;
	};

	struct VulkanP0ArtifactParts
	{
		std::vector<std::byte> rodata;
		std::vector<std::byte> instructions;
		std::vector<std::byte> constants;
		std::vector<std::byte> weights;
		std::vector<CompiledModuleExternalTensorInfo> externalTensorInfos;
		std::vector<CompiledTensorSpec> inputSpecs;
		std::vector<CompiledTensorSpec> outputSpecs;
	};

	std::optional<std::uint32_t> VulkanP0ShapeNumElementsU32(std::span<const std::size_t> shape);
	std::optional<VulkanP0LinearChainPlan> MatchVulkanP0LinearChainF32(const Graph& graph,
	                                                                   VulkanP0ExternalTensorBuilder* externalBuilder);

	std::optional<std::uint32_t> GetVulkanP0ParamIndex(const Subgraph& subgraph, NodeOutput output)
	{
		if (output.port != 0 || output.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto* param = std::get_if<ParamRefNode>(&subgraph.GetNodeEntry(output.node).node);
		if (!param || param->paramIndex >= subgraph.Params().size() ||
		    param->paramIndex > std::numeric_limits<std::uint32_t>::max())
		{
			return std::nullopt;
		}
		return static_cast<std::uint32_t>(param->paramIndex);
	}

	std::optional<std::uint32_t> AppendVulkanP0ExternalTensor(VulkanP0ExternalTensorBuilder& builder, std::string name,
	                                                          std::string_view regionName,
	                                                          const Tensor<PolymorphicDevice>& tensor,
	                                                          const OutputInfo& output)
	{
		if (builder.externalTensorInfos.size() > std::numeric_limits<std::uint32_t>::max())
		{
			return std::nullopt;
		}

		constexpr std::uint64_t kAlignment = 64;
		auto& regionBytes = regionName == kWeightsRegionName ? builder.weights : builder.constants;
		const auto offset = AppendTensorPayloadBytes(regionBytes, tensor, output.dtype, output.shape, kAlignment);
		if (!offset)
		{
			return std::nullopt;
		}

		const auto byteSize = TensorByteSizeForShape(output.dtype, output.shape);
		builder.externalTensorInfos.push_back(MakeExternalTensorInfo(
		    std::move(name), regionName, output.dtype, regionBytes, output.shape, *offset, byteSize, kAlignment));
		return static_cast<std::uint32_t>(builder.externalTensorInfos.size() - 1);
	}

	bool VulkanP0ExternalTensorInfoMatches(const CompiledModuleExternalTensorInfo& info, const OutputInfo& output)
	{
		return info.type.dtype == output.dtype && info.type.StaticShape() == output.shape;
	}

	std::optional<VulkanP0TensorRef> GetVulkanP0TensorRef(const Graph& graph, const Subgraph& subgraph,
	                                                      NodeOutput output, VulkanP0ExternalTensorBuilder* builder)
	{
		if (output.port != 0 || output.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& entry = subgraph.GetNodeEntry(output.node);
		if (entry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}
		const auto& info = entry.outputInfos[0];
		const auto elementCount = VulkanP0ShapeNumElementsU32(info.shape);
		if (!elementCount)
		{
			return std::nullopt;
		}

		if (const auto* param = std::get_if<ParamRefNode>(&entry.node))
		{
			if (param->paramIndex >= subgraph.Params().size() ||
			    param->paramIndex > std::numeric_limits<std::uint32_t>::max())
			{
				return std::nullopt;
			}
			return VulkanP0TensorRef{
				.argumentKind = VulkanNativeArgumentKind::InputTensor,
				.argumentIndex = static_cast<std::uint32_t>(param->paramIndex),
				.dtype = info.dtype,
				.shape = info.shape,
				.elementCount = *elementCount,
			};
		}

		if (const auto* variable = std::get_if<VariableRefNode>(&entry.node))
		{
			if (variable->variableIndex >= graph.VariableCount())
			{
				return std::nullopt;
			}
			std::uint32_t externalId = 0;
			if (builder != nullptr)
			{
				auto [it, inserted] = builder->variableExternalIds.emplace(variable->variableIndex, 0);
				if (inserted)
				{
					auto name = graph.VariableName(variable->variableIndex);
					if (name.empty())
					{
						name = std::format("variable{}", variable->variableIndex);
					}
					const auto appended =
					    AppendVulkanP0ExternalTensor(*builder, std::move(name), kWeightsRegionName,
					                                 graph.GetVariable(variable->variableIndex)->Data(), info);
					if (!appended)
					{
						builder->variableExternalIds.erase(it);
						return std::nullopt;
					}
					it->second = *appended;
				}
				externalId = it->second;
				if (!VulkanP0ExternalTensorInfoMatches(builder->externalTensorInfos[externalId], info))
				{
					return std::nullopt;
				}
			}
			return VulkanP0TensorRef{
				.argumentKind = VulkanNativeArgumentKind::ExternalTensor,
				.argumentIndex = externalId,
				.dtype = info.dtype,
				.shape = info.shape,
				.elementCount = *elementCount,
			};
		}

		if (const auto* constant = std::get_if<ConstantNode>(&entry.node))
		{
			std::uint32_t externalId = 0;
			if (builder != nullptr)
			{
				auto [it, inserted] = builder->constantExternalIds.emplace(output.node, 0);
				if (inserted)
				{
					const auto appended = AppendVulkanP0ExternalTensor(
					    *builder, std::format("constant_{}", output.node), kConstantsRegionName, constant->value, info);
					if (!appended)
					{
						builder->constantExternalIds.erase(it);
						return std::nullopt;
					}
					it->second = *appended;
				}
				externalId = it->second;
				if (!VulkanP0ExternalTensorInfoMatches(builder->externalTensorInfos[externalId], info))
				{
					return std::nullopt;
				}
			}
			return VulkanP0TensorRef{
				.argumentKind = VulkanNativeArgumentKind::ExternalTensor,
				.argumentIndex = externalId,
				.dtype = info.dtype,
				.shape = info.shape,
				.elementCount = *elementCount,
			};
		}

		return std::nullopt;
	}

	VulkanNativeArgumentSpec VulkanP0TensorArgument(const VulkanP0TensorRef& ref, std::uint32_t binding)
	{
		return VulkanNativeArgumentSpec{
			.kind = ref.argumentKind,
			.index = ref.argumentIndex,
			.binding = binding,
			.byteOffset = 0,
			.byteSize = TensorByteSizeForShape(ref.dtype, ref.shape),
		};
	}

	bool IsVulkanP0SingleForwardGraph(const Graph& graph)
	{
		return graph.Forward() < graph.SubgraphCount() && !graph.Backward().has_value() &&
		       graph.ActivationSlotCount() == 0 && graph.TapeSlotCount() == 0;
	}

	std::optional<std::uint32_t> VulkanP0ShapeNumElementsU32(std::span<const std::size_t> shape)
	{
		std::uint64_t count = 1;
		for (const auto dim : shape)
		{
			if (dim == 0)
			{
				return std::nullopt;
			}
			count *= static_cast<std::uint64_t>(dim);
			if (count > std::numeric_limits<std::uint32_t>::max())
			{
				return std::nullopt;
			}
		}
		return static_cast<std::uint32_t>(count);
	}

	std::uint32_t VulkanP0ElementwiseGroupCount(std::uint32_t elementCount)
	{
		return (elementCount / kVulkanNativeElementwiseWorkgroupSize) +
		       (elementCount % kVulkanNativeElementwiseWorkgroupSize == 0 ? 0u : 1u);
	}

	std::uint32_t VulkanP0MatMulGroupCount(std::uint32_t elementCount)
	{
		return (elementCount / kVulkanNativeMatMulWorkgroupSize) +
		       (elementCount % kVulkanNativeMatMulWorkgroupSize == 0 ? 0u : 1u);
	}

	VulkanNativeKernelRequirements VulkanP0KernelRequirements(std::uint32_t localSize)
	{
		return VulkanNativeKernelRequirements{
			.descriptorAbiVersion = 1,
			.localSize = { .x = localSize, .y = 1, .z = 1 },
		};
	}

	void AddVulkanP0DTypeDeviceRequirements(VulkanNativeDeviceRequirementSet& requirements, DataType dtype)
	{
		switch (dtype)
		{
		case DataType::Float16:
			requirements.AddRequirement(VulkanNativeDeviceRequirement::ShaderFloat16);
			requirements.AddRequirement(VulkanNativeDeviceRequirement::StorageBuffer16BitAccess);
			break;
		case DataType::Int8:
		case DataType::UInt8:
			requirements.AddRequirement(VulkanNativeDeviceRequirement::ShaderInt8);
			requirements.AddRequirement(VulkanNativeDeviceRequirement::StorageBuffer8BitAccess);
			break;
		default:
			break;
		}
	}

	VulkanNativeKernelRequirements VulkanP0CastKernelRequirements(DataType srcType, DataType dstType)
	{
		auto requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize);
		AddVulkanP0DTypeDeviceRequirements(requirements.deviceRequirements, srcType);
		AddVulkanP0DTypeDeviceRequirements(requirements.deviceRequirements, dstType);
		return requirements;
	}

	VulkanNativeSupportReport VulkanNativeSupported(std::string capability)
	{
		return VulkanNativeSupportReport{ .supported = true, .capability = std::move(capability) };
	}

	VulkanNativeSupportReport VulkanNativeUnsupported(std::string reason)
	{
		return VulkanNativeSupportReport{ .supported = false, .reason = std::move(reason) };
	}

	std::string VulkanNativeOpName(UnaryOp op)
	{
		return std::string(EnumToString<EnumToStringStyle::Unqualified>(op));
	}

	std::string VulkanNativeOpName(BinaryOp op)
	{
		return std::string(EnumToString<EnumToStringStyle::Unqualified>(op));
	}

	std::string VulkanNativeOpName(ReduceOp op)
	{
		return std::string(EnumToString<EnumToStringStyle::Unqualified>(op));
	}

	std::string VulkanNativeOpName(NormalizationMode mode)
	{
		return std::string(EnumToString<EnumToStringStyle::Unqualified>(mode));
	}

	std::string VulkanNativeOpName(PoolMode mode)
	{
		return std::string(EnumToString<EnumToStringStyle::Unqualified>(mode));
	}

	std::string_view VulkanNativeShortDTypeName(DataType dtype)
	{
		switch (dtype)
		{
		case DataType::Float32:
			return "f32";
		case DataType::Float16:
			return "f16";
		case DataType::Int8:
			return "int8";
		case DataType::UInt8:
			return "uint8";
		default:
			return DataTypeName(dtype);
		}
	}

	std::vector<std::size_t> VulkanP0ReduceOutputShape(std::span<const std::size_t> inputShape, std::size_t axis)
	{
		std::vector<std::size_t> outputShape;
		if (axis >= inputShape.size())
		{
			return outputShape;
		}
		outputShape.reserve(inputShape.size() - 1);
		for (std::size_t i = 0; i < inputShape.size(); ++i)
		{
			if (i != axis)
			{
				outputShape.push_back(inputShape[i]);
			}
		}
		return outputShape;
	}

	bool VulkanP0SupportsNormalizationAffineShape(std::span<const std::size_t> inputShape, std::size_t axis,
	                                              std::span<const std::size_t> affineShape)
	{
		if (axis >= inputShape.size())
		{
			return false;
		}
		const auto axisSize = inputShape[axis];
		return (affineShape.size() == 1 && affineShape[0] == axisSize) ||
		       (affineShape.size() == 2 && affineShape[0] == 1 && affineShape[1] == axisSize);
	}

	std::optional<std::size_t> VulkanP0GroupNormGroupedVolume(std::span<const std::size_t> inputShape)
	{
		if (inputShape.empty() || inputShape.size() > 4)
		{
			return std::nullopt;
		}
		std::uint64_t groupedVolume = 1;
		for (std::size_t dim = 0; dim < std::min<std::size_t>(inputShape.size(), 3); ++dim)
		{
			if (inputShape[dim] == 0)
			{
				return std::nullopt;
			}
			groupedVolume *= static_cast<std::uint64_t>(inputShape[dim]);
			if (groupedVolume > std::numeric_limits<std::size_t>::max())
			{
				return std::nullopt;
			}
		}
		return static_cast<std::size_t>(groupedVolume);
	}

	bool VulkanP0SupportsGroupNormAffineShape(std::span<const std::size_t> inputShape,
	                                          std::span<const std::size_t> affineShape)
	{
		const auto groupedVolume = VulkanP0GroupNormGroupedVolume(inputShape);
		if (!groupedVolume)
		{
			return false;
		}
		return (affineShape.size() == 1 && affineShape[0] == *groupedVolume) ||
		       (affineShape.size() == 2 && affineShape[0] == 1 && affineShape[1] == *groupedVolume);
	}

	VulkanNativeSupportReport DiagnoseVulkanP0SingleForwardShape(std::span<const std::size_t> shape,
	                                                             std::string_view label)
	{
		const auto elementCount = VulkanP0ShapeNumElementsU32(shape);
		if (!elementCount)
		{
			return VulkanNativeUnsupported(std::format(
			    "{} shape {} must be non-empty, non-zero, and contain at most uint32_t elements for Vulkan native P0",
			    label, Validation::ShapeToString(shape)));
		}
		return VulkanNativeSupported("");
	}

	template <typename ResolveParamOperand>
	std::optional<VulkanP0BinaryChainPlan>
	MatchVulkanP0SameShapeBinaryF32ChainInSubgraph(const Subgraph& subgraph, NodeOutput result,
	                                               std::span<const std::size_t> chainShape, std::uint32_t elementCount,
	                                               ResolveParamOperand&& resolveParamOperand)
	{
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1 || resultEntry.outputInfos[0].dtype != DataType::Float32)
		{
			return std::nullopt;
		}
		if (!std::ranges::equal(resultEntry.outputInfos[0].shape, chainShape))
		{
			return std::nullopt;
		}

		std::vector<bool> visited(subgraph.NodeCount(), false);
		VulkanP0BinaryChainPlan plan{ .elementCount = elementCount };

		const auto isBinaryNodeOutput = [&](NodeOutput output) {
			return output.port == 0 && output.node < subgraph.NodeCount() &&
			       std::holds_alternative<BinaryOpNode>(subgraph.GetNodeEntry(output.node).node);
		};

		const auto collect = [&](auto&& self, NodeOutput output) -> bool {
			if (output.port != 0 || output.node >= subgraph.NodeCount())
			{
				return false;
			}
			const auto& entry = subgraph.GetNodeEntry(output.node);
			if (entry.outputInfos.size() != 1 || entry.outputInfos[0].dtype != DataType::Float32 ||
			    !std::ranges::equal(entry.outputInfos[0].shape, chainShape))
			{
				return false;
			}
			const auto* binary = std::get_if<BinaryOpNode>(&entry.node);
			if (!binary || !VulkanNativeSupportsSameShapeBinaryF32(binary->op))
			{
				return false;
			}

			const bool lhsIsAccumulator = isBinaryNodeOutput(binary->lhs);
			const bool rhsIsAccumulator = isBinaryNodeOutput(binary->rhs);
			if (lhsIsAccumulator && rhsIsAccumulator)
			{
				return false;
			}

			VulkanP0BinaryChainKernelPlan kernel{ .op = binary->op };
			if (lhsIsAccumulator)
			{
				if (!self(self, binary->lhs))
				{
					return false;
				}
				const auto rhs = resolveParamOperand(binary->rhs);
				if (!rhs)
				{
					return false;
				}
				kernel.lhs = { .accumulator = true };
				kernel.rhs = *rhs;
			}
			else if (rhsIsAccumulator)
			{
				if (!self(self, binary->rhs))
				{
					return false;
				}
				const auto lhs = resolveParamOperand(binary->lhs);
				if (!lhs)
				{
					return false;
				}
				kernel.lhs = *lhs;
				kernel.rhs = { .accumulator = true };
			}
			else
			{
				const auto lhs = resolveParamOperand(binary->lhs);
				const auto rhs = resolveParamOperand(binary->rhs);
				if (!lhs || !rhs)
				{
					return false;
				}
				kernel.lhs = *lhs;
				kernel.rhs = *rhs;
			}

			visited[output.node] = true;
			plan.kernels.push_back(kernel);
			return true;
		};

		if (!collect(collect, result) || plan.kernels.size() < 2)
		{
			return std::nullopt;
		}
		for (std::size_t nodeIndex = 0; nodeIndex < subgraph.NodeCount(); ++nodeIndex)
		{
			const auto& entry = subgraph.GetNodeEntry(nodeIndex);
			if (std::holds_alternative<ParamRefNode>(entry.node))
			{
				continue;
			}
			if (!visited[nodeIndex])
			{
				return std::nullopt;
			}
		}
		return plan;
	}

	std::optional<VulkanP0BinaryChainPlan> MatchVulkanP0SameShapeBinaryF32Chain(const Graph& graph)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Results().size() != 1)
		{
			return std::nullopt;
		}
		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1 || resultEntry.outputInfos[0].dtype != DataType::Float32)
		{
			return std::nullopt;
		}
		const auto& chainShape = resultEntry.outputInfos[0].shape;
		const auto elementCount = VulkanP0ShapeNumElementsU32(chainShape);
		if (!elementCount)
		{
			return std::nullopt;
		}

		if (const auto* fused = std::get_if<FusedOpNode>(&resultEntry.node);
		    fused && fused->pattern == FusionPattern::ElementWiseChain)
		{
			if (fused->body >= graph.SubgraphCount())
			{
				return std::nullopt;
			}
			const auto& body = graph.GetSubgraph(fused->body);
			if (body.Results().size() != 1 || body.Params().size() != fused->args.size())
			{
				return std::nullopt;
			}

			const auto resolveFusedBodyParam = [&](NodeOutput output) -> std::optional<VulkanP0BinaryChainOperand> {
				const auto bodyParamIndex = GetVulkanP0ParamIndex(body, output);
				if (!bodyParamIndex || *bodyParamIndex >= body.Params().size() || *bodyParamIndex >= fused->args.size())
				{
					return std::nullopt;
				}
				const auto& bodyParam = body.Params()[*bodyParamIndex];
				if (bodyParam.dtype != DataType::Float32 || !std::ranges::equal(bodyParam.shape, chainShape))
				{
					return std::nullopt;
				}
				const auto parentInputIndex = GetVulkanP0ParamIndex(subgraph, fused->args[*bodyParamIndex]);
				if (!parentInputIndex)
				{
					return std::nullopt;
				}
				const auto& parentParam = subgraph.Params()[*parentInputIndex];
				if (parentParam.dtype != DataType::Float32 || !std::ranges::equal(parentParam.shape, chainShape))
				{
					return std::nullopt;
				}
				return VulkanP0BinaryChainOperand{ .inputIndex = *parentInputIndex };
			};
			return MatchVulkanP0SameShapeBinaryF32ChainInSubgraph(body, body.Results()[0], chainShape, *elementCount,
			                                                      resolveFusedBodyParam);
		}

		const auto resolveParentParam = [&](NodeOutput output) -> std::optional<VulkanP0BinaryChainOperand> {
			const auto inputIndex = GetVulkanP0ParamIndex(subgraph, output);
			if (!inputIndex)
			{
				return std::nullopt;
			}
			const auto& param = subgraph.Params()[*inputIndex];
			if (param.dtype != DataType::Float32 || !std::ranges::equal(param.shape, chainShape))
			{
				return std::nullopt;
			}
			return VulkanP0BinaryChainOperand{ .inputIndex = *inputIndex };
		};
		return MatchVulkanP0SameShapeBinaryF32ChainInSubgraph(subgraph, result, chainShape, *elementCount,
		                                                      resolveParentParam);
	}

	std::optional<VulkanP0BinaryDAGPlan> MatchVulkanP0SameShapeBinaryF32DAG(const Graph& graph)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Results().size() != 1)
		{
			return std::nullopt;
		}
		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1 || resultEntry.outputInfos[0].dtype != DataType::Float32)
		{
			return std::nullopt;
		}
		const auto& shape = resultEntry.outputInfos[0].shape;
		const auto elementCount = VulkanP0ShapeNumElementsU32(shape);
		if (!elementCount)
		{
			return std::nullopt;
		}

		VulkanP0BinaryDAGPlan plan{ .elementCount = *elementCount };
		std::vector<bool> visited(subgraph.NodeCount(), false);
		std::unordered_map<NodeId, std::uint32_t> kernelIndexByNode;

		const auto collect = [&](auto&& self, NodeOutput output) -> std::optional<VulkanP0BinaryDAGOperand> {
			const auto inputIndex = GetVulkanP0ParamIndex(subgraph, output);
			if (inputIndex)
			{
				const auto& param = subgraph.Params()[*inputIndex];
				if (param.dtype != DataType::Float32 || param.shape != shape)
				{
					return std::nullopt;
				}
				return VulkanP0BinaryDAGOperand{ .kind = VulkanP0BinaryDAGOperandKind::Input, .index = *inputIndex };
			}
			if (output.port != 0 || output.node >= subgraph.NodeCount())
			{
				return std::nullopt;
			}
			if (const auto found = kernelIndexByNode.find(output.node); found != kernelIndexByNode.end())
			{
				return VulkanP0BinaryDAGOperand{ .kind = VulkanP0BinaryDAGOperandKind::Intermediate,
					                             .index = found->second };
			}

			const auto& entry = subgraph.GetNodeEntry(output.node);
			if (entry.outputInfos.size() != 1 || entry.outputInfos[0].dtype != DataType::Float32 ||
			    entry.outputInfos[0].shape != shape)
			{
				return std::nullopt;
			}
			if (const auto* fused = std::get_if<FusedOpNode>(&entry.node);
			    fused && fused->pattern == FusionPattern::ElementWiseChain)
			{
				if (fused->body >= graph.SubgraphCount())
				{
					return std::nullopt;
				}
				const auto& body = graph.GetSubgraph(fused->body);
				if (body.Results().size() != 1 || body.Params().size() != fused->args.size())
				{
					return std::nullopt;
				}

				std::vector<bool> bodyVisited(body.NodeCount(), false);
				std::unordered_map<NodeId, std::uint32_t> bodyKernelIndexByNode;
				const auto collectBody = [&](auto&& bodySelf,
				                             NodeOutput bodyOutput) -> std::optional<VulkanP0BinaryDAGOperand> {
					const auto bodyParamIndex = GetVulkanP0ParamIndex(body, bodyOutput);
					if (bodyParamIndex)
					{
						if (*bodyParamIndex >= body.Params().size() || *bodyParamIndex >= fused->args.size())
						{
							return std::nullopt;
						}
						const auto& bodyParam = body.Params()[*bodyParamIndex];
						if (bodyParam.dtype != DataType::Float32 || bodyParam.shape != shape)
						{
							return std::nullopt;
						}
						return self(self, fused->args[*bodyParamIndex]);
					}
					if (bodyOutput.port != 0 || bodyOutput.node >= body.NodeCount())
					{
						return std::nullopt;
					}
					if (const auto found = bodyKernelIndexByNode.find(bodyOutput.node);
					    found != bodyKernelIndexByNode.end())
					{
						return VulkanP0BinaryDAGOperand{ .kind = VulkanP0BinaryDAGOperandKind::Intermediate,
							                             .index = found->second };
					}
					const auto& bodyEntry = body.GetNodeEntry(bodyOutput.node);
					if (bodyEntry.outputInfos.size() != 1 || bodyEntry.outputInfos[0].dtype != DataType::Float32 ||
					    bodyEntry.outputInfos[0].shape != shape)
					{
						return std::nullopt;
					}
					const auto* bodyBinary = std::get_if<BinaryOpNode>(&bodyEntry.node);
					if (!bodyBinary || !VulkanNativeSupportsSameShapeBinaryF32(bodyBinary->op))
					{
						return std::nullopt;
					}
					const auto lhs = bodySelf(bodySelf, bodyBinary->lhs);
					const auto rhs = bodySelf(bodySelf, bodyBinary->rhs);
					if (!lhs || !rhs)
					{
						return std::nullopt;
					}
					if (plan.kernels.size() > std::numeric_limits<std::uint32_t>::max())
					{
						return std::nullopt;
					}
					const auto kernelIndex = static_cast<std::uint32_t>(plan.kernels.size());
					bodyVisited[bodyOutput.node] = true;
					bodyKernelIndexByNode.emplace(bodyOutput.node, kernelIndex);
					plan.kernels.push_back(
					    VulkanP0BinaryDAGKernelPlan{ .op = bodyBinary->op, .lhs = *lhs, .rhs = *rhs });
					return VulkanP0BinaryDAGOperand{ .kind = VulkanP0BinaryDAGOperandKind::Intermediate,
						                             .index = kernelIndex };
				};

				const auto fusedOperand = collectBody(collectBody, body.Results()[0]);
				if (!fusedOperand || fusedOperand->kind != VulkanP0BinaryDAGOperandKind::Intermediate)
				{
					return std::nullopt;
				}
				for (std::size_t bodyNodeIndex = 0; bodyNodeIndex < body.NodeCount(); ++bodyNodeIndex)
				{
					const auto& bodyEntry = body.GetNodeEntry(bodyNodeIndex);
					if (std::holds_alternative<ParamRefNode>(bodyEntry.node))
					{
						continue;
					}
					if (!bodyVisited[bodyNodeIndex])
					{
						return std::nullopt;
					}
				}
				visited[output.node] = true;
				kernelIndexByNode.emplace(output.node, fusedOperand->index);
				return fusedOperand;
			}
			const auto* binary = std::get_if<BinaryOpNode>(&entry.node);
			if (!binary || !VulkanNativeSupportsSameShapeBinaryF32(binary->op))
			{
				return std::nullopt;
			}
			const auto lhs = self(self, binary->lhs);
			const auto rhs = self(self, binary->rhs);
			if (!lhs || !rhs)
			{
				return std::nullopt;
			}
			if (plan.kernels.size() > std::numeric_limits<std::uint32_t>::max())
			{
				return std::nullopt;
			}
			const auto kernelIndex = static_cast<std::uint32_t>(plan.kernels.size());
			visited[output.node] = true;
			kernelIndexByNode.emplace(output.node, kernelIndex);
			plan.kernels.push_back(VulkanP0BinaryDAGKernelPlan{ .op = binary->op, .lhs = *lhs, .rhs = *rhs });
			return VulkanP0BinaryDAGOperand{ .kind = VulkanP0BinaryDAGOperandKind::Intermediate, .index = kernelIndex };
		};

		const auto outputOperand = collect(collect, result);
		if (!outputOperand || outputOperand->kind != VulkanP0BinaryDAGOperandKind::Intermediate ||
		    plan.kernels.size() < 2)
		{
			return std::nullopt;
		}
		plan.outputKernelIndex = outputOperand->index;
		for (std::size_t nodeIndex = 0; nodeIndex < subgraph.NodeCount(); ++nodeIndex)
		{
			const auto& entry = subgraph.GetNodeEntry(nodeIndex);
			if (std::holds_alternative<ParamRefNode>(entry.node))
			{
				continue;
			}
			if (!visited[nodeIndex])
			{
				return std::nullopt;
			}
		}
		return plan;
	}

	std::optional<VulkanP0ElementwiseDAGPlan> MatchVulkanP0SameShapeElementwiseF32DAG(const Graph& graph)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Results().size() != 1)
		{
			return std::nullopt;
		}
		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1 || resultEntry.outputInfos[0].dtype != DataType::Float32)
		{
			return std::nullopt;
		}
		const auto& shape = resultEntry.outputInfos[0].shape;
		const auto elementCount = VulkanP0ShapeNumElementsU32(shape);
		if (!elementCount)
		{
			return std::nullopt;
		}

		VulkanP0ElementwiseDAGPlan plan{ .elementCount = *elementCount };
		std::vector<bool> visited(subgraph.NodeCount(), false);
		std::unordered_map<NodeId, std::uint32_t> kernelIndexByNode;
		bool hasUnary = false;

		const auto collect = [&](auto&& self, NodeOutput output) -> std::optional<VulkanP0ElementwiseDAGOperand> {
			const auto inputIndex = GetVulkanP0ParamIndex(subgraph, output);
			if (inputIndex)
			{
				const auto& param = subgraph.Params()[*inputIndex];
				if (param.dtype != DataType::Float32 || param.shape != shape)
				{
					return std::nullopt;
				}
				return VulkanP0ElementwiseDAGOperand{ .kind = VulkanP0ElementwiseDAGOperandKind::Input,
					                                  .index = *inputIndex };
			}
			if (output.port != 0 || output.node >= subgraph.NodeCount())
			{
				return std::nullopt;
			}
			if (const auto found = kernelIndexByNode.find(output.node); found != kernelIndexByNode.end())
			{
				return VulkanP0ElementwiseDAGOperand{ .kind = VulkanP0ElementwiseDAGOperandKind::Intermediate,
					                                  .index = found->second };
			}

			const auto& entry = subgraph.GetNodeEntry(output.node);
			if (entry.outputInfos.size() != 1 || entry.outputInfos[0].dtype != DataType::Float32 ||
			    entry.outputInfos[0].shape != shape)
			{
				return std::nullopt;
			}
			if (const auto* fused = std::get_if<FusedOpNode>(&entry.node);
			    fused && fused->pattern == FusionPattern::ElementWiseChain)
			{
				if (fused->body >= graph.SubgraphCount())
				{
					return std::nullopt;
				}
				const auto& body = graph.GetSubgraph(fused->body);
				if (body.Results().size() != 1 || body.Params().size() != fused->args.size())
				{
					return std::nullopt;
				}

				std::vector<bool> bodyVisited(body.NodeCount(), false);
				std::unordered_map<NodeId, std::uint32_t> bodyKernelIndexByNode;
				const auto collectBody = [&](auto&& bodySelf,
				                             NodeOutput bodyOutput) -> std::optional<VulkanP0ElementwiseDAGOperand> {
					const auto bodyParamIndex = GetVulkanP0ParamIndex(body, bodyOutput);
					if (bodyParamIndex)
					{
						if (*bodyParamIndex >= body.Params().size() || *bodyParamIndex >= fused->args.size())
						{
							return std::nullopt;
						}
						const auto& bodyParam = body.Params()[*bodyParamIndex];
						if (bodyParam.dtype != DataType::Float32 || bodyParam.shape != shape)
						{
							return std::nullopt;
						}
						return self(self, fused->args[*bodyParamIndex]);
					}
					if (bodyOutput.port != 0 || bodyOutput.node >= body.NodeCount())
					{
						return std::nullopt;
					}
					if (const auto found = bodyKernelIndexByNode.find(bodyOutput.node);
					    found != bodyKernelIndexByNode.end())
					{
						return VulkanP0ElementwiseDAGOperand{ .kind = VulkanP0ElementwiseDAGOperandKind::Intermediate,
							                                  .index = found->second };
					}
					const auto& bodyEntry = body.GetNodeEntry(bodyOutput.node);
					if (bodyEntry.outputInfos.size() != 1 || bodyEntry.outputInfos[0].dtype != DataType::Float32 ||
					    bodyEntry.outputInfos[0].shape != shape)
					{
						return std::nullopt;
					}
					if (const auto* bodyUnary = std::get_if<UnaryOpNode>(&bodyEntry.node))
					{
						if (!VulkanNativeSupportsSameShapeUnaryF32(bodyUnary->op))
						{
							return std::nullopt;
						}
						const auto input = bodySelf(bodySelf, bodyUnary->input);
						if (!input)
						{
							return std::nullopt;
						}
						if (plan.kernels.size() > std::numeric_limits<std::uint32_t>::max())
						{
							return std::nullopt;
						}
						const auto kernelIndex = static_cast<std::uint32_t>(plan.kernels.size());
						bodyVisited[bodyOutput.node] = true;
						bodyKernelIndexByNode.emplace(bodyOutput.node, kernelIndex);
						hasUnary = true;
						plan.kernels.push_back(VulkanP0ElementwiseDAGKernelPlan{
						    .kind = VulkanNativeElementwiseF32KernelKind::Unary,
						    .unaryOp = bodyUnary->op,
						    .input = *input,
						});
						return VulkanP0ElementwiseDAGOperand{ .kind = VulkanP0ElementwiseDAGOperandKind::Intermediate,
							                                  .index = kernelIndex };
					}
					const auto* bodyBinary = std::get_if<BinaryOpNode>(&bodyEntry.node);
					if (!bodyBinary || !VulkanNativeSupportsSameShapeBinaryF32(bodyBinary->op))
					{
						return std::nullopt;
					}
					const auto lhs = bodySelf(bodySelf, bodyBinary->lhs);
					const auto rhs = bodySelf(bodySelf, bodyBinary->rhs);
					if (!lhs || !rhs)
					{
						return std::nullopt;
					}
					if (plan.kernels.size() > std::numeric_limits<std::uint32_t>::max())
					{
						return std::nullopt;
					}
					const auto kernelIndex = static_cast<std::uint32_t>(plan.kernels.size());
					bodyVisited[bodyOutput.node] = true;
					bodyKernelIndexByNode.emplace(bodyOutput.node, kernelIndex);
					plan.kernels.push_back(VulkanP0ElementwiseDAGKernelPlan{
					    .kind = VulkanNativeElementwiseF32KernelKind::Binary,
					    .binaryOp = bodyBinary->op,
					    .lhs = *lhs,
					    .rhs = *rhs,
					});
					return VulkanP0ElementwiseDAGOperand{ .kind = VulkanP0ElementwiseDAGOperandKind::Intermediate,
						                                  .index = kernelIndex };
				};

				const auto fusedOperand = collectBody(collectBody, body.Results()[0]);
				if (!fusedOperand || fusedOperand->kind != VulkanP0ElementwiseDAGOperandKind::Intermediate)
				{
					return std::nullopt;
				}
				for (std::size_t bodyNodeIndex = 0; bodyNodeIndex < body.NodeCount(); ++bodyNodeIndex)
				{
					const auto& bodyEntry = body.GetNodeEntry(bodyNodeIndex);
					if (std::holds_alternative<ParamRefNode>(bodyEntry.node))
					{
						continue;
					}
					if (!bodyVisited[bodyNodeIndex])
					{
						return std::nullopt;
					}
				}
				visited[output.node] = true;
				kernelIndexByNode.emplace(output.node, fusedOperand->index);
				return fusedOperand;
			}
			if (const auto* unary = std::get_if<UnaryOpNode>(&entry.node))
			{
				if (!VulkanNativeSupportsSameShapeUnaryF32(unary->op))
				{
					return std::nullopt;
				}
				const auto input = self(self, unary->input);
				if (!input)
				{
					return std::nullopt;
				}
				if (plan.kernels.size() > std::numeric_limits<std::uint32_t>::max())
				{
					return std::nullopt;
				}
				const auto kernelIndex = static_cast<std::uint32_t>(plan.kernels.size());
				visited[output.node] = true;
				kernelIndexByNode.emplace(output.node, kernelIndex);
				hasUnary = true;
				plan.kernels.push_back(VulkanP0ElementwiseDAGKernelPlan{
				    .kind = VulkanNativeElementwiseF32KernelKind::Unary,
				    .unaryOp = unary->op,
				    .input = *input,
				});
				return VulkanP0ElementwiseDAGOperand{ .kind = VulkanP0ElementwiseDAGOperandKind::Intermediate,
					                                  .index = kernelIndex };
			}
			const auto* binary = std::get_if<BinaryOpNode>(&entry.node);
			if (!binary || !VulkanNativeSupportsSameShapeBinaryF32(binary->op))
			{
				return std::nullopt;
			}
			const auto lhs = self(self, binary->lhs);
			const auto rhs = self(self, binary->rhs);
			if (!lhs || !rhs)
			{
				return std::nullopt;
			}
			if (plan.kernels.size() > std::numeric_limits<std::uint32_t>::max())
			{
				return std::nullopt;
			}
			const auto kernelIndex = static_cast<std::uint32_t>(plan.kernels.size());
			visited[output.node] = true;
			kernelIndexByNode.emplace(output.node, kernelIndex);
			plan.kernels.push_back(VulkanP0ElementwiseDAGKernelPlan{
			    .kind = VulkanNativeElementwiseF32KernelKind::Binary,
			    .binaryOp = binary->op,
			    .lhs = *lhs,
			    .rhs = *rhs,
			});
			return VulkanP0ElementwiseDAGOperand{ .kind = VulkanP0ElementwiseDAGOperandKind::Intermediate,
				                                  .index = kernelIndex };
		};

		const auto outputOperand = collect(collect, result);
		if (!outputOperand || outputOperand->kind != VulkanP0ElementwiseDAGOperandKind::Intermediate ||
		    plan.kernels.size() < 2 || !hasUnary)
		{
			return std::nullopt;
		}
		plan.outputKernelIndex = outputOperand->index;
		for (std::size_t nodeIndex = 0; nodeIndex < subgraph.NodeCount(); ++nodeIndex)
		{
			const auto& entry = subgraph.GetNodeEntry(nodeIndex);
			if (std::holds_alternative<ParamRefNode>(entry.node))
			{
				continue;
			}
			if (!visited[nodeIndex])
			{
				return std::nullopt;
			}
		}
		return plan;
	}

	VulkanNativeSupportReport DiagnoseVulkanNativeSupport(const Graph& graph)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return VulkanNativeUnsupported(
			    std::format("Vulkan native currently requires a forward-only graph with no activations or tape slots; "
			                "got subgraphs={}, forward={}, backward={}, variables={}, activationSlots={}, tapeSlots={}",
			                graph.SubgraphCount(), graph.Forward(), graph.Backward().has_value(), graph.VariableCount(),
			                graph.ActivationSlotCount(), graph.TapeSlotCount()));
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Results().size() != 1)
		{
			return VulkanNativeUnsupported(std::format(
			    "Vulkan native currently supports exactly one graph result; got {}", subgraph.Results().size()));
		}
		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return VulkanNativeUnsupported("Vulkan native result must reference output port 0 of an existing node");
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return VulkanNativeUnsupported(std::format("Vulkan native result node must have exactly one output; got {}",
			                                           resultEntry.outputInfos.size()));
		}
		const auto& output = resultEntry.outputInfos[0];

		if (const auto chain = MatchVulkanP0LinearChainF32(graph, nullptr))
		{
			return VulkanNativeSupported(std::format("f32 linear chain ({} kernels)", chain->kernels.size()));
		}

		if (const auto* fused = std::get_if<FusedOpNode>(&resultEntry.node);
		    fused != nullptr &&
		    (fused->pattern == FusionPattern::MatMulBiasAdd || fused->pattern == FusionPattern::MatMulBiasAddReLU))
		{
			if (fused->args.size() < 3)
			{
				return VulkanNativeUnsupported("Vulkan native MatMulBias fused node must have at least three args");
			}
			const auto lhs = GetVulkanP0TensorRef(graph, subgraph, fused->args[0], nullptr);
			const auto rhs = GetVulkanP0TensorRef(graph, subgraph, fused->args[1], nullptr);
			const auto bias = GetVulkanP0TensorRef(graph, subgraph, fused->args[2], nullptr);
			if (!lhs || !rhs || !bias)
			{
				return VulkanNativeUnsupported(
				    "Vulkan native MatMulBias inputs must be graph parameters, variables, or constants");
			}
			if (lhs->dtype != DataType::Float32 || rhs->dtype != DataType::Float32 ||
			    bias->dtype != DataType::Float32 || output.dtype != DataType::Float32)
			{
				return VulkanNativeUnsupported(std::format(
				    "Vulkan native MatMulBias requires Float32 lhs/rhs/bias/output, got lhs={} rhs={} bias={} "
				    "output={}",
				    DataTypeName(lhs->dtype), DataTypeName(rhs->dtype), DataTypeName(bias->dtype),
				    DataTypeName(output.dtype)));
			}
			if (lhs->shape.size() != 2 || rhs->shape.size() != 2 || bias->shape.size() != 2 || output.shape.size() != 2)
			{
				return VulkanNativeUnsupported(std::format(
				    "Vulkan native MatMulBias requires rank-2 lhs/rhs/bias/output, got lhs={} rhs={} bias={} "
				    "output={}",
				    Validation::ShapeToString(lhs->shape), Validation::ShapeToString(rhs->shape),
				    Validation::ShapeToString(bias->shape), Validation::ShapeToString(output.shape)));
			}
			if (lhs->shape[1] != rhs->shape[0] || output.shape[0] != lhs->shape[0] ||
			    output.shape[1] != rhs->shape[1] || bias->shape[1] != output.shape[1] ||
			    (bias->shape[0] != 1 && bias->shape[0] != output.shape[0]))
			{
				return VulkanNativeUnsupported(std::format(
				    "Vulkan native MatMulBias shape mismatch, expected [M,K] @ [K,N] + [1|M,N] -> [M,N], got "
				    "lhs={} rhs={} bias={} output={}",
				    Validation::ShapeToString(lhs->shape), Validation::ShapeToString(rhs->shape),
				    Validation::ShapeToString(bias->shape), Validation::ShapeToString(output.shape)));
			}
			return VulkanNativeSupported(fused->pattern == FusionPattern::MatMulBiasAddReLU ? "f32 matmul bias relu"
			                                                                                : "f32 matmul bias add");
		}

		if (const auto chain = MatchVulkanP0SameShapeBinaryF32Chain(graph))
		{
			return VulkanNativeSupported(
			    std::format("same-shape f32 binary chain ({} kernels)", chain->kernels.size()));
		}
		if (const auto dag = MatchVulkanP0SameShapeBinaryF32DAG(graph))
		{
			return VulkanNativeSupported(std::format("same-shape f32 binary DAG ({} kernels)", dag->kernels.size()));
		}
		if (const auto dag = MatchVulkanP0SameShapeElementwiseF32DAG(graph))
		{
			return VulkanNativeSupported(
			    std::format("same-shape f32 elementwise DAG ({} kernels)", dag->kernels.size()));
		}

		if (const auto* conv = std::get_if<Conv2DNode>(&resultEntry.node))
		{
			const auto input = GetVulkanP0TensorRef(graph, subgraph, conv->input, nullptr);
			const auto weight = GetVulkanP0TensorRef(graph, subgraph, conv->weight, nullptr);
			std::optional<VulkanP0TensorRef> bias;
			if (conv->bias)
			{
				bias = GetVulkanP0TensorRef(graph, subgraph, *conv->bias, nullptr);
				if (!bias)
				{
					return VulkanNativeUnsupported("Vulkan native Conv2D bias tensor is not a supported tensor ref");
				}
			}
			if (!input || !weight)
			{
				return VulkanNativeUnsupported("Vulkan native Conv2D input/weight must be supported tensor refs");
			}
			if (input->dtype != DataType::Float32 || weight->dtype != DataType::Float32 ||
			    output.dtype != DataType::Float32 || (bias && bias->dtype != DataType::Float32))
			{
				return VulkanNativeUnsupported(std::format(
				    "Vulkan native Conv2D requires Float32 input/weight/bias/output, got input={} weight={} "
				    "bias={} output={}",
				    DataTypeName(input->dtype), DataTypeName(weight->dtype), bias ? DataTypeName(bias->dtype) : "none",
				    DataTypeName(output.dtype)));
			}
			if (bias && !(bias->shape.size() == 1 || bias->shape.size() == 4))
			{
				return VulkanNativeUnsupported(std::format("Vulkan native Conv2D bias shape is unsupported: {}",
				                                           Validation::ShapeToString(bias->shape)));
			}
			if (bias && ((bias->shape.size() == 1 && bias->shape[0] != output.shape[1]) ||
			             (bias->shape.size() == 4 && (bias->shape[0] != 1 || bias->shape[1] != output.shape[1] ||
			                                          bias->shape[2] != 1 || bias->shape[3] != 1))))
			{
				return VulkanNativeUnsupported(std::format(
				    "Vulkan native Conv2D bias must be [outChannels] or [1,outChannels,1,1], got bias={} output={}",
				    Validation::ShapeToString(bias->shape), Validation::ShapeToString(output.shape)));
			}
			if (!VulkanNativeSupportsConv2DF32(input->shape, weight->shape, output.shape, conv->strides,
			                                   conv->dilations, conv->lowPads, conv->highPads, conv->groupCount))
			{
				return VulkanNativeUnsupported(std::format(
				    "Vulkan native Conv2D requires static rank-4 f32 tensors, got input={} weight={} output={} "
				    "strides={} dilations={} lowPads={} highPads={} groupCount={}",
				    Validation::ShapeToString(input->shape), Validation::ShapeToString(weight->shape),
				    Validation::ShapeToString(output.shape), Validation::ShapeToString(conv->strides),
				    Validation::ShapeToString(conv->dilations), Validation::ShapeToString(conv->lowPads),
				    Validation::ShapeToString(conv->highPads), conv->groupCount));
			}
			return VulkanNativeSupported(
			    std::format("f32 Conv2D groupCount={} bias={}", conv->groupCount, bias.has_value()));
		}

		if (const auto* convT = std::get_if<ConvTranspose2DNode>(&resultEntry.node))
		{
			const auto input = GetVulkanP0TensorRef(graph, subgraph, convT->input, nullptr);
			const auto weight = GetVulkanP0TensorRef(graph, subgraph, convT->weight, nullptr);
			std::optional<VulkanP0TensorRef> bias;
			if (convT->bias)
			{
				bias = GetVulkanP0TensorRef(graph, subgraph, *convT->bias, nullptr);
				if (!bias)
				{
					return VulkanNativeUnsupported(
					    "Vulkan native ConvTranspose2D bias tensor is not a supported tensor ref");
				}
			}
			if (!input || !weight)
			{
				return VulkanNativeUnsupported(
				    "Vulkan native ConvTranspose2D input/weight must be supported tensor refs");
			}
			if (input->dtype != DataType::Float32 || weight->dtype != DataType::Float32 ||
			    output.dtype != DataType::Float32 || (bias && bias->dtype != DataType::Float32))
			{
				return VulkanNativeUnsupported(std::format(
				    "Vulkan native ConvTranspose2D requires Float32 input/weight/bias/output, got input={} weight={} "
				    "bias={} output={}",
				    DataTypeName(input->dtype), DataTypeName(weight->dtype), bias ? DataTypeName(bias->dtype) : "none",
				    DataTypeName(output.dtype)));
			}
			if (bias)
			{
				const auto biasOk = (bias->shape.size() == 1 && bias->shape[0] == output.shape[1]) ||
				                    (bias->shape.size() == 4 && bias->shape[0] == 1 &&
				                     bias->shape[1] == output.shape[1] && bias->shape[2] == 1 && bias->shape[3] == 1);
				if (!biasOk)
				{
					return VulkanNativeUnsupported(std::format(
					    "Vulkan native ConvTranspose2D bias must be [outChannels] or [1,outChannels,1,1], got "
					    "bias={} output={}",
					    Validation::ShapeToString(bias->shape), Validation::ShapeToString(output.shape)));
				}
			}
			if (!VulkanNativeSupportsConvTranspose2DF32(input->shape, weight->shape, output.shape, convT->strides,
			                                            convT->dilations, convT->lowPads, convT->highPads,
			                                            convT->outputPads, convT->groupCount))
			{
				return VulkanNativeUnsupported(
				    std::format("Vulkan native ConvTranspose2D requires static rank-4 f32 tensors, got input={} "
				                "weight={} output={} "
				                "strides={} dilations={} lowPads={} highPads={} outputPads={} groupCount={}",
				                Validation::ShapeToString(input->shape), Validation::ShapeToString(weight->shape),
				                Validation::ShapeToString(output.shape), Validation::ShapeToString(convT->strides),
				                Validation::ShapeToString(convT->dilations), Validation::ShapeToString(convT->lowPads),
				                Validation::ShapeToString(convT->highPads),
				                Validation::ShapeToString(convT->outputPads), convT->groupCount));
			}
			return VulkanNativeSupported(
			    std::format("f32 ConvTranspose2D groupCount={} bias={}", convT->groupCount, bias.has_value()));
		}

		if (subgraph.Params().size() == 1 && subgraph.NodeCount() == 2)
		{
			if (const auto* unary = std::get_if<UnaryOpNode>(&resultEntry.node))
			{
				if (!VulkanNativeSupportsSameShapeUnaryF32(unary->op))
				{
					return VulkanNativeUnsupported(
					    std::format("unsupported unary op {} for Vulkan native same-shape unary slice",
					                VulkanNativeOpName(unary->op)));
				}
				const auto inputIndex = GetVulkanP0ParamIndex(subgraph, unary->input);
				if (!inputIndex)
				{
					return VulkanNativeUnsupported("Vulkan native unary input must be a direct graph parameter");
				}
				const auto& input = subgraph.Params()[*inputIndex];
				if (input.dtype != output.dtype)
				{
					return VulkanNativeUnsupported(
					    std::format("Vulkan native unary slice requires matching input/output dtypes, got {} -> {}",
					                DataTypeName(input.dtype), DataTypeName(output.dtype)));
				}
				if (!VulkanNativeSupportsSameShapeUnary(input.dtype, unary->op))
				{
					return VulkanNativeUnsupported(std::format(
					    "Vulkan native unary slice requires Float32/Float16 or Int8 Negate/Abs input/output, got {}",
					    DataTypeName(input.dtype)));
				}
				if (input.shape != output.shape)
				{
					return VulkanNativeUnsupported(
					    std::format("Vulkan native unary slice requires identical input/output shapes, got {} -> {}",
					                Validation::ShapeToString(input.shape), Validation::ShapeToString(output.shape)));
				}
				if (auto shapeReport = DiagnoseVulkanP0SingleForwardShape(output.shape, "unary output");
				    !shapeReport.supported)
				{
					return shapeReport;
				}
				return VulkanNativeSupported(std::format(
				    "same-shape {} unary {}", VulkanNativeShortDTypeName(output.dtype), VulkanNativeOpName(unary->op)));
			}

			if (const auto* cast = std::get_if<CastNode>(&resultEntry.node))
			{
				const auto inputIndex = GetVulkanP0ParamIndex(subgraph, cast->input);
				if (!inputIndex)
				{
					return VulkanNativeUnsupported("Vulkan native cast input must be a direct graph parameter");
				}
				const auto& input = subgraph.Params()[*inputIndex];
				if (output.dtype != cast->targetType)
				{
					return VulkanNativeUnsupported(
					    std::format("Vulkan native cast output dtype must match target type, got output={} target={}",
					                DataTypeName(output.dtype), DataTypeName(cast->targetType)));
				}
				if (input.shape != output.shape)
				{
					return VulkanNativeUnsupported(
					    std::format("Vulkan native cast slice requires identical input/output shapes, got {} -> {}",
					                Validation::ShapeToString(input.shape), Validation::ShapeToString(output.shape)));
				}
				if (!VulkanNativeSupportsSameShapeCast(input.dtype, output.dtype))
				{
					return VulkanNativeUnsupported(
					    std::format("unsupported cast {} -> {} for Vulkan native same-shape cast slice",
					                DataTypeName(input.dtype), DataTypeName(output.dtype)));
				}
				if (auto shapeReport = DiagnoseVulkanP0SingleForwardShape(output.shape, "cast output");
				    !shapeReport.supported)
				{
					return shapeReport;
				}
				return VulkanNativeSupported(
				    std::format("same-shape cast {} -> {}", DataTypeName(input.dtype), DataTypeName(output.dtype)));
			}

			if (const auto* reduce = std::get_if<ReduceOpNode>(&resultEntry.node))
			{
				const auto inputIndex = GetVulkanP0ParamIndex(subgraph, reduce->input);
				if (!inputIndex)
				{
					return VulkanNativeUnsupported("Vulkan native reduce input must be a direct graph parameter");
				}
				const auto& input = subgraph.Params()[*inputIndex];
				if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32)
				{
					return VulkanNativeUnsupported(
					    std::format("Vulkan native reduce slice requires Float32 input/output, got {} -> {}",
					                DataTypeName(input.dtype), DataTypeName(output.dtype)));
				}
				const auto expectedShape = VulkanP0ReduceOutputShape(input.shape, reduce->axis);
				if (reduce->axis >= input.shape.size() || expectedShape != output.shape)
				{
					return VulkanNativeUnsupported(std::format(
					    "Vulkan native reduce shape mismatch, got input={} axis={} output={}",
					    Validation::ShapeToString(input.shape), reduce->axis, Validation::ShapeToString(output.shape)));
				}
				if (!VulkanNativeSupportsReduceF32(reduce->op, input.shape, reduce->axis))
				{
					return VulkanNativeUnsupported(
					    std::format("unsupported reduce op {} or shape for Vulkan native f32 reduce slice",
					                VulkanNativeOpName(reduce->op)));
				}
				return VulkanNativeSupported(
				    std::format("f32 reduce {} axis={}", VulkanNativeOpName(reduce->op), reduce->axis));
			}

			if (const auto* softmax = std::get_if<SoftmaxNode>(&resultEntry.node))
			{
				const auto inputIndex = GetVulkanP0ParamIndex(subgraph, softmax->input);
				if (!inputIndex)
				{
					return VulkanNativeUnsupported("Vulkan native softmax input must be a direct graph parameter");
				}
				const auto& input = subgraph.Params()[*inputIndex];
				if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32)
				{
					return VulkanNativeUnsupported(
					    std::format("Vulkan native softmax slice requires Float32 input/output, got {} -> {}",
					                DataTypeName(input.dtype), DataTypeName(output.dtype)));
				}
				if (input.shape != output.shape)
				{
					return VulkanNativeUnsupported(
					    std::format("Vulkan native softmax slice requires identical input/output shapes, got {} -> {}",
					                Validation::ShapeToString(input.shape), Validation::ShapeToString(output.shape)));
				}
				if (!VulkanNativeSupportsSoftmaxF32(input.shape, softmax->axis))
				{
					return VulkanNativeUnsupported(std::format(
					    "Vulkan native softmax requires static non-empty shape and in-range axis, got input={} axis={}",
					    Validation::ShapeToString(input.shape), softmax->axis));
				}
				return VulkanNativeSupported(std::format("f32 softmax axis={}", softmax->axis));
			}

			if (const auto* norm = std::get_if<NormalizationNode>(&resultEntry.node))
			{
				const auto inputIndex = GetVulkanP0ParamIndex(subgraph, norm->input);
				if (!inputIndex)
				{
					return VulkanNativeUnsupported(
					    "Vulkan native normalization input must be a direct graph parameter");
				}
				const auto& input = subgraph.Params()[*inputIndex];
				if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32)
				{
					return VulkanNativeUnsupported(
					    std::format("Vulkan native normalization slice requires Float32 input/output, got {} -> {}",
					                DataTypeName(input.dtype), DataTypeName(output.dtype)));
				}
				if (input.shape != output.shape)
				{
					return VulkanNativeUnsupported(std::format(
					    "Vulkan native normalization slice requires identical input/output shapes, got {} -> {}",
					    Validation::ShapeToString(input.shape), Validation::ShapeToString(output.shape)));
				}
				if (!VulkanNativeSupportsNormalizationF32(norm->mode, input.shape, norm->axis, norm->groupCount))
				{
					return VulkanNativeUnsupported(std::format(
					    "Vulkan native normalization requires supported static non-empty shape, got mode={} input={} "
					    "axis={} groupCount={}",
					    VulkanNativeOpName(norm->mode), Validation::ShapeToString(input.shape), norm->axis,
					    norm->groupCount));
				}
				const auto checkAffine = [&](NodeOutput affine, std::string_view label) -> VulkanNativeSupportReport {
					if (affine.port != 0 || affine.node >= subgraph.NodeCount())
					{
						return VulkanNativeUnsupported(
						    std::format("Vulkan native normalization {} tensor reference is invalid", label));
					}
					const auto& affineEntry = subgraph.GetNodeEntry(affine.node);
					if (affineEntry.outputInfos.size() != 1)
					{
						return VulkanNativeUnsupported(
						    std::format("Vulkan native normalization {} tensor must have one output", label));
					}
					const auto& affineInfo = affineEntry.outputInfos[0];
					if (affineInfo.dtype != DataType::Float32)
					{
						return VulkanNativeUnsupported(
						    std::format("Vulkan native normalization {} tensor must be Float32, got {}", label,
						                DataTypeName(affineInfo.dtype)));
					}
					const auto affineShapeSupported =
					    norm->mode == NormalizationMode::GroupNorm
					        ? VulkanP0SupportsGroupNormAffineShape(input.shape, affineInfo.shape)
					        : VulkanP0SupportsNormalizationAffineShape(input.shape, norm->axis, affineInfo.shape);
					if (!affineShapeSupported)
					{
						return VulkanNativeUnsupported(std::format(
						    "Vulkan native normalization {} tensor has unsupported affine shape, got mode={} input={} "
						    "axis={} groupCount={} {}={}",
						    label, VulkanNativeOpName(norm->mode), Validation::ShapeToString(input.shape), norm->axis,
						    norm->groupCount, label, Validation::ShapeToString(affineInfo.shape)));
					}
					return VulkanNativeSupported({});
				};
				if (norm->scale)
				{
					if (auto report = checkAffine(*norm->scale, "scale"); !report.supported)
					{
						return report;
					}
				}
				if (norm->bias)
				{
					if (auto report = checkAffine(*norm->bias, "bias"); !report.supported)
					{
						return report;
					}
				}
				return VulkanNativeSupported(std::format("f32 normalization {} axis={} groupCount={} affine={}/{}",
				                                         VulkanNativeOpName(norm->mode), norm->axis, norm->groupCount,
				                                         norm->scale.has_value(), norm->bias.has_value()));
			}

			if (const auto* pool = std::get_if<Pool2DNode>(&resultEntry.node))
			{
				const auto inputIndex = GetVulkanP0ParamIndex(subgraph, pool->input);
				if (!inputIndex)
				{
					return VulkanNativeUnsupported("Vulkan native Pool2D input must be a direct graph parameter");
				}
				const auto& input = subgraph.Params()[*inputIndex];
				if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32)
				{
					return VulkanNativeUnsupported(
					    std::format("Vulkan native Pool2D slice requires Float32 input/output, got {} -> {}",
					                DataTypeName(input.dtype), DataTypeName(output.dtype)));
				}
				if (!VulkanNativeSupportsPool2DF32(pool->mode, input.shape, output.shape, pool->kernelShape,
				                                   pool->strides, pool->lowPads, pool->highPads, pool->countIncludePad))
				{
					return VulkanNativeUnsupported(
					    std::format("Vulkan native Pool2D requires static rank-4 f32 input/output, got input={} "
					                "output={} kernel={} strides={} lowPads={} highPads={} countIncludePad={}",
					                Validation::ShapeToString(input.shape), Validation::ShapeToString(output.shape),
					                Validation::ShapeToString(pool->kernelShape),
					                Validation::ShapeToString(pool->strides), Validation::ShapeToString(pool->lowPads),
					                Validation::ShapeToString(pool->highPads), pool->countIncludePad));
				}
				return VulkanNativeSupported(std::format("f32 Pool2D {}", VulkanNativeOpName(pool->mode)));
			}

			if (const auto* upsample = std::get_if<UpsampleNode>(&resultEntry.node))
			{
				const auto inputIndex = GetVulkanP0ParamIndex(subgraph, upsample->input);
				if (!inputIndex)
				{
					return VulkanNativeUnsupported(
					    "Vulkan native nearest Upsample input must be a direct graph parameter");
				}
				const auto& input = subgraph.Params()[*inputIndex];
				if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32)
				{
					return VulkanNativeUnsupported(
					    std::format("Vulkan native nearest Upsample slice requires Float32 input/output, got {} -> {}",
					                DataTypeName(input.dtype), DataTypeName(output.dtype)));
				}
				if (upsample->mode != UpsampleMode::Nearest)
				{
					return VulkanNativeUnsupported("Vulkan native Upsample currently supports nearest mode only");
				}
				if (!VulkanNativeSupportsUpsampleNearestF32(input.shape, output.shape, upsample->alignCorners))
				{
					return VulkanNativeUnsupported(std::format("Vulkan native nearest Upsample requires static rank-4 "
					                                           "f32 input/output and alignCorners=false, "
					                                           "got input={} output={} alignCorners={}",
					                                           Validation::ShapeToString(input.shape),
					                                           Validation::ShapeToString(output.shape),
					                                           upsample->alignCorners));
				}
				return VulkanNativeSupported("f32 nearest Upsample");
			}

			if (const auto* slice = std::get_if<SliceNode>(&resultEntry.node))
			{
				const auto inputIndex = GetVulkanP0ParamIndex(subgraph, slice->input);
				if (!inputIndex)
				{
					return VulkanNativeUnsupported("Vulkan native Slice input must be a direct graph parameter");
				}
				const auto& input = subgraph.Params()[*inputIndex];
				if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32)
				{
					return VulkanNativeUnsupported(
					    std::format("Vulkan native Slice requires Float32 input/output, got {} -> {}",
					                DataTypeName(input.dtype), DataTypeName(output.dtype)));
				}
				if (!VulkanNativeSupportsSliceF32(input.shape, output.shape, slice->axis, slice->start, slice->length))
				{
					return VulkanNativeUnsupported(std::format(
					    "Vulkan native Slice requires compatible static f32 input/output, got input={} output={} "
					    "axis={} start={} length={}",
					    Validation::ShapeToString(input.shape), Validation::ShapeToString(output.shape), slice->axis,
					    slice->start, slice->length));
				}
				return VulkanNativeSupported(
				    std::format("f32 Slice axis={} start={} length={}", slice->axis, slice->start, slice->length));
			}

			return VulkanNativeUnsupported(
			    "Vulkan native one-input slice currently supports only UnaryOpNode, CastNode, ReduceOpNode, "
			    "SoftmaxNode, NormalizationNode, Pool2DNode, UpsampleNode, or SliceNode result nodes");
		}

		if (subgraph.Params().size() == 2 && subgraph.NodeCount() == 3)
		{
			if (const auto* concat = std::get_if<ConcatNode>(&resultEntry.node))
			{
				if (concat->inputs.size() != 2)
				{
					return VulkanNativeUnsupported(std::format(
					    "Vulkan native Concat currently supports exactly 2 inputs, got {}", concat->inputs.size()));
				}
				const auto lhsIndex = GetVulkanP0ParamIndex(subgraph, concat->inputs[0]);
				const auto rhsIndex = GetVulkanP0ParamIndex(subgraph, concat->inputs[1]);
				if (!lhsIndex || !rhsIndex)
				{
					return VulkanNativeUnsupported("Vulkan native Concat inputs must be direct graph parameters");
				}
				const auto& lhs = subgraph.Params()[*lhsIndex];
				const auto& rhs = subgraph.Params()[*rhsIndex];
				if (lhs.dtype != DataType::Float32 || rhs.dtype != DataType::Float32 ||
				    output.dtype != DataType::Float32)
				{
					return VulkanNativeUnsupported(
					    std::format("Vulkan native Concat requires Float32 inputs/output, got lhs={} rhs={} output={}",
					                DataTypeName(lhs.dtype), DataTypeName(rhs.dtype), DataTypeName(output.dtype)));
				}
				if (!VulkanNativeSupportsConcatF32(lhs.shape, rhs.shape, output.shape, concat->axis))
				{
					return VulkanNativeUnsupported(std::format("Vulkan native Concat requires compatible static f32 "
					                                           "shapes, got lhs={} rhs={} output={} axis={}",
					                                           Validation::ShapeToString(lhs.shape),
					                                           Validation::ShapeToString(rhs.shape),
					                                           Validation::ShapeToString(output.shape), concat->axis));
				}
				return VulkanNativeSupported(std::format("f32 Concat axis={}", concat->axis));
			}

			const auto* binary = std::get_if<BinaryOpNode>(&resultEntry.node);
			if (!binary)
			{
				return VulkanNativeUnsupported(
				    "Vulkan native two-input slice currently supports only BinaryOpNode or ConcatNode result nodes");
			}
			if (binary->op == BinaryOp::MatMul)
			{
				const auto lhsInputIndex = GetVulkanP0ParamIndex(subgraph, binary->lhs);
				const auto rhsInputIndex = GetVulkanP0ParamIndex(subgraph, binary->rhs);
				if (!lhsInputIndex || !rhsInputIndex)
				{
					return VulkanNativeUnsupported("Vulkan native MatMul inputs must be direct graph parameters");
				}
				const auto& lhsParam = subgraph.Params()[*lhsInputIndex];
				const auto& rhsParam = subgraph.Params()[*rhsInputIndex];
				if (lhsParam.dtype != DataType::Float32 || rhsParam.dtype != DataType::Float32 ||
				    output.dtype != DataType::Float32)
				{
					return VulkanNativeUnsupported(std::format(
					    "Vulkan native MatMul requires Float32 lhs/rhs/output, got lhs={} rhs={} output={}",
					    DataTypeName(lhsParam.dtype), DataTypeName(rhsParam.dtype), DataTypeName(output.dtype)));
				}
				if (lhsParam.shape.size() != 2 || rhsParam.shape.size() != 2 || output.shape.size() != 2)
				{
					return VulkanNativeUnsupported(std::format(
					    "Vulkan native MatMul requires rank-2 lhs/rhs/output, got lhs={} rhs={} output={}",
					    Validation::ShapeToString(lhsParam.shape), Validation::ShapeToString(rhsParam.shape),
					    Validation::ShapeToString(output.shape)));
				}
				if (lhsParam.shape[1] != rhsParam.shape[0] || output.shape[0] != lhsParam.shape[0] ||
				    output.shape[1] != rhsParam.shape[1])
				{
					return VulkanNativeUnsupported(std::format(
					    "Vulkan native MatMul shape mismatch, expected [M,K] @ [K,N] -> [M,N], got lhs={} rhs={} "
					    "output={}",
					    Validation::ShapeToString(lhsParam.shape), Validation::ShapeToString(rhsParam.shape),
					    Validation::ShapeToString(output.shape)));
				}
				if (!VulkanP0ShapeNumElementsU32(lhsParam.shape) || !VulkanP0ShapeNumElementsU32(rhsParam.shape) ||
				    !VulkanP0ShapeNumElementsU32(output.shape))
				{
					return VulkanNativeUnsupported(
					    "Vulkan native MatMul shapes must be non-empty, non-zero, and contain at most uint32_t "
					    "elements");
				}
				return VulkanNativeSupported("f32 matmul");
			}
			if (!VulkanNativeSupportsSameShapeBinaryF32(binary->op))
			{
				return VulkanNativeUnsupported(
				    std::format("unsupported binary op {} for Vulkan native same-shape binary slice",
				                VulkanNativeOpName(binary->op)));
			}
			const auto lhsInputIndex = GetVulkanP0ParamIndex(subgraph, binary->lhs);
			const auto rhsInputIndex = GetVulkanP0ParamIndex(subgraph, binary->rhs);
			if (!lhsInputIndex || !rhsInputIndex)
			{
				return VulkanNativeUnsupported("Vulkan native binary inputs must be direct graph parameters");
			}
			const auto& lhsParam = subgraph.Params()[*lhsInputIndex];
			const auto& rhsParam = subgraph.Params()[*rhsInputIndex];
			if (lhsParam.dtype != rhsParam.dtype || lhsParam.dtype != output.dtype)
			{
				return VulkanNativeUnsupported(std::format(
				    "Vulkan native binary slice requires matching lhs/rhs/output dtypes, got lhs={} rhs={} output={}",
				    DataTypeName(lhsParam.dtype), DataTypeName(rhsParam.dtype), DataTypeName(output.dtype)));
			}
			if (!VulkanNativeSupportsSameShapeBinary(lhsParam.dtype, binary->op))
			{
				return VulkanNativeUnsupported(std::format(
				    "Vulkan native binary slice requires Float32, Float16, Int8, or UInt8 lhs/rhs/output, got {}",
				    DataTypeName(lhsParam.dtype)));
			}
			if (lhsParam.shape != output.shape || rhsParam.shape != output.shape)
			{
				return VulkanNativeUnsupported(std::format(
				    "Vulkan native binary slice requires identical lhs/rhs/output shapes, got lhs={} rhs={} output={}",
				    Validation::ShapeToString(lhsParam.shape), Validation::ShapeToString(rhsParam.shape),
				    Validation::ShapeToString(output.shape)));
			}
			if (auto shapeReport = DiagnoseVulkanP0SingleForwardShape(output.shape, "binary output");
			    !shapeReport.supported)
			{
				return shapeReport;
			}
			return VulkanNativeSupported(std::format(
			    "same-shape {} binary {}", VulkanNativeShortDTypeName(output.dtype), VulkanNativeOpName(binary->op)));
		}

		return VulkanNativeUnsupported(std::format(
		    "Vulkan native currently supports one-input unary/cast, two-input binary single-kernel graphs, or "
		    "same-shape f32 binary chains; got params={} nodes={}",
		    subgraph.Params().size(), subgraph.NodeCount()));
	}

	std::optional<VulkanP0UnaryPlan> MatchVulkanP0SameShapeUnary(const Graph& graph)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 1 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 2)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* unary = std::get_if<UnaryOpNode>(&resultEntry.node);
		if (!unary)
		{
			return std::nullopt;
		}

		const auto inputIndex = GetVulkanP0ParamIndex(subgraph, unary->input);
		if (!inputIndex)
		{
			return std::nullopt;
		}

		const auto& input = subgraph.Params()[*inputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (input.dtype != output.dtype || !VulkanNativeSupportsSameShapeUnary(input.dtype, unary->op) ||
		    input.shape != output.shape)
		{
			return std::nullopt;
		}

		const auto elementCount = VulkanP0ShapeNumElementsU32(output.shape);
		if (!elementCount)
		{
			return std::nullopt;
		}

		return VulkanP0UnaryPlan{
			.op = unary->op,
			.inputIndex = *inputIndex,
			.dtype = output.dtype,
			.elementCount = *elementCount,
		};
	}

	std::optional<VulkanP0BinaryPlan> MatchVulkanP0SameShapeBinary(const Graph& graph)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 2 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 3)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* binary = std::get_if<BinaryOpNode>(&resultEntry.node);
		if (!binary)
		{
			return std::nullopt;
		}

		const auto lhsInputIndex = GetVulkanP0ParamIndex(subgraph, binary->lhs);
		const auto rhsInputIndex = GetVulkanP0ParamIndex(subgraph, binary->rhs);
		if (!lhsInputIndex || !rhsInputIndex)
		{
			return std::nullopt;
		}

		const auto& lhsParam = subgraph.Params()[*lhsInputIndex];
		const auto& rhsParam = subgraph.Params()[*rhsInputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (lhsParam.dtype != rhsParam.dtype || lhsParam.dtype != output.dtype ||
		    !VulkanNativeSupportsSameShapeBinary(lhsParam.dtype, binary->op) || lhsParam.shape != output.shape ||
		    rhsParam.shape != output.shape)
		{
			return std::nullopt;
		}

		const auto elementCount = VulkanP0ShapeNumElementsU32(output.shape);
		if (!elementCount)
		{
			return std::nullopt;
		}

		return VulkanP0BinaryPlan{
			.op = binary->op,
			.lhsInputIndex = *lhsInputIndex,
			.rhsInputIndex = *rhsInputIndex,
			.dtype = output.dtype,
			.elementCount = *elementCount,
		};
	}

	std::optional<VulkanP0MatMulPlan> MatchVulkanP0MatMulF32(const Graph& graph)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 2 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 3)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* binary = std::get_if<BinaryOpNode>(&resultEntry.node);
		if (!binary || binary->op != BinaryOp::MatMul)
		{
			return std::nullopt;
		}

		const auto lhsInputIndex = GetVulkanP0ParamIndex(subgraph, binary->lhs);
		const auto rhsInputIndex = GetVulkanP0ParamIndex(subgraph, binary->rhs);
		if (!lhsInputIndex || !rhsInputIndex)
		{
			return std::nullopt;
		}

		const auto& lhsParam = subgraph.Params()[*lhsInputIndex];
		const auto& rhsParam = subgraph.Params()[*rhsInputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (lhsParam.dtype != DataType::Float32 || rhsParam.dtype != DataType::Float32 ||
		    output.dtype != DataType::Float32 || lhsParam.shape.size() != 2 || rhsParam.shape.size() != 2 ||
		    output.shape.size() != 2)
		{
			return std::nullopt;
		}
		if (lhsParam.shape[1] != rhsParam.shape[0] || output.shape[0] != lhsParam.shape[0] ||
		    output.shape[1] != rhsParam.shape[1])
		{
			return std::nullopt;
		}

		const auto lhsElementCount = VulkanP0ShapeNumElementsU32(lhsParam.shape);
		const auto rhsElementCount = VulkanP0ShapeNumElementsU32(rhsParam.shape);
		const auto outputElementCount = VulkanP0ShapeNumElementsU32(output.shape);
		if (!lhsElementCount || !rhsElementCount || !outputElementCount)
		{
			return std::nullopt;
		}
		if (lhsParam.shape[0] > std::numeric_limits<std::uint32_t>::max() ||
		    lhsParam.shape[1] > std::numeric_limits<std::uint32_t>::max() ||
		    rhsParam.shape[1] > std::numeric_limits<std::uint32_t>::max())
		{
			return std::nullopt;
		}

		const auto m = static_cast<std::uint32_t>(lhsParam.shape[0]);
		const auto k = static_cast<std::uint32_t>(lhsParam.shape[1]);
		const auto n = static_cast<std::uint32_t>(rhsParam.shape[1]);
		if (!VulkanNativeSupportsMatMulF32(m, k, n))
		{
			return std::nullopt;
		}

		return VulkanP0MatMulPlan{
			.lhsInputIndex = *lhsInputIndex,
			.rhsInputIndex = *rhsInputIndex,
			.m = m,
			.k = k,
			.n = n,
			.outputElementCount = *outputElementCount,
		};
	}

	std::optional<VulkanP0MatMulBiasPlan> MakeVulkanP0MatMulBiasF32Plan(VulkanP0TensorRef lhs, VulkanP0TensorRef rhs,
	                                                                    VulkanP0TensorRef bias,
	                                                                    const OutputInfo& output, bool relu)
	{
		if (lhs.dtype != DataType::Float32 || rhs.dtype != DataType::Float32 || bias.dtype != DataType::Float32 ||
		    output.dtype != DataType::Float32 || lhs.shape.size() != 2 || rhs.shape.size() != 2 ||
		    bias.shape.size() != 2 || output.shape.size() != 2 || lhs.shape[1] != rhs.shape[0] ||
		    output.shape[0] != lhs.shape[0] || output.shape[1] != rhs.shape[1] || bias.shape[1] != output.shape[1] ||
		    (bias.shape[0] != 1 && bias.shape[0] != output.shape[0]))
		{
			return std::nullopt;
		}

		const auto outputElementCount = VulkanP0ShapeNumElementsU32(output.shape);
		if (!outputElementCount)
		{
			return std::nullopt;
		}
		if (lhs.shape[0] > std::numeric_limits<std::uint32_t>::max() ||
		    lhs.shape[1] > std::numeric_limits<std::uint32_t>::max() ||
		    rhs.shape[1] > std::numeric_limits<std::uint32_t>::max() ||
		    bias.shape[0] > std::numeric_limits<std::uint32_t>::max())
		{
			return std::nullopt;
		}

		const auto m = static_cast<std::uint32_t>(lhs.shape[0]);
		const auto k = static_cast<std::uint32_t>(lhs.shape[1]);
		const auto n = static_cast<std::uint32_t>(rhs.shape[1]);
		const auto biasRows = static_cast<std::uint32_t>(bias.shape[0]);
		if (!VulkanNativeSupportsMatMulBiasF32(m, k, n, biasRows))
		{
			return std::nullopt;
		}

		return VulkanP0MatMulBiasPlan{
			.lhs = std::move(lhs),
			.rhs = std::move(rhs),
			.bias = std::move(bias),
			.m = m,
			.k = k,
			.n = n,
			.biasRows = biasRows,
			.outputElementCount = *outputElementCount,
			.relu = relu,
		};
	}

	std::optional<VulkanP0MatMulBiasPlan>
	MatchVulkanP0MatMulBiasF32(const Graph& graph, VulkanP0ExternalTensorBuilder* externalBuilder = nullptr)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Results().size() != 1)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* fused = std::get_if<FusedOpNode>(&resultEntry.node);
		if (!fused ||
		    (fused->pattern != FusionPattern::MatMulBiasAdd && fused->pattern != FusionPattern::MatMulBiasAddReLU) ||
		    fused->args.size() < 3)
		{
			return std::nullopt;
		}
		auto lhs = GetVulkanP0TensorRef(graph, subgraph, fused->args[0], externalBuilder);
		auto rhs = GetVulkanP0TensorRef(graph, subgraph, fused->args[1], externalBuilder);
		auto bias = GetVulkanP0TensorRef(graph, subgraph, fused->args[2], externalBuilder);
		if (!lhs || !rhs || !bias)
		{
			return std::nullopt;
		}

		return MakeVulkanP0MatMulBiasF32Plan(std::move(*lhs), std::move(*rhs), std::move(*bias),
		                                     resultEntry.outputInfos[0],
		                                     fused->pattern == FusionPattern::MatMulBiasAddReLU);
	}

	std::optional<VulkanP0LinearChainPlan>
	MatchVulkanP0LinearChainF32(const Graph& graph, VulkanP0ExternalTensorBuilder* externalBuilder = nullptr)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Results().size() != 1)
		{
			return std::nullopt;
		}
		const auto finalResult = subgraph.Results()[0];
		if (finalResult.port != 0 || finalResult.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		std::vector<std::size_t> useCount(subgraph.NodeCount(), 0);
		for (NodeId nodeId = 0; nodeId < subgraph.NodeCount(); ++nodeId)
		{
			const auto& entry = subgraph.GetNodeEntry(nodeId);
			const auto* fused = std::get_if<FusedOpNode>(&entry.node);
			if (!fused)
			{
				continue;
			}
			for (const auto arg : fused->args)
			{
				if (arg.port == 0 && arg.node < useCount.size())
				{
					++useCount[arg.node];
				}
			}
		}

		VulkanP0WorkspacePlanner workspacePlanner;
		std::vector<std::optional<VulkanP0TensorRef>> values(subgraph.NodeCount());
		VulkanP0LinearChainPlan chain;

		const auto requireValue = [&](NodeOutput output) -> std::optional<VulkanP0TensorRef> {
			if (output.port != 0 || output.node >= values.size() || !values[output.node])
			{
				return std::nullopt;
			}
			return *values[output.node];
		};

		for (NodeId nodeId = 0; nodeId < subgraph.NodeCount(); ++nodeId)
		{
			const auto& entry = subgraph.GetNodeEntry(nodeId);
			if (entry.outputInfos.size() != 1)
			{
				return std::nullopt;
			}
			const auto& output = entry.outputInfos[0];

			if (std::holds_alternative<ParamRefNode>(entry.node) ||
			    std::holds_alternative<VariableRefNode>(entry.node) || std::holds_alternative<ConstantNode>(entry.node))
			{
				values[nodeId] = GetVulkanP0TensorRef(graph, subgraph, { nodeId, 0 }, externalBuilder);
				if (!values[nodeId])
				{
					return std::nullopt;
				}
				continue;
			}

			const auto* fused = std::get_if<FusedOpNode>(&entry.node);
			if (!fused ||
			    (fused->pattern != FusionPattern::MatMulBiasAdd &&
			     fused->pattern != FusionPattern::MatMulBiasAddReLU) ||
			    fused->args.size() < 3)
			{
				return std::nullopt;
			}
			if (NodeOutput{ nodeId, 0 } != finalResult && useCount[nodeId] != 1)
			{
				return std::nullopt;
			}

			auto lhs = requireValue(fused->args[0]);
			auto rhs = requireValue(fused->args[1]);
			auto bias = requireValue(fused->args[2]);
			if (!lhs || !rhs || !bias)
			{
				return std::nullopt;
			}
			auto layer = MakeVulkanP0MatMulBiasF32Plan(std::move(*lhs), std::move(*rhs), std::move(*bias), output,
			                                           fused->pattern == FusionPattern::MatMulBiasAddReLU);
			if (!layer)
			{
				return std::nullopt;
			}
			const auto outputRef = [&]() -> VulkanP0TensorRef {
				if (NodeOutput{ nodeId, 0 } == finalResult)
				{
					return VulkanP0TensorRef{
						.argumentKind = VulkanNativeArgumentKind::OutputTensor,
						.argumentIndex = 0,
						.dtype = output.dtype,
						.shape = output.shape,
						.elementCount = layer->outputElementCount,
					};
				}
				const auto workspaceIndex =
				    workspacePlanner.Allocate(TensorByteSizeForShape(output.dtype, output.shape), alignof(float),
				                              chain.kernels.size(), subgraph.NodeCount());
				return VulkanP0TensorRef{
					.argumentKind = VulkanNativeArgumentKind::WorkspaceTensor,
					.argumentIndex = workspaceIndex,
					.dtype = output.dtype,
					.shape = output.shape,
					.elementCount = layer->outputElementCount,
				};
			}();
			values[nodeId] = outputRef;
			chain.kernels.push_back({
			    .lhs = std::move(layer->lhs),
			    .rhs = std::move(layer->rhs),
			    .bias = std::move(layer->bias),
			    .output = outputRef,
			    .m = layer->m,
			    .k = layer->k,
			    .n = layer->n,
			    .biasRows = layer->biasRows,
			    .outputElementCount = layer->outputElementCount,
			    .relu = layer->relu,
			});
		}

		if (chain.kernels.size() < 2 || !values[finalResult.node] ||
		    values[finalResult.node]->argumentKind != VulkanNativeArgumentKind::OutputTensor)
		{
			return std::nullopt;
		}
		chain.workspaceTensors = workspacePlanner.TakeWorkspaceTensors();
		if (chain.workspaceTensors.empty())
		{
			return std::nullopt;
		}
		return chain;
	}

	std::optional<VulkanP0CastPlan> MatchVulkanP0SameShapeCast(const Graph& graph)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 1 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 2)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* cast = std::get_if<CastNode>(&resultEntry.node);
		if (!cast)
		{
			return std::nullopt;
		}

		const auto inputIndex = GetVulkanP0ParamIndex(subgraph, cast->input);
		if (!inputIndex)
		{
			return std::nullopt;
		}

		const auto& input = subgraph.Params()[*inputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (output.dtype != cast->targetType || input.shape != output.shape ||
		    !VulkanNativeSupportsSameShapeCast(input.dtype, output.dtype))
		{
			return std::nullopt;
		}

		const auto elementCount = VulkanP0ShapeNumElementsU32(output.shape);
		if (!elementCount)
		{
			return std::nullopt;
		}

		return VulkanP0CastPlan{
			.inputIndex = *inputIndex,
			.elementCount = *elementCount,
			.srcType = input.dtype,
			.dstType = output.dtype,
		};
	}

	std::optional<VulkanP0ReducePlan> MatchVulkanP0ReduceF32(const Graph& graph)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 1 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 2)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* reduce = std::get_if<ReduceOpNode>(&resultEntry.node);
		if (!reduce)
		{
			return std::nullopt;
		}
		const auto inputIndex = GetVulkanP0ParamIndex(subgraph, reduce->input);
		if (!inputIndex)
		{
			return std::nullopt;
		}

		const auto& input = subgraph.Params()[*inputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32 || reduce->axis >= input.shape.size())
		{
			return std::nullopt;
		}
		if (VulkanP0ReduceOutputShape(input.shape, reduce->axis) != output.shape)
		{
			return std::nullopt;
		}
		if (!VulkanNativeSupportsReduceF32(reduce->op, input.shape, reduce->axis))
		{
			return std::nullopt;
		}
		const auto inputElementCount = VulkanP0ShapeNumElementsU32(input.shape);
		const auto outputElementCount = VulkanP0ShapeNumElementsU32(output.shape);
		if (!inputElementCount || !outputElementCount)
		{
			return std::nullopt;
		}

		return VulkanP0ReducePlan{
			.op = reduce->op,
			.inputIndex = *inputIndex,
			.inputElementCount = *inputElementCount,
			.outputElementCount = *outputElementCount,
			.axis = reduce->axis,
			.inputShape = input.shape,
			.outputShape = output.shape,
		};
	}

	std::optional<VulkanP0SoftmaxPlan> MatchVulkanP0SoftmaxF32(const Graph& graph)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 1 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 2)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* softmax = std::get_if<SoftmaxNode>(&resultEntry.node);
		if (!softmax)
		{
			return std::nullopt;
		}
		const auto inputIndex = GetVulkanP0ParamIndex(subgraph, softmax->input);
		if (!inputIndex)
		{
			return std::nullopt;
		}

		const auto& input = subgraph.Params()[*inputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32 || input.shape != output.shape ||
		    !VulkanNativeSupportsSoftmaxF32(input.shape, softmax->axis))
		{
			return std::nullopt;
		}
		const auto elementCount = VulkanP0ShapeNumElementsU32(output.shape);
		if (!elementCount)
		{
			return std::nullopt;
		}

		return VulkanP0SoftmaxPlan{
			.inputIndex = *inputIndex,
			.elementCount = *elementCount,
			.axis = softmax->axis,
			.inputShape = input.shape,
		};
	}

	std::optional<VulkanP0Pool2DPlan> MatchVulkanP0Pool2DF32(const Graph& graph)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 1 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 2)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* pool = std::get_if<Pool2DNode>(&resultEntry.node);
		if (!pool)
		{
			return std::nullopt;
		}
		const auto inputIndex = GetVulkanP0ParamIndex(subgraph, pool->input);
		if (!inputIndex)
		{
			return std::nullopt;
		}

		const auto& input = subgraph.Params()[*inputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32 ||
		    !VulkanNativeSupportsPool2DF32(pool->mode, input.shape, output.shape, pool->kernelShape, pool->strides,
		                                   pool->lowPads, pool->highPads, pool->countIncludePad))
		{
			return std::nullopt;
		}
		const auto inputElementCount = VulkanP0ShapeNumElementsU32(input.shape);
		const auto outputElementCount = VulkanP0ShapeNumElementsU32(output.shape);
		if (!inputElementCount || !outputElementCount)
		{
			return std::nullopt;
		}

		return VulkanP0Pool2DPlan{
			.mode = pool->mode,
			.inputIndex = *inputIndex,
			.inputElementCount = *inputElementCount,
			.outputElementCount = *outputElementCount,
			.inputShape = input.shape,
			.outputShape = output.shape,
			.kernelShape = pool->kernelShape,
			.strides = pool->strides,
			.lowPads = pool->lowPads,
			.highPads = pool->highPads,
			.countIncludePad = pool->countIncludePad,
		};
	}

	std::optional<VulkanP0Conv2DPlan> MatchVulkanP0Conv2DF32(const Graph& graph,
	                                                         VulkanP0ExternalTensorBuilder* externalBuilder = nullptr)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().empty() || subgraph.Results().size() != 1)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* conv = std::get_if<Conv2DNode>(&resultEntry.node);
		if (!conv)
		{
			return std::nullopt;
		}

		auto input = GetVulkanP0TensorRef(graph, subgraph, conv->input, externalBuilder);
		auto weight = GetVulkanP0TensorRef(graph, subgraph, conv->weight, externalBuilder);
		if (!input || !weight)
		{
			return std::nullopt;
		}
		std::optional<VulkanP0TensorRef> bias;
		if (conv->bias)
		{
			bias = GetVulkanP0TensorRef(graph, subgraph, *conv->bias, externalBuilder);
			if (!bias)
			{
				return std::nullopt;
			}
		}

		const auto& output = resultEntry.outputInfos[0];
		if (input->dtype != DataType::Float32 || weight->dtype != DataType::Float32 ||
		    output.dtype != DataType::Float32 || (bias && bias->dtype != DataType::Float32) ||
		    !VulkanNativeSupportsConv2DF32(input->shape, weight->shape, output.shape, conv->strides, conv->dilations,
		                                   conv->lowPads, conv->highPads, conv->groupCount))
		{
			return std::nullopt;
		}
		if (bias)
		{
			const auto biasOk = (bias->shape.size() == 1 && bias->shape[0] == output.shape[1]) ||
			                    (bias->shape.size() == 4 && bias->shape[0] == 1 && bias->shape[1] == output.shape[1] &&
			                     bias->shape[2] == 1 && bias->shape[3] == 1);
			if (!biasOk)
			{
				return std::nullopt;
			}
		}

		const auto outputElementCount = VulkanP0ShapeNumElementsU32(output.shape);
		if (!outputElementCount)
		{
			return std::nullopt;
		}
		auto inputShape = input->shape;
		auto weightShape = weight->shape;

		return VulkanP0Conv2DPlan{
			.input = std::move(*input),
			.weight = std::move(*weight),
			.bias = std::move(bias),
			.outputElementCount = *outputElementCount,
			.inputShape = std::move(inputShape),
			.weightShape = std::move(weightShape),
			.outputShape = output.shape,
			.strides = conv->strides,
			.dilations = conv->dilations,
			.lowPads = conv->lowPads,
			.highPads = conv->highPads,
			.groupCount = conv->groupCount,
		};
	}

	std::optional<VulkanP0UpsamplePlan> MatchVulkanP0UpsampleNearestF32(const Graph& graph)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 1 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 2)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* upsample = std::get_if<UpsampleNode>(&resultEntry.node);
		if (!upsample || upsample->mode != UpsampleMode::Nearest)
		{
			return std::nullopt;
		}
		const auto inputIndex = GetVulkanP0ParamIndex(subgraph, upsample->input);
		if (!inputIndex)
		{
			return std::nullopt;
		}

		const auto& input = subgraph.Params()[*inputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32 ||
		    !VulkanNativeSupportsUpsampleNearestF32(input.shape, output.shape, upsample->alignCorners))
		{
			return std::nullopt;
		}
		const auto inputElementCount = VulkanP0ShapeNumElementsU32(input.shape);
		const auto outputElementCount = VulkanP0ShapeNumElementsU32(output.shape);
		if (!inputElementCount || !outputElementCount)
		{
			return std::nullopt;
		}

		return VulkanP0UpsamplePlan{
			.inputIndex = *inputIndex,
			.inputElementCount = *inputElementCount,
			.outputElementCount = *outputElementCount,
			.inputShape = input.shape,
			.outputShape = output.shape,
		};
	}

	std::optional<VulkanP0SlicePlan> MatchVulkanP0SliceF32(const Graph& graph)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 1 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 2)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* slice = std::get_if<SliceNode>(&resultEntry.node);
		if (!slice)
		{
			return std::nullopt;
		}
		const auto inputIndex = GetVulkanP0ParamIndex(subgraph, slice->input);
		if (!inputIndex)
		{
			return std::nullopt;
		}

		const auto& input = subgraph.Params()[*inputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32 ||
		    !VulkanNativeSupportsSliceF32(input.shape, output.shape, slice->axis, slice->start, slice->length))
		{
			return std::nullopt;
		}
		const auto inputElementCount = VulkanP0ShapeNumElementsU32(input.shape);
		const auto outputElementCount = VulkanP0ShapeNumElementsU32(output.shape);
		if (!inputElementCount || !outputElementCount)
		{
			return std::nullopt;
		}

		return VulkanP0SlicePlan{
			.inputIndex = *inputIndex,
			.inputElementCount = *inputElementCount,
			.outputElementCount = *outputElementCount,
			.inputShape = input.shape,
			.outputShape = output.shape,
			.axis = slice->axis,
			.start = slice->start,
			.length = slice->length,
		};
	}

	std::optional<VulkanP0ConcatPlan> MatchVulkanP0ConcatF32(const Graph& graph)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().size() != 2 || subgraph.Results().size() != 1 || subgraph.NodeCount() != 3)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* concat = std::get_if<ConcatNode>(&resultEntry.node);
		if (!concat || concat->inputs.size() != 2)
		{
			return std::nullopt;
		}
		const auto lhsIndex = GetVulkanP0ParamIndex(subgraph, concat->inputs[0]);
		const auto rhsIndex = GetVulkanP0ParamIndex(subgraph, concat->inputs[1]);
		if (!lhsIndex || !rhsIndex)
		{
			return std::nullopt;
		}

		const auto& lhs = subgraph.Params()[*lhsIndex];
		const auto& rhs = subgraph.Params()[*rhsIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (lhs.dtype != DataType::Float32 || rhs.dtype != DataType::Float32 || output.dtype != DataType::Float32 ||
		    !VulkanNativeSupportsConcatF32(lhs.shape, rhs.shape, output.shape, concat->axis))
		{
			return std::nullopt;
		}
		const auto lhsElementCount = VulkanP0ShapeNumElementsU32(lhs.shape);
		const auto rhsElementCount = VulkanP0ShapeNumElementsU32(rhs.shape);
		const auto outputElementCount = VulkanP0ShapeNumElementsU32(output.shape);
		if (!lhsElementCount || !rhsElementCount || !outputElementCount)
		{
			return std::nullopt;
		}

		return VulkanP0ConcatPlan{
			.lhsIndex = *lhsIndex,
			.rhsIndex = *rhsIndex,
			.lhsElementCount = *lhsElementCount,
			.rhsElementCount = *rhsElementCount,
			.outputElementCount = *outputElementCount,
			.lhsShape = lhs.shape,
			.rhsShape = rhs.shape,
			.outputShape = output.shape,
			.axis = concat->axis,
		};
	}

	std::optional<VulkanP0ConvTranspose2DPlan>
	MatchVulkanP0ConvTranspose2DF32(const Graph& graph, VulkanP0ExternalTensorBuilder* externalBuilder = nullptr)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().empty() || subgraph.Results().size() != 1)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* convT = std::get_if<ConvTranspose2DNode>(&resultEntry.node);
		if (!convT)
		{
			return std::nullopt;
		}

		auto input = GetVulkanP0TensorRef(graph, subgraph, convT->input, externalBuilder);
		auto weight = GetVulkanP0TensorRef(graph, subgraph, convT->weight, externalBuilder);
		if (!input || !weight)
		{
			return std::nullopt;
		}
		std::optional<VulkanP0TensorRef> bias;
		if (convT->bias)
		{
			bias = GetVulkanP0TensorRef(graph, subgraph, *convT->bias, externalBuilder);
			if (!bias)
			{
				return std::nullopt;
			}
		}

		const auto& output = resultEntry.outputInfos[0];
		if (input->dtype != DataType::Float32 || weight->dtype != DataType::Float32 ||
		    output.dtype != DataType::Float32 || (bias && bias->dtype != DataType::Float32) ||
		    !VulkanNativeSupportsConvTranspose2DF32(input->shape, weight->shape, output.shape, convT->strides,
		                                            convT->dilations, convT->lowPads, convT->highPads,
		                                            convT->outputPads, convT->groupCount))
		{
			return std::nullopt;
		}
		if (bias)
		{
			const auto biasOk = (bias->shape.size() == 1 && bias->shape[0] == output.shape[1]) ||
			                    (bias->shape.size() == 4 && bias->shape[0] == 1 && bias->shape[1] == output.shape[1] &&
			                     bias->shape[2] == 1 && bias->shape[3] == 1);
			if (!biasOk)
			{
				return std::nullopt;
			}
		}

		const auto outputElementCount = VulkanP0ShapeNumElementsU32(output.shape);
		if (!outputElementCount)
		{
			return std::nullopt;
		}
		auto inputShape = input->shape;
		auto weightShape = weight->shape;

		return VulkanP0ConvTranspose2DPlan{
			.input = std::move(*input),
			.weight = std::move(*weight),
			.bias = std::move(bias),
			.outputElementCount = *outputElementCount,
			.inputShape = std::move(inputShape),
			.weightShape = std::move(weightShape),
			.outputShape = output.shape,
			.strides = convT->strides,
			.dilations = convT->dilations,
			.lowPads = convT->lowPads,
			.highPads = convT->highPads,
			.outputPads = convT->outputPads,
			.groupCount = convT->groupCount,
		};
	}

	std::optional<VulkanP0NormalizationPlan>
	MatchVulkanP0NormalizationF32(const Graph& graph, VulkanP0ExternalTensorBuilder* externalBuilder = nullptr)
	{
		if (!IsVulkanP0SingleForwardGraph(graph))
		{
			return std::nullopt;
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Params().empty() || subgraph.Results().size() != 1)
		{
			return std::nullopt;
		}

		const auto result = subgraph.Results()[0];
		if (result.port != 0 || result.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}

		const auto& resultEntry = subgraph.GetNodeEntry(result.node);
		if (resultEntry.outputInfos.size() != 1)
		{
			return std::nullopt;
		}

		const auto* norm = std::get_if<NormalizationNode>(&resultEntry.node);
		if (!norm)
		{
			return std::nullopt;
		}
		const auto inputIndex = GetVulkanP0ParamIndex(subgraph, norm->input);
		if (!inputIndex)
		{
			return std::nullopt;
		}

		const auto& input = subgraph.Params()[*inputIndex];
		const auto& output = resultEntry.outputInfos[0];
		if (input.dtype != DataType::Float32 || output.dtype != DataType::Float32 || input.shape != output.shape ||
		    !VulkanNativeSupportsNormalizationF32(norm->mode, input.shape, norm->axis, norm->groupCount))
		{
			return std::nullopt;
		}
		const auto elementCount = VulkanP0ShapeNumElementsU32(output.shape);
		if (!elementCount)
		{
			return std::nullopt;
		}
		const auto getAffine = [&](const std::optional<NodeOutput>& output) -> std::optional<VulkanP0TensorRef> {
			if (!output)
			{
				return VulkanP0TensorRef{};
			}
			auto ref = GetVulkanP0TensorRef(graph, subgraph, *output, externalBuilder);
			if (!ref || ref->dtype != DataType::Float32)
			{
				return std::nullopt;
			}
			const auto affineShapeSupported =
			    norm->mode == NormalizationMode::GroupNorm
			        ? VulkanP0SupportsGroupNormAffineShape(input.shape, ref->shape)
			        : VulkanP0SupportsNormalizationAffineShape(input.shape, norm->axis, ref->shape);
			if (!affineShapeSupported)
			{
				return std::nullopt;
			}
			return ref;
		};
		std::optional<VulkanP0TensorRef> scale;
		if (norm->scale)
		{
			scale = getAffine(norm->scale);
			if (!scale)
			{
				return std::nullopt;
			}
		}
		std::optional<VulkanP0TensorRef> bias;
		if (norm->bias)
		{
			bias = getAffine(norm->bias);
			if (!bias)
			{
				return std::nullopt;
			}
		}

		return VulkanP0NormalizationPlan{
			.mode = norm->mode,
			.inputIndex = *inputIndex,
			.scale = std::move(scale),
			.bias = std::move(bias),
			.elementCount = *elementCount,
			.axis = norm->axis,
			.groupCount = norm->groupCount,
			.epsilon = norm->epsilon,
			.inputShape = input.shape,
		};
	}

	VulkanNativeFeature VulkanNativeUnaryF32FeatureFlag(UnaryOp op)
	{
		switch (op)
		{
		case UnaryOp::Negate:
			return VulkanNativeFeature::SameShapeElementwiseNegateF32;
		case UnaryOp::Abs:
			return VulkanNativeFeature::SameShapeElementwiseAbsF32;
		case UnaryOp::Sqrt:
			return VulkanNativeFeature::SameShapeElementwiseSqrtF32;
		case UnaryOp::Exp:
			return VulkanNativeFeature::SameShapeElementwiseExpF32;
		case UnaryOp::Log:
			return VulkanNativeFeature::SameShapeElementwiseLogF32;
		case UnaryOp::Sin:
			return VulkanNativeFeature::SameShapeElementwiseSinF32;
		case UnaryOp::Cos:
			return VulkanNativeFeature::SameShapeElementwiseCosF32;
		default:
			throw std::runtime_error("Unsupported Vulkan native unary op");
		}
	}

	VulkanNativeFeature VulkanNativeCastFeatureFlag(DataType srcType, DataType dstType)
	{
		if (srcType == DataType::Float32 && dstType == DataType::Int32)
		{
			return VulkanNativeFeature::SameShapeCastFloat32ToInt32;
		}
		if (srcType == DataType::Int32 && dstType == DataType::Float32)
		{
			return VulkanNativeFeature::SameShapeCastInt32ToFloat32;
		}
		if (VulkanNativeSupportsSameShapeCast(srcType, dstType))
		{
			return VulkanNativeFeature::SameShapeCastLowPrecision;
		}
		throw std::runtime_error("Unsupported Vulkan native cast");
	}

	VulkanNativeFeature VulkanNativeBinaryF32FeatureFlag(BinaryOp op)
	{
		switch (op)
		{
		case BinaryOp::Add:
			return VulkanNativeFeature::SameShapeElementwiseAddF32;
		case BinaryOp::Subtract:
			return VulkanNativeFeature::SameShapeElementwiseSubtractF32;
		case BinaryOp::Multiply:
			return VulkanNativeFeature::SameShapeElementwiseMultiplyF32;
		case BinaryOp::Divide:
			return VulkanNativeFeature::SameShapeElementwiseDivideF32;
		case BinaryOp::Max:
			return VulkanNativeFeature::SameShapeElementwiseMaxF32;
		case BinaryOp::Min:
			return VulkanNativeFeature::SameShapeElementwiseMinF32;
		default:
			throw std::runtime_error("Unsupported Vulkan native same-shape f32 binary op");
		}
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeSameShapeUnaryP0(const Graph& graph)
	{
		const auto plan = MatchVulkanP0SameShapeUnary(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		payload.featureSet.AddFeature(plan->dtype == DataType::Float32
		                                  ? VulkanNativeUnaryF32FeatureFlag(plan->op)
		                                  : VulkanNativeFeature::SameShapeElementwiseUnaryLowPrecision);
		auto spirv = VulkanNativeSameShapeUnarySPIRV(plan->dtype, plan->op, plan->elementCount);
		payload.spirv = std::move(spirv.words);

		const auto byteSize = static_cast<std::uint64_t>(plan->elementCount) * ElementByteSize(plan->dtype);
		auto requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize);
		AddVulkanP0DTypeDeviceRequirements(requirements.deviceRequirements, plan->dtype);
		payload.kernels.push_back({
		    .entryPoint = "main",
		    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->elementCount), .y = 1, .z = 1 },
		    .requirements = requirements,
		    .arguments = {
		        { .kind = VulkanNativeArgumentKind::InputTensor,
		          .index = plan->inputIndex,
		          .binding = 0,
		          .byteOffset = 0,
		          .byteSize = byteSize },
		        { .kind = VulkanNativeArgumentKind::OutputTensor,
		          .index = 0,
		          .binding = 1,
		          .byteOffset = 0,
		          .byteSize = byteSize },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeSameShapeCastP0(const Graph& graph)
	{
		const auto plan = MatchVulkanP0SameShapeCast(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		payload.featureSet.AddFeature(VulkanNativeCastFeatureFlag(plan->srcType, plan->dstType));
		auto spirv = VulkanNativeSameShapeCastSPIRV(plan->srcType, plan->dstType, plan->elementCount);
		payload.spirv = std::move(spirv.words);

		payload.kernels.push_back({
		    .entryPoint = "main",
		    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->elementCount), .y = 1, .z = 1 },
		    .requirements = VulkanP0CastKernelRequirements(plan->srcType, plan->dstType),
		    .arguments = {
		        { .kind = VulkanNativeArgumentKind::InputTensor,
		          .index = plan->inputIndex,
		          .binding = 0,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->elementCount) * ElementByteSize(plan->srcType) },
		        { .kind = VulkanNativeArgumentKind::OutputTensor,
		          .index = 0,
		          .binding = 1,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->elementCount) * ElementByteSize(plan->dstType) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeReduceF32P0(const Graph& graph)
	{
		const auto plan = MatchVulkanP0ReduceF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		payload.featureSet.AddFeature(VulkanNativeFeature::ReduceF32);
		auto spirv = VulkanNativeReduceF32SPIRV(plan->op, plan->inputShape, plan->axis);
		payload.spirv = std::move(spirv.words);

		payload.kernels.push_back({
		    .entryPoint = std::string(VulkanNativeReduceF32KernelName(plan->op)),
		    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->outputElementCount), .y = 1, .z = 1 },
		    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
		    .arguments = {
		        { .kind = VulkanNativeArgumentKind::InputTensor,
		          .index = plan->inputIndex,
		          .binding = 0,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->inputElementCount) * sizeof(float) },
		        { .kind = VulkanNativeArgumentKind::OutputTensor,
		          .index = 0,
		          .binding = 1,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->outputElementCount) * sizeof(float) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeSoftmaxF32P0(const Graph& graph)
	{
		const auto plan = MatchVulkanP0SoftmaxF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		payload.featureSet.AddFeature(VulkanNativeFeature::SoftmaxF32);
		auto spirv = VulkanNativeSoftmaxF32SPIRV(plan->inputShape, plan->axis);
		payload.spirv = std::move(spirv.words);

		const auto byteSize = static_cast<std::uint64_t>(plan->elementCount) * sizeof(float);
		const auto axisSize = static_cast<std::uint32_t>(plan->inputShape[plan->axis]);
		const auto rowCount = plan->elementCount / axisSize;
		const auto rowWorkspaceBytes = static_cast<std::uint64_t>(rowCount) * sizeof(float);
		VulkanP0WorkspacePlanner workspacePlanner;
		const auto rowMaxWorkspace = workspacePlanner.Allocate(rowWorkspaceBytes, alignof(float), 0, 2);
		const auto rowSumWorkspace = workspacePlanner.Allocate(rowWorkspaceBytes, alignof(float), 1, 2);

		payload.kernels.push_back({
		    .entryPoint = std::string(VulkanNativeSoftmaxRowMaxF32KernelName()),
		    .groups = { .x = VulkanP0ElementwiseGroupCount(rowCount), .y = 1, .z = 1 },
		    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
		    .arguments = {
		        { .kind = VulkanNativeArgumentKind::InputTensor,
		          .index = plan->inputIndex,
		          .binding = 0,
		          .byteOffset = 0,
		          .byteSize = byteSize },
		        workspacePlanner.Argument(rowMaxWorkspace, 1, rowWorkspaceBytes),
		    },
		});
		payload.kernels.push_back({
		    .entryPoint = std::string(VulkanNativeSoftmaxRowSumF32KernelName()),
		    .groups = { .x = VulkanP0ElementwiseGroupCount(rowCount), .y = 1, .z = 1 },
		    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
		    .arguments = {
		        { .kind = VulkanNativeArgumentKind::InputTensor,
		          .index = plan->inputIndex,
		          .binding = 0,
		          .byteOffset = 0,
		          .byteSize = byteSize },
		        workspacePlanner.Argument(rowMaxWorkspace, 1, rowWorkspaceBytes),
		        workspacePlanner.Argument(rowSumWorkspace, 2, rowWorkspaceBytes),
		    },
		});
		payload.kernels.push_back({
		    .entryPoint = std::string(VulkanNativeSoftmaxWriteF32KernelName()),
		    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->elementCount), .y = 1, .z = 1 },
		    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
		    .arguments = {
		        { .kind = VulkanNativeArgumentKind::InputTensor,
		          .index = plan->inputIndex,
		          .binding = 0,
		          .byteOffset = 0,
		          .byteSize = byteSize },
		        workspacePlanner.Argument(rowMaxWorkspace, 1, rowWorkspaceBytes),
		        workspacePlanner.Argument(rowSumWorkspace, 2, rowWorkspaceBytes),
		        { .kind = VulkanNativeArgumentKind::OutputTensor,
		          .index = 0,
		          .binding = 3,
		          .byteOffset = 0,
		          .byteSize = byteSize },
		    },
		});
		payload.workspaceTensors = workspacePlanner.TakeWorkspaceTensors();

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeNormalizationF32P0(const Graph& graph)
	{
		VulkanP0ExternalTensorBuilder externalBuilder;
		const auto plan = MatchVulkanP0NormalizationF32(graph, &externalBuilder);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		payload.featureSet.AddFeature(VulkanNativeFeature::NormalizationF32);

		const auto outputByteSize = static_cast<std::uint64_t>(plan->elementCount) * sizeof(float);
		const auto useStagedAxisNormalization = plan->groupCount == 1 && plan->mode == NormalizationMode::LayerNorm;
		if (useStagedAxisNormalization)
		{
			auto spirv = VulkanNativeAxisNormalizationF32SPIRV(plan->mode, plan->inputShape, plan->axis, plan->epsilon,
			                                                   plan->scale.has_value(), plan->bias.has_value());
			payload.spirv = std::move(spirv.words);

			const auto axisSize = static_cast<std::uint32_t>(plan->inputShape[plan->axis]);
			const auto rowCount = plan->elementCount / axisSize;
			const auto rowWorkspaceBytes = static_cast<std::uint64_t>(rowCount) * sizeof(float);
			VulkanP0WorkspacePlanner workspacePlanner;
			std::optional<std::uint32_t> rowMeanWorkspace;
			if (plan->mode == NormalizationMode::LayerNorm)
			{
				rowMeanWorkspace = workspacePlanner.Allocate(rowWorkspaceBytes, alignof(float), 0, 1);
			}
			const auto rowDenomWorkspace = workspacePlanner.Allocate(rowWorkspaceBytes, alignof(float), 0, 1);

			std::vector<VulkanNativeArgumentSpec> statsArgs{
				{ .kind = VulkanNativeArgumentKind::InputTensor,
				  .index = plan->inputIndex,
				  .binding = 0,
				  .byteOffset = 0,
				  .byteSize = outputByteSize },
			};
			std::uint32_t nextBinding = 1;
			if (rowMeanWorkspace)
			{
				statsArgs.push_back(workspacePlanner.Argument(*rowMeanWorkspace, nextBinding++, rowWorkspaceBytes));
			}
			statsArgs.push_back(workspacePlanner.Argument(rowDenomWorkspace, nextBinding, rowWorkspaceBytes));
			payload.kernels.push_back({
			    .entryPoint = std::string(VulkanNativeAxisNormalizationStatsF32KernelName(plan->mode)),
			    .groups = { .x = VulkanP0ElementwiseGroupCount(rowCount), .y = 1, .z = 1 },
			    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
			    .arguments = std::move(statsArgs),
			});

			std::vector<VulkanNativeArgumentSpec> writeArgs{
				{ .kind = VulkanNativeArgumentKind::InputTensor,
				  .index = plan->inputIndex,
				  .binding = 0,
				  .byteOffset = 0,
				  .byteSize = outputByteSize },
			};
			nextBinding = 1;
			if (rowMeanWorkspace)
			{
				writeArgs.push_back(workspacePlanner.Argument(*rowMeanWorkspace, nextBinding++, rowWorkspaceBytes));
			}
			writeArgs.push_back(workspacePlanner.Argument(rowDenomWorkspace, nextBinding++, rowWorkspaceBytes));
			const auto appendAffine = [&](const VulkanP0TensorRef& ref) {
				writeArgs.push_back({ .kind = ref.argumentKind,
				                      .index = ref.argumentIndex,
				                      .binding = nextBinding++,
				                      .byteOffset = 0,
				                      .byteSize = static_cast<std::uint64_t>(ref.elementCount) * sizeof(float) });
			};
			if (plan->scale)
			{
				appendAffine(*plan->scale);
			}
			if (plan->bias)
			{
				appendAffine(*plan->bias);
			}
			writeArgs.push_back({ .kind = VulkanNativeArgumentKind::OutputTensor,
			                      .index = 0,
			                      .binding = nextBinding,
			                      .byteOffset = 0,
			                      .byteSize = outputByteSize });
			payload.kernels.push_back({
			    .entryPoint = std::string(VulkanNativeAxisNormalizationWriteF32KernelName(plan->mode)),
			    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->elementCount), .y = 1, .z = 1 },
			    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
			    .arguments = std::move(writeArgs),
			});
			payload.workspaceTensors = workspacePlanner.TakeWorkspaceTensors();
		}
		else
		{
			auto spirv =
			    VulkanNativeNormalizationF32SPIRV(plan->mode, plan->inputShape, plan->axis, plan->epsilon,
			                                      plan->scale.has_value(), plan->bias.has_value(), plan->groupCount);
			payload.spirv = std::move(spirv.words);

			std::vector<VulkanNativeArgumentSpec> arguments;
			arguments.push_back({ .kind = VulkanNativeArgumentKind::InputTensor,
			                      .index = plan->inputIndex,
			                      .binding = 0,
			                      .byteOffset = 0,
			                      .byteSize = outputByteSize });
			std::uint32_t nextBinding = 1;
			const auto appendAffine = [&](const VulkanP0TensorRef& ref) {
				arguments.push_back({ .kind = ref.argumentKind,
				                      .index = ref.argumentIndex,
				                      .binding = nextBinding++,
				                      .byteOffset = 0,
				                      .byteSize = static_cast<std::uint64_t>(ref.elementCount) * sizeof(float) });
			};
			if (plan->scale)
			{
				appendAffine(*plan->scale);
			}
			if (plan->bias)
			{
				appendAffine(*plan->bias);
			}
			arguments.push_back({ .kind = VulkanNativeArgumentKind::OutputTensor,
			                      .index = 0,
			                      .binding = nextBinding,
			                      .byteOffset = 0,
			                      .byteSize = outputByteSize });
			payload.kernels.push_back({
			    .entryPoint = std::string(VulkanNativeNormalizationF32KernelName(plan->mode)),
			    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->elementCount), .y = 1, .z = 1 },
			    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
			    .arguments = std::move(arguments),
			});
		}

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.constants = std::move(externalBuilder.constants),
			.weights = std::move(externalBuilder.weights),
			.externalTensorInfos = std::move(externalBuilder.externalTensorInfos),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativePool2DF32P0(const Graph& graph)
	{
		const auto plan = MatchVulkanP0Pool2DF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		payload.featureSet.AddFeature(VulkanNativeFeature::Pool2DF32);
		auto spirv = VulkanNativePool2DF32SPIRV(plan->mode, plan->inputShape, plan->outputShape, plan->kernelShape,
		                                        plan->strides, plan->lowPads, plan->highPads, plan->countIncludePad);
		payload.spirv = std::move(spirv.words);

		payload.kernels.push_back({
		    .entryPoint = std::string(VulkanNativePool2DF32KernelName(plan->mode)),
		    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->outputElementCount), .y = 1, .z = 1 },
		    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
		    .arguments = {
		        { .kind = VulkanNativeArgumentKind::InputTensor,
		          .index = plan->inputIndex,
		          .binding = 0,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->inputElementCount) * sizeof(float) },
		        { .kind = VulkanNativeArgumentKind::OutputTensor,
		          .index = 0,
		          .binding = 1,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->outputElementCount) * sizeof(float) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeSameShapeBinaryP0(const Graph& graph)
	{
		const auto plan = MatchVulkanP0SameShapeBinary(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		payload.featureSet.AddFeature(plan->dtype == DataType::Float32
		                                  ? VulkanNativeBinaryF32FeatureFlag(plan->op)
		                                  : VulkanNativeFeature::SameShapeElementwiseBinaryLowPrecision);
		auto spirv = VulkanNativeSameShapeBinarySPIRV(plan->dtype, plan->op, plan->elementCount);
		payload.spirv = std::move(spirv.words);

		const auto byteSize = static_cast<std::uint64_t>(plan->elementCount) * ElementByteSize(plan->dtype);
		auto requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize);
		AddVulkanP0DTypeDeviceRequirements(requirements.deviceRequirements, plan->dtype);
		payload.kernels.push_back({
		    .entryPoint = "main",
		    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->elementCount), .y = 1, .z = 1 },
		    .requirements = requirements,
		    .arguments = {
		        { .kind = VulkanNativeArgumentKind::InputTensor,
		          .index = plan->lhsInputIndex,
		          .binding = 0,
		          .byteOffset = 0,
		          .byteSize = byteSize },
		        { .kind = VulkanNativeArgumentKind::InputTensor,
		          .index = plan->rhsInputIndex,
		          .binding = 1,
		          .byteOffset = 0,
		          .byteSize = byteSize },
		        { .kind = VulkanNativeArgumentKind::OutputTensor,
		          .index = 0,
		          .binding = 2,
		          .byteOffset = 0,
		          .byteSize = byteSize },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	VulkanNativeArgumentSpec VulkanP0BinaryChainArgument(VulkanP0BinaryChainOperand operand, std::uint32_t binding,
	                                                     std::uint64_t byteSize, std::uint32_t workspaceIndex)
	{
		if (operand.accumulator)
		{
			return VulkanNativeArgumentSpec{
				.kind = VulkanNativeArgumentKind::WorkspaceTensor,
				.index = workspaceIndex,
				.binding = binding,
				.byteOffset = 0,
				.byteSize = byteSize,
			};
		}
		return VulkanNativeArgumentSpec{
			.kind = VulkanNativeArgumentKind::InputTensor,
			.index = operand.inputIndex,
			.binding = binding,
			.byteOffset = 0,
			.byteSize = byteSize,
		};
	}

	VulkanNativeArgumentSpec VulkanP0BinaryDAGArgument(VulkanP0BinaryDAGOperand operand, std::uint32_t binding,
	                                                   std::uint64_t byteSize,
	                                                   std::span<const std::uint32_t> workspaceByKernel)
	{
		if (operand.kind == VulkanP0BinaryDAGOperandKind::Intermediate)
		{
			return VulkanP0ScheduleWorkspaceArgument(operand.index, binding, byteSize, workspaceByKernel);
		}
		return VulkanNativeArgumentSpec{
			.kind = VulkanNativeArgumentKind::InputTensor,
			.index = operand.index,
			.binding = binding,
			.byteOffset = 0,
			.byteSize = byteSize,
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeSameShapeBinaryF32ChainP0(const Graph& graph)
	{
		const auto plan = MatchVulkanP0SameShapeBinaryF32Chain(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		std::vector<BinaryOp> kernelOps;
		kernelOps.reserve(plan->kernels.size());
		for (const auto& kernelPlan : plan->kernels)
		{
			payload.featureSet.AddFeature(VulkanNativeBinaryF32FeatureFlag(kernelPlan.op));
			kernelOps.push_back(kernelPlan.op);
		}
		auto spirv = VulkanNativeSameShapeBinaryF32ChainSPIRV(kernelOps, plan->elementCount);
		payload.spirv = std::move(spirv.words);

		const auto byteSize = static_cast<std::uint64_t>(plan->elementCount) * sizeof(float);
		VulkanP0WorkspacePlanner workspacePlanner;
		const auto accumulatorWorkspace =
		    workspacePlanner.Allocate(byteSize, alignof(float), 0, plan->kernels.size() - 1);
		for (std::size_t kernelIndex = 0; kernelIndex < plan->kernels.size(); ++kernelIndex)
		{
			const auto& kernelPlan = plan->kernels[kernelIndex];
			const auto isFinalKernel = kernelIndex + 1 == plan->kernels.size();
			payload.kernels.push_back({
			    .entryPoint = VulkanNativeSameShapeBinaryF32KernelName(kernelPlan.op),
			    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->elementCount), .y = 1, .z = 1 },
			    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
			    .arguments = {
			        VulkanP0BinaryChainArgument(kernelPlan.lhs, 0, byteSize, accumulatorWorkspace),
			        VulkanP0BinaryChainArgument(kernelPlan.rhs, 1, byteSize, accumulatorWorkspace),
			        isFinalKernel ? VulkanNativeArgumentSpec{ .kind = VulkanNativeArgumentKind::OutputTensor,
			                                                 .index = 0,
			                                                 .binding = 2,
			                                                 .byteOffset = 0,
			                                                 .byteSize = byteSize }
			                      : workspacePlanner.Argument(accumulatorWorkspace, 2, byteSize),
			    },
			});
		}
		payload.workspaceTensors = workspacePlanner.TakeWorkspaceTensors();

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeSameShapeBinaryF32DAGP0(const Graph& graph)
	{
		const auto plan = MatchVulkanP0SameShapeBinaryF32DAG(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		std::vector<BinaryOp> kernelOps;
		kernelOps.reserve(plan->kernels.size());
		for (const auto& kernelPlan : plan->kernels)
		{
			payload.featureSet.AddFeature(VulkanNativeBinaryF32FeatureFlag(kernelPlan.op));
			kernelOps.push_back(kernelPlan.op);
		}
		auto spirv = VulkanNativeSameShapeBinaryF32ChainSPIRV(kernelOps, plan->elementCount);
		payload.spirv = std::move(spirv.words);

		const auto byteSize = static_cast<std::uint64_t>(plan->elementCount) * sizeof(float);
		const auto lastUse = BuildVulkanP0ScheduleLastUses(
		    plan->kernels.size(), [&](std::size_t kernelIndex, const auto& markIntermediateUse) {
			    const auto markUse = [&](VulkanP0BinaryDAGOperand operand) {
				    if (operand.kind == VulkanP0BinaryDAGOperandKind::Intermediate)
				    {
					    markIntermediateUse(operand.index);
				    }
			    };
			    markUse(plan->kernels[kernelIndex].lhs);
			    markUse(plan->kernels[kernelIndex].rhs);
		    });
		auto workspaceAllocation = AllocateVulkanP0ScheduleWorkspaces(plan->kernels.size(), plan->outputKernelIndex,
		                                                              lastUse, byteSize, alignof(float));

		for (std::size_t kernelIndex = 0; kernelIndex < plan->kernels.size(); ++kernelIndex)
		{
			const auto& kernelPlan = plan->kernels[kernelIndex];
			payload.kernels.push_back({
			    .entryPoint = VulkanNativeSameShapeBinaryF32KernelName(kernelPlan.op),
			    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->elementCount), .y = 1, .z = 1 },
			    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
			    .arguments = {
			        VulkanP0BinaryDAGArgument(kernelPlan.lhs, 0, byteSize, workspaceAllocation.workspaceByKernel),
			        VulkanP0BinaryDAGArgument(kernelPlan.rhs, 1, byteSize, workspaceAllocation.workspaceByKernel),
			        VulkanP0ScheduleOutputArgument(kernelIndex, plan->outputKernelIndex, 2, byteSize,
			                                       workspaceAllocation.workspaceByKernel),
			    },
			});
		}
		payload.workspaceTensors = std::move(workspaceAllocation.workspaceTensors);

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeSameShapeElementwiseF32DAGP0(const Graph& graph)
	{
		const auto plan = MatchVulkanP0SameShapeElementwiseF32DAG(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		std::vector<VulkanNativeElementwiseF32KernelOp> kernelOps;
		kernelOps.reserve(plan->kernels.size());
		for (const auto& kernelPlan : plan->kernels)
		{
			if (kernelPlan.kind == VulkanNativeElementwiseF32KernelKind::Unary)
			{
				payload.featureSet.AddFeature(VulkanNativeUnaryF32FeatureFlag(kernelPlan.unaryOp));
				kernelOps.push_back(
				    { .kind = VulkanNativeElementwiseF32KernelKind::Unary, .unaryOp = kernelPlan.unaryOp });
			}
			else
			{
				payload.featureSet.AddFeature(VulkanNativeBinaryF32FeatureFlag(kernelPlan.binaryOp));
				kernelOps.push_back(
				    { .kind = VulkanNativeElementwiseF32KernelKind::Binary, .binaryOp = kernelPlan.binaryOp });
			}
		}
		auto spirv = VulkanNativeSameShapeElementwiseF32DAGSPIRV(kernelOps, plan->elementCount);
		payload.spirv = std::move(spirv.words);

		const auto byteSize = static_cast<std::uint64_t>(plan->elementCount) * sizeof(float);
		const auto lastUse = BuildVulkanP0ScheduleLastUses(
		    plan->kernels.size(), [&](std::size_t kernelIndex, const auto& markIntermediateUse) {
			    const auto markUse = [&](VulkanP0ElementwiseDAGOperand operand) {
				    if (operand.kind == VulkanP0ElementwiseDAGOperandKind::Intermediate)
				    {
					    markIntermediateUse(operand.index);
				    }
			    };
			    const auto& kernelPlan = plan->kernels[kernelIndex];
			    if (kernelPlan.kind == VulkanNativeElementwiseF32KernelKind::Unary)
			    {
				    markUse(kernelPlan.input);
			    }
			    else
			    {
				    markUse(kernelPlan.lhs);
				    markUse(kernelPlan.rhs);
			    }
		    });
		auto workspaceAllocation = AllocateVulkanP0ScheduleWorkspaces(plan->kernels.size(), plan->outputKernelIndex,
		                                                              lastUse, byteSize, alignof(float));

		for (std::size_t kernelIndex = 0; kernelIndex < plan->kernels.size(); ++kernelIndex)
		{
			const auto& kernelPlan = plan->kernels[kernelIndex];
			VulkanNativeArgumentSpec outputArgument = VulkanP0ScheduleOutputArgument(
			    kernelIndex, plan->outputKernelIndex, 2, byteSize, workspaceAllocation.workspaceByKernel);

			if (kernelPlan.kind == VulkanNativeElementwiseF32KernelKind::Unary)
			{
				payload.kernels.push_back({
				    .entryPoint = VulkanNativeSameShapeUnaryF32KernelName(kernelPlan.unaryOp),
				    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->elementCount), .y = 1, .z = 1 },
				    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
				    .arguments = {
				        VulkanP0BinaryDAGArgument(kernelPlan.input, 0, byteSize, workspaceAllocation.workspaceByKernel),
				        VulkanP0BinaryDAGArgument(kernelPlan.input, 1, byteSize, workspaceAllocation.workspaceByKernel),
				        outputArgument,
				    },
				});
			}
			else
			{
				payload.kernels.push_back({
				    .entryPoint = VulkanNativeSameShapeBinaryF32KernelName(kernelPlan.binaryOp),
				    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->elementCount), .y = 1, .z = 1 },
				    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
				    .arguments = {
				        VulkanP0BinaryDAGArgument(kernelPlan.lhs, 0, byteSize, workspaceAllocation.workspaceByKernel),
				        VulkanP0BinaryDAGArgument(kernelPlan.rhs, 1, byteSize, workspaceAllocation.workspaceByKernel),
				        outputArgument,
				    },
				});
			}
		}
		payload.workspaceTensors = std::move(workspaceAllocation.workspaceTensors);

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeConv2DF32P0(const Graph& graph)
	{
		VulkanP0ExternalTensorBuilder externalBuilder;
		const auto plan = MatchVulkanP0Conv2DF32(graph, &externalBuilder);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		payload.featureSet.AddFeature(VulkanNativeFeature::Conv2DF32);
		auto spirv = VulkanNativeConv2DF32SPIRV(plan->inputShape, plan->weightShape, plan->outputShape, plan->strides,
		                                        plan->dilations, plan->lowPads, plan->highPads, plan->groupCount,
		                                        plan->bias.has_value());
		payload.spirv = std::move(spirv.words);

		std::vector<VulkanNativeArgumentSpec> arguments;
		std::uint32_t nextBinding = 0;
		const auto appendTensor = [&](const VulkanP0TensorRef& ref) {
			arguments.push_back({ .kind = ref.argumentKind,
			                      .index = ref.argumentIndex,
			                      .binding = nextBinding++,
			                      .byteOffset = 0,
			                      .byteSize = static_cast<std::uint64_t>(ref.elementCount) * sizeof(float) });
		};
		appendTensor(plan->input);
		appendTensor(plan->weight);
		if (plan->bias)
		{
			appendTensor(*plan->bias);
		}
		arguments.push_back({ .kind = VulkanNativeArgumentKind::OutputTensor,
		                      .index = 0,
		                      .binding = nextBinding,
		                      .byteOffset = 0,
		                      .byteSize = static_cast<std::uint64_t>(plan->outputElementCount) * sizeof(float) });

		payload.kernels.push_back({
		    .entryPoint = std::string(VulkanNativeConv2DF32KernelName()),
		    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->outputElementCount), .y = 1, .z = 1 },
		    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
		    .arguments = std::move(arguments),
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.constants = std::move(externalBuilder.constants),
			.weights = std::move(externalBuilder.weights),
			.externalTensorInfos = std::move(externalBuilder.externalTensorInfos),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeUpsampleNearestF32P0(const Graph& graph)
	{
		const auto plan = MatchVulkanP0UpsampleNearestF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		payload.featureSet.AddFeature(VulkanNativeFeature::UpsampleNearestF32);
		auto spirv = VulkanNativeUpsampleNearestF32SPIRV(plan->inputShape, plan->outputShape, false);
		payload.spirv = std::move(spirv.words);

		payload.kernels.push_back({
		    .entryPoint = std::string(VulkanNativeUpsampleNearestF32KernelName()),
		    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->outputElementCount), .y = 1, .z = 1 },
		    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
		    .arguments = {
		        { .kind = VulkanNativeArgumentKind::InputTensor,
		          .index = plan->inputIndex,
		          .binding = 0,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->inputElementCount) * sizeof(float) },
		        { .kind = VulkanNativeArgumentKind::OutputTensor,
		          .index = 0,
		          .binding = 1,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->outputElementCount) * sizeof(float) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeSliceF32P0(const Graph& graph)
	{
		const auto plan = MatchVulkanP0SliceF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		payload.featureSet.AddFeature(VulkanNativeFeature::SliceF32);
		auto spirv =
		    VulkanNativeSliceF32SPIRV(plan->inputShape, plan->outputShape, plan->axis, plan->start, plan->length);
		payload.spirv = std::move(spirv.words);

		payload.kernels.push_back({
		    .entryPoint = std::string(VulkanNativeSliceF32KernelName()),
		    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->outputElementCount), .y = 1, .z = 1 },
		    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
		    .arguments = {
		        { .kind = VulkanNativeArgumentKind::InputTensor,
		          .index = plan->inputIndex,
		          .binding = 0,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->inputElementCount) * sizeof(float) },
		        { .kind = VulkanNativeArgumentKind::OutputTensor,
		          .index = 0,
		          .binding = 1,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->outputElementCount) * sizeof(float) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeConcatF32P0(const Graph& graph)
	{
		const auto plan = MatchVulkanP0ConcatF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		payload.featureSet.AddFeature(VulkanNativeFeature::ConcatF32);
		auto spirv = VulkanNativeConcatF32SPIRV(plan->lhsShape, plan->rhsShape, plan->outputShape, plan->axis);
		payload.spirv = std::move(spirv.words);

		payload.kernels.push_back({
		    .entryPoint = std::string(VulkanNativeConcatF32KernelName()),
		    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->outputElementCount), .y = 1, .z = 1 },
		    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
		    .arguments = {
		        { .kind = VulkanNativeArgumentKind::InputTensor,
		          .index = plan->lhsIndex,
		          .binding = 0,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->lhsElementCount) * sizeof(float) },
		        { .kind = VulkanNativeArgumentKind::InputTensor,
		          .index = plan->rhsIndex,
		          .binding = 1,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->rhsElementCount) * sizeof(float) },
		        { .kind = VulkanNativeArgumentKind::OutputTensor,
		          .index = 0,
		          .binding = 2,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->outputElementCount) * sizeof(float) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeConvTranspose2DF32P0(const Graph& graph)
	{
		VulkanP0ExternalTensorBuilder externalBuilder;
		const auto plan = MatchVulkanP0ConvTranspose2DF32(graph, &externalBuilder);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		payload.featureSet.AddFeature(VulkanNativeFeature::ConvTranspose2DF32);
		auto spirv = VulkanNativeConvTranspose2DF32SPIRV(plan->inputShape, plan->weightShape, plan->outputShape,
		                                                 plan->strides, plan->dilations, plan->lowPads, plan->highPads,
		                                                 plan->outputPads, plan->groupCount, plan->bias.has_value());
		payload.spirv = std::move(spirv.words);

		std::vector<VulkanNativeArgumentSpec> arguments;
		std::uint32_t nextBinding = 0;
		const auto appendTensor = [&](const VulkanP0TensorRef& ref) {
			arguments.push_back({ .kind = ref.argumentKind,
			                      .index = ref.argumentIndex,
			                      .binding = nextBinding++,
			                      .byteOffset = 0,
			                      .byteSize = static_cast<std::uint64_t>(ref.elementCount) * sizeof(float) });
		};
		appendTensor(plan->input);
		appendTensor(plan->weight);
		if (plan->bias)
		{
			appendTensor(*plan->bias);
		}
		arguments.push_back({ .kind = VulkanNativeArgumentKind::OutputTensor,
		                      .index = 0,
		                      .binding = nextBinding,
		                      .byteOffset = 0,
		                      .byteSize = static_cast<std::uint64_t>(plan->outputElementCount) * sizeof(float) });

		payload.kernels.push_back({
		    .entryPoint = std::string(VulkanNativeConvTranspose2DF32KernelName()),
		    .groups = { .x = VulkanP0ElementwiseGroupCount(plan->outputElementCount), .y = 1, .z = 1 },
		    .requirements = VulkanP0KernelRequirements(kVulkanNativeElementwiseWorkgroupSize),
		    .arguments = std::move(arguments),
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.constants = std::move(externalBuilder.constants),
			.weights = std::move(externalBuilder.weights),
			.externalTensorInfos = std::move(externalBuilder.externalTensorInfos),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeMatMulF32P0(const Graph& graph)
	{
		const auto plan = MatchVulkanP0MatMulF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		payload.featureSet.AddFeature(VulkanNativeFeature::MatMulF32);
		auto spirv = VulkanNativeMatMulF32SPIRV(plan->m, plan->k, plan->n);
		payload.spirv = std::move(spirv.words);

		payload.kernels.push_back({
		    .entryPoint = "main",
		    .groups = { .x = VulkanP0MatMulGroupCount(plan->outputElementCount), .y = 1, .z = 1 },
		    .requirements = VulkanP0KernelRequirements(kVulkanNativeMatMulWorkgroupSize),
		    .arguments = {
		        { .kind = VulkanNativeArgumentKind::InputTensor,
		          .index = plan->lhsInputIndex,
		          .binding = 0,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->m) * plan->k * sizeof(float) },
		        { .kind = VulkanNativeArgumentKind::InputTensor,
		          .index = plan->rhsInputIndex,
		          .binding = 1,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->k) * plan->n * sizeof(float) },
		        { .kind = VulkanNativeArgumentKind::OutputTensor,
		          .index = 0,
		          .binding = 2,
		          .byteOffset = 0,
		          .byteSize = static_cast<std::uint64_t>(plan->outputElementCount) * sizeof(float) },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeMatMulBiasF32P0(const Graph& graph)
	{
		VulkanP0ExternalTensorBuilder externalBuilder;
		const auto plan = MatchVulkanP0MatMulBiasF32(graph, &externalBuilder);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		payload.featureSet.AddFeature(plan->relu ? VulkanNativeFeature::MatMulBiasAddReLUF32
		                                         : VulkanNativeFeature::MatMulBiasAddF32);
		auto spirv = VulkanNativeMatMulBiasF32SPIRV(plan->m, plan->k, plan->n, plan->biasRows, plan->relu);
		payload.spirv = std::move(spirv.words);

		const auto lhsByteSize = static_cast<std::uint64_t>(plan->lhs.elementCount) * sizeof(float);
		const auto rhsByteSize = static_cast<std::uint64_t>(plan->rhs.elementCount) * sizeof(float);
		const auto biasByteSize = static_cast<std::uint64_t>(plan->bias.elementCount) * sizeof(float);
		const auto outputByteSize = static_cast<std::uint64_t>(plan->outputElementCount) * sizeof(float);
		payload.kernels.push_back({
		    .entryPoint = "main",
		    .groups = { .x = VulkanP0MatMulGroupCount(plan->outputElementCount), .y = 1, .z = 1 },
		    .requirements = VulkanP0KernelRequirements(kVulkanNativeMatMulWorkgroupSize),
		    .arguments = {
		        { .kind = plan->lhs.argumentKind,
		          .index = plan->lhs.argumentIndex,
		          .binding = 0,
		          .byteOffset = 0,
		          .byteSize = lhsByteSize },
		        { .kind = plan->rhs.argumentKind,
		          .index = plan->rhs.argumentIndex,
		          .binding = 1,
		          .byteOffset = 0,
		          .byteSize = rhsByteSize },
		        { .kind = plan->bias.argumentKind,
		          .index = plan->bias.argumentIndex,
		          .binding = 2,
		          .byteOffset = 0,
		          .byteSize = biasByteSize },
		        { .kind = VulkanNativeArgumentKind::OutputTensor,
		          .index = 0,
		          .binding = 3,
		          .byteOffset = 0,
		          .byteSize = outputByteSize },
		    },
		});

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.constants = std::move(externalBuilder.constants),
			.weights = std::move(externalBuilder.weights),
			.externalTensorInfos = std::move(externalBuilder.externalTensorInfos),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeLinearChainF32P0(const Graph& graph)
	{
		VulkanP0ExternalTensorBuilder externalBuilder;
		const auto plan = MatchVulkanP0LinearChainF32(graph, &externalBuilder);
		if (!plan)
		{
			return std::nullopt;
		}

		VulkanNativeInstructionPayload payload;
		payload.featureSet.AddFeature(VulkanNativeFeature::StaticShape);
		payload.featureSet.AddFeature(VulkanNativeFeature::SingleSubgraph);
		payload.workspaceTensors = plan->workspaceTensors;

		std::vector<VulkanNativeMatMulBiasF32KernelSpec> codegenSpecs;
		codegenSpecs.reserve(plan->kernels.size());
		for (std::size_t kernelIndex = 0; kernelIndex < plan->kernels.size(); ++kernelIndex)
		{
			const auto& kernelPlan = plan->kernels[kernelIndex];
			payload.featureSet.AddFeature(kernelPlan.relu ? VulkanNativeFeature::MatMulBiasAddReLUF32
			                                              : VulkanNativeFeature::MatMulBiasAddF32);
			codegenSpecs.push_back({
			    .kernelName = std::format("matmul_bias_f32_{}", kernelIndex),
			    .m = kernelPlan.m,
			    .k = kernelPlan.k,
			    .n = kernelPlan.n,
			    .biasRows = kernelPlan.biasRows,
			    .relu = kernelPlan.relu,
			});
		}
		auto spirv = VulkanNativeMatMulBiasF32SPIRV(codegenSpecs);
		payload.spirv = std::move(spirv.words);

		for (std::size_t kernelIndex = 0; kernelIndex < plan->kernels.size(); ++kernelIndex)
		{
			const auto& kernelPlan = plan->kernels[kernelIndex];
			payload.kernels.push_back({
			    .entryPoint = codegenSpecs[kernelIndex].kernelName,
			    .groups = { .x = VulkanP0MatMulGroupCount(kernelPlan.outputElementCount), .y = 1, .z = 1 },
			    .requirements = VulkanP0KernelRequirements(kVulkanNativeMatMulWorkgroupSize),
			    .arguments = {
			        VulkanP0TensorArgument(kernelPlan.lhs, 0),
			        VulkanP0TensorArgument(kernelPlan.rhs, 1),
			        VulkanP0TensorArgument(kernelPlan.bias, 2),
			        VulkanP0TensorArgument(kernelPlan.output, 3),
			    },
			});
		}

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, llvm::sys::getDefaultTargetTriple(),
		                              CompiledModuleBackend::VulkanNative);
		auto instructions = SerializeVulkanNativeInstructionPayload(payload);
		return VulkanP0ArtifactParts{
			.rodata = std::move(rodata),
			.instructions = std::move(instructions),
			.constants = std::move(externalBuilder.constants),
			.weights = std::move(externalBuilder.weights),
			.externalTensorInfos = std::move(externalBuilder.externalTensorInfos),
			.inputSpecs = std::move(inputSpecs),
			.outputSpecs = std::move(outputSpecs),
		};
	}

	std::optional<VulkanP0ArtifactParts> TryCompileVulkanNativeP0(const Graph& graph)
	{
		if (auto nativeParts = TryCompileVulkanNativeSameShapeUnaryP0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativeSameShapeCastP0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativeReduceF32P0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativeSoftmaxF32P0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativeNormalizationF32P0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativePool2DF32P0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativeUpsampleNearestF32P0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativeSliceF32P0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativeConcatF32P0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativeConv2DF32P0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativeConvTranspose2DF32P0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativeMatMulBiasF32P0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativeLinearChainF32P0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativeMatMulF32P0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativeSameShapeBinaryF32ChainP0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativeSameShapeBinaryF32DAGP0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativeSameShapeElementwiseF32DAGP0(graph))
		{
			return std::move(nativeParts);
		}
		if (auto nativeParts = TryCompileVulkanNativeSameShapeBinaryP0(graph))
		{
			return std::move(nativeParts);
		}
		return std::nullopt;
	}

	CompiledArtifactParts MakeVulkanNativeCompiledArtifactParts(VulkanP0ArtifactParts parts)
	{
		return MakeCompiledArtifactParts(std::move(parts.rodata), std::move(parts.instructions),
		                                 std::move(parts.inputSpecs), std::move(parts.outputSpecs),
		                                 CompiledModuleBackend::VulkanNative, std::move(parts.constants),
		                                 std::move(parts.weights), std::move(parts.externalTensorInfos));
	}
#endif

	mlir::OwningOpRef<mlir::ModuleOp> BuildLoweredMLIRModule(const Graph& graph, mlir::MLIRContext& ctx,
	                                                         const CompilerOptions& options)
	{
		auto module = TimedCompileDiagnostic(options, "cpu-mlir translate graph", [&] {
			return litenn::translateExecutablePlanToMLIR(Detail::BuildExecutablePlanFromGraph(graph), ctx);
		});
		if (!module)
		{
			throw std::runtime_error("Failed to translate LiteNN executable plan to MLIR");
		}

		TimedCompileDiagnostic(options, "cpu-mlir lower LiteNN dialect", [&] {
			mlir::PassManager pm(&ctx);
			pm.addPass(litenn::createLowerLiteNNPass());
			if (mlir::failed(pm.run(*module)))
			{
				throw std::runtime_error("LiteNN dialect lowering pipeline failed");
			}
		});
		TimedCompileDiagnostic(options, "cpu-mlir bufferize", [&] {
			mlir::PassManager pm(&ctx);
			litenn::addBufferizationPipeline(pm);
			if (mlir::failed(pm.run(*module)))
			{
				throw std::runtime_error("LiteNN bufferization pipeline failed");
			}
		});
		TimedCompileDiagnostic(options, "cpu-mlir lower LLVM dialect", [&] {
			mlir::PassManager pm(&ctx);
			litenn::addLLVMCodegenPipeline(pm);
			if (mlir::failed(pm.run(*module)))
			{
				throw std::runtime_error("LiteNN LLVM codegen pipeline failed");
			}
		});
		if (mlir::failed(mlir::verify(*module)))
		{
			throw std::runtime_error("LiteNN lowered MLIR module verification failed");
		}
		return module;
	}

	void SetupCompilerMLIRContext(mlir::MLIRContext& ctx)
	{
		ctx.disableMultithreading();

		mlir::DialectRegistry registry;
		litenn::registerBufferizationModels(registry);
		litenn::registerLLVMTranslations(registry);

		ctx.appendDialectRegistry(registry);
		ctx.loadDialect<litenn::LiteNNDialect, mlir::arith::ArithDialect, mlir::bufferization::BufferizationDialect,
		                mlir::cf::ControlFlowDialect, mlir::func::FuncDialect, mlir::linalg::LinalgDialect,
		                mlir::LLVM::LLVMDialect, mlir::math::MathDialect, mlir::memref::MemRefDialect,
		                mlir::scf::SCFDialect, mlir::tensor::TensorDialect, mlir::vector::VectorDialect>();
	}

	std::optional<CompiledArtifactParts> TryCompileCPUMLIRExternalRegions(const Graph& graph,
	                                                                      const CompilerOptions& options)
	{
		if (!IsCPUExternalRegionsEnabled(options))
		{
			return std::nullopt;
		}

		auto externalized = TimedCompileDiagnostic(options, "cpu-mlir externalize graph",
		                                           [&] { return BuildCPUMLIRExternalizedGraph(graph, options); });
		if (!externalized)
		{
			LogCompileDiagnostic(options, "cpu-mlir externalize graph: unsupported, falling back");
			return std::nullopt;
		}
		LogCompileDiagnostic(options,
		                     std::format("cpu-mlir external regions: constants={} bytes weights={} bytes tensors={}",
		                                 externalized->constants.size(), externalized->weights.size(),
		                                 externalized->externalTensorInfos.size()));

		mlir::MLIRContext ctx;
		TimedCompileDiagnostic(options, "cpu-mlir setup context", [&] { SetupCompilerMLIRContext(ctx); });
		auto mlirModule = BuildLoweredMLIRModule(externalized->graph, ctx, options);

		llvm::LLVMContext llvmCtx;
		auto llvmModule = TimedCompileDiagnostic(options, "cpu-mlir translate to LLVM IR",
		                                         [&] { return litenn::translateToLLVMIR(*mlirModule, llvmCtx); });
		if (!llvmModule)
		{
			throw std::runtime_error("Failed to translate externalized LiteNN MLIR module to LLVM IR");
		}

		auto config = TimedCompileDiagnostic(options, "cpu-aot create target machine",
		                                     [&] { return CreateNativeTargetMachine(); });
		TimedCompileDiagnostic(options, "cpu-aot configure module",
		                       [&] { ConfigureForNativeObject(*llvmModule, config); });

		auto inputSpecs =
		    TimedCompileDiagnostic(options, "cpu-aot build input specs", [&] { return BuildInputSpecs(graph); });
		auto outputSpecs =
		    TimedCompileDiagnostic(options, "cpu-aot build output specs", [&] { return BuildOutputSpecs(graph); });
		TimedCompileDiagnostic(options, "cpu-aot add uniform entry wrapper", [&] {
			AddUniformEntryWrapper(*llvmModule, "subgraph_" + std::to_string(graph.Forward()), inputSpecs, outputSpecs,
			                       externalized->entryExternalTensorInfos);
		});
		TimedCompileDiagnostic(
		    options, std::format("cpu-aot optimize LLVM module O{}", options.cpuAOTLLVMOptLevel),
		    [&] { OptimizeLLVMModule(*llvmModule, *config.targetMachine, options.cpuAOTLLVMOptLevel); });

		auto rodata = TimedCompileDiagnostic(options, "cpu-aot serialize rodata", [&] {
			return SerializeRodata(inputSpecs, outputSpecs, config.triple, CompiledModuleBackend::CPUNative);
		});
		auto instructions =
		    TimedCompileDiagnostic(options, "cpu-aot emit object file", [&] { return EmitObjectFile(*llvmModule); });
		return CompiledArtifactParts{ std::move(rodata),
			                          std::move(instructions),
			                          std::move(externalized->constants),
			                          std::move(externalized->weights),
			                          std::move(externalized->externalTensorInfos),
			                          std::move(inputSpecs),
			                          std::move(outputSpecs) };
	}
} // namespace

struct CompiledModule<CPU>::Impl
{
	std::vector<std::byte> rodata;
	std::vector<std::byte> instructions;
	std::vector<std::byte> externalConstants;
	std::vector<std::byte> externalWeights;
	std::span<const std::byte> borrowedExternalConstants;
	std::span<const std::byte> borrowedExternalWeights;
	std::vector<CompiledTensorSpec> inputSpecs;
	std::vector<CompiledTensorSpec> outputSpecs;
	CompiledModuleBackend backend{ CompiledModuleBackend::CPUNative };
	std::unique_ptr<llvm::LLVMContext> jitContext;
	std::unique_ptr<llvm::ExecutionEngine> jit;
	EntryFn entry{};

	const void* ExternalConstantsData() const
	{
		if (!externalConstants.empty())
		{
			return externalConstants.data();
		}
		return borrowedExternalConstants.empty() ? nullptr : borrowedExternalConstants.data();
	}

	const void* ExternalWeightsData() const
	{
		if (!externalWeights.empty())
		{
			return externalWeights.data();
		}
		return borrowedExternalWeights.empty() ? nullptr : borrowedExternalWeights.data();
	}
};

CompiledModule<CPU>::CompiledModule() = default;

CompilerOptions CompilerOptions::Defaults()
{
	return {};
}

CompileBudgetEstimate LiteNN::EstimateCompileBudget(const ExecutablePlan& plan, const CompilerOptions& options)
{
	CompileBudgetEstimate estimate;
	estimate.subgraphCount = plan.subgraphs.size();
	estimate.variableCount = plan.variables.size();
	estimate.cpuAOTExternalRegionsEnabled = options.enableCPUAOTExternalRegions;

	for (const auto& variable : plan.variables)
	{
		const auto byteSize = static_cast<std::uint64_t>(variable.LogicalByteSize().value_or(variable.region.byteSize));
		estimate.variablePayloadBytes = SaturatedAddU64(estimate.variablePayloadBytes, byteSize);
		if (options.enableCPUAOTExternalRegions)
		{
			estimate.projectedExternalWeightBytes = SaturatedAddU64(estimate.projectedExternalWeightBytes, byteSize);
		}
		else
		{
			estimate.projectedInlineMLIRPayloadBytes =
			    SaturatedAddU64(estimate.projectedInlineMLIRPayloadBytes, byteSize);
		}
	}

	for (const auto& subgraph : plan.subgraphs)
	{
		estimate.nodeCount += subgraph.nodes.size();
		for (const auto& entry : subgraph.nodes)
		{
			std::visit(
			    [&](const auto& node) {
				    using T = std::decay_t<decltype(node)>;
				    if constexpr (std::same_as<T, VariableRefNode>)
				    {
					    ++estimate.variableRefNodeCount;
				    }
				    else if constexpr (std::same_as<T, ConstantNode>)
				    {
					    ++estimate.constantNodeCount;
					    const auto byteSize = static_cast<std::uint64_t>(node.value.NumElements()) *
					                          LiteNN::ElementByteSize(node.value.DType());
					    estimate.constantPayloadBytes = SaturatedAddU64(estimate.constantPayloadBytes, byteSize);
					    const bool externalized = options.enableCPUAOTExternalRegions &&
					                              CanExternalizeCPUTensorInMLIR(node.value.DType()) &&
					                              byteSize >= options.cpuAOTExternalConstantMinBytes;
					    if (externalized)
					    {
						    estimate.projectedExternalConstantBytes =
						        SaturatedAddU64(estimate.projectedExternalConstantBytes, byteSize);
					    }
					    else
					    {
						    estimate.projectedInlineMLIRPayloadBytes =
						        SaturatedAddU64(estimate.projectedInlineMLIRPayloadBytes, byteSize);
					    }
				    }
				    else if constexpr (std::same_as<T, QuantizedConstantNode>)
				    {
					    ++estimate.quantizedConstantNodeCount;
					    const auto byteSize = static_cast<std::uint64_t>(node.storage.NumElements()) *
					                          LiteNN::ElementByteSize(node.storage.DType());
					    estimate.quantizedConstantPayloadBytes =
					        SaturatedAddU64(estimate.quantizedConstantPayloadBytes, byteSize);
					    estimate.projectedInlineMLIRPayloadBytes =
					        SaturatedAddU64(estimate.projectedInlineMLIRPayloadBytes, byteSize);
				    }
			    },
			    entry.node);
		}
	}
	return estimate;
}

CompileBudgetEstimate LiteNN::EstimateCompileBudget(const Graph& graph, const CompilerOptions& options)
{
	return EstimateCompileBudget(Detail::BuildExecutablePlanFromGraph(graph), options);
}

CompiledModuleSeparatedArtifact::CompiledModuleSeparatedArtifact(
    std::vector<std::byte> metadata, std::vector<std::byte> constants, std::vector<std::byte> weights,
    std::vector<std::byte> instructions, std::vector<CompiledTensorSpec> inputSpecs,
    std::vector<CompiledTensorSpec> outputSpecs, CompiledModuleBackend backend)
    : metadata_(std::move(metadata)), constants_(std::move(constants)), weights_(std::move(weights)),
      instructions_(std::move(instructions)), inputSpecs_(std::move(inputSpecs)), outputSpecs_(std::move(outputSpecs)),
      backend_(backend)
{
}

CompiledModuleSeparatedArtifact CompiledModuleSeparatedArtifact::CopyFromImage(CompiledModuleSeparatedImage image)
{
	auto separatedMetadata = ValidateSeparatedImage(image);
	return CompiledModuleSeparatedArtifact(
	    ToByteVector(image.metadata, kMetadataRegionName), ToByteVector(image.constants, kConstantsRegionName),
	    ToByteVector(image.weights, kWeightsRegionName), ToByteVector(image.instructions, kInstructionsRegionName),
	    std::move(separatedMetadata.legacyMetadata.inputSpecs), std::move(separatedMetadata.legacyMetadata.outputSpecs),
	    separatedMetadata.legacyMetadata.backend);
}

CompiledModuleSeparatedArtifact CompiledModuleSeparatedArtifact::FromOwnedRegions(std::vector<std::byte> metadata,
                                                                                  std::vector<std::byte> constants,
                                                                                  std::vector<std::byte> weights,
                                                                                  std::vector<std::byte> instructions)
{
	auto separatedMetadata = ValidateSeparatedImage({
	    .metadata = { .data = metadata.data(), .size = metadata.size() },
	    .constants = { .data = constants.data(), .size = constants.size() },
	    .weights = { .data = weights.data(), .size = weights.size() },
	    .instructions = { .data = instructions.data(), .size = instructions.size() },
	});
	return CompiledModuleSeparatedArtifact(
	    std::move(metadata), std::move(constants), std::move(weights), std::move(instructions),
	    std::move(separatedMetadata.legacyMetadata.inputSpecs), std::move(separatedMetadata.legacyMetadata.outputSpecs),
	    separatedMetadata.legacyMetadata.backend);
}

CompiledModuleSeparatedArtifact
CompiledModuleSeparatedArtifact::FromExportedSymbols(CompiledModuleSeparatedExportedSymbols symbols)
{
	return CopyFromImage({
	    .metadata = {
	        .data = symbols.metadata,
	        .size = ReadExportedSymbolSize(symbols.metadataSize, "metadata_size"),
	    },
	    .constants = {
	        .data = symbols.constants,
	        .size = ReadExportedSymbolSize(symbols.constantsSize, "constants_size"),
	    },
	    .weights = {
	        .data = symbols.weights,
	        .size = ReadExportedSymbolSize(symbols.weightsSize, "weights_size"),
	    },
	    .instructions = {
	        .data = symbols.instructions,
	        .size = ReadExportedSymbolSize(symbols.instructionsSize, "instructions_size"),
	    },
	});
}

CompiledModule<CPU> CompiledModuleSeparatedArtifact::Load() const
{
	return CompiledModule<CPU>::Load(Image());
}

CompiledModule<CPU> CompiledModuleSeparatedArtifact::LoadBorrowedExternalRegions() const
{
	return CompiledModule<CPU>::LoadBorrowedExternalRegions(Image());
}

#ifdef LITENN_ENABLE_CUDA
CompiledModule<CUDA> CompiledModuleSeparatedArtifact::Load(CUDA device) const
{
	return CompiledModule<CUDA>::Load(Image(), std::move(device));
}

CompiledModule<CUDA> CompiledModuleSeparatedArtifact::LoadBorrowedExternalRegions(CUDA device) const
{
	return CompiledModule<CUDA>::LoadBorrowedExternalRegions(Image(), std::move(device));
}
#endif

CompiledModuleSeparatedArtifact
CompiledModuleSeparatedArtifact::WithReboundConstants(CompiledModuleRegion constants) const
{
	auto metadata = DeserializeSeparatedMetadata(metadata_);
	ValidateSeparatedRegion(constants, FindRegionInfo(metadata.regions, kConstantsRegionName));
	return CompiledModuleSeparatedArtifact(metadata_, ToByteVector(constants, kConstantsRegionName), weights_,
	                                       instructions_, inputSpecs_, outputSpecs_, backend_);
}

CompiledModuleSeparatedArtifact CompiledModuleSeparatedArtifact::WithReboundWeights(CompiledModuleRegion weights) const
{
	auto metadata = DeserializeSeparatedMetadata(metadata_);
	ValidateSeparatedRegion(weights, FindRegionInfo(metadata.regions, kWeightsRegionName));
	return CompiledModuleSeparatedArtifact(metadata_, constants_, ToByteVector(weights, kWeightsRegionName),
	                                       instructions_, inputSpecs_, outputSpecs_, backend_);
}

CompiledModuleSeparatedImage CompiledModuleSeparatedArtifact::Image() const
{
	return {
		.metadata = {
		    .data = metadata_.data(),
		    .size = metadata_.size(),
		},
		.constants = {
		    .data = constants_.data(),
		    .size = constants_.size(),
		},
		.weights = {
		    .data = weights_.data(),
		    .size = weights_.size(),
		},
		.instructions = {
		    .data = instructions_.data(),
		    .size = instructions_.size(),
		},
	};
}

std::span<const std::byte> CompiledModuleSeparatedArtifact::Metadata() const
{
	return metadata_;
}

std::span<const std::byte> CompiledModuleSeparatedArtifact::Constants() const
{
	return constants_;
}

std::span<const std::byte> CompiledModuleSeparatedArtifact::Weights() const
{
	return weights_;
}

std::span<const std::byte> CompiledModuleSeparatedArtifact::Instructions() const
{
	return instructions_;
}

std::vector<CompiledModuleRegionInfo> CompiledModuleSeparatedArtifact::RegionInfos() const
{
	auto infos = std::vector<CompiledModuleRegionInfo>{ MakeRegionInfo(kMetadataRegionName, metadata_) };
	auto metadata = DeserializeSeparatedMetadata(metadata_);
	infos.insert(infos.end(), metadata.regions.begin(), metadata.regions.end());
	return infos;
}

std::vector<CompiledModuleExternalTensorInfo> CompiledModuleSeparatedArtifact::ExternalTensorInfos() const
{
	return DeserializeSeparatedMetadata(metadata_).externalTensorInfos;
}

std::span<const CompiledTensorSpec> CompiledModuleSeparatedArtifact::InputSpecs() const
{
	return inputSpecs_;
}

std::span<const CompiledTensorSpec> CompiledModuleSeparatedArtifact::OutputSpecs() const
{
	return outputSpecs_;
}

CompiledModuleBackend CompiledModuleSeparatedArtifact::Backend() const
{
	return backend_;
}

std::optional<std::size_t> CompiledModuleSeparatedArtifact::FindInput(std::string_view name) const
{
	return FindSpecIndex(inputSpecs_, name);
}

std::optional<std::size_t> CompiledModuleSeparatedArtifact::FindOutput(std::string_view name) const
{
	return FindSpecIndex(outputSpecs_, name);
}

void CompiledModuleSeparatedArtifact::WriteObjectFile(const std::filesystem::path& path,
                                                      std::string_view symbolPrefix) const
{
	const auto objectBytes = EmitSeparatedCarrierObject(metadata_, constants_, weights_, instructions_, symbolPrefix);
	WriteAllBytes(path, objectBytes);
}

void CompiledModuleSeparatedArtifact::WriteObjectFiles(const std::filesystem::path& directory,
                                                       std::string_view symbolPrefix) const
{
	std::filesystem::create_directories(directory);
	const auto prefix = std::string(symbolPrefix);
	WriteAllBytes(directory / (prefix + "_metadata.o"),
	              EmitSingleRegionCarrierObject(metadata_, symbolPrefix, kMetadataRegionName, ".litenn_metadata"));
	WriteAllBytes(directory / (prefix + "_constants.o"),
	              EmitSingleRegionCarrierObject(constants_, symbolPrefix, kConstantsRegionName, ".litenn_constants"));
	WriteAllBytes(directory / (prefix + "_weights.o"),
	              EmitSingleRegionCarrierObject(weights_, symbolPrefix, kWeightsRegionName, ".litenn_weights"));
	WriteAllBytes(
	    directory / (prefix + "_instructions.o"),
	    EmitSingleRegionCarrierObject(instructions_, symbolPrefix, kInstructionsRegionName, ".litenn_instructions"));
}

void CompiledModuleSeparatedArtifact::WriteRegionFiles(const std::filesystem::path& directory,
                                                       std::string_view filePrefix) const
{
	std::filesystem::create_directories(directory);
	const auto prefix = std::string(filePrefix);
	WriteAllBytes(directory / (prefix + ".metadata.bin"), metadata_);
	WriteAllBytes(directory / (prefix + ".constants.bin"), constants_);
	WriteAllBytes(directory / (prefix + ".weights.bin"), weights_);
	WriteAllBytes(directory / (prefix + ".instructions.bin"), instructions_);
}

CompiledModuleArtifact::CompiledModuleArtifact(std::vector<std::byte> rodata, std::vector<std::byte> instructions,
                                               std::vector<CompiledTensorSpec> inputSpecs,
                                               std::vector<CompiledTensorSpec> outputSpecs,
                                               CompiledModuleBackend backend, std::vector<std::byte> constants,
                                               std::vector<std::byte> weights,
                                               std::vector<CompiledModuleExternalTensorInfo> externalTensorInfos)
    : rodata_(std::move(rodata)), instructions_(std::move(instructions)), constants_(std::move(constants)),
      weights_(std::move(weights)), externalTensorInfos_(std::move(externalTensorInfos)),
      inputSpecs_(std::move(inputSpecs)), outputSpecs_(std::move(outputSpecs)), backend_(backend)
{
}

CompiledModuleArtifact CompiledModuleArtifact::CopyFromImage(CompiledModuleImage image)
{
	auto rodata = ToByteVector(image.rodata, image.rodataSize);
	auto instructions = ToByteVector(image.instructions, image.instructionSize);
	auto metadata = DeserializeRodata(rodata);
	return CompiledModuleArtifact(std::move(rodata), std::move(instructions), std::move(metadata.inputSpecs),
	                              std::move(metadata.outputSpecs), metadata.backend);
}

CompiledModuleArtifact CompiledModuleArtifact::FromExportedSymbols(CompiledModuleExportedSymbols symbols)
{
	return CopyFromImage({
	    .rodata = symbols.rodata,
	    .rodataSize = ReadExportedSymbolSize(symbols.rodataSize, "rodata_size"),
	    .instructions = symbols.instructions,
	    .instructionSize = ReadExportedSymbolSize(symbols.instructionSize, "instructions_size"),
	});
}

CompiledModuleSeparatedArtifact CompiledModuleArtifact::SeparateRodata() const
{
	std::vector<std::byte> constants(constants_.begin(), constants_.end());
	std::vector<std::byte> weights(weights_.begin(), weights_.end());
	std::vector<std::byte> instructions(instructions_.begin(), instructions_.end());

	if (backend_ == CompiledModuleBackend::CUDANative)
	{
		auto payload = DeserializeCUDANativeInstructionPayload(instructions_);
		if (constants.empty())
		{
			constants = std::move(payload.constantData);
		}
		else
		{
			constants.insert(constants.end(), payload.constantData.begin(), payload.constantData.end());
		}
		payload.constantData.clear();
		instructions = SerializeCUDANativeInstructionPayload(payload);
	}

	auto metadata = SerializeSeparatedMetadata(rodata_, constants, weights, instructions, externalTensorInfos_);
	return CompiledModuleSeparatedArtifact(std::move(metadata), std::move(constants), std::move(weights),
	                                       std::move(instructions), inputSpecs_, outputSpecs_, backend_);
}

CompiledModule<CPU> CompiledModuleArtifact::Load() const
{
	auto module = CompiledModule<CPU>::Load(Image());
	if (!constants_.empty())
	{
		module.impl_->externalConstants = constants_;
	}
	if (!weights_.empty())
	{
		module.impl_->externalWeights = weights_;
	}
	return module;
}

CompiledModule<CPU> CompiledModuleArtifact::Load() &&
{
	auto module = CompiledModule<CPU>::Load(Image());
	module.impl_->externalConstants = std::move(constants_);
	module.impl_->externalWeights = std::move(weights_);
	return module;
}

CompiledModuleImage CompiledModuleArtifact::Image() const
{
	return {
		.rodata = rodata_.data(),
		.rodataSize = rodata_.size(),
		.instructions = instructions_.data(),
		.instructionSize = instructions_.size(),
	};
}

std::span<const std::byte> CompiledModuleArtifact::Rodata() const
{
	return rodata_;
}

std::span<const std::byte> CompiledModuleArtifact::Instructions() const
{
	return instructions_;
}

std::span<const std::byte> CompiledModuleArtifact::Constants() const
{
	return constants_;
}

std::span<const std::byte> CompiledModuleArtifact::Weights() const
{
	return weights_;
}

std::span<const CompiledTensorSpec> CompiledModuleArtifact::InputSpecs() const
{
	return inputSpecs_;
}

std::span<const CompiledTensorSpec> CompiledModuleArtifact::OutputSpecs() const
{
	return outputSpecs_;
}

CompiledModuleBackend CompiledModuleArtifact::Backend() const
{
	return backend_;
}

std::span<const CompiledModuleExternalTensorInfo> CompiledModuleArtifact::ExternalTensorInfos() const
{
	return externalTensorInfos_;
}

std::optional<std::size_t> CompiledModuleArtifact::FindInput(std::string_view name) const
{
	return FindSpecIndex(inputSpecs_, name);
}

std::optional<std::size_t> CompiledModuleArtifact::FindOutput(std::string_view name) const
{
	return FindSpecIndex(outputSpecs_, name);
}

std::vector<std::byte> CompiledModuleArtifact::BuildSeparatedMetadata() const
{
	return SerializeSeparatedMetadata(rodata_, constants_, weights_, instructions_, externalTensorInfos_);
}

void CompiledModuleArtifact::WriteObjectFile(const std::filesystem::path& path, std::string_view symbolPrefix) const
{
	const auto objectBytes = EmitCarrierObject(rodata_, instructions_, symbolPrefix);
	WriteAllBytes(path, objectBytes);
}
CompiledModule<CPU>::CompiledModule(const CompiledModule&) = default;
CompiledModule<CPU>::CompiledModule(CompiledModule&&) noexcept = default;
CompiledModule<CPU>& CompiledModule<CPU>::operator=(const CompiledModule&) = default;
CompiledModule<CPU>& CompiledModule<CPU>::operator=(CompiledModule&&) noexcept = default;
CompiledModule<CPU>::~CompiledModule() = default;

CompiledModule<CPU>::CompiledModule(std::shared_ptr<Impl> impl) : impl_(std::move(impl))
{
}

CompiledModule<CPU> CompiledModule<CPU>::Load(CompiledModuleImage image)
{
	auto impl = std::make_shared<Impl>();
	impl->rodata = ToByteVector(image.rodata, image.rodataSize);
	impl->instructions = ToByteVector(image.instructions, image.instructionSize);

	auto metadata = DeserializeRodata(impl->rodata);
	if (metadata.backend != CompiledModuleBackend::CPUNative)
	{
		throw std::runtime_error("CompiledModule<CPU> can only load CPU native compiled module images");
	}
	impl->backend = metadata.backend;
	impl->inputSpecs = std::move(metadata.inputSpecs);
	impl->outputSpecs = std::move(metadata.outputSpecs);
	auto loadedJit = LoadJIT(impl->instructions);
	impl->jitContext = std::move(loadedJit.context);
	impl->jit = std::move(loadedJit.engine);
	impl->entry = loadedJit.entry;
	return CompiledModule(std::move(impl));
}

CompiledModule<CPU> CompiledModule<CPU>::Load(CompiledModuleSeparatedImage image)
{
	auto metadata = ValidateSeparatedImage(image);
	auto constants = RegionBytes(image.constants, kConstantsRegionName);
	auto weights = RegionBytes(image.weights, kWeightsRegionName);
	auto instructions = RestoreLegacyInstructionsFromSeparated(
	    metadata.legacyMetadata.backend, RegionBytes(image.instructions, kInstructionsRegionName), constants);
	auto module = Load({
	    .rodata = metadata.legacyRodata.data(),
	    .rodataSize = metadata.legacyRodata.size(),
	    .instructions = instructions.data(),
	    .instructionSize = instructions.size(),
	});
	module.impl_->externalConstants.assign(constants.begin(), constants.end());
	module.impl_->externalWeights.assign(weights.begin(), weights.end());
	return module;
}

CompiledModule<CPU> CompiledModule<CPU>::LoadBorrowedExternalRegions(CompiledModuleSeparatedImage image)
{
	auto metadata = ValidateSeparatedImage(image);
	auto constants = RegionBytes(image.constants, kConstantsRegionName);
	auto weights = RegionBytes(image.weights, kWeightsRegionName);
	auto instructions = RestoreLegacyInstructionsFromSeparated(
	    metadata.legacyMetadata.backend, RegionBytes(image.instructions, kInstructionsRegionName), constants);
	auto module = Load({
	    .rodata = metadata.legacyRodata.data(),
	    .rodataSize = metadata.legacyRodata.size(),
	    .instructions = instructions.data(),
	    .instructionSize = instructions.size(),
	});
	module.impl_->borrowedExternalConstants = constants;
	module.impl_->borrowedExternalWeights = weights;
	return module;
}

std::vector<Tensor<CPU>> CompiledModule<CPU>::RunTensors(std::span<const Tensor<CPU>> inputs) const
{
	if (!impl_ || !impl_->entry)
	{
		throw std::runtime_error("CompiledModule is empty");
	}
	std::vector<Tensor<CPU>> outputs;
	outputs.reserve(impl_->outputSpecs.size());
	for (const auto& spec : impl_->outputSpecs)
	{
		outputs.emplace_back(Uninitialized, ShapeView{ spec.type.StaticShape() }, spec.type.dtype, CPU{});
	}
	RunTensorsInto(inputs, outputs);
	return outputs;
}

void CompiledModule<CPU>::RunTensorsInto(std::span<const Tensor<CPU>> inputs, std::span<Tensor<CPU>> outputs) const
{
	llvm::SmallVector<CompiledTensorBinding, 8> inputBindings;
	inputBindings.reserve(inputs.size());
	for (std::size_t i = 0; i < inputs.size(); ++i)
	{
		const auto name = i < InputSpecs().size() ? InputSpecs()[i].name : std::string{};
		inputBindings.push_back(MakeBindingFromTensor(inputs[i], name));
	}

	llvm::SmallVector<CompiledTensorBinding, 8> outputBindings;
	outputBindings.reserve(outputs.size());
	for (std::size_t i = 0; i < outputs.size(); ++i)
	{
		const auto name = i < OutputSpecs().size() ? OutputSpecs()[i].name : std::string{};
		outputBindings.push_back(MakeBindingFromTensor(outputs[i], name));
	}

	RunIntoBindings(inputBindings, outputBindings);
}

void CompiledModule<CPU>::RunIntoBindings(std::span<const CompiledTensorBinding> inputs,
                                          std::span<const CompiledTensorBinding> outputs) const
{
	if (!impl_ || !impl_->entry)
	{
		throw std::runtime_error("CompiledModule is empty");
	}
	if (inputs.size() != impl_->inputSpecs.size())
	{
		throw std::runtime_error(std::format("CompiledModule input count mismatch: expected {}, got {}",
		                                     impl_->inputSpecs.size(), inputs.size()));
	}
	if (outputs.size() != impl_->outputSpecs.size())
	{
		throw std::runtime_error(std::format("CompiledModule output count mismatch: expected {}, got {}",
		                                     impl_->outputSpecs.size(), outputs.size()));
	}

	llvm::SmallVector<void*, 8> inputPtrs;
	inputPtrs.reserve(inputs.size());
	for (std::size_t i = 0; i < inputs.size(); ++i)
	{
		ValidateBindingAgainstSpec(inputs[i], impl_->inputSpecs[i], i, "input");
		inputPtrs.push_back(inputs[i].data);
	}

	llvm::SmallVector<void*, 8> outputPtrs;
	outputPtrs.reserve(outputs.size());
	for (std::size_t i = 0; i < outputs.size(); ++i)
	{
		ValidateBindingAgainstSpec(outputs[i], impl_->outputSpecs[i], i, "output");
		outputPtrs.push_back(outputs[i].data);
	}

	ScopedCPUExternalRegions scopedRegions(impl_->ExternalConstantsData(), impl_->ExternalWeightsData());
	impl_->entry(inputPtrs.data(), outputPtrs.data());
}

void CompiledModule<CPU>::RunManyTensorsInto(std::span<const CompiledModuleTensorInvocation> invocations,
                                             std::size_t threadCount) const
{
	std::vector<CompiledModuleBindingInvocation> bindingInvocations;
	bindingInvocations.reserve(invocations.size());
	std::vector<llvm::SmallVector<CompiledTensorBinding, 8>> inputBindings(invocations.size());
	std::vector<llvm::SmallVector<CompiledTensorBinding, 8>> outputBindings(invocations.size());
	for (std::size_t invocationIndex = 0; invocationIndex < invocations.size(); ++invocationIndex)
	{
		const auto& invocation = invocations[invocationIndex];
		auto& inputs = inputBindings[invocationIndex];
		inputs.reserve(invocation.inputs.size());
		for (std::size_t i = 0; i < invocation.inputs.size(); ++i)
		{
			const auto name = i < InputSpecs().size() ? InputSpecs()[i].name : std::string{};
			inputs.push_back(MakeBindingFromTensor(invocation.inputs[i], name));
		}

		auto& outputs = outputBindings[invocationIndex];
		outputs.reserve(invocation.outputs.size());
		for (std::size_t i = 0; i < invocation.outputs.size(); ++i)
		{
			const auto name = i < OutputSpecs().size() ? OutputSpecs()[i].name : std::string{};
			outputs.push_back(MakeBindingFromTensor(invocation.outputs[i], name));
		}
		bindingInvocations.push_back({ .inputs = inputs, .outputs = outputs });
	}
	RunManyIntoBindings(bindingInvocations, threadCount);
}

void CompiledModule<CPU>::RunManyIntoBindings(std::span<const CompiledModuleBindingInvocation> invocations,
                                              std::size_t threadCount) const
{
	const auto workerCount = NormalizeThreadCount(threadCount, invocations.size());
	if (workerCount == 0)
	{
		return;
	}
	if (workerCount == 1)
	{
		for (const auto& invocation : invocations)
		{
			RunIntoBindings(invocation.inputs, invocation.outputs);
		}
		return;
	}

	std::atomic<std::size_t> next{ 0 };
	std::atomic_bool stop{ false };
	std::exception_ptr firstError;
	std::mutex errorMutex;

	auto worker = [&] {
		while (!stop.load(std::memory_order_relaxed))
		{
			const auto index = next.fetch_add(1, std::memory_order_relaxed);
			if (index >= invocations.size())
			{
				break;
			}

			try
			{
				const auto& invocation = invocations[index];
				RunIntoBindings(invocation.inputs, invocation.outputs);
			}
			catch (...)
			{
				{
					std::lock_guard lock(errorMutex);
					if (!firstError)
					{
						firstError = std::current_exception();
					}
				}
				stop.store(true, std::memory_order_relaxed);
				break;
			}
		}
	};

	std::vector<std::thread> workers;
	workers.reserve(workerCount);
	for (std::size_t i = 0; i < workerCount; ++i)
	{
		workers.emplace_back(worker);
	}
	for (auto& thread : workers)
	{
		thread.join();
	}

	if (firstError)
	{
		std::rethrow_exception(firstError);
	}
}

CompiledModuleImage CompiledModule<CPU>::Image() const
{
	if (!impl_)
	{
		return {};
	}
	return {
		.rodata = impl_->rodata.data(),
		.rodataSize = impl_->rodata.size(),
		.instructions = impl_->instructions.data(),
		.instructionSize = impl_->instructions.size(),
	};
}

std::span<const std::byte> CompiledModule<CPU>::Rodata() const
{
	if (!impl_)
	{
		return {};
	}
	return impl_->rodata;
}

std::span<const std::byte> CompiledModule<CPU>::Instructions() const
{
	if (!impl_)
	{
		return {};
	}
	return impl_->instructions;
}

std::span<const CompiledTensorSpec> CompiledModule<CPU>::InputSpecs() const
{
	if (!impl_)
	{
		return {};
	}
	return impl_->inputSpecs;
}

std::span<const CompiledTensorSpec> CompiledModule<CPU>::OutputSpecs() const
{
	if (!impl_)
	{
		return {};
	}
	return impl_->outputSpecs;
}

CompiledModuleBackend CompiledModule<CPU>::Backend() const
{
	return impl_ ? impl_->backend : CompiledModuleBackend::CPUNative;
}

std::optional<std::size_t> CompiledModule<CPU>::FindInput(std::string_view name) const
{
	if (!impl_)
	{
		return std::nullopt;
	}
	return FindSpecIndex(impl_->inputSpecs, name);
}

std::optional<std::size_t> CompiledModule<CPU>::FindOutput(std::string_view name) const
{
	if (!impl_)
	{
		return std::nullopt;
	}
	return FindSpecIndex(impl_->outputSpecs, name);
}

void CompiledModule<CPU>::WriteObjectFile(const std::filesystem::path& path, std::string_view symbolPrefix) const
{
	if (!impl_)
	{
		throw std::runtime_error("CompiledModule is empty");
	}
	const auto objectBytes = EmitCarrierObject(impl_->rodata, impl_->instructions, symbolPrefix);
	WriteAllBytes(path, objectBytes);
}

#ifdef LITENN_ENABLE_CUDA
namespace
{
	void RequireCUDACPUBridgeAllowed(const CUDA& device, std::string_view operation)
	{
		if (device.hostFallbackPolicy != CUDAHostFallbackPolicy::Allow)
		{
			throw std::runtime_error(std::format(
			    "CompiledModule<CUDA> CPU bridge for {} is disabled; load with CUDAHostFallbackPolicy::Allow or lower "
			    "the runtime schedule to a native CUDA artifact with explicit fallback steps",
			    operation));
		}
	}
} // namespace

struct CompiledModule<CUDA>::Impl
{
	std::vector<std::byte> rodata;
	std::vector<std::byte> instructions;
	std::vector<CompiledTensorSpec> inputSpecs;
	std::vector<CompiledTensorSpec> outputSpecs;
	CompiledModuleBackend backend{ CompiledModuleBackend::CPUNative };
	CompiledModule<CPU> cpuModule;
	CUDA device;
	CUDANativeInstructionPayload cudaPayload;
	CUDADriverModule cudaModule;
	std::optional<CUDANativeWorkspaceBuffer> cudaWorkspace;
	std::optional<CUDANativeConstantBuffer> cudaConstants;
	mutable std::mutex cudaWorkspaceMutex;
	mutable std::mutex cudaGraphReplayMutex;
	mutable CUDAGraphReplayCache cudaGraphReplayCache;
};

CompiledModule<CUDA>::CompiledModule() = default;
CompiledModule<CUDA>::CompiledModule(const CompiledModule&) = default;
CompiledModule<CUDA>::CompiledModule(CompiledModule&&) noexcept = default;
CompiledModule<CUDA>& CompiledModule<CUDA>::operator=(const CompiledModule&) = default;
CompiledModule<CUDA>& CompiledModule<CUDA>::operator=(CompiledModule&&) noexcept = default;
CompiledModule<CUDA>::~CompiledModule() = default;

CompiledModule<CUDA>::CompiledModule(std::shared_ptr<Impl> impl) : impl_(std::move(impl))
{
}

CompiledModule<CUDA> CompiledModuleArtifact::Load(CUDA device) const
{
	if (!constants_.empty() || !weights_.empty())
	{
		return SeparateRodata().Load(std::move(device));
	}
	return CompiledModule<CUDA>::Load(Image(), std::move(device));
}

CompiledModule<CUDA> CompiledModule<CUDA>::Load(CompiledModuleImage image, CUDA device)
{
	auto impl = std::make_shared<Impl>();
	impl->rodata = ToByteVector(image.rodata, image.rodataSize);
	impl->instructions = ToByteVector(image.instructions, image.instructionSize);

	auto metadata = DeserializeRodata(impl->rodata);
	impl->backend = metadata.backend;
	impl->inputSpecs = std::move(metadata.inputSpecs);
	impl->outputSpecs = std::move(metadata.outputSpecs);
	impl->device = std::move(device);

	if (impl->backend == CompiledModuleBackend::CPUNative)
	{
		RequireCUDACPUBridgeAllowed(impl->device, "CPU-native artifact");
		impl->cpuModule = CompiledModule<CPU>::Load({
		    .rodata = impl->rodata.data(),
		    .rodataSize = impl->rodata.size(),
		    .instructions = impl->instructions.data(),
		    .instructionSize = impl->instructions.size(),
		});
	}
	else if (impl->backend == CompiledModuleBackend::CUDANative)
	{
		impl->cudaPayload = DeserializeCUDANativeInstructionPayload(impl->instructions);
		if (impl->cudaPayload.binaryKind != CUDANativeBinaryKind::LibraryCall)
		{
			impl->cudaModule = CUDADriverModule(impl->device, impl->cudaPayload.binary);
			for (const auto& kernel : impl->cudaPayload.kernels)
			{
				if (!IsCUDANativeLibraryCallKernel(kernel.name))
				{
					impl->cudaModule.CacheFunction(kernel.name);
				}
			}
		}
		impl->cudaWorkspace.emplace(impl->device, CUDANativeWorkspaceByteSize(impl->cudaPayload));
		impl->cudaConstants.emplace(impl->device, impl->cudaPayload.constantData);
	}
	else
	{
		throw std::runtime_error("CompiledModule<CUDA> received an unsupported backend");
	}

	return CompiledModule(std::move(impl));
}

CompiledModule<CUDA> CompiledModule<CUDA>::Load(CompiledModuleSeparatedImage image, CUDA device)
{
	auto metadata = ValidateSeparatedImage(image);
	auto constants = RegionBytes(image.constants, kConstantsRegionName);
	auto instructions = RestoreLegacyInstructionsFromSeparated(
	    metadata.legacyMetadata.backend, RegionBytes(image.instructions, kInstructionsRegionName), constants);
	auto module = Load(
	    {
	        .rodata = metadata.legacyRodata.data(),
	        .rodataSize = metadata.legacyRodata.size(),
	        .instructions = instructions.data(),
	        .instructionSize = instructions.size(),
	    },
	    std::move(device));
	if (metadata.legacyMetadata.backend == CompiledModuleBackend::CPUNative)
	{
		module.impl_->cpuModule = CompiledModule<CPU>::Load(image);
	}
	return module;
}

CompiledModule<CUDA> CompiledModule<CUDA>::LoadBorrowedExternalRegions(CompiledModuleSeparatedImage image, CUDA device)
{
	auto metadata = ValidateSeparatedImage(image);
	if (metadata.legacyMetadata.backend == CompiledModuleBackend::CPUNative)
	{
		auto instructions = RestoreLegacyInstructionsFromSeparated(
		    metadata.legacyMetadata.backend, RegionBytes(image.instructions, kInstructionsRegionName),
		    RegionBytes(image.constants, kConstantsRegionName));
		auto module = Load(
		    {
		        .rodata = metadata.legacyRodata.data(),
		        .rodataSize = metadata.legacyRodata.size(),
		        .instructions = instructions.data(),
		        .instructionSize = instructions.size(),
		    },
		    std::move(device));
		module.impl_->cpuModule = CompiledModule<CPU>::LoadBorrowedExternalRegions(image);
		return module;
	}

	// CUDA-native kernels consume constants from a device allocation. The separated
	// host constants region is validated here and copied to device memory during Load().
	return Load(image, std::move(device));
}

std::vector<Tensor<CUDA>> CompiledModule<CUDA>::RunTensors(std::span<const Tensor<CUDA>> inputs) const
{
	return RunTensors(inputs, CompiledModuleCUDARunOptions{});
}

std::vector<Tensor<CUDA>> CompiledModule<CUDA>::RunTensors(std::span<const Tensor<CUDA>> inputs,
                                                           CompiledModuleCUDARunOptions options) const
{
	if (!impl_)
	{
		throw std::runtime_error("CompiledModule is empty");
	}
	if (inputs.size() != impl_->inputSpecs.size())
	{
		throw std::runtime_error(std::format("CompiledModule input count mismatch: expected {}, got {}",
		                                     impl_->inputSpecs.size(), inputs.size()));
	}

	for (std::size_t i = 0; i < inputs.size(); ++i)
	{
		ValidateTensorAgainstSpec(inputs[i], impl_->inputSpecs[i], i);
	}

	std::vector<Tensor<CUDA>> outputs;
	outputs.reserve(impl_->outputSpecs.size());
	for (const auto& spec : impl_->outputSpecs)
	{
		outputs.emplace_back(Uninitialized, ShapeView{ spec.type.StaticShape() }, spec.type.dtype, impl_->device);
	}
	RunTensorsInto(inputs, outputs, options);
	return outputs;
}

void CompiledModule<CUDA>::RunTensorsInto(std::span<const Tensor<CUDA>> inputs, std::span<Tensor<CUDA>> outputs) const
{
	RunTensorsInto(inputs, outputs, CompiledModuleCUDARunOptions{});
}

void CompiledModule<CUDA>::RunTensorsInto(std::span<const Tensor<CUDA>> inputs, std::span<Tensor<CUDA>> outputs,
                                          CompiledModuleCUDARunOptions options) const
{
	if (!impl_)
	{
		throw std::runtime_error("CompiledModule is empty");
	}
	if (inputs.size() != impl_->inputSpecs.size())
	{
		throw std::runtime_error(std::format("CompiledModule input count mismatch: expected {}, got {}",
		                                     impl_->inputSpecs.size(), inputs.size()));
	}
	if (outputs.size() != impl_->outputSpecs.size())
	{
		throw std::runtime_error(std::format("CompiledModule output count mismatch: expected {}, got {}",
		                                     impl_->outputSpecs.size(), outputs.size()));
	}

	for (std::size_t i = 0; i < inputs.size(); ++i)
	{
		ValidateTensorAgainstSpec(inputs[i], impl_->inputSpecs[i], i);
	}
	for (std::size_t i = 0; i < outputs.size(); ++i)
	{
		ValidateOutputTensorAgainstSpec(outputs[i], impl_->outputSpecs[i], i);
	}

	if (impl_->backend == CompiledModuleBackend::CUDANative)
	{
		if (!impl_->cudaWorkspace)
		{
			throw std::runtime_error("CUDA native workspace buffer is not initialized");
		}
		if (!impl_->cudaConstants)
		{
			throw std::runtime_error("CUDA native constant buffer is not initialized");
		}
		if (impl_->cudaWorkspace->ByteSize() != 0 && !options.synchronize)
		{
			throw std::runtime_error("CUDA native asynchronous execution with shared workspace is not supported");
		}
		if (RequestsCUDAGraphReplay(options) && !CanUseCUDAGraphReplay(options))
		{
			throw std::runtime_error(
			    "CUDA graph replay requires synchronized execution on the module-owned default stream");
		}
		if (CanUseCUDAGraphReplay(options))
		{
			std::scoped_lock lock(impl_->cudaWorkspaceMutex, impl_->cudaGraphReplayMutex);
			RunCUDANativePayloadWithGraphReplay(impl_->cudaGraphReplayCache, impl_->device, impl_->cudaPayload,
			                                    impl_->cudaModule, *impl_->cudaWorkspace, *impl_->cudaConstants, inputs,
			                                    outputs, options);
			return;
		}
		if (impl_->cudaWorkspace->ByteSize() == 0)
		{
			RunCUDANativePayload(impl_->device, impl_->cudaPayload, impl_->cudaModule, *impl_->cudaWorkspace,
			                     *impl_->cudaConstants, inputs, outputs, options);
		}
		else
		{
			std::lock_guard lock(impl_->cudaWorkspaceMutex);
			RunCUDANativePayload(impl_->device, impl_->cudaPayload, impl_->cudaModule, *impl_->cudaWorkspace,
			                     *impl_->cudaConstants, inputs, outputs, options);
		}
		return;
	}
	if (!options.synchronize)
	{
		throw std::runtime_error("CompiledModule<CUDA> CPU bridge does not support asynchronous execution");
	}

	std::vector<Tensor<CPU>> cpuInputs;
	cpuInputs.reserve(inputs.size());
	for (std::size_t i = 0; i < inputs.size(); ++i)
	{
		Tensor<CPU> cpuInput(Uninitialized, inputs[i].Shape(), inputs[i].DType(), CPU{});
		auto inputDevice = inputs[i].CurDevice();
		DeviceTraits<CUDA>::CopyToCPU(
		    inputDevice, inputs[i].DType(), inputs[i].UnsafeRawData(), inputs[i].NumElements(), cpuInput.DType(),
		    cpuInput.UnsafeRawData(),
		    CUDAExecutionOptions{ .stream = options.stream, .synchronize = true, .allowHostFallback = true });
		cpuInputs.push_back(std::move(cpuInput));
	}

	std::vector<Tensor<CPU>> cpuOutputs;
	cpuOutputs.reserve(impl_->outputSpecs.size());
	for (const auto& spec : impl_->outputSpecs)
	{
		cpuOutputs.emplace_back(Uninitialized, ShapeView{ spec.type.StaticShape() }, spec.type.dtype, CPU{});
	}
	impl_->cpuModule.RunTensorsInto(cpuInputs, cpuOutputs);

	for (std::size_t i = 0; i < outputs.size(); ++i)
	{
		DeviceTraits<CUDA>::CopyFromCPU(
		    outputs[i].CurDevice(), outputs[i].DType(), outputs[i].UnsafeRawData(), cpuOutputs[i].DType(),
		    cpuOutputs[i].UnsafeRawData(), cpuOutputs[i].NumElements(),
		    CUDAExecutionOptions{ .stream = options.stream, .synchronize = true, .allowHostFallback = true });
	}
}

void CompiledModule<CUDA>::RunManyTensorsInto(std::span<const CompiledModuleCUDATensorInvocation> invocations,
                                              std::size_t threadCount) const
{
	const auto workerCount = NormalizeThreadCount(threadCount, invocations.size());
	if (workerCount == 0)
	{
		return;
	}
	if (workerCount == 1)
	{
		for (const auto& invocation : invocations)
		{
			RunTensorsInto(invocation.inputs, invocation.outputs, invocation.options);
		}
		return;
	}

	std::atomic<std::size_t> next{ 0 };
	std::atomic_bool stop{ false };
	std::exception_ptr firstError;
	std::mutex errorMutex;

	auto worker = [&] {
		while (!stop.load(std::memory_order_relaxed))
		{
			const auto index = next.fetch_add(1, std::memory_order_relaxed);
			if (index >= invocations.size())
			{
				break;
			}

			try
			{
				const auto& invocation = invocations[index];
				RunTensorsInto(invocation.inputs, invocation.outputs, invocation.options);
			}
			catch (...)
			{
				{
					std::lock_guard lock(errorMutex);
					if (!firstError)
					{
						firstError = std::current_exception();
					}
				}
				stop.store(true, std::memory_order_relaxed);
				break;
			}
		}
	};

	std::vector<std::thread> workers;
	workers.reserve(workerCount);
	for (std::size_t i = 0; i < workerCount; ++i)
	{
		workers.emplace_back(worker);
	}
	for (auto& thread : workers)
	{
		thread.join();
	}

	if (firstError)
	{
		std::rethrow_exception(firstError);
	}
}

CompiledModuleImage CompiledModule<CUDA>::Image() const
{
	return impl_ ? CompiledModuleImage{
	                   .rodata = impl_->rodata.data(),
	                   .rodataSize = impl_->rodata.size(),
	                   .instructions = impl_->instructions.data(),
	                   .instructionSize = impl_->instructions.size(),
	               }
	             : CompiledModuleImage{};
}

std::span<const std::byte> CompiledModule<CUDA>::Rodata() const
{
	return impl_ ? std::span<const std::byte>{ impl_->rodata.data(), impl_->rodata.size() }
	             : std::span<const std::byte>{};
}

std::span<const std::byte> CompiledModule<CUDA>::Instructions() const
{
	return impl_ ? std::span<const std::byte>{ impl_->instructions.data(), impl_->instructions.size() }
	             : std::span<const std::byte>{};
}

std::span<const CompiledTensorSpec> CompiledModule<CUDA>::InputSpecs() const
{
	return impl_ ? std::span<const CompiledTensorSpec>{ impl_->inputSpecs.data(), impl_->inputSpecs.size() }
	             : std::span<const CompiledTensorSpec>{};
}

std::span<const CompiledTensorSpec> CompiledModule<CUDA>::OutputSpecs() const
{
	return impl_ ? std::span<const CompiledTensorSpec>{ impl_->outputSpecs.data(), impl_->outputSpecs.size() }
	             : std::span<const CompiledTensorSpec>{};
}

CompiledModuleBackend CompiledModule<CUDA>::Backend() const
{
	return impl_ ? impl_->backend : CompiledModuleBackend::CPUNative;
}

std::optional<std::size_t> CompiledModule<CUDA>::FindInput(std::string_view name) const
{
	return impl_ ? FindSpecIndex(impl_->inputSpecs, name) : std::nullopt;
}

std::optional<std::size_t> CompiledModule<CUDA>::FindOutput(std::string_view name) const
{
	return impl_ ? FindSpecIndex(impl_->outputSpecs, name) : std::nullopt;
}

void CompiledModule<CUDA>::WriteObjectFile(const std::filesystem::path& path, std::string_view symbolPrefix) const
{
	if (!impl_)
	{
		throw std::runtime_error("CompiledModule is empty");
	}
	const auto objectBytes = EmitCarrierObject(impl_->rodata, impl_->instructions, symbolPrefix);
	WriteAllBytes(path, objectBytes);
}
#endif

#ifdef LITENN_ENABLE_VULKAN
namespace
{
	void RequireVulkanCPUBridgeAllowed(const Vulkan& device, std::string_view operation)
	{
		if (device.hostFallbackPolicy != VulkanHostFallbackPolicy::Allow)
		{
			throw std::runtime_error(std::format(
			    "CompiledModule<Vulkan> CPU bridge for {} is disabled; load with VulkanHostFallbackPolicy::Allow or "
			    "lower the runtime schedule to a native Vulkan artifact with explicit fallback steps",
			    operation));
		}
	}

	std::uint32_t VulkanDescriptorCount(const VulkanNativeKernelSpec& kernel)
	{
		std::uint32_t count = 0;
		for (const auto& argument : kernel.arguments)
		{
			if (argument.binding == std::numeric_limits<std::uint32_t>::max())
			{
				throw std::runtime_error("Vulkan native argument binding is invalid");
			}
			count = std::max(count, argument.binding + 1);
		}
		if (count == 0)
		{
			throw std::runtime_error("Vulkan native kernel has no descriptor bindings");
		}
		return count;
	}

	std::vector<VulkanSpecializationConstant> VulkanSpecializationConstants(const VulkanNativeKernelSpec& kernel)
	{
		std::vector<VulkanSpecializationConstant> constants;
		constants.reserve(kernel.specializationConstants.size());
		for (const auto& constant : kernel.specializationConstants)
		{
			constants.push_back({
			    .constantId = constant.constantId,
			    .byteOffset = constant.byteOffset,
			    .byteSize = constant.byteSize,
			});
		}
		return constants;
	}

	std::uint64_t VulkanTensorByteSize(DataType dtype, std::size_t elementCount)
	{
		if (elementCount > std::numeric_limits<std::uint64_t>::max() / ElementByteSize(dtype))
		{
			throw std::runtime_error("Vulkan native tensor byte size overflows uint64_t");
		}
		return static_cast<std::uint64_t>(elementCount) * ElementByteSize(dtype);
	}

	void ValidateVulkanArgumentRange(const VulkanNativeArgumentSpec& argument, DataType dtype, std::size_t elementCount,
	                                 std::string_view label)
	{
		const auto byteSize = VulkanTensorByteSize(dtype, elementCount);
		if (argument.byteOffset > byteSize || argument.byteSize > byteSize - argument.byteOffset)
		{
			throw std::runtime_error(std::format("Vulkan native {} argument byte range is out of bounds", label));
		}
		if (argument.byteOffset != 0)
		{
			throw std::runtime_error("Vulkan native P0 payload does not support tensor byte offsets");
		}
	}

	void ValidateVulkanWorkspaceArgumentRange(const VulkanNativeArgumentSpec& argument,
	                                          const VulkanNativeWorkspaceSpec& workspace)
	{
		if (argument.byteOffset > workspace.byteSize || argument.byteSize > workspace.byteSize - argument.byteOffset)
		{
			throw std::runtime_error("Vulkan native workspace argument byte range is out of bounds");
		}
		if (argument.byteOffset != 0)
		{
			throw std::runtime_error("Vulkan native P0 payload does not support workspace byte offsets");
		}
	}

	bool VulkanNativeHasFeature(const VulkanNativeInstructionPayload& payload, VulkanNativeFeature feature)
	{
		return (payload.featureSet.flags & (1ull << static_cast<std::uint32_t>(feature))) != 0;
	}

	bool VulkanApiVersionAtLeast(const VulkanDeviceCapabilities& capabilities, std::uint32_t major, std::uint32_t minor)
	{
		return capabilities.apiVersionMajor > major ||
		       (capabilities.apiVersionMajor == major && capabilities.apiVersionMinor >= minor);
	}

	bool VulkanNativeSpecUsesDType(std::span<const CompiledTensorSpec> specs, DataType dtype)
	{
		return std::ranges::any_of(specs, [&](const CompiledTensorSpec& spec) { return spec.type.dtype == dtype; });
	}

	std::string VulkanDeviceCapabilityName(const VulkanDeviceCapabilities& capabilities)
	{
		return capabilities.deviceName.empty() ? std::string("selected Vulkan device") : capabilities.deviceName;
	}

	void RequireVulkanNativeDeviceFeature(bool enabled, bool available, std::string_view featureName,
	                                      const VulkanDeviceCapabilities& capabilities)
	{
		if (enabled)
		{
			return;
		}
		throw std::runtime_error(std::format(
		    "Vulkan native payload requires {}, but device '{}' reports available={} and LiteNN logical-device "
		    "enabled={}",
		    featureName, VulkanDeviceCapabilityName(capabilities), available, enabled));
	}

	void RequireVulkanNativeDeviceFeature(const VulkanNativeDeviceRequirementSet& requirements,
	                                      VulkanNativeDeviceRequirement requirement, bool enabled, bool available,
	                                      std::string_view featureName, const VulkanDeviceCapabilities& capabilities)
	{
		if (!requirements.HasRequirement(requirement))
		{
			return;
		}
		RequireVulkanNativeDeviceFeature(enabled, available, featureName, capabilities);
	}

	std::uint64_t VulkanDispatchDimProduct(VulkanNativeDispatchDim dim)
	{
		return static_cast<std::uint64_t>(dim.x) * dim.y * dim.z;
	}

	void ValidateVulkanNativeDeviceCapabilities(const VulkanNativeInstructionPayload& payload,
	                                            std::span<const CompiledTensorSpec> inputSpecs,
	                                            std::span<const CompiledTensorSpec> outputSpecs, const Vulkan& device)
	{
		const auto capabilities = QueryVulkanDeviceCapabilities(device);
		if (payload.target != "vulkan1.1")
		{
			throw std::runtime_error(std::format("Unsupported Vulkan native target '{}'", payload.target));
		}
		if (!VulkanApiVersionAtLeast(capabilities, 1, 1))
		{
			throw std::runtime_error(
			    std::format("Vulkan native payload requires Vulkan 1.1 or newer; device '{}' reports {}.{}.{}",
			                VulkanDeviceCapabilityName(capabilities), capabilities.apiVersionMajor,
			                capabilities.apiVersionMinor, capabilities.apiVersionPatch));
		}
		for (std::size_t kernelIndex = 0; kernelIndex < payload.kernels.size(); ++kernelIndex)
		{
			const auto& kernel = payload.kernels[kernelIndex];
			const auto& requirements = kernel.requirements;
			if (requirements.descriptorAbiVersion != 1)
			{
				throw std::runtime_error(
				    std::format("Vulkan native kernel {} requires unsupported descriptor ABI version {}", kernelIndex,
				                requirements.descriptorAbiVersion));
			}
			if (kernel.groups.x == 0 || kernel.groups.y == 0 || kernel.groups.z == 0)
			{
				throw std::runtime_error(
				    std::format("Vulkan native kernel {} has zero dispatch dimension", kernelIndex));
			}
			if (kernel.groups.x > capabilities.maxComputeWorkGroupCount[0] ||
			    kernel.groups.y > capabilities.maxComputeWorkGroupCount[1] ||
			    kernel.groups.z > capabilities.maxComputeWorkGroupCount[2])
			{
				throw std::runtime_error(
				    std::format("Vulkan native kernel {} requires dispatch groups {}x{}x{}, but device '{}' supports "
				                "maxComputeWorkGroupCount {}x{}x{}",
				                kernelIndex, kernel.groups.x, kernel.groups.y, kernel.groups.z,
				                VulkanDeviceCapabilityName(capabilities), capabilities.maxComputeWorkGroupCount[0],
				                capabilities.maxComputeWorkGroupCount[1], capabilities.maxComputeWorkGroupCount[2]));
			}
			const auto descriptorCount = VulkanDescriptorCount(kernel);
			if (capabilities.maxBoundDescriptorSets < 1)
			{
				throw std::runtime_error(std::format(
				    "Vulkan native kernel {} requires one descriptor set, but device '{}' reports "
				    "maxBoundDescriptorSets={}",
				    kernelIndex, VulkanDeviceCapabilityName(capabilities), capabilities.maxBoundDescriptorSets));
			}
			if (descriptorCount > capabilities.maxPerStageDescriptorStorageBuffers ||
			    descriptorCount > capabilities.maxDescriptorSetStorageBuffers)
			{
				throw std::runtime_error(std::format(
				    "Vulkan native kernel {} requires {} storage-buffer descriptor(s), but device '{}' reports "
				    "maxPerStageDescriptorStorageBuffers={} and maxDescriptorSetStorageBuffers={}",
				    kernelIndex, descriptorCount, VulkanDeviceCapabilityName(capabilities),
				    capabilities.maxPerStageDescriptorStorageBuffers, capabilities.maxDescriptorSetStorageBuffers));
			}
			const auto localInvocations = VulkanDispatchDimProduct(requirements.localSize);
			if (localInvocations > capabilities.maxComputeWorkGroupInvocations ||
			    requirements.localSize.x > capabilities.maxComputeWorkGroupSize[0] ||
			    requirements.localSize.y > capabilities.maxComputeWorkGroupSize[1] ||
			    requirements.localSize.z > capabilities.maxComputeWorkGroupSize[2])
			{
				throw std::runtime_error(std::format(
				    "Vulkan native kernel {} requires local size {}x{}x{} ({} invocations), but device '{}' "
				    "supports max size {}x{}x{} and {} invocations",
				    kernelIndex, requirements.localSize.x, requirements.localSize.y, requirements.localSize.z,
				    localInvocations, VulkanDeviceCapabilityName(capabilities), capabilities.maxComputeWorkGroupSize[0],
				    capabilities.maxComputeWorkGroupSize[1], capabilities.maxComputeWorkGroupSize[2],
				    capabilities.maxComputeWorkGroupInvocations));
			}
			if (requirements.requiredSubgroupSize != 0 &&
			    (!capabilities.subgroupComputeAvailable || !capabilities.subgroupBasicAvailable ||
			     capabilities.subgroupSize != requirements.requiredSubgroupSize))
			{
				throw std::runtime_error(
				    std::format("Vulkan native kernel {} requires compute subgroup size {}, but device '{}' reports "
				                "subgroupSize={}, compute={}, basic={}",
				                kernelIndex, requirements.requiredSubgroupSize,
				                VulkanDeviceCapabilityName(capabilities), capabilities.subgroupSize,
				                capabilities.subgroupComputeAvailable, capabilities.subgroupBasicAvailable));
			}
			RequireVulkanNativeDeviceFeature(
			    requirements.deviceRequirements, VulkanNativeDeviceRequirement::SubgroupArithmetic,
			    capabilities.subgroupComputeAvailable && capabilities.subgroupArithmeticAvailable,
			    capabilities.subgroupArithmeticAvailable, "subgroupArithmetic", capabilities);
			RequireVulkanNativeDeviceFeature(
			    requirements.deviceRequirements, VulkanNativeDeviceRequirement::SubgroupBallot,
			    capabilities.subgroupComputeAvailable && capabilities.subgroupBallotAvailable,
			    capabilities.subgroupBallotAvailable, "subgroupBallot", capabilities);
			RequireVulkanNativeDeviceFeature(
			    requirements.deviceRequirements, VulkanNativeDeviceRequirement::SubgroupShuffle,
			    capabilities.subgroupComputeAvailable && capabilities.subgroupShuffleAvailable,
			    capabilities.subgroupShuffleAvailable, "subgroupShuffle", capabilities);
			RequireVulkanNativeDeviceFeature(
			    requirements.deviceRequirements, VulkanNativeDeviceRequirement::ShaderFloat16,
			    capabilities.shaderFloat16Enabled, capabilities.shaderFloat16Available, "shaderFloat16", capabilities);
			RequireVulkanNativeDeviceFeature(requirements.deviceRequirements, VulkanNativeDeviceRequirement::ShaderInt8,
			                                 capabilities.shaderInt8Enabled, capabilities.shaderInt8Available,
			                                 "shaderInt8", capabilities);
			RequireVulkanNativeDeviceFeature(
			    requirements.deviceRequirements, VulkanNativeDeviceRequirement::StorageBuffer16BitAccess,
			    capabilities.storageBuffer16BitAccessEnabled, capabilities.storageBuffer16BitAccessAvailable,
			    "storageBuffer16BitAccess", capabilities);
			RequireVulkanNativeDeviceFeature(
			    requirements.deviceRequirements, VulkanNativeDeviceRequirement::StorageBuffer8BitAccess,
			    capabilities.storageBuffer8BitAccessEnabled, capabilities.storageBuffer8BitAccessAvailable,
			    "storageBuffer8BitAccess", capabilities);
			RequireVulkanNativeDeviceFeature(requirements.deviceRequirements,
			                                 VulkanNativeDeviceRequirement::ShaderStorageBufferArrayNonUniformIndexing,
			                                 capabilities.shaderStorageBufferArrayNonUniformIndexingEnabled,
			                                 capabilities.shaderStorageBufferArrayNonUniformIndexingAvailable,
			                                 "shaderStorageBufferArrayNonUniformIndexing", capabilities);
			RequireVulkanNativeDeviceFeature(
			    requirements.deviceRequirements,
			    VulkanNativeDeviceRequirement::DescriptorBindingStorageBufferUpdateAfterBind,
			    capabilities.descriptorBindingStorageBufferUpdateAfterBindEnabled,
			    capabilities.descriptorBindingStorageBufferUpdateAfterBindAvailable,
			    "descriptorBindingStorageBufferUpdateAfterBind", capabilities);
			RequireVulkanNativeDeviceFeature(
			    requirements.deviceRequirements, VulkanNativeDeviceRequirement::DescriptorBindingPartiallyBound,
			    capabilities.descriptorBindingPartiallyBoundEnabled,
			    capabilities.descriptorBindingPartiallyBoundAvailable, "descriptorBindingPartiallyBound", capabilities);
			RequireVulkanNativeDeviceFeature(requirements.deviceRequirements,
			                                 VulkanNativeDeviceRequirement::DescriptorBindingVariableDescriptorCount,
			                                 capabilities.descriptorBindingVariableDescriptorCountEnabled,
			                                 capabilities.descriptorBindingVariableDescriptorCountAvailable,
			                                 "descriptorBindingVariableDescriptorCount", capabilities);
			RequireVulkanNativeDeviceFeature(
			    requirements.deviceRequirements, VulkanNativeDeviceRequirement::RuntimeDescriptorArray,
			    capabilities.runtimeDescriptorArrayEnabled, capabilities.runtimeDescriptorArrayAvailable,
			    "runtimeDescriptorArray", capabilities);
			const auto requiredAlignment = std::max<std::uint64_t>(requirements.requiredStorageBufferOffsetAlignment,
			                                                       capabilities.minStorageBufferOffsetAlignment);
			for (const auto& argument : kernel.arguments)
			{
				if (argument.byteSize > capabilities.maxStorageBufferRange)
				{
					throw std::runtime_error(std::format(
					    "Vulkan native kernel {} binding {} requires storage-buffer range {} bytes, but device "
					    "'{}' supports at most {} bytes",
					    kernelIndex, argument.binding, argument.byteSize, VulkanDeviceCapabilityName(capabilities),
					    capabilities.maxStorageBufferRange));
				}
				if (requiredAlignment != 0 && argument.byteOffset % requiredAlignment != 0)
				{
					throw std::runtime_error(
					    std::format("Vulkan native kernel {} binding {} byte offset {} is not aligned to {} bytes",
					                kernelIndex, argument.binding, argument.byteOffset, requiredAlignment));
				}
			}
		}

		if (VulkanNativeHasFeature(payload, VulkanNativeFeature::SameShapeCastLowPrecision))
		{
			if (VulkanNativeSpecUsesDType(inputSpecs, DataType::Float16) ||
			    VulkanNativeSpecUsesDType(outputSpecs, DataType::Float16))
			{
				RequireVulkanNativeDeviceFeature(capabilities.shaderFloat16Enabled, capabilities.shaderFloat16Available,
				                                 "shaderFloat16", capabilities);
				RequireVulkanNativeDeviceFeature(capabilities.storageBuffer16BitAccessEnabled,
				                                 capabilities.storageBuffer16BitAccessAvailable,
				                                 "storageBuffer16BitAccess", capabilities);
			}
			if (VulkanNativeSpecUsesDType(inputSpecs, DataType::Int8) ||
			    VulkanNativeSpecUsesDType(outputSpecs, DataType::Int8) ||
			    VulkanNativeSpecUsesDType(inputSpecs, DataType::UInt8) ||
			    VulkanNativeSpecUsesDType(outputSpecs, DataType::UInt8))
			{
				RequireVulkanNativeDeviceFeature(capabilities.shaderInt8Enabled, capabilities.shaderInt8Available,
				                                 "shaderInt8", capabilities);
				RequireVulkanNativeDeviceFeature(capabilities.storageBuffer8BitAccessEnabled,
				                                 capabilities.storageBuffer8BitAccessAvailable,
				                                 "storageBuffer8BitAccess", capabilities);
			}
		}
	}

	std::vector<Tensor<Vulkan>> LoadVulkanExternalTensors(const SeparatedMetadata& metadata,
	                                                      CompiledModuleSeparatedImage image, Vulkan device)
	{
		std::vector<Tensor<Vulkan>> tensors;
		tensors.reserve(metadata.externalTensorInfos.size());
		for (const auto& info : metadata.externalTensorInfos)
		{
			const auto region = SeparatedImageRegionBytes(image, info.region);
			if (info.byteOffset > region.size() || info.byteSize > region.size() - info.byteOffset)
			{
				throw std::runtime_error(
				    std::format("Vulkan native external tensor '{}' byte range is out of bounds", info.name));
			}
			const auto shape = info.type.StaticShape();
			const auto expectedByteSize = TensorByteSizeForShape(info.type.dtype, shape);
			if (info.byteSize != expectedByteSize)
			{
				throw std::runtime_error(
				    std::format("Vulkan native external tensor '{}' byte size does not match its type", info.name));
			}

			Tensor<CPU> host(Uninitialized, ShapeView{ shape }, info.type.dtype, CPU{});
			std::memcpy(host.UnsafeRawData(), region.data() + static_cast<std::ptrdiff_t>(info.byteOffset),
			            static_cast<std::size_t>(info.byteSize));
			tensors.push_back(host.CopyToDevice(device));
		}
		return tensors;
	}

	std::vector<Tensor<Vulkan>> AllocateVulkanWorkspaceTensors(const VulkanNativeInstructionPayload& payload,
	                                                           Vulkan device)
	{
		std::vector<Tensor<Vulkan>> tensors;
		tensors.reserve(payload.workspaceTensors.size());
		for (const auto& workspace : payload.workspaceTensors)
		{
			if (workspace.byteSize > std::numeric_limits<std::size_t>::max())
			{
				throw std::runtime_error("Vulkan native workspace tensor is too large for this host");
			}
			const std::array<std::size_t, 1> shape{ static_cast<std::size_t>(workspace.byteSize) };
			tensors.emplace_back(Uninitialized, ShapeView{ shape }, DataType::UInt8, device);
		}
		return tensors;
	}

	void RunVulkanNativePayload(const VulkanNativeInstructionPayload& payload,
	                            std::span<const VulkanComputeModule> modules, std::span<const Tensor<Vulkan>> inputs,
	                            std::span<const Tensor<Vulkan>> externalTensors,
	                            std::span<Tensor<Vulkan>> workspaceTensors, std::span<Tensor<Vulkan>> outputs,
	                            CompiledModuleVulkanRunOptions options)
	{
		if (!options.synchronize)
		{
			throw std::runtime_error(
			    "CompiledModule<Vulkan> native backend does not expose asynchronous execution yet");
		}
		if (modules.size() != payload.kernels.size())
		{
			throw std::runtime_error("Vulkan native module count does not match payload kernel count");
		}
		const bool batchSubmit = options.profileEvents == nullptr && payload.kernels.size() > 1;
		std::vector<std::vector<const void*>> batchDescriptorStorage;
		std::vector<VulkanComputeBatchDispatch> batchDispatches;
		if (batchSubmit)
		{
			batchDescriptorStorage.reserve(payload.kernels.size());
			batchDispatches.reserve(payload.kernels.size());
		}
		const auto dispatchKernel = [&](std::size_t kernelIndex) {
			const auto& kernel = payload.kernels[kernelIndex];
			std::vector<const void*> descriptors(VulkanDescriptorCount(kernel), nullptr);
			for (const auto& argument : kernel.arguments)
			{
				if (argument.binding >= descriptors.size())
				{
					throw std::runtime_error("Vulkan native argument binding is out of bounds");
				}
				switch (argument.kind)
				{
				case VulkanNativeArgumentKind::InputTensor:
					if (argument.index >= inputs.size())
					{
						throw std::runtime_error("Vulkan native input argument index is out of bounds");
					}
					ValidateVulkanArgumentRange(argument, inputs[argument.index].DType(),
					                            inputs[argument.index].NumElements(), "input");
					descriptors[argument.binding] = inputs[argument.index].UnsafeRawData();
					break;
				case VulkanNativeArgumentKind::ExternalTensor:
					if (argument.index >= externalTensors.size())
					{
						throw std::runtime_error("Vulkan native external tensor argument index is out of bounds");
					}
					ValidateVulkanArgumentRange(argument, externalTensors[argument.index].DType(),
					                            externalTensors[argument.index].NumElements(), "external tensor");
					descriptors[argument.binding] = externalTensors[argument.index].UnsafeRawData();
					break;
				case VulkanNativeArgumentKind::WorkspaceTensor:
					if (argument.index >= workspaceTensors.size() || argument.index >= payload.workspaceTensors.size())
					{
						throw std::runtime_error("Vulkan native workspace argument index is out of bounds");
					}
					ValidateVulkanWorkspaceArgumentRange(argument, payload.workspaceTensors[argument.index]);
					descriptors[argument.binding] = workspaceTensors[argument.index].UnsafeRawData();
					break;
				case VulkanNativeArgumentKind::OutputTensor:
					if (argument.index >= outputs.size())
					{
						throw std::runtime_error("Vulkan native output argument index is out of bounds");
					}
					ValidateVulkanArgumentRange(argument, outputs[argument.index].DType(),
					                            outputs[argument.index].NumElements(), "output");
					descriptors[argument.binding] = outputs[argument.index].UnsafeRawData();
					break;
				}
			}
			if (std::ranges::any_of(descriptors, [](const void* ptr) { return ptr == nullptr; }))
			{
				throw std::runtime_error("Vulkan native kernel has an unbound descriptor");
			}
			const VulkanDispatchDim dispatchGroups{
				.x = kernel.groups.x,
				.y = kernel.groups.y,
				.z = kernel.groups.z,
			};
			if (batchSubmit)
			{
				batchDescriptorStorage.push_back(std::move(descriptors));
				auto& storedDescriptors = batchDescriptorStorage.back();
				batchDispatches.push_back({
				    .module = &modules[kernelIndex],
				    .descriptorBuffers = std::span<const void*>(storedDescriptors.data(), storedDescriptors.size()),
				    .groups = dispatchGroups,
				});
				return;
			}
			VulkanDispatchTiming gpuTiming;
			const auto dispatchBegin = std::chrono::steady_clock::now();
			modules[kernelIndex].Dispatch(descriptors, dispatchGroups,
			                              VulkanExecutionOptions{
			                                  .synchronize = true,
			                                  .timing = options.profileEvents == nullptr ? nullptr : &gpuTiming,
			                              });
			const auto dispatchEnd = std::chrono::steady_clock::now();
			if (options.profileEvents != nullptr)
			{
				options.profileEvents->push_back({
				    .kernelIndex = kernelIndex,
				    .entryPoint = kernel.entryPoint,
				    .groups = dispatchGroups,
				    .localSize = {
				        .x = kernel.requirements.localSize.x,
				        .y = kernel.requirements.localSize.y,
				        .z = kernel.requirements.localSize.z,
				    },
				    .descriptorCount = static_cast<std::uint32_t>(descriptors.size()),
				    .moduleCreationWallMs = modules[kernelIndex].CreationWallTimeMs(),
				    .dispatchWallMs = std::chrono::duration<double, std::milli>(dispatchEnd - dispatchBegin).count(),
				    .gpuTimestampAvailable = gpuTiming.gpuTimestampAvailable,
				    .gpuElapsedMs = gpuTiming.gpuElapsedMs,
				});
			}
		};
		for (std::size_t kernelIndex = 0; kernelIndex < payload.kernels.size(); ++kernelIndex)
		{
			dispatchKernel(kernelIndex);
		}
		if (batchSubmit)
		{
			VulkanComputeModule::DispatchBatch(batchDispatches);
		}
	}
} // namespace

struct CompiledModule<Vulkan>::Impl
{
	std::vector<std::byte> rodata;
	std::vector<std::byte> instructions;
	std::vector<CompiledTensorSpec> inputSpecs;
	std::vector<CompiledTensorSpec> outputSpecs;
	CompiledModuleBackend backend{ CompiledModuleBackend::CPUNative };
	CompiledModule<CPU> cpuModule;
	Vulkan device;
	VulkanNativeInstructionPayload vulkanPayload;
	std::vector<VulkanComputeModule> vulkanModules;
	std::vector<Tensor<Vulkan>> vulkanExternalTensors;
	std::vector<Tensor<Vulkan>> vulkanWorkspaceTensors;
	mutable std::mutex vulkanNativeRunMutex;
};

CompiledModule<Vulkan>::CompiledModule() = default;
CompiledModule<Vulkan>::CompiledModule(const CompiledModule&) = default;
CompiledModule<Vulkan>::CompiledModule(CompiledModule&&) noexcept = default;
CompiledModule<Vulkan>& CompiledModule<Vulkan>::operator=(const CompiledModule&) = default;
CompiledModule<Vulkan>& CompiledModule<Vulkan>::operator=(CompiledModule&&) noexcept = default;
CompiledModule<Vulkan>::~CompiledModule() = default;

CompiledModule<Vulkan>::CompiledModule(std::shared_ptr<Impl> impl) : impl_(std::move(impl))
{
}

CompiledModule<Vulkan> CompiledModuleArtifact::Load(Vulkan device) const
{
	if (!constants_.empty() || !weights_.empty())
	{
		return SeparateRodata().Load(std::move(device));
	}
	return CompiledModule<Vulkan>::Load(Image(), std::move(device));
}

CompiledModule<Vulkan> CompiledModuleSeparatedArtifact::Load(Vulkan device) const
{
	return CompiledModule<Vulkan>::Load(Image(), std::move(device));
}

CompiledModule<Vulkan> CompiledModuleSeparatedArtifact::LoadBorrowedExternalRegions(Vulkan device) const
{
	return CompiledModule<Vulkan>::LoadBorrowedExternalRegions(Image(), std::move(device));
}

CompiledModule<Vulkan> CompiledModule<Vulkan>::Load(CompiledModuleImage image, Vulkan device)
{
	auto impl = std::make_shared<Impl>();
	impl->rodata = ToByteVector(image.rodata, image.rodataSize);
	impl->instructions = ToByteVector(image.instructions, image.instructionSize);

	auto metadata = DeserializeRodata(impl->rodata);
	impl->backend = metadata.backend;
	impl->inputSpecs = std::move(metadata.inputSpecs);
	impl->outputSpecs = std::move(metadata.outputSpecs);
	impl->device = std::move(device);

	if (impl->backend == CompiledModuleBackend::CPUNative)
	{
		RequireVulkanCPUBridgeAllowed(impl->device, "CPU-native artifact");
		impl->cpuModule = CompiledModule<CPU>::Load({
		    .rodata = impl->rodata.data(),
		    .rodataSize = impl->rodata.size(),
		    .instructions = impl->instructions.data(),
		    .instructionSize = impl->instructions.size(),
		});
	}
	else if (impl->backend == CompiledModuleBackend::VulkanNative)
	{
		impl->vulkanPayload = DeserializeVulkanNativeInstructionPayload(impl->instructions);
		ValidateVulkanNativeDeviceCapabilities(impl->vulkanPayload, impl->inputSpecs, impl->outputSpecs, impl->device);
		impl->vulkanWorkspaceTensors = AllocateVulkanWorkspaceTensors(impl->vulkanPayload, impl->device);
		impl->vulkanModules.reserve(impl->vulkanPayload.kernels.size());
		for (const auto& kernel : impl->vulkanPayload.kernels)
		{
			const auto specializationConstants = VulkanSpecializationConstants(kernel);
			impl->vulkanModules.emplace_back(impl->device, impl->vulkanPayload.spirv, kernel.entryPoint,
			                                 VulkanDescriptorCount(kernel), specializationConstants,
			                                 kernel.specializationData);
		}
	}
	else
	{
		throw std::runtime_error("CompiledModule<Vulkan> received an unsupported backend");
	}

	return CompiledModule(std::move(impl));
}

CompiledModule<Vulkan> CompiledModule<Vulkan>::Load(CompiledModuleSeparatedImage image, Vulkan device)
{
	auto metadata = ValidateSeparatedImage(image);
	auto constants = RegionBytes(image.constants, kConstantsRegionName);
	auto instructions = RestoreLegacyInstructionsFromSeparated(
	    metadata.legacyMetadata.backend, RegionBytes(image.instructions, kInstructionsRegionName), constants);
	auto module = Load(
	    {
	        .rodata = metadata.legacyRodata.data(),
	        .rodataSize = metadata.legacyRodata.size(),
	        .instructions = instructions.data(),
	        .instructionSize = instructions.size(),
	    },
	    std::move(device));
	if (metadata.legacyMetadata.backend == CompiledModuleBackend::CPUNative)
	{
		module.impl_->cpuModule = CompiledModule<CPU>::Load(image);
	}
	else if (metadata.legacyMetadata.backend == CompiledModuleBackend::VulkanNative)
	{
		module.impl_->vulkanExternalTensors = LoadVulkanExternalTensors(metadata, image, module.impl_->device);
	}
	return module;
}

CompiledModule<Vulkan> CompiledModule<Vulkan>::LoadBorrowedExternalRegions(CompiledModuleSeparatedImage image,
                                                                           Vulkan device)
{
	auto metadata = ValidateSeparatedImage(image);
	if (metadata.legacyMetadata.backend == CompiledModuleBackend::CPUNative)
	{
		auto instructions = RestoreLegacyInstructionsFromSeparated(
		    metadata.legacyMetadata.backend, RegionBytes(image.instructions, kInstructionsRegionName),
		    RegionBytes(image.constants, kConstantsRegionName));
		auto module = Load(
		    {
		        .rodata = metadata.legacyRodata.data(),
		        .rodataSize = metadata.legacyRodata.size(),
		        .instructions = instructions.data(),
		        .instructionSize = instructions.size(),
		    },
		    std::move(device));
		module.impl_->cpuModule = CompiledModule<CPU>::LoadBorrowedExternalRegions(image);
		return module;
	}
	return Load(image, std::move(device));
}

std::vector<Tensor<Vulkan>> CompiledModule<Vulkan>::RunTensors(std::span<const Tensor<Vulkan>> inputs) const
{
	return RunTensors(inputs, CompiledModuleVulkanRunOptions{});
}

std::vector<Tensor<Vulkan>> CompiledModule<Vulkan>::RunTensors(std::span<const Tensor<Vulkan>> inputs,
                                                               CompiledModuleVulkanRunOptions options) const
{
	if (!impl_)
	{
		throw std::runtime_error("CompiledModule is empty");
	}
	if (inputs.size() != impl_->inputSpecs.size())
	{
		throw std::runtime_error(std::format("CompiledModule input count mismatch: expected {}, got {}",
		                                     impl_->inputSpecs.size(), inputs.size()));
	}
	for (std::size_t i = 0; i < inputs.size(); ++i)
	{
		ValidateTensorAgainstSpec(inputs[i], impl_->inputSpecs[i], i);
	}

	auto outputs = AllocateOutputTensors();
	RunTensorsInto(inputs, outputs, options);
	return outputs;
}

std::vector<Tensor<Vulkan>> CompiledModule<Vulkan>::AllocateOutputTensors() const
{
	if (!impl_)
	{
		throw std::runtime_error("CompiledModule is empty");
	}

	std::vector<Tensor<Vulkan>> outputs;
	outputs.reserve(impl_->outputSpecs.size());
	for (const auto& spec : impl_->outputSpecs)
	{
		outputs.emplace_back(Uninitialized, ShapeView{ spec.type.StaticShape() }, spec.type.dtype, impl_->device);
	}
	return outputs;
}

CompiledModuleVulkanRunWorkspace CompiledModule<Vulkan>::CreateRunWorkspace() const
{
	if (!impl_)
	{
		throw std::runtime_error("CompiledModule is empty");
	}

	CompiledModuleVulkanRunWorkspace workspace;
	workspace.outputs_ = AllocateOutputTensors();
	if (impl_->backend == CompiledModuleBackend::CPUNative)
	{
		workspace.cpuInputs_.reserve(impl_->inputSpecs.size());
		for (const auto& spec : impl_->inputSpecs)
		{
			workspace.cpuInputs_.emplace_back(Uninitialized, ShapeView{ spec.type.StaticShape() }, spec.type.dtype,
			                                  CPU{});
		}
		workspace.cpuOutputs_.reserve(impl_->outputSpecs.size());
		for (const auto& spec : impl_->outputSpecs)
		{
			workspace.cpuOutputs_.emplace_back(Uninitialized, ShapeView{ spec.type.StaticShape() }, spec.type.dtype,
			                                   CPU{});
		}
	}
	return workspace;
}

std::span<Tensor<Vulkan>> CompiledModule<Vulkan>::RunTensors(std::span<const Tensor<Vulkan>> inputs,
                                                             CompiledModuleVulkanRunWorkspace& workspace) const
{
	return RunTensors(inputs, workspace, CompiledModuleVulkanRunOptions{});
}

std::span<Tensor<Vulkan>> CompiledModule<Vulkan>::RunTensors(std::span<const Tensor<Vulkan>> inputs,
                                                             CompiledModuleVulkanRunWorkspace& workspace,
                                                             CompiledModuleVulkanRunOptions options) const
{
	if (!impl_)
	{
		throw std::runtime_error("CompiledModule is empty");
	}
	if (inputs.size() != impl_->inputSpecs.size())
	{
		throw std::runtime_error(std::format("CompiledModule input count mismatch: expected {}, got {}",
		                                     impl_->inputSpecs.size(), inputs.size()));
	}
	if (workspace.outputs_.size() != impl_->outputSpecs.size())
	{
		throw std::runtime_error(
		    std::format("CompiledModule Vulkan workspace output count mismatch: expected {}, got {}",
		                impl_->outputSpecs.size(), workspace.outputs_.size()));
	}
	for (std::size_t i = 0; i < inputs.size(); ++i)
	{
		ValidateTensorAgainstSpec(inputs[i], impl_->inputSpecs[i], i);
	}
	for (std::size_t i = 0; i < workspace.outputs_.size(); ++i)
	{
		ValidateOutputTensorAgainstSpec(workspace.outputs_[i], impl_->outputSpecs[i], i);
	}

	if (impl_->backend == CompiledModuleBackend::VulkanNative)
	{
		std::lock_guard lock(impl_->vulkanNativeRunMutex);
		RunVulkanNativePayload(impl_->vulkanPayload, impl_->vulkanModules, inputs, impl_->vulkanExternalTensors,
		                       impl_->vulkanWorkspaceTensors, workspace.outputs_, options);
		return workspace.Outputs();
	}
	if (!options.synchronize)
	{
		throw std::runtime_error("CompiledModule<Vulkan> CPU bridge does not support asynchronous execution");
	}
	if (workspace.cpuInputs_.size() != impl_->inputSpecs.size() ||
	    workspace.cpuOutputs_.size() != impl_->outputSpecs.size())
	{
		throw std::runtime_error("CompiledModule Vulkan workspace is not initialized for CPU bridge execution");
	}

	for (std::size_t i = 0; i < inputs.size(); ++i)
	{
		ValidateOutputTensorAgainstSpec(workspace.cpuInputs_[i], impl_->inputSpecs[i], i);
		auto inputDevice = inputs[i].CurDevice();
		DeviceTraits<Vulkan>::CopyToCPU(inputDevice, inputs[i].DType(), inputs[i].UnsafeRawData(),
		                                inputs[i].NumElements(), workspace.cpuInputs_[i].DType(),
		                                workspace.cpuInputs_[i].UnsafeRawData());
	}

	impl_->cpuModule.RunTensorsInto(workspace.cpuInputs_, workspace.cpuOutputs_);

	for (std::size_t i = 0; i < workspace.outputs_.size(); ++i)
	{
		DeviceTraits<Vulkan>::CopyFromCPU(workspace.outputs_[i].CurDevice(), workspace.outputs_[i].DType(),
		                                  workspace.outputs_[i].UnsafeRawData(), workspace.cpuOutputs_[i].DType(),
		                                  workspace.cpuOutputs_[i].UnsafeRawData(),
		                                  workspace.cpuOutputs_[i].NumElements());
	}
	return workspace.Outputs();
}

void CompiledModule<Vulkan>::RunTensorsInto(std::span<const Tensor<Vulkan>> inputs,
                                            std::span<Tensor<Vulkan>> outputs) const
{
	RunTensorsInto(inputs, outputs, CompiledModuleVulkanRunOptions{});
}

void CompiledModule<Vulkan>::RunTensorsInto(std::span<const Tensor<Vulkan>> inputs, std::span<Tensor<Vulkan>> outputs,
                                            CompiledModuleVulkanRunOptions options) const
{
	if (!impl_)
	{
		throw std::runtime_error("CompiledModule is empty");
	}
	if (inputs.size() != impl_->inputSpecs.size())
	{
		throw std::runtime_error(std::format("CompiledModule input count mismatch: expected {}, got {}",
		                                     impl_->inputSpecs.size(), inputs.size()));
	}
	if (outputs.size() != impl_->outputSpecs.size())
	{
		throw std::runtime_error(std::format("CompiledModule output count mismatch: expected {}, got {}",
		                                     impl_->outputSpecs.size(), outputs.size()));
	}
	for (std::size_t i = 0; i < inputs.size(); ++i)
	{
		ValidateTensorAgainstSpec(inputs[i], impl_->inputSpecs[i], i);
	}
	for (std::size_t i = 0; i < outputs.size(); ++i)
	{
		ValidateOutputTensorAgainstSpec(outputs[i], impl_->outputSpecs[i], i);
	}

	if (impl_->backend == CompiledModuleBackend::VulkanNative)
	{
		std::lock_guard lock(impl_->vulkanNativeRunMutex);
		RunVulkanNativePayload(impl_->vulkanPayload, impl_->vulkanModules, inputs, impl_->vulkanExternalTensors,
		                       impl_->vulkanWorkspaceTensors, outputs, options);
		return;
	}
	if (!options.synchronize)
	{
		throw std::runtime_error("CompiledModule<Vulkan> CPU bridge does not support asynchronous execution");
	}

	std::vector<Tensor<CPU>> cpuInputs;
	cpuInputs.reserve(inputs.size());
	for (const auto& input : inputs)
	{
		Tensor<CPU> cpuInput(Uninitialized, input.Shape(), input.DType(), CPU{});
		auto inputDevice = input.CurDevice();
		DeviceTraits<Vulkan>::CopyToCPU(inputDevice, input.DType(), input.UnsafeRawData(), input.NumElements(),
		                                cpuInput.DType(), cpuInput.UnsafeRawData());
		cpuInputs.push_back(std::move(cpuInput));
	}

	std::vector<Tensor<CPU>> cpuOutputs;
	cpuOutputs.reserve(impl_->outputSpecs.size());
	for (const auto& spec : impl_->outputSpecs)
	{
		cpuOutputs.emplace_back(Uninitialized, ShapeView{ spec.type.StaticShape() }, spec.type.dtype, CPU{});
	}
	impl_->cpuModule.RunTensorsInto(cpuInputs, cpuOutputs);

	for (std::size_t i = 0; i < outputs.size(); ++i)
	{
		DeviceTraits<Vulkan>::CopyFromCPU(outputs[i].CurDevice(), outputs[i].DType(), outputs[i].UnsafeRawData(),
		                                  cpuOutputs[i].DType(), cpuOutputs[i].UnsafeRawData(),
		                                  cpuOutputs[i].NumElements());
	}
}

void CompiledModule<Vulkan>::RunManyTensorsInto(std::span<const CompiledModuleVulkanTensorInvocation> invocations,
                                                std::size_t threadCount) const
{
	const auto workerCount = NormalizeThreadCount(threadCount, invocations.size());
	if (workerCount == 0)
	{
		return;
	}
	if (workerCount > 1 && std::ranges::any_of(invocations, [](const CompiledModuleVulkanTensorInvocation& invocation) {
		    return invocation.options.profileEvents != nullptr;
	    }))
	{
		throw std::runtime_error(
		    "CompiledModule<Vulkan> profile event sinks are not thread-safe with RunManyTensorsInto");
	}
	if (workerCount == 1)
	{
		for (const auto& invocation : invocations)
		{
			RunTensorsInto(invocation.inputs, invocation.outputs, invocation.options);
		}
		return;
	}

	std::atomic<std::size_t> next{ 0 };
	std::atomic_bool stop{ false };
	std::exception_ptr firstError;
	std::mutex errorMutex;

	auto worker = [&] {
		while (!stop.load(std::memory_order_relaxed))
		{
			const auto index = next.fetch_add(1, std::memory_order_relaxed);
			if (index >= invocations.size())
			{
				break;
			}
			try
			{
				const auto& invocation = invocations[index];
				RunTensorsInto(invocation.inputs, invocation.outputs, invocation.options);
			}
			catch (...)
			{
				{
					std::lock_guard lock(errorMutex);
					if (!firstError)
					{
						firstError = std::current_exception();
					}
				}
				stop.store(true, std::memory_order_relaxed);
				break;
			}
		}
	};

	std::vector<std::thread> workers;
	workers.reserve(workerCount);
	for (std::size_t i = 0; i < workerCount; ++i)
	{
		workers.emplace_back(worker);
	}
	for (auto& thread : workers)
	{
		thread.join();
	}
	if (firstError)
	{
		std::rethrow_exception(firstError);
	}
}

CompiledModuleImage CompiledModule<Vulkan>::Image() const
{
	return impl_ ? CompiledModuleImage{
	                   .rodata = impl_->rodata.data(),
	                   .rodataSize = impl_->rodata.size(),
	                   .instructions = impl_->instructions.data(),
	                   .instructionSize = impl_->instructions.size(),
	               }
	             : CompiledModuleImage{};
}

std::span<const std::byte> CompiledModule<Vulkan>::Rodata() const
{
	return impl_ ? std::span<const std::byte>{ impl_->rodata.data(), impl_->rodata.size() }
	             : std::span<const std::byte>{};
}

std::span<const std::byte> CompiledModule<Vulkan>::Instructions() const
{
	return impl_ ? std::span<const std::byte>{ impl_->instructions.data(), impl_->instructions.size() }
	             : std::span<const std::byte>{};
}

std::span<const CompiledTensorSpec> CompiledModule<Vulkan>::InputSpecs() const
{
	return impl_ ? std::span<const CompiledTensorSpec>{ impl_->inputSpecs.data(), impl_->inputSpecs.size() }
	             : std::span<const CompiledTensorSpec>{};
}

std::span<const CompiledTensorSpec> CompiledModule<Vulkan>::OutputSpecs() const
{
	return impl_ ? std::span<const CompiledTensorSpec>{ impl_->outputSpecs.data(), impl_->outputSpecs.size() }
	             : std::span<const CompiledTensorSpec>{};
}

CompiledModuleBackend CompiledModule<Vulkan>::Backend() const
{
	return impl_ ? impl_->backend : CompiledModuleBackend::CPUNative;
}

std::optional<std::size_t> CompiledModule<Vulkan>::FindInput(std::string_view name) const
{
	return impl_ ? FindSpecIndex(impl_->inputSpecs, name) : std::nullopt;
}

std::optional<std::size_t> CompiledModule<Vulkan>::FindOutput(std::string_view name) const
{
	return impl_ ? FindSpecIndex(impl_->outputSpecs, name) : std::nullopt;
}

void CompiledModule<Vulkan>::WriteObjectFile(const std::filesystem::path& path, std::string_view symbolPrefix) const
{
	if (!impl_)
	{
		throw std::runtime_error("CompiledModule is empty");
	}
	const auto objectBytes = EmitCarrierObject(impl_->rodata, impl_->instructions, symbolPrefix);
	WriteAllBytes(path, objectBytes);
}
#endif

namespace
{
	Graph BuildCompilerGraphFromPlan(const ExecutablePlan& plan)
	{
		ValidateExecutablePlan(plan);
		Graph graph;

		for (std::size_t i = 0; i < plan.variables.size(); ++i)
		{
			const auto& storage = plan.variables[i];
			if (!storage.type.IsFullyStatic())
			{
				throw std::runtime_error(std::format("Cannot compile plan variable {} with non-static tensor type", i));
			}
			if (!storage.region.data)
			{
				throw std::runtime_error(std::format("Cannot compile plan variable {} without bound storage", i));
			}
			if (storage.type.memorySpace == TensorMemorySpace::Device)
			{
				throw std::runtime_error(
				    std::format("Cannot compile plan variable {} from device-only storage without host binding", i));
			}
			const auto shape = storage.type.StaticShape();
			const auto* bytes = static_cast<const std::byte*>(storage.region.data) + storage.region.byteOffset +
			                    storage.storageOffsetBytes;
			auto hostView = Tensor<PolymorphicDevice>::UnsafeBorrowed(const_cast<std::byte*>(bytes), ShapeView{ shape },
			                                                          storage.type.dtype, PolymorphicDevice{ CPU{} });
			if (storage.quantization)
			{
				graph.AddVariable(Variable::CreateFrozenQuantized(std::move(hostView), *storage.quantization));
			}
			else
			{
				graph.AddVariable(Variable::CreateFrozen(std::move(hostView)));
			}
		}

		for (const auto& type : plan.activationSlots)
		{
			graph.AddActivationSlot(type);
		}
		for (const auto& type : plan.tapeSlots)
		{
			graph.AddTapeSlot(type);
		}

		for (const auto& planSubgraph : plan.subgraphs)
		{
			Subgraph subgraph;
			for (std::size_t paramIndex = 0; paramIndex < planSubgraph.params.size(); ++paramIndex)
			{
				if (paramIndex >= planSubgraph.nodes.size())
				{
					throw std::runtime_error("ExecutablePlan subgraph has fewer nodes than params");
				}
				const auto* paramNode = std::get_if<ParamRefNode>(&planSubgraph.nodes[paramIndex].node);
				if (!paramNode || paramNode->paramIndex != paramIndex)
				{
					throw std::runtime_error(
					    "ExecutablePlan compiler bridge expects leading ParamRefNode entries for subgraph params");
				}
				subgraph.AddParam(planSubgraph.params[paramIndex]);
			}
			for (std::size_t nodeIndex = planSubgraph.params.size(); nodeIndex < planSubgraph.nodes.size(); ++nodeIndex)
			{
				const auto& planNode = planSubgraph.nodes[nodeIndex];
				std::vector<OutputInfo> outputs;
				outputs.reserve(planNode.outputs.size());
				for (const auto& output : planNode.outputs)
				{
					outputs.push_back(OutputInfo::FromType(output));
				}
				subgraph.AddNode(planNode.node, std::move(outputs));
			}
			subgraph.SetResults(std::vector<NodeOutput>(planSubgraph.results.begin(), planSubgraph.results.end()));
			const auto id = graph.AddSubgraph(std::move(subgraph));
			if (id != planSubgraph.sourceSubgraph)
			{
				throw std::runtime_error("ExecutablePlan subgraph order is not compatible with compiler lowering");
			}
		}

		graph.SetForward(plan.forward);
		if (plan.backward)
		{
			graph.SetBackward(*plan.backward);
		}

		std::vector<std::string> inputNames;
		inputNames.reserve(plan.inputs.size());
		for (std::size_t i = 0; i < plan.inputs.size(); ++i)
		{
			inputNames.push_back(plan.inputs[i].name.empty() ? std::format("input{}", i) : plan.inputs[i].name);
		}
		graph.SetInputNames(std::move(inputNames));

		std::vector<std::string> outputNames;
		outputNames.reserve(plan.outputs.size());
		for (std::size_t i = 0; i < plan.outputs.size(); ++i)
		{
			outputNames.push_back(plan.outputs[i].name.empty() ? std::format("output{}", i) : plan.outputs[i].name);
		}
		graph.SetOutputNames(std::move(outputNames));

		std::vector<std::string> variableNames;
		variableNames.reserve(plan.variables.size());
		for (std::size_t i = 0; i < plan.variables.size(); ++i)
		{
			const auto& name = i < plan.variableNames.size() ? plan.variableNames[i] : plan.variables[i].region.name;
			variableNames.push_back(name.empty() ? std::format("variable{}", i) : name);
		}
		graph.SetVariableNames(std::move(variableNames));

		Validation::ValidateGraph(graph);
		return graph;
	}

	CompiledArtifactParts CompileCPUArtifactPartsFromGraph(const Graph& graph, const CompilerOptions& options)
	{
		Validation::ValidateGraph(graph);
		if (auto parallelParts = TryCompileCPUParallelLinearChainF32WithExternalRegionFusion(graph, options))
		{
			return MakeCompiledArtifactParts(std::move(parallelParts->rodata), std::move(parallelParts->instructions),
			                                 std::move(parallelParts->inputSpecs),
			                                 std::move(parallelParts->outputSpecs), CompiledModuleBackend::CPUNative,
			                                 std::move(parallelParts->constants), std::move(parallelParts->weights),
			                                 std::move(parallelParts->externalTensorInfos));
		}
		if (auto externalParts = TryCompileCPUMLIRExternalRegions(graph, options))
		{
			return MakeCompiledArtifactParts(std::move(externalParts->rodata), std::move(externalParts->instructions),
			                                 std::move(externalParts->inputSpecs),
			                                 std::move(externalParts->outputSpecs), CompiledModuleBackend::CPUNative,
			                                 std::move(externalParts->constants), std::move(externalParts->weights),
			                                 std::move(externalParts->externalTensorInfos));
		}

		mlir::MLIRContext ctx;
		SetupCompilerMLIRContext(ctx);
		auto mlirModule = BuildLoweredMLIRModule(graph, ctx, options);

		llvm::LLVMContext llvmCtx;
		auto llvmModule = litenn::translateToLLVMIR(*mlirModule, llvmCtx);
		if (!llvmModule)
		{
			throw std::runtime_error("Failed to translate lowered MLIR module to LLVM IR");
		}

		auto config = CreateNativeTargetMachine();
		ConfigureForNativeObject(*llvmModule, config);

		const auto inputSpecs = BuildInputSpecs(graph);
		const auto outputSpecs = BuildOutputSpecs(graph);
		AddUniformEntryWrapper(*llvmModule, "subgraph_" + std::to_string(graph.Forward()), inputSpecs, outputSpecs);
		OptimizeLLVMModule(*llvmModule, *config.targetMachine, options.cpuAOTLLVMOptLevel);

		auto rodata = SerializeRodata(inputSpecs, outputSpecs, config.triple, CompiledModuleBackend::CPUNative);
		auto instructions = EmitObjectFile(*llvmModule);
		return MakeCompiledArtifactParts(std::move(rodata), std::move(instructions), inputSpecs, outputSpecs,
		                                 CompiledModuleBackend::CPUNative);
	}

#ifdef LITENN_ENABLE_CUDA
	CompiledArtifactParts CompileCUDAArtifactPartsFromGraph(const Graph& graph, const CompilerOptions& options)
	{
		Validation::ValidateGraph(graph);
		if (options.enableCUDANativeAOT)
		{
			if (auto nativeParts = TryCompileCUDANativeCast(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeUnaryF32(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeSiLUF32(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeGGMLBlockQuantizedMatMul(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeLinearChain(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeMatMulBias(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeMatMulF32(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeBatchMatMulF32(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeScatterUpdateF32(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeMatMulLowPrecision(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeBinaryF32(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeReduceF32(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeSoftmaxF32(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeGetRowsF32(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeRMSNormF32(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeRoPEF32(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeConcatF32(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
			if (auto nativeParts = TryCompileCUDANativeSliceF32(graph))
			{
				return MakeCompiledArtifactParts(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
				                                 std::move(nativeParts->inputSpecs),
				                                 std::move(nativeParts->outputSpecs),
				                                 CompiledModuleBackend::CUDANative);
			}
		}
		return CompileCPUArtifactPartsFromGraph(graph, options);
	}
#endif
#ifdef LITENN_ENABLE_VULKAN
	CompiledArtifactParts CompileVulkanArtifactPartsFromGraph(const Graph& graph, const CompilerOptions& options)
	{
		Validation::ValidateGraph(graph);
		if (options.enableVulkanNativeAOT)
		{
			if (auto nativeParts = TryCompileVulkanNativeP0(graph))
			{
				return MakeVulkanNativeCompiledArtifactParts(std::move(*nativeParts));
			}

			Graph optimized = graph;
			FusionPass{}.Run(optimized);
			if (auto nativeParts = TryCompileVulkanNativeP0(optimized))
			{
				return MakeVulkanNativeCompiledArtifactParts(std::move(*nativeParts));
			}

			const auto report = DiagnoseVulkanNativeSupport(graph);
			if (!report.supported)
			{
				LogCompileDiagnostic(options, "vulkan-native unsupported: " + report.reason);
			}
		}
		return CompileCPUArtifactPartsFromGraph(graph, options);
	}
#endif
} // namespace

CompiledModuleArtifact Compiler<CPU>::CompileArtifact(const ExecutablePlan& plan)
{
	return CompileArtifact(plan, CompilerOptions::Defaults());
}

CompiledModuleArtifact Compiler<CPU>::CompileArtifact(const ExecutablePlan& plan, const CompilerOptions& options)
{
	ValidateExecutablePlan(plan);
	auto parts = CompileCPUArtifactPartsFromGraph(BuildCompilerGraphFromPlan(plan), options);
	return CompiledModuleArtifact(std::move(parts.rodata), std::move(parts.instructions), std::move(parts.inputSpecs),
	                              std::move(parts.outputSpecs), parts.backend, std::move(parts.constants),
	                              std::move(parts.weights), std::move(parts.externalTensorInfos));
}

CompiledModule<CPU> Compiler<CPU>::Compile(const ExecutablePlan& plan)
{
	return Compile(plan, CompilerOptions::Defaults());
}

CompiledModule<CPU> Compiler<CPU>::Compile(const ExecutablePlan& plan, const CompilerOptions& options)
{
	auto artifact = CompileArtifact(plan, options);
	return std::move(artifact).Load();
}

#ifdef LITENN_ENABLE_CUDA
CompiledModuleArtifact Compiler<CUDA>::CompileArtifact(const ExecutablePlan& plan)
{
	return CompileArtifact(plan, CompilerOptions::Defaults());
}

CompiledModuleArtifact Compiler<CUDA>::CompileArtifact(const ExecutablePlan& plan, const CompilerOptions& options)
{
	ValidateExecutablePlan(plan);
	auto parts = CompileCUDAArtifactPartsFromGraph(BuildCompilerGraphFromPlan(plan), options);
	return CompiledModuleArtifact(std::move(parts.rodata), std::move(parts.instructions), std::move(parts.inputSpecs),
	                              std::move(parts.outputSpecs), parts.backend, std::move(parts.constants),
	                              std::move(parts.weights), std::move(parts.externalTensorInfos));
}

CompiledModule<CUDA> Compiler<CUDA>::Compile(const ExecutablePlan& plan, CUDA device)
{
	return Compile(plan, std::move(device), CompilerOptions::Defaults());
}

CompiledModule<CUDA> Compiler<CUDA>::Compile(const ExecutablePlan& plan, const CompilerOptions& options)
{
	return Compile(plan, CUDA{}, options);
}

CompiledModule<CUDA> Compiler<CUDA>::Compile(const ExecutablePlan& plan, CUDA device, const CompilerOptions& options)
{
	return CompileArtifact(plan, options).Load(std::move(device));
}
#endif

#ifdef LITENN_ENABLE_VULKAN
VulkanNativeSupportReport Compiler<Vulkan>::QueryNativeSupport(const ExecutablePlan& plan)
{
	ValidateExecutablePlan(plan);
	auto graph = BuildCompilerGraphFromPlan(plan);
	auto report = DiagnoseVulkanNativeSupport(graph);
	if (report.supported)
	{
		return report;
	}
	FusionPass{}.Run(graph);
	auto fusedReport = DiagnoseVulkanNativeSupport(graph);
	return fusedReport.supported ? fusedReport : report;
}

CompiledModuleArtifact Compiler<Vulkan>::CompileArtifact(const ExecutablePlan& plan)
{
	return CompileArtifact(plan, CompilerOptions::Defaults());
}

CompiledModuleArtifact Compiler<Vulkan>::CompileArtifact(const ExecutablePlan& plan, const CompilerOptions& options)
{
	ValidateExecutablePlan(plan);
	auto parts = CompileVulkanArtifactPartsFromGraph(BuildCompilerGraphFromPlan(plan), options);
	return CompiledModuleArtifact(std::move(parts.rodata), std::move(parts.instructions), std::move(parts.inputSpecs),
	                              std::move(parts.outputSpecs), parts.backend, std::move(parts.constants),
	                              std::move(parts.weights), std::move(parts.externalTensorInfos));
}

CompiledModule<Vulkan> Compiler<Vulkan>::Compile(const ExecutablePlan& plan, Vulkan device)
{
	return Compile(plan, std::move(device), CompilerOptions::Defaults());
}

CompiledModule<Vulkan> Compiler<Vulkan>::Compile(const ExecutablePlan& plan, const CompilerOptions& options)
{
	return Compile(plan, Vulkan{}, options);
}

CompiledModule<Vulkan> Compiler<Vulkan>::Compile(const ExecutablePlan& plan, Vulkan device,
                                                 const CompilerOptions& options)
{
	return CompileArtifact(plan, options).Load(std::move(device));
}
#endif
