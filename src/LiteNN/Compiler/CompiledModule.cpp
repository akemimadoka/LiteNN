#include "CompiledModule.h"

#include "CUDANativeCodegen.h"
#include "CUDANativePayload.h"
#include "Dialect/LiteNNDialect.h"
#include "Dialect/LiteNNOps.h"
#include "Pass/BufferizationPipeline.h"
#include "Pass/LLVMCodegenPipeline.h"
#include "Pass/LowerLiteNNPass.h"
#include "Translation/GraphToMLIR.h"

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
#include "llvm/Support/Error.h"
#include "llvm/Support/DynamicLibrary.h"
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

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <fstream>
#include <limits>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <utility>
#include <variant>

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

	bool IsCUDANativeAOTDisabled()
	{
		const char* value = std::getenv("LITENN_CUDA_DISABLE_NATIVE_AOT");
		if (!value)
		{
			return false;
		}
		const std::string_view text = value;
		return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
	}

	using LiteNNCPUParallelForBody = void (*)(std::uint64_t begin, std::uint64_t end, void* userData);

	std::size_t LiteNNCPUAOTThreadCount()
	{
		if (const char* value = std::getenv("LITENN_CPU_AOT_THREADS"))
		{
			char* end = nullptr;
			const auto parsed = std::strtoull(value, &end, 10);
			if (end != value && parsed > 0)
			{
				return static_cast<std::size_t>(parsed);
			}
		}
		const auto hardware = std::thread::hardware_concurrency();
		return hardware == 0 ? 1 : static_cast<std::size_t>(hardware);
	}

	std::size_t LiteNNCPUMaxThreadCount()
	{
		const auto hardware = std::thread::hardware_concurrency();
		return hardware == 0 ? 1 : static_cast<std::size_t>(hardware);
	}

	std::uint64_t LiteNNCPUParallelMinFlops()
	{
		if (const char* value = std::getenv("LITENN_CPU_AOT_PARALLEL_MIN_FLOPS"))
		{
			char* end = nullptr;
			const auto parsed = std::strtoull(value, &end, 10);
			if (end != value)
			{
				return static_cast<std::uint64_t>(parsed);
			}
		}
		return 1ull << 28;
	}

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

		void ParallelFor(std::uint64_t begin, std::uint64_t end, std::uint64_t grain,
		                 LiteNNCPUParallelForBody body, void* userData, std::size_t requestedThreads)
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
			const auto desiredWorkers = std::min<std::size_t>(
			    static_cast<std::size_t>(participantCount - 1), workers_.size());
			{
				std::lock_guard lock(mutex_);
				begin_ = begin;
				end_ = end;
				grain_ = grain;
				body_ = body;
				userData_ = userData;
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
			std::uint64_t seenGeneration = 0;
			while (true)
			{
				bool participate = false;
				{
					std::unique_lock lock(mutex_);
					start_.wait(lock, [&] { return stopping_ || generation_ != seenGeneration; });
					if (stopping_)
					{
						return;
					}
					seenGeneration = generation_;
					participate = workerIndex < desiredWorkers_;
				}

				if (participate)
				{
					RunTasks();

					std::lock_guard lock(mutex_);
					++workersDone_;
					if (workersDone_ == desiredWorkers_)
					{
						done_.notify_one();
					}
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
	                          LiteNNCPUParallelForBody body, void* userData)
	{
		GetLiteNNCPUThreadPool().ParallelFor(begin, end, grain, body, userData, LiteNNCPUAOTThreadCount());
	}

	void LiteNNCPUMatMulBiasReLURange(const float* LITENN_RESTRICT lhs, const float* LITENN_RESTRICT rhs,
	                                  const float* LITENN_RESTRICT bias, float* LITENN_RESTRICT out,
	                                  std::uint64_t rowBegin, std::uint64_t rowEnd, std::uint64_t k,
	                                  std::uint64_t n, std::uint64_t biasRows, bool relu)
	{
		for (std::uint64_t row = rowBegin; row < rowEnd; ++row)
		{
			float* outRow = out + row * n;
			const float* biasRow = bias + (biasRows == 1 ? 0 : row) * n;
			std::memcpy(outRow, biasRow, static_cast<std::size_t>(n) * sizeof(float));

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

			if (relu)
			{
				LITENN_GCC_IVDEP
				for (std::uint64_t col = 0; col < n; ++col)
				{
					if (outRow[col] < 0.0f)
					{
						outRow[col] = 0.0f;
					}
				}
			}
		}
	}

	void LiteNNCPUMatMulBiasReLUParallel(const float* LITENN_RESTRICT lhs, const float* LITENN_RESTRICT rhs,
	                                     const float* LITENN_RESTRICT bias, float* LITENN_RESTRICT out,
	                                     std::uint64_t m, std::uint64_t k, std::uint64_t n,
	                                     std::uint64_t biasRows, bool relu)
	{
		const auto flops = m * k * n * 2;
		const auto threadCount = std::min<std::uint64_t>(LiteNNCPUAOTThreadCount(), m);
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
			LiteNNCPUMatMulBiasReLURange(ctx.lhs, ctx.rhs, ctx.bias, ctx.out, begin, end,
			                             ctx.k, ctx.n, ctx.biasRows, ctx.relu);
		};

		const auto grain = std::max<std::uint64_t>(1, (m + threadCount * 4 - 1) / (threadCount * 4));
		LiteNNCPUParallelFor(0, m, grain, body, &context);
	}

	extern "C" void litenn_cpu_matmul_bias_relu_parallel_f32(const float* lhs, const float* rhs,
	                                                         const float* bias, float* out,
	                                                         std::uint64_t m, std::uint64_t k,
	                                                         std::uint64_t n, std::uint64_t biasRows,
	                                                         bool relu)
	{
		LiteNNCPUMatMulBiasReLUParallel(lhs, rhs, bias, out, m, k, n, biasRows, relu);
	}

	constexpr std::string_view kEntrySymbol = "litenn_forward";
	constexpr std::array<std::byte, 8> kRodataMagic = {
		std::byte{ 'L' }, std::byte{ 'T' }, std::byte{ 'N' }, std::byte{ 'N' },
		std::byte{ 'C' }, std::byte{ 'M' }, std::byte{ '0' }, std::byte{ 0 },
	};
	constexpr std::uint32_t kRodataVersion = 4;
	constexpr std::uint32_t kRodataLittleEndian = 1;
	constexpr std::uint32_t kRodataBigEndian = 2;

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
		if (targetTriple.isOSDarwin() && targetTriple.getArch() == llvm::Triple::aarch64)
		{
			// MCJIT may place code outside ARM64 branch26 range from libSystem; use
			// address materialization for external calls instead of direct BL relocations.
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

	void OptimizeLLVMModule(llvm::Module& module, llvm::TargetMachine& targetMachine)
	{
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

		auto modulePipeline = passBuilder.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
		modulePipeline.run(module, moduleAnalysisManager);
	}

	std::vector<std::byte> EmitObjectFile(llvm::Module& module)
	{
		auto config = CreateNativeTargetMachine();
		ConfigureForNativeObject(module, config);

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

	void AppendQuantizationParams(std::vector<std::byte>& out, const QuantizationParams& params)
	{
		AppendU32(out, static_cast<std::uint32_t>(params.scheme));
		AppendU32(out, static_cast<std::uint32_t>(params.granularity));
		AppendU32(out, static_cast<std::uint32_t>(params.blockFormat));
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
			AppendU32(rodata, static_cast<std::uint32_t>(spec.dtype));
			AppendU32(rodata, static_cast<std::uint32_t>(spec.shape.size()));
			for (auto dim : spec.shape)
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
			spec.dtype = static_cast<DataType>(dtypeValue);

			const auto rank = ReadU32(rodata, offset);
			spec.shape.reserve(rank);
			for (std::uint32_t i = 0; i < rank; ++i)
			{
				const auto dim = ReadU64(rodata, offset);
				if (dim > std::numeric_limits<std::size_t>::max())
				{
					throw std::runtime_error("Compiled module rodata shape dimension is too large");
				}
				spec.shape.push_back(static_cast<std::size_t>(dim));
			}
			if (version >= 2)
			{
				spec.name = ReadString(rodata, offset);
			}
			if (version >= 4 && ReadU32(rodata, offset) != 0)
			{
				spec.quantization = ReadQuantizationParams(rodata, offset);
				try
				{
					ValidateQuantizationParams(*spec.quantization, ShapeView{ spec.shape }, spec.dtype);
				}
				catch (const std::exception& ex)
				{
					throw std::runtime_error(std::format("Compiled module rodata quantization metadata is invalid: {}",
					                                     ex.what()));
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

	std::vector<CompiledTensorSpec> BuildInputSpecs(const Graph& graph)
	{
		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		std::vector<CompiledTensorSpec> specs;
		specs.reserve(subgraph.Params().size());
		for (std::size_t i = 0; i < subgraph.Params().size(); ++i)
		{
			const auto& param = subgraph.Params()[i];
			specs.push_back({ param.dtype, param.shape, graph.InputName(i) });
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
		std::vector<CompiledTensorSpec> specs;
		specs.reserve(subgraph.Results().size());
		for (std::size_t i = 0; i < subgraph.Results().size(); ++i)
		{
			const auto output = subgraph.Results()[i];
			const auto& info = subgraph.GetOutputInfo(output);
			specs.push_back({ info.dtype, info.shape, graph.OutputName(i), InferOutputQuantization(graph, subgraph, output) });
		}
		return specs;
	}

	struct CompiledArtifactParts
	{
		std::vector<std::byte> rodata;
		std::vector<std::byte> instructions;
		std::vector<CompiledTensorSpec> inputSpecs;
		std::vector<CompiledTensorSpec> outputSpecs;
	};

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
		if (cpuTensor.DType() != DataType::Float32 || !Validation::SameShape(cpuTensor.Shape().Dims, expectedShape))
		{
			return std::nullopt;
		}
		std::vector<float> values(cpuTensor.NumElements());
		std::memcpy(values.data(), cpuTensor.RawData(), values.size() * sizeof(float));
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

	std::optional<CompiledArtifactParts> TryCompileCPUParallelLinearChainF32(const Graph& graph)
	{
		if (LiteNNCPUAOTThreadCount() <= 1)
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
		    llvm::FunctionType::get(voidTy, { ptrTy, ptrTy, ptrTy, ptrTy, i64Ty, i64Ty, i64Ty, i64Ty, i1Ty }, false));

		std::vector<std::optional<ValueRef>> values(subgraph.NodeCount());
		std::vector<llvm::Value*> heapAllocations;
		std::size_t fusedLayerCount = 0;
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
		const auto tensorBytes = [&](const OutputInfo& info) {
			return TensorByteSizeForShape(info.dtype, info.shape);
		};

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
				values[nodeId] = ValueRef{
					.ptr = AddF32ConstantGlobal(*module, builder,
					                            std::format("litenn_cpu_const_{}", nodeId), *constantData),
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
				values[nodeId] = ValueRef{
					.ptr = AddF32ConstantGlobal(*module, builder,
					                            std::format("litenn_cpu_const_{}", nodeId), *constantData),
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
			    bias->dtype != DataType::Float32 || output.dtype != DataType::Float32 ||
			    lhs->shape.size() != 2 || rhs->shape.size() != 2 || output.shape.size() != 2 ||
			    bias->shape.size() != output.shape.size() || lhs->shape[1] != rhs->shape[0] ||
			    output.shape[0] != lhs->shape[0] || output.shape[1] != rhs->shape[1] ||
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
			totalFlops = SaturatedAddU64(totalFlops, layerFlops);
			builder.CreateCall(kernelFn, { lhs->ptr, rhs->ptr, bias->ptr, outPtr, builder.getInt64(m),
			                               builder.getInt64(k), builder.getInt64(n),
			                               builder.getInt64(static_cast<std::uint64_t>(bias->shape[0])),
			                               builder.getInt1(fused->pattern == FusionPattern::MatMulBiasAddReLU) });
			values[nodeId] = ValueRef{ .ptr = outPtr, .dtype = output.dtype, .shape = output.shape };
			++fusedLayerCount;
		}

		if (fusedLayerCount == 0 || !values[finalResult.node] || totalFlops < LiteNNCPUParallelMinFlops())
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
		OptimizeLLVMModule(*module, *config.targetMachine);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, config.triple, CompiledModuleBackend::CPUNative);
		auto instructions = EmitObjectFile(*module);
		return CompiledArtifactParts{ std::move(rodata), std::move(instructions), inputSpecs, outputSpecs };
	}

	std::uint64_t NumElements(const CompiledTensorSpec& spec)
	{
		std::uint64_t n = 1;
		for (const auto dim : spec.shape)
		{
			n *= static_cast<std::uint64_t>(dim);
		}
		return n;
	}

	std::vector<std::uint64_t> ContiguousStrides(const CompiledTensorSpec& spec)
	{
		std::vector<std::uint64_t> strides(spec.shape.size());
		if (!strides.empty())
		{
			strides.back() = 1;
			for (std::size_t i = strides.size() - 1; i > 0; --i)
			{
				strides[i - 1] = strides[i] * static_cast<std::uint64_t>(spec.shape[i]);
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
		auto* descTy = GetMemRefDescriptorType(ctx, spec.shape.size());
		llvm::Value* desc = llvm::PoisonValue::get(descTy);
		desc = builder.CreateInsertValue(desc, data, { 0 });
		desc = builder.CreateInsertValue(desc, data, { 1 });
		desc = builder.CreateInsertValue(desc, builder.getInt64(0), { 2 });

		std::vector<std::uint64_t> sizes(spec.shape.begin(), spec.shape.end());
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
		const auto byteCount = NumElements(spec) * LiteNN::ElementByteSize(spec.dtype);
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

	void AddUniformEntryWrapper(llvm::Module& module, std::string_view calleeName,
	                            std::span<const CompiledTensorSpec> inputs, std::span<const CompiledTensorSpec> outputs)
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
		descriptors.reserve(inputs.size());
		for (std::size_t i = 0; i < inputs.size(); ++i)
		{
			auto* inputSlot = builder.CreateGEP(ptrTy, inputArray, builder.getInt64(i));
			auto* inputData = builder.CreateLoad(ptrTy, inputSlot);
			descriptors.push_back(BuildMemRefDescriptor(builder, inputData, inputs[i]));
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
			if (outputs.size() == 1 && IsMemRefDescriptorType(ctx, sretType, outputs[0].shape.size()))
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
			auto* expectedDescTy = GetMemRefDescriptorType(ctx, outputs[0].shape.size());
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
			throw std::runtime_error(std::format(
			    "Compiled module exported symbol '{}' does not fit in size_t on this host", label));
		}
		return static_cast<std::size_t>(rawSize);
	}

	struct LoadedJIT
	{
		std::unique_ptr<llvm::LLVMContext> context;
		std::unique_ptr<llvm::ExecutionEngine> engine;
		EntryFn entry{};
	};

	void RegisterJITRuntimeSymbol(std::string_view name, void* address)
	{
		const auto symbolName = std::string(name);
		llvm::sys::DynamicLibrary::AddSymbol(symbolName, address);
		llvm::sys::DynamicLibrary::AddSymbol("_" + symbolName, address);
	}

	LoadedJIT LoadJIT(std::span<const std::byte> instructions)
	{
		InitializeNativeLLVM();
		RegisterJITRuntimeSymbol("malloc", reinterpret_cast<void*>(static_cast<void* (*)(std::size_t)>(&std::malloc)));
		RegisterJITRuntimeSymbol("free", reinterpret_cast<void*>(static_cast<void (*)(void*)>(&std::free)));
		RegisterJITRuntimeSymbol("litenn_cpu_matmul_bias_relu_parallel_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_matmul_bias_relu_parallel_f32));

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

	void AddByteArraySymbol(llvm::Module& module, std::string_view name, std::span<const std::byte> bytes)
	{
		auto& ctx = module.getContext();
		auto* init = ByteArrayConstant(ctx, bytes);
		auto* global = new llvm::GlobalVariable(module, init->getType(), true, llvm::GlobalValue::ExternalLinkage, init,
		                                        std::string(name));
		global->setAlignment(llvm::Align(1));
	}

	void AddSizeSymbol(llvm::Module& module, std::string_view name, std::size_t size)
	{
		auto& ctx = module.getContext();
		auto* i64Ty = llvm::Type::getInt64Ty(ctx);
		auto* init = llvm::ConstantInt::get(i64Ty, static_cast<std::uint64_t>(size));
		new llvm::GlobalVariable(module, i64Ty, true, llvm::GlobalValue::ExternalLinkage, init, std::string(name));
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

	void WriteAllBytes(const std::filesystem::path& path, std::span<const std::byte> bytes)
	{
		std::ofstream out(path, std::ios::binary);
		if (!out)
		{
			throw std::runtime_error("Failed to open output object file");
		}
		out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		if (!out)
		{
			throw std::runtime_error("Failed to write output object file");
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
		if (tensor.DType() != spec.dtype || !std::ranges::equal(tensor.Shape().Dims, spec.shape))
		{
			const auto label =
			    spec.name.empty() ? std::to_string(inputIndex) : std::format("{} ('{}')", inputIndex, spec.name);
			throw std::runtime_error(std::format("CompiledModule input {} mismatch: expected {}, got {}", label,
			                                     Validation::FormatInfo(spec.dtype, spec.shape),
			                                     Validation::FormatInfo(tensor.DType(), tensor.Shape().Dims)));
		}
	}

	template <Device D>
	void ValidateOutputTensorAgainstSpec(const Tensor<D>& tensor, const CompiledTensorSpec& spec,
	                                     std::size_t outputIndex)
	{
		if (tensor.DType() != spec.dtype || !std::ranges::equal(tensor.Shape().Dims, spec.shape))
		{
			const auto label =
			    spec.name.empty() ? std::to_string(outputIndex) : std::format("{} ('{}')", outputIndex, spec.name);
			throw std::runtime_error(std::format("CompiledModule output {} mismatch: expected {}, got {}", label,
			                                     Validation::FormatInfo(spec.dtype, spec.shape),
			                                     Validation::FormatInfo(tensor.DType(), tensor.Shape().Dims)));
		}
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
	struct CUDANativeBinaryPlan
	{
		BinaryOp op{ BinaryOp::Add };
		std::uint32_t lhsInputIndex{};
		std::uint32_t rhsInputIndex{};
		std::uint32_t elementCount{};
		std::uint32_t lhsElementCount{};
		std::uint32_t rhsElementCount{};
		bool requiresBroadcast{};
		std::vector<std::size_t> outputShape;
		std::vector<std::size_t> lhsShape;
		std::vector<std::size_t> rhsShape;
	};

	struct CUDANativeUnaryPlan
	{
		UnaryOp op{ UnaryOp::Negate };
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

	struct CUDANativeTensorRef
	{
		CUDANativeArgumentKind kind{ CUDANativeArgumentKind::InputTensor };
		std::uint32_t index{};
		std::uint64_t byteOffset{};
		std::uint64_t byteSize{};
		DataType dtype{ DataType::Float32 };
		std::vector<std::size_t> shape;
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

	bool SameShape(std::span<const std::size_t> lhs, std::span<const std::size_t> rhs)
	{
		return std::ranges::equal(lhs, rhs);
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
		const auto* begin = reinterpret_cast<const std::byte*>(cpuTensor.RawData());
		payload.constantData.insert(payload.constantData.end(), begin, begin + byteSize);
		return offset;
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
		if (param.dtype != DataType::Float32 || output.dtype != DataType::Float32 ||
		    !SameShape(param.shape, output.shape))
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

	std::optional<CUDANativeBinaryPlan> MatchCUDANativeBinaryF32(const Graph& graph)
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
		if (!binary || !IsSupportedCUDANativeBinaryF32Op(binary->op))
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
		if (output.dtype != DataType::Float32)
		{
			return std::nullopt;
		}
		if (lhsParam.dtype != DataType::Float32 || rhsParam.dtype != DataType::Float32 ||
		    !IsSameRankBroadcastCompatible(lhsParam.shape, rhsParam.shape, output.shape))
		{
			return std::nullopt;
		}

		const auto elementCount = ShapeView{ output.shape }.NumElements();
		const auto lhsElementCount = ShapeView{ lhsParam.shape }.NumElements();
		const auto rhsElementCount = ShapeView{ rhsParam.shape }.NumElements();
		if (elementCount == 0 || elementCount > std::numeric_limits<std::uint32_t>::max() ||
		    lhsElementCount == 0 || lhsElementCount > std::numeric_limits<std::uint32_t>::max() ||
		    rhsElementCount == 0 || rhsElementCount > std::numeric_limits<std::uint32_t>::max())
		{
			return std::nullopt;
		}

		return CUDANativeBinaryPlan{
			.op = binary->op,
			.lhsInputIndex = *lhsInputIndex,
			.rhsInputIndex = *rhsInputIndex,
			.elementCount = static_cast<std::uint32_t>(elementCount),
			.lhsElementCount = static_cast<std::uint32_t>(lhsElementCount),
			.rhsElementCount = static_cast<std::uint32_t>(rhsElementCount),
			.requiresBroadcast = !SameShape(lhsParam.shape, output.shape) || !SameShape(rhsParam.shape, output.shape),
			.outputShape = output.shape,
			.lhsShape = lhsParam.shape,
			.rhsShape = rhsParam.shape,
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

	bool IsSupportedCUDANativeLowPrecisionMatMulType(DataType dtype)
	{
		switch (dtype)
		{
		case DataType::Float16:
		case DataType::BFloat16:
		case DataType::Float8E4M3:
		case DataType::Float8E5M2:
		case DataType::Int8:
		case DataType::UInt8:
			return true;
		default:
			return false;
		}
	}

	bool IsSupportedCUDANativeMatMulBiasType(DataType dtype)
	{
		return dtype == DataType::Float32 || dtype == DataType::Float16 || dtype == DataType::BFloat16 ||
		       dtype == DataType::Int8 || dtype == DataType::UInt8;
	}

	void AddCUDANativeMatMulFeatureFlag(CUDANativeInstructionPayload& payload, DataType dtype)
	{
		payload.featureFlags |=
		    dtype == DataType::Float32 ? kCUDANativeFeatureMatMulCUBLASF32 : kCUDANativeFeatureMatMulCUBLASLowPrecision;
	}

	void AddCUDANativeMatMulBiasFeatureFlags(CUDANativeInstructionPayload& payload, DataType dtype, bool relu)
	{
		if (dtype == DataType::Float32)
		{
			payload.featureFlags |= relu ? kCUDANativeFeatureMatMulBiasAddReLUF32
			                             : kCUDANativeFeatureMatMulBiasAddF32;
			return;
		}
		payload.featureFlags |= relu ? kCUDANativeFeatureMatMulBiasAddReLULowPrecision
		                             : kCUDANativeFeatureMatMulBiasAddLowPrecision;
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

	std::optional<CUDANativeMatMulBiasPlan> MakeCUDANativeMatMulBiasPlan(
	    const Subgraph& subgraph, std::uint32_t lhsInputIndex, std::uint32_t rhsInputIndex,
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
		DeviceTraits<CPU>::ConvertTo(cpu, cpuTensor.DType(), cpuTensor.RawData(), cpuTensor.NumElements(),
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
		return CUDANativeReducePlan{ reduce->op, *inputIndex, *inputElementCount, *outputElementCount,
			                         reduce->axis, input.shape, output.shape };
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
		if (output.dtype != castNode->targetType || !SameShape(input.shape, output.shape) ||
		    !CUDANativeSupportsCast(input.dtype, output.dtype))
		{
			return std::nullopt;
		}
		const auto elementCount = ShapeNumElementsU32(output.shape);
		if (!elementCount)
		{
			return std::nullopt;
		}
		return CUDANativeCastPlan{ .inputIndex = *inputIndex,
			                       .elementCount = *elementCount,
			                       .srcType = input.dtype,
			                       .dstType = output.dtype };
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
		return CUDANativeSlicePlan{ *inputIndex, *inputElementCount, *outputElementCount, slice->axis, slice->start,
			                        input.shape, output.shape };
	}

	std::uint64_t CUDANativeBinaryF32FeatureFlag(BinaryOp op)
	{
		switch (op)
		{
		case BinaryOp::Add:
			return kCUDANativeFeatureElementwiseAddF32;
		case BinaryOp::Subtract:
			return kCUDANativeFeatureElementwiseSubtractF32;
		case BinaryOp::Multiply:
			return kCUDANativeFeatureElementwiseMultiplyF32;
		case BinaryOp::Divide:
			return kCUDANativeFeatureElementwiseDivideF32;
		case BinaryOp::Max:
			return kCUDANativeFeatureElementwiseMaxF32;
		case BinaryOp::Min:
			return kCUDANativeFeatureElementwiseMinF32;
		default:
			throw std::runtime_error("Unsupported CUDA native binary op");
		}
	}

	std::uint64_t CUDANativeUnaryF32FeatureFlag(UnaryOp op)
	{
		switch (op)
		{
		case UnaryOp::Negate:
			return kCUDANativeFeatureElementwiseNegateF32;
		case UnaryOp::Abs:
			return kCUDANativeFeatureElementwiseAbsF32;
		case UnaryOp::Sqrt:
			return kCUDANativeFeatureElementwiseSqrtF32;
		case UnaryOp::Exp:
			return kCUDANativeFeatureElementwiseExpF32;
		case UnaryOp::Log:
			return kCUDANativeFeatureElementwiseLogF32;
		case UnaryOp::Sin:
			return kCUDANativeFeatureElementwiseSinF32;
		case UnaryOp::Cos:
			return kCUDANativeFeatureElementwiseCosF32;
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
		payload.featureFlags =
		    kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph | kCUDANativeFeatureMultiKernelLaunch;
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
				if (variableTensor.DType() != output.dtype || !SameShape(variableTensor.Shape().Dims, output.shape))
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
			hasChainSpecificStorage = hasChainSpecificStorage ||
			                          target.kind == CUDANativeArgumentKind::Workspace ||
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
			const auto epilogueName = std::format("{}_{}",
			                                      CUDANativeMatMulBiasEpilogueKernelName(output.dtype, relu),
			                                      fusedLayerCount);
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
			payload.featureFlags |= kCUDANativeFeatureWorkspace;
		}
		if (!payload.constantData.empty())
		{
			payload.featureFlags |= kCUDANativeFeatureConstantTensor;
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
		payload.featureFlags = kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph |
		                       CUDANativeUnaryF32FeatureFlag(plan->op);
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
		payload.featureFlags = kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph |
		                       kCUDANativeFeatureCast;
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
		          .byteSize = TensorByteSize(outputSpecs[0].dtype, outputSpecs[0].shape) },
		        { .kind = CUDANativeArgumentKind::InputTensor,
		          .index = plan->inputIndex,
		          .byteOffset = 0,
		          .byteSize = TensorByteSize(inputSpecs[plan->inputIndex].dtype, inputSpecs[plan->inputIndex].shape) },
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
		const auto plan = MatchCUDANativeBinaryF32(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::PTX;
		payload.featureFlags = kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph |
		                       CUDANativeBinaryF32FeatureFlag(plan->op);
		if (plan->requiresBroadcast)
		{
			payload.featureFlags |= kCUDANativeFeatureElementwiseBroadcastF32;
		}
		payload.target = CUDANativeNVPTXTargetChip();
		std::string ptx;
		if (plan->requiresBroadcast)
		{
			const auto spec = CUDANativeBroadcastBinaryF32CodegenSpec{
			    .op = plan->op,
			    .outputShape = plan->outputShape,
			    .lhsShape = plan->lhsShape,
			    .rhsShape = plan->rhsShape,
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
		const auto lhsByteSize = static_cast<std::uint64_t>(plan->lhsElementCount) * sizeof(float);
		const auto rhsByteSize = static_cast<std::uint64_t>(plan->rhsElementCount) * sizeof(float);
		payload.kernels.push_back({
		    .name = std::string(CUDANativeBinaryF32KernelName(plan->op, plan->requiresBroadcast)),
		    .grid = { .x = gridSize, .y = 1, .z = 1 },
		    .block = { .x = blockSize, .y = 1, .z = 1 },
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
		payload.featureFlags =
		    kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph | kCUDANativeFeatureMatMulCUBLASF32;
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

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeMatMulLowPrecision(const Graph& graph)
	{
		const auto plan = MatchCUDANativeMatMulLowPrecision(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		auto inputSpecs = BuildInputSpecs(graph);
		auto outputSpecs = BuildOutputSpecs(graph);
		const auto dtype = outputSpecs[0].dtype;

		CUDANativeInstructionPayload payload;
		payload.binaryKind = CUDANativeBinaryKind::LibraryCall;
		payload.featureFlags = kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph |
		                       kCUDANativeFeatureMatMulCUBLASLowPrecision;
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

	std::optional<CUDANativeArtifactParts> TryCompileCUDANativeMatMulBias(const Graph& graph)
	{
#ifdef LITENN_ENABLE_CUDA_DRIVER
		const auto plan = MatchCUDANativeMatMulBias(graph);
		if (!plan)
		{
			return std::nullopt;
		}

		const auto epiloguePtx = TryCUDANativeMatMulBiasEpiloguePTXFromMLIRNVPTX(
		    CUDANativeMatMulBiasEpilogueCodegenSpec{
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
		payload.featureFlags =
		    kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph | kCUDANativeFeatureMultiKernelLaunch;
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
		payload.featureFlags = kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph |
		                       kCUDANativeFeatureReduceF32;
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
		payload.featureFlags = kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph |
		                       kCUDANativeFeatureConcatF32 | kCUDANativeFeatureMultiKernelLaunch;
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
		payload.featureFlags = kCUDANativeFeatureStaticShape | kCUDANativeFeatureSingleSubgraph |
		                       kCUDANativeFeatureSliceF32;
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
		CUDANativeConstantBuffer(CUDA& device, std::span<const std::byte> bytes) : device_(&device), byteSize_(bytes.size())
		{
			if (byteSize_ == 0)
			{
				return;
			}
			wordCount_ = (byteSize_ + sizeof(std::uint32_t) - 1) / sizeof(std::uint32_t);
			std::vector<std::uint32_t> padded(wordCount_);
			std::memcpy(padded.data(), bytes.data(), bytes.size());
			data_ = DeviceTraits<CUDA>::Allocate(device, DataType::Int32, wordCount_);
			DeviceTraits<CUDA>::CopyFromCPU(device, DataType::Int32, data_, DataType::Int32, padded.data(),
			                                wordCount_);
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
		for (const auto dtype : { DataType::Float32, DataType::Float16, DataType::BFloat16,
		                         DataType::Float8E4M3, DataType::Float8E5M2, DataType::Int8,
		                         DataType::UInt8 })
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
		return CUDAExecutionOptions{ .stream = options.stream, .synchronize = options.synchronize };
	}

	bool IsCUDAGraphReplayEnabled()
	{
		const char* value = std::getenv("LITENN_CUDA_ENABLE_GRAPH_REPLAY");
		if (value == nullptr)
		{
			return false;
		}
		const std::string_view text = value;
		return text == "1" || text == "true" || text == "TRUE" || text == "on" || text == "ON";
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

	CUDAGraphBindingKey MakeCUDAGraphBindingKey(std::span<const Tensor<CUDA>> inputs,
	                                            std::span<Tensor<CUDA>> outputs)
	{
		CUDAGraphBindingKey key;
		key.pointers.reserve(inputs.size() + outputs.size());
		for (const auto& input : inputs)
		{
			key.pointers.push_back(reinterpret_cast<std::uintptr_t>(input.RawData()));
		}
		for (auto& output : outputs)
		{
			key.pointers.push_back(reinterpret_cast<std::uintptr_t>(output.RawData()));
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
				(void)cudaStreamDestroy(stream_);
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
				(void)cudaGraphExecDestroy(exec_);
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

	void* TensorArgumentPointer(const CUDANativeArgumentSpec& argument, Tensor<CUDA>& tensor,
	                            std::string_view label)
	{
		const auto tensorSize = TensorByteSize(tensor);
		if (argument.byteOffset > tensorSize ||
		    (argument.byteSize != 0 && argument.byteSize > tensorSize - argument.byteOffset))
		{
			throw std::runtime_error(std::format("CUDA native {} argument byte range is out of bounds", label));
		}
		auto* base = static_cast<std::byte*>(tensor.RawData());
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
		auto* base = reinterpret_cast<const std::byte*>(tensor.RawData());
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
	                              const CUDANativeInstructionPayload& payload,
	                              CUDANativeWorkspaceBuffer& workspace, CUDANativeConstantBuffer& constants,
	                              std::span<const Tensor<CUDA>> inputs, std::span<Tensor<CUDA>> outputs,
	                              CompiledModuleCUDARunOptions options)
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

			(void)TensorArgumentPointer(outputArg, output, "output");
			(void)ConstTensorArgumentPointer(lhsArg, lhs, "input");
			(void)ConstTensorArgumentPointer(rhsArg, rhs, "input");
			DeviceTraits<CUDA>::DoBinaryOp(device, BinaryOp::MatMul, output.RawData(), lhs.DType(), lhs.Shape(),
			                                lhs.RawData(), rhs.DType(), rhs.Shape(), rhs.RawData(),
			                                ToCUDAExecutionOptions(options));
			return;
		}

		const auto m = static_cast<std::size_t>(ReadScalarU32(payload, kernel.arguments[3]));
		const auto k = static_cast<std::size_t>(ReadScalarU32(payload, kernel.arguments[4]));
		const auto n = static_cast<std::size_t>(ReadScalarU32(payload, kernel.arguments[5]));
		void* outputPtr = CUDANativeDevicePointer(kernel.arguments[0], inputs, outputs, workspace, constants, "output");
		void* lhsPtr = CUDANativeDevicePointer(kernel.arguments[1], inputs, outputs, workspace, constants, "lhs");
		void* rhsPtr = CUDANativeDevicePointer(kernel.arguments[2], inputs, outputs, workspace, constants, "rhs");
		Tensor<CUDA> outputView(outputPtr, { m, n }, *dtype, device);
		Tensor<CUDA> lhsView(lhsPtr, { m, k }, *dtype, device);
		Tensor<CUDA> rhsView(rhsPtr, { k, n }, *dtype, device);
		DeviceTraits<CUDA>::DoBinaryOp(device, BinaryOp::MatMul, outputView.RawData(), lhsView.DType(),
		                                lhsView.Shape(), lhsView.RawData(), rhsView.DType(), rhsView.Shape(),
		                                rhsView.RawData(),
		                                ToCUDAExecutionOptions(options));
	}

	void RunCUDANativePayload(CUDA& device, const CUDANativeInstructionPayload& payload,
	                          const CUDADriverModule& module, CUDANativeWorkspaceBuffer& workspace,
	                          CUDANativeConstantBuffer& constants,
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
	                                             const CUDADriverModule& module,
	                                             CUDANativeWorkspaceBuffer& workspace,
	                                             CUDANativeConstantBuffer& constants,
	                                             std::span<const Tensor<CUDA>> inputs,
	                                             std::span<Tensor<CUDA>> outputs)
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
			                     });

			CheckCUDARuntime(cudaStreamBeginCapture(captureStream.Get(), cudaStreamCaptureModeThreadLocal),
			                 "cudaStreamBeginCapture");
			capturing = true;

			RunCUDANativePayload(device, payload, module, workspace, constants, inputs, outputs,
			                     CompiledModuleCUDARunOptions{
			                         .stream = captureStream.Get(),
			                         .synchronize = false,
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
				(void)cudaStreamEndCapture(captureStream.Get(), &discardedGraph);
				if (discardedGraph != nullptr)
				{
					(void)cudaGraphDestroy(discardedGraph);
				}
			}
			if (graph != nullptr)
			{
				(void)cudaGraphDestroy(graph);
			}
			throw;
		}
	}

	void RunCUDANativePayloadWithGraphReplay(CUDAGraphReplayCache& cache,
	                                         CUDA& device, const CUDANativeInstructionPayload& payload,
	                                         const CUDADriverModule& module,
	                                         CUDANativeWorkspaceBuffer& workspace,
	                                         CUDANativeConstantBuffer& constants,
	                                         std::span<const Tensor<CUDA>> inputs,
	                                         std::span<Tensor<CUDA>> outputs,
	                                         CompiledModuleCUDARunOptions options)
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

	mlir::OwningOpRef<mlir::ModuleOp> BuildLoweredMLIRModule(const Graph& graph, mlir::MLIRContext& ctx)
	{
		auto module = litenn::translateGraphToMLIR(graph, ctx);
		if (!module)
		{
			throw std::runtime_error("Failed to translate LiteNN graph to MLIR");
		}

		mlir::PassManager pm(&ctx);
		pm.addPass(litenn::createLowerLiteNNPass());
		litenn::addBufferizationPipeline(pm);
		litenn::addLLVMCodegenPipeline(pm);
		if (mlir::failed(pm.run(*module)))
		{
			throw std::runtime_error("LiteNN MLIR lowering/codegen pipeline failed");
		}
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
} // namespace

struct CompiledModule<CPU>::Impl
{
	std::vector<std::byte> rodata;
	std::vector<std::byte> instructions;
	std::vector<CompiledTensorSpec> inputSpecs;
	std::vector<CompiledTensorSpec> outputSpecs;
	CompiledModuleBackend backend{ CompiledModuleBackend::CPUNative };
	std::unique_ptr<llvm::LLVMContext> jitContext;
	std::unique_ptr<llvm::ExecutionEngine> jit;
	EntryFn entry{};
};

CompiledModule<CPU>::CompiledModule() = default;

CompiledModuleArtifact::CompiledModuleArtifact(std::vector<std::byte> rodata,
	                                           std::vector<std::byte> instructions,
	                                           std::vector<CompiledTensorSpec> inputSpecs,
	                                           std::vector<CompiledTensorSpec> outputSpecs,
	                                           CompiledModuleBackend backend)
	: rodata_(std::move(rodata)),
	  instructions_(std::move(instructions)),
	  inputSpecs_(std::move(inputSpecs)),
	  outputSpecs_(std::move(outputSpecs)),
	  backend_(backend)
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

CompiledModule<CPU> CompiledModuleArtifact::Load() const
{
	return CompiledModule<CPU>::Load(Image());
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

std::optional<std::size_t> CompiledModuleArtifact::FindInput(std::string_view name) const
{
	return FindSpecIndex(inputSpecs_, name);
}

std::optional<std::size_t> CompiledModuleArtifact::FindOutput(std::string_view name) const
{
	return FindSpecIndex(outputSpecs_, name);
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

std::vector<Tensor<CPU>> CompiledModule<CPU>::Run(std::span<const Tensor<CPU>> inputs) const
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

	llvm::SmallVector<void*, 8> inputPtrs;
	inputPtrs.reserve(inputs.size());
	for (std::size_t i = 0; i < inputs.size(); ++i)
	{
		ValidateTensorAgainstSpec(inputs[i], impl_->inputSpecs[i], i);
		inputPtrs.push_back(const_cast<void*>(inputs[i].RawData()));
	}

	std::vector<Tensor<CPU>> outputs;
	outputs.reserve(impl_->outputSpecs.size());
	llvm::SmallVector<void*, 8> outputPtrs;
	outputPtrs.reserve(impl_->outputSpecs.size());
	for (const auto& spec : impl_->outputSpecs)
	{
		outputs.emplace_back(Uninitialized, ShapeView{ spec.shape }, spec.dtype, CPU{});
		outputPtrs.push_back(outputs.back().RawData());
	}

	impl_->entry(inputPtrs.data(), outputPtrs.data());
	return outputs;
}

void CompiledModule<CPU>::RunInto(std::span<const Tensor<CPU>> inputs, std::span<Tensor<CPU>> outputs) const
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
		ValidateTensorAgainstSpec(inputs[i], impl_->inputSpecs[i], i);
		inputPtrs.push_back(const_cast<void*>(inputs[i].RawData()));
	}

	llvm::SmallVector<void*, 8> outputPtrs;
	outputPtrs.reserve(outputs.size());
	for (std::size_t i = 0; i < outputs.size(); ++i)
	{
		ValidateOutputTensorAgainstSpec(outputs[i], impl_->outputSpecs[i], i);
		outputPtrs.push_back(outputs[i].RawData());
	}

	impl_->entry(inputPtrs.data(), outputPtrs.data());
}

void CompiledModule<CPU>::RunManyInto(std::span<const CompiledModuleInvocation> invocations,
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
			RunInto(invocation.inputs, invocation.outputs);
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
				RunInto(invocation.inputs, invocation.outputs);
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

std::vector<Tensor<CUDA>> CompiledModule<CUDA>::Run(std::span<const Tensor<CUDA>> inputs) const
{
	return Run(inputs, CompiledModuleCUDARunOptions{});
}

std::vector<Tensor<CUDA>> CompiledModule<CUDA>::Run(std::span<const Tensor<CUDA>> inputs,
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
		outputs.emplace_back(Uninitialized, ShapeView{ spec.shape }, spec.dtype, impl_->device);
	}
	RunInto(inputs, outputs, options);
	return outputs;
}

void CompiledModule<CUDA>::RunInto(std::span<const Tensor<CUDA>> inputs, std::span<Tensor<CUDA>> outputs) const
{
	RunInto(inputs, outputs, CompiledModuleCUDARunOptions{});
}

void CompiledModule<CUDA>::RunInto(std::span<const Tensor<CUDA>> inputs, std::span<Tensor<CUDA>> outputs,
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
		if (IsCUDAGraphReplayEnabled() && options.synchronize && options.stream == nullptr)
		{
			std::scoped_lock lock(impl_->cudaWorkspaceMutex, impl_->cudaGraphReplayMutex);
			RunCUDANativePayloadWithGraphReplay(impl_->cudaGraphReplayCache, impl_->device, impl_->cudaPayload,
			                                    impl_->cudaModule, *impl_->cudaWorkspace, *impl_->cudaConstants,
			                                    inputs, outputs, options);
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
		DeviceTraits<CUDA>::CopyToCPU(inputDevice, inputs[i].DType(), inputs[i].RawData(),
		                                inputs[i].NumElements(), cpuInput.DType(), cpuInput.RawData(),
		                                CUDAExecutionOptions{ .stream = options.stream, .synchronize = true });
		cpuInputs.push_back(std::move(cpuInput));
	}

	std::vector<Tensor<CPU>> cpuOutputs;
	cpuOutputs.reserve(impl_->outputSpecs.size());
	for (const auto& spec : impl_->outputSpecs)
	{
		cpuOutputs.emplace_back(Uninitialized, ShapeView{ spec.shape }, spec.dtype, CPU{});
	}
	impl_->cpuModule.RunInto(cpuInputs, cpuOutputs);

	for (std::size_t i = 0; i < outputs.size(); ++i)
	{
		DeviceTraits<CUDA>::CopyFromCPU(outputs[i].CurDevice(), outputs[i].DType(), outputs[i].RawData(),
		                                cpuOutputs[i].DType(), cpuOutputs[i].RawData(), cpuOutputs[i].NumElements(),
		                                CUDAExecutionOptions{ .stream = options.stream, .synchronize = true });
	}
}

void CompiledModule<CUDA>::RunManyInto(std::span<const CompiledModuleCUDAInvocation> invocations,
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
			RunInto(invocation.inputs, invocation.outputs, invocation.options);
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
				RunInto(invocation.inputs, invocation.outputs, invocation.options);
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

CompiledModuleArtifact Compiler<CPU>::CompileArtifact(const Graph& graph)
{
	if (auto parallelParts = TryCompileCPUParallelLinearChainF32(graph))
	{
		return CompiledModuleArtifact(std::move(parallelParts->rodata), std::move(parallelParts->instructions),
		                              std::move(parallelParts->inputSpecs), std::move(parallelParts->outputSpecs),
		                              CompiledModuleBackend::CPUNative);
	}

	mlir::MLIRContext ctx;
	SetupCompilerMLIRContext(ctx);
	auto mlirModule = BuildLoweredMLIRModule(graph, ctx);

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
	OptimizeLLVMModule(*llvmModule, *config.targetMachine);

	auto rodata = SerializeRodata(inputSpecs, outputSpecs, config.triple, CompiledModuleBackend::CPUNative);
	auto instructions = EmitObjectFile(*llvmModule);
	return CompiledModuleArtifact(std::move(rodata), std::move(instructions), inputSpecs, outputSpecs,
	                              CompiledModuleBackend::CPUNative);
}

CompiledModule<CPU> Compiler<CPU>::Compile(const Graph& graph)
{
	return CompileArtifact(graph).Load();
}

#ifdef LITENN_ENABLE_CUDA
CompiledModuleArtifact Compiler<CUDA>::CompileArtifact(const Graph& graph)
{
	Validation::ValidateGraph(graph);
	if (!IsCUDANativeAOTDisabled())
	{
		if (auto nativeParts = TryCompileCUDANativeCast(graph))
		{
			return CompiledModuleArtifact(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
			                              std::move(nativeParts->inputSpecs), std::move(nativeParts->outputSpecs),
			                              CompiledModuleBackend::CUDANative);
		}
		if (auto nativeParts = TryCompileCUDANativeUnaryF32(graph))
		{
			return CompiledModuleArtifact(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
			                              std::move(nativeParts->inputSpecs), std::move(nativeParts->outputSpecs),
			                              CompiledModuleBackend::CUDANative);
		}
		if (auto nativeParts = TryCompileCUDANativeLinearChain(graph))
		{
			return CompiledModuleArtifact(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
			                              std::move(nativeParts->inputSpecs), std::move(nativeParts->outputSpecs),
			                              CompiledModuleBackend::CUDANative);
		}
		if (auto nativeParts = TryCompileCUDANativeMatMulBias(graph))
		{
			return CompiledModuleArtifact(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
			                              std::move(nativeParts->inputSpecs), std::move(nativeParts->outputSpecs),
			                              CompiledModuleBackend::CUDANative);
		}
		if (auto nativeParts = TryCompileCUDANativeMatMulF32(graph))
		{
			return CompiledModuleArtifact(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
			                              std::move(nativeParts->inputSpecs), std::move(nativeParts->outputSpecs),
			                              CompiledModuleBackend::CUDANative);
		}
		if (auto nativeParts = TryCompileCUDANativeMatMulLowPrecision(graph))
		{
			return CompiledModuleArtifact(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
			                              std::move(nativeParts->inputSpecs), std::move(nativeParts->outputSpecs),
			                              CompiledModuleBackend::CUDANative);
		}
		if (auto nativeParts = TryCompileCUDANativeBinaryF32(graph))
		{
			return CompiledModuleArtifact(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
			                              std::move(nativeParts->inputSpecs), std::move(nativeParts->outputSpecs),
			                              CompiledModuleBackend::CUDANative);
		}
		if (auto nativeParts = TryCompileCUDANativeReduceF32(graph))
		{
			return CompiledModuleArtifact(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
			                              std::move(nativeParts->inputSpecs), std::move(nativeParts->outputSpecs),
			                              CompiledModuleBackend::CUDANative);
		}
		if (auto nativeParts = TryCompileCUDANativeConcatF32(graph))
		{
			return CompiledModuleArtifact(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
			                              std::move(nativeParts->inputSpecs), std::move(nativeParts->outputSpecs),
			                              CompiledModuleBackend::CUDANative);
		}
		if (auto nativeParts = TryCompileCUDANativeSliceF32(graph))
		{
			return CompiledModuleArtifact(std::move(nativeParts->rodata), std::move(nativeParts->instructions),
			                              std::move(nativeParts->inputSpecs), std::move(nativeParts->outputSpecs),
			                              CompiledModuleBackend::CUDANative);
		}
	}
	return Compiler<CPU>::CompileArtifact(graph);
}

CompiledModule<CUDA> Compiler<CUDA>::Compile(const Graph& graph, CUDA device)
{
	return CompileArtifact(graph).Load(std::move(device));
}
#endif
