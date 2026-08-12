#include "CompiledModule.h"

#include "CUDANativeCodegen.h"
#include "CUDANativePayload.h"
#include "Dialect/LiteNNDialect.h"
#include "Dialect/LiteNNOps.h"
#include "Pass/BufferizationPipeline.h"
#include "Pass/LLVMCodegenPipeline.h"
#include "Pass/LowerLiteNNPass.h"
#include "Runtime/CPUGGMLV4Microkernels.h"
#include "Translation/GraphToMLIR.h"
#ifdef LITENN_ENABLE_VULKAN
#include "VulkanNativeCodegen.h"
#include "VulkanNativePayload.h"
#endif

#include <LiteNN/Misc.h>
#include <LiteNN/OpSchema.h>
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
#include "mlir/Pass/Pass.h"
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

#if (defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)) &&                               \
    (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#define LITENN_HAS_X86_AVX2_TARGET 1
#define LITENN_TARGET_AVX2 __attribute__((target("avx2")))
#define LITENN_TARGET_AVX2_F16C __attribute__((target("avx2,f16c")))
#else
#define LITENN_HAS_X86_AVX2_TARGET 0
#define LITENN_TARGET_AVX2
#define LITENN_TARGET_AVX2_F16C
#endif

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <semaphore>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

using namespace LiteNN;

namespace LiteNN
{
	struct CompiledModuleCPUHelperProfiler::Impl
	{
		struct NodeFrame
		{
			std::uint64_t subgraphId{};
			std::uint64_t nodeId{};
			std::uint32_t schemaId{};
			std::chrono::steady_clock::time_point start;
			double helperMillisecondsAtStart{};
			double childInclusiveMilliseconds{};
			double childHelperMilliseconds{};
			double childInstrumentationMilliseconds{};
		};

		CompiledModuleCPUHelperProfiler::Impl* previous{};
		std::unordered_map<std::string, CompiledModuleCPUHelperProfileEvent> events;
		std::unordered_map<std::string, CompiledModuleCPUNodeProfileEvent> nodeEvents;
		std::vector<CompiledModuleCPUParallelProfileEvent> parallelEvents;
		std::vector<NodeFrame> nodeStack;
		double helperMilliseconds{};
		double nodeInstrumentationMilliseconds{};
	};

	struct CompiledModuleCPUHelperProfilerAccess
	{
		static thread_local CompiledModuleCPUHelperProfiler::Impl* current;

		static bool Enabled()
		{
			return current != nullptr;
		}

		static void Record(std::string_view helper, std::string_view detail, double milliseconds)
		{
			if (current == nullptr)
			{
				return;
			}
			std::string key(helper);
			key.push_back('\n');
			key.append(detail);
			auto& event = current->events[key];
			event.helper = std::string(helper);
			event.detail = std::string(detail);
			++event.calls;
			event.totalMilliseconds += milliseconds;
			current->helperMilliseconds += milliseconds;
		}

		static void BeginNode(std::uint64_t subgraphId, std::uint64_t nodeId, std::uint32_t schemaId)
		{
			if (current == nullptr)
			{
				return;
			}
			const auto callbackStart = std::chrono::steady_clock::now();
			const auto parentIndex = current->nodeStack.size();
			current->nodeStack.push_back({
			    .subgraphId = subgraphId,
			    .nodeId = nodeId,
			    .schemaId = schemaId,
			    .helperMillisecondsAtStart = current->helperMilliseconds,
			});
			const auto callbackEnd = std::chrono::steady_clock::now();
			current->nodeStack.back().start = callbackEnd;
			current->nodeInstrumentationMilliseconds +=
			    std::chrono::duration<double, std::milli>(callbackEnd - callbackStart).count();
			if (parentIndex > 0)
			{
				current->nodeStack[parentIndex - 1].childInstrumentationMilliseconds +=
				    std::chrono::duration<double, std::milli>(callbackEnd - callbackStart).count();
			}
		}

		static void RecordParallel(CompiledModuleCPUParallelProfileEvent event)
		{
			if (current != nullptr)
			{
				current->parallelEvents.push_back(std::move(event));
			}
		}

		static void EndNode(std::uint64_t subgraphId, std::uint64_t nodeId, std::uint32_t schemaId)
		{
			if (current == nullptr || current->nodeStack.empty())
			{
				return;
			}

			const auto callbackStart = std::chrono::steady_clock::now();
			auto frame = current->nodeStack.back();
			current->nodeStack.pop_back();
			if (frame.subgraphId != subgraphId || frame.nodeId != nodeId || frame.schemaId != schemaId)
			{
				current->nodeStack.clear();
				current->nodeInstrumentationMilliseconds +=
				    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - callbackStart).count();
				return;
			}

			const auto inclusive = std::chrono::duration<double, std::milli>(callbackStart - frame.start).count();
			const auto totalHelper = std::max(0.0, current->helperMilliseconds - frame.helperMillisecondsAtStart);
			const auto directHelper = std::max(0.0, totalHelper - frame.childHelperMilliseconds);
			const auto self = std::max(0.0, inclusive - frame.childInclusiveMilliseconds -
			                                    frame.childInstrumentationMilliseconds - directHelper);
			const auto key = std::format("{}:{}:{}", subgraphId, nodeId, schemaId);
			auto& event = current->nodeEvents[key];
			event.subgraphId = subgraphId;
			event.nodeId = nodeId;
			event.schemaId = schemaId;
			++event.calls;
			event.inclusiveMilliseconds += inclusive;
			event.selfMilliseconds += self;
			event.helperMilliseconds += directHelper;

			const auto callbackEnd = std::chrono::steady_clock::now();
			current->nodeInstrumentationMilliseconds +=
			    std::chrono::duration<double, std::milli>(callbackEnd - callbackStart).count();
			if (!current->nodeStack.empty())
			{
				auto& parent = current->nodeStack.back();
				parent.childInclusiveMilliseconds += inclusive;
				parent.childHelperMilliseconds += totalHelper;
				parent.childInstrumentationMilliseconds +=
				    std::chrono::duration<double, std::milli>(callbackEnd - callbackStart).count();
			}
		}
	};

	thread_local CompiledModuleCPUHelperProfiler::Impl* CompiledModuleCPUHelperProfilerAccess::current = nullptr;
} // namespace LiteNN

namespace
{
	using LiteNN::Detail::AccumulateGGMLQ4KFieldInterleavedV4BlockQ8Kx16AVX2;
	using LiteNN::Detail::AccumulateGGMLQ4KFieldInterleavedV4BlockQ8Kx8AVX2;
	using LiteNN::Detail::AccumulateGGMLQ6KFieldInterleavedV4BlockQ8Kx16AVX2;
	using LiteNN::Detail::AccumulateGGMLQ6KFieldInterleavedV4BlockQ8Kx16AVX512;
	using LiteNN::Detail::AccumulateGGMLQ6KFieldInterleavedV4BlockQ8Kx8AVX2;
	using LiteNN::Detail::CPUHasGGMLV4AVX512F16C;
	using LiteNN::Detail::GGMLQ4KFieldInterleaved8Block;
	using LiteNN::Detail::GGMLQ6KFieldInterleaved8Block;
	using LiteNN::Detail::GGMLQ8KActivationBlock;

	extern "C" void litenn_cpu_profile_node_begin(std::uint64_t subgraphId, std::uint64_t nodeId,
	                                              std::uint64_t schemaId)
	{
		CompiledModuleCPUHelperProfilerAccess::BeginNode(subgraphId, nodeId, static_cast<std::uint32_t>(schemaId));
	}

	extern "C" void litenn_cpu_profile_node_end(std::uint64_t subgraphId, std::uint64_t nodeId, std::uint64_t schemaId)
	{
		CompiledModuleCPUHelperProfilerAccess::EndNode(subgraphId, nodeId, static_cast<std::uint32_t>(schemaId));
	}

#if defined(__GNUC__) || defined(__clang__)
#define LITENN_RESTRICT __restrict__
#define LITENN_GCC_IVDEP _Pragma("GCC ivdep")
#else
#define LITENN_RESTRICT
#define LITENN_GCC_IVDEP
#endif

	class CPUAOTHelperProfileTimer
	{
	public:
		explicit CPUAOTHelperProfileTimer(std::string_view helper)
		    : helper_(helper), enabled_(CompiledModuleCPUHelperProfilerAccess::Enabled())
		{
			if (enabled_)
			{
				start_ = std::chrono::steady_clock::now();
			}
		}

		CPUAOTHelperProfileTimer(std::string_view helper, std::string detail)
		    : helper_(helper), detail_(std::move(detail)), enabled_(CompiledModuleCPUHelperProfilerAccess::Enabled())
		{
			if (enabled_)
			{
				start_ = std::chrono::steady_clock::now();
			}
		}

		~CPUAOTHelperProfileTimer()
		{
			if (!enabled_)
			{
				return;
			}
			const auto elapsed = std::chrono::steady_clock::now() - start_;
			CompiledModuleCPUHelperProfilerAccess::Record(helper_, detail_,
			                                              std::chrono::duration<double, std::milli>(elapsed).count());
		}

	private:
		std::string_view helper_;
		std::string detail_;
		bool enabled_{};
		std::chrono::steady_clock::time_point start_;
	};

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

	struct MLIRModuleStats
	{
		std::size_t operationCount{};
		std::size_t functionCount{};
		std::size_t blockCount{};
	};

	MLIRModuleStats CollectMLIRModuleStats(mlir::ModuleOp module)
	{
		MLIRModuleStats stats;
		module.walk([&](mlir::Operation* op) {
			++stats.operationCount;
			stats.blockCount += op->getRegions().empty()
			                        ? 0
			                        : std::accumulate(op->getRegions().begin(), op->getRegions().end(), std::size_t{},
			                                          [](std::size_t total, mlir::Region& region) {
				                                          return total + std::distance(region.begin(), region.end());
			                                          });
			if (llvm::isa<mlir::func::FuncOp>(op))
			{
				++stats.functionCount;
			}
		});
		return stats;
	}

	struct LLVMModuleStats
	{
		std::size_t functionCount{};
		std::size_t declarationCount{};
		std::size_t basicBlockCount{};
		std::size_t instructionCount{};
		std::size_t globalVariableCount{};
		std::size_t aliasCount{};
		std::vector<std::pair<std::string, std::size_t>> largestFunctions;
	};

	LLVMModuleStats CollectLLVMModuleStats(const llvm::Module& module)
	{
		LLVMModuleStats stats;
		for (const auto& function : module)
		{
			if (function.isDeclaration())
			{
				++stats.declarationCount;
				continue;
			}
			++stats.functionCount;
			stats.basicBlockCount += function.size();
			std::size_t functionInstructionCount = 0;
			for (const auto& block : function)
			{
				stats.instructionCount += block.size();
				functionInstructionCount += block.size();
			}
			stats.largestFunctions.emplace_back(function.getName().str(), functionInstructionCount);
		}
		stats.globalVariableCount = std::distance(module.global_begin(), module.global_end());
		stats.aliasCount = std::distance(module.alias_begin(), module.alias_end());
		std::ranges::sort(stats.largestFunctions,
		                  [](const auto& lhs, const auto& rhs) { return lhs.second > rhs.second; });
		if (stats.largestFunctions.size() > 5)
		{
			stats.largestFunctions.resize(5);
		}
		return stats;
	}

	void LogMLIRModuleStats(const CompilerOptions& options, std::string_view label, mlir::ModuleOp module)
	{
		if (!options.enableCompileDiagnostics)
		{
			return;
		}
		const auto stats = CollectMLIRModuleStats(module);
		LogCompileDiagnostic(options, std::format("{} stats: ops={} funcs={} blocks={}", label, stats.operationCount,
		                                          stats.functionCount, stats.blockCount));
	}

	void LogLLVMModuleStats(const CompilerOptions& options, std::string_view label, const llvm::Module& module)
	{
		if (!options.enableCompileDiagnostics)
		{
			return;
		}
		const auto stats = CollectLLVMModuleStats(module);
		LogCompileDiagnostic(options,
		                     std::format("{} stats: funcs={} decls={} blocks={} insts={} globals={} aliases={}", label,
		                                 stats.functionCount, stats.declarationCount, stats.basicBlockCount,
		                                 stats.instructionCount, stats.globalVariableCount, stats.aliasCount));
		for (std::size_t i = 0; i < stats.largestFunctions.size(); ++i)
		{
			const auto& [name, instructions] = stats.largestFunctions[i];
			LogCompileDiagnostic(options,
			                     std::format("{} top_function[{}]: name={} insts={}", label, i, name, instructions));
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

	void LiteNNCPUThreadRelax()
	{
#if LITENN_HAS_X86_AVX2_TARGET
		_mm_pause();
#elif defined(__aarch64__)
		__asm__ volatile("yield" ::: "memory");
#else
		std::this_thread::yield();
#endif
	}

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

	struct LiteNNCPUThreadAffinityTarget
	{
#ifdef _WIN32
		WORD group{};
		KAFFINITY mask{};
#elif defined(__linux__)
		std::size_t processor{};
#endif
	};

	struct LiteNNCPUAffinityTopology
	{
		std::vector<LiteNNCPUThreadAffinityTarget> compact;
		std::vector<LiteNNCPUThreadAffinityTarget> spread;
	};

	const LiteNNCPUAffinityTopology& GetLiteNNCPUAffinityTopology()
	{
		static const auto topology = [] {
			std::vector<std::vector<LiteNNCPUThreadAffinityTarget>> cores;
#ifdef _WIN32
			DWORD bytes = 0;
			if (!GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &bytes) &&
			    GetLastError() == ERROR_INSUFFICIENT_BUFFER && bytes != 0)
			{
				std::vector<std::byte> storage(bytes);
				if (GetLogicalProcessorInformationEx(
				        RelationProcessorCore,
				        reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(storage.data()), &bytes))
				{
					std::size_t offset = 0;
					while (offset + sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX) <= bytes)
					{
						const auto* info =
						    reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(storage.data() + offset);
						if (info->Size == 0 || offset + info->Size > bytes)
						{
							break;
						}
						std::vector<LiteNNCPUThreadAffinityTarget> siblings;
						for (WORD groupIndex = 0; groupIndex < info->Processor.GroupCount; ++groupIndex)
						{
							const auto& groupMask = info->Processor.GroupMask[groupIndex];
							for (std::size_t bit = 0; bit < sizeof(KAFFINITY) * 8; ++bit)
							{
								const auto mask = static_cast<KAFFINITY>(1) << bit;
								if ((groupMask.Mask & mask) != 0)
								{
									siblings.push_back({ .group = groupMask.Group, .mask = mask });
								}
							}
						}
						if (!siblings.empty())
						{
							cores.push_back(std::move(siblings));
						}
						offset += info->Size;
					}
				}
			}
#elif defined(__linux__)
			cpu_set_t available;
			CPU_ZERO(&available);
			if (pthread_getaffinity_np(pthread_self(), sizeof(available), &available) == 0)
			{
				struct LinuxCore
				{
					int package{};
					int core{};
					std::vector<LiteNNCPUThreadAffinityTarget> siblings;
				};
				std::vector<LinuxCore> linuxCores;
				auto readTopologyValue = [](std::size_t processor, std::string_view name) -> std::optional<int> {
					std::ifstream input(std::format("/sys/devices/system/cpu/cpu{}/topology/{}", processor, name));
					int value = 0;
					return input >> value ? std::optional<int>{ value } : std::nullopt;
				};
				for (std::size_t processor = 0; processor < CPU_SETSIZE; ++processor)
				{
					if (!CPU_ISSET(processor, &available))
					{
						continue;
					}
					const auto package = readTopologyValue(processor, "physical_package_id");
					const auto core = readTopologyValue(processor, "core_id");
					if (!package || !core)
					{
						continue;
					}
					auto found = std::ranges::find_if(linuxCores, [&](const LinuxCore& candidate) {
						return candidate.package == *package && candidate.core == *core;
					});
					if (found == linuxCores.end())
					{
						linuxCores.push_back({ .package = *package, .core = *core });
						found = std::prev(linuxCores.end());
					}
					found->siblings.push_back({ .processor = processor });
				}
				for (auto& core : linuxCores)
				{
					cores.push_back(std::move(core.siblings));
				}
			}
#endif

			LiteNNCPUAffinityTopology result;
			for (std::size_t siblingIndex = 0;; ++siblingIndex)
			{
				const auto previousSize = result.compact.size();
				for (const auto& core : cores)
				{
					if (siblingIndex < core.size())
					{
						result.compact.push_back(core[siblingIndex]);
					}
				}
				if (result.compact.size() == previousSize)
				{
					break;
				}
			}
			if (!cores.empty())
			{
				// Firmware topology enumeration commonly keeps cache domains contiguous. Interleave its lower and upper
				// halves without changing Compact semantics; exact LLC/NUMA placement remains a separate policy.
				const auto secondHalf = (cores.size() + 1) / 2;
				for (std::size_t siblingIndex = 0;; ++siblingIndex)
				{
					const auto previousSize = result.spread.size();
					for (std::size_t slot = 0; slot < cores.size(); ++slot)
					{
						const auto coreIndex = slot % 2 == 0 ? slot / 2 : secondHalf + slot / 2;
						if (coreIndex < cores.size() && siblingIndex < cores[coreIndex].size())
						{
							result.spread.push_back(cores[coreIndex][siblingIndex]);
						}
					}
					if (result.spread.size() == previousSize)
					{
						break;
					}
				}
			}
			if (result.compact.empty())
			{
#ifdef _WIN32
				const auto groupCount = GetActiveProcessorGroupCount();
				for (WORD group = 0; group < groupCount; ++group)
				{
					const auto processorCount = GetActiveProcessorCount(group);
					for (DWORD processor = 0; processor < processorCount && processor < sizeof(KAFFINITY) * 8;
					     ++processor)
					{
						result.compact.push_back(
						    { .group = group,
						      .mask = static_cast<KAFFINITY>(1) << static_cast<std::size_t>(processor) });
					}
				}
#elif defined(__linux__)
				cpu_set_t available;
				CPU_ZERO(&available);
				if (pthread_getaffinity_np(pthread_self(), sizeof(available), &available) == 0)
				{
					for (std::size_t processor = 0; processor < CPU_SETSIZE; ++processor)
					{
						if (CPU_ISSET(processor, &available))
						{
							result.compact.push_back({ .processor = processor });
						}
					}
				}
#endif
			}
			if (result.spread.empty())
			{
				result.spread = result.compact;
			}
			return result;
		}();
		return topology;
	}

	constexpr std::uint64_t kCPUAOTSchedulingFieldMask = 0xff;
	constexpr std::uint64_t kCPUAOTWorkerWaitPolicyShift = 8;

	std::uint64_t EncodeCPUAOTSchedulingPolicy(CPUAOTAffinityPolicy affinityPolicy, CPUAOTWorkerWaitPolicy waitPolicy)
	{
		return static_cast<std::uint64_t>(affinityPolicy) |
		       (static_cast<std::uint64_t>(waitPolicy) << kCPUAOTWorkerWaitPolicyShift);
	}

	CPUAOTAffinityPolicy ResolveCPUAOTAffinityPolicy(std::uint64_t value)
	{
		switch (static_cast<CPUAOTAffinityPolicy>(value & kCPUAOTSchedulingFieldMask))
		{
		case CPUAOTAffinityPolicy::Compact:
			return CPUAOTAffinityPolicy::Compact;
		case CPUAOTAffinityPolicy::Spread:
			return CPUAOTAffinityPolicy::Spread;
		default:
			return CPUAOTAffinityPolicy::None;
		}
	}

	CPUAOTWorkerWaitPolicy ResolveCPUAOTWorkerWaitPolicy(std::uint64_t value)
	{
		switch (
		    static_cast<CPUAOTWorkerWaitPolicy>((value >> kCPUAOTWorkerWaitPolicyShift) & kCPUAOTSchedulingFieldMask))
		{
		case CPUAOTWorkerWaitPolicy::LowPower:
			return CPUAOTWorkerWaitPolicy::LowPower;
		case CPUAOTWorkerWaitPolicy::Latency:
			return CPUAOTWorkerWaitPolicy::Latency;
		default:
			return CPUAOTWorkerWaitPolicy::Adaptive;
		}
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
			if (policy == CPUAOTAffinityPolicy::None)
			{
				return;
			}
			const auto& topology = GetLiteNNCPUAffinityTopology();
			const auto& targets = policy == CPUAOTAffinityPolicy::Spread ? topology.spread : topology.compact;
			if (workerSlot >= targets.size())
			{
				return;
			}
#ifdef _WIN32
			const GROUP_AFFINITY target{
				.Mask = targets[workerSlot].mask,
				.Group = targets[workerSlot].group,
			};
			if (SetThreadGroupAffinity(GetCurrentThread(), &target, &previousGroupAffinity_))
			{
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
			CPU_SET(targets[workerSlot].processor, &targetSet);
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
			SetThreadGroupAffinity(GetCurrentThread(), &previousGroupAffinity_, nullptr);
#elif defined(__linux__)
			pthread_setaffinity_np(pthread_self(), sizeof(previousSet_), &previousSet_);
#endif
			activePolicy_ = CPUAOTAffinityPolicy::None;
		}

		CPUAOTAffinityPolicy activePolicy_{ CPUAOTAffinityPolicy::None };
#ifdef _WIN32
		GROUP_AFFINITY previousGroupAffinity_{};
#elif defined(__linux__)
		cpu_set_t previousSet_{};
#endif
	};

	class LiteNNCPUThreadPool
	{
	public:
		struct ParticipantProfile
		{
			std::uint64_t taskClaims{};
			std::uint64_t workUnits{};
			double usefulMilliseconds{};
		};

		struct ParallelForProfile
		{
			std::vector<ParticipantProfile> participants;
			std::uint64_t signaledWorkerCount{};
			double lockWaitMilliseconds{};
			double dispatchMilliseconds{};
			double wallMilliseconds{};
			double barrierWaitMilliseconds{};
		};

		explicit LiteNNCPUThreadPool(std::size_t threadCount)
		{
			const auto workerCount = threadCount > 1 ? threadCount - 1 : 0;
			workers_.reserve(workerCount);
			for (std::size_t i = 0; i < workerCount; ++i)
			{
				auto worker = std::make_unique<Worker>();
				auto* workerPtr = worker.get();
				worker->thread = std::thread([this, workerPtr, i] { WorkerLoop(*workerPtr, i); });
				workers_.push_back(std::move(worker));
			}
		}

		~LiteNNCPUThreadPool()
		{
			stopping_.store(true, std::memory_order_release);
			for (auto& worker : workers_)
			{
				worker->generation.fetch_add(1, std::memory_order_release);
				worker->start.release();
			}
			for (auto& worker : workers_)
			{
				if (worker->thread.joinable())
				{
					worker->thread.join();
				}
			}
		}

		LiteNNCPUThreadPool(const LiteNNCPUThreadPool&) = delete;
		LiteNNCPUThreadPool& operator=(const LiteNNCPUThreadPool&) = delete;

		void ParallelFor(std::uint64_t begin, std::uint64_t end, std::uint64_t grain, LiteNNCPUParallelForBody body,
		                 void* userData, std::size_t requestedThreads, CPUAOTAffinityPolicy affinityPolicy,
		                 CPUAOTWorkerWaitPolicy waitPolicy, ParallelForProfile* profile = nullptr)
		{
			if (begin >= end)
			{
				return;
			}
			grain = std::max<std::uint64_t>(1, grain);
			const auto taskCount = (end - begin + grain - 1) / grain;
			const auto participantCount = std::min<std::uint64_t>(
			    std::max<std::uint64_t>(1, static_cast<std::uint64_t>(requestedThreads)), taskCount);
			static thread_local LiteNNCPUThreadAffinityState callerAffinity;
			callerAffinity.Apply(affinityPolicy, 0);
			if (participantCount <= 1 || workers_.empty())
			{
				const auto start = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
				body(begin, end, userData);
				if (profile)
				{
					const auto elapsed =
					    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
					profile->participants = { ParticipantProfile{
						.taskClaims = 1, .workUnits = end - begin, .usefulMilliseconds = elapsed } };
					profile->wallMilliseconds = elapsed;
				}
				return;
			}

			const auto lockStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
			std::unique_lock runLock(runMutex_);
			const auto desiredWorkers =
			    std::min<std::size_t>(static_cast<std::size_t>(participantCount - 1), workers_.size());
			if (profile)
			{
				profile->lockWaitMilliseconds =
				    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - lockStart).count();
				profile->participants.assign(desiredWorkers + 1, {});
			}
			begin_ = begin;
			end_ = end;
			grain_ = grain;
			body_ = body;
			userData_ = userData;
			affinityPolicy_ = affinityPolicy;
			waitPolicy_.store(waitPolicy, std::memory_order_release);
			next_.store(begin, std::memory_order_relaxed);
			workersDone_.store(0, std::memory_order_relaxed);
			desiredWorkers_ = desiredWorkers;
			profile_ = profile;
			const auto parallelStart =
			    profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
			for (std::size_t i = 0; i < desiredWorkers; ++i)
			{
				workers_[i]->generation.fetch_add(1, std::memory_order_release);
				workers_[i]->start.release();
				if (profile)
				{
					++profile->signaledWorkerCount;
				}
			}
			if (profile)
			{
				profile->dispatchMilliseconds =
				    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - parallelStart).count();
				RunTasksProfiled(0);
			}
			else
			{
				RunTasks();
			}

			const auto barrierStart =
			    profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
			while (workersDone_.load(std::memory_order_acquire) != desiredWorkers)
			{
				LiteNNCPUThreadRelax();
			}
			if (profile)
			{
				const auto endTime = std::chrono::steady_clock::now();
				profile->barrierWaitMilliseconds =
				    std::chrono::duration<double, std::milli>(endTime - barrierStart).count();
				profile->wallMilliseconds = std::chrono::duration<double, std::milli>(endTime - parallelStart).count();
			}
			profile_ = nullptr;
		}

	private:
		struct Worker
		{
			std::binary_semaphore start{ 0 };
			std::atomic<std::uint64_t> generation{};
			std::thread thread;
		};

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

		void RunTasksProfiled(std::size_t participantIndex)
		{
			auto& participant = profile_->participants[participantIndex];
			while (true)
			{
				const auto taskBegin = next_.fetch_add(grain_, std::memory_order_relaxed);
				if (taskBegin >= end_)
				{
					break;
				}
				const auto taskEnd = std::min<std::uint64_t>(taskBegin + grain_, end_);
				const auto start = std::chrono::steady_clock::now();
				body_(taskBegin, taskEnd, userData_);
				participant.usefulMilliseconds +=
				    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
				++participant.taskClaims;
				participant.workUnits += taskEnd - taskBegin;
			}
		}

		void WorkerLoop(Worker& worker, std::size_t workerIndex)
		{
			constexpr std::size_t kAdaptiveInitialPollRounds = LITENN_HAS_X86_AVX2_TARGET ? 65536 : 64;
			constexpr std::size_t kAdaptiveMinPollRounds = LITENN_HAS_X86_AVX2_TARGET ? 1024 : 16;
			constexpr std::size_t kAdaptiveMaxPollRounds = LITENN_HAS_X86_AVX2_TARGET ? 1u << 20 : 1024;
			LiteNNCPUThreadAffinityState affinity;
			std::size_t adaptivePollRounds = kAdaptiveInitialPollRounds;
			auto observedGeneration = worker.generation.load(std::memory_order_relaxed);
			while (true)
			{
				const auto waitPolicy = waitPolicy_.load(std::memory_order_acquire);
				const auto pollRounds =
				    waitPolicy == CPUAOTWorkerWaitPolicy::LowPower
				        ? std::size_t{ 0 }
				        : (waitPolicy == CPUAOTWorkerWaitPolicy::Latency ? kAdaptiveMaxPollRounds : adaptivePollRounds);
				bool observedWorkWhilePolling = false;
				for (std::size_t round = 0; round < pollRounds; ++round)
				{
					if (worker.generation.load(std::memory_order_acquire) != observedGeneration)
					{
						observedWorkWhilePolling = true;
						break;
					}
					LiteNNCPUThreadRelax();
				}
				worker.start.acquire();
				if (waitPolicy == CPUAOTWorkerWaitPolicy::Adaptive)
				{
					adaptivePollRounds = observedWorkWhilePolling
					                         ? std::min(kAdaptiveMaxPollRounds, adaptivePollRounds * 2)
					                         : std::max(kAdaptiveMinPollRounds, adaptivePollRounds / 2);
				}
				observedGeneration = worker.generation.load(std::memory_order_acquire);
				if (stopping_.load(std::memory_order_acquire))
				{
					return;
				}

				affinity.Apply(affinityPolicy_, workerIndex + 1);
				if (profile_)
				{
					RunTasksProfiled(workerIndex + 1);
				}
				else
				{
					RunTasks();
				}
				workersDone_.fetch_add(1, std::memory_order_release);
			}
		}

		std::vector<std::unique_ptr<Worker>> workers_;
		std::mutex runMutex_;
		std::atomic<std::uint64_t> next_{ 0 };
		std::uint64_t begin_{};
		std::uint64_t end_{};
		std::uint64_t grain_{ 1 };
		LiteNNCPUParallelForBody body_{};
		void* userData_{};
		CPUAOTAffinityPolicy affinityPolicy_{ CPUAOTAffinityPolicy::None };
		std::atomic<CPUAOTWorkerWaitPolicy> waitPolicy_{ CPUAOTWorkerWaitPolicy::Adaptive };
		ParallelForProfile* profile_{};
		std::size_t desiredWorkers_{};
		std::atomic<std::size_t> workersDone_{};
		std::atomic<bool> stopping_{};
	};

	LiteNNCPUThreadPool& GetLiteNNCPUThreadPool()
	{
		static LiteNNCPUThreadPool pool(LiteNNCPUMaxThreadCount());
		return pool;
	}

	void LiteNNCPUParallelFor(std::uint64_t begin, std::uint64_t end, std::uint64_t grain,
	                          LiteNNCPUParallelForBody body, void* userData, std::uint64_t threadCount,
	                          CPUAOTAffinityPolicy affinityPolicy,
	                          CPUAOTWorkerWaitPolicy waitPolicy = CPUAOTWorkerWaitPolicy::Adaptive,
	                          LiteNNCPUThreadPool::ParallelForProfile* profile = nullptr)
	{
		GetLiteNNCPUThreadPool().ParallelFor(begin, end, grain, body, userData, static_cast<std::size_t>(threadCount),
		                                     affinityPolicy, waitPolicy, profile);
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
	                                     CPUAOTAffinityPolicy affinityPolicy, CPUAOTWorkerWaitPolicy waitPolicy)
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
		LiteNNCPUParallelFor(0, m, grain, body, &context, threadCount, affinityPolicy, waitPolicy);
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
		const auto policy = ResolveCPUAOTAffinityPolicy(affinityPolicy);
		const auto waitPolicy = ResolveCPUAOTWorkerWaitPolicy(affinityPolicy);
		LiteNNCPUMatMulBiasReLUParallel(lhs, rhs, bias, out, m, k, n, biasRows, threadCount, relu, policy, waitPolicy);
	}

	extern "C" void litenn_cpu_swiglu_f32(const float*, const float* gateAligned, std::int64_t gateOffset,
	                                      std::int64_t gateRows, std::int64_t gateColumns, std::int64_t gateRowStride,
	                                      std::int64_t gateColumnStride, const float*, const float* upAligned,
	                                      std::int64_t upOffset, std::int64_t upRows, std::int64_t upColumns,
	                                      std::int64_t upRowStride, std::int64_t upColumnStride, float*,
	                                      float* outAligned, std::int64_t outOffset, std::int64_t outRows,
	                                      std::int64_t outColumns, std::int64_t outRowStride,
	                                      std::int64_t outColumnStride)
	{
		CPUAOTHelperProfileTimer profileTimer("litenn_cpu_swiglu_f32",
		                                      CompiledModuleCPUHelperProfilerAccess::Enabled()
		                                          ? std::format("gate={}x{} up={}x{} out={}x{}", gateRows, gateColumns,
		                                                        upRows, upColumns, outRows, outColumns)
		                                          : std::string{});
		if (gateRows <= 0 || gateColumns <= 0 || gateRows != upRows || gateColumns != upColumns ||
		    gateRows != outRows || gateColumns != outColumns)
		{
			return;
		}
		for (std::int64_t row = 0; row < gateRows; ++row)
		{
			for (std::int64_t column = 0; column < gateColumns; ++column)
			{
				const auto gate = gateAligned[gateOffset + row * gateRowStride + column * gateColumnStride];
				const auto up = upAligned[upOffset + row * upRowStride + column * upColumnStride];
				outAligned[outOffset + row * outRowStride + column * outColumnStride] =
				    gate / (1.0F + std::exp(-gate)) * up;
			}
		}
	}

	struct RoPEAtPositionsThreadCache
	{
		std::int64_t columns{};
		double base{};
		double frequencyScale{};
		std::int64_t position{};
		bool frequenciesValid{};
		bool anglesValid{};
		std::vector<double> frequencies;
		std::vector<double> cosines;
		std::vector<double> sines;

		void Prepare(std::int64_t newColumns, double newBase, double newFrequencyScale, std::int64_t newPosition)
		{
			const auto pairCount = static_cast<std::size_t>(newColumns / 2);
			if (!frequenciesValid || columns != newColumns || base != newBase)
			{
				columns = newColumns;
				base = newBase;
				frequencies.resize(pairCount);
				cosines.resize(pairCount);
				sines.resize(pairCount);
				for (std::size_t pair = 0; pair < pairCount; ++pair)
				{
					frequencies[pair] = std::pow(base, -2.0 * static_cast<double>(pair) / static_cast<double>(columns));
				}
				frequenciesValid = true;
				anglesValid = false;
			}

			if (!anglesValid || position != newPosition || frequencyScale != newFrequencyScale)
			{
				position = newPosition;
				frequencyScale = newFrequencyScale;
				const auto positionValue = static_cast<double>(position);
				for (std::size_t pair = 0; pair < pairCount; ++pair)
				{
					const auto angle = positionValue * frequencies[pair] * frequencyScale;
					cosines[pair] = std::cos(angle);
					sines[pair] = std::sin(angle);
				}
				anglesValid = true;
			}
		}
	};

	extern "C" void litenn_cpu_rope_at_positions_f32(const float*, const float* inputAligned, std::int64_t inputOffset,
	                                                 std::int64_t inputRows, std::int64_t inputColumns,
	                                                 std::int64_t inputRowStride, std::int64_t inputColumnStride,
	                                                 const std::int64_t*, const std::int64_t* positionAligned,
	                                                 std::int64_t positionOffset, std::int64_t positionSize,
	                                                 std::int64_t positionStride, float*, float* outAligned,
	                                                 std::int64_t outOffset, std::int64_t outRows,
	                                                 std::int64_t outColumns, std::int64_t outRowStride,
	                                                 std::int64_t outColumnStride, double base, double frequencyScale)
	{
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_rope_at_positions_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? std::format("input={}x{} out={}x{}", inputRows, inputColumns, outRows, outColumns)
		        : std::string{});
		if (inputRows <= 0 || inputColumns <= 0 || (inputColumns % 2) != 0 || inputRows != outRows ||
		    inputColumns != outColumns || positionSize != inputRows || !std::isfinite(base) || base <= 0.0 ||
		    !std::isfinite(frequencyScale) || frequencyScale <= 0.0)
		{
			return;
		}

		const auto* input = inputAligned + inputOffset;
		auto* out = outAligned + outOffset;
		thread_local RoPEAtPositionsThreadCache cache;
		for (std::int64_t row = 0; row < inputRows; ++row)
		{
			const auto position = positionAligned[positionOffset + row * positionStride];
			cache.Prepare(inputColumns, base, frequencyScale, position);
			for (std::int64_t pair = 0; pair < inputColumns / 2; ++pair)
			{
				const auto cosine = cache.cosines[static_cast<std::size_t>(pair)];
				const auto sine = cache.sines[static_cast<std::size_t>(pair)];
				const auto first = static_cast<double>(input[row * inputRowStride + pair * 2 * inputColumnStride]);
				const auto second =
				    static_cast<double>(input[row * inputRowStride + (pair * 2 + 1) * inputColumnStride]);
				out[row * outRowStride + pair * 2 * outColumnStride] =
				    static_cast<float>(first * cosine - second * sine);
				out[row * outRowStride + (pair * 2 + 1) * outColumnStride] =
				    static_cast<float>(first * sine + second * cosine);
			}
		}
	}

	extern "C" void litenn_cpu_rms_norm_f32(const float*, const float* inputAligned, std::int64_t inputOffset,
	                                        std::int64_t inputRows, std::int64_t inputColumns,
	                                        std::int64_t inputRowStride, std::int64_t inputColumnStride, const float*,
	                                        const float* scaleAligned, std::int64_t scaleOffset, std::int64_t scaleRows,
	                                        std::int64_t scaleColumns, std::int64_t scaleRowStride,
	                                        std::int64_t scaleColumnStride, float*, float* outAligned,
	                                        std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns,
	                                        std::int64_t outRowStride, std::int64_t outColumnStride, double epsilon)
	{
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_rms_norm_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? std::format("input={}x{} output={}x{}", inputRows, inputColumns, outRows, outColumns)
		        : std::string{});
		if (!inputAligned || !scaleAligned || !outAligned || inputOffset < 0 || scaleOffset < 0 || outOffset < 0 ||
		    inputRows <= 0 || inputColumns <= 0 || inputRowStride <= 0 || inputColumnStride <= 0 || scaleRows != 1 ||
		    scaleColumns != inputColumns || scaleRowStride <= 0 || scaleColumnStride <= 0 || outRows != inputRows ||
		    outColumns != inputColumns || outRowStride <= 0 || outColumnStride <= 0 || !std::isfinite(epsilon) ||
		    epsilon <= 0.0)
		{
			return;
		}
		for (std::int64_t row = 0; row < inputRows; ++row)
		{
			float sumSquares = 0.0F;
			for (std::int64_t column = 0; column < inputColumns; ++column)
			{
				const auto value = inputAligned[inputOffset + row * inputRowStride + column * inputColumnStride];
				sumSquares += value * value;
			}
			const auto inverseRms =
			    1.0F / std::sqrt(sumSquares / static_cast<float>(inputColumns) + static_cast<float>(epsilon));
			for (std::int64_t column = 0; column < inputColumns; ++column)
			{
				const auto inputIndex = inputOffset + row * inputRowStride + column * inputColumnStride;
				const auto scaleIndex = scaleOffset + column * scaleColumnStride;
				const auto outputIndex = outOffset + row * outRowStride + column * outColumnStride;
				outAligned[outputIndex] = inputAligned[inputIndex] * inverseRms * scaleAligned[scaleIndex];
			}
		}
	}

	void ComputeActivePrefixAttentionF32(const float* query, std::int64_t queryColumns, std::int64_t queryColumnStride,
	                                     const float* keys, std::int64_t keyRowStride, std::int64_t keyColumnStride,
	                                     const float* values, std::int64_t valueRowStride,
	                                     std::int64_t valueColumnStride, std::int64_t activeRows, float* out,
	                                     std::int64_t outColumns, std::int64_t outColumnStride, double scale,
	                                     std::span<float> scores)
	{
		if (scores.size() < static_cast<std::size_t>(activeRows))
		{
			return;
		}
		for (std::int64_t col = 0; col < outColumns; ++col)
		{
			out[col * outColumnStride] = 0.0F;
		}

		float maxScore = -std::numeric_limits<float>::infinity();
		for (std::int64_t row = 0; row < activeRows; ++row)
		{
			float score = 0.0F;
			const auto* keyRow = keys + row * keyRowStride;
			for (std::int64_t col = 0; col < queryColumns; ++col)
			{
				score += query[col * queryColumnStride] * keyRow[col * keyColumnStride];
			}
			score *= static_cast<float>(scale);
			scores[static_cast<std::size_t>(row)] = score;
			maxScore = std::max(maxScore, score);
		}

		float denominator = 0.0F;
		for (auto& score : scores)
		{
			score = std::exp(score - maxScore);
			denominator += score;
		}
		if (denominator == 0.0F)
		{
			return;
		}

		const auto invDenominator = 1.0F / denominator;
		for (std::int64_t row = 0; row < activeRows; ++row)
		{
			const auto weight = scores[static_cast<std::size_t>(row)] * invDenominator;
			const auto* valueRow = values + row * valueRowStride;
			for (std::int64_t col = 0; col < outColumns; ++col)
			{
				out[col * outColumnStride] += weight * valueRow[col * valueColumnStride];
			}
		}
	}

	std::span<float> PrepareActivePrefixAttentionScores(std::size_t activeRows)
	{
		thread_local std::vector<float> scores;
		scores.resize(activeRows);
		return scores;
	}

	std::uint64_t ResolveGroupedActivePrefixAttentionThreadCount(std::int64_t queryRows, std::int64_t queryColumns,
	                                                             std::int64_t keyHeads, std::int64_t outColumns,
	                                                             std::int64_t activeRows,
	                                                             std::int64_t queryGroupsPerKVHead,
	                                                             std::uint64_t requestedThreadCount)
	{
		constexpr std::uint64_t kMinimumParallelWork = 1ull << 20;
		const auto hardware = static_cast<std::uint64_t>(LiteNNCPUHardwareThreadCount());
		const auto requestedOrHardware =
		    requestedThreadCount == 0 ? hardware : std::min(requestedThreadCount, hardware);
		const auto activeKVHeads = static_cast<std::uint64_t>(
		    std::min<std::int64_t>(keyHeads, (queryRows + queryGroupsPerKVHead - 1) / queryGroupsPerKVHead));
		if (requestedOrHardware <= 1 || activeKVHeads <= 1)
		{
			return 1;
		}

		auto remaining = kMinimumParallelWork - 1;
		for (const auto extent : { queryRows, activeRows, queryColumns + outColumns })
		{
			const auto unsignedExtent = static_cast<std::uint64_t>(extent);
			if (unsignedExtent > remaining)
			{
				return std::min(requestedOrHardware, activeKVHeads);
			}
			remaining /= unsignedExtent;
		}
		return 1;
	}

	struct GroupedActivePrefixAttentionContext
	{
		const float* queries{};
		std::int64_t queryRows{};
		std::int64_t queryColumns{};
		std::int64_t queryRowStride{};
		std::int64_t queryColumnStride{};
		const float* keys{};
		std::int64_t keyHeadStride{};
		std::int64_t keyRowStride{};
		std::int64_t keyColumnStride{};
		const float* values{};
		std::int64_t valueHeadStride{};
		std::int64_t valueRowStride{};
		std::int64_t valueColumnStride{};
		std::int64_t activeRows{};
		float* out{};
		std::int64_t outColumns{};
		std::int64_t outRowStride{};
		std::int64_t outColumnStride{};
		double scale{};
		std::int64_t queryGroupsPerKVHead{};
	};

	void ComputeGroupedActivePrefixAttentionRange(std::uint64_t begin, std::uint64_t end, void* userData)
	{
		const auto& ctx = *static_cast<const GroupedActivePrefixAttentionContext*>(userData);
		auto scores = PrepareActivePrefixAttentionScores(static_cast<std::size_t>(ctx.activeRows));
		for (auto kvHead = begin; kvHead < end; ++kvHead)
		{
			const auto queryBegin = static_cast<std::int64_t>(kvHead) * ctx.queryGroupsPerKVHead;
			const auto queryEnd = std::min(ctx.queryRows, queryBegin + ctx.queryGroupsPerKVHead);
			for (auto queryHead = queryBegin; queryHead < queryEnd; ++queryHead)
			{
				ComputeActivePrefixAttentionF32(
				    ctx.queries + queryHead * ctx.queryRowStride, ctx.queryColumns, ctx.queryColumnStride,
				    ctx.keys + static_cast<std::int64_t>(kvHead) * ctx.keyHeadStride, ctx.keyRowStride,
				    ctx.keyColumnStride, ctx.values + static_cast<std::int64_t>(kvHead) * ctx.valueHeadStride,
				    ctx.valueRowStride, ctx.valueColumnStride, ctx.activeRows, ctx.out + queryHead * ctx.outRowStride,
				    ctx.outColumns, ctx.outColumnStride, ctx.scale, scores);
			}
		}
	}

	void RecordGroupedActivePrefixAttentionParallelProfile(std::string_view detail, std::uint64_t activeKVHeads,
	                                                       const LiteNNCPUThreadPool::ParallelForProfile& parallel)
	{
		if (!CompiledModuleCPUHelperProfilerAccess::Enabled())
		{
			return;
		}
		CompiledModuleCPUParallelProfileEvent event{
			.helper = "litenn_cpu_active_prefix_attention_f32_rank3_grouped",
			.detail = std::string(detail),
			.workUnits = activeKVHeads,
			.participantCount = parallel.participants.size(),
			.signaledWorkerCount = parallel.signaledWorkerCount,
			.threadPoolLockWaitMilliseconds = parallel.lockWaitMilliseconds,
			.dispatchMilliseconds = parallel.dispatchMilliseconds,
			.parallelWallMilliseconds = parallel.wallMilliseconds,
			.barrierWaitMilliseconds = parallel.barrierWaitMilliseconds,
		};
		if (!parallel.participants.empty())
		{
			event.minParticipantTaskClaims = std::numeric_limits<std::uint64_t>::max();
			event.minParticipantWorkUnits = std::numeric_limits<std::uint64_t>::max();
			event.minParticipantUsefulMilliseconds = std::numeric_limits<double>::max();
			for (std::size_t i = 0; i < parallel.participants.size(); ++i)
			{
				const auto& participant = parallel.participants[i];
				event.taskClaims += participant.taskClaims;
				event.minParticipantTaskClaims = std::min(event.minParticipantTaskClaims, participant.taskClaims);
				event.maxParticipantTaskClaims = std::max(event.maxParticipantTaskClaims, participant.taskClaims);
				event.minParticipantWorkUnits = std::min(event.minParticipantWorkUnits, participant.workUnits);
				event.maxParticipantWorkUnits = std::max(event.maxParticipantWorkUnits, participant.workUnits);
				event.minParticipantUsefulMilliseconds =
				    std::min(event.minParticipantUsefulMilliseconds, participant.usefulMilliseconds);
				event.maxParticipantUsefulMilliseconds =
				    std::max(event.maxParticipantUsefulMilliseconds, participant.usefulMilliseconds);
				if (i == 0)
				{
					event.callerUsefulMilliseconds = participant.usefulMilliseconds;
				}
				else
				{
					event.workerUsefulMilliseconds += participant.usefulMilliseconds;
				}
			}
		}
		CompiledModuleCPUHelperProfilerAccess::RecordParallel(std::move(event));
	}

	extern "C" void litenn_cpu_active_prefix_attention_f32(
	    const float*, const float* queryAligned, std::int64_t queryOffset, std::int64_t queryRows,
	    std::int64_t queryColumns, std::int64_t queryRowStride, std::int64_t queryColumnStride, const float*,
	    const float* keysAligned, std::int64_t keysOffset, std::int64_t keyRows, std::int64_t keyColumns,
	    std::int64_t keyRowStride, std::int64_t keyColumnStride, const float*, const float* valuesAligned,
	    std::int64_t valuesOffset, std::int64_t valueRows, std::int64_t valueColumns, std::int64_t valueRowStride,
	    std::int64_t valueColumnStride, const std::int64_t*, const std::int64_t* positionAligned,
	    std::int64_t positionOffset, std::int64_t positionSize, std::int64_t positionStride, float*, float* outAligned,
	    std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride,
	    std::int64_t outColumnStride, double scale)
	{
		CPUAOTHelperProfileTimer profileTimer("litenn_cpu_active_prefix_attention_f32",
		                                      CompiledModuleCPUHelperProfilerAccess::Enabled()
		                                          ? std::format("query={}x{} keys={}x{} out={}x{}", queryRows,
		                                                        queryColumns, keyRows, keyColumns, outRows, outColumns)
		                                          : std::string{});
		if (queryRows != 1 || outRows != 1 || positionSize != 1 || queryColumns <= 0 || keyRows <= 0 ||
		    keyColumns != queryColumns || valueRows != keyRows || valueColumns != outColumns || outColumns <= 0)
		{
			return;
		}
		const auto currentPosition = positionAligned[positionOffset + positionStride * 0];
		if (currentPosition < 0)
		{
			return;
		}
		const auto activeRows = std::min<std::int64_t>(
		    keyRows, currentPosition > std::numeric_limits<std::int64_t>::max() - 1 ? keyRows : currentPosition + 1);
		if (activeRows <= 0)
		{
			return;
		}
		const auto* query = queryAligned + queryOffset;
		const auto* keys = keysAligned + keysOffset;
		const auto* values = valuesAligned + valuesOffset;
		auto* out = outAligned + outOffset;
		auto scores = PrepareActivePrefixAttentionScores(static_cast<std::size_t>(activeRows));
		ComputeActivePrefixAttentionF32(query, queryColumns, queryColumnStride, keys, keyRowStride, keyColumnStride,
		                                values, valueRowStride, valueColumnStride, activeRows, out, outColumns,
		                                outColumnStride, scale, scores);
	}

	extern "C" void litenn_cpu_scatter_update_axis0_f32_rank3(
	    const float*, const float* dataAligned, std::int64_t dataOffset, std::int64_t dataDim0, std::int64_t dataDim1,
	    std::int64_t dataDim2, std::int64_t dataStride0, std::int64_t dataStride1, std::int64_t dataStride2,
	    const std::int64_t*, const std::int64_t* indicesAligned, std::int64_t indicesOffset, std::int64_t indicesSize,
	    std::int64_t indicesStride, const float*, const float* updatesAligned, std::int64_t updatesOffset,
	    std::int64_t updatesDim0, std::int64_t updatesDim1, std::int64_t updatesDim2, std::int64_t updatesStride0,
	    std::int64_t updatesStride1, std::int64_t updatesStride2, float*, float* outAligned, std::int64_t outOffset,
	    std::int64_t outDim0, std::int64_t outDim1, std::int64_t outDim2, std::int64_t outStride0,
	    std::int64_t outStride1, std::int64_t outStride2)
	{
		CPUAOTHelperProfileTimer profileTimer("litenn_cpu_scatter_update_axis0_f32_rank3",
		                                      CompiledModuleCPUHelperProfilerAccess::Enabled()
		                                          ? std::format("data={}x{}x{} updates={}x{}x{} out={}x{}x{}", dataDim0,
		                                                        dataDim1, dataDim2, updatesDim0, updatesDim1,
		                                                        updatesDim2, outDim0, outDim1, outDim2)
		                                          : std::string{});
		if (indicesSize != 1 || dataDim0 != outDim0 || dataDim1 != outDim1 || dataDim2 != outDim2 || updatesDim0 != 1 ||
		    updatesDim1 != dataDim1 || updatesDim2 != dataDim2 || dataDim0 <= 0 || dataDim1 <= 0 || dataDim2 <= 0)
		{
			return;
		}
		const auto row = indicesAligned[indicesOffset + indicesStride * 0];
		if (row < 0 || row >= dataDim0)
		{
			return;
		}

		const auto* data = dataAligned + dataOffset;
		const auto* updates = updatesAligned + updatesOffset;
		auto* out = outAligned + outOffset;
		if (data != out)
		{
			for (std::int64_t i = 0; i < dataDim0; ++i)
			{
				for (std::int64_t j = 0; j < dataDim1; ++j)
				{
					for (std::int64_t k = 0; k < dataDim2; ++k)
					{
						out[i * outStride0 + j * outStride1 + k * outStride2] =
						    data[i * dataStride0 + j * dataStride1 + k * dataStride2];
					}
				}
			}
		}

		for (std::int64_t j = 0; j < dataDim1; ++j)
		{
			for (std::int64_t k = 0; k < dataDim2; ++k)
			{
				out[row * outStride0 + j * outStride1 + k * outStride2] =
				    updates[j * updatesStride1 + k * updatesStride2];
			}
		}
	}

	extern "C" void litenn_cpu_active_prefix_attention_f32_rank3(
	    const float*, const float* queryAligned, std::int64_t queryOffset, std::int64_t queryRows,
	    std::int64_t queryColumns, std::int64_t queryRowStride, std::int64_t queryColumnStride, const float*,
	    const float* keysAligned, std::int64_t keysOffset, std::int64_t keyRows, std::int64_t keyHeads,
	    std::int64_t keyColumns, std::int64_t keyRowStride, std::int64_t keyHeadStride, std::int64_t keyColumnStride,
	    const float*, const float* valuesAligned, std::int64_t valuesOffset, std::int64_t valueRows,
	    std::int64_t valueHeads, std::int64_t valueColumns, std::int64_t valueRowStride, std::int64_t valueHeadStride,
	    std::int64_t valueColumnStride, const std::int64_t*, const std::int64_t* positionAligned,
	    std::int64_t positionOffset, std::int64_t positionSize, std::int64_t positionStride, float*, float* outAligned,
	    std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride,
	    std::int64_t outColumnStride, double scale, std::int64_t kvHead)
	{
		CPUAOTHelperProfileTimer profileTimer("litenn_cpu_active_prefix_attention_f32_rank3",
		                                      CompiledModuleCPUHelperProfilerAccess::Enabled()
		                                          ? std::format("query={}x{} keys={}x{}x{} out={}x{} kv_head={}",
		                                                        queryRows, queryColumns, keyRows, keyHeads, keyColumns,
		                                                        outRows, outColumns, kvHead)
		                                          : std::string{});
		if (queryRows != 1 || outRows != 1 || positionSize != 1 || queryColumns <= 0 || keyRows <= 0 ||
		    keyColumns != queryColumns || valueRows != keyRows || valueHeads != keyHeads || kvHead < 0 ||
		    kvHead >= keyHeads || valueColumns != outColumns || outColumns <= 0)
		{
			return;
		}
		const auto currentPosition = positionAligned[positionOffset + positionStride * 0];
		if (currentPosition < 0)
		{
			return;
		}
		const auto activeRows = std::min<std::int64_t>(
		    keyRows, currentPosition > std::numeric_limits<std::int64_t>::max() - 1 ? keyRows : currentPosition + 1);
		if (activeRows <= 0)
		{
			return;
		}
		const auto* query = queryAligned + queryOffset;
		const auto* keys = keysAligned + keysOffset;
		const auto* values = valuesAligned + valuesOffset;
		auto* out = outAligned + outOffset;
		auto scores = PrepareActivePrefixAttentionScores(static_cast<std::size_t>(activeRows));
		ComputeActivePrefixAttentionF32(query, queryColumns, queryColumnStride, keys + kvHead * keyHeadStride,
		                                keyRowStride, keyColumnStride, values + kvHead * valueHeadStride,
		                                valueRowStride, valueColumnStride, activeRows, out, outColumns, outColumnStride,
		                                scale, scores);
	}

	extern "C" void litenn_cpu_active_prefix_attention_f32_rank3_grouped(
	    const float*, const float* queryAligned, std::int64_t queryOffset, std::int64_t queryRows,
	    std::int64_t queryColumns, std::int64_t queryRowStride, std::int64_t queryColumnStride, const float*,
	    const float* keysAligned, std::int64_t keysOffset, std::int64_t keyRows, std::int64_t keyHeads,
	    std::int64_t keyColumns, std::int64_t keyRowStride, std::int64_t keyHeadStride, std::int64_t keyColumnStride,
	    const float*, const float* valuesAligned, std::int64_t valuesOffset, std::int64_t valueRows,
	    std::int64_t valueHeads, std::int64_t valueColumns, std::int64_t valueRowStride, std::int64_t valueHeadStride,
	    std::int64_t valueColumnStride, const std::int64_t*, const std::int64_t* positionAligned,
	    std::int64_t positionOffset, std::int64_t positionSize, std::int64_t positionStride, float*, float* outAligned,
	    std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride,
	    std::int64_t outColumnStride, double scale, std::int64_t queryGroupsPerKVHead,
	    std::uint64_t requestedThreadCount, std::uint64_t schedulingPolicyValue)
	{
		if (queryRows <= 0 || outRows != queryRows || positionSize != 1 || queryColumns <= 0 || keyRows <= 0 ||
		    keyHeads <= 0 || queryGroupsPerKVHead <= 0 || keyColumns != queryColumns || valueRows != keyRows ||
		    valueHeads != keyHeads || valueColumns != outColumns || outColumns <= 0 ||
		    queryRows > keyHeads * queryGroupsPerKVHead)
		{
			return;
		}
		const auto currentPosition = positionAligned[positionOffset + positionStride * 0];
		if (currentPosition < 0)
		{
			return;
		}
		const auto activeRows = std::min<std::int64_t>(
		    keyRows, currentPosition > std::numeric_limits<std::int64_t>::max() - 1 ? keyRows : currentPosition + 1);
		if (activeRows <= 0)
		{
			return;
		}
		const auto threadCount = ResolveGroupedActivePrefixAttentionThreadCount(
		    queryRows, queryColumns, keyHeads, outColumns, activeRows, queryGroupsPerKVHead, requestedThreadCount);
		const auto profileDetail =
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? std::format("queries={}x{} keys={}x{}x{} out={}x{} groups_per_kv={} "
		                      "requested_threads={} resolved_threads={}",
		                      queryRows, queryColumns, keyRows, keyHeads, keyColumns, outRows, outColumns,
		                      queryGroupsPerKVHead, requestedThreadCount, threadCount)
		        : std::string{};
		CPUAOTHelperProfileTimer profileTimer("litenn_cpu_active_prefix_attention_f32_rank3_grouped", profileDetail);

		GroupedActivePrefixAttentionContext context{
			.queries = queryAligned + queryOffset,
			.queryRows = queryRows,
			.queryColumns = queryColumns,
			.queryRowStride = queryRowStride,
			.queryColumnStride = queryColumnStride,
			.keys = keysAligned + keysOffset,
			.keyHeadStride = keyHeadStride,
			.keyRowStride = keyRowStride,
			.keyColumnStride = keyColumnStride,
			.values = valuesAligned + valuesOffset,
			.valueHeadStride = valueHeadStride,
			.valueRowStride = valueRowStride,
			.valueColumnStride = valueColumnStride,
			.activeRows = activeRows,
			.out = outAligned + outOffset,
			.outColumns = outColumns,
			.outRowStride = outRowStride,
			.outColumnStride = outColumnStride,
			.scale = scale,
			.queryGroupsPerKVHead = queryGroupsPerKVHead,
		};
		const auto activeKVHeads = static_cast<std::uint64_t>(
		    std::min<std::int64_t>(keyHeads, (queryRows + queryGroupsPerKVHead - 1) / queryGroupsPerKVHead));
		if (threadCount <= 1)
		{
			LiteNNCPUThreadPool::ParallelForProfile parallelProfile;
			const auto start = CompiledModuleCPUHelperProfilerAccess::Enabled()
			                       ? std::chrono::steady_clock::now()
			                       : std::chrono::steady_clock::time_point{};
			ComputeGroupedActivePrefixAttentionRange(0, activeKVHeads, &context);
			if (CompiledModuleCPUHelperProfilerAccess::Enabled())
			{
				const auto elapsed =
				    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
				parallelProfile.participants = { LiteNNCPUThreadPool::ParticipantProfile{
					.taskClaims = 1, .workUnits = activeKVHeads, .usefulMilliseconds = elapsed } };
				parallelProfile.wallMilliseconds = elapsed;
			}
			RecordGroupedActivePrefixAttentionParallelProfile(profileDetail, activeKVHeads, parallelProfile);
			return;
		}
		LiteNNCPUThreadPool::ParallelForProfile parallelProfile;
		LiteNNCPUParallelFor(0, activeKVHeads, 1, ComputeGroupedActivePrefixAttentionRange, &context, threadCount,
		                     ResolveCPUAOTAffinityPolicy(schedulingPolicyValue),
		                     ResolveCPUAOTWorkerWaitPolicy(schedulingPolicyValue),
		                     CompiledModuleCPUHelperProfilerAccess::Enabled() ? &parallelProfile : nullptr);
		RecordGroupedActivePrefixAttentionParallelProfile(profileDetail, activeKVHeads, parallelProfile);
	}

	extern "C" void litenn_cpu_grouped_paged_attention_f32(
	    const float*, const float* queryAligned, std::int64_t queryOffset, std::int64_t queryRows,
	    std::int64_t queryColumns, std::int64_t queryRowStride, std::int64_t queryColumnStride, const float*,
	    const float* kvAligned, std::int64_t kvOffset, std::int64_t kvPlanes, std::int64_t residentPages,
	    std::int64_t pageSize, std::int64_t kvHeads, std::int64_t kvColumns, std::int64_t kvPlaneStride,
	    std::int64_t kvPageStride, std::int64_t kvTokenStride, std::int64_t kvHeadStride, std::int64_t kvColumnStride,
	    const std::int64_t*, const std::int64_t* pageTableAligned, std::int64_t pageTableOffset,
	    std::int64_t pageTableSize, std::int64_t pageTableStride, const std::int64_t*,
	    const std::int64_t* pageDescriptorAligned, std::int64_t pageDescriptorOffset, std::int64_t pageDescriptorRows,
	    std::int64_t pageDescriptorColumns, std::int64_t pageDescriptorRowStride,
	    std::int64_t pageDescriptorColumnStride, const std::int64_t*, const std::int64_t* activeLengthAligned,
	    std::int64_t activeLengthOffset, std::int64_t activeLengthSize, std::int64_t activeLengthStride, float*,
	    float* outAligned, std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns,
	    std::int64_t outRowStride, std::int64_t outColumnStride, double scale, std::int64_t queryGroupsPerKVHead)
	{
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_grouped_paged_attention_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? std::format("queries={}x{} kv={}x{}x{}x{}x{} out={}x{} groups_per_kv={}", queryRows, queryColumns,
		                      kvPlanes, residentPages, pageSize, kvHeads, kvColumns, outRows, outColumns,
		                      queryGroupsPerKVHead)
		        : std::string{});
		if (queryRows <= 0 || queryColumns <= 0 || outRows != queryRows || outColumns != queryColumns ||
		    kvPlanes != 2 || residentPages <= 0 || pageSize <= 0 || kvHeads <= 0 || kvColumns != queryColumns ||
		    queryGroupsPerKVHead <= 0 || queryRows > kvHeads * queryGroupsPerKVHead || pageTableSize <= 0 ||
		    pageDescriptorRows != residentPages || pageDescriptorColumns < 4 || activeLengthSize != 1)
		{
			return;
		}

		const auto activeLength = activeLengthAligned[activeLengthOffset];
		if (activeLength <= 0 || (activeLength + pageSize - 1) / pageSize > pageTableSize)
		{
			return;
		}

		const auto* queries = queryAligned + queryOffset;
		const auto* kv = kvAligned + kvOffset;
		const auto* pageTable = pageTableAligned + pageTableOffset;
		const auto* pageDescriptors = pageDescriptorAligned + pageDescriptorOffset;
		auto* out = outAligned + outOffset;
		std::vector<float> scores(static_cast<std::size_t>(activeLength));

		for (std::int64_t queryHead = 0; queryHead < queryRows; ++queryHead)
		{
			const auto kvHead = queryHead / queryGroupsPerKVHead;
			for (std::int64_t col = 0; col < outColumns; ++col)
			{
				out[queryHead * outRowStride + col * outColumnStride] = 0.0F;
			}

			float maxScore = -std::numeric_limits<float>::infinity();
			for (std::int64_t token = 0; token < activeLength; ++token)
			{
				const auto logicalPage = token / pageSize;
				const auto tokenInPage = token % pageSize;
				const auto resident = pageTable[logicalPage * pageTableStride];
				if (resident < 0 || resident >= residentPages)
				{
					return;
				}
				const auto descriptor = pageDescriptors + resident * pageDescriptorRowStride;
				const auto descriptorLogicalPage = descriptor[0 * pageDescriptorColumnStride];
				const auto descriptorFirstToken = descriptor[1 * pageDescriptorColumnStride];
				const auto descriptorTokenCount = descriptor[2 * pageDescriptorColumnStride];
				const auto descriptorFlags = descriptor[3 * pageDescriptorColumnStride];
				if (descriptorLogicalPage != logicalPage || descriptorFirstToken > token ||
				    descriptorFirstToken + descriptorTokenCount <= token || (descriptorFlags & 1) == 0)
				{
					return;
				}

				const auto* query = queries + queryHead * queryRowStride;
				const auto* key = kv + resident * kvPageStride + tokenInPage * kvTokenStride + kvHead * kvHeadStride;
				float score = 0.0F;
				for (std::int64_t col = 0; col < queryColumns; ++col)
				{
					score += query[col * queryColumnStride] * key[col * kvColumnStride];
				}
				score *= static_cast<float>(scale);
				scores[static_cast<std::size_t>(token)] = score;
				maxScore = std::max(maxScore, score);
			}

			float denominator = 0.0F;
			for (auto& score : scores)
			{
				score = std::exp(score - maxScore);
				denominator += score;
			}
			if (denominator == 0.0F)
			{
				return;
			}

			const auto invDenominator = 1.0F / denominator;
			for (std::int64_t token = 0; token < activeLength; ++token)
			{
				const auto logicalPage = token / pageSize;
				const auto tokenInPage = token % pageSize;
				const auto resident = pageTable[logicalPage * pageTableStride];
				const auto* value =
				    kv + kvPlaneStride + resident * kvPageStride + tokenInPage * kvTokenStride + kvHead * kvHeadStride;
				const auto weight = scores[static_cast<std::size_t>(token)] * invDenominator;
				for (std::int64_t col = 0; col < outColumns; ++col)
				{
					out[queryHead * outRowStride + col * outColumnStride] += weight * value[col * kvColumnStride];
				}
			}
		}
	}

	std::uint8_t GGMLBlockByte(const std::uint8_t* block, std::int64_t byteStride, std::uint64_t byteOffset)
	{
		return block[static_cast<std::int64_t>(byteOffset) * byteStride];
	}

	float ReadGGMLF16Strided(const std::uint8_t* block, std::int64_t byteStride, std::uint64_t byteOffset)
	{
		const auto low = static_cast<std::uint16_t>(GGMLBlockByte(block, byteStride, byteOffset));
		const auto high = static_cast<std::uint16_t>(GGMLBlockByte(block, byteStride, byteOffset + 1));
		return QuantizationDetail::Float16BitsToFloat32(static_cast<std::uint16_t>(low | (high << 8U)));
	}

	void GGMLQ4Or5KScaleMin(const std::uint8_t* block, std::int64_t byteStride, std::uint64_t subblock,
	                        std::uint32_t& scale, std::uint32_t& minimum)
	{
		const auto belowFour = subblock < 4;
		const auto scaleLowOffset = belowFour ? subblock : subblock + 4;
		const auto scaleLow = static_cast<std::uint32_t>(GGMLBlockByte(block, byteStride, 4 + scaleLowOffset));
		const auto minLow = static_cast<std::uint32_t>(GGMLBlockByte(block, byteStride, 4 + subblock + 4));
		scale = scaleLow & 63U;
		minimum = minLow & 63U;
		if (!belowFour)
		{
			const auto highOffset = subblock - 4;
			const auto highSource = static_cast<std::uint32_t>(GGMLBlockByte(block, byteStride, 4 + highOffset));
			const auto minHighSource = static_cast<std::uint32_t>(GGMLBlockByte(block, byteStride, 4 + subblock));
			scale = (scaleLow & 15U) | ((highSource >> 6U) << 4U);
			minimum = (minLow >> 4U) | ((minHighSource >> 6U) << 4U);
		}
	}

	float DotGGMLQ8_0BlockF32(const std::uint8_t* block, std::int64_t byteStride, const float* lhs,
	                          std::int64_t lhsStride)
	{
		const auto d = ReadGGMLF16Strided(block, byteStride, 0);
		float sum = 0.0F;
		for (std::uint64_t lane = 0; lane < 32; lane += 4)
		{
			const auto base = static_cast<std::int64_t>(lane) * lhsStride;
			sum += lhs[base] * static_cast<float>(static_cast<std::int8_t>(GGMLBlockByte(block, byteStride, 2 + lane)));
			sum += lhs[base + lhsStride] *
			       static_cast<float>(static_cast<std::int8_t>(GGMLBlockByte(block, byteStride, 3 + lane)));
			sum += lhs[base + lhsStride * 2] *
			       static_cast<float>(static_cast<std::int8_t>(GGMLBlockByte(block, byteStride, 4 + lane)));
			sum += lhs[base + lhsStride * 3] *
			       static_cast<float>(static_cast<std::int8_t>(GGMLBlockByte(block, byteStride, 5 + lane)));
		}
		return d * sum;
	}

	void AccumulateGGMLQ8_0BlockF32x4(const std::uint8_t* const blocks[4], const bool valid[4], std::int64_t byteStride,
	                                  const float* lhs, std::int64_t lhsStride, float acc[4])
	{
		float d[4] = {};
		for (int column = 0; column < 4; ++column)
		{
			if (valid[column])
			{
				d[column] = ReadGGMLF16Strided(blocks[column], byteStride, 0);
			}
		}

		float sum[4] = {};
		for (std::uint64_t lane = 0; lane < 32; lane += 4)
		{
			for (std::uint64_t local = 0; local < 4; ++local)
			{
				const auto localLane = lane + local;
				const auto lhsValue = lhs[static_cast<std::int64_t>(localLane) * lhsStride];
				for (int column = 0; column < 4; ++column)
				{
					if (!valid[column])
					{
						continue;
					}
					const auto quant = static_cast<float>(
					    static_cast<std::int8_t>(GGMLBlockByte(blocks[column], byteStride, 2 + localLane)));
					sum[column] += lhsValue * quant;
				}
			}
		}

		for (int column = 0; column < 4; ++column)
		{
			if (valid[column])
			{
				acc[column] += d[column] * sum[column];
			}
		}
	}

	enum class GGMLActivationDotMode
	{
		DirectFloat32,
		Q8KStaged,
	};

	constexpr std::uint64_t kGGMLQ8KActivationBlockBytes = sizeof(GGMLQ8KActivationBlock);

	bool IsGGMLQ8KStagedMatMulFormat(QuantizedBlockFormat format)
	{
		return format == QuantizedBlockFormat::GGML_Q4_K || format == QuantizedBlockFormat::GGML_Q5_K ||
		       format == QuantizedBlockFormat::GGML_Q6_K;
	}

	std::uint64_t ResolveGGMLBlockMatMulThreadCount(QuantizedBlockFormat format,
	                                                GGMLActivationDotMode activationDotMode, std::uint64_t operations,
	                                                std::uint64_t outputUnits, std::uint64_t requestedThreadCount)
	{
		if (outputUnits <= 1 || operations < (1ull << 20))
		{
			return 1;
		}

		const auto requestedOrHardware = requestedThreadCount == 0
		                                     ? static_cast<std::uint64_t>(LiteNNCPUHardwareThreadCount())
		                                     : requestedThreadCount;
		if (requestedOrHardware <= 1)
		{
			return 1;
		}

		auto capped = requestedOrHardware;
		if (requestedThreadCount == 0)
		{
			capped = std::min<std::uint64_t>(capped, 32);
			if (outputUnits <= 32)
			{
				capped = std::min<std::uint64_t>(capped, 4);
			}
			else if (outputUnits <= 128)
			{
				capped = std::min<std::uint64_t>(capped, 8);
			}
			if (activationDotMode == GGMLActivationDotMode::Q8KStaged &&
			    (format == QuantizedBlockFormat::GGML_Q4_K || format == QuantizedBlockFormat::GGML_Q5_K))
			{
				capped = std::min<std::uint64_t>(capped, 8);
			}
		}

		return std::max<std::uint64_t>(1, std::min<std::uint64_t>(capped, outputUnits));
	}

	std::uint64_t ResolveGGMLFieldInterleavedV4ThreadCount(QuantizedBlockFormat format, std::int64_t lhsRows,
	                                                       std::int64_t lhsColumns, std::int64_t outColumns,
	                                                       std::uint64_t outputUnits,
	                                                       std::uint64_t requestedThreadCount, bool grouped)
	{
		const auto operations = static_cast<std::uint64_t>(std::max<std::int64_t>(0, lhsRows)) *
		                        static_cast<std::uint64_t>(std::max<std::int64_t>(0, lhsColumns)) *
		                        static_cast<std::uint64_t>(std::max<std::int64_t>(0, outColumns));
		if (outputUnits <= 1 || operations < (1ull << 20))
		{
			return 1;
		}
		if (lhsRows != 1)
		{
			return ResolveGGMLBlockMatMulThreadCount(format, GGMLActivationDotMode::Q8KStaged, operations, outputUnits,
			                                         requestedThreadCount);
		}

		const auto requestedOrHardware = requestedThreadCount == 0
		                                     ? static_cast<std::uint64_t>(LiteNNCPUHardwareThreadCount())
		                                     : requestedThreadCount;
		auto shapeLimit = requestedThreadCount == 0 ? std::uint64_t{ 8 } : std::uint64_t{ 16 };
		if (!grouped && outColumns <= 2048)
		{
			shapeLimit = 2;
		}
		else if (!grouped && format == QuantizedBlockFormat::GGML_Q4_K && lhsColumns == outColumns &&
		         outColumns <= 8192)
		{
			shapeLimit = 4;
		}
		else if (!grouped && format == QuantizedBlockFormat::GGML_Q4_K && outColumns <= 8192)
		{
			shapeLimit = 8;
		}
		return std::max<std::uint64_t>(
		    1, std::min({ requestedOrHardware, shapeLimit, std::max<std::uint64_t>(1, outputUnits) }));
	}

	std::string BuildGGMLBlockMatMulProfileDetail(QuantizedBlockFormat format, GGMLActivationDotMode activationDotMode,
	                                              std::int64_t lhsRows, std::int64_t lhsColumns, std::int64_t outRows,
	                                              std::int64_t outColumns, std::uint64_t requestedThreadCount,
	                                              std::optional<std::uint64_t> resolvedThreadCount = std::nullopt)
	{
		const auto positiveLhsRows = static_cast<std::uint64_t>(std::max<std::int64_t>(0, lhsRows));
		const auto positiveLhsColumns = static_cast<std::uint64_t>(std::max<std::int64_t>(0, lhsColumns));
		const auto positiveOutColumns = static_cast<std::uint64_t>(std::max<std::int64_t>(0, outColumns));
		const auto outputElements = positiveLhsRows * positiveOutColumns;
		const auto operations = outputElements * positiveLhsColumns;
		auto outputUnits = outputElements;
		if (format == QuantizedBlockFormat::GGML_Q8_0 || format == QuantizedBlockFormat::GGML_Q4_K ||
		    format == QuantizedBlockFormat::GGML_Q5_K || format == QuantizedBlockFormat::GGML_Q6_K)
		{
			outputUnits = positiveLhsRows * ((positiveOutColumns + 3) / 4);
		}
		return std::format("format={} activation={} lhs={}x{} out={}x{} requested_threads={} resolved_threads={}",
		                   QuantizedBlockFormatName(format),
		                   activationDotMode == GGMLActivationDotMode::Q8KStaged ? "q8k_staged" : "direct", lhsRows,
		                   lhsColumns, outRows, outColumns, requestedThreadCount,
		                   resolvedThreadCount.value_or(ResolveGGMLBlockMatMulThreadCount(
		                       format, activationDotMode, operations, outputUnits, requestedThreadCount)));
	}

	std::string BuildGGMLMixedBlockMatMulProfileDetail(std::span<const QuantizedBlockFormat> formats,
	                                                   GGMLActivationDotMode activationDotMode, std::int64_t lhsRows,
	                                                   std::int64_t lhsColumns, std::int64_t outRows,
	                                                   std::int64_t outColumns, std::uint64_t requestedThreadCount)
	{
		const auto profileFormat = std::ranges::contains(formats, QuantizedBlockFormat::GGML_Q6_K)
		                               ? QuantizedBlockFormat::GGML_Q6_K
		                               : formats.front();
		auto detail = BuildGGMLBlockMatMulProfileDetail(profileFormat, activationDotMode, lhsRows, lhsColumns, outRows,
		                                                outColumns, requestedThreadCount);
		detail.append(" formats=");
		for (std::size_t i = 0; i < formats.size(); ++i)
		{
			if (i != 0)
			{
				detail.push_back(',');
			}
			detail.append(QuantizedBlockFormatName(formats[i]));
		}
		return detail;
	}

#if LITENN_HAS_X86_AVX2_TARGET
	bool LiteNNCPUHasAVX2()
	{
		static const bool supported = [] {
			__builtin_cpu_init();
			return __builtin_cpu_supports("avx2");
		}();
		return supported;
	}

	bool LiteNNCPUHasAVX2F16C()
	{
		static const bool supported = [] {
			__builtin_cpu_init();
			return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");
		}();
		return supported;
	}

	LITENN_TARGET_AVX2 std::int32_t DotGGMLQ8KWithU8Vector16AVX2(const std::int8_t* q8, __m128i rawBytes,
	                                                             std::int16_t zeroPoint)
	{
		const auto q8Bytes = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8));
		const auto rawTimesQ8Pairs = _mm_maddubs_epi16(rawBytes, q8Bytes);
		auto sums = _mm256_cvtepi16_epi32(rawTimesQ8Pairs);
		if (zeroPoint != 0)
		{
			const auto q8I16 = _mm256_cvtepi8_epi16(q8Bytes);
			const auto q8PairSums = _mm256_madd_epi16(q8I16, _mm256_set1_epi16(1));
			sums = _mm256_sub_epi32(sums, _mm256_mullo_epi32(q8PairSums, _mm256_set1_epi32(zeroPoint)));
		}
		auto sum128 = _mm_add_epi32(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
		sum128 = _mm_hadd_epi32(sum128, sum128);
		sum128 = _mm_hadd_epi32(sum128, sum128);
		return _mm_cvtsi128_si32(sum128);
	}

	LITENN_TARGET_AVX2 __m128i DotGGMLQ8KWithU8Vector16PairAVX2(const std::int8_t* q8, __m128i rawBytes0,
	                                                            __m128i rawBytes1, std::int16_t zeroPoint)
	{
		auto rawBytes = _mm256_castsi128_si256(rawBytes0);
		rawBytes = _mm256_inserti128_si256(rawBytes, rawBytes1, 1);
		const auto q8Bytes128 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(q8));
		const auto q8Bytes = _mm256_broadcastsi128_si256(q8Bytes128);
		auto sums = _mm256_madd_epi16(_mm256_maddubs_epi16(rawBytes, q8Bytes), _mm256_set1_epi16(1));
		sums = _mm256_hadd_epi32(sums, sums);
		sums = _mm256_hadd_epi32(sums, sums);
		auto pair = _mm_unpacklo_epi32(_mm256_castsi256_si128(sums), _mm256_extracti128_si256(sums, 1));
		if (zeroPoint != 0)
		{
			auto q8Sum = _mm_madd_epi16(_mm_maddubs_epi16(_mm_set1_epi8(1), q8Bytes128), _mm_set1_epi16(1));
			q8Sum = _mm_hadd_epi32(q8Sum, q8Sum);
			q8Sum = _mm_hadd_epi32(q8Sum, q8Sum);
			pair = _mm_sub_epi32(pair, _mm_set1_epi32(_mm_cvtsi128_si32(q8Sum) * zeroPoint));
		}
		return pair;
	}

	LITENN_TARGET_AVX2 std::int32_t DotGGMLQ8KWithU8Raw16AVX2(const std::int8_t* q8, const std::uint8_t* raw,
	                                                          std::int16_t zeroPoint)
	{
		return DotGGMLQ8KWithU8Vector16AVX2(q8, _mm_loadu_si128(reinterpret_cast<const __m128i*>(raw)), zeroPoint);
	}
#endif

	int RoundGGMLQ8KValue(float value)
	{
		constexpr float roundToNearestBias = 12582912.0F;
		const auto bits = std::bit_cast<std::int32_t>(value + roundToNearestBias);
		return (bits & 0x007fffff) - 0x00400000;
	}

	void QuantizeGGMLQ8KActivationBlock(const float* lhs, std::int64_t lhsStride, GGMLQ8KActivationBlock& out)
	{
		float signedMax = 0.0F;
		float absMax = 0.0F;
		for (std::uint64_t lane = 0; lane < 256; ++lane)
		{
			const auto value = lhs[static_cast<std::int64_t>(lane) * lhsStride];
			const auto absValue = std::fabs(value);
			if (absValue > absMax)
			{
				absMax = absValue;
				signedMax = value;
			}
		}

		std::memset(out.qs, 0, sizeof(out.qs));
		std::memset(out.bsums, 0, sizeof(out.bsums));
		if (absMax == 0.0F)
		{
			out.d = 0.0F;
			return;
		}

		const auto inverseScale = -127.0F / signedMax;
		for (std::uint64_t lane = 0; lane < 256; ++lane)
		{
			const auto value = lhs[static_cast<std::int64_t>(lane) * lhsStride];
			auto quant = RoundGGMLQ8KValue(inverseScale * value);
			quant = std::clamp(quant, -127, 127);
			out.qs[lane] = static_cast<std::int8_t>(quant);
		}

		for (std::uint64_t group = 0; group < 16; ++group)
		{
			int sum = 0;
			for (std::uint64_t lane = 0; lane < 16; ++lane)
			{
				sum += out.qs[group * 16 + lane];
			}
			out.bsums[group] = static_cast<std::int16_t>(sum);
		}
		out.d = 1.0F / inverseScale;
	}

	struct GGMLQ8KActivationPreparationProfile
	{
		bool cacheHit{};
		double lookupMilliseconds{};
		double copyMilliseconds{};
		double quantizeMilliseconds{};
	};

	class GGMLQ8KActivationThreadCache
	{
	public:
		const GGMLQ8KActivationBlock* Prepare(const float* lhs, std::int64_t rows, std::int64_t columns,
		                                      std::int64_t rowStride, std::int64_t columnStride,
		                                      GGMLQ8KActivationPreparationProfile* profile = nullptr)
		{
			const auto elementCount = static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
			bool matches = false;
			if (profile)
			{
				const auto start = std::chrono::steady_clock::now();
				matches = Matches(lhs, rows, columns, rowStride, columnStride, elementCount);
				profile->lookupMilliseconds =
				    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
				profile->cacheHit = matches;
			}
			else
			{
				matches = Matches(lhs, rows, columns, rowStride, columnStride, elementCount);
			}
			if (matches)
			{
				return blocks_.data();
			}

			const auto copyStart = profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
			source_.resize(elementCount);
			if (columnStride == 1 && rowStride == columns)
			{
				std::memcpy(source_.data(), lhs, elementCount * sizeof(float));
			}
			else
			{
				for (std::int64_t row = 0; row < rows; ++row)
				{
					for (std::int64_t column = 0; column < columns; ++column)
					{
						source_[static_cast<std::size_t>(row * columns + column)] =
						    lhs[row * rowStride + column * columnStride];
					}
				}
			}
			if (profile)
			{
				profile->copyMilliseconds =
				    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - copyStart).count();
			}

			const auto quantizeStart =
			    profile ? std::chrono::steady_clock::now() : std::chrono::steady_clock::time_point{};
			const auto blockCount = static_cast<std::size_t>(columns / 256);
			blocks_.resize(static_cast<std::size_t>(rows) * blockCount);
			for (std::int64_t row = 0; row < rows; ++row)
			{
				const auto* sourceRow = source_.data() + static_cast<std::size_t>(row * columns);
				for (std::size_t block = 0; block < blockCount; ++block)
				{
					QuantizeGGMLQ8KActivationBlock(sourceRow + block * 256, 1,
					                               blocks_[static_cast<std::size_t>(row) * blockCount + block]);
				}
			}
			rows_ = rows;
			columns_ = columns;
			if (profile)
			{
				profile->quantizeMilliseconds =
				    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - quantizeStart).count();
			}
			return blocks_.data();
		}

		const float* PrepareSwiGLU(const float* gate, std::int64_t gateOffset, std::int64_t rows, std::int64_t columns,
		                           std::int64_t gateRowStride, std::int64_t gateColumnStride, const float* up,
		                           std::int64_t upOffset, std::int64_t upRowStride, std::int64_t upColumnStride)
		{
			const auto elementCount = static_cast<std::size_t>(rows) * static_cast<std::size_t>(columns);
			const auto blockCount = static_cast<std::size_t>(columns / 256);
			source_.resize(elementCount);
			blocks_.resize(static_cast<std::size_t>(rows) * blockCount);
			for (std::int64_t row = 0; row < rows; ++row)
			{
				auto* sourceRow = source_.data() + static_cast<std::size_t>(row * columns);
				for (std::size_t block = 0; block < blockCount; ++block)
				{
					const auto columnBase = static_cast<std::int64_t>(block * 256);
					for (std::int64_t lane = 0; lane < 256; ++lane)
					{
						const auto column = columnBase + lane;
						const auto gateValue = gate[gateOffset + row * gateRowStride + column * gateColumnStride];
						const auto upValue = up[upOffset + row * upRowStride + column * upColumnStride];
						sourceRow[column] = gateValue / (1.0F + std::exp(-gateValue)) * upValue;
					}
					QuantizeGGMLQ8KActivationBlock(sourceRow + block * 256, 1,
					                               blocks_[static_cast<std::size_t>(row) * blockCount + block]);
				}
			}
			rows_ = rows;
			columns_ = columns;
			return source_.data();
		}

	private:
		bool Matches(const float* lhs, std::int64_t rows, std::int64_t columns, std::int64_t rowStride,
		             std::int64_t columnStride, std::size_t elementCount) const
		{
			if (rows_ != rows || columns_ != columns || source_.size() != elementCount)
			{
				return false;
			}
			if (columnStride == 1 && rowStride == columns)
			{
				return std::memcmp(source_.data(), lhs, elementCount * sizeof(float)) == 0;
			}
			for (std::int64_t row = 0; row < rows; ++row)
			{
				for (std::int64_t column = 0; column < columns; ++column)
				{
					const auto& cached = source_[static_cast<std::size_t>(row * columns + column)];
					const auto& current = lhs[row * rowStride + column * columnStride];
					if (std::memcmp(&cached, &current, sizeof(float)) != 0)
					{
						return false;
					}
				}
			}
			return true;
		}

		std::int64_t rows_{ -1 };
		std::int64_t columns_{ -1 };
		std::vector<float> source_;
		std::vector<GGMLQ8KActivationBlock> blocks_;
	};

	GGMLQ8KActivationThreadCache& GetGGMLQ8KActivationThreadCache()
	{
		thread_local GGMLQ8KActivationThreadCache cache;
		return cache;
	}

	const GGMLQ8KActivationBlock* PrepareCachedGGMLQ8KActivation(const float* lhs, std::int64_t rows,
	                                                             std::int64_t columns, std::int64_t rowStride,
	                                                             std::int64_t columnStride,
	                                                             GGMLQ8KActivationPreparationProfile* profile = nullptr)
	{
		return GetGGMLQ8KActivationThreadCache().Prepare(lhs, rows, columns, rowStride, columnStride, profile);
	}

	extern "C" std::uint64_t litenn_cpu_ggml_q8k_activation_block_bytes()
	{
		return kGGMLQ8KActivationBlockBytes;
	}

	extern "C" void litenn_cpu_ggml_prepare_q8k_activation_f32(const float*, const float* lhsAligned,
	                                                           std::int64_t lhsOffset, std::int64_t lhsRows,
	                                                           std::int64_t lhsColumns, std::int64_t lhsRowStride,
	                                                           std::int64_t lhsColumnStride, std::uint8_t*,
	                                                           std::uint8_t* stagedAligned, std::int64_t stagedOffset,
	                                                           std::int64_t stagedBytes, std::int64_t stagedStride)
	{
		if (!lhsAligned || !stagedAligned || lhsOffset < 0 || lhsRows < 0 || lhsColumns <= 0 || lhsRowStride <= 0 ||
		    lhsColumnStride <= 0 || stagedOffset < 0 || stagedBytes < 0 || stagedStride != 1 ||
		    static_cast<std::uint64_t>(lhsColumns) % 256 != 0)
		{
			return;
		}
		const auto blockCount = static_cast<std::uint64_t>(lhsColumns) / 256;
		const auto requiredBytes = static_cast<std::uint64_t>(lhsRows) * blockCount * kGGMLQ8KActivationBlockBytes;
		if (static_cast<std::uint64_t>(stagedBytes) < requiredBytes)
		{
			return;
		}
		for (std::int64_t row = 0; row < lhsRows; ++row)
		{
			const auto* lhsRow = lhsAligned + lhsOffset + row * lhsRowStride;
			for (std::uint64_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
			{
				const auto* lhsBlock = lhsRow + static_cast<std::int64_t>(blockIndex * 256) * lhsColumnStride;
				auto* stagedBlockBytes =
				    stagedAligned + stagedOffset +
				    (static_cast<std::uint64_t>(row) * blockCount + blockIndex) * kGGMLQ8KActivationBlockBytes;
				GGMLQ8KActivationBlock stagedBlock;
				QuantizeGGMLQ8KActivationBlock(lhsBlock, lhsColumnStride, stagedBlock);
				std::memcpy(stagedBlockBytes, &stagedBlock, sizeof(stagedBlock));
			}
		}
	}

	float DotGGMLQ4KBlockF32(const std::uint8_t* block, std::int64_t byteStride, const float* lhs,
	                         std::int64_t lhsStride, const float* lhsSubblockSums)
	{
		const auto d = ReadGGMLF16Strided(block, byteStride, 0);
		const auto dmin = ReadGGMLF16Strided(block, byteStride, 2);
		float acc = 0.0F;
		for (std::uint64_t subblock = 0; subblock < 8; ++subblock)
		{
			std::uint32_t scale = 0;
			std::uint32_t minimum = 0;
			GGMLQ4Or5KScaleMin(block, byteStride, subblock, scale, minimum);
			const auto quantPairOffset = (subblock / 2) * 32;
			const auto useHighNibble = (subblock % 2) != 0;
			float quantSum = 0.0F;
			float lhsSum = lhsSubblockSums ? lhsSubblockSums[subblock] : 0.0F;
			if (lhsSubblockSums)
			{
				for (std::uint64_t laneInSubblock = 0; laneInSubblock < 32; laneInSubblock += 4)
				{
					for (std::uint64_t local = 0; local < 4; ++local)
					{
						const auto localLane = laneInSubblock + local;
						const auto lane = subblock * 32 + localLane;
						const auto quantByte = static_cast<std::uint32_t>(
						    GGMLBlockByte(block, byteStride, 16 + quantPairOffset + localLane));
						const auto quant = useHighNibble ? ((quantByte >> 4U) & 15U) : (quantByte & 15U);
						const auto lhsValue = lhs[static_cast<std::int64_t>(lane) * lhsStride];
						quantSum += lhsValue * static_cast<float>(quant);
					}
				}
			}
			else
			{
				for (std::uint64_t laneInSubblock = 0; laneInSubblock < 32; laneInSubblock += 4)
				{
					for (std::uint64_t local = 0; local < 4; ++local)
					{
						const auto localLane = laneInSubblock + local;
						const auto lane = subblock * 32 + localLane;
						const auto quantByte = static_cast<std::uint32_t>(
						    GGMLBlockByte(block, byteStride, 16 + quantPairOffset + localLane));
						const auto quant = useHighNibble ? ((quantByte >> 4U) & 15U) : (quantByte & 15U);
						const auto lhsValue = lhs[static_cast<std::int64_t>(lane) * lhsStride];
						quantSum += lhsValue * static_cast<float>(quant);
						lhsSum += lhsValue;
					}
				}
			}
			acc += d * static_cast<float>(scale) * quantSum - dmin * static_cast<float>(minimum) * lhsSum;
		}
		return acc;
	}

	void AccumulateGGMLQ4KBlockF32x4(const std::uint8_t* const blocks[4], const bool valid[4], std::int64_t byteStride,
	                                 const float* lhs, std::int64_t lhsStride, const float* lhsSubblockSums,
	                                 float acc[4])
	{
		const bool allValid = valid[0] && valid[1] && valid[2] && valid[3];
		float d[4] = {};
		float dmin[4] = {};
		for (int column = 0; column < 4; ++column)
		{
			if (!allValid && !valid[column])
			{
				continue;
			}
			d[column] = ReadGGMLF16Strided(blocks[column], byteStride, 0);
			dmin[column] = ReadGGMLF16Strided(blocks[column], byteStride, 2);
		}

		for (std::uint64_t subblock = 0; subblock < 8; ++subblock)
		{
			std::uint32_t scale[4] = {};
			std::uint32_t minimum[4] = {};
			for (int column = 0; column < 4; ++column)
			{
				if (allValid || valid[column])
				{
					GGMLQ4Or5KScaleMin(blocks[column], byteStride, subblock, scale[column], minimum[column]);
				}
			}

			const auto quantPairOffset = (subblock / 2) * 32;
			const auto useHighNibble = (subblock % 2) != 0;
			float quantSum[4] = {};
			float lhsSum = lhsSubblockSums ? lhsSubblockSums[subblock] : 0.0F;
			for (std::uint64_t laneInSubblock = 0; laneInSubblock < 32; laneInSubblock += 4)
			{
				for (std::uint64_t local = 0; local < 4; ++local)
				{
					const auto localLane = laneInSubblock + local;
					const auto lane = subblock * 32 + localLane;
					const auto lhsValue = lhs[static_cast<std::int64_t>(lane) * lhsStride];
					if (!lhsSubblockSums)
					{
						lhsSum += lhsValue;
					}

					if (allValid)
					{
						for (int column = 0; column < 4; ++column)
						{
							const auto quantByte = static_cast<std::uint32_t>(
							    GGMLBlockByte(blocks[column], byteStride, 16 + quantPairOffset + localLane));
							const auto quant = useHighNibble ? ((quantByte >> 4U) & 15U) : (quantByte & 15U);
							quantSum[column] += lhsValue * static_cast<float>(quant);
						}
					}
					else
					{
						for (int column = 0; column < 4; ++column)
						{
							if (!valid[column])
							{
								continue;
							}
							const auto quantByte = static_cast<std::uint32_t>(
							    GGMLBlockByte(blocks[column], byteStride, 16 + quantPairOffset + localLane));
							const auto quant = useHighNibble ? ((quantByte >> 4U) & 15U) : (quantByte & 15U);
							quantSum[column] += lhsValue * static_cast<float>(quant);
						}
					}
				}
			}

			for (int column = 0; column < 4; ++column)
			{
				if (allValid || valid[column])
				{
					acc[column] += d[column] * static_cast<float>(scale[column]) * quantSum[column] -
					               dmin[column] * static_cast<float>(minimum[column]) * lhsSum;
				}
			}
		}
	}

	void AccumulateGGMLQ4KBlockQ8Kx4(const std::uint8_t* const blocks[4], const bool valid[4], std::int64_t byteStride,
	                                 const GGMLQ8KActivationBlock& lhs, float acc[4])
	{
		float d[4] = {};
		float dmin[4] = {};
		for (int column = 0; column < 4; ++column)
		{
			if (!valid[column])
			{
				continue;
			}
			d[column] = ReadGGMLF16Strided(blocks[column], byteStride, 0);
			dmin[column] = ReadGGMLF16Strided(blocks[column], byteStride, 2);
		}

		for (std::uint64_t subblock = 0; subblock < 8; ++subblock)
		{
			std::uint32_t scale[4] = {};
			std::uint32_t minimum[4] = {};
			for (int column = 0; column < 4; ++column)
			{
				if (valid[column])
				{
					GGMLQ4Or5KScaleMin(blocks[column], byteStride, subblock, scale[column], minimum[column]);
				}
			}

			const auto quantPairOffset = (subblock / 2) * 32;
			const auto useHighNibble = (subblock % 2) != 0;
			std::int32_t quantSum[4] = {};
			for (std::uint64_t laneInSubblock = 0; laneInSubblock < 32; laneInSubblock += 4)
			{
				for (std::uint64_t local = 0; local < 4; ++local)
				{
					const auto localLane = laneInSubblock + local;
					const auto lane = subblock * 32 + localLane;
					const auto lhsQuant = static_cast<std::int32_t>(lhs.qs[lane]);
					for (int column = 0; column < 4; ++column)
					{
						if (!valid[column])
						{
							continue;
						}
						const auto quantByte = static_cast<std::uint32_t>(
						    GGMLBlockByte(blocks[column], byteStride, 16 + quantPairOffset + localLane));
						const auto quant = useHighNibble ? ((quantByte >> 4U) & 15U) : (quantByte & 15U);
						quantSum[column] += lhsQuant * static_cast<std::int32_t>(quant);
					}
				}
			}

			const auto lhsSum = static_cast<std::int32_t>(lhs.bsums[subblock * 2]) +
			                    static_cast<std::int32_t>(lhs.bsums[subblock * 2 + 1]);
			for (int column = 0; column < 4; ++column)
			{
				if (valid[column])
				{
					const auto combinedScale = lhs.d * d[column];
					acc[column] +=
					    combinedScale * static_cast<float>(scale[column]) * static_cast<float>(quantSum[column]) -
					    lhs.d * dmin[column] * static_cast<float>(minimum[column]) * static_cast<float>(lhsSum);
				}
			}
		}
	}

#if LITENN_HAS_X86_AVX2_TARGET
	LITENN_TARGET_AVX2 void AccumulateGGMLQ4KBlockQ8Kx4AVX2(const std::uint8_t* const blocks[4], const bool valid[4],
	                                                        std::int64_t byteStride, const GGMLQ8KActivationBlock& lhs,
	                                                        float acc[4])
	{
		float d[4] = {};
		float dmin[4] = {};
		for (int column = 0; column < 4; ++column)
		{
			if (!valid[column])
			{
				continue;
			}
			d[column] = ReadGGMLF16Strided(blocks[column], byteStride, 0);
			dmin[column] = ReadGGMLF16Strided(blocks[column], byteStride, 2);
		}

		for (std::uint64_t subblock = 0; subblock < 8; ++subblock)
		{
			std::uint32_t scale[4] = {};
			std::uint32_t minimum[4] = {};
			for (int column = 0; column < 4; ++column)
			{
				if (valid[column])
				{
					GGMLQ4Or5KScaleMin(blocks[column], byteStride, subblock, scale[column], minimum[column]);
				}
			}

			const auto quantPairOffset = (subblock / 2) * 32;
			const auto useHighNibble = (subblock % 2) != 0;
			std::int32_t quantSum[4] = {};
			for (std::uint64_t group = 0; group < 2; ++group)
			{
				std::uint8_t raw[4][16] = {};
				for (std::uint64_t local = 0; local < 16; ++local)
				{
					const auto localLane = group * 16 + local;
					for (int column = 0; column < 4; ++column)
					{
						if (!valid[column])
						{
							continue;
						}
						const auto quantByte = static_cast<std::uint32_t>(
						    GGMLBlockByte(blocks[column], byteStride, 16 + quantPairOffset + localLane));
						raw[column][local] =
						    static_cast<std::uint8_t>(useHighNibble ? ((quantByte >> 4U) & 15U) : (quantByte & 15U));
					}
				}
				const auto q8Offset = subblock * 32 + group * 16;
				for (int column = 0; column < 4; ++column)
				{
					if (valid[column])
					{
						quantSum[column] += DotGGMLQ8KWithU8Raw16AVX2(lhs.qs + q8Offset, raw[column], 0);
					}
				}
			}

			const auto lhsSum = static_cast<std::int32_t>(lhs.bsums[subblock * 2]) +
			                    static_cast<std::int32_t>(lhs.bsums[subblock * 2 + 1]);
			for (int column = 0; column < 4; ++column)
			{
				if (valid[column])
				{
					const auto combinedScale = lhs.d * d[column];
					acc[column] +=
					    combinedScale * static_cast<float>(scale[column]) * static_cast<float>(quantSum[column]) -
					    lhs.d * dmin[column] * static_cast<float>(minimum[column]) * static_cast<float>(lhsSum);
				}
			}
		}
	}

	LITENN_TARGET_AVX2 void AccumulateGGMLQ4KBlockQ8Kx4AVX2AllValid(const std::uint8_t* const blocks[4],
	                                                                std::int64_t byteStride,
	                                                                const GGMLQ8KActivationBlock& lhs, float acc[4])
	{
		const float d[4] = {
			ReadGGMLF16Strided(blocks[0], byteStride, 0),
			ReadGGMLF16Strided(blocks[1], byteStride, 0),
			ReadGGMLF16Strided(blocks[2], byteStride, 0),
			ReadGGMLF16Strided(blocks[3], byteStride, 0),
		};
		const float dmin[4] = {
			ReadGGMLF16Strided(blocks[0], byteStride, 2),
			ReadGGMLF16Strided(blocks[1], byteStride, 2),
			ReadGGMLF16Strided(blocks[2], byteStride, 2),
			ReadGGMLF16Strided(blocks[3], byteStride, 2),
		};

		for (std::uint64_t subblock = 0; subblock < 8; ++subblock)
		{
			std::uint32_t scale[4] = {};
			std::uint32_t minimum[4] = {};
			for (int column = 0; column < 4; ++column)
			{
				GGMLQ4Or5KScaleMin(blocks[column], byteStride, subblock, scale[column], minimum[column]);
			}

			const auto quantPairOffset = (subblock / 2) * 32;
			const auto useHighNibble = (subblock % 2) != 0;
			std::int32_t quantSum[4] = {};
			for (std::uint64_t group = 0; group < 2; ++group)
			{
				std::uint8_t raw[4][16] = {};
				for (std::uint64_t local = 0; local < 16; ++local)
				{
					const auto localLane = group * 16 + local;
					for (int column = 0; column < 4; ++column)
					{
						const auto quantByte = static_cast<std::uint32_t>(
						    GGMLBlockByte(blocks[column], byteStride, 16 + quantPairOffset + localLane));
						raw[column][local] =
						    static_cast<std::uint8_t>(useHighNibble ? ((quantByte >> 4U) & 15U) : (quantByte & 15U));
					}
				}
				const auto q8Offset = subblock * 32 + group * 16;
				for (int column = 0; column < 4; ++column)
				{
					quantSum[column] += DotGGMLQ8KWithU8Raw16AVX2(lhs.qs + q8Offset, raw[column], 0);
				}
			}

			const auto lhsSum = static_cast<std::int32_t>(lhs.bsums[subblock * 2]) +
			                    static_cast<std::int32_t>(lhs.bsums[subblock * 2 + 1]);
			for (int column = 0; column < 4; ++column)
			{
				const auto combinedScale = lhs.d * d[column];
				acc[column] +=
				    combinedScale * static_cast<float>(scale[column]) * static_cast<float>(quantSum[column]) -
				    lhs.d * dmin[column] * static_cast<float>(minimum[column]) * static_cast<float>(lhsSum);
			}
		}
	}

	LITENN_TARGET_AVX2 void AccumulateGGMLQ4KBlockQ8Kx4AVX2ContiguousAllValid(const std::uint8_t* const blocks[4],
	                                                                          const GGMLQ8KActivationBlock& lhs,
	                                                                          float acc[4])
	{
		const float d[4] = {
			ReadGGMLF16Strided(blocks[0], 1, 0),
			ReadGGMLF16Strided(blocks[1], 1, 0),
			ReadGGMLF16Strided(blocks[2], 1, 0),
			ReadGGMLF16Strided(blocks[3], 1, 0),
		};
		const float dmin[4] = {
			ReadGGMLF16Strided(blocks[0], 1, 2),
			ReadGGMLF16Strided(blocks[1], 1, 2),
			ReadGGMLF16Strided(blocks[2], 1, 2),
			ReadGGMLF16Strided(blocks[3], 1, 2),
		};
		const auto nibbleMask = _mm_set1_epi8(0x0f);

		for (std::uint64_t subblock = 0; subblock < 8; ++subblock)
		{
			std::uint32_t scale[4] = {};
			std::uint32_t minimum[4] = {};
			for (int column = 0; column < 4; ++column)
			{
				GGMLQ4Or5KScaleMin(blocks[column], 1, subblock, scale[column], minimum[column]);
			}

			const auto quantPairOffset = (subblock / 2) * 32;
			const auto useHighNibble = (subblock % 2) != 0;
			std::int32_t quantSum[4] = {};
			for (std::uint64_t group = 0; group < 2; ++group)
			{
				const auto q8Offset = subblock * 32 + group * 16;
				const auto quantOffset = 16 + quantPairOffset + group * 16;
				__m128i quantBytes[4];
				for (int column = 0; column < 4; ++column)
				{
					quantBytes[column] =
					    _mm_loadu_si128(reinterpret_cast<const __m128i*>(blocks[column] + quantOffset));
					if (useHighNibble)
					{
						quantBytes[column] = _mm_srli_epi16(quantBytes[column], 4);
					}
					quantBytes[column] = _mm_and_si128(quantBytes[column], nibbleMask);
				}
				const auto sums01 =
				    DotGGMLQ8KWithU8Vector16PairAVX2(lhs.qs + q8Offset, quantBytes[0], quantBytes[1], 0);
				const auto sums23 =
				    DotGGMLQ8KWithU8Vector16PairAVX2(lhs.qs + q8Offset, quantBytes[2], quantBytes[3], 0);
				quantSum[0] += _mm_cvtsi128_si32(sums01);
				quantSum[1] += _mm_extract_epi32(sums01, 1);
				quantSum[2] += _mm_cvtsi128_si32(sums23);
				quantSum[3] += _mm_extract_epi32(sums23, 1);
			}

			const auto lhsSum = static_cast<std::int32_t>(lhs.bsums[subblock * 2]) +
			                    static_cast<std::int32_t>(lhs.bsums[subblock * 2 + 1]);
			for (int column = 0; column < 4; ++column)
			{
				const auto combinedScale = lhs.d * d[column];
				acc[column] +=
				    combinedScale * static_cast<float>(scale[column]) * static_cast<float>(quantSum[column]) -
				    lhs.d * dmin[column] * static_cast<float>(minimum[column]) * static_cast<float>(lhsSum);
			}
		}
	}
#endif

	float DotGGMLQ5KBlockF32(const std::uint8_t* block, std::int64_t byteStride, const float* lhs,
	                         std::int64_t lhsStride, const float* lhsSubblockSums)
	{
		const auto d = ReadGGMLF16Strided(block, byteStride, 0);
		const auto dmin = ReadGGMLF16Strided(block, byteStride, 2);
		float acc = 0.0F;
		for (std::uint64_t subblock = 0; subblock < 8; ++subblock)
		{
			std::uint32_t scale = 0;
			std::uint32_t minimum = 0;
			GGMLQ4Or5KScaleMin(block, byteStride, subblock, scale, minimum);
			const auto quantPairOffset = (subblock / 2) * 32;
			const auto useHighNibble = (subblock % 2) != 0;
			float quantSum = 0.0F;
			float lhsSum = lhsSubblockSums ? lhsSubblockSums[subblock] : 0.0F;
			if (lhsSubblockSums)
			{
				for (std::uint64_t laneInSubblock = 0; laneInSubblock < 32; laneInSubblock += 4)
				{
					for (std::uint64_t local = 0; local < 4; ++local)
					{
						const auto localLane = laneInSubblock + local;
						const auto lane = subblock * 32 + localLane;
						const auto quantByte = static_cast<std::uint32_t>(
						    GGMLBlockByte(block, byteStride, 48 + quantPairOffset + localLane));
						const auto highBits =
						    static_cast<std::uint32_t>(GGMLBlockByte(block, byteStride, 16 + localLane));
						auto quant = useHighNibble ? ((quantByte >> 4U) & 15U) : (quantByte & 15U);
						quant |= ((highBits >> subblock) & 1U) << 4U;
						const auto lhsValue = lhs[static_cast<std::int64_t>(lane) * lhsStride];
						quantSum += lhsValue * static_cast<float>(quant);
					}
				}
			}
			else
			{
				for (std::uint64_t laneInSubblock = 0; laneInSubblock < 32; laneInSubblock += 4)
				{
					for (std::uint64_t local = 0; local < 4; ++local)
					{
						const auto localLane = laneInSubblock + local;
						const auto lane = subblock * 32 + localLane;
						const auto quantByte = static_cast<std::uint32_t>(
						    GGMLBlockByte(block, byteStride, 48 + quantPairOffset + localLane));
						const auto highBits =
						    static_cast<std::uint32_t>(GGMLBlockByte(block, byteStride, 16 + localLane));
						auto quant = useHighNibble ? ((quantByte >> 4U) & 15U) : (quantByte & 15U);
						quant |= ((highBits >> subblock) & 1U) << 4U;
						const auto lhsValue = lhs[static_cast<std::int64_t>(lane) * lhsStride];
						quantSum += lhsValue * static_cast<float>(quant);
						lhsSum += lhsValue;
					}
				}
			}
			acc += d * static_cast<float>(scale) * quantSum - dmin * static_cast<float>(minimum) * lhsSum;
		}
		return acc;
	}

	void AccumulateGGMLQ5KBlockF32x4(const std::uint8_t* const blocks[4], const bool valid[4], std::int64_t byteStride,
	                                 const float* lhs, std::int64_t lhsStride, const float* lhsSubblockSums,
	                                 float acc[4])
	{
		float d[4] = {};
		float dmin[4] = {};
		for (int column = 0; column < 4; ++column)
		{
			if (!valid[column])
			{
				continue;
			}
			d[column] = ReadGGMLF16Strided(blocks[column], byteStride, 0);
			dmin[column] = ReadGGMLF16Strided(blocks[column], byteStride, 2);
		}

		for (std::uint64_t subblock = 0; subblock < 8; ++subblock)
		{
			std::uint32_t scale[4] = {};
			std::uint32_t minimum[4] = {};
			for (int column = 0; column < 4; ++column)
			{
				if (valid[column])
				{
					GGMLQ4Or5KScaleMin(blocks[column], byteStride, subblock, scale[column], minimum[column]);
				}
			}

			const auto quantPairOffset = (subblock / 2) * 32;
			const auto useHighNibble = (subblock % 2) != 0;
			float quantSum[4] = {};
			float lhsSum = lhsSubblockSums ? lhsSubblockSums[subblock] : 0.0F;
			for (std::uint64_t laneInSubblock = 0; laneInSubblock < 32; laneInSubblock += 4)
			{
				for (std::uint64_t local = 0; local < 4; ++local)
				{
					const auto localLane = laneInSubblock + local;
					const auto lane = subblock * 32 + localLane;
					const auto lhsValue = lhs[static_cast<std::int64_t>(lane) * lhsStride];
					if (!lhsSubblockSums)
					{
						lhsSum += lhsValue;
					}

					for (int column = 0; column < 4; ++column)
					{
						if (!valid[column])
						{
							continue;
						}
						const auto quantByte = static_cast<std::uint32_t>(
						    GGMLBlockByte(blocks[column], byteStride, 48 + quantPairOffset + localLane));
						const auto highBits =
						    static_cast<std::uint32_t>(GGMLBlockByte(blocks[column], byteStride, 16 + localLane));
						auto quant = useHighNibble ? ((quantByte >> 4U) & 15U) : (quantByte & 15U);
						quant |= ((highBits >> subblock) & 1U) << 4U;
						quantSum[column] += lhsValue * static_cast<float>(quant);
					}
				}
			}

			for (int column = 0; column < 4; ++column)
			{
				if (valid[column])
				{
					acc[column] += d[column] * static_cast<float>(scale[column]) * quantSum[column] -
					               dmin[column] * static_cast<float>(minimum[column]) * lhsSum;
				}
			}
		}
	}

	void AccumulateGGMLQ5KBlockQ8Kx4(const std::uint8_t* const blocks[4], const bool valid[4], std::int64_t byteStride,
	                                 const GGMLQ8KActivationBlock& lhs, float acc[4])
	{
		float d[4] = {};
		float dmin[4] = {};
		for (int column = 0; column < 4; ++column)
		{
			if (!valid[column])
			{
				continue;
			}
			d[column] = ReadGGMLF16Strided(blocks[column], byteStride, 0);
			dmin[column] = ReadGGMLF16Strided(blocks[column], byteStride, 2);
		}

		for (std::uint64_t subblock = 0; subblock < 8; ++subblock)
		{
			std::uint32_t scale[4] = {};
			std::uint32_t minimum[4] = {};
			for (int column = 0; column < 4; ++column)
			{
				if (valid[column])
				{
					GGMLQ4Or5KScaleMin(blocks[column], byteStride, subblock, scale[column], minimum[column]);
				}
			}

			const auto quantPairOffset = (subblock / 2) * 32;
			const auto useHighNibble = (subblock % 2) != 0;
			std::int32_t quantSum[4] = {};
			for (std::uint64_t laneInSubblock = 0; laneInSubblock < 32; laneInSubblock += 4)
			{
				for (std::uint64_t local = 0; local < 4; ++local)
				{
					const auto localLane = laneInSubblock + local;
					const auto lane = subblock * 32 + localLane;
					const auto lhsQuant = static_cast<std::int32_t>(lhs.qs[lane]);
					for (int column = 0; column < 4; ++column)
					{
						if (!valid[column])
						{
							continue;
						}
						const auto quantByte = static_cast<std::uint32_t>(
						    GGMLBlockByte(blocks[column], byteStride, 48 + quantPairOffset + localLane));
						const auto highBits =
						    static_cast<std::uint32_t>(GGMLBlockByte(blocks[column], byteStride, 16 + localLane));
						auto quant = useHighNibble ? ((quantByte >> 4U) & 15U) : (quantByte & 15U);
						quant |= ((highBits >> subblock) & 1U) << 4U;
						quantSum[column] += lhsQuant * static_cast<std::int32_t>(quant);
					}
				}
			}

			const auto lhsSum = static_cast<std::int32_t>(lhs.bsums[subblock * 2]) +
			                    static_cast<std::int32_t>(lhs.bsums[subblock * 2 + 1]);
			for (int column = 0; column < 4; ++column)
			{
				if (valid[column])
				{
					const auto combinedScale = lhs.d * d[column];
					acc[column] +=
					    combinedScale * static_cast<float>(scale[column]) * static_cast<float>(quantSum[column]) -
					    lhs.d * dmin[column] * static_cast<float>(minimum[column]) * static_cast<float>(lhsSum);
				}
			}
		}
	}

#if LITENN_HAS_X86_AVX2_TARGET
	LITENN_TARGET_AVX2 void AccumulateGGMLQ5KBlockQ8Kx4AVX2(const std::uint8_t* const blocks[4], const bool valid[4],
	                                                        std::int64_t byteStride, const GGMLQ8KActivationBlock& lhs,
	                                                        float acc[4])
	{
		float d[4] = {};
		float dmin[4] = {};
		for (int column = 0; column < 4; ++column)
		{
			if (!valid[column])
			{
				continue;
			}
			d[column] = ReadGGMLF16Strided(blocks[column], byteStride, 0);
			dmin[column] = ReadGGMLF16Strided(blocks[column], byteStride, 2);
		}

		for (std::uint64_t subblock = 0; subblock < 8; ++subblock)
		{
			std::uint32_t scale[4] = {};
			std::uint32_t minimum[4] = {};
			for (int column = 0; column < 4; ++column)
			{
				if (valid[column])
				{
					GGMLQ4Or5KScaleMin(blocks[column], byteStride, subblock, scale[column], minimum[column]);
				}
			}

			const auto quantPairOffset = (subblock / 2) * 32;
			const auto useHighNibble = (subblock % 2) != 0;
			std::int32_t quantSum[4] = {};
			for (std::uint64_t group = 0; group < 2; ++group)
			{
				std::uint8_t raw[4][16] = {};
				for (std::uint64_t local = 0; local < 16; ++local)
				{
					const auto localLane = group * 16 + local;
					for (int column = 0; column < 4; ++column)
					{
						if (!valid[column])
						{
							continue;
						}
						const auto quantByte = static_cast<std::uint32_t>(
						    GGMLBlockByte(blocks[column], byteStride, 48 + quantPairOffset + localLane));
						const auto highBits =
						    static_cast<std::uint32_t>(GGMLBlockByte(blocks[column], byteStride, 16 + localLane));
						auto quant = useHighNibble ? ((quantByte >> 4U) & 15U) : (quantByte & 15U);
						quant |= ((highBits >> subblock) & 1U) << 4U;
						raw[column][local] = static_cast<std::uint8_t>(quant);
					}
				}
				const auto q8Offset = subblock * 32 + group * 16;
				for (int column = 0; column < 4; ++column)
				{
					if (valid[column])
					{
						quantSum[column] += DotGGMLQ8KWithU8Raw16AVX2(lhs.qs + q8Offset, raw[column], 0);
					}
				}
			}

			const auto lhsSum = static_cast<std::int32_t>(lhs.bsums[subblock * 2]) +
			                    static_cast<std::int32_t>(lhs.bsums[subblock * 2 + 1]);
			for (int column = 0; column < 4; ++column)
			{
				if (valid[column])
				{
					const auto combinedScale = lhs.d * d[column];
					acc[column] +=
					    combinedScale * static_cast<float>(scale[column]) * static_cast<float>(quantSum[column]) -
					    lhs.d * dmin[column] * static_cast<float>(minimum[column]) * static_cast<float>(lhsSum);
				}
			}
		}
	}

	LITENN_TARGET_AVX2 void AccumulateGGMLQ5KBlockQ8Kx4AVX2AllValid(const std::uint8_t* const blocks[4],
	                                                                std::int64_t byteStride,
	                                                                const GGMLQ8KActivationBlock& lhs, float acc[4])
	{
		const float d[4] = {
			ReadGGMLF16Strided(blocks[0], byteStride, 0),
			ReadGGMLF16Strided(blocks[1], byteStride, 0),
			ReadGGMLF16Strided(blocks[2], byteStride, 0),
			ReadGGMLF16Strided(blocks[3], byteStride, 0),
		};
		const float dmin[4] = {
			ReadGGMLF16Strided(blocks[0], byteStride, 2),
			ReadGGMLF16Strided(blocks[1], byteStride, 2),
			ReadGGMLF16Strided(blocks[2], byteStride, 2),
			ReadGGMLF16Strided(blocks[3], byteStride, 2),
		};

		for (std::uint64_t subblock = 0; subblock < 8; ++subblock)
		{
			std::uint32_t scale[4] = {};
			std::uint32_t minimum[4] = {};
			for (int column = 0; column < 4; ++column)
			{
				GGMLQ4Or5KScaleMin(blocks[column], byteStride, subblock, scale[column], minimum[column]);
			}

			const auto quantPairOffset = (subblock / 2) * 32;
			const auto useHighNibble = (subblock % 2) != 0;
			std::int32_t quantSum[4] = {};
			for (std::uint64_t group = 0; group < 2; ++group)
			{
				std::uint8_t raw[4][16] = {};
				for (std::uint64_t local = 0; local < 16; ++local)
				{
					const auto localLane = group * 16 + local;
					for (int column = 0; column < 4; ++column)
					{
						const auto quantByte = static_cast<std::uint32_t>(
						    GGMLBlockByte(blocks[column], byteStride, 48 + quantPairOffset + localLane));
						const auto highBits =
						    static_cast<std::uint32_t>(GGMLBlockByte(blocks[column], byteStride, 16 + localLane));
						auto quant = useHighNibble ? ((quantByte >> 4U) & 15U) : (quantByte & 15U);
						quant |= ((highBits >> subblock) & 1U) << 4U;
						raw[column][local] = static_cast<std::uint8_t>(quant);
					}
				}
				const auto q8Offset = subblock * 32 + group * 16;
				for (int column = 0; column < 4; ++column)
				{
					quantSum[column] += DotGGMLQ8KWithU8Raw16AVX2(lhs.qs + q8Offset, raw[column], 0);
				}
			}

			const auto lhsSum = static_cast<std::int32_t>(lhs.bsums[subblock * 2]) +
			                    static_cast<std::int32_t>(lhs.bsums[subblock * 2 + 1]);
			for (int column = 0; column < 4; ++column)
			{
				const auto combinedScale = lhs.d * d[column];
				acc[column] +=
				    combinedScale * static_cast<float>(scale[column]) * static_cast<float>(quantSum[column]) -
				    lhs.d * dmin[column] * static_cast<float>(minimum[column]) * static_cast<float>(lhsSum);
			}
		}
	}
#endif

	float DotGGMLQ6KBlockF32(const std::uint8_t* block, std::int64_t byteStride, const float* lhs,
	                         std::int64_t lhsStride)
	{
		const auto d = ReadGGMLF16Strided(block, byteStride, 208);
		float acc = 0.0F;
		for (std::uint64_t halfBlock = 0; halfBlock < 2; ++halfBlock)
		{
			for (std::uint64_t segment = 0; segment < 4; ++segment)
			{
				for (std::uint64_t group = 0; group < 2; ++group)
				{
					const auto scaleOffset = halfBlock * 8 + group + segment * 2;
					const auto scale = static_cast<float>(
					    static_cast<std::int8_t>(GGMLBlockByte(block, byteStride, 192 + scaleOffset)));
					float quantSum = 0.0F;
					for (std::uint64_t local = 0; local < 16; local += 4)
					{
						for (std::uint64_t i = 0; i < 4; ++i)
						{
							const auto laneInSegment = group * 16 + local + i;
							const auto lane = halfBlock * 128 + segment * 32 + laneInSegment;
							const auto qlOffset = halfBlock * 64 + laneInSegment + (segment % 2) * 32;
							const auto ql = static_cast<std::uint32_t>(GGMLBlockByte(block, byteStride, qlOffset));
							const auto lowFour = segment >= 2 ? ((ql >> 4U) & 15U) : (ql & 15U);
							const auto qhOffset = halfBlock * 32 + laneInSegment;
							const auto qh =
							    static_cast<std::uint32_t>(GGMLBlockByte(block, byteStride, 128 + qhOffset));
							const auto highTwo = (qh >> (segment * 2)) & 3U;
							const auto quant = static_cast<std::int32_t>(lowFour | (highTwo << 4U)) - 32;
							quantSum += lhs[static_cast<std::int64_t>(lane) * lhsStride] * static_cast<float>(quant);
						}
					}
					acc += d * scale * quantSum;
				}
			}
		}
		return acc;
	}

	void AccumulateGGMLQ6KBlockF32x4(const std::uint8_t* const blocks[4], const bool valid[4], std::int64_t byteStride,
	                                 const float* lhs, std::int64_t lhsStride, float acc[4])
	{
		const bool allValid = valid[0] && valid[1] && valid[2] && valid[3];
		float d[4] = {};
		for (int column = 0; column < 4; ++column)
		{
			if (allValid || valid[column])
			{
				d[column] = ReadGGMLF16Strided(blocks[column], byteStride, 208);
			}
		}

		for (std::uint64_t halfBlock = 0; halfBlock < 2; ++halfBlock)
		{
			for (std::uint64_t segment = 0; segment < 4; ++segment)
			{
				for (std::uint64_t group = 0; group < 2; ++group)
				{
					const auto scaleOffset = halfBlock * 8 + group + segment * 2;
					float scale[4] = {};
					for (int column = 0; column < 4; ++column)
					{
						if (allValid || valid[column])
						{
							scale[column] = static_cast<float>(
							    static_cast<std::int8_t>(GGMLBlockByte(blocks[column], byteStride, 192 + scaleOffset)));
						}
					}

					float quantSum[4] = {};
					for (std::uint64_t local = 0; local < 16; local += 4)
					{
						for (std::uint64_t i = 0; i < 4; ++i)
						{
							const auto laneInSegment = group * 16 + local + i;
							const auto lane = halfBlock * 128 + segment * 32 + laneInSegment;
							const auto qlOffset = halfBlock * 64 + laneInSegment + (segment % 2) * 32;
							const auto qhOffset = halfBlock * 32 + laneInSegment;
							const auto lhsValue = lhs[static_cast<std::int64_t>(lane) * lhsStride];
							if (allValid)
							{
								for (int column = 0; column < 4; ++column)
								{
									const auto ql =
									    static_cast<std::uint32_t>(GGMLBlockByte(blocks[column], byteStride, qlOffset));
									const auto lowFour = segment >= 2 ? ((ql >> 4U) & 15U) : (ql & 15U);
									const auto qh = static_cast<std::uint32_t>(
									    GGMLBlockByte(blocks[column], byteStride, 128 + qhOffset));
									const auto highTwo = (qh >> (segment * 2)) & 3U;
									const auto quant = static_cast<std::int32_t>(lowFour | (highTwo << 4U)) - 32;
									quantSum[column] += lhsValue * static_cast<float>(quant);
								}
							}
							else
							{
								for (int column = 0; column < 4; ++column)
								{
									if (!valid[column])
									{
										continue;
									}
									const auto ql =
									    static_cast<std::uint32_t>(GGMLBlockByte(blocks[column], byteStride, qlOffset));
									const auto lowFour = segment >= 2 ? ((ql >> 4U) & 15U) : (ql & 15U);
									const auto qh = static_cast<std::uint32_t>(
									    GGMLBlockByte(blocks[column], byteStride, 128 + qhOffset));
									const auto highTwo = (qh >> (segment * 2)) & 3U;
									const auto quant = static_cast<std::int32_t>(lowFour | (highTwo << 4U)) - 32;
									quantSum[column] += lhsValue * static_cast<float>(quant);
								}
							}
						}
					}

					for (int column = 0; column < 4; ++column)
					{
						if (allValid || valid[column])
						{
							acc[column] += d[column] * scale[column] * quantSum[column];
						}
					}
				}
			}
		}
	}

	constexpr std::uint64_t kGGMLQ6KPreparedScaleCount = 16;
	constexpr std::uint64_t kGGMLQ6KPreparedLanes = 256;
	constexpr std::uint64_t kGGMLQ6KPreparedScaleBytes = kGGMLQ6KPreparedScaleCount * sizeof(float);
	constexpr std::uint64_t kGGMLQ6KPreparedQuantOffset = kGGMLQ6KPreparedScaleBytes;
	constexpr std::uint64_t kGGMLQ6KPreparedBlockBytes = kGGMLQ6KPreparedQuantOffset + kGGMLQ6KPreparedLanes;
	constexpr std::uint64_t kGGMLQ4KPreparedSubblockCount = 8;
	constexpr std::uint64_t kGGMLQ4KPreparedLanes = 256;
	constexpr std::uint64_t kGGMLQ4KPreparedScaleBytes = kGGMLQ4KPreparedSubblockCount * sizeof(float);
	constexpr std::uint64_t kGGMLQ4KPreparedMinOffset = kGGMLQ4KPreparedScaleBytes;
	constexpr std::uint64_t kGGMLQ4KPreparedQuantOffset = kGGMLQ4KPreparedMinOffset + kGGMLQ4KPreparedScaleBytes;
	constexpr std::uint64_t kGGMLQ4KPreparedBlockBytes = kGGMLQ4KPreparedQuantOffset + kGGMLQ4KPreparedLanes;

	void StoreGGMLQ6KPreparedScale(std::uint8_t* block, std::uint64_t group, float value)
	{
		std::memcpy(block + group * sizeof(float), &value, sizeof(float));
	}

	float LoadGGMLQ6KPreparedScale(const std::uint8_t* block, std::uint64_t group)
	{
		float value = 0.0F;
		std::memcpy(&value, block + group * sizeof(float), sizeof(float));
		return value;
	}

	std::int8_t LoadGGMLQ6KPreparedQuant(const std::uint8_t* block, std::uint64_t lane)
	{
		return static_cast<std::int8_t>(block[kGGMLQ6KPreparedQuantOffset + lane]);
	}

	void StoreGGMLQ4KPreparedScale(std::uint8_t* block, std::uint64_t subblock, float scale, float minimum)
	{
		std::memcpy(block + subblock * sizeof(float), &scale, sizeof(float));
		std::memcpy(block + kGGMLQ4KPreparedMinOffset + subblock * sizeof(float), &minimum, sizeof(float));
	}

	float LoadGGMLQ4KPreparedScale(const std::uint8_t* block, std::uint64_t subblock)
	{
		float value = 0.0F;
		std::memcpy(&value, block + subblock * sizeof(float), sizeof(float));
		return value;
	}

	float LoadGGMLQ4KPreparedMin(const std::uint8_t* block, std::uint64_t subblock)
	{
		float value = 0.0F;
		std::memcpy(&value, block + kGGMLQ4KPreparedMinOffset + subblock * sizeof(float), sizeof(float));
		return value;
	}

	std::uint8_t LoadGGMLQ4KPreparedQuant(const std::uint8_t* block, std::uint64_t lane)
	{
		return block[kGGMLQ4KPreparedQuantOffset + lane];
	}

	void PrepareGGMLQ4KBlockF32(const std::uint8_t* block, std::int64_t byteStride, std::uint8_t* out)
	{
		const auto d = ReadGGMLF16Strided(block, byteStride, 0);
		const auto dmin = ReadGGMLF16Strided(block, byteStride, 2);
		for (std::uint64_t subblock = 0; subblock < kGGMLQ4KPreparedSubblockCount; ++subblock)
		{
			std::uint32_t scale = 0;
			std::uint32_t minimum = 0;
			GGMLQ4Or5KScaleMin(block, byteStride, subblock, scale, minimum);
			StoreGGMLQ4KPreparedScale(out, subblock, d * static_cast<float>(scale), dmin * static_cast<float>(minimum));
			const auto quantPairOffset = (subblock / 2) * 32;
			const auto useHighNibble = (subblock % 2) != 0;
			for (std::uint64_t laneInSubblock = 0; laneInSubblock < 32; ++laneInSubblock)
			{
				const auto quantByte =
				    static_cast<std::uint32_t>(GGMLBlockByte(block, byteStride, 16 + quantPairOffset + laneInSubblock));
				const auto quant = useHighNibble ? ((quantByte >> 4U) & 15U) : (quantByte & 15U);
				out[kGGMLQ4KPreparedQuantOffset + subblock * 32 + laneInSubblock] = static_cast<std::uint8_t>(quant);
			}
		}
	}

	void AccumulateGGMLQ4KPreparedBlockF32x4(const std::uint8_t* const blocks[4], const bool valid[4], const float* lhs,
	                                         std::int64_t lhsStride, const float* lhsSubblockSums, float acc[4])
	{
		const bool allValid = valid[0] && valid[1] && valid[2] && valid[3];
		for (std::uint64_t subblock = 0; subblock < kGGMLQ4KPreparedSubblockCount; ++subblock)
		{
			float quantSum[4] = {};
			float lhsSum = lhsSubblockSums ? lhsSubblockSums[subblock] : 0.0F;
			for (std::uint64_t laneInSubblock = 0; laneInSubblock < 32; ++laneInSubblock)
			{
				const auto lane = subblock * 32 + laneInSubblock;
				const auto lhsValue = lhs[static_cast<std::int64_t>(lane) * lhsStride];
				if (!lhsSubblockSums)
				{
					lhsSum += lhsValue;
				}
				if (allValid)
				{
					for (int column = 0; column < 4; ++column)
					{
						quantSum[column] +=
						    lhsValue * static_cast<float>(LoadGGMLQ4KPreparedQuant(blocks[column], lane));
					}
				}
				else
				{
					for (int column = 0; column < 4; ++column)
					{
						if (!valid[column])
						{
							continue;
						}
						quantSum[column] +=
						    lhsValue * static_cast<float>(LoadGGMLQ4KPreparedQuant(blocks[column], lane));
					}
				}
			}
			for (int column = 0; column < 4; ++column)
			{
				if (allValid || valid[column])
				{
					acc[column] += LoadGGMLQ4KPreparedScale(blocks[column], subblock) * quantSum[column] -
					               LoadGGMLQ4KPreparedMin(blocks[column], subblock) * lhsSum;
				}
			}
		}
	}

	void AccumulateGGMLQ4KPreparedBlockF32x4AllValid(const std::uint8_t* const blocks[4], const float* lhs,
	                                                 std::int64_t lhsStride, const float* lhsSubblockSums, float acc[4])
	{
		for (std::uint64_t subblock = 0; subblock < kGGMLQ4KPreparedSubblockCount; ++subblock)
		{
			float quantSum[4] = {};
			float lhsSum = lhsSubblockSums ? lhsSubblockSums[subblock] : 0.0F;
			for (std::uint64_t laneInSubblock = 0; laneInSubblock < 32; ++laneInSubblock)
			{
				const auto lane = subblock * 32 + laneInSubblock;
				const auto lhsValue = lhs[static_cast<std::int64_t>(lane) * lhsStride];
				if (!lhsSubblockSums)
				{
					lhsSum += lhsValue;
				}
				for (int column = 0; column < 4; ++column)
				{
					quantSum[column] += lhsValue * static_cast<float>(LoadGGMLQ4KPreparedQuant(blocks[column], lane));
				}
			}
			for (int column = 0; column < 4; ++column)
			{
				acc[column] += LoadGGMLQ4KPreparedScale(blocks[column], subblock) * quantSum[column] -
				               LoadGGMLQ4KPreparedMin(blocks[column], subblock) * lhsSum;
			}
		}
	}

#if LITENN_HAS_X86_AVX2_TARGET
	LITENN_TARGET_AVX2 float HorizontalSumF32AVX2(__m256 values)
	{
		const auto halves = _mm_add_ps(_mm256_castps256_ps128(values), _mm256_extractf128_ps(values, 1));
		const auto pairs = _mm_hadd_ps(halves, halves);
		return _mm_cvtss_f32(_mm_hadd_ps(pairs, pairs));
	}

	LITENN_TARGET_AVX2 void AccumulateGGMLQ4KPreparedBlockF32x4AllValidAVX2(const std::uint8_t* const blocks[4],
	                                                                        const float* lhs,
	                                                                        const float* lhsSubblockSums, float acc[4])
	{
		for (std::uint64_t subblock = 0; subblock < kGGMLQ4KPreparedSubblockCount; ++subblock)
		{
			__m256 quantSum[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
			auto lhsSumVector = _mm256_setzero_ps();
			for (std::uint64_t laneInSubblock = 0; laneInSubblock < 32; laneInSubblock += 8)
			{
				const auto lane = subblock * 32 + laneInSubblock;
				const auto lhsValues = _mm256_loadu_ps(lhs + lane);
				if (!lhsSubblockSums)
				{
					lhsSumVector = _mm256_add_ps(lhsSumVector, lhsValues);
				}
				for (int column = 0; column < 4; ++column)
				{
					const auto quantBytes = _mm_loadl_epi64(
					    reinterpret_cast<const __m128i*>(blocks[column] + kGGMLQ4KPreparedQuantOffset + lane));
					const auto quant32 = _mm256_cvtepu8_epi32(quantBytes);
					const auto quantF32 = _mm256_cvtepi32_ps(quant32);
					quantSum[column] = _mm256_add_ps(quantSum[column], _mm256_mul_ps(lhsValues, quantF32));
				}
			}
			const auto lhsSum = lhsSubblockSums ? lhsSubblockSums[subblock] : HorizontalSumF32AVX2(lhsSumVector);
			for (int column = 0; column < 4; ++column)
			{
				acc[column] +=
				    LoadGGMLQ4KPreparedScale(blocks[column], subblock) * HorizontalSumF32AVX2(quantSum[column]) -
				    LoadGGMLQ4KPreparedMin(blocks[column], subblock) * lhsSum;
			}
		}
	}
#endif

	void PrepareGGMLQ6KBlockF32(const std::uint8_t* block, std::int64_t byteStride, std::uint8_t* out)
	{
		const auto d = ReadGGMLF16Strided(block, byteStride, 208);
		for (std::uint64_t halfBlock = 0; halfBlock < 2; ++halfBlock)
		{
			for (std::uint64_t segment = 0; segment < 4; ++segment)
			{
				for (std::uint64_t group = 0; group < 2; ++group)
				{
					const auto groupIndex = halfBlock * 8 + segment * 2 + group;
					const auto scaleOffset = halfBlock * 8 + group + segment * 2;
					const auto scale = d * static_cast<float>(static_cast<std::int8_t>(
					                           GGMLBlockByte(block, byteStride, 192 + scaleOffset)));
					StoreGGMLQ6KPreparedScale(out, groupIndex, scale);
					for (std::uint64_t local = 0; local < 16; ++local)
					{
						const auto laneInSegment = group * 16 + local;
						const auto lane = groupIndex * 16 + local;
						const auto qlOffset = halfBlock * 64 + laneInSegment + (segment % 2) * 32;
						const auto ql = static_cast<std::uint32_t>(GGMLBlockByte(block, byteStride, qlOffset));
						const auto lowFour = segment >= 2 ? ((ql >> 4U) & 15U) : (ql & 15U);
						const auto qhOffset = halfBlock * 32 + laneInSegment;
						const auto qh = static_cast<std::uint32_t>(GGMLBlockByte(block, byteStride, 128 + qhOffset));
						const auto highTwo = (qh >> (segment * 2)) & 3U;
						const auto quant = static_cast<std::int32_t>(lowFour | (highTwo << 4U)) - 32;
						out[kGGMLQ6KPreparedQuantOffset + lane] =
						    static_cast<std::uint8_t>(static_cast<std::int8_t>(quant));
					}
				}
			}
		}
	}

	void AccumulateGGMLQ6KPreparedBlockF32x4(const std::uint8_t* const blocks[4], const bool valid[4], const float* lhs,
	                                         std::int64_t lhsStride, float acc[4])
	{
		const bool allValid = valid[0] && valid[1] && valid[2] && valid[3];
		for (std::uint64_t group = 0; group < kGGMLQ6KPreparedScaleCount; ++group)
		{
			float quantSum[4] = {};
			for (std::uint64_t local = 0; local < 16; ++local)
			{
				const auto lane = group * 16 + local;
				const auto lhsValue = lhs[static_cast<std::int64_t>(lane) * lhsStride];
				if (allValid)
				{
					for (int column = 0; column < 4; ++column)
					{
						quantSum[column] +=
						    lhsValue * static_cast<float>(LoadGGMLQ6KPreparedQuant(blocks[column], lane));
					}
				}
				else
				{
					for (int column = 0; column < 4; ++column)
					{
						if (!valid[column])
						{
							continue;
						}
						quantSum[column] +=
						    lhsValue * static_cast<float>(LoadGGMLQ6KPreparedQuant(blocks[column], lane));
					}
				}
			}

			for (int column = 0; column < 4; ++column)
			{
				if (allValid || valid[column])
				{
					acc[column] += LoadGGMLQ6KPreparedScale(blocks[column], group) * quantSum[column];
				}
			}
		}
	}

	void AccumulateGGMLQ6KPreparedBlockF32x4AllValid(const std::uint8_t* const blocks[4], const float* lhs,
	                                                 std::int64_t lhsStride, float acc[4])
	{
		for (std::uint64_t group = 0; group < kGGMLQ6KPreparedScaleCount; ++group)
		{
			float quantSum[4] = {};
			for (std::uint64_t local = 0; local < 16; ++local)
			{
				const auto lane = group * 16 + local;
				const auto lhsValue = lhs[static_cast<std::int64_t>(lane) * lhsStride];
				for (int column = 0; column < 4; ++column)
				{
					quantSum[column] += lhsValue * static_cast<float>(LoadGGMLQ6KPreparedQuant(blocks[column], lane));
				}
			}

			for (int column = 0; column < 4; ++column)
			{
				acc[column] += LoadGGMLQ6KPreparedScale(blocks[column], group) * quantSum[column];
			}
		}
	}

#if LITENN_HAS_X86_AVX2_TARGET
	LITENN_TARGET_AVX2 void AccumulateGGMLQ6KPreparedBlockF32x4AllValidAVX2(const std::uint8_t* const blocks[4],
	                                                                        const float* lhs, float acc[4])
	{
		for (std::uint64_t group = 0; group < kGGMLQ6KPreparedScaleCount; ++group)
		{
			__m256 quantSum[4] = { _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps(), _mm256_setzero_ps() };
			for (std::uint64_t local = 0; local < 16; local += 8)
			{
				const auto lane = group * 16 + local;
				const auto lhsValues = _mm256_loadu_ps(lhs + lane);
				for (int column = 0; column < 4; ++column)
				{
					const auto quantBytes = _mm_loadl_epi64(
					    reinterpret_cast<const __m128i*>(blocks[column] + kGGMLQ6KPreparedQuantOffset + lane));
					const auto quant32 = _mm256_cvtepi8_epi32(quantBytes);
					const auto quantF32 = _mm256_cvtepi32_ps(quant32);
					quantSum[column] = _mm256_add_ps(quantSum[column], _mm256_mul_ps(lhsValues, quantF32));
				}
			}

			for (int column = 0; column < 4; ++column)
			{
				acc[column] += LoadGGMLQ6KPreparedScale(blocks[column], group) * HorizontalSumF32AVX2(quantSum[column]);
			}
		}
	}
#endif

	void AccumulateGGMLQ6KBlockQ8Kx4(const std::uint8_t* const blocks[4], const bool valid[4], std::int64_t byteStride,
	                                 const GGMLQ8KActivationBlock& lhs, float acc[4])
	{
		float d[4] = {};
		for (int column = 0; column < 4; ++column)
		{
			if (valid[column])
			{
				d[column] = ReadGGMLF16Strided(blocks[column], byteStride, 208);
			}
		}

		for (std::uint64_t halfBlock = 0; halfBlock < 2; ++halfBlock)
		{
			for (std::uint64_t segment = 0; segment < 4; ++segment)
			{
				for (std::uint64_t group = 0; group < 2; ++group)
				{
					const auto scaleOffset = halfBlock * 8 + group + segment * 2;
					float scale[4] = {};
					for (int column = 0; column < 4; ++column)
					{
						if (valid[column])
						{
							scale[column] = static_cast<float>(
							    static_cast<std::int8_t>(GGMLBlockByte(blocks[column], byteStride, 192 + scaleOffset)));
						}
					}

					std::int32_t quantSum[4] = {};
					for (std::uint64_t local = 0; local < 16; local += 4)
					{
						for (std::uint64_t i = 0; i < 4; ++i)
						{
							const auto laneInSegment = group * 16 + local + i;
							const auto lane = halfBlock * 128 + segment * 32 + laneInSegment;
							const auto qlOffset = halfBlock * 64 + laneInSegment + (segment % 2) * 32;
							const auto qhOffset = halfBlock * 32 + laneInSegment;
							const auto lhsQuant = static_cast<std::int32_t>(lhs.qs[lane]);
							for (int column = 0; column < 4; ++column)
							{
								if (!valid[column])
								{
									continue;
								}
								const auto ql =
								    static_cast<std::uint32_t>(GGMLBlockByte(blocks[column], byteStride, qlOffset));
								const auto lowFour = segment >= 2 ? ((ql >> 4U) & 15U) : (ql & 15U);
								const auto qh = static_cast<std::uint32_t>(
								    GGMLBlockByte(blocks[column], byteStride, 128 + qhOffset));
								const auto highTwo = (qh >> (segment * 2)) & 3U;
								const auto quant = static_cast<std::int32_t>(lowFour | (highTwo << 4U)) - 32;
								quantSum[column] += lhsQuant * quant;
							}
						}
					}

					for (int column = 0; column < 4; ++column)
					{
						if (valid[column])
						{
							acc[column] += lhs.d * d[column] * scale[column] * static_cast<float>(quantSum[column]);
						}
					}
				}
			}
		}
	}

#if LITENN_HAS_X86_AVX2_TARGET
	LITENN_TARGET_AVX2 void AccumulateGGMLQ6KBlockQ8Kx4AVX2(const std::uint8_t* const blocks[4], const bool valid[4],
	                                                        std::int64_t byteStride, const GGMLQ8KActivationBlock& lhs,
	                                                        float acc[4])
	{
		float d[4] = {};
		for (int column = 0; column < 4; ++column)
		{
			if (valid[column])
			{
				d[column] = ReadGGMLF16Strided(blocks[column], byteStride, 208);
			}
		}

		for (std::uint64_t halfBlock = 0; halfBlock < 2; ++halfBlock)
		{
			for (std::uint64_t segment = 0; segment < 4; ++segment)
			{
				for (std::uint64_t group = 0; group < 2; ++group)
				{
					const auto scaleOffset = halfBlock * 8 + group + segment * 2;
					float scale[4] = {};
					for (int column = 0; column < 4; ++column)
					{
						if (valid[column])
						{
							scale[column] = static_cast<float>(
							    static_cast<std::int8_t>(GGMLBlockByte(blocks[column], byteStride, 192 + scaleOffset)));
						}
					}

					std::int32_t quantSum[4] = {};
					std::uint8_t raw[4][16] = {};
					for (std::uint64_t local = 0; local < 16; ++local)
					{
						const auto laneInSegment = group * 16 + local;
						const auto qlOffset = halfBlock * 64 + laneInSegment + (segment % 2) * 32;
						const auto qhOffset = halfBlock * 32 + laneInSegment;
						for (int column = 0; column < 4; ++column)
						{
							if (!valid[column])
							{
								continue;
							}
							const auto ql =
							    static_cast<std::uint32_t>(GGMLBlockByte(blocks[column], byteStride, qlOffset));
							const auto lowFour = segment >= 2 ? ((ql >> 4U) & 15U) : (ql & 15U);
							const auto qh =
							    static_cast<std::uint32_t>(GGMLBlockByte(blocks[column], byteStride, 128 + qhOffset));
							const auto highTwo = (qh >> (segment * 2)) & 3U;
							raw[column][local] = static_cast<std::uint8_t>(lowFour | (highTwo << 4U));
						}
					}

					const auto q8Offset = halfBlock * 128 + segment * 32 + group * 16;
					for (int column = 0; column < 4; ++column)
					{
						if (valid[column])
						{
							quantSum[column] = DotGGMLQ8KWithU8Raw16AVX2(lhs.qs + q8Offset, raw[column], 32);
							acc[column] += lhs.d * d[column] * scale[column] * static_cast<float>(quantSum[column]);
						}
					}
				}
			}
		}
	}

	LITENN_TARGET_AVX2 void AccumulateGGMLQ6KBlockQ8Kx4AVX2AllValid(const std::uint8_t* const blocks[4],
	                                                                std::int64_t byteStride,
	                                                                const GGMLQ8KActivationBlock& lhs, float acc[4])
	{
		const float d[4] = {
			ReadGGMLF16Strided(blocks[0], byteStride, 208),
			ReadGGMLF16Strided(blocks[1], byteStride, 208),
			ReadGGMLF16Strided(blocks[2], byteStride, 208),
			ReadGGMLF16Strided(blocks[3], byteStride, 208),
		};

		for (std::uint64_t halfBlock = 0; halfBlock < 2; ++halfBlock)
		{
			for (std::uint64_t segment = 0; segment < 4; ++segment)
			{
				for (std::uint64_t group = 0; group < 2; ++group)
				{
					const auto scaleOffset = halfBlock * 8 + group + segment * 2;
					const float scale[4] = {
						static_cast<float>(
						    static_cast<std::int8_t>(GGMLBlockByte(blocks[0], byteStride, 192 + scaleOffset))),
						static_cast<float>(
						    static_cast<std::int8_t>(GGMLBlockByte(blocks[1], byteStride, 192 + scaleOffset))),
						static_cast<float>(
						    static_cast<std::int8_t>(GGMLBlockByte(blocks[2], byteStride, 192 + scaleOffset))),
						static_cast<float>(
						    static_cast<std::int8_t>(GGMLBlockByte(blocks[3], byteStride, 192 + scaleOffset))),
					};

					std::uint8_t raw[4][16] = {};
					for (std::uint64_t local = 0; local < 16; ++local)
					{
						const auto laneInSegment = group * 16 + local;
						const auto qlOffset = halfBlock * 64 + laneInSegment + (segment % 2) * 32;
						const auto qhOffset = halfBlock * 32 + laneInSegment;
						for (int column = 0; column < 4; ++column)
						{
							const auto ql =
							    static_cast<std::uint32_t>(GGMLBlockByte(blocks[column], byteStride, qlOffset));
							const auto lowFour = segment >= 2 ? ((ql >> 4U) & 15U) : (ql & 15U);
							const auto qh =
							    static_cast<std::uint32_t>(GGMLBlockByte(blocks[column], byteStride, 128 + qhOffset));
							const auto highTwo = (qh >> (segment * 2)) & 3U;
							raw[column][local] = static_cast<std::uint8_t>(lowFour | (highTwo << 4U));
						}
					}

					const auto q8Offset = halfBlock * 128 + segment * 32 + group * 16;
					for (int column = 0; column < 4; ++column)
					{
						const auto quantSum = DotGGMLQ8KWithU8Raw16AVX2(lhs.qs + q8Offset, raw[column], 32);
						acc[column] += lhs.d * d[column] * scale[column] * static_cast<float>(quantSum);
					}
				}
			}
		}
	}

	LITENN_TARGET_AVX2 void AccumulateGGMLQ6KBlockQ8Kx4AVX2ContiguousAllValid(const std::uint8_t* const blocks[4],
	                                                                          const GGMLQ8KActivationBlock& lhs,
	                                                                          float acc[4])
	{
		const float d[4] = {
			ReadGGMLF16Strided(blocks[0], 1, 208),
			ReadGGMLF16Strided(blocks[1], 1, 208),
			ReadGGMLF16Strided(blocks[2], 1, 208),
			ReadGGMLF16Strided(blocks[3], 1, 208),
		};
		const auto lowFourMask = _mm_set1_epi8(0x0f);
		const auto highTwoMask = _mm_set1_epi8(0x03);

		for (std::uint64_t halfBlock = 0; halfBlock < 2; ++halfBlock)
		{
			for (std::uint64_t segment = 0; segment < 4; ++segment)
			{
				for (std::uint64_t group = 0; group < 2; ++group)
				{
					const auto scaleOffset = halfBlock * 8 + group + segment * 2;
					const auto laneInSegment = group * 16;
					const auto qlOffset = halfBlock * 64 + laneInSegment + (segment % 2) * 32;
					const auto qhOffset = 128 + halfBlock * 32 + laneInSegment;
					const auto q8Offset = halfBlock * 128 + segment * 32 + laneInSegment;
					float scale[4];
					__m128i quantBytes[4];
					for (int column = 0; column < 4; ++column)
					{
						scale[column] = static_cast<float>(static_cast<std::int8_t>(blocks[column][192 + scaleOffset]));
						auto lowFour = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blocks[column] + qlOffset));
						if (segment >= 2)
						{
							lowFour = _mm_srli_epi16(lowFour, 4);
						}
						lowFour = _mm_and_si128(lowFour, lowFourMask);
						auto highTwo = _mm_loadu_si128(reinterpret_cast<const __m128i*>(blocks[column] + qhOffset));
						highTwo = _mm_and_si128(_mm_srli_epi16(highTwo, static_cast<int>(segment * 2)), highTwoMask);
						quantBytes[column] = _mm_or_si128(lowFour, _mm_slli_epi16(highTwo, 4));
					}
					const auto sums01 =
					    DotGGMLQ8KWithU8Vector16PairAVX2(lhs.qs + q8Offset, quantBytes[0], quantBytes[1], 32);
					const auto sums23 =
					    DotGGMLQ8KWithU8Vector16PairAVX2(lhs.qs + q8Offset, quantBytes[2], quantBytes[3], 32);
					const std::int32_t quantSum[4] = {
						_mm_cvtsi128_si32(sums01),
						_mm_extract_epi32(sums01, 1),
						_mm_cvtsi128_si32(sums23),
						_mm_extract_epi32(sums23, 1),
					};
					for (int column = 0; column < 4; ++column)
					{
						acc[column] += lhs.d * d[column] * scale[column] * static_cast<float>(quantSum[column]);
					}
				}
			}
		}
	}

#endif

	float DotGGMLBlockF32(const std::uint8_t* block, std::int64_t byteStride, const float* lhs, std::int64_t lhsStride,
	                      QuantizedBlockFormat format, const float* lhsSubblockSums = nullptr)
	{
		switch (format)
		{
		case QuantizedBlockFormat::GGML_Q8_0:
			return DotGGMLQ8_0BlockF32(block, byteStride, lhs, lhsStride);
		case QuantizedBlockFormat::GGML_Q4_K:
			return DotGGMLQ4KBlockF32(block, byteStride, lhs, lhsStride, lhsSubblockSums);
		case QuantizedBlockFormat::GGML_Q5_K:
			return DotGGMLQ5KBlockF32(block, byteStride, lhs, lhsStride, lhsSubblockSums);
		case QuantizedBlockFormat::GGML_Q6_K:
			return DotGGMLQ6KBlockF32(block, byteStride, lhs, lhsStride);
		default:
			return 0.0F;
		}
	}

	void DecodeGGMLQ8_0BlockF32(const std::uint8_t* block, std::int64_t byteStride, float* out, std::int64_t outStride)
	{
		const auto d = ReadGGMLF16Strided(block, byteStride, 0);
		for (std::uint64_t lane = 0; lane < 32; ++lane)
		{
			out[static_cast<std::int64_t>(lane) * outStride] =
			    d * static_cast<float>(static_cast<std::int8_t>(GGMLBlockByte(block, byteStride, 2 + lane)));
		}
	}

	void DecodeGGMLQ4Or5KBlockF32(const std::uint8_t* block, std::int64_t byteStride, float* out,
	                              std::int64_t outStride, QuantizedBlockFormat format)
	{
		const auto d = ReadGGMLF16Strided(block, byteStride, 0);
		const auto dmin = ReadGGMLF16Strided(block, byteStride, 2);
		const auto quantBaseOffset = format == QuantizedBlockFormat::GGML_Q5_K ? 48u : 16u;
		for (std::uint64_t subblock = 0; subblock < 8; ++subblock)
		{
			std::uint32_t scale = 0;
			std::uint32_t minimum = 0;
			GGMLQ4Or5KScaleMin(block, byteStride, subblock, scale, minimum);
			const auto quantPairOffset = (subblock / 2) * 32;
			const auto useHighNibble = (subblock % 2) != 0;
			const auto scaleF32 = d * static_cast<float>(scale);
			const auto minF32 = dmin * static_cast<float>(minimum);
			for (std::uint64_t laneInSubblock = 0; laneInSubblock < 32; ++laneInSubblock)
			{
				const auto lane = subblock * 32 + laneInSubblock;
				const auto quantByte = static_cast<std::uint32_t>(
				    GGMLBlockByte(block, byteStride, quantBaseOffset + quantPairOffset + laneInSubblock));
				auto quant = useHighNibble ? ((quantByte >> 4U) & 15U) : (quantByte & 15U);
				if (format == QuantizedBlockFormat::GGML_Q5_K)
				{
					const auto highBits =
					    static_cast<std::uint32_t>(GGMLBlockByte(block, byteStride, 16 + laneInSubblock));
					quant |= ((highBits >> subblock) & 1U) << 4U;
				}
				out[static_cast<std::int64_t>(lane) * outStride] = scaleF32 * static_cast<float>(quant) - minF32;
			}
		}
	}

	void DecodeGGMLQ6KBlockF32(const std::uint8_t* block, std::int64_t byteStride, float* out, std::int64_t outStride)
	{
		const auto d = ReadGGMLF16Strided(block, byteStride, 208);
		for (std::uint64_t halfBlock = 0; halfBlock < 2; ++halfBlock)
		{
			for (std::uint64_t segment = 0; segment < 4; ++segment)
			{
				for (std::uint64_t group = 0; group < 2; ++group)
				{
					const auto scaleOffset = halfBlock * 8 + group + segment * 2;
					const auto scale = d * static_cast<float>(static_cast<std::int8_t>(
					                           GGMLBlockByte(block, byteStride, 192 + scaleOffset)));
					for (std::uint64_t local = 0; local < 16; ++local)
					{
						const auto laneInSegment = group * 16 + local;
						const auto lane = halfBlock * 128 + segment * 32 + laneInSegment;
						const auto qlOffset = halfBlock * 64 + laneInSegment + (segment % 2) * 32;
						const auto ql = static_cast<std::uint32_t>(GGMLBlockByte(block, byteStride, qlOffset));
						const auto lowFour = segment >= 2 ? ((ql >> 4U) & 15U) : (ql & 15U);
						const auto qhOffset = halfBlock * 32 + laneInSegment;
						const auto qh = static_cast<std::uint32_t>(GGMLBlockByte(block, byteStride, 128 + qhOffset));
						const auto highTwo = (qh >> (segment * 2)) & 3U;
						const auto quant = static_cast<std::int32_t>(lowFour | (highTwo << 4U)) - 32;
						out[static_cast<std::int64_t>(lane) * outStride] = scale * static_cast<float>(quant);
					}
				}
			}
		}
	}

	void DecodeGGMLBlockF32(const std::uint8_t* block, std::int64_t byteStride, float* out, std::int64_t outStride,
	                        QuantizedBlockFormat format)
	{
		switch (format)
		{
		case QuantizedBlockFormat::GGML_Q8_0:
			DecodeGGMLQ8_0BlockF32(block, byteStride, out, outStride);
			break;
		case QuantizedBlockFormat::GGML_Q4_K:
		case QuantizedBlockFormat::GGML_Q5_K:
			DecodeGGMLQ4Or5KBlockF32(block, byteStride, out, outStride, format);
			break;
		case QuantizedBlockFormat::GGML_Q6_K:
			DecodeGGMLQ6KBlockF32(block, byteStride, out, outStride);
			break;
		default:
			break;
		}
	}

	struct GGMLBlockMatMulProjection
	{
		const std::uint8_t* rhsAligned{};
		std::int64_t rhsOffset{};
		std::int64_t rhsBytes{};
		std::int64_t rhsStride{};
		std::int64_t outColumns{};
	};

	constexpr std::uint64_t kGGMLCompactInterleavedMagic = 0x33564C49434E4E4Cull;
	constexpr std::uint32_t kGGMLCompactInterleavedVersion = 3;

	struct GGMLCompactInterleavedHeader
	{
		std::uint64_t magic{ kGGMLCompactInterleavedMagic };
		std::uint32_t version{ kGGMLCompactInterleavedVersion };
		std::uint32_t format{};
		std::uint64_t rows{};
		std::uint64_t columns{};
		std::uint64_t blockCount{};
		std::uint64_t bytesPerBlock{};
		std::uint64_t payloadBytes{};
		std::uint64_t reserved{};
	};

	static_assert(sizeof(GGMLCompactInterleavedHeader) == 64);

	constexpr std::uint64_t kGGMLFieldInterleavedV4Magic = 0x34564C49464E4E4Cull;
	constexpr std::uint32_t kGGMLFieldInterleavedV4Version = 4;

	struct GGMLFieldInterleavedV4Header
	{
		std::uint64_t magic{ kGGMLFieldInterleavedV4Magic };
		std::uint32_t version{ kGGMLFieldInterleavedV4Version };
		std::uint32_t format{};
		std::uint64_t rows{};
		std::uint64_t columns{};
		std::uint64_t blockCount{};
		std::uint64_t bytesPerGroupBlock{};
		std::uint64_t payloadBytes{};
		std::uint64_t reserved{};
	};

	static_assert(sizeof(GGMLFieldInterleavedV4Header) == 64);

	std::optional<std::uint64_t> GGMLFieldInterleavedV4GroupBlockBytes(QuantizedBlockFormat format)
	{
		switch (format)
		{
		case QuantizedBlockFormat::GGML_Q4_K:
			return sizeof(GGMLQ4KFieldInterleaved8Block);
		case QuantizedBlockFormat::GGML_Q6_K:
			return sizeof(GGMLQ6KFieldInterleaved8Block);
		default:
			return std::nullopt;
		}
	}

	std::optional<std::uint64_t> GGMLFieldInterleavedV4ByteSize(QuantizedBlockFormat format, std::uint64_t rows,
	                                                            std::uint64_t columns)
	{
		const auto layout = GetQuantizedBlockLayout(format);
		const auto bytesPerGroupBlock = GGMLFieldInterleavedV4GroupBlockBytes(format);
		if (!layout || !bytesPerGroupBlock || columns == 0 || columns % layout->elementsPerBlock != 0 ||
		    rows > std::numeric_limits<std::uint64_t>::max() - 7)
		{
			return std::nullopt;
		}
		const auto blockCount = columns / layout->elementsPerBlock;
		const auto rowGroups = (rows + 7) / 8;
		if (rowGroups != 0 &&
		    blockCount > (std::numeric_limits<std::uint64_t>::max() - sizeof(GGMLFieldInterleavedV4Header)) /
		                     rowGroups / *bytesPerGroupBlock)
		{
			return std::nullopt;
		}
		return sizeof(GGMLFieldInterleavedV4Header) + rowGroups * blockCount * *bytesPerGroupBlock;
	}

	const GGMLFieldInterleavedV4Header* ResolveGGMLFieldInterleavedV4Header(const std::uint8_t* aligned,
	                                                                        std::int64_t offset, std::int64_t bytes,
	                                                                        std::int64_t stride,
	                                                                        QuantizedBlockFormat format,
	                                                                        std::uint64_t rows, std::uint64_t columns)
	{
		if (!aligned || offset < 0 || bytes < 0 || stride != 1 ||
		    static_cast<std::uint64_t>(offset) > static_cast<std::uint64_t>(bytes) ||
		    static_cast<std::uint64_t>(bytes) - static_cast<std::uint64_t>(offset) <
		        sizeof(GGMLFieldInterleavedV4Header))
		{
			return nullptr;
		}
		const auto* header =
		    reinterpret_cast<const GGMLFieldInterleavedV4Header*>(aligned + static_cast<std::uint64_t>(offset));
		const auto requiredBytes = GGMLFieldInterleavedV4ByteSize(format, rows, columns);
		const auto groupBlockBytes = GGMLFieldInterleavedV4GroupBlockBytes(format);
		const auto layout = GetQuantizedBlockLayout(format);
		if (!requiredBytes || !groupBlockBytes || !layout || header->magic != kGGMLFieldInterleavedV4Magic ||
		    header->version != kGGMLFieldInterleavedV4Version || header->format != static_cast<std::uint32_t>(format) ||
		    header->rows != rows || header->columns != columns ||
		    header->blockCount != columns / layout->elementsPerBlock ||
		    header->bytesPerGroupBlock != *groupBlockBytes ||
		    header->payloadBytes != *requiredBytes - sizeof(*header) ||
		    static_cast<std::uint64_t>(bytes) - static_cast<std::uint64_t>(offset) < *requiredBytes)
		{
			return nullptr;
		}
		return header;
	}

	std::optional<std::uint64_t> GGMLCompactInterleavedByteSize(QuantizedBlockFormat format, std::uint64_t rows,
	                                                            std::uint64_t columns)
	{
		const auto layout = GetQuantizedBlockLayout(format);
		if (!layout || !IsGGMLQ8KStagedMatMulFormat(format) || columns == 0 ||
		    columns % layout->elementsPerBlock != 0 || rows > std::numeric_limits<std::uint64_t>::max() - 3)
		{
			return std::nullopt;
		}
		const auto blockCount = columns / layout->elementsPerBlock;
		const auto paddedRows = (rows + 3) & ~std::uint64_t{ 3 };
		if (paddedRows != 0 &&
		    blockCount > (std::numeric_limits<std::uint64_t>::max() - sizeof(GGMLCompactInterleavedHeader)) /
		                     paddedRows / layout->bytesPerBlock)
		{
			return std::nullopt;
		}
		return sizeof(GGMLCompactInterleavedHeader) + paddedRows * blockCount * layout->bytesPerBlock;
	}

	const GGMLCompactInterleavedHeader* ResolveGGMLCompactInterleavedHeader(const std::uint8_t* aligned,
	                                                                        std::int64_t offset, std::int64_t bytes,
	                                                                        std::int64_t stride,
	                                                                        QuantizedBlockFormat format,
	                                                                        std::uint64_t rows, std::uint64_t columns)
	{
		if (!aligned || offset < 0 || bytes < 0 || stride != 1 ||
		    static_cast<std::uint64_t>(offset) > static_cast<std::uint64_t>(bytes) ||
		    static_cast<std::uint64_t>(bytes) - static_cast<std::uint64_t>(offset) <
		        sizeof(GGMLCompactInterleavedHeader))
		{
			return nullptr;
		}
		const auto* header =
		    reinterpret_cast<const GGMLCompactInterleavedHeader*>(aligned + static_cast<std::uint64_t>(offset));
		const auto requiredBytes = GGMLCompactInterleavedByteSize(format, rows, columns);
		if (!requiredBytes || header->magic != kGGMLCompactInterleavedMagic ||
		    header->version != kGGMLCompactInterleavedVersion || header->format != static_cast<std::uint32_t>(format) ||
		    header->rows != rows || header->columns != columns ||
		    header->payloadBytes != *requiredBytes - sizeof(*header) ||
		    static_cast<std::uint64_t>(bytes) - static_cast<std::uint64_t>(offset) < *requiredBytes)
		{
			return nullptr;
		}
		const auto layout = GetQuantizedBlockLayout(format);
		if (!layout || header->blockCount != columns / layout->elementsPerBlock ||
		    header->bytesPerBlock != layout->bytesPerBlock)
		{
			return nullptr;
		}
		return header;
	}

	const GGMLBlockMatMulProjection*
	ResolveGGMLBlockMatMulProjection(std::span<const GGMLBlockMatMulProjection> projections, std::uint64_t column,
	                                 std::uint64_t& localColumn)
	{
		localColumn = column;
		for (const auto& projection : projections)
		{
			const auto width = static_cast<std::uint64_t>(projection.outColumns);
			if (localColumn < width)
			{
				return &projection;
			}
			localColumn -= width;
		}
		return nullptr;
	}

	void LiteNNCPUGGMLBlockMatMulProjectedF32(
	    const float*, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows, std::int64_t lhsColumns,
	    std::int64_t lhsRowStride, std::int64_t lhsColumnStride, std::span<const GGMLBlockMatMulProjection> projections,
	    float*, float* outAligned, std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns,
	    std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t formatValue,
	    std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue, GGMLActivationDotMode activationDotMode,
	    const GGMLQ8KActivationBlock* preparedQ8KActivationBlocks = nullptr)
	{
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		const auto layout = GetQuantizedBlockLayout(format);
		if (!layout || projections.empty() || lhsRows < 0 || lhsColumns < 0 || outRows < 0 || outColumns < 0 ||
		    lhsRows != outRows || lhsColumns == 0 || outColumns == 0 ||
		    static_cast<std::uint64_t>(lhsColumns) % layout->elementsPerBlock != 0)
		{
			return;
		}
		std::int64_t totalProjectionColumns = 0;
		const auto rhsStride = projections.front().rhsStride;
		for (const auto& projection : projections)
		{
			if (projection.outColumns < 0 || projection.rhsBytes < 0 || projection.rhsStride != rhsStride)
			{
				return;
			}
			if (projection.outColumns > std::numeric_limits<std::int64_t>::max() - totalProjectionColumns)
			{
				return;
			}
			totalProjectionColumns += projection.outColumns;
		}
		if (totalProjectionColumns != outColumns)
		{
			return;
		}

		const auto rowBytes =
		    (static_cast<std::uint64_t>(lhsColumns) / layout->elementsPerBlock) * layout->bytesPerBlock;
		for (const auto& projection : projections)
		{
			if (static_cast<std::uint64_t>(projection.rhsBytes) <
			    static_cast<std::uint64_t>(projection.outColumns) * rowBytes)
			{
				return;
			}
		}

		if (rhsStride <= 0)
		{
			return;
		}

		struct Context
		{
			const float* lhsAligned{};
			std::int64_t lhsOffset{};
			std::int64_t lhsColumns{};
			std::int64_t lhsRowStride{};
			std::int64_t lhsColumnStride{};
			std::int64_t rhsStride{};
			const GGMLBlockMatMulProjection* projections{};
			std::uint64_t projectionCount{};
			float* outAligned{};
			std::int64_t outOffset{};
			std::int64_t outColumns{};
			std::int64_t outRowStride{};
			std::int64_t outColumnStride{};
			std::uint64_t rowBytes{};
			QuantizedBlockFormat format{};
			std::uint64_t elementsPerBlock{};
			std::uint64_t bytesPerBlock{};
			std::uint64_t blockCount{};
			std::uint64_t columnGroupsPerRow{};
			GGMLActivationDotMode activationDotMode{ GGMLActivationDotMode::DirectFloat32 };
			const float* lhsSubblockSums{};
			const GGMLQ8KActivationBlock* lhsQ8KBlocks{};
		};
		const auto blockCount = static_cast<std::uint64_t>(lhsColumns) / layout->elementsPerBlock;
		const auto effectiveActivationDotMode =
		    activationDotMode == GGMLActivationDotMode::Q8KStaged && IsGGMLQ8KStagedMatMulFormat(format)
		        ? GGMLActivationDotMode::Q8KStaged
		        : GGMLActivationDotMode::DirectFloat32;
		std::vector<float> lhsSubblockSums;
		std::vector<GGMLQ8KActivationBlock> lhsQ8KBlocks;
		const GGMLQ8KActivationBlock* effectiveQ8KBlocks = preparedQ8KActivationBlocks;
		if (effectiveActivationDotMode == GGMLActivationDotMode::Q8KStaged)
		{
			if (!effectiveQ8KBlocks)
			{
				if (!lhsAligned)
				{
					return;
				}
				lhsQ8KBlocks.resize(static_cast<std::size_t>(static_cast<std::uint64_t>(lhsRows) * blockCount));
				for (std::int64_t row = 0; row < lhsRows; ++row)
				{
					const auto* lhsRow = lhsAligned + lhsOffset + row * lhsRowStride;
					for (std::uint64_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
					{
						const auto* lhsBlock =
						    lhsRow + static_cast<std::int64_t>(blockIndex * layout->elementsPerBlock) * lhsColumnStride;
						const auto offset = static_cast<std::uint64_t>(row) * blockCount + blockIndex;
						QuantizeGGMLQ8KActivationBlock(lhsBlock, lhsColumnStride,
						                               lhsQ8KBlocks[static_cast<std::size_t>(offset)]);
					}
				}
				effectiveQ8KBlocks = lhsQ8KBlocks.data();
			}
		}
		else if (format == QuantizedBlockFormat::GGML_Q4_K || format == QuantizedBlockFormat::GGML_Q5_K)
		{
			if (!lhsAligned)
			{
				return;
			}
			constexpr std::uint64_t kSubblocksPerKBlock = 8;
			constexpr std::uint64_t kLanesPerSubblock = 32;
			lhsSubblockSums.resize(
			    static_cast<std::size_t>(static_cast<std::uint64_t>(lhsRows) * blockCount * kSubblocksPerKBlock));
			for (std::int64_t row = 0; row < lhsRows; ++row)
			{
				const auto* lhsRow = lhsAligned + lhsOffset + row * lhsRowStride;
				for (std::uint64_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
				{
					for (std::uint64_t subblock = 0; subblock < kSubblocksPerKBlock; ++subblock)
					{
						float sum = 0.0F;
						const auto baseLane = blockIndex * layout->elementsPerBlock + subblock * kLanesPerSubblock;
						for (std::uint64_t lane = 0; lane < kLanesPerSubblock; ++lane)
						{
							sum += lhsRow[static_cast<std::int64_t>(baseLane + lane) * lhsColumnStride];
						}
						const auto offset =
						    (static_cast<std::uint64_t>(row) * blockCount + blockIndex) * kSubblocksPerKBlock +
						    subblock;
						lhsSubblockSums[static_cast<std::size_t>(offset)] = sum;
					}
				}
			}
		}
		Context context{ .lhsAligned = lhsAligned,
			             .lhsOffset = lhsOffset,
			             .lhsColumns = lhsColumns,
			             .lhsRowStride = lhsRowStride,
			             .lhsColumnStride = lhsColumnStride,
			             .rhsStride = rhsStride,
			             .projections = projections.data(),
			             .projectionCount = projections.size(),
			             .outAligned = outAligned,
			             .outOffset = outOffset,
			             .outColumns = outColumns,
			             .outRowStride = outRowStride,
			             .outColumnStride = outColumnStride,
			             .rowBytes = rowBytes,
			             .format = format,
			             .elementsPerBlock = layout->elementsPerBlock,
			             .bytesPerBlock = layout->bytesPerBlock,
			             .blockCount = blockCount,
			             .columnGroupsPerRow = (static_cast<std::uint64_t>(outColumns) + 3) / 4,
			             .activationDotMode = effectiveActivationDotMode,
			             .lhsSubblockSums = lhsSubblockSums.empty() ? nullptr : lhsSubblockSums.data(),
			             .lhsQ8KBlocks = effectiveQ8KBlocks };
		const auto outputElements = static_cast<std::uint64_t>(lhsRows) * static_cast<std::uint64_t>(outColumns);
		const auto body = [](std::uint64_t begin, std::uint64_t end, void* userData) {
			const auto& ctx = *static_cast<const Context*>(userData);
			for (std::uint64_t outputIndex = begin; outputIndex < end; ++outputIndex)
			{
				const auto row = static_cast<std::int64_t>(outputIndex / static_cast<std::uint64_t>(ctx.outColumns));
				const auto column = static_cast<std::int64_t>(outputIndex % static_cast<std::uint64_t>(ctx.outColumns));
				std::uint64_t localColumn = 0;
				const auto* projection = ResolveGGMLBlockMatMulProjection(
				    { ctx.projections, ctx.projectionCount }, static_cast<std::uint64_t>(column), localColumn);
				if (!projection)
				{
					continue;
				}
				const auto* weightRow = projection->rhsAligned + projection->rhsOffset +
				                        static_cast<std::int64_t>(localColumn * ctx.rowBytes) * ctx.rhsStride;
				float acc = 0.0F;
				const auto* lhsRow = ctx.lhsAligned + ctx.lhsOffset + row * ctx.lhsRowStride;
				for (std::uint64_t blockIndex = 0; blockIndex < ctx.blockCount; ++blockIndex)
				{
					const auto* block = weightRow + blockIndex * ctx.bytesPerBlock * ctx.rhsStride;
					const auto* lhsBlock =
					    lhsRow + static_cast<std::int64_t>(blockIndex * ctx.elementsPerBlock) * ctx.lhsColumnStride;
					const float* lhsSums = nullptr;
					if (ctx.lhsSubblockSums)
					{
						constexpr std::uint64_t kSubblocksPerKBlock = 8;
						const auto sumOffset =
						    (static_cast<std::uint64_t>(row) * ctx.blockCount + blockIndex) * kSubblocksPerKBlock;
						lhsSums = ctx.lhsSubblockSums + static_cast<std::size_t>(sumOffset);
					}
					acc += DotGGMLBlockF32(block, ctx.rhsStride, lhsBlock, ctx.lhsColumnStride, ctx.format, lhsSums);
				}
				ctx.outAligned[ctx.outOffset + row * ctx.outRowStride + column * ctx.outColumnStride] =
				    static_cast<float>(acc);
			}
		};
		const auto operations = outputElements * static_cast<std::uint64_t>(lhsColumns);
		const auto threadCount = ResolveGGMLBlockMatMulThreadCount(format, effectiveActivationDotMode, operations,
		                                                           outputElements, requestedThreadCount);
		const auto affinityPolicy = ResolveCPUAOTAffinityPolicy(affinityPolicyValue);
		const auto waitPolicy = ResolveCPUAOTWorkerWaitPolicy(affinityPolicyValue);
		if (format == QuantizedBlockFormat::GGML_Q8_0 || format == QuantizedBlockFormat::GGML_Q4_K ||
		    format == QuantizedBlockFormat::GGML_Q5_K || format == QuantizedBlockFormat::GGML_Q6_K)
		{
			const auto groupedBody = [](std::uint64_t begin, std::uint64_t end, void* userData) {
				const auto& ctx = *static_cast<const Context*>(userData);
				for (std::uint64_t groupIndex = begin; groupIndex < end; ++groupIndex)
				{
					const auto row = static_cast<std::int64_t>(groupIndex / ctx.columnGroupsPerRow);
					const auto columnBase = (groupIndex % ctx.columnGroupsPerRow) * static_cast<std::uint64_t>(4);
					const auto* lhsRow = ctx.activationDotMode == GGMLActivationDotMode::Q8KStaged
					                         ? nullptr
					                         : ctx.lhsAligned + ctx.lhsOffset + row * ctx.lhsRowStride;
					float acc[4] = {};
					bool valid[4] = { columnBase < static_cast<std::uint64_t>(ctx.outColumns),
						              columnBase + 1 < static_cast<std::uint64_t>(ctx.outColumns),
						              columnBase + 2 < static_cast<std::uint64_t>(ctx.outColumns),
						              columnBase + 3 < static_cast<std::uint64_t>(ctx.outColumns) };
					const std::uint8_t* weightRows[4] = {};
					for (int localColumn = 0; localColumn < 4; ++localColumn)
					{
						if (!valid[localColumn])
						{
							continue;
						}
						const auto column =
						    static_cast<std::int64_t>(columnBase + static_cast<std::uint64_t>(localColumn));
						std::uint64_t projectionColumn = 0;
						const auto* projection =
						    ResolveGGMLBlockMatMulProjection({ ctx.projections, ctx.projectionCount },
						                                     static_cast<std::uint64_t>(column), projectionColumn);
						if (!projection)
						{
							valid[localColumn] = false;
							continue;
						}
						weightRows[localColumn] =
						    projection->rhsAligned + projection->rhsOffset +
						    static_cast<std::int64_t>(projectionColumn * ctx.rowBytes) * ctx.rhsStride;
					}
					for (std::uint64_t blockIndex = 0; blockIndex < ctx.blockCount; ++blockIndex)
					{
						const float* lhsBlock = nullptr;
						if (ctx.activationDotMode != GGMLActivationDotMode::Q8KStaged)
						{
							lhsBlock = lhsRow + static_cast<std::int64_t>(blockIndex * ctx.elementsPerBlock) *
							                        ctx.lhsColumnStride;
						}
						const float* lhsSums = nullptr;
						if (ctx.lhsSubblockSums)
						{
							constexpr std::uint64_t kSubblocksPerKBlock = 8;
							const auto sumOffset =
							    (static_cast<std::uint64_t>(row) * ctx.blockCount + blockIndex) * kSubblocksPerKBlock;
							lhsSums = ctx.lhsSubblockSums + static_cast<std::size_t>(sumOffset);
						}

						const std::uint8_t* blocks[4] = {};
						for (int localColumn = 0; localColumn < 4; ++localColumn)
						{
							if (!valid[localColumn])
							{
								continue;
							}
							blocks[localColumn] =
							    weightRows[localColumn] +
							    static_cast<std::int64_t>(blockIndex * ctx.bytesPerBlock) * ctx.rhsStride;
						}
						if (ctx.activationDotMode == GGMLActivationDotMode::Q8KStaged)
						{
							const auto* lhsQ8KBlock =
							    ctx.lhsQ8KBlocks + static_cast<std::uint64_t>(row) * ctx.blockCount + blockIndex;
							if (ctx.format == QuantizedBlockFormat::GGML_Q4_K)
							{
#if LITENN_HAS_X86_AVX2_TARGET
								if (LiteNNCPUHasAVX2())
								{
									if (valid[0] && valid[1] && valid[2] && valid[3])
									{
										AccumulateGGMLQ4KBlockQ8Kx4AVX2AllValid(blocks, ctx.rhsStride, *lhsQ8KBlock,
										                                        acc);
									}
									else
									{
										AccumulateGGMLQ4KBlockQ8Kx4AVX2(blocks, valid, ctx.rhsStride, *lhsQ8KBlock,
										                                acc);
									}
								}
								else
#endif
									AccumulateGGMLQ4KBlockQ8Kx4(blocks, valid, ctx.rhsStride, *lhsQ8KBlock, acc);
							}
							else if (ctx.format == QuantizedBlockFormat::GGML_Q5_K)
							{
#if LITENN_HAS_X86_AVX2_TARGET
								if (LiteNNCPUHasAVX2())
								{
									if (valid[0] && valid[1] && valid[2] && valid[3])
									{
										AccumulateGGMLQ5KBlockQ8Kx4AVX2AllValid(blocks, ctx.rhsStride, *lhsQ8KBlock,
										                                        acc);
									}
									else
									{
										AccumulateGGMLQ5KBlockQ8Kx4AVX2(blocks, valid, ctx.rhsStride, *lhsQ8KBlock,
										                                acc);
									}
								}
								else
#endif
									AccumulateGGMLQ5KBlockQ8Kx4(blocks, valid, ctx.rhsStride, *lhsQ8KBlock, acc);
							}
							else
							{
#if LITENN_HAS_X86_AVX2_TARGET
								if (LiteNNCPUHasAVX2())
								{
									if (valid[0] && valid[1] && valid[2] && valid[3])
									{
										AccumulateGGMLQ6KBlockQ8Kx4AVX2AllValid(blocks, ctx.rhsStride, *lhsQ8KBlock,
										                                        acc);
									}
									else
									{
										AccumulateGGMLQ6KBlockQ8Kx4AVX2(blocks, valid, ctx.rhsStride, *lhsQ8KBlock,
										                                acc);
									}
								}
								else
#endif
									AccumulateGGMLQ6KBlockQ8Kx4(blocks, valid, ctx.rhsStride, *lhsQ8KBlock, acc);
							}
						}
						else if (ctx.format == QuantizedBlockFormat::GGML_Q8_0)
						{
							AccumulateGGMLQ8_0BlockF32x4(blocks, valid, ctx.rhsStride, lhsBlock, ctx.lhsColumnStride,
							                             acc);
						}
						else if (ctx.format == QuantizedBlockFormat::GGML_Q4_K)
						{
							AccumulateGGMLQ4KBlockF32x4(blocks, valid, ctx.rhsStride, lhsBlock, ctx.lhsColumnStride,
							                            lhsSums, acc);
						}
						else if (ctx.format == QuantizedBlockFormat::GGML_Q5_K)
						{
							AccumulateGGMLQ5KBlockF32x4(blocks, valid, ctx.rhsStride, lhsBlock, ctx.lhsColumnStride,
							                            lhsSums, acc);
						}
						else
						{
							AccumulateGGMLQ6KBlockF32x4(blocks, valid, ctx.rhsStride, lhsBlock, ctx.lhsColumnStride,
							                            acc);
						}
					}

					for (int localColumn = 0; localColumn < 4; ++localColumn)
					{
						if (!valid[localColumn])
						{
							continue;
						}
						const auto column =
						    static_cast<std::int64_t>(columnBase + static_cast<std::uint64_t>(localColumn));
						ctx.outAligned[ctx.outOffset + row * ctx.outRowStride + column * ctx.outColumnStride] =
						    acc[localColumn];
					}
				}
			};
			const auto outputGroups = static_cast<std::uint64_t>(lhsRows) * context.columnGroupsPerRow;
			const auto groupedThreadCount = ResolveGGMLBlockMatMulThreadCount(
			    format, effectiveActivationDotMode, operations, outputGroups, requestedThreadCount);
			const auto groupedGrain =
			    std::max<std::uint64_t>(1, outputGroups / (std::max<std::uint64_t>(1, groupedThreadCount) * 8));
			if (groupedThreadCount <= 1)
			{
				groupedBody(0, outputGroups, &context);
				return;
			}
			LiteNNCPUParallelFor(0, outputGroups, groupedGrain, groupedBody, &context, groupedThreadCount,
			                     affinityPolicy, waitPolicy);
			return;
		}
		const auto grain = std::max<std::uint64_t>(1, outputElements / (std::max<std::uint64_t>(1, threadCount) * 8));
		if (threadCount <= 1)
		{
			body(0, outputElements, &context);
			return;
		}
		LiteNNCPUParallelFor(0, outputElements, grain, body, &context, threadCount, affinityPolicy, waitPolicy);
	}

	void LiteNNCPUGGMLBlockMatMulF32(const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset,
	                                 std::int64_t lhsRows, std::int64_t lhsColumns, std::int64_t lhsRowStride,
	                                 std::int64_t lhsColumnStride, const std::uint8_t*, const std::uint8_t* rhsAligned,
	                                 std::int64_t rhsOffset, std::int64_t rhsBytes, std::int64_t rhsStride, float*,
	                                 float* outAligned, std::int64_t outOffset, std::int64_t outRows,
	                                 std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride,
	                                 std::uint64_t formatValue, std::uint64_t requestedThreadCount,
	                                 std::uint64_t affinityPolicyValue, GGMLActivationDotMode activationDotMode)
	{
		const GGMLBlockMatMulProjection projection{
			.rhsAligned = rhsAligned,
			.rhsOffset = rhsOffset,
			.rhsBytes = rhsBytes,
			.rhsStride = rhsStride,
			.outColumns = outColumns,
		};
		LiteNNCPUGGMLBlockMatMulProjectedF32(lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride,
		                                     lhsColumnStride,
		                                     std::span<const GGMLBlockMatMulProjection>{ &projection, 1 }, nullptr,
		                                     outAligned, outOffset, outRows, outColumns, outRowStride, outColumnStride,
		                                     formatValue, requestedThreadCount, affinityPolicyValue, activationDotMode);
	}

	extern "C" void litenn_cpu_ggml_block_matmul_f32(
	    const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows,
	    std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t* rhsBase,
	    const std::uint8_t* rhsAligned, std::int64_t rhsOffset, std::int64_t rhsBytes, std::int64_t rhsStride,
	    float* outBase, float* outAligned, std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns,
	    std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t formatValue,
	    std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_matmul_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLBlockMatMulProfileDetail(format, GGMLActivationDotMode::DirectFloat32, lhsRows, lhsColumns,
		                                            outRows, outColumns, requestedThreadCount)
		        : std::string{});
		LiteNNCPUGGMLBlockMatMulF32(lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride, lhsColumnStride,
		                            rhsBase, rhsAligned, rhsOffset, rhsBytes, rhsStride, outBase, outAligned, outOffset,
		                            outRows, outColumns, outRowStride, outColumnStride, formatValue,
		                            requestedThreadCount, affinityPolicyValue, GGMLActivationDotMode::DirectFloat32);
	}

	extern "C" void litenn_cpu_ggml_block_matmul_q8k_staged_f32(
	    const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows,
	    std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t* rhsBase,
	    const std::uint8_t* rhsAligned, std::int64_t rhsOffset, std::int64_t rhsBytes, std::int64_t rhsStride,
	    float* outBase, float* outAligned, std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns,
	    std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t formatValue,
	    std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_matmul_q8k_staged_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLBlockMatMulProfileDetail(format, GGMLActivationDotMode::Q8KStaged, lhsRows, lhsColumns,
		                                            outRows, outColumns, requestedThreadCount)
		        : std::string{});
		LiteNNCPUGGMLBlockMatMulF32(lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride, lhsColumnStride,
		                            rhsBase, rhsAligned, rhsOffset, rhsBytes, rhsStride, outBase, outAligned, outOffset,
		                            outRows, outColumns, outRowStride, outColumnStride, formatValue,
		                            requestedThreadCount, affinityPolicyValue, GGMLActivationDotMode::Q8KStaged);
	}

	extern "C" std::uint64_t litenn_cpu_ggml_compact_interleaved_bytes(std::uint64_t formatValue, std::int64_t rows,
	                                                                   std::int64_t columns)
	{
		if (rows < 0 || columns < 0)
		{
			return 0;
		}
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		if (format != QuantizedBlockFormat::GGML_Q4_K && format != QuantizedBlockFormat::GGML_Q6_K)
		{
			return 0;
		}
		return GGMLCompactInterleavedByteSize(format, static_cast<std::uint64_t>(rows),
		                                      static_cast<std::uint64_t>(columns))
		    .value_or(0);
	}

	extern "C" void litenn_cpu_ggml_prepack_compact_interleaved(
	    const std::uint8_t*, const std::uint8_t* rhsAligned, std::int64_t rhsOffset, std::int64_t rhsBytes,
	    std::int64_t rhsStride, std::int64_t rows, std::int64_t columns, std::uint64_t formatValue, std::uint8_t*,
	    std::uint8_t* packedAligned, std::int64_t packedOffset, std::int64_t packedBytes, std::int64_t packedStride)
	{
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		const auto layout = GetQuantizedBlockLayout(format);
		const auto requiredBytes = rows >= 0 && columns >= 0
		                               ? GGMLCompactInterleavedByteSize(format, static_cast<std::uint64_t>(rows),
		                                                                static_cast<std::uint64_t>(columns))
		                               : std::nullopt;
		if (!layout || !requiredBytes || !rhsAligned || !packedAligned || rhsOffset < 0 || rhsBytes < 0 ||
		    packedOffset < 0 || packedBytes < 0 || rhsStride != 1 || packedStride != 1 ||
		    (format != QuantizedBlockFormat::GGML_Q4_K && format != QuantizedBlockFormat::GGML_Q6_K))
		{
			return;
		}
		const auto sourceBytes = static_cast<std::uint64_t>(rows) *
		                         (static_cast<std::uint64_t>(columns) / layout->elementsPerBlock) *
		                         layout->bytesPerBlock;
		if (static_cast<std::uint64_t>(rhsOffset) > static_cast<std::uint64_t>(rhsBytes) ||
		    static_cast<std::uint64_t>(rhsBytes) - static_cast<std::uint64_t>(rhsOffset) < sourceBytes ||
		    static_cast<std::uint64_t>(packedOffset) > static_cast<std::uint64_t>(packedBytes) ||
		    static_cast<std::uint64_t>(packedBytes) - static_cast<std::uint64_t>(packedOffset) < *requiredBytes)
		{
			return;
		}

		auto* target = packedAligned + static_cast<std::uint64_t>(packedOffset);
		const auto blockCount = static_cast<std::uint64_t>(columns) / layout->elementsPerBlock;
		const auto paddedRows = (static_cast<std::uint64_t>(rows) + 3) & ~std::uint64_t{ 3 };
		const auto payloadBytes = *requiredBytes - sizeof(GGMLCompactInterleavedHeader);
		const GGMLCompactInterleavedHeader header{
			.format = static_cast<std::uint32_t>(format),
			.rows = static_cast<std::uint64_t>(rows),
			.columns = static_cast<std::uint64_t>(columns),
			.blockCount = blockCount,
			.bytesPerBlock = layout->bytesPerBlock,
			.payloadBytes = payloadBytes,
		};
		std::memcpy(target, &header, sizeof(header));
		auto* payload = target + sizeof(header);
		std::memset(payload, 0, static_cast<std::size_t>(payloadBytes));
		const auto* source = rhsAligned + static_cast<std::uint64_t>(rhsOffset);
		const auto sourceRowBytes = blockCount * layout->bytesPerBlock;
		for (std::uint64_t row = 0; row < paddedRows; ++row)
		{
			if (row >= static_cast<std::uint64_t>(rows))
			{
				break;
			}
			const auto group = row / 4;
			const auto lane = row % 4;
			for (std::uint64_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
			{
				const auto* sourceBlock = source + row * sourceRowBytes + blockIndex * layout->bytesPerBlock;
				auto* targetBlock = payload + (group * blockCount + blockIndex) * layout->bytesPerBlock * 4 +
				                    lane * layout->bytesPerBlock;
				std::memcpy(targetBlock, sourceBlock, static_cast<std::size_t>(layout->bytesPerBlock));
			}
		}
	}

	extern "C" std::uint64_t litenn_cpu_ggml_field_interleaved_v4_bytes(std::uint64_t formatValue, std::int64_t rows,
	                                                                    std::int64_t columns)
	{
		if (rows < 0 || columns < 0)
		{
			return 0;
		}
		return GGMLFieldInterleavedV4ByteSize(static_cast<QuantizedBlockFormat>(formatValue),
		                                      static_cast<std::uint64_t>(rows), static_cast<std::uint64_t>(columns))
		    .value_or(0);
	}

	extern "C" void litenn_cpu_ggml_prepack_field_interleaved_v4(
	    const std::uint8_t*, const std::uint8_t* rhsAligned, std::int64_t rhsOffset, std::int64_t rhsBytes,
	    std::int64_t rhsStride, std::int64_t rows, std::int64_t columns, std::uint64_t formatValue, std::uint8_t*,
	    std::uint8_t* packedAligned, std::int64_t packedOffset, std::int64_t packedBytes, std::int64_t packedStride)
	{
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		const auto layout = GetQuantizedBlockLayout(format);
		const auto groupBlockBytes = GGMLFieldInterleavedV4GroupBlockBytes(format);
		const auto requiredBytes = rows >= 0 && columns >= 0
		                               ? GGMLFieldInterleavedV4ByteSize(format, static_cast<std::uint64_t>(rows),
		                                                                static_cast<std::uint64_t>(columns))
		                               : std::nullopt;
		if (!layout || !groupBlockBytes || !requiredBytes || !rhsAligned || !packedAligned || rhsOffset < 0 ||
		    rhsBytes < 0 || packedOffset < 0 || packedBytes < 0 || rhsStride != 1 || packedStride != 1)
		{
			return;
		}
		const auto blockCount = static_cast<std::uint64_t>(columns) / layout->elementsPerBlock;
		const auto sourceBytes = static_cast<std::uint64_t>(rows) * blockCount * layout->bytesPerBlock;
		if (static_cast<std::uint64_t>(rhsOffset) > static_cast<std::uint64_t>(rhsBytes) ||
		    static_cast<std::uint64_t>(rhsBytes) - static_cast<std::uint64_t>(rhsOffset) < sourceBytes ||
		    static_cast<std::uint64_t>(packedOffset) > static_cast<std::uint64_t>(packedBytes) ||
		    static_cast<std::uint64_t>(packedBytes) - static_cast<std::uint64_t>(packedOffset) < *requiredBytes)
		{
			return;
		}

		auto* target = packedAligned + static_cast<std::uint64_t>(packedOffset);
		const GGMLFieldInterleavedV4Header header{
			.format = static_cast<std::uint32_t>(format),
			.rows = static_cast<std::uint64_t>(rows),
			.columns = static_cast<std::uint64_t>(columns),
			.blockCount = blockCount,
			.bytesPerGroupBlock = *groupBlockBytes,
			.payloadBytes = *requiredBytes - sizeof(GGMLFieldInterleavedV4Header),
		};
		std::memcpy(target, &header, sizeof(header));
		auto* payload = target + sizeof(header);
		std::memset(payload, 0, static_cast<std::size_t>(header.payloadBytes));
		const auto* source = rhsAligned + static_cast<std::uint64_t>(rhsOffset);
		const auto sourceRowBytes = blockCount * layout->bytesPerBlock;
		const auto rowGroups = (static_cast<std::uint64_t>(rows) + 7) / 8;
		for (std::uint64_t rowGroup = 0; rowGroup < rowGroups; ++rowGroup)
		{
			for (std::uint64_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
			{
				auto* packedBlock = payload + (rowGroup * blockCount + blockIndex) * *groupBlockBytes;
				for (std::uint64_t lane = 0; lane < 8; ++lane)
				{
					const auto row = rowGroup * 8 + lane;
					if (row >= static_cast<std::uint64_t>(rows))
					{
						continue;
					}
					const auto* sourceBlock = source + row * sourceRowBytes + blockIndex * layout->bytesPerBlock;
					if (format == QuantizedBlockFormat::GGML_Q4_K)
					{
						auto* packedQ4 = reinterpret_cast<GGMLQ4KFieldInterleaved8Block*>(packedBlock);
						std::memcpy(&packedQ4->d[lane], sourceBlock, sizeof(std::uint16_t));
						std::memcpy(&packedQ4->dmin[lane], sourceBlock + 2, sizeof(std::uint16_t));
						for (std::uint64_t subblock = 0; subblock < 8; ++subblock)
						{
							std::uint32_t scale = 0;
							std::uint32_t minimum = 0;
							GGMLQ4Or5KScaleMin(sourceBlock, 1, subblock, scale, minimum);
							packedQ4->scales[subblock][lane] = static_cast<std::uint8_t>(scale);
							packedQ4->minimums[subblock][lane] = static_cast<std::uint8_t>(minimum);
						}
						for (std::uint64_t quantOffset = 0; quantOffset < 128; quantOffset += 4)
						{
							std::memcpy(packedQ4->qs + (quantOffset / 4) * 32 + lane * 4,
							            sourceBlock + 16 + quantOffset, 4);
						}
					}
					else
					{
						auto* packedQ6 = reinterpret_cast<GGMLQ6KFieldInterleaved8Block*>(packedBlock);
						std::memcpy(&packedQ6->d[lane], sourceBlock + 208, sizeof(std::uint16_t));
						for (std::uint64_t subblock = 0; subblock < 16; ++subblock)
						{
							packedQ6->scales[subblock][lane] = static_cast<std::int8_t>(sourceBlock[192 + subblock]);
						}
						for (std::uint64_t quantOffset = 0; quantOffset < 128; quantOffset += 4)
						{
							std::memcpy(packedQ6->ql + (quantOffset / 4) * 32 + lane * 4, sourceBlock + quantOffset, 4);
						}
						for (std::uint64_t quantOffset = 0; quantOffset < 64; quantOffset += 4)
						{
							std::memcpy(packedQ6->qh + (quantOffset / 4) * 32 + lane * 4,
							            sourceBlock + 128 + quantOffset, 4);
						}
					}
				}
			}
		}
	}

	void AccumulateGGMLQ4KFieldInterleavedV4BlockQ8Kx8(const GGMLQ4KFieldInterleaved8Block& block, const bool valid[8],
	                                                   const GGMLQ8KActivationBlock& lhs, float acc[8])
	{
		for (std::uint64_t lane = 0; lane < 8; ++lane)
		{
			if (!valid[lane])
			{
				continue;
			}
			const auto d = ReadGGMLF16Strided(reinterpret_cast<const std::uint8_t*>(&block.d[lane]), 1, 0);
			const auto dmin = ReadGGMLF16Strided(reinterpret_cast<const std::uint8_t*>(&block.dmin[lane]), 1, 0);
			for (std::uint64_t subblock = 0; subblock < 8; ++subblock)
			{
				std::int32_t quantSum = 0;
				for (std::uint64_t local = 0; local < 32; ++local)
				{
					const auto sourceOffset = (subblock / 2) * 32 + local;
					const auto quantByte = block.qs[(sourceOffset / 4) * 32 + lane * 4 + sourceOffset % 4];
					const auto quant = subblock % 2 == 0 ? quantByte & 15U : (quantByte >> 4U) & 15U;
					quantSum +=
					    static_cast<std::int32_t>(quant) * static_cast<std::int32_t>(lhs.qs[subblock * 32 + local]);
				}
				const auto lhsSum = static_cast<std::int32_t>(lhs.bsums[subblock * 2]) +
				                    static_cast<std::int32_t>(lhs.bsums[subblock * 2 + 1]);
				acc[lane] +=
				    lhs.d * d * static_cast<float>(block.scales[subblock][lane]) * static_cast<float>(quantSum) -
				    lhs.d * dmin * static_cast<float>(block.minimums[subblock][lane]) * static_cast<float>(lhsSum);
			}
		}
	}

	void AccumulateGGMLQ6KFieldInterleavedV4BlockQ8Kx8(const GGMLQ6KFieldInterleaved8Block& block, const bool valid[8],
	                                                   const GGMLQ8KActivationBlock& lhs, float acc[8])
	{
		for (std::uint64_t lane = 0; lane < 8; ++lane)
		{
			if (!valid[lane])
			{
				continue;
			}
			const auto d = ReadGGMLF16Strided(reinterpret_cast<const std::uint8_t*>(&block.d[lane]), 1, 0);
			for (std::uint64_t halfBlock = 0; halfBlock < 2; ++halfBlock)
			{
				for (std::uint64_t segment = 0; segment < 4; ++segment)
				{
					for (std::uint64_t group = 0; group < 2; ++group)
					{
						const auto scaleOffset = halfBlock * 8 + group + segment * 2;
						const auto q8Offset = halfBlock * 128 + segment * 32 + group * 16;
						std::int32_t quantSum = 0;
						for (std::uint64_t local = 0; local < 16; ++local)
						{
							const auto qlOffset = halfBlock * 64 + group * 16 + local + (segment % 2) * 32;
							const auto qhOffset = halfBlock * 32 + group * 16 + local;
							const auto ql = block.ql[(qlOffset / 4) * 32 + lane * 4 + qlOffset % 4];
							const auto qh = block.qh[(qhOffset / 4) * 32 + lane * 4 + qhOffset % 4];
							const auto lowFour = segment >= 2 ? (ql >> 4U) & 15U : ql & 15U;
							const auto highTwo = (qh >> (segment * 2)) & 3U;
							const auto quant = static_cast<std::int32_t>(lowFour | (highTwo << 4U)) - 32;
							quantSum += quant * static_cast<std::int32_t>(lhs.qs[q8Offset + local]);
						}
						acc[lane] += lhs.d * d * static_cast<float>(block.scales[scaleOffset][lane]) *
						             static_cast<float>(quantSum);
					}
				}
			}
		}
	}

	void RecordGGMLFieldInterleavedV4ParallelProfile(std::string_view helper, std::string_view detail,
	                                                 std::uint64_t workUnits, std::uint64_t weightBytes,
	                                                 const GGMLQ8KActivationPreparationProfile& activation,
	                                                 const LiteNNCPUThreadPool::ParallelForProfile& parallel)
	{
		if (!CompiledModuleCPUHelperProfilerAccess::Enabled())
		{
			return;
		}

		CompiledModuleCPUParallelProfileEvent event{
			.helper = std::string(helper),
			.detail = std::string(detail),
			.workUnits = workUnits,
			.weightBytes = weightBytes,
			.participantCount = parallel.participants.size(),
			.signaledWorkerCount = parallel.signaledWorkerCount,
			.activationCacheHit = activation.cacheHit,
			.activationLookupMilliseconds = activation.lookupMilliseconds,
			.activationCopyMilliseconds = activation.copyMilliseconds,
			.activationQuantizeMilliseconds = activation.quantizeMilliseconds,
			.threadPoolLockWaitMilliseconds = parallel.lockWaitMilliseconds,
			.dispatchMilliseconds = parallel.dispatchMilliseconds,
			.parallelWallMilliseconds = parallel.wallMilliseconds,
			.barrierWaitMilliseconds = parallel.barrierWaitMilliseconds,
		};
		if (!parallel.participants.empty())
		{
			event.minParticipantTaskClaims = std::numeric_limits<std::uint64_t>::max();
			event.minParticipantWorkUnits = std::numeric_limits<std::uint64_t>::max();
			event.minParticipantUsefulMilliseconds = std::numeric_limits<double>::max();
			for (std::size_t i = 0; i < parallel.participants.size(); ++i)
			{
				const auto& participant = parallel.participants[i];
				event.taskClaims += participant.taskClaims;
				event.minParticipantTaskClaims = std::min(event.minParticipantTaskClaims, participant.taskClaims);
				event.maxParticipantTaskClaims = std::max(event.maxParticipantTaskClaims, participant.taskClaims);
				event.minParticipantWorkUnits = std::min(event.minParticipantWorkUnits, participant.workUnits);
				event.maxParticipantWorkUnits = std::max(event.maxParticipantWorkUnits, participant.workUnits);
				event.minParticipantUsefulMilliseconds =
				    std::min(event.minParticipantUsefulMilliseconds, participant.usefulMilliseconds);
				event.maxParticipantUsefulMilliseconds =
				    std::max(event.maxParticipantUsefulMilliseconds, participant.usefulMilliseconds);
				if (i == 0)
				{
					event.callerUsefulMilliseconds = participant.usefulMilliseconds;
				}
				else
				{
					event.workerUsefulMilliseconds += participant.usefulMilliseconds;
				}
			}
		}
		CompiledModuleCPUHelperProfilerAccess::RecordParallel(std::move(event));
	}

	extern "C" void litenn_cpu_ggml_block_matmul_field_interleaved_v4_q8k_f32(
	    const float*, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows, std::int64_t lhsColumns,
	    std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t*, const std::uint8_t* rhsAligned,
	    std::int64_t rhsOffset, std::int64_t rhsBytes, std::int64_t rhsStride, float*, float* outAligned,
	    std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride,
	    std::int64_t outColumnStride, std::uint64_t formatValue, std::uint64_t requestedThreadCount,
	    std::uint64_t affinityPolicyValue)
	{
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		const auto layout = GetQuantizedBlockLayout(format);
		if (!layout || !lhsAligned || !outAligned || lhsRows < 0 || lhsColumns <= 0 || outRows < 0 || outColumns <= 0 ||
		    lhsRows != outRows || static_cast<std::uint64_t>(lhsColumns) % layout->elementsPerBlock != 0)
		{
			return;
		}
		const auto* header = ResolveGGMLFieldInterleavedV4Header(rhsAligned, rhsOffset, rhsBytes, rhsStride, format,
		                                                         static_cast<std::uint64_t>(outColumns),
		                                                         static_cast<std::uint64_t>(lhsColumns));
		if (!header)
		{
			return;
		}
		const auto schedulingUnits =
		    static_cast<std::uint64_t>(lhsRows) * ((static_cast<std::uint64_t>(outColumns) + 7) / 8);
		const auto threadCount = ResolveGGMLFieldInterleavedV4ThreadCount(format, lhsRows, lhsColumns, outColumns,
		                                                                  schedulingUnits, requestedThreadCount, false);

		constexpr std::string_view helper = "litenn_cpu_ggml_block_matmul_field_interleaved_v4_q8k_f32";
		const auto profileEnabled = CompiledModuleCPUHelperProfilerAccess::Enabled();
		const auto profileDetail =
		    profileEnabled
		        ? BuildGGMLBlockMatMulProfileDetail(format, GGMLActivationDotMode::Q8KStaged, lhsRows, lhsColumns,
		                                            outRows, outColumns, requestedThreadCount, threadCount)
		        : std::string{};
		CPUAOTHelperProfileTimer profileTimer(helper, profileDetail);
		const auto blockCount = header->blockCount;
		GGMLQ8KActivationPreparationProfile activationProfile;
		const auto* staged =
		    PrepareCachedGGMLQ8KActivation(lhsAligned + lhsOffset, lhsRows, lhsColumns, lhsRowStride, lhsColumnStride,
		                                   profileEnabled ? &activationProfile : nullptr);
		if (lhsRows > 0 && !staged)
		{
			return;
		}

		bool useAVX2 = false;
		bool useAVX512 = false;
#if LITENN_HAS_X86_AVX2_TARGET
		useAVX2 = LiteNNCPUHasAVX2F16C();
		useAVX512 = format == QuantizedBlockFormat::GGML_Q6_K && CPUHasGGMLV4AVX512F16C();
#endif
		struct Context
		{
			const std::uint8_t* payload{};
			const GGMLQ8KActivationBlock* staged{};
			float* outAligned{};
			std::int64_t outOffset{};
			std::int64_t outColumns{};
			std::int64_t outRowStride{};
			std::int64_t outColumnStride{};
			std::uint64_t blockCount{};
			std::uint64_t groupBlockBytes{};
			std::uint64_t groupsPerRow{};
			std::uint64_t groupSpan{};
			std::uint64_t workItemsPerRow{};
			QuantizedBlockFormat format{};
			bool useAVX2{};
			bool useAVX512{};
		};
		const auto groupsPerRow = (static_cast<std::uint64_t>(outColumns) + 7) / 8;
		const auto wideOutputBenefitsFromX16 = outColumns >= 32768;
		const auto groupSpan = useAVX2 && (useAVX512 || wideOutputBenefitsFromX16) && (outColumns % 16) == 0
		                           ? std::uint64_t{ 2 }
		                           : std::uint64_t{ 1 };
		const auto workItemsPerRow = (groupsPerRow + groupSpan - 1) / groupSpan;
		Context context{
			.payload = reinterpret_cast<const std::uint8_t*>(header + 1),
			.staged = staged,
			.outAligned = outAligned,
			.outOffset = outOffset,
			.outColumns = outColumns,
			.outRowStride = outRowStride,
			.outColumnStride = outColumnStride,
			.blockCount = blockCount,
			.groupBlockBytes = header->bytesPerGroupBlock,
			.groupsPerRow = groupsPerRow,
			.groupSpan = groupSpan,
			.workItemsPerRow = workItemsPerRow,
			.format = format,
			.useAVX2 = useAVX2,
			.useAVX512 = useAVX512,
		};
		const auto body = [](std::uint64_t begin, std::uint64_t end, void* userData) {
			const auto& ctx = *static_cast<const Context*>(userData);
			for (std::uint64_t workItem = begin; workItem < end; ++workItem)
			{
				const auto row = workItem / ctx.workItemsPerRow;
				const auto group = (workItem % ctx.workItemsPerRow) * ctx.groupSpan;
				const auto columnBase = group * 8;
#if LITENN_HAS_X86_AVX2_TARGET
				if (ctx.groupSpan == 2)
				{
					float acc[16] = {};
					for (std::uint64_t blockIndex = 0; blockIndex < ctx.blockCount; ++blockIndex)
					{
						const auto* packedBlock0 =
						    ctx.payload + (group * ctx.blockCount + blockIndex) * ctx.groupBlockBytes;
						const auto* packedBlock1 =
						    ctx.payload + ((group + 1) * ctx.blockCount + blockIndex) * ctx.groupBlockBytes;
						const auto& lhsBlock = ctx.staged[row * ctx.blockCount + blockIndex];
						if (ctx.format == QuantizedBlockFormat::GGML_Q4_K)
						{
							AccumulateGGMLQ4KFieldInterleavedV4BlockQ8Kx16AVX2(
							    *reinterpret_cast<const GGMLQ4KFieldInterleaved8Block*>(packedBlock0),
							    *reinterpret_cast<const GGMLQ4KFieldInterleaved8Block*>(packedBlock1), lhsBlock, acc);
						}
						else
						{
							const auto& block0 = *reinterpret_cast<const GGMLQ6KFieldInterleaved8Block*>(packedBlock0);
							const auto& block1 = *reinterpret_cast<const GGMLQ6KFieldInterleaved8Block*>(packedBlock1);
							if (ctx.useAVX512)
							{
								AccumulateGGMLQ6KFieldInterleavedV4BlockQ8Kx16AVX512(block0, block1, lhsBlock, acc);
							}
							else
							{
								AccumulateGGMLQ6KFieldInterleavedV4BlockQ8Kx16AVX2(block0, block1, lhsBlock, acc);
							}
						}
					}
					for (std::uint64_t lane = 0; lane < 16; ++lane)
					{
						ctx.outAligned[ctx.outOffset + static_cast<std::int64_t>(row) * ctx.outRowStride +
						               static_cast<std::int64_t>(columnBase + lane) * ctx.outColumnStride] = acc[lane];
					}
					continue;
				}
#endif
				bool valid[8];
				bool allValid = true;
				for (std::uint64_t lane = 0; lane < 8; ++lane)
				{
					valid[lane] = columnBase + lane < static_cast<std::uint64_t>(ctx.outColumns);
					allValid = allValid && valid[lane];
				}
				float acc[8] = {};
				for (std::uint64_t blockIndex = 0; blockIndex < ctx.blockCount; ++blockIndex)
				{
					const auto* packedBlock = ctx.payload + (group * ctx.blockCount + blockIndex) * ctx.groupBlockBytes;
					const auto& lhsBlock = ctx.staged[row * ctx.blockCount + blockIndex];
					if (ctx.format == QuantizedBlockFormat::GGML_Q4_K)
					{
						const auto& block = *reinterpret_cast<const GGMLQ4KFieldInterleaved8Block*>(packedBlock);
#if LITENN_HAS_X86_AVX2_TARGET
						if (allValid && ctx.useAVX2)
						{
							AccumulateGGMLQ4KFieldInterleavedV4BlockQ8Kx8AVX2(block, lhsBlock, acc);
						}
						else
#endif
						{
							AccumulateGGMLQ4KFieldInterleavedV4BlockQ8Kx8(block, valid, lhsBlock, acc);
						}
					}
					else
					{
						const auto& block = *reinterpret_cast<const GGMLQ6KFieldInterleaved8Block*>(packedBlock);
#if LITENN_HAS_X86_AVX2_TARGET
						if (allValid && ctx.useAVX2)
						{
							AccumulateGGMLQ6KFieldInterleavedV4BlockQ8Kx8AVX2(block, lhsBlock, acc);
						}
						else
#endif
						{
							AccumulateGGMLQ6KFieldInterleavedV4BlockQ8Kx8(block, valid, lhsBlock, acc);
						}
					}
				}
				for (std::uint64_t lane = 0; lane < 8; ++lane)
				{
					if (valid[lane])
					{
						ctx.outAligned[ctx.outOffset + static_cast<std::int64_t>(row) * ctx.outRowStride +
						               static_cast<std::int64_t>(columnBase + lane) * ctx.outColumnStride] = acc[lane];
					}
				}
			}
		};
		const auto outputGroups = static_cast<std::uint64_t>(lhsRows) * workItemsPerRow;
		const auto grain = std::max<std::uint64_t>(1, outputGroups / (std::max<std::uint64_t>(1, threadCount) * 4));
		const auto affinityPolicy = ResolveCPUAOTAffinityPolicy(affinityPolicyValue);
		const auto waitPolicy = ResolveCPUAOTWorkerWaitPolicy(affinityPolicyValue);
		LiteNNCPUThreadPool::ParallelForProfile parallelProfile;
		if (threadCount <= 1)
		{
			if (profileEnabled)
			{
				const auto start = std::chrono::steady_clock::now();
				body(0, outputGroups, &context);
				const auto elapsed =
				    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
				parallelProfile.participants = { LiteNNCPUThreadPool::ParticipantProfile{
					.taskClaims = 1, .workUnits = outputGroups, .usefulMilliseconds = elapsed } };
				parallelProfile.wallMilliseconds = elapsed;
			}
			else
			{
				body(0, outputGroups, &context);
			}
			RecordGGMLFieldInterleavedV4ParallelProfile(helper, profileDetail, outputGroups, header->payloadBytes,
			                                            activationProfile, parallelProfile);
			return;
		}
		LiteNNCPUParallelFor(0, outputGroups, grain, body, &context, threadCount, affinityPolicy, waitPolicy,
		                     profileEnabled ? &parallelProfile : nullptr);
		RecordGGMLFieldInterleavedV4ParallelProfile(helper, profileDetail, outputGroups, header->payloadBytes,
		                                            activationProfile, parallelProfile);
	}

	extern "C" void litenn_cpu_swiglu_ggml_block_matmul_field_interleaved_v4_q8k_f32(
	    const float*, const float* gateAligned, std::int64_t gateOffset, std::int64_t gateRows,
	    std::int64_t gateColumns, std::int64_t gateRowStride, std::int64_t gateColumnStride, const float*,
	    const float* upAligned, std::int64_t upOffset, std::int64_t upRows, std::int64_t upColumns,
	    std::int64_t upRowStride, std::int64_t upColumnStride, const std::uint8_t*, const std::uint8_t* rhsAligned,
	    std::int64_t rhsOffset, std::int64_t rhsBytes, std::int64_t rhsStride, float*, float* outAligned,
	    std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride,
	    std::int64_t outColumnStride, std::uint64_t formatValue, std::uint64_t requestedThreadCount,
	    std::uint64_t affinityPolicyValue)
	{
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		const auto layout = GetQuantizedBlockLayout(format);
		if (!layout || (format != QuantizedBlockFormat::GGML_Q4_K && format != QuantizedBlockFormat::GGML_Q6_K) ||
		    !gateAligned || !upAligned || !rhsAligned || !outAligned || gateOffset < 0 || upOffset < 0 ||
		    gateRows <= 0 || gateColumns <= 0 || gateRows != upRows || gateColumns != upColumns ||
		    gateRows != outRows || outColumns <= 0 || gateRowStride <= 0 || gateColumnStride <= 0 || upRowStride <= 0 ||
		    upColumnStride <= 0 || static_cast<std::uint64_t>(gateColumns) % layout->elementsPerBlock != 0)
		{
			return;
		}

		const float* activation = nullptr;
		{
			CPUAOTHelperProfileTimer profileTimer(
			    "litenn_cpu_swiglu_prepare_q8k_activation_f32",
			    CompiledModuleCPUHelperProfilerAccess::Enabled()
			        ? std::format("gate={}x{} up={}x{}", gateRows, gateColumns, upRows, upColumns)
			        : std::string{});
			activation = GetGGMLQ8KActivationThreadCache().PrepareSwiGLU(gateAligned, gateOffset, gateRows, gateColumns,
			                                                             gateRowStride, gateColumnStride, upAligned,
			                                                             upOffset, upRowStride, upColumnStride);
		}
		litenn_cpu_ggml_block_matmul_field_interleaved_v4_q8k_f32(
		    nullptr, activation, 0, gateRows, gateColumns, gateColumns, 1, nullptr, rhsAligned, rhsOffset, rhsBytes,
		    rhsStride, nullptr, outAligned, outOffset, outRows, outColumns, outRowStride, outColumnStride, formatValue,
		    requestedThreadCount, affinityPolicyValue);
	}

	struct GGMLFieldInterleavedV4MatMulProjection
	{
		const std::uint8_t* payload{};
		std::uint64_t outColumns{};
		std::uint64_t outputColumnOffset{};
		std::uint64_t groupOffset{};
		std::uint64_t groupBlockBytes{};
		QuantizedBlockFormat format{};
	};

	void LiteNNCPUGGMLBlockGroupedFieldInterleavedV4Q8KF32(
	    const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows, std::int64_t lhsColumns,
	    std::int64_t lhsRowStride, std::int64_t lhsColumnStride,
	    std::span<const GGMLBlockMatMulProjection> sourceProjections, float* outAligned, std::int64_t outOffset,
	    std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride,
	    std::span<const QuantizedBlockFormat> formats, std::uint64_t requestedThreadCount,
	    std::uint64_t affinityPolicyValue)
	{
		const auto layout = formats.empty() ? std::nullopt : GetQuantizedBlockLayout(formats.front());
		if (!layout || !lhsAligned || !outAligned || lhsRows < 0 || lhsColumns <= 0 || outRows < 0 || outColumns <= 0 ||
		    lhsRows != outRows || sourceProjections.size() < 2 || sourceProjections.size() > 3 ||
		    sourceProjections.size() != formats.size() ||
		    static_cast<std::uint64_t>(lhsColumns) % layout->elementsPerBlock != 0)
		{
			return;
		}

		std::vector<GGMLFieldInterleavedV4MatMulProjection> projections;
		projections.reserve(sourceProjections.size());
		std::uint64_t outputColumnOffset = 0;
		std::uint64_t groupOffset = 0;
		for (std::size_t projectionIndex = 0; projectionIndex < sourceProjections.size(); ++projectionIndex)
		{
			const auto& source = sourceProjections[projectionIndex];
			const auto format = formats[projectionIndex];
			const auto projectionLayout = GetQuantizedBlockLayout(format);
			if (source.outColumns <= 0 ||
			    static_cast<std::uint64_t>(source.outColumns) >
			        static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) - outputColumnOffset ||
			    !projectionLayout || projectionLayout->elementsPerBlock != layout->elementsPerBlock ||
			    (format != QuantizedBlockFormat::GGML_Q4_K && format != QuantizedBlockFormat::GGML_Q6_K))
			{
				return;
			}
			const auto* header = ResolveGGMLFieldInterleavedV4Header(
			    source.rhsAligned, source.rhsOffset, source.rhsBytes, source.rhsStride, format,
			    static_cast<std::uint64_t>(source.outColumns), static_cast<std::uint64_t>(lhsColumns));
			if (!header)
			{
				return;
			}
			const auto width = static_cast<std::uint64_t>(source.outColumns);
			projections.push_back({
			    .payload = reinterpret_cast<const std::uint8_t*>(header + 1),
			    .outColumns = width,
			    .outputColumnOffset = outputColumnOffset,
			    .groupOffset = groupOffset,
			    .groupBlockBytes = header->bytesPerGroupBlock,
			    .format = format,
			});
			outputColumnOffset += width;
			groupOffset += (width + 7) / 8;
		}
		if (outputColumnOffset != static_cast<std::uint64_t>(outColumns))
		{
			return;
		}

		const auto blockCount = static_cast<std::uint64_t>(lhsColumns) / layout->elementsPerBlock;
		std::vector<GGMLQ8KActivationBlock> staged(
		    static_cast<std::size_t>(static_cast<std::uint64_t>(lhsRows) * blockCount));
		for (std::int64_t row = 0; row < lhsRows; ++row)
		{
			const auto* lhsRow = lhsAligned + lhsOffset + row * lhsRowStride;
			for (std::uint64_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
			{
				QuantizeGGMLQ8KActivationBlock(
				    lhsRow + static_cast<std::int64_t>(blockIndex * layout->elementsPerBlock) * lhsColumnStride,
				    lhsColumnStride,
				    staged[static_cast<std::size_t>(static_cast<std::uint64_t>(row) * blockCount + blockIndex)]);
			}
		}

		bool useAVX2 = false;
#if LITENN_HAS_X86_AVX2_TARGET
		useAVX2 = LiteNNCPUHasAVX2F16C();
#endif
		struct Context
		{
			const GGMLFieldInterleavedV4MatMulProjection* projections{};
			std::uint64_t projectionCount{};
			const GGMLQ8KActivationBlock* staged{};
			float* outAligned{};
			std::int64_t outOffset{};
			std::int64_t outRowStride{};
			std::int64_t outColumnStride{};
			std::uint64_t blockCount{};
			std::uint64_t groupsPerRow{};
			bool useAVX2{};
		};
		Context context{
			.projections = projections.data(),
			.projectionCount = projections.size(),
			.staged = staged.data(),
			.outAligned = outAligned,
			.outOffset = outOffset,
			.outRowStride = outRowStride,
			.outColumnStride = outColumnStride,
			.blockCount = blockCount,
			.groupsPerRow = groupOffset,
			.useAVX2 = useAVX2,
		};
		const auto body = [](std::uint64_t begin, std::uint64_t end, void* userData) {
			const auto& ctx = *static_cast<const Context*>(userData);
			for (std::uint64_t outputGroup = begin; outputGroup < end; ++outputGroup)
			{
				const auto row = outputGroup / ctx.groupsPerRow;
				const auto groupWithinRow = outputGroup % ctx.groupsPerRow;
				const GGMLFieldInterleavedV4MatMulProjection* projection = nullptr;
				for (std::uint64_t projectionIndex = 0; projectionIndex < ctx.projectionCount; ++projectionIndex)
				{
					const auto& candidate = ctx.projections[projectionIndex];
					if (groupWithinRow >= candidate.groupOffset &&
					    groupWithinRow < candidate.groupOffset + (candidate.outColumns + 7) / 8)
					{
						projection = &candidate;
						break;
					}
				}
				if (!projection)
				{
					continue;
				}
				const auto projectionGroup = groupWithinRow - projection->groupOffset;
				const auto columnBase = projectionGroup * 8;
				bool valid[8];
				bool allValid = true;
				for (std::uint64_t lane = 0; lane < 8; ++lane)
				{
					valid[lane] = columnBase + lane < projection->outColumns;
					allValid = allValid && valid[lane];
				}
				float acc[8] = {};
				for (std::uint64_t blockIndex = 0; blockIndex < ctx.blockCount; ++blockIndex)
				{
					const auto* packedBlock = projection->payload + (projectionGroup * ctx.blockCount + blockIndex) *
					                                                    projection->groupBlockBytes;
					const auto& lhsBlock = ctx.staged[row * ctx.blockCount + blockIndex];
					if (projection->format == QuantizedBlockFormat::GGML_Q4_K)
					{
						const auto& block = *reinterpret_cast<const GGMLQ4KFieldInterleaved8Block*>(packedBlock);
#if LITENN_HAS_X86_AVX2_TARGET
						if (allValid && ctx.useAVX2)
						{
							AccumulateGGMLQ4KFieldInterleavedV4BlockQ8Kx8AVX2(block, lhsBlock, acc);
						}
						else
#endif
						{
							AccumulateGGMLQ4KFieldInterleavedV4BlockQ8Kx8(block, valid, lhsBlock, acc);
						}
					}
					else
					{
						const auto& block = *reinterpret_cast<const GGMLQ6KFieldInterleaved8Block*>(packedBlock);
#if LITENN_HAS_X86_AVX2_TARGET
						if (allValid && ctx.useAVX2)
						{
							AccumulateGGMLQ6KFieldInterleavedV4BlockQ8Kx8AVX2(block, lhsBlock, acc);
						}
						else
#endif
						{
							AccumulateGGMLQ6KFieldInterleavedV4BlockQ8Kx8(block, valid, lhsBlock, acc);
						}
					}
				}
				for (std::uint64_t lane = 0; lane < 8; ++lane)
				{
					if (valid[lane])
					{
						const auto outputColumn = projection->outputColumnOffset + columnBase + lane;
						ctx.outAligned[ctx.outOffset + static_cast<std::int64_t>(row) * ctx.outRowStride +
						               static_cast<std::int64_t>(outputColumn) * ctx.outColumnStride] = acc[lane];
					}
				}
			}
		};
		const auto outputGroups = static_cast<std::uint64_t>(lhsRows) * groupOffset;
		const auto schedulingFormat = std::ranges::contains(formats, QuantizedBlockFormat::GGML_Q6_K)
		                                  ? QuantizedBlockFormat::GGML_Q6_K
		                                  : formats.front();
		const auto threadCount = ResolveGGMLFieldInterleavedV4ThreadCount(
		    schedulingFormat, lhsRows, lhsColumns, outColumns, outputGroups, requestedThreadCount, true);
		const auto grain = std::max<std::uint64_t>(1, outputGroups / (std::max<std::uint64_t>(1, threadCount) * 4));
		const auto affinityPolicy = ResolveCPUAOTAffinityPolicy(affinityPolicyValue);
		const auto waitPolicy = ResolveCPUAOTWorkerWaitPolicy(affinityPolicyValue);
		if (threadCount <= 1)
		{
			body(0, outputGroups, &context);
			return;
		}
		LiteNNCPUParallelFor(0, outputGroups, grain, body, &context, threadCount, affinityPolicy, waitPolicy);
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul2_field_interleaved_v4_q8k_f32(
	    const float*, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows, std::int64_t lhsColumns,
	    std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t*, const std::uint8_t* rhs0Aligned,
	    std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride, const std::uint8_t*,
	    const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes, std::int64_t rhs1Stride,
	    float*, float* outAligned, std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns,
	    std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t formatValue, std::uint64_t out0Columns,
	    std::uint64_t out1Columns, std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
		{
			return;
		}
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		const std::array formats{ format, format };
		const auto profileOutputUnits = static_cast<std::uint64_t>(std::max<std::int64_t>(0, lhsRows)) *
		                                ((static_cast<std::uint64_t>(std::max<std::int64_t>(0, outColumns)) + 7) / 8);
		const auto profileThreadCount = ResolveGGMLFieldInterleavedV4ThreadCount(
		    format, lhsRows, lhsColumns, outColumns, profileOutputUnits, requestedThreadCount, true);
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_grouped_matmul2_field_interleaved_v4_q8k_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLBlockMatMulProfileDetail(format, GGMLActivationDotMode::Q8KStaged, lhsRows, lhsColumns,
		                                            outRows, outColumns, requestedThreadCount, profileThreadCount)
		        : std::string{});
		const std::array projections{
			GGMLBlockMatMulProjection{ .rhsAligned = rhs0Aligned,
			                           .rhsOffset = rhs0Offset,
			                           .rhsBytes = rhs0Bytes,
			                           .rhsStride = rhs0Stride,
			                           .outColumns = static_cast<std::int64_t>(out0Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs1Aligned,
			                           .rhsOffset = rhs1Offset,
			                           .rhsBytes = rhs1Bytes,
			                           .rhsStride = rhs1Stride,
			                           .outColumns = static_cast<std::int64_t>(out1Columns) },
		};
		LiteNNCPUGGMLBlockGroupedFieldInterleavedV4Q8KF32(lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride,
		                                                  lhsColumnStride, projections, outAligned, outOffset, outRows,
		                                                  outColumns, outRowStride, outColumnStride, formats,
		                                                  requestedThreadCount, affinityPolicyValue);
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul3_field_interleaved_v4_q8k_f32(
	    const float*, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows, std::int64_t lhsColumns,
	    std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t*, const std::uint8_t* rhs0Aligned,
	    std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride, const std::uint8_t*,
	    const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes, std::int64_t rhs1Stride,
	    const std::uint8_t*, const std::uint8_t* rhs2Aligned, std::int64_t rhs2Offset, std::int64_t rhs2Bytes,
	    std::int64_t rhs2Stride, float*, float* outAligned, std::int64_t outOffset, std::int64_t outRows,
	    std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t formatValue,
	    std::uint64_t out0Columns, std::uint64_t out1Columns, std::uint64_t out2Columns,
	    std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out2Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
		{
			return;
		}
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		const std::array formats{ format, format, format };
		const auto profileOutputUnits = static_cast<std::uint64_t>(std::max<std::int64_t>(0, lhsRows)) *
		                                ((static_cast<std::uint64_t>(std::max<std::int64_t>(0, outColumns)) + 7) / 8);
		const auto profileThreadCount = ResolveGGMLFieldInterleavedV4ThreadCount(
		    format, lhsRows, lhsColumns, outColumns, profileOutputUnits, requestedThreadCount, true);
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_grouped_matmul3_field_interleaved_v4_q8k_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLBlockMatMulProfileDetail(format, GGMLActivationDotMode::Q8KStaged, lhsRows, lhsColumns,
		                                            outRows, outColumns, requestedThreadCount, profileThreadCount)
		        : std::string{});
		const std::array projections{
			GGMLBlockMatMulProjection{ .rhsAligned = rhs0Aligned,
			                           .rhsOffset = rhs0Offset,
			                           .rhsBytes = rhs0Bytes,
			                           .rhsStride = rhs0Stride,
			                           .outColumns = static_cast<std::int64_t>(out0Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs1Aligned,
			                           .rhsOffset = rhs1Offset,
			                           .rhsBytes = rhs1Bytes,
			                           .rhsStride = rhs1Stride,
			                           .outColumns = static_cast<std::int64_t>(out1Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs2Aligned,
			                           .rhsOffset = rhs2Offset,
			                           .rhsBytes = rhs2Bytes,
			                           .rhsStride = rhs2Stride,
			                           .outColumns = static_cast<std::int64_t>(out2Columns) },
		};
		LiteNNCPUGGMLBlockGroupedFieldInterleavedV4Q8KF32(lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride,
		                                                  lhsColumnStride, projections, outAligned, outOffset, outRows,
		                                                  outColumns, outRowStride, outColumnStride, formats,
		                                                  requestedThreadCount, affinityPolicyValue);
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul2_mixed_field_interleaved_v4_q8k_f32(
	    const float*, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows, std::int64_t lhsColumns,
	    std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t*, const std::uint8_t* rhs0Aligned,
	    std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride, const std::uint8_t*,
	    const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes, std::int64_t rhs1Stride,
	    float*, float* outAligned, std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns,
	    std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t format0Value, std::uint64_t format1Value,
	    std::uint64_t out0Columns, std::uint64_t out1Columns, std::uint64_t requestedThreadCount,
	    std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
		{
			return;
		}
		const std::array formats{ static_cast<QuantizedBlockFormat>(format0Value),
			                      static_cast<QuantizedBlockFormat>(format1Value) };
		const std::array projections{
			GGMLBlockMatMulProjection{ .rhsAligned = rhs0Aligned,
			                           .rhsOffset = rhs0Offset,
			                           .rhsBytes = rhs0Bytes,
			                           .rhsStride = rhs0Stride,
			                           .outColumns = static_cast<std::int64_t>(out0Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs1Aligned,
			                           .rhsOffset = rhs1Offset,
			                           .rhsBytes = rhs1Bytes,
			                           .rhsStride = rhs1Stride,
			                           .outColumns = static_cast<std::int64_t>(out1Columns) },
		};
		const auto profileFormat = std::ranges::contains(formats, QuantizedBlockFormat::GGML_Q6_K)
		                               ? QuantizedBlockFormat::GGML_Q6_K
		                               : formats.front();
		const auto profileOutputUnits = static_cast<std::uint64_t>(std::max<std::int64_t>(0, lhsRows)) *
		                                ((static_cast<std::uint64_t>(std::max<std::int64_t>(0, outColumns)) + 7) / 8);
		const auto profileThreadCount = ResolveGGMLFieldInterleavedV4ThreadCount(
		    profileFormat, lhsRows, lhsColumns, outColumns, profileOutputUnits, requestedThreadCount, true);
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_grouped_matmul2_mixed_field_interleaved_v4_q8k_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLBlockMatMulProfileDetail(profileFormat, GGMLActivationDotMode::Q8KStaged, lhsRows,
		                                            lhsColumns, outRows, outColumns, requestedThreadCount,
		                                            profileThreadCount)
		        : std::string{});
		LiteNNCPUGGMLBlockGroupedFieldInterleavedV4Q8KF32(lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride,
		                                                  lhsColumnStride, projections, outAligned, outOffset, outRows,
		                                                  outColumns, outRowStride, outColumnStride, formats,
		                                                  requestedThreadCount, affinityPolicyValue);
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul3_mixed_field_interleaved_v4_q8k_f32(
	    const float*, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows, std::int64_t lhsColumns,
	    std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t*, const std::uint8_t* rhs0Aligned,
	    std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride, const std::uint8_t*,
	    const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes, std::int64_t rhs1Stride,
	    const std::uint8_t*, const std::uint8_t* rhs2Aligned, std::int64_t rhs2Offset, std::int64_t rhs2Bytes,
	    std::int64_t rhs2Stride, float*, float* outAligned, std::int64_t outOffset, std::int64_t outRows,
	    std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t format0Value,
	    std::uint64_t format1Value, std::uint64_t format2Value, std::uint64_t out0Columns, std::uint64_t out1Columns,
	    std::uint64_t out2Columns, std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out2Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
		{
			return;
		}
		const std::array formats{ static_cast<QuantizedBlockFormat>(format0Value),
			                      static_cast<QuantizedBlockFormat>(format1Value),
			                      static_cast<QuantizedBlockFormat>(format2Value) };
		const std::array projections{
			GGMLBlockMatMulProjection{ .rhsAligned = rhs0Aligned,
			                           .rhsOffset = rhs0Offset,
			                           .rhsBytes = rhs0Bytes,
			                           .rhsStride = rhs0Stride,
			                           .outColumns = static_cast<std::int64_t>(out0Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs1Aligned,
			                           .rhsOffset = rhs1Offset,
			                           .rhsBytes = rhs1Bytes,
			                           .rhsStride = rhs1Stride,
			                           .outColumns = static_cast<std::int64_t>(out1Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs2Aligned,
			                           .rhsOffset = rhs2Offset,
			                           .rhsBytes = rhs2Bytes,
			                           .rhsStride = rhs2Stride,
			                           .outColumns = static_cast<std::int64_t>(out2Columns) },
		};
		const auto profileFormat = std::ranges::contains(formats, QuantizedBlockFormat::GGML_Q6_K)
		                               ? QuantizedBlockFormat::GGML_Q6_K
		                               : formats.front();
		const auto profileOutputUnits = static_cast<std::uint64_t>(std::max<std::int64_t>(0, lhsRows)) *
		                                ((static_cast<std::uint64_t>(std::max<std::int64_t>(0, outColumns)) + 7) / 8);
		const auto profileThreadCount = ResolveGGMLFieldInterleavedV4ThreadCount(
		    profileFormat, lhsRows, lhsColumns, outColumns, profileOutputUnits, requestedThreadCount, true);
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_grouped_matmul3_mixed_field_interleaved_v4_q8k_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLBlockMatMulProfileDetail(profileFormat, GGMLActivationDotMode::Q8KStaged, lhsRows,
		                                            lhsColumns, outRows, outColumns, requestedThreadCount,
		                                            profileThreadCount)
		        : std::string{});
		LiteNNCPUGGMLBlockGroupedFieldInterleavedV4Q8KF32(lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride,
		                                                  lhsColumnStride, projections, outAligned, outOffset, outRows,
		                                                  outColumns, outRowStride, outColumnStride, formats,
		                                                  requestedThreadCount, affinityPolicyValue);
	}

	void AccumulateGGMLCompactBlockQ8Kx4(QuantizedBlockFormat format, const std::uint8_t* const blocks[4],
	                                     const bool valid[4], const GGMLQ8KActivationBlock& lhsBlock, float acc[4])
	{
		if (format == QuantizedBlockFormat::GGML_Q4_K)
		{
#if LITENN_HAS_X86_AVX2_TARGET
			if (LiteNNCPUHasAVX2())
			{
				if (valid[0] && valid[1] && valid[2] && valid[3])
				{
					AccumulateGGMLQ4KBlockQ8Kx4AVX2ContiguousAllValid(blocks, lhsBlock, acc);
				}
				else
				{
					AccumulateGGMLQ4KBlockQ8Kx4AVX2(blocks, valid, 1, lhsBlock, acc);
				}
				return;
			}
#endif
			AccumulateGGMLQ4KBlockQ8Kx4(blocks, valid, 1, lhsBlock, acc);
			return;
		}
#if LITENN_HAS_X86_AVX2_TARGET
		if (LiteNNCPUHasAVX2())
		{
			if (valid[0] && valid[1] && valid[2] && valid[3])
			{
				AccumulateGGMLQ6KBlockQ8Kx4AVX2ContiguousAllValid(blocks, lhsBlock, acc);
			}
			else
			{
				AccumulateGGMLQ6KBlockQ8Kx4AVX2(blocks, valid, 1, lhsBlock, acc);
			}
			return;
		}
#endif
		AccumulateGGMLQ6KBlockQ8Kx4(blocks, valid, 1, lhsBlock, acc);
	}

	extern "C" void litenn_cpu_ggml_block_matmul_compact_q8k_f32(
	    const float*, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows, std::int64_t lhsColumns,
	    std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t*, const std::uint8_t* rhsAligned,
	    std::int64_t rhsOffset, std::int64_t rhsBytes, std::int64_t rhsStride, float*, float* outAligned,
	    std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride,
	    std::int64_t outColumnStride, std::uint64_t formatValue, std::uint64_t requestedThreadCount,
	    std::uint64_t affinityPolicyValue)
	{
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		const auto layout = GetQuantizedBlockLayout(format);
		if (!layout || !lhsAligned || !outAligned || lhsRows < 0 || lhsColumns <= 0 || outRows < 0 || outColumns <= 0 ||
		    lhsRows != outRows ||
		    (format != QuantizedBlockFormat::GGML_Q4_K && format != QuantizedBlockFormat::GGML_Q6_K) ||
		    static_cast<std::uint64_t>(lhsColumns) % layout->elementsPerBlock != 0)
		{
			return;
		}
		const auto* header = ResolveGGMLCompactInterleavedHeader(rhsAligned, rhsOffset, rhsBytes, rhsStride, format,
		                                                         static_cast<std::uint64_t>(outColumns),
		                                                         static_cast<std::uint64_t>(lhsColumns));
		if (!header)
		{
			return;
		}

		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_matmul_compact_q8k_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLBlockMatMulProfileDetail(format, GGMLActivationDotMode::Q8KStaged, lhsRows, lhsColumns,
		                                            outRows, outColumns, requestedThreadCount)
		        : std::string{});
		const auto blockCount = header->blockCount;
		std::vector<GGMLQ8KActivationBlock> staged(
		    static_cast<std::size_t>(static_cast<std::uint64_t>(lhsRows) * blockCount));
		for (std::int64_t row = 0; row < lhsRows; ++row)
		{
			const auto* lhsRow = lhsAligned + lhsOffset + row * lhsRowStride;
			for (std::uint64_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
			{
				QuantizeGGMLQ8KActivationBlock(
				    lhsRow + static_cast<std::int64_t>(blockIndex * layout->elementsPerBlock) * lhsColumnStride,
				    lhsColumnStride,
				    staged[static_cast<std::size_t>(static_cast<std::uint64_t>(row) * blockCount + blockIndex)]);
			}
		}

		struct Context
		{
			const std::uint8_t* payload{};
			const GGMLQ8KActivationBlock* staged{};
			float* outAligned{};
			std::int64_t outOffset{};
			std::int64_t outColumns{};
			std::int64_t outRowStride{};
			std::int64_t outColumnStride{};
			std::uint64_t blockCount{};
			std::uint64_t bytesPerBlock{};
			std::uint64_t columnGroupsPerRow{};
			QuantizedBlockFormat format{};
		};
		const auto columnGroupsPerRow = (static_cast<std::uint64_t>(outColumns) + 3) / 4;
		Context context{
			.payload = reinterpret_cast<const std::uint8_t*>(header + 1),
			.staged = staged.data(),
			.outAligned = outAligned,
			.outOffset = outOffset,
			.outColumns = outColumns,
			.outRowStride = outRowStride,
			.outColumnStride = outColumnStride,
			.blockCount = blockCount,
			.bytesPerBlock = layout->bytesPerBlock,
			.columnGroupsPerRow = columnGroupsPerRow,
			.format = format,
		};
		const auto body = [](std::uint64_t begin, std::uint64_t end, void* userData) {
			const auto& ctx = *static_cast<const Context*>(userData);
			for (std::uint64_t outputGroup = begin; outputGroup < end; ++outputGroup)
			{
				const auto row = outputGroup / ctx.columnGroupsPerRow;
				const auto columnGroup = outputGroup % ctx.columnGroupsPerRow;
				const auto columnBase = columnGroup * 4;
				const bool valid[4] = {
					columnBase < static_cast<std::uint64_t>(ctx.outColumns),
					columnBase + 1 < static_cast<std::uint64_t>(ctx.outColumns),
					columnBase + 2 < static_cast<std::uint64_t>(ctx.outColumns),
					columnBase + 3 < static_cast<std::uint64_t>(ctx.outColumns),
				};
				float acc[4] = {};
				for (std::uint64_t blockIndex = 0; blockIndex < ctx.blockCount; ++blockIndex)
				{
					const auto* interleavedBlock =
					    ctx.payload + (columnGroup * ctx.blockCount + blockIndex) * ctx.bytesPerBlock * 4;
					const std::uint8_t* blocks[4] = {
						interleavedBlock,
						interleavedBlock + ctx.bytesPerBlock,
						interleavedBlock + ctx.bytesPerBlock * 2,
						interleavedBlock + ctx.bytesPerBlock * 3,
					};
					const auto& lhsBlock = ctx.staged[row * ctx.blockCount + blockIndex];
					AccumulateGGMLCompactBlockQ8Kx4(ctx.format, blocks, valid, lhsBlock, acc);
				}
				for (std::uint64_t lane = 0; lane < 4; ++lane)
				{
					if (valid[lane])
					{
						ctx.outAligned[ctx.outOffset + static_cast<std::int64_t>(row) * ctx.outRowStride +
						               static_cast<std::int64_t>(columnBase + lane) * ctx.outColumnStride] = acc[lane];
					}
				}
			}
		};
		const auto outputGroups = static_cast<std::uint64_t>(lhsRows) * columnGroupsPerRow;
		const auto operations = static_cast<std::uint64_t>(lhsRows) * static_cast<std::uint64_t>(lhsColumns) *
		                        static_cast<std::uint64_t>(outColumns);
		const auto threadCount = ResolveGGMLBlockMatMulThreadCount(format, GGMLActivationDotMode::Q8KStaged, operations,
		                                                           outputGroups, requestedThreadCount);
		const auto grain = std::max<std::uint64_t>(1, outputGroups / (std::max<std::uint64_t>(1, threadCount) * 8));
		const auto affinityPolicy = ResolveCPUAOTAffinityPolicy(affinityPolicyValue);
		const auto waitPolicy = ResolveCPUAOTWorkerWaitPolicy(affinityPolicyValue);
		if (threadCount <= 1)
		{
			body(0, outputGroups, &context);
			return;
		}
		LiteNNCPUParallelFor(0, outputGroups, grain, body, &context, threadCount, affinityPolicy, waitPolicy);
	}

	struct GGMLCompactMatMulProjection
	{
		const std::uint8_t* payload{};
		std::uint64_t outColumns{};
	};

	void LiteNNCPUGGMLBlockGroupedCompactQ8KF32(const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows,
	                                            std::int64_t lhsColumns, std::int64_t lhsRowStride,
	                                            std::int64_t lhsColumnStride,
	                                            std::span<const GGMLBlockMatMulProjection> sourceProjections,
	                                            float* outAligned, std::int64_t outOffset, std::int64_t outRows,
	                                            std::int64_t outColumns, std::int64_t outRowStride,
	                                            std::int64_t outColumnStride, std::uint64_t formatValue,
	                                            std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		const auto layout = GetQuantizedBlockLayout(format);
		if (!layout || !lhsAligned || !outAligned || lhsRows < 0 || lhsColumns <= 0 || outRows < 0 || outColumns <= 0 ||
		    lhsRows != outRows || sourceProjections.size() < 2 || sourceProjections.size() > 3 ||
		    (format != QuantizedBlockFormat::GGML_Q4_K && format != QuantizedBlockFormat::GGML_Q6_K) ||
		    static_cast<std::uint64_t>(lhsColumns) % layout->elementsPerBlock != 0)
		{
			return;
		}

		std::vector<GGMLCompactMatMulProjection> projections;
		projections.reserve(sourceProjections.size());
		std::int64_t totalOutputColumns = 0;
		for (const auto& source : sourceProjections)
		{
			if (source.outColumns <= 0 ||
			    source.outColumns > std::numeric_limits<std::int64_t>::max() - totalOutputColumns)
			{
				return;
			}
			const auto* header = ResolveGGMLCompactInterleavedHeader(
			    source.rhsAligned, source.rhsOffset, source.rhsBytes, source.rhsStride, format,
			    static_cast<std::uint64_t>(source.outColumns), static_cast<std::uint64_t>(lhsColumns));
			if (!header)
			{
				return;
			}
			projections.push_back(
			    { reinterpret_cast<const std::uint8_t*>(header + 1), static_cast<std::uint64_t>(source.outColumns) });
			totalOutputColumns += source.outColumns;
		}
		if (totalOutputColumns != outColumns)
		{
			return;
		}

		const auto blockCount = static_cast<std::uint64_t>(lhsColumns) / layout->elementsPerBlock;
		std::vector<GGMLQ8KActivationBlock> staged(
		    static_cast<std::size_t>(static_cast<std::uint64_t>(lhsRows) * blockCount));
		for (std::int64_t row = 0; row < lhsRows; ++row)
		{
			const auto* lhsRow = lhsAligned + lhsOffset + row * lhsRowStride;
			for (std::uint64_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
			{
				QuantizeGGMLQ8KActivationBlock(
				    lhsRow + static_cast<std::int64_t>(blockIndex * layout->elementsPerBlock) * lhsColumnStride,
				    lhsColumnStride,
				    staged[static_cast<std::size_t>(static_cast<std::uint64_t>(row) * blockCount + blockIndex)]);
			}
		}

		struct Context
		{
			const GGMLCompactMatMulProjection* projections{};
			std::uint64_t projectionCount{};
			const GGMLQ8KActivationBlock* staged{};
			float* outAligned{};
			std::int64_t outOffset{};
			std::int64_t outColumns{};
			std::int64_t outRowStride{};
			std::int64_t outColumnStride{};
			std::uint64_t blockCount{};
			std::uint64_t bytesPerBlock{};
			std::uint64_t columnGroupsPerRow{};
			QuantizedBlockFormat format{};
		};
		const auto columnGroupsPerRow = (static_cast<std::uint64_t>(outColumns) + 3) / 4;
		Context context{
			.projections = projections.data(),
			.projectionCount = projections.size(),
			.staged = staged.data(),
			.outAligned = outAligned,
			.outOffset = outOffset,
			.outColumns = outColumns,
			.outRowStride = outRowStride,
			.outColumnStride = outColumnStride,
			.blockCount = blockCount,
			.bytesPerBlock = layout->bytesPerBlock,
			.columnGroupsPerRow = columnGroupsPerRow,
			.format = format,
		};
		const auto body = [](std::uint64_t begin, std::uint64_t end, void* userData) {
			const auto& ctx = *static_cast<const Context*>(userData);
			for (std::uint64_t outputGroup = begin; outputGroup < end; ++outputGroup)
			{
				const auto row = outputGroup / ctx.columnGroupsPerRow;
				const auto columnBase = (outputGroup % ctx.columnGroupsPerRow) * 4;
				float acc[4] = {};
				bool valid[4] = {
					columnBase < static_cast<std::uint64_t>(ctx.outColumns),
					columnBase + 1 < static_cast<std::uint64_t>(ctx.outColumns),
					columnBase + 2 < static_cast<std::uint64_t>(ctx.outColumns),
					columnBase + 3 < static_cast<std::uint64_t>(ctx.outColumns),
				};
				for (std::uint64_t blockIndex = 0; blockIndex < ctx.blockCount; ++blockIndex)
				{
					const std::uint8_t* blocks[4] = {};
					for (std::uint64_t lane = 0; lane < 4; ++lane)
					{
						if (!valid[lane])
						{
							continue;
						}
						auto localColumn = columnBase + lane;
						for (std::uint64_t projectionIndex = 0; projectionIndex < ctx.projectionCount;
						     ++projectionIndex)
						{
							const auto& projection = ctx.projections[projectionIndex];
							if (localColumn < projection.outColumns)
							{
								const auto projectionGroup = localColumn / 4;
								const auto projectionLane = localColumn % 4;
								blocks[lane] = projection.payload +
								               (projectionGroup * ctx.blockCount + blockIndex) * ctx.bytesPerBlock * 4 +
								               projectionLane * ctx.bytesPerBlock;
								break;
							}
							localColumn -= projection.outColumns;
						}
						if (!blocks[lane])
						{
							valid[lane] = false;
						}
					}
					const auto& lhsBlock = ctx.staged[row * ctx.blockCount + blockIndex];
					AccumulateGGMLCompactBlockQ8Kx4(ctx.format, blocks, valid, lhsBlock, acc);
				}
				for (std::uint64_t lane = 0; lane < 4; ++lane)
				{
					if (valid[lane])
					{
						ctx.outAligned[ctx.outOffset + static_cast<std::int64_t>(row) * ctx.outRowStride +
						               static_cast<std::int64_t>(columnBase + lane) * ctx.outColumnStride] = acc[lane];
					}
				}
			}
		};
		const auto outputGroups = static_cast<std::uint64_t>(lhsRows) * columnGroupsPerRow;
		const auto operations = static_cast<std::uint64_t>(lhsRows) * static_cast<std::uint64_t>(lhsColumns) *
		                        static_cast<std::uint64_t>(outColumns);
		const auto threadCount = ResolveGGMLBlockMatMulThreadCount(format, GGMLActivationDotMode::Q8KStaged, operations,
		                                                           outputGroups, requestedThreadCount);
		const auto grain = std::max<std::uint64_t>(1, outputGroups / (std::max<std::uint64_t>(1, threadCount) * 8));
		const auto affinityPolicy = ResolveCPUAOTAffinityPolicy(affinityPolicyValue);
		const auto waitPolicy = ResolveCPUAOTWorkerWaitPolicy(affinityPolicyValue);
		if (threadCount <= 1)
		{
			body(0, outputGroups, &context);
			return;
		}
		LiteNNCPUParallelFor(0, outputGroups, grain, body, &context, threadCount, affinityPolicy, waitPolicy);
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul2_compact_q8k_f32(
	    const float*, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows, std::int64_t lhsColumns,
	    std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t*, const std::uint8_t* rhs0Aligned,
	    std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride, const std::uint8_t*,
	    const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes, std::int64_t rhs1Stride,
	    float*, float* outAligned, std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns,
	    std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t formatValue, std::uint64_t out0Columns,
	    std::uint64_t out1Columns, std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
		{
			return;
		}
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_grouped_matmul2_compact_q8k_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLBlockMatMulProfileDetail(format, GGMLActivationDotMode::Q8KStaged, lhsRows, lhsColumns,
		                                            outRows, outColumns, requestedThreadCount)
		        : std::string{});
		const std::array projections{
			GGMLBlockMatMulProjection{ .rhsAligned = rhs0Aligned,
			                           .rhsOffset = rhs0Offset,
			                           .rhsBytes = rhs0Bytes,
			                           .rhsStride = rhs0Stride,
			                           .outColumns = static_cast<std::int64_t>(out0Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs1Aligned,
			                           .rhsOffset = rhs1Offset,
			                           .rhsBytes = rhs1Bytes,
			                           .rhsStride = rhs1Stride,
			                           .outColumns = static_cast<std::int64_t>(out1Columns) },
		};
		LiteNNCPUGGMLBlockGroupedCompactQ8KF32(lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride,
		                                       lhsColumnStride, projections, outAligned, outOffset, outRows, outColumns,
		                                       outRowStride, outColumnStride, formatValue, requestedThreadCount,
		                                       affinityPolicyValue);
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul3_compact_q8k_f32(
	    const float*, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows, std::int64_t lhsColumns,
	    std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t*, const std::uint8_t* rhs0Aligned,
	    std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride, const std::uint8_t*,
	    const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes, std::int64_t rhs1Stride,
	    const std::uint8_t*, const std::uint8_t* rhs2Aligned, std::int64_t rhs2Offset, std::int64_t rhs2Bytes,
	    std::int64_t rhs2Stride, float*, float* outAligned, std::int64_t outOffset, std::int64_t outRows,
	    std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t formatValue,
	    std::uint64_t out0Columns, std::uint64_t out1Columns, std::uint64_t out2Columns,
	    std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out2Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
		{
			return;
		}
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_grouped_matmul3_compact_q8k_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLBlockMatMulProfileDetail(format, GGMLActivationDotMode::Q8KStaged, lhsRows, lhsColumns,
		                                            outRows, outColumns, requestedThreadCount)
		        : std::string{});
		const std::array projections{
			GGMLBlockMatMulProjection{ .rhsAligned = rhs0Aligned,
			                           .rhsOffset = rhs0Offset,
			                           .rhsBytes = rhs0Bytes,
			                           .rhsStride = rhs0Stride,
			                           .outColumns = static_cast<std::int64_t>(out0Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs1Aligned,
			                           .rhsOffset = rhs1Offset,
			                           .rhsBytes = rhs1Bytes,
			                           .rhsStride = rhs1Stride,
			                           .outColumns = static_cast<std::int64_t>(out1Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs2Aligned,
			                           .rhsOffset = rhs2Offset,
			                           .rhsBytes = rhs2Bytes,
			                           .rhsStride = rhs2Stride,
			                           .outColumns = static_cast<std::int64_t>(out2Columns) },
		};
		LiteNNCPUGGMLBlockGroupedCompactQ8KF32(lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride,
		                                       lhsColumnStride, projections, outAligned, outOffset, outRows, outColumns,
		                                       outRowStride, outColumnStride, formatValue, requestedThreadCount,
		                                       affinityPolicyValue);
	}

	const GGMLQ8KActivationBlock* ResolveGGMLQ8KActivationBlocks(QuantizedBlockFormat format,
	                                                             const std::uint8_t* lhsQ8KAligned,
	                                                             std::int64_t lhsQ8KOffset, std::int64_t lhsQ8KBytes,
	                                                             std::int64_t lhsQ8KStride, std::int64_t lhsRows,
	                                                             std::int64_t lhsColumns)
	{
		const auto layout = GetQuantizedBlockLayout(format);
		if (!layout || !IsGGMLQ8KStagedMatMulFormat(format) || !lhsQ8KAligned || lhsQ8KOffset < 0 || lhsQ8KBytes < 0 ||
		    lhsQ8KStride != 1 || lhsRows < 0 || lhsColumns <= 0 ||
		    static_cast<std::uint64_t>(lhsColumns) % layout->elementsPerBlock != 0)
		{
			return nullptr;
		}
		const auto blockCount = static_cast<std::uint64_t>(lhsColumns) / layout->elementsPerBlock;
		const auto requiredBytes = static_cast<std::uint64_t>(lhsRows) * blockCount * kGGMLQ8KActivationBlockBytes;
		const auto offsetBytes = static_cast<std::uint64_t>(lhsQ8KOffset);
		const auto availableBytes = static_cast<std::uint64_t>(lhsQ8KBytes);
		if (offsetBytes > availableBytes || availableBytes - offsetBytes < requiredBytes)
		{
			return nullptr;
		}
		return reinterpret_cast<const GGMLQ8KActivationBlock*>(lhsQ8KAligned + lhsQ8KOffset);
	}

	extern "C" void litenn_cpu_ggml_block_matmul_q8k_prepared_activation_f32(
	    const std::uint8_t*, const std::uint8_t* lhsQ8KAligned, std::int64_t lhsQ8KOffset, std::int64_t lhsQ8KBytes,
	    std::int64_t lhsQ8KStride, std::int64_t lhsRows, std::int64_t lhsColumns, const std::uint8_t*,
	    const std::uint8_t* rhsAligned, std::int64_t rhsOffset, std::int64_t rhsBytes, std::int64_t rhsStride, float*,
	    float* outAligned, std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns,
	    std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t formatValue,
	    std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_matmul_q8k_prepared_activation_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLBlockMatMulProfileDetail(format, GGMLActivationDotMode::Q8KStaged, lhsRows, lhsColumns,
		                                            outRows, outColumns, requestedThreadCount)
		        : std::string{});
		const auto* preparedBlocks = ResolveGGMLQ8KActivationBlocks(format, lhsQ8KAligned, lhsQ8KOffset, lhsQ8KBytes,
		                                                            lhsQ8KStride, lhsRows, lhsColumns);
		if (!preparedBlocks)
		{
			return;
		}
		const GGMLBlockMatMulProjection projection{
			.rhsAligned = rhsAligned,
			.rhsOffset = rhsOffset,
			.rhsBytes = rhsBytes,
			.rhsStride = rhsStride,
			.outColumns = outColumns,
		};
		LiteNNCPUGGMLBlockMatMulProjectedF32(nullptr, nullptr, 0, lhsRows, lhsColumns, 0, 0,
		                                     std::span<const GGMLBlockMatMulProjection>{ &projection, 1 }, nullptr,
		                                     outAligned, outOffset, outRows, outColumns, outRowStride, outColumnStride,
		                                     formatValue, requestedThreadCount, affinityPolicyValue,
		                                     GGMLActivationDotMode::Q8KStaged, preparedBlocks);
	}

	extern "C" std::uint64_t litenn_cpu_ggml_q4k_prepacked_block_bytes()
	{
		return kGGMLQ4KPreparedBlockBytes;
	}

	extern "C" void litenn_cpu_ggml_prepack_q4k_f32(const std::uint8_t*, const std::uint8_t* rhsAligned,
	                                                std::int64_t rhsOffset, std::int64_t rhsBytes,
	                                                std::int64_t rhsStride, std::int64_t rows, std::int64_t columns,
	                                                std::uint8_t*, std::uint8_t* packedAligned,
	                                                std::int64_t packedOffset, std::int64_t packedBytes,
	                                                std::int64_t packedStride)
	{
		const auto layout = GetQuantizedBlockLayout(QuantizedBlockFormat::GGML_Q4_K);
		if (!layout || !rhsAligned || !packedAligned || rhsOffset < 0 || rhsBytes < 0 || rhsStride <= 0 || rows < 0 ||
		    columns < 0 || packedOffset < 0 || packedBytes < 0 || packedStride != 1 ||
		    static_cast<std::uint64_t>(columns) % layout->elementsPerBlock != 0)
		{
			return;
		}
		const auto blockCount = static_cast<std::uint64_t>(columns) / layout->elementsPerBlock;
		const auto rowBytes = blockCount * layout->bytesPerBlock;
		const auto packedRowBytes = blockCount * kGGMLQ4KPreparedBlockBytes;
		if (static_cast<std::uint64_t>(rhsBytes) < static_cast<std::uint64_t>(rows) * rowBytes ||
		    static_cast<std::uint64_t>(packedBytes) < static_cast<std::uint64_t>(rows) * packedRowBytes)
		{
			return;
		}
		for (std::int64_t row = 0; row < rows; ++row)
		{
			for (std::uint64_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
			{
				const auto* sourceBlock =
				    rhsAligned + rhsOffset +
				    (static_cast<std::uint64_t>(row) * rowBytes + blockIndex * layout->bytesPerBlock) *
				        static_cast<std::uint64_t>(rhsStride);
				auto* targetBlock = packedAligned + packedOffset + static_cast<std::uint64_t>(row) * packedRowBytes +
				                    blockIndex * kGGMLQ4KPreparedBlockBytes;
				PrepareGGMLQ4KBlockF32(sourceBlock, rhsStride, targetBlock);
			}
		}
	}

	extern "C" void litenn_cpu_ggml_block_matmul_q4k_prepacked_f32(
	    const float*, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows, std::int64_t lhsColumns,
	    std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t*, const std::uint8_t* rhsAligned,
	    std::int64_t rhsOffset, std::int64_t rhsBytes, std::int64_t rhsStride, float*, float* outAligned,
	    std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride,
	    std::int64_t outColumnStride, std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_matmul_q4k_prepacked_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? std::format("format=GGML_Q4_K activation=direct_prepacked lhs={}x{} out={}x{} requested_threads={}",
		                      lhsRows, lhsColumns, outRows, outColumns, requestedThreadCount)
		        : std::string{});
		const auto layout = GetQuantizedBlockLayout(QuantizedBlockFormat::GGML_Q4_K);
		if (!layout || !lhsAligned || !rhsAligned || !outAligned || lhsRows < 0 || lhsColumns < 0 || outRows < 0 ||
		    outColumns < 0 || lhsRows != outRows || lhsColumns == 0 || outColumns == 0 || rhsOffset < 0 ||
		    rhsBytes < 0 || rhsStride != 1 || static_cast<std::uint64_t>(lhsColumns) % layout->elementsPerBlock != 0)
		{
			return;
		}
		const auto blockCount = static_cast<std::uint64_t>(lhsColumns) / layout->elementsPerBlock;
		const auto rowBytes = blockCount * kGGMLQ4KPreparedBlockBytes;
		if (static_cast<std::uint64_t>(rhsBytes) < static_cast<std::uint64_t>(outColumns) * rowBytes)
		{
			return;
		}

		const auto positiveRows = static_cast<std::uint64_t>(lhsRows);
		std::vector<float> lhsSubblockSums(
		    static_cast<std::size_t>(positiveRows * blockCount * kGGMLQ4KPreparedSubblockCount));
		for (std::int64_t row = 0; row < lhsRows; ++row)
		{
			const auto* lhsRow = lhsAligned + lhsOffset + row * lhsRowStride;
			for (std::uint64_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
			{
				for (std::uint64_t subblock = 0; subblock < kGGMLQ4KPreparedSubblockCount; ++subblock)
				{
					float sum = 0.0F;
					const auto baseLane = blockIndex * kGGMLQ4KPreparedLanes + subblock * 32;
					for (std::uint64_t lane = 0; lane < 32; ++lane)
					{
						sum += lhsRow[static_cast<std::int64_t>(baseLane + lane) * lhsColumnStride];
					}
					const auto offset =
					    (static_cast<std::uint64_t>(row) * blockCount + blockIndex) * kGGMLQ4KPreparedSubblockCount +
					    subblock;
					lhsSubblockSums[static_cast<std::size_t>(offset)] = sum;
				}
			}
		}

		struct Context
		{
			const float* lhsAligned{};
			std::int64_t lhsOffset{};
			std::int64_t lhsRowStride{};
			std::int64_t lhsColumnStride{};
			const std::uint8_t* rhsAligned{};
			std::int64_t rhsOffset{};
			float* outAligned{};
			std::int64_t outOffset{};
			std::int64_t outColumns{};
			std::int64_t outRowStride{};
			std::int64_t outColumnStride{};
			const float* lhsSubblockSums{};
			std::uint64_t blockCount{};
			std::uint64_t rowBytes{};
			std::uint64_t columnGroupsPerRow{};
			bool useAVX2Prepared{};
		};
#if LITENN_HAS_X86_AVX2_TARGET
		const auto useAVX2Prepared = lhsColumnStride == 1 && LiteNNCPUHasAVX2();
#else
		const auto useAVX2Prepared = false;
#endif
		Context context{ .lhsAligned = lhsAligned,
			             .lhsOffset = lhsOffset,
			             .lhsRowStride = lhsRowStride,
			             .lhsColumnStride = lhsColumnStride,
			             .rhsAligned = rhsAligned,
			             .rhsOffset = rhsOffset,
			             .outAligned = outAligned,
			             .outOffset = outOffset,
			             .outColumns = outColumns,
			             .outRowStride = outRowStride,
			             .outColumnStride = outColumnStride,
			             .lhsSubblockSums = lhsSubblockSums.data(),
			             .blockCount = blockCount,
			             .rowBytes = rowBytes,
			             .columnGroupsPerRow = (static_cast<std::uint64_t>(outColumns) + 3) / 4,
			             .useAVX2Prepared = useAVX2Prepared };
		const auto body = [](std::uint64_t begin, std::uint64_t end, void* userData) {
			const auto& ctx = *static_cast<const Context*>(userData);
			for (std::uint64_t groupIndex = begin; groupIndex < end; ++groupIndex)
			{
				const auto row = static_cast<std::int64_t>(groupIndex / ctx.columnGroupsPerRow);
				const auto columnBase = (groupIndex % ctx.columnGroupsPerRow) * static_cast<std::uint64_t>(4);
				const auto* lhsRow = ctx.lhsAligned + ctx.lhsOffset + row * ctx.lhsRowStride;
				float acc[4] = {};
				bool valid[4] = { columnBase < static_cast<std::uint64_t>(ctx.outColumns),
					              columnBase + 1 < static_cast<std::uint64_t>(ctx.outColumns),
					              columnBase + 2 < static_cast<std::uint64_t>(ctx.outColumns),
					              columnBase + 3 < static_cast<std::uint64_t>(ctx.outColumns) };
				const auto allValid = columnBase + 3 < static_cast<std::uint64_t>(ctx.outColumns);
				for (std::uint64_t blockIndex = 0; blockIndex < ctx.blockCount; ++blockIndex)
				{
					const auto* lhsBlock =
					    lhsRow + static_cast<std::int64_t>(blockIndex * kGGMLQ4KPreparedLanes) * ctx.lhsColumnStride;
					const auto* lhsSums =
					    ctx.lhsSubblockSums +
					    (static_cast<std::uint64_t>(row) * ctx.blockCount + blockIndex) * kGGMLQ4KPreparedSubblockCount;
					const std::uint8_t* blocks[4] = {};
					if (allValid)
					{
						for (int localColumn = 0; localColumn < 4; ++localColumn)
						{
							const auto column = columnBase + static_cast<std::uint64_t>(localColumn);
							blocks[localColumn] = ctx.rhsAligned + ctx.rhsOffset + column * ctx.rowBytes +
							                      blockIndex * kGGMLQ4KPreparedBlockBytes;
						}
#if LITENN_HAS_X86_AVX2_TARGET
						if (ctx.useAVX2Prepared)
						{
							AccumulateGGMLQ4KPreparedBlockF32x4AllValidAVX2(blocks, lhsBlock, lhsSums, acc);
							continue;
						}
#endif
						AccumulateGGMLQ4KPreparedBlockF32x4AllValid(blocks, lhsBlock, ctx.lhsColumnStride, lhsSums,
						                                            acc);
						continue;
					}

					for (int localColumn = 0; localColumn < 4; ++localColumn)
					{
						if (!valid[localColumn])
						{
							continue;
						}
						const auto column = columnBase + static_cast<std::uint64_t>(localColumn);
						blocks[localColumn] = ctx.rhsAligned + ctx.rhsOffset + column * ctx.rowBytes +
						                      blockIndex * kGGMLQ4KPreparedBlockBytes;
					}
					AccumulateGGMLQ4KPreparedBlockF32x4(blocks, valid, lhsBlock, ctx.lhsColumnStride, lhsSums, acc);
				}
				if (allValid)
				{
					for (int localColumn = 0; localColumn < 4; ++localColumn)
					{
						const auto column =
						    static_cast<std::int64_t>(columnBase + static_cast<std::uint64_t>(localColumn));
						ctx.outAligned[ctx.outOffset + row * ctx.outRowStride + column * ctx.outColumnStride] =
						    acc[localColumn];
					}
					continue;
				}
				for (int localColumn = 0; localColumn < 4; ++localColumn)
				{
					if (!valid[localColumn])
					{
						continue;
					}
					const auto column = static_cast<std::int64_t>(columnBase + static_cast<std::uint64_t>(localColumn));
					ctx.outAligned[ctx.outOffset + row * ctx.outRowStride + column * ctx.outColumnStride] =
					    acc[localColumn];
				}
			}
		};
		const auto outputGroups = static_cast<std::uint64_t>(lhsRows) * context.columnGroupsPerRow;
		const auto operations = static_cast<std::uint64_t>(lhsRows) * static_cast<std::uint64_t>(outColumns) *
		                        static_cast<std::uint64_t>(lhsColumns);
		const auto threadCount =
		    ResolveGGMLBlockMatMulThreadCount(QuantizedBlockFormat::GGML_Q4_K, GGMLActivationDotMode::DirectFloat32,
		                                      operations, outputGroups, requestedThreadCount);
		const auto grain = std::max<std::uint64_t>(1, outputGroups / (std::max<std::uint64_t>(1, threadCount) * 8));
		const auto affinityPolicy = ResolveCPUAOTAffinityPolicy(affinityPolicyValue);
		const auto waitPolicy = ResolveCPUAOTWorkerWaitPolicy(affinityPolicyValue);
		if (threadCount <= 1)
		{
			body(0, outputGroups, &context);
			return;
		}
		LiteNNCPUParallelFor(0, outputGroups, grain, body, &context, threadCount, affinityPolicy, waitPolicy);
	}

	extern "C" std::uint64_t litenn_cpu_ggml_q6k_prepacked_block_bytes()
	{
		return kGGMLQ6KPreparedBlockBytes;
	}

	extern "C" void litenn_cpu_ggml_prepack_q6k_f32(const std::uint8_t*, const std::uint8_t* rhsAligned,
	                                                std::int64_t rhsOffset, std::int64_t rhsBytes,
	                                                std::int64_t rhsStride, std::int64_t rows, std::int64_t columns,
	                                                std::uint8_t*, std::uint8_t* packedAligned,
	                                                std::int64_t packedOffset, std::int64_t packedBytes,
	                                                std::int64_t packedStride)
	{
		const auto layout = GetQuantizedBlockLayout(QuantizedBlockFormat::GGML_Q6_K);
		if (!layout || !rhsAligned || !packedAligned || rhsOffset < 0 || rhsBytes < 0 || rhsStride <= 0 || rows < 0 ||
		    columns < 0 || packedOffset < 0 || packedBytes < 0 || packedStride != 1 ||
		    static_cast<std::uint64_t>(columns) % layout->elementsPerBlock != 0)
		{
			return;
		}
		const auto blockCount = static_cast<std::uint64_t>(columns) / layout->elementsPerBlock;
		const auto rowBytes = blockCount * layout->bytesPerBlock;
		const auto packedRowBytes = blockCount * kGGMLQ6KPreparedBlockBytes;
		if (static_cast<std::uint64_t>(rhsBytes) < static_cast<std::uint64_t>(rows) * rowBytes ||
		    static_cast<std::uint64_t>(packedBytes) < static_cast<std::uint64_t>(rows) * packedRowBytes)
		{
			return;
		}
		for (std::int64_t row = 0; row < rows; ++row)
		{
			for (std::uint64_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
			{
				const auto* sourceBlock =
				    rhsAligned + rhsOffset +
				    (static_cast<std::uint64_t>(row) * rowBytes + blockIndex * layout->bytesPerBlock) *
				        static_cast<std::uint64_t>(rhsStride);
				auto* targetBlock = packedAligned + packedOffset + static_cast<std::uint64_t>(row) * packedRowBytes +
				                    blockIndex * kGGMLQ6KPreparedBlockBytes;
				PrepareGGMLQ6KBlockF32(sourceBlock, rhsStride, targetBlock);
			}
		}
	}

	extern "C" void litenn_cpu_ggml_block_matmul_q6k_prepacked_f32(
	    const float*, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows, std::int64_t lhsColumns,
	    std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t*, const std::uint8_t* rhsAligned,
	    std::int64_t rhsOffset, std::int64_t rhsBytes, std::int64_t rhsStride, float*, float* outAligned,
	    std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride,
	    std::int64_t outColumnStride, std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_matmul_q6k_prepacked_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? std::format("format=GGML_Q6_K activation=direct_prepacked lhs={}x{} out={}x{} requested_threads={}",
		                      lhsRows, lhsColumns, outRows, outColumns, requestedThreadCount)
		        : std::string{});
		const auto layout = GetQuantizedBlockLayout(QuantizedBlockFormat::GGML_Q6_K);
		if (!layout || !lhsAligned || !rhsAligned || !outAligned || lhsRows < 0 || lhsColumns < 0 || outRows < 0 ||
		    outColumns < 0 || lhsRows != outRows || lhsColumns == 0 || outColumns == 0 || rhsOffset < 0 ||
		    rhsBytes < 0 || rhsStride != 1 || static_cast<std::uint64_t>(lhsColumns) % layout->elementsPerBlock != 0)
		{
			return;
		}
		const auto blockCount = static_cast<std::uint64_t>(lhsColumns) / layout->elementsPerBlock;
		const auto rowBytes = blockCount * kGGMLQ6KPreparedBlockBytes;
		if (static_cast<std::uint64_t>(rhsBytes) < static_cast<std::uint64_t>(outColumns) * rowBytes)
		{
			return;
		}

		struct Context
		{
			const float* lhsAligned{};
			std::int64_t lhsOffset{};
			std::int64_t lhsRowStride{};
			std::int64_t lhsColumnStride{};
			const std::uint8_t* rhsAligned{};
			std::int64_t rhsOffset{};
			float* outAligned{};
			std::int64_t outOffset{};
			std::int64_t outColumns{};
			std::int64_t outRowStride{};
			std::int64_t outColumnStride{};
			std::uint64_t blockCount{};
			std::uint64_t rowBytes{};
			std::uint64_t columnGroupsPerRow{};
			bool useAVX2Prepared{};
		};
#if LITENN_HAS_X86_AVX2_TARGET
		const auto useAVX2Prepared = lhsColumnStride == 1 && LiteNNCPUHasAVX2();
#else
		const auto useAVX2Prepared = false;
#endif
		Context context{ .lhsAligned = lhsAligned,
			             .lhsOffset = lhsOffset,
			             .lhsRowStride = lhsRowStride,
			             .lhsColumnStride = lhsColumnStride,
			             .rhsAligned = rhsAligned,
			             .rhsOffset = rhsOffset,
			             .outAligned = outAligned,
			             .outOffset = outOffset,
			             .outColumns = outColumns,
			             .outRowStride = outRowStride,
			             .outColumnStride = outColumnStride,
			             .blockCount = blockCount,
			             .rowBytes = rowBytes,
			             .columnGroupsPerRow = (static_cast<std::uint64_t>(outColumns) + 3) / 4,
			             .useAVX2Prepared = useAVX2Prepared };
		const auto body = [](std::uint64_t begin, std::uint64_t end, void* userData) {
			const auto& ctx = *static_cast<const Context*>(userData);
			for (std::uint64_t groupIndex = begin; groupIndex < end; ++groupIndex)
			{
				const auto row = static_cast<std::int64_t>(groupIndex / ctx.columnGroupsPerRow);
				const auto columnBase = (groupIndex % ctx.columnGroupsPerRow) * static_cast<std::uint64_t>(4);
				const auto* lhsRow = ctx.lhsAligned + ctx.lhsOffset + row * ctx.lhsRowStride;
				float acc[4] = {};
				bool valid[4] = { columnBase < static_cast<std::uint64_t>(ctx.outColumns),
					              columnBase + 1 < static_cast<std::uint64_t>(ctx.outColumns),
					              columnBase + 2 < static_cast<std::uint64_t>(ctx.outColumns),
					              columnBase + 3 < static_cast<std::uint64_t>(ctx.outColumns) };
				const auto allValid = columnBase + 3 < static_cast<std::uint64_t>(ctx.outColumns);
				for (std::uint64_t blockIndex = 0; blockIndex < ctx.blockCount; ++blockIndex)
				{
					const auto* lhsBlock =
					    lhsRow + static_cast<std::int64_t>(blockIndex * kGGMLQ6KPreparedLanes) * ctx.lhsColumnStride;
					const std::uint8_t* blocks[4] = {};
					if (allValid)
					{
						for (int localColumn = 0; localColumn < 4; ++localColumn)
						{
							const auto column = columnBase + static_cast<std::uint64_t>(localColumn);
							blocks[localColumn] = ctx.rhsAligned + ctx.rhsOffset + column * ctx.rowBytes +
							                      blockIndex * kGGMLQ6KPreparedBlockBytes;
						}
#if LITENN_HAS_X86_AVX2_TARGET
						if (ctx.useAVX2Prepared)
						{
							AccumulateGGMLQ6KPreparedBlockF32x4AllValidAVX2(blocks, lhsBlock, acc);
							continue;
						}
#endif
						AccumulateGGMLQ6KPreparedBlockF32x4AllValid(blocks, lhsBlock, ctx.lhsColumnStride, acc);
						continue;
					}

					for (int localColumn = 0; localColumn < 4; ++localColumn)
					{
						if (!valid[localColumn])
						{
							continue;
						}
						const auto column = columnBase + static_cast<std::uint64_t>(localColumn);
						blocks[localColumn] = ctx.rhsAligned + ctx.rhsOffset + column * ctx.rowBytes +
						                      blockIndex * kGGMLQ6KPreparedBlockBytes;
					}
					AccumulateGGMLQ6KPreparedBlockF32x4(blocks, valid, lhsBlock, ctx.lhsColumnStride, acc);
				}
				if (allValid)
				{
					for (int localColumn = 0; localColumn < 4; ++localColumn)
					{
						const auto column =
						    static_cast<std::int64_t>(columnBase + static_cast<std::uint64_t>(localColumn));
						ctx.outAligned[ctx.outOffset + row * ctx.outRowStride + column * ctx.outColumnStride] =
						    acc[localColumn];
					}
					continue;
				}
				for (int localColumn = 0; localColumn < 4; ++localColumn)
				{
					if (!valid[localColumn])
					{
						continue;
					}
					const auto column = static_cast<std::int64_t>(columnBase + static_cast<std::uint64_t>(localColumn));
					ctx.outAligned[ctx.outOffset + row * ctx.outRowStride + column * ctx.outColumnStride] =
					    acc[localColumn];
				}
			}
		};
		const auto outputGroups = static_cast<std::uint64_t>(lhsRows) * context.columnGroupsPerRow;
		const auto operations = static_cast<std::uint64_t>(lhsRows) * static_cast<std::uint64_t>(outColumns) *
		                        static_cast<std::uint64_t>(lhsColumns);
		const auto threadCount =
		    ResolveGGMLBlockMatMulThreadCount(QuantizedBlockFormat::GGML_Q6_K, GGMLActivationDotMode::DirectFloat32,
		                                      operations, outputGroups, requestedThreadCount);
		const auto grain = std::max<std::uint64_t>(1, outputGroups / (std::max<std::uint64_t>(1, threadCount) * 8));
		const auto affinityPolicy = ResolveCPUAOTAffinityPolicy(affinityPolicyValue);
		const auto waitPolicy = ResolveCPUAOTWorkerWaitPolicy(affinityPolicyValue);
		if (threadCount <= 1)
		{
			body(0, outputGroups, &context);
			return;
		}
		LiteNNCPUParallelFor(0, outputGroups, grain, body, &context, threadCount, affinityPolicy, waitPolicy);
	}

	using GGMLPrepackedMatMulHelperFn = void (*)(const float*, const float*, std::int64_t, std::int64_t, std::int64_t,
	                                             std::int64_t, std::int64_t, const std::uint8_t*, const std::uint8_t*,
	                                             std::int64_t, std::int64_t, std::int64_t, float*, float*, std::int64_t,
	                                             std::int64_t, std::int64_t, std::int64_t, std::int64_t, std::uint64_t,
	                                             std::uint64_t);

	GGMLPrepackedMatMulHelperFn GGMLPrepackedMatMulHelperFor(QuantizedBlockFormat format)
	{
		switch (format)
		{
		case QuantizedBlockFormat::GGML_Q4_K:
			return litenn_cpu_ggml_block_matmul_q4k_prepacked_f32;
		case QuantizedBlockFormat::GGML_Q6_K:
			return litenn_cpu_ggml_block_matmul_q6k_prepacked_f32;
		default:
			return nullptr;
		}
	}

	void LiteNNCPUGGMLBlockGroupedMatMul2PrepackedF32(
	    QuantizedBlockFormat format, const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset,
	    std::int64_t lhsRows, std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride,
	    const std::uint8_t* rhs0Base, const std::uint8_t* rhs0Aligned, std::int64_t rhs0Offset, std::int64_t rhs0Bytes,
	    std::int64_t rhs0Stride, const std::uint8_t* rhs1Base, const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset,
	    std::int64_t rhs1Bytes, std::int64_t rhs1Stride, float* outBase, float* outAligned, std::int64_t outOffset,
	    std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride,
	    std::uint64_t out0Columns, std::uint64_t out1Columns, std::uint64_t requestedThreadCount,
	    std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out0Columns + out1Columns != static_cast<std::uint64_t>(std::max<std::int64_t>(0, outColumns)))
		{
			return;
		}
		const auto invoke = GGMLPrepackedMatMulHelperFor(format);
		if (!invoke)
		{
			return;
		}
		invoke(lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride, lhsColumnStride, rhs0Base,
		       rhs0Aligned, rhs0Offset, rhs0Bytes, rhs0Stride, outBase, outAligned, outOffset, outRows,
		       static_cast<std::int64_t>(out0Columns), outRowStride, outColumnStride, requestedThreadCount,
		       affinityPolicyValue);
		invoke(lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride, lhsColumnStride, rhs1Base,
		       rhs1Aligned, rhs1Offset, rhs1Bytes, rhs1Stride, outBase, outAligned,
		       outOffset + static_cast<std::int64_t>(out0Columns) * outColumnStride, outRows,
		       static_cast<std::int64_t>(out1Columns), outRowStride, outColumnStride, requestedThreadCount,
		       affinityPolicyValue);
	}

	void LiteNNCPUGGMLBlockGroupedMatMul3PrepackedF32(
	    QuantizedBlockFormat format, const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset,
	    std::int64_t lhsRows, std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride,
	    const std::uint8_t* rhs0Base, const std::uint8_t* rhs0Aligned, std::int64_t rhs0Offset, std::int64_t rhs0Bytes,
	    std::int64_t rhs0Stride, const std::uint8_t* rhs1Base, const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset,
	    std::int64_t rhs1Bytes, std::int64_t rhs1Stride, const std::uint8_t* rhs2Base, const std::uint8_t* rhs2Aligned,
	    std::int64_t rhs2Offset, std::int64_t rhs2Bytes, std::int64_t rhs2Stride, float* outBase, float* outAligned,
	    std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride,
	    std::int64_t outColumnStride, std::uint64_t out0Columns, std::uint64_t out1Columns, std::uint64_t out2Columns,
	    std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out2Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out0Columns + out1Columns + out2Columns !=
		        static_cast<std::uint64_t>(std::max<std::int64_t>(0, outColumns)))
		{
			return;
		}
		const auto invoke = GGMLPrepackedMatMulHelperFor(format);
		if (!invoke)
		{
			return;
		}
		const auto out0 = static_cast<std::int64_t>(out0Columns);
		const auto out1 = static_cast<std::int64_t>(out1Columns);
		const auto out2 = static_cast<std::int64_t>(out2Columns);
		invoke(lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride, lhsColumnStride, rhs0Base,
		       rhs0Aligned, rhs0Offset, rhs0Bytes, rhs0Stride, outBase, outAligned, outOffset, outRows, out0,
		       outRowStride, outColumnStride, requestedThreadCount, affinityPolicyValue);
		invoke(lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride, lhsColumnStride, rhs1Base,
		       rhs1Aligned, rhs1Offset, rhs1Bytes, rhs1Stride, outBase, outAligned, outOffset + out0 * outColumnStride,
		       outRows, out1, outRowStride, outColumnStride, requestedThreadCount, affinityPolicyValue);
		invoke(lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride, lhsColumnStride, rhs2Base,
		       rhs2Aligned, rhs2Offset, rhs2Bytes, rhs2Stride, outBase, outAligned,
		       outOffset + (out0 + out1) * outColumnStride, outRows, out2, outRowStride, outColumnStride,
		       requestedThreadCount, affinityPolicyValue);
	}

	void LiteNNCPUGGMLBlockMixedProjectedF32(const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset,
	                                         std::int64_t lhsRows, std::int64_t lhsColumns, std::int64_t lhsRowStride,
	                                         std::int64_t lhsColumnStride,
	                                         std::span<const GGMLBlockMatMulProjection> projections,
	                                         std::span<const QuantizedBlockFormat> formats, float* outBase,
	                                         float* outAligned, std::int64_t outOffset, std::int64_t outRows,
	                                         std::int64_t outColumns, std::int64_t outRowStride,
	                                         std::int64_t outColumnStride, std::uint64_t requestedThreadCount,
	                                         std::uint64_t affinityPolicyValue, GGMLActivationDotMode activationDotMode)
	{
		if (projections.size() != formats.size() || projections.size() < 2 || projections.size() > 3)
		{
			return;
		}
		std::int64_t totalOutputColumns = 0;
		for (const auto& projection : projections)
		{
			if (projection.outColumns <= 0 ||
			    projection.outColumns > std::numeric_limits<std::int64_t>::max() - totalOutputColumns)
			{
				return;
			}
			totalOutputColumns += projection.outColumns;
		}
		if (totalOutputColumns != outColumns)
		{
			return;
		}

		std::int64_t outputColumnOffset = 0;
		for (std::size_t projectionIndex = 0; projectionIndex < projections.size(); ++projectionIndex)
		{
			const auto& projection = projections[projectionIndex];
			LiteNNCPUGGMLBlockMatMulProjectedF32(lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride,
			                                     lhsColumnStride, std::span{ &projection, std::size_t{ 1 } }, outBase,
			                                     outAligned, outOffset + outputColumnOffset * outColumnStride, outRows,
			                                     projection.outColumns, outRowStride, outColumnStride,
			                                     static_cast<std::uint64_t>(formats[projectionIndex]),
			                                     requestedThreadCount, affinityPolicyValue, activationDotMode);
			outputColumnOffset += projection.outColumns;
		}
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul2_mixed_f32(
	    const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows,
	    std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t* rhs0Base,
	    const std::uint8_t* rhs0Aligned, std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride,
	    const std::uint8_t* rhs1Base, const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes,
	    std::int64_t rhs1Stride, float* outBase, float* outAligned, std::int64_t outOffset, std::int64_t outRows,
	    std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t format0Value,
	    std::uint64_t format1Value, std::uint64_t out0Columns, std::uint64_t out1Columns,
	    std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
		{
			return;
		}
		const std::array formats{ static_cast<QuantizedBlockFormat>(format0Value),
			                      static_cast<QuantizedBlockFormat>(format1Value) };
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_grouped_matmul2_mixed_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLMixedBlockMatMulProfileDetail(formats, GGMLActivationDotMode::DirectFloat32, lhsRows,
		                                                 lhsColumns, outRows, outColumns, requestedThreadCount)
		        : std::string{});
		const std::array projections{
			GGMLBlockMatMulProjection{ rhs0Aligned, rhs0Offset, rhs0Bytes, rhs0Stride,
			                           static_cast<std::int64_t>(out0Columns) },
			GGMLBlockMatMulProjection{ rhs1Aligned, rhs1Offset, rhs1Bytes, rhs1Stride,
			                           static_cast<std::int64_t>(out1Columns) },
		};
		LiteNNCPUGGMLBlockMixedProjectedF32(lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride,
		                                    lhsColumnStride, projections, formats, outBase, outAligned, outOffset,
		                                    outRows, outColumns, outRowStride, outColumnStride, requestedThreadCount,
		                                    affinityPolicyValue, GGMLActivationDotMode::DirectFloat32);
		(void) rhs0Base;
		(void) rhs1Base;
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul2_mixed_q8k_staged_f32(
	    const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows,
	    std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t* rhs0Base,
	    const std::uint8_t* rhs0Aligned, std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride,
	    const std::uint8_t* rhs1Base, const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes,
	    std::int64_t rhs1Stride, float* outBase, float* outAligned, std::int64_t outOffset, std::int64_t outRows,
	    std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t format0Value,
	    std::uint64_t format1Value, std::uint64_t out0Columns, std::uint64_t out1Columns,
	    std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
		{
			return;
		}
		const std::array formats{ static_cast<QuantizedBlockFormat>(format0Value),
			                      static_cast<QuantizedBlockFormat>(format1Value) };
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_grouped_matmul2_mixed_q8k_staged_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLMixedBlockMatMulProfileDetail(formats, GGMLActivationDotMode::Q8KStaged, lhsRows, lhsColumns,
		                                                 outRows, outColumns, requestedThreadCount)
		        : std::string{});
		const std::array projections{
			GGMLBlockMatMulProjection{ rhs0Aligned, rhs0Offset, rhs0Bytes, rhs0Stride,
			                           static_cast<std::int64_t>(out0Columns) },
			GGMLBlockMatMulProjection{ rhs1Aligned, rhs1Offset, rhs1Bytes, rhs1Stride,
			                           static_cast<std::int64_t>(out1Columns) },
		};
		LiteNNCPUGGMLBlockMixedProjectedF32(lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride,
		                                    lhsColumnStride, projections, formats, outBase, outAligned, outOffset,
		                                    outRows, outColumns, outRowStride, outColumnStride, requestedThreadCount,
		                                    affinityPolicyValue, GGMLActivationDotMode::Q8KStaged);
		(void) rhs0Base;
		(void) rhs1Base;
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul3_mixed_f32(
	    const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows,
	    std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t* rhs0Base,
	    const std::uint8_t* rhs0Aligned, std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride,
	    const std::uint8_t* rhs1Base, const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes,
	    std::int64_t rhs1Stride, const std::uint8_t* rhs2Base, const std::uint8_t* rhs2Aligned, std::int64_t rhs2Offset,
	    std::int64_t rhs2Bytes, std::int64_t rhs2Stride, float* outBase, float* outAligned, std::int64_t outOffset,
	    std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride,
	    std::uint64_t format0Value, std::uint64_t format1Value, std::uint64_t format2Value, std::uint64_t out0Columns,
	    std::uint64_t out1Columns, std::uint64_t out2Columns, std::uint64_t requestedThreadCount,
	    std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out2Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
		{
			return;
		}
		const std::array formats{ static_cast<QuantizedBlockFormat>(format0Value),
			                      static_cast<QuantizedBlockFormat>(format1Value),
			                      static_cast<QuantizedBlockFormat>(format2Value) };
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_grouped_matmul3_mixed_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLMixedBlockMatMulProfileDetail(formats, GGMLActivationDotMode::DirectFloat32, lhsRows,
		                                                 lhsColumns, outRows, outColumns, requestedThreadCount)
		        : std::string{});
		const std::array projections{
			GGMLBlockMatMulProjection{ rhs0Aligned, rhs0Offset, rhs0Bytes, rhs0Stride,
			                           static_cast<std::int64_t>(out0Columns) },
			GGMLBlockMatMulProjection{ rhs1Aligned, rhs1Offset, rhs1Bytes, rhs1Stride,
			                           static_cast<std::int64_t>(out1Columns) },
			GGMLBlockMatMulProjection{ rhs2Aligned, rhs2Offset, rhs2Bytes, rhs2Stride,
			                           static_cast<std::int64_t>(out2Columns) },
		};
		LiteNNCPUGGMLBlockMixedProjectedF32(lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride,
		                                    lhsColumnStride, projections, formats, outBase, outAligned, outOffset,
		                                    outRows, outColumns, outRowStride, outColumnStride, requestedThreadCount,
		                                    affinityPolicyValue, GGMLActivationDotMode::DirectFloat32);
		(void) rhs0Base;
		(void) rhs1Base;
		(void) rhs2Base;
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul3_mixed_q8k_staged_f32(
	    const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows,
	    std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t* rhs0Base,
	    const std::uint8_t* rhs0Aligned, std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride,
	    const std::uint8_t* rhs1Base, const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes,
	    std::int64_t rhs1Stride, const std::uint8_t* rhs2Base, const std::uint8_t* rhs2Aligned, std::int64_t rhs2Offset,
	    std::int64_t rhs2Bytes, std::int64_t rhs2Stride, float* outBase, float* outAligned, std::int64_t outOffset,
	    std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride,
	    std::uint64_t format0Value, std::uint64_t format1Value, std::uint64_t format2Value, std::uint64_t out0Columns,
	    std::uint64_t out1Columns, std::uint64_t out2Columns, std::uint64_t requestedThreadCount,
	    std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out2Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
		{
			return;
		}
		const std::array formats{ static_cast<QuantizedBlockFormat>(format0Value),
			                      static_cast<QuantizedBlockFormat>(format1Value),
			                      static_cast<QuantizedBlockFormat>(format2Value) };
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_grouped_matmul3_mixed_q8k_staged_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLMixedBlockMatMulProfileDetail(formats, GGMLActivationDotMode::Q8KStaged, lhsRows, lhsColumns,
		                                                 outRows, outColumns, requestedThreadCount)
		        : std::string{});
		const std::array projections{
			GGMLBlockMatMulProjection{ rhs0Aligned, rhs0Offset, rhs0Bytes, rhs0Stride,
			                           static_cast<std::int64_t>(out0Columns) },
			GGMLBlockMatMulProjection{ rhs1Aligned, rhs1Offset, rhs1Bytes, rhs1Stride,
			                           static_cast<std::int64_t>(out1Columns) },
			GGMLBlockMatMulProjection{ rhs2Aligned, rhs2Offset, rhs2Bytes, rhs2Stride,
			                           static_cast<std::int64_t>(out2Columns) },
		};
		LiteNNCPUGGMLBlockMixedProjectedF32(lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride,
		                                    lhsColumnStride, projections, formats, outBase, outAligned, outOffset,
		                                    outRows, outColumns, outRowStride, outColumnStride, requestedThreadCount,
		                                    affinityPolicyValue, GGMLActivationDotMode::Q8KStaged);
		(void) rhs0Base;
		(void) rhs1Base;
		(void) rhs2Base;
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul2_f32(
	    const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows,
	    std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t* rhs0Base,
	    const std::uint8_t* rhs0Aligned, std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride,
	    const std::uint8_t* rhs1Base, const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes,
	    std::int64_t rhs1Stride, float* outBase, float* outAligned, std::int64_t outOffset, std::int64_t outRows,
	    std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t formatValue,
	    std::uint64_t out0Columns, std::uint64_t out1Columns, std::uint64_t requestedThreadCount,
	    std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
		{
			return;
		}
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_grouped_matmul2_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLBlockMatMulProfileDetail(format, GGMLActivationDotMode::DirectFloat32, lhsRows, lhsColumns,
		                                            outRows, outColumns, requestedThreadCount)
		        : std::string{});
		const std::array projections{
			GGMLBlockMatMulProjection{ .rhsAligned = rhs0Aligned,
			                           .rhsOffset = rhs0Offset,
			                           .rhsBytes = rhs0Bytes,
			                           .rhsStride = rhs0Stride,
			                           .outColumns = static_cast<std::int64_t>(out0Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs1Aligned,
			                           .rhsOffset = rhs1Offset,
			                           .rhsBytes = rhs1Bytes,
			                           .rhsStride = rhs1Stride,
			                           .outColumns = static_cast<std::int64_t>(out1Columns) },
		};
		LiteNNCPUGGMLBlockMatMulProjectedF32(
		    lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride, lhsColumnStride, projections, outBase,
		    outAligned, outOffset, outRows, outColumns, outRowStride, outColumnStride, formatValue,
		    requestedThreadCount, affinityPolicyValue, GGMLActivationDotMode::DirectFloat32);
		(void) rhs0Base;
		(void) rhs1Base;
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul2_q8k_staged_f32(
	    const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows,
	    std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t* rhs0Base,
	    const std::uint8_t* rhs0Aligned, std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride,
	    const std::uint8_t* rhs1Base, const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes,
	    std::int64_t rhs1Stride, float* outBase, float* outAligned, std::int64_t outOffset, std::int64_t outRows,
	    std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t formatValue,
	    std::uint64_t out0Columns, std::uint64_t out1Columns, std::uint64_t requestedThreadCount,
	    std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
		{
			return;
		}
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_grouped_matmul2_q8k_staged_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLBlockMatMulProfileDetail(format, GGMLActivationDotMode::Q8KStaged, lhsRows, lhsColumns,
		                                            outRows, outColumns, requestedThreadCount)
		        : std::string{});
		const std::array projections{
			GGMLBlockMatMulProjection{ .rhsAligned = rhs0Aligned,
			                           .rhsOffset = rhs0Offset,
			                           .rhsBytes = rhs0Bytes,
			                           .rhsStride = rhs0Stride,
			                           .outColumns = static_cast<std::int64_t>(out0Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs1Aligned,
			                           .rhsOffset = rhs1Offset,
			                           .rhsBytes = rhs1Bytes,
			                           .rhsStride = rhs1Stride,
			                           .outColumns = static_cast<std::int64_t>(out1Columns) },
		};
		LiteNNCPUGGMLBlockMatMulProjectedF32(
		    lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride, lhsColumnStride, projections, outBase,
		    outAligned, outOffset, outRows, outColumns, outRowStride, outColumnStride, formatValue,
		    requestedThreadCount, affinityPolicyValue, GGMLActivationDotMode::Q8KStaged);
		(void) rhs0Base;
		(void) rhs1Base;
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul2_q8k_prepared_activation_f32(
	    const std::uint8_t*, const std::uint8_t* lhsQ8KAligned, std::int64_t lhsQ8KOffset, std::int64_t lhsQ8KBytes,
	    std::int64_t lhsQ8KStride, std::int64_t lhsRows, std::int64_t lhsColumns, const std::uint8_t* rhs0Base,
	    const std::uint8_t* rhs0Aligned, std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride,
	    const std::uint8_t* rhs1Base, const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes,
	    std::int64_t rhs1Stride, float* outBase, float* outAligned, std::int64_t outOffset, std::int64_t outRows,
	    std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t formatValue,
	    std::uint64_t out0Columns, std::uint64_t out1Columns, std::uint64_t requestedThreadCount,
	    std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
		{
			return;
		}
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_grouped_matmul2_q8k_prepared_activation_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLBlockMatMulProfileDetail(format, GGMLActivationDotMode::Q8KStaged, lhsRows, lhsColumns,
		                                            outRows, outColumns, requestedThreadCount)
		        : std::string{});
		const auto* preparedBlocks = ResolveGGMLQ8KActivationBlocks(format, lhsQ8KAligned, lhsQ8KOffset, lhsQ8KBytes,
		                                                            lhsQ8KStride, lhsRows, lhsColumns);
		if (!preparedBlocks)
		{
			return;
		}
		const std::array projections{
			GGMLBlockMatMulProjection{ .rhsAligned = rhs0Aligned,
			                           .rhsOffset = rhs0Offset,
			                           .rhsBytes = rhs0Bytes,
			                           .rhsStride = rhs0Stride,
			                           .outColumns = static_cast<std::int64_t>(out0Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs1Aligned,
			                           .rhsOffset = rhs1Offset,
			                           .rhsBytes = rhs1Bytes,
			                           .rhsStride = rhs1Stride,
			                           .outColumns = static_cast<std::int64_t>(out1Columns) },
		};
		LiteNNCPUGGMLBlockMatMulProjectedF32(nullptr, nullptr, 0, lhsRows, lhsColumns, 0, 0, projections, outBase,
		                                     outAligned, outOffset, outRows, outColumns, outRowStride, outColumnStride,
		                                     formatValue, requestedThreadCount, affinityPolicyValue,
		                                     GGMLActivationDotMode::Q8KStaged, preparedBlocks);
		(void) rhs0Base;
		(void) rhs1Base;
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul3_f32(
	    const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows,
	    std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t* rhs0Base,
	    const std::uint8_t* rhs0Aligned, std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride,
	    const std::uint8_t* rhs1Base, const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes,
	    std::int64_t rhs1Stride, const std::uint8_t* rhs2Base, const std::uint8_t* rhs2Aligned, std::int64_t rhs2Offset,
	    std::int64_t rhs2Bytes, std::int64_t rhs2Stride, float* outBase, float* outAligned, std::int64_t outOffset,
	    std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride,
	    std::uint64_t formatValue, std::uint64_t out0Columns, std::uint64_t out1Columns, std::uint64_t out2Columns,
	    std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out2Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
		{
			return;
		}
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_grouped_matmul3_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLBlockMatMulProfileDetail(format, GGMLActivationDotMode::DirectFloat32, lhsRows, lhsColumns,
		                                            outRows, outColumns, requestedThreadCount)
		        : std::string{});
		const std::array projections{
			GGMLBlockMatMulProjection{ .rhsAligned = rhs0Aligned,
			                           .rhsOffset = rhs0Offset,
			                           .rhsBytes = rhs0Bytes,
			                           .rhsStride = rhs0Stride,
			                           .outColumns = static_cast<std::int64_t>(out0Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs1Aligned,
			                           .rhsOffset = rhs1Offset,
			                           .rhsBytes = rhs1Bytes,
			                           .rhsStride = rhs1Stride,
			                           .outColumns = static_cast<std::int64_t>(out1Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs2Aligned,
			                           .rhsOffset = rhs2Offset,
			                           .rhsBytes = rhs2Bytes,
			                           .rhsStride = rhs2Stride,
			                           .outColumns = static_cast<std::int64_t>(out2Columns) },
		};
		LiteNNCPUGGMLBlockMatMulProjectedF32(
		    lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride, lhsColumnStride, projections, outBase,
		    outAligned, outOffset, outRows, outColumns, outRowStride, outColumnStride, formatValue,
		    requestedThreadCount, affinityPolicyValue, GGMLActivationDotMode::DirectFloat32);
		(void) rhs0Base;
		(void) rhs1Base;
		(void) rhs2Base;
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul3_q8k_staged_f32(
	    const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows,
	    std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t* rhs0Base,
	    const std::uint8_t* rhs0Aligned, std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride,
	    const std::uint8_t* rhs1Base, const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes,
	    std::int64_t rhs1Stride, const std::uint8_t* rhs2Base, const std::uint8_t* rhs2Aligned, std::int64_t rhs2Offset,
	    std::int64_t rhs2Bytes, std::int64_t rhs2Stride, float* outBase, float* outAligned, std::int64_t outOffset,
	    std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride,
	    std::uint64_t formatValue, std::uint64_t out0Columns, std::uint64_t out1Columns, std::uint64_t out2Columns,
	    std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out2Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
		{
			return;
		}
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_grouped_matmul3_q8k_staged_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLBlockMatMulProfileDetail(format, GGMLActivationDotMode::Q8KStaged, lhsRows, lhsColumns,
		                                            outRows, outColumns, requestedThreadCount)
		        : std::string{});
		const std::array projections{
			GGMLBlockMatMulProjection{ .rhsAligned = rhs0Aligned,
			                           .rhsOffset = rhs0Offset,
			                           .rhsBytes = rhs0Bytes,
			                           .rhsStride = rhs0Stride,
			                           .outColumns = static_cast<std::int64_t>(out0Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs1Aligned,
			                           .rhsOffset = rhs1Offset,
			                           .rhsBytes = rhs1Bytes,
			                           .rhsStride = rhs1Stride,
			                           .outColumns = static_cast<std::int64_t>(out1Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs2Aligned,
			                           .rhsOffset = rhs2Offset,
			                           .rhsBytes = rhs2Bytes,
			                           .rhsStride = rhs2Stride,
			                           .outColumns = static_cast<std::int64_t>(out2Columns) },
		};
		LiteNNCPUGGMLBlockMatMulProjectedF32(
		    lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride, lhsColumnStride, projections, outBase,
		    outAligned, outOffset, outRows, outColumns, outRowStride, outColumnStride, formatValue,
		    requestedThreadCount, affinityPolicyValue, GGMLActivationDotMode::Q8KStaged);
		(void) rhs0Base;
		(void) rhs1Base;
		(void) rhs2Base;
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul3_q8k_prepared_activation_f32(
	    const std::uint8_t*, const std::uint8_t* lhsQ8KAligned, std::int64_t lhsQ8KOffset, std::int64_t lhsQ8KBytes,
	    std::int64_t lhsQ8KStride, std::int64_t lhsRows, std::int64_t lhsColumns, const std::uint8_t* rhs0Base,
	    const std::uint8_t* rhs0Aligned, std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride,
	    const std::uint8_t* rhs1Base, const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes,
	    std::int64_t rhs1Stride, const std::uint8_t* rhs2Base, const std::uint8_t* rhs2Aligned, std::int64_t rhs2Offset,
	    std::int64_t rhs2Bytes, std::int64_t rhs2Stride, float* outBase, float* outAligned, std::int64_t outOffset,
	    std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride,
	    std::uint64_t formatValue, std::uint64_t out0Columns, std::uint64_t out1Columns, std::uint64_t out2Columns,
	    std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		if (out0Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out1Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) ||
		    out2Columns > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()))
		{
			return;
		}
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_grouped_matmul3_q8k_prepared_activation_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? BuildGGMLBlockMatMulProfileDetail(format, GGMLActivationDotMode::Q8KStaged, lhsRows, lhsColumns,
		                                            outRows, outColumns, requestedThreadCount)
		        : std::string{});
		const auto* preparedBlocks = ResolveGGMLQ8KActivationBlocks(format, lhsQ8KAligned, lhsQ8KOffset, lhsQ8KBytes,
		                                                            lhsQ8KStride, lhsRows, lhsColumns);
		if (!preparedBlocks)
		{
			return;
		}
		const std::array projections{
			GGMLBlockMatMulProjection{ .rhsAligned = rhs0Aligned,
			                           .rhsOffset = rhs0Offset,
			                           .rhsBytes = rhs0Bytes,
			                           .rhsStride = rhs0Stride,
			                           .outColumns = static_cast<std::int64_t>(out0Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs1Aligned,
			                           .rhsOffset = rhs1Offset,
			                           .rhsBytes = rhs1Bytes,
			                           .rhsStride = rhs1Stride,
			                           .outColumns = static_cast<std::int64_t>(out1Columns) },
			GGMLBlockMatMulProjection{ .rhsAligned = rhs2Aligned,
			                           .rhsOffset = rhs2Offset,
			                           .rhsBytes = rhs2Bytes,
			                           .rhsStride = rhs2Stride,
			                           .outColumns = static_cast<std::int64_t>(out2Columns) },
		};
		LiteNNCPUGGMLBlockMatMulProjectedF32(nullptr, nullptr, 0, lhsRows, lhsColumns, 0, 0, projections, outBase,
		                                     outAligned, outOffset, outRows, outColumns, outRowStride, outColumnStride,
		                                     formatValue, requestedThreadCount, affinityPolicyValue,
		                                     GGMLActivationDotMode::Q8KStaged, preparedBlocks);
		(void) rhs0Base;
		(void) rhs1Base;
		(void) rhs2Base;
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul2_q4k_prepacked_f32(
	    const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows,
	    std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t* rhs0Base,
	    const std::uint8_t* rhs0Aligned, std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride,
	    const std::uint8_t* rhs1Base, const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes,
	    std::int64_t rhs1Stride, float* outBase, float* outAligned, std::int64_t outOffset, std::int64_t outRows,
	    std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t out0Columns,
	    std::uint64_t out1Columns, std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		LiteNNCPUGGMLBlockGroupedMatMul2PrepackedF32(
		    QuantizedBlockFormat::GGML_Q4_K, lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride,
		    lhsColumnStride, rhs0Base, rhs0Aligned, rhs0Offset, rhs0Bytes, rhs0Stride, rhs1Base, rhs1Aligned,
		    rhs1Offset, rhs1Bytes, rhs1Stride, outBase, outAligned, outOffset, outRows, outColumns, outRowStride,
		    outColumnStride, out0Columns, out1Columns, requestedThreadCount, affinityPolicyValue);
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul2_q6k_prepacked_f32(
	    const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows,
	    std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t* rhs0Base,
	    const std::uint8_t* rhs0Aligned, std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride,
	    const std::uint8_t* rhs1Base, const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes,
	    std::int64_t rhs1Stride, float* outBase, float* outAligned, std::int64_t outOffset, std::int64_t outRows,
	    std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride, std::uint64_t out0Columns,
	    std::uint64_t out1Columns, std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		LiteNNCPUGGMLBlockGroupedMatMul2PrepackedF32(
		    QuantizedBlockFormat::GGML_Q6_K, lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride,
		    lhsColumnStride, rhs0Base, rhs0Aligned, rhs0Offset, rhs0Bytes, rhs0Stride, rhs1Base, rhs1Aligned,
		    rhs1Offset, rhs1Bytes, rhs1Stride, outBase, outAligned, outOffset, outRows, outColumns, outRowStride,
		    outColumnStride, out0Columns, out1Columns, requestedThreadCount, affinityPolicyValue);
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul3_q4k_prepacked_f32(
	    const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows,
	    std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t* rhs0Base,
	    const std::uint8_t* rhs0Aligned, std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride,
	    const std::uint8_t* rhs1Base, const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes,
	    std::int64_t rhs1Stride, const std::uint8_t* rhs2Base, const std::uint8_t* rhs2Aligned, std::int64_t rhs2Offset,
	    std::int64_t rhs2Bytes, std::int64_t rhs2Stride, float* outBase, float* outAligned, std::int64_t outOffset,
	    std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride,
	    std::uint64_t out0Columns, std::uint64_t out1Columns, std::uint64_t out2Columns,
	    std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		LiteNNCPUGGMLBlockGroupedMatMul3PrepackedF32(
		    QuantizedBlockFormat::GGML_Q4_K, lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride,
		    lhsColumnStride, rhs0Base, rhs0Aligned, rhs0Offset, rhs0Bytes, rhs0Stride, rhs1Base, rhs1Aligned,
		    rhs1Offset, rhs1Bytes, rhs1Stride, rhs2Base, rhs2Aligned, rhs2Offset, rhs2Bytes, rhs2Stride, outBase,
		    outAligned, outOffset, outRows, outColumns, outRowStride, outColumnStride, out0Columns, out1Columns,
		    out2Columns, requestedThreadCount, affinityPolicyValue);
	}

	extern "C" void litenn_cpu_ggml_block_grouped_matmul3_q6k_prepacked_f32(
	    const float* lhsBase, const float* lhsAligned, std::int64_t lhsOffset, std::int64_t lhsRows,
	    std::int64_t lhsColumns, std::int64_t lhsRowStride, std::int64_t lhsColumnStride, const std::uint8_t* rhs0Base,
	    const std::uint8_t* rhs0Aligned, std::int64_t rhs0Offset, std::int64_t rhs0Bytes, std::int64_t rhs0Stride,
	    const std::uint8_t* rhs1Base, const std::uint8_t* rhs1Aligned, std::int64_t rhs1Offset, std::int64_t rhs1Bytes,
	    std::int64_t rhs1Stride, const std::uint8_t* rhs2Base, const std::uint8_t* rhs2Aligned, std::int64_t rhs2Offset,
	    std::int64_t rhs2Bytes, std::int64_t rhs2Stride, float* outBase, float* outAligned, std::int64_t outOffset,
	    std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride,
	    std::uint64_t out0Columns, std::uint64_t out1Columns, std::uint64_t out2Columns,
	    std::uint64_t requestedThreadCount, std::uint64_t affinityPolicyValue)
	{
		LiteNNCPUGGMLBlockGroupedMatMul3PrepackedF32(
		    QuantizedBlockFormat::GGML_Q6_K, lhsBase, lhsAligned, lhsOffset, lhsRows, lhsColumns, lhsRowStride,
		    lhsColumnStride, rhs0Base, rhs0Aligned, rhs0Offset, rhs0Bytes, rhs0Stride, rhs1Base, rhs1Aligned,
		    rhs1Offset, rhs1Bytes, rhs1Stride, rhs2Base, rhs2Aligned, rhs2Offset, rhs2Bytes, rhs2Stride, outBase,
		    outAligned, outOffset, outRows, outColumns, outRowStride, outColumnStride, out0Columns, out1Columns,
		    out2Columns, requestedThreadCount, affinityPolicyValue);
	}

	template <typename IndexT>
	void LiteNNCPUGGMLBlockGetRowsF32(const std::uint8_t* storageAligned, std::int64_t storageOffset,
	                                  std::int64_t storageBytes, std::int64_t storageStride,
	                                  const IndexT* indicesAligned, std::int64_t indicesOffset,
	                                  std::int64_t indicesCount, std::int64_t indicesStride, float* outAligned,
	                                  std::int64_t outOffset, std::int64_t outRows, std::int64_t outColumns,
	                                  std::int64_t outRowStride, std::int64_t outColumnStride,
	                                  std::uint64_t formatValue)
	{
		const auto format = static_cast<QuantizedBlockFormat>(formatValue);
		const auto layout = GetQuantizedBlockLayout(format);
		if (!layout || storageBytes < 0 || indicesCount < 0 || outRows < 0 || outColumns < 0 ||
		    indicesCount != outRows || outColumns == 0 ||
		    static_cast<std::uint64_t>(outColumns) % layout->elementsPerBlock != 0)
		{
			return;
		}
		const auto rowBytes =
		    (static_cast<std::uint64_t>(outColumns) / layout->elementsPerBlock) * layout->bytesPerBlock;
		if (rowBytes == 0 || static_cast<std::uint64_t>(storageBytes) < rowBytes)
		{
			return;
		}
		const auto tableRows = static_cast<std::uint64_t>(storageBytes) / rowBytes;
		for (std::int64_t outputRow = 0; outputRow < outRows; ++outputRow)
		{
			const auto sourceRow = indicesAligned[indicesOffset + outputRow * indicesStride];
			if (sourceRow < 0 || static_cast<std::uint64_t>(sourceRow) >= tableRows)
			{
				return;
			}
			const auto* rowBase =
			    storageAligned + storageOffset +
			    static_cast<std::int64_t>(sourceRow) * static_cast<std::int64_t>(rowBytes) * storageStride;
			const auto blockCount = static_cast<std::uint64_t>(outColumns) / layout->elementsPerBlock;
			auto* outRow = outAligned + outOffset + outputRow * outRowStride;
			for (std::uint64_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
			{
				const auto* block = rowBase + blockIndex * layout->bytesPerBlock * storageStride;
				auto* outBlock =
				    outRow + static_cast<std::int64_t>(blockIndex * layout->elementsPerBlock) * outColumnStride;
				DecodeGGMLBlockF32(block, storageStride, outBlock, outColumnStride, format);
			}
		}
	}

	extern "C" void litenn_cpu_ggml_block_get_rows_i32_f32(
	    const std::uint8_t*, const std::uint8_t* storageAligned, std::int64_t storageOffset, std::int64_t storageBytes,
	    std::int64_t storageStride, const std::int32_t*, const std::int32_t* indicesAligned, std::int64_t indicesOffset,
	    std::int64_t indicesCount, std::int64_t indicesStride, float*, float* outAligned, std::int64_t outOffset,
	    std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride,
	    std::uint64_t formatValue)
	{
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_get_rows_i32_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? std::format("format={} rows={} columns={} storage_bytes={}",
		                      QuantizedBlockFormatName(static_cast<QuantizedBlockFormat>(formatValue)), outRows,
		                      outColumns, storageBytes)
		        : std::string{});
		LiteNNCPUGGMLBlockGetRowsF32(storageAligned, storageOffset, storageBytes, storageStride, indicesAligned,
		                             indicesOffset, indicesCount, indicesStride, outAligned, outOffset, outRows,
		                             outColumns, outRowStride, outColumnStride, formatValue);
	}

	extern "C" void litenn_cpu_ggml_block_get_rows_i64_f32(
	    const std::uint8_t*, const std::uint8_t* storageAligned, std::int64_t storageOffset, std::int64_t storageBytes,
	    std::int64_t storageStride, const std::int64_t*, const std::int64_t* indicesAligned, std::int64_t indicesOffset,
	    std::int64_t indicesCount, std::int64_t indicesStride, float*, float* outAligned, std::int64_t outOffset,
	    std::int64_t outRows, std::int64_t outColumns, std::int64_t outRowStride, std::int64_t outColumnStride,
	    std::uint64_t formatValue)
	{
		CPUAOTHelperProfileTimer profileTimer(
		    "litenn_cpu_ggml_block_get_rows_i64_f32",
		    CompiledModuleCPUHelperProfilerAccess::Enabled()
		        ? std::format("format={} rows={} columns={} storage_bytes={}",
		                      QuantizedBlockFormatName(static_cast<QuantizedBlockFormat>(formatValue)), outRows,
		                      outColumns, storageBytes)
		        : std::string{});
		LiteNNCPUGGMLBlockGetRowsF32(storageAligned, storageOffset, storageBytes, storageStride, indicesAligned,
		                             indicesOffset, indicesCount, indicesStride, outAligned, outOffset, outRows,
		                             outColumns, outRowStride, outColumnStride, formatValue);
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
	constexpr std::uint32_t kRodataVersion = 6;
	constexpr std::uint32_t kSeparatedMetadataVersion = 2;
	constexpr std::uint32_t kRodataLittleEndian = 1;
	constexpr std::uint32_t kRodataBigEndian = 2;
	constexpr std::string_view kMetadataRegionName = "metadata";
	constexpr std::string_view kConstantsRegionName = "constants";
	constexpr std::string_view kWeightsRegionName = "weights";
	constexpr std::string_view kInstructionsRegionName = "instructions";
	constexpr std::uint64_t kCPUAOTBoundedActivationMathFeature = UINT64_C(1) << 0;
	constexpr std::uint64_t kKnownCompiledModuleRuntimeFeatures = kCPUAOTBoundedActivationMathFeature;

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
		std::uint64_t requiredRuntimeFeatures{};
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

	void StripStateAliasUnsafeAttributes(llvm::Module& module)
	{
		constexpr std::array aliasSensitiveAttrs = {
			llvm::Attribute::NoAlias,
			llvm::Attribute::ReadNone,
			llvm::Attribute::ReadOnly,
			llvm::Attribute::WriteOnly,
		};
		const auto stripParamAttrs = [&](auto& callable, unsigned argCount) {
			for (unsigned arg = 0; arg < argCount; ++arg)
			{
				for (const auto attr : aliasSensitiveAttrs)
				{
					callable.removeParamAttr(arg, attr);
				}
			}
		};
		const auto stripFnAttrs = [](auto& callable) {
			callable.removeFnAttr(llvm::Attribute::ReadNone);
			callable.removeFnAttr(llvm::Attribute::ReadOnly);
			callable.removeFnAttr(llvm::Attribute::WriteOnly);
		};

		for (auto& function : module)
		{
			stripParamAttrs(function, function.arg_size());
			stripFnAttrs(function);
			for (auto& block : function)
			{
				for (auto& instruction : block)
				{
					auto* call = llvm::dyn_cast<llvm::CallBase>(&instruction);
					if (!call)
					{
						continue;
					}
					stripParamAttrs(*call, call->arg_size());
					stripFnAttrs(*call);
				}
			}
		}
	}

	std::uint8_t EffectiveCPUAOTLLVMOptLevel(const CompilerOptions& options,
	                                         const Runtime::RuntimeScheduleOutputProjection* outputProjection)
	{
		if (options.cpuAOTLLVMOptLevel <= 2 || outputProjection == nullptr || outputProjection->stateAliases.empty())
		{
			return options.cpuAOTLLVMOptLevel;
		}
		LogCompileDiagnostic(
		    options,
		    "cpu-aot llvm opt level O3 is downgraded to O2 for state-alias schedules until O3 alias safety is proven");
		return 2;
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
		AppendU32(out, static_cast<std::uint32_t>(params.storageLayout));
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
		params.storageLayout = static_cast<QuantizedStorageLayout>(ReadU32(bytes, offset));
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

	std::uint64_t CPUAOTRequiredRuntimeFeatures(const CompilerOptions& options)
	{
		return options.cpuAOTActivationMathPolicy == CPUAOTActivationMathPolicy::Bounded
		           ? kCPUAOTBoundedActivationMathFeature
		           : 0;
	}

	void ValidateCPUAOTCompilerOptions(const CompilerOptions& options)
	{
		switch (options.cpuAOTActivationMathPolicy)
		{
		case CPUAOTActivationMathPolicy::Strict:
			return;
		case CPUAOTActivationMathPolicy::Bounded:
			if (IsCPUAOTActivationMathPolicySupported(options.cpuAOTActivationMathPolicy))
			{
				return;
			}
			throw std::invalid_argument("CPU AOT bounded activation math requires a configured vector-math provider");
		default:
			throw std::invalid_argument("CPU AOT activation math policy is invalid");
		}
	}

	void ValidateCompiledModuleRuntimeFeatures(CompiledModuleBackend backend, std::uint64_t features)
	{
		if ((features & ~kKnownCompiledModuleRuntimeFeatures) != 0)
		{
			throw std::runtime_error("Compiled module rodata requires unknown runtime features");
		}
		if ((features & kCPUAOTBoundedActivationMathFeature) != 0)
		{
			if (backend != CompiledModuleBackend::CPUNative)
			{
				throw std::runtime_error(
				    "Compiled module rodata applies the CPU bounded activation feature to a non-CPU backend");
			}
			if (!IsCPUAOTActivationMathPolicySupported(CPUAOTActivationMathPolicy::Bounded))
			{
				throw std::runtime_error(
				    "Compiled module requires CPU bounded activation math, but no vector-math provider is available");
			}
		}
	}

	std::vector<std::byte> SerializeRodata(std::span<const CompiledTensorSpec> inputs,
	                                       std::span<const CompiledTensorSpec> outputs, std::string_view targetTriple,
	                                       CompiledModuleBackend backend, std::uint64_t requiredRuntimeFeatures = 0)
	{
		ValidateCompiledModuleRuntimeFeatures(backend, requiredRuntimeFeatures);
		std::vector<std::byte> rodata;
		rodata.insert(rodata.end(), kRodataMagic.begin(), kRodataMagic.end());
		AppendU32(rodata, kRodataVersion);
		AppendU32(rodata, static_cast<std::uint32_t>(sizeof(void*)));
		AppendU32(rodata, NativeEndianTag());
		AppendString(rodata, targetTriple);
		AppendU32(rodata, static_cast<std::uint32_t>(backend));
		AppendU64(rodata, requiredRuntimeFeatures);
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
		if (version != kRodataVersion)
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
		const auto requiredRuntimeFeatures = ReadU64(rodata, offset);
		ValidateCompiledModuleRuntimeFeatures(backend, requiredRuntimeFeatures);

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
			.requiredRuntimeFeatures = requiredRuntimeFeatures,
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

	void ValidateSeparatedRegion(CompiledModuleRegion region, const CompiledModuleRegionInfo& expected,
	                             bool validateChecksum = true)
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
		if (validateChecksum && ChecksumBytes(bytes) != expected.checksum)
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

	SeparatedMetadata ValidateSeparatedImage(CompiledModuleSeparatedImage image, bool validateWeightsChecksum = true)
	{
		const auto metadataBytes = RegionBytes(image.metadata, kMetadataRegionName);
		auto metadata = DeserializeSeparatedMetadata(metadataBytes);
		ValidateSeparatedRegion(image.constants, FindRegionInfo(metadata.regions, kConstantsRegionName));
		ValidateSeparatedRegion(image.weights, FindRegionInfo(metadata.regions, kWeightsRegionName),
		                        validateWeightsChecksum);
		ValidateSeparatedRegion(image.instructions, FindRegionInfo(metadata.regions, kInstructionsRegionName));
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

	std::vector<CompiledTensorSpec>
	BuildProjectedOutputSpecs(std::span<const CompiledTensorSpec> functionalOutputs,
	                          const Runtime::RuntimeScheduleOutputProjection& projection)
	{
		std::vector<CompiledTensorSpec> specs;
		specs.reserve(projection.publicOutputIndices.size());
		for (const auto outputIndex : projection.publicOutputIndices)
		{
			if (outputIndex >= functionalOutputs.size())
			{
				throw std::runtime_error("Runtime schedule output projection references an unknown output spec");
			}
			specs.push_back(functionalOutputs[outputIndex]);
		}
		return specs;
	}

	std::vector<CompiledTensorSpec> BuildEntryOutputSpecs(std::span<const CompiledTensorSpec> functionalOutputs,
	                                                      const Runtime::RuntimeScheduleOutputProjection* projection)
	{
		if (!projection)
		{
			return std::vector<CompiledTensorSpec>(functionalOutputs.begin(), functionalOutputs.end());
		}
		return BuildProjectedOutputSpecs(functionalOutputs, *projection);
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
		return builder.CreateGEP(i8Ty, base, builder.getInt64(offset));
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

	bool IsCPUAOTPrepackedGGMLFormat(QuantizedBlockFormat format)
	{
		return format == QuantizedBlockFormat::GGML_Q4_K || format == QuantizedBlockFormat::GGML_Q6_K;
	}

	bool ShouldPrepackCPUAOTGGMLFormat(QuantizedBlockFormat format, const CompilerOptions& options)
	{
		if (options.enableCPUAOTGGMLPrepackedWeights)
		{
			return IsCPUAOTPrepackedGGMLFormat(format);
		}
		switch (options.cpuAOTGGMLPrepackedWeightPolicy)
		{
		case CPUAOTGGMLPrepackedWeightPolicy::Disabled:
			return false;
		case CPUAOTGGMLPrepackedWeightPolicy::Profitable:
			return format == QuantizedBlockFormat::GGML_Q6_K;
		case CPUAOTGGMLPrepackedWeightPolicy::All:
			return IsCPUAOTPrepackedGGMLFormat(format);
		}
		return false;
	}

	std::optional<std::uint64_t> GGMLPrepackedBlockBytes(QuantizedBlockFormat format)
	{
		switch (format)
		{
		case QuantizedBlockFormat::GGML_Q4_K:
			return kGGMLQ4KPreparedBlockBytes;
		case QuantizedBlockFormat::GGML_Q6_K:
			return kGGMLQ6KPreparedBlockBytes;
		default:
			return std::nullopt;
		}
	}

	constexpr std::string_view kGGMLPrepackedExpandedF32ScalesV1Name = "expanded_f32_scales_v1";
	constexpr std::string_view kGGMLPrepackedCompactBlockGroupedV3Name = "compact_block_grouped_v3";
	constexpr std::string_view kGGMLPrepackedFieldInterleavedV4Name = "field_interleaved_v4";

	std::string_view GGMLPrepackedLayoutName(CPUAOTGGMLPrepackedWeightLayout preparedLayout)
	{
		switch (preparedLayout)
		{
		case CPUAOTGGMLPrepackedWeightLayout::ExpandedF32ScalesV1:
			return kGGMLPrepackedExpandedF32ScalesV1Name;
		case CPUAOTGGMLPrepackedWeightLayout::CompactBlockGroupedV3:
			return kGGMLPrepackedCompactBlockGroupedV3Name;
		case CPUAOTGGMLPrepackedWeightLayout::FieldInterleavedV4:
			return kGGMLPrepackedFieldInterleavedV4Name;
		}
		return "unknown";
	}

	QuantizedStorageLayout GGMLPrepackedQuantizedStorageLayout(CPUAOTGGMLPrepackedWeightLayout preparedLayout)
	{
		switch (preparedLayout)
		{
		case CPUAOTGGMLPrepackedWeightLayout::ExpandedF32ScalesV1:
			return QuantizedStorageLayout::GGMLExpandedF32ScalesV1;
		case CPUAOTGGMLPrepackedWeightLayout::CompactBlockGroupedV3:
			return QuantizedStorageLayout::GGMLCompactBlockGroupedV3;
		case CPUAOTGGMLPrepackedWeightLayout::FieldInterleavedV4:
			return QuantizedStorageLayout::GGMLFieldInterleavedV4;
		}
		return QuantizedStorageLayout::Source;
	}

	std::optional<std::vector<std::size_t>> GGMLPrepackedStorageShape(const QuantizationParams& params,
	                                                                  CPUAOTGGMLPrepackedWeightLayout preparedLayout)
	{
		if (params.scheme != QuantizationScheme::Block || !IsCPUAOTPrepackedGGMLFormat(params.blockFormat) ||
		    params.storageType != DataType::UInt8 || params.expressedShape.size() != 2)
		{
			return std::nullopt;
		}
		const auto layout = GetQuantizedBlockLayout(params.blockFormat);
		if (!layout || params.expressedShape[1] % layout->elementsPerBlock != 0)
		{
			return std::nullopt;
		}
		if (preparedLayout == CPUAOTGGMLPrepackedWeightLayout::CompactBlockGroupedV3)
		{
			const auto compactBytes =
			    GGMLCompactInterleavedByteSize(params.blockFormat, static_cast<std::uint64_t>(params.expressedShape[0]),
			                                   static_cast<std::uint64_t>(params.expressedShape[1]));
			if (!compactBytes || *compactBytes > std::numeric_limits<std::size_t>::max())
			{
				return std::nullopt;
			}
			return std::vector<std::size_t>{ static_cast<std::size_t>(*compactBytes) };
		}
		if (preparedLayout == CPUAOTGGMLPrepackedWeightLayout::FieldInterleavedV4)
		{
			const auto interleavedBytes =
			    GGMLFieldInterleavedV4ByteSize(params.blockFormat, static_cast<std::uint64_t>(params.expressedShape[0]),
			                                   static_cast<std::uint64_t>(params.expressedShape[1]));
			if (!interleavedBytes || *interleavedBytes > std::numeric_limits<std::size_t>::max())
			{
				return std::nullopt;
			}
			return std::vector<std::size_t>{ static_cast<std::size_t>(*interleavedBytes) };
		}
		if (preparedLayout != CPUAOTGGMLPrepackedWeightLayout::ExpandedF32ScalesV1)
		{
			return std::nullopt;
		}
		const auto preparedBlockBytes = GGMLPrepackedBlockBytes(params.blockFormat);
		if (!preparedBlockBytes)
		{
			return std::nullopt;
		}
		const auto blockCount = params.expressedShape[1] / layout->elementsPerBlock;
		if (params.expressedShape[0] > std::numeric_limits<std::size_t>::max() / blockCount ||
		    params.expressedShape[0] * blockCount > std::numeric_limits<std::size_t>::max() / *preparedBlockBytes)
		{
			return std::nullopt;
		}
		return std::vector<std::size_t>{ params.expressedShape[0] * blockCount *
			                             static_cast<std::size_t>(*preparedBlockBytes) };
	}

	std::optional<std::uint64_t> AppendGGMLPrepackedPayloadBytes(std::vector<std::byte>& bytes,
	                                                             const Tensor<PolymorphicDevice>& tensor,
	                                                             const QuantizationParams& params,
	                                                             CPUAOTGGMLPrepackedWeightLayout preparedLayout,
	                                                             std::uint64_t alignment)
	{
		const auto preparedShape = GGMLPrepackedStorageShape(params, preparedLayout);
		const auto layout = GetQuantizedBlockLayout(params.blockFormat);
		if (!preparedShape || !layout || tensor.DType() != DataType::UInt8 || tensor.Shape().NumDim() != 1)
		{
			return std::nullopt;
		}
		const auto rows = params.expressedShape[0];
		const auto columns = params.expressedShape[1];
		const auto blockCount = columns / layout->elementsPerBlock;
		const auto sourceRowBytes = blockCount * layout->bytesPerBlock;
		if (tensor.NumElements() < rows * sourceRowBytes)
		{
			return std::nullopt;
		}

		const auto offset = AlignUpU64(static_cast<std::uint64_t>(bytes.size()), alignment);
		if (bytes.size() < offset)
		{
			bytes.resize(static_cast<std::size_t>(offset));
		}
		const auto oldSize = bytes.size();
		const auto preparedByteSize = (*preparedShape)[0];
		bytes.resize(oldSize + preparedByteSize);
		std::optional<Tensor<CPU>> ownedCPU;
		const void* sourceRaw = tensor.UnsafeRawData();
		if (tensor.CurDevice().template As<CPU>() == nullptr)
		{
			ownedCPU = tensor.CopyToDevice(CPU{});
			sourceRaw = ownedCPU->UnsafeRawData();
		}
		const auto* source = static_cast<const std::uint8_t*>(sourceRaw);
		auto* target = reinterpret_cast<std::uint8_t*>(bytes.data() + oldSize);
		if (preparedLayout == CPUAOTGGMLPrepackedWeightLayout::CompactBlockGroupedV3)
		{
			litenn_cpu_ggml_prepack_compact_interleaved(
			    source, source, 0, static_cast<std::int64_t>(tensor.NumElements()), 1, static_cast<std::int64_t>(rows),
			    static_cast<std::int64_t>(columns), static_cast<std::uint64_t>(params.blockFormat), target, target, 0,
			    static_cast<std::int64_t>(preparedByteSize), 1);
			return offset;
		}
		if (preparedLayout == CPUAOTGGMLPrepackedWeightLayout::FieldInterleavedV4)
		{
			litenn_cpu_ggml_prepack_field_interleaved_v4(
			    source, source, 0, static_cast<std::int64_t>(tensor.NumElements()), 1, static_cast<std::int64_t>(rows),
			    static_cast<std::int64_t>(columns), static_cast<std::uint64_t>(params.blockFormat), target, target, 0,
			    static_cast<std::int64_t>(preparedByteSize), 1);
			return offset;
		}
		const auto preparedBlockBytes = GGMLPrepackedBlockBytes(params.blockFormat);
		if (!preparedBlockBytes)
		{
			bytes.resize(oldSize);
			return std::nullopt;
		}
		const auto preparedRowBytes = blockCount * *preparedBlockBytes;
		for (std::size_t row = 0; row < rows; ++row)
		{
			for (std::size_t blockIndex = 0; blockIndex < blockCount; ++blockIndex)
			{
				const auto* sourceBlock = source + row * sourceRowBytes + blockIndex * layout->bytesPerBlock;
				auto* targetBlock = target + row * preparedRowBytes + blockIndex * *preparedBlockBytes;
				if (params.blockFormat == QuantizedBlockFormat::GGML_Q4_K)
				{
					PrepareGGMLQ4KBlockF32(sourceBlock, 1, targetBlock);
				}
				else
				{
					PrepareGGMLQ6KBlockF32(sourceBlock, 1, targetBlock);
				}
			}
		}
		return offset;
	}

	std::optional<std::size_t> TryGetVariableRefIndex(const Subgraph& subgraph, NodeOutput output)
	{
		if (output.port != 0 || output.node >= subgraph.NodeCount())
		{
			return std::nullopt;
		}
		const auto* variable = std::get_if<VariableRefNode>(&subgraph.GetNodeEntry(output.node).node);
		return variable ? std::optional<std::size_t>{ variable->variableIndex } : std::nullopt;
	}

	std::vector<std::optional<QuantizationParams>> BuildCPUAOTPrepackedGGMLVariablePlans(const Graph& graph,
	                                                                                     const CompilerOptions& options)
	{
		std::vector<std::optional<QuantizationParams>> plans(graph.VariableCount());
		if (!options.enableCPUAOTGGMLPrepackedWeights &&
		    options.cpuAOTGGMLPrepackedWeightPolicy == CPUAOTGGMLPrepackedWeightPolicy::Disabled)
		{
			return plans;
		}

		std::vector<std::size_t> variableRefCounts(graph.VariableCount());
		std::vector<std::size_t> prepackUseCounts(graph.VariableCount());
		std::vector<bool> rejected(graph.VariableCount());
		const auto recordPrepackUse = [&](const Subgraph& subgraph, NodeOutput storage,
		                                  const QuantizationParams& params, bool forceForGroup = false) {
			if (!forceForGroup && !ShouldPrepackCPUAOTGGMLFormat(params.blockFormat, options))
			{
				return;
			}
			const auto variableIndex = TryGetVariableRefIndex(subgraph, storage);
			if (!variableIndex || *variableIndex >= plans.size())
			{
				return;
			}
			auto& plan = plans[*variableIndex];
			if (plan && (plan->blockFormat != params.blockFormat || plan->expressedShape != params.expressedShape))
			{
				rejected[*variableIndex] = true;
				return;
			}
			plan = params;
			++prepackUseCounts[*variableIndex];
		};
		for (SubgraphId subgraphId = 0; subgraphId < graph.SubgraphCount(); ++subgraphId)
		{
			const auto& subgraph = graph.GetSubgraph(subgraphId);
			for (const auto& entry : subgraph.Nodes())
			{
				if (const auto* variable = std::get_if<VariableRefNode>(&entry.node);
				    variable && variable->variableIndex < variableRefCounts.size())
				{
					++variableRefCounts[variable->variableIndex];
				}
			}
			for (const auto& entry : subgraph.Nodes())
			{
				if (const auto* matmul = std::get_if<QuantizedMatMulNode>(&entry.node))
				{
					if (!matmul->transposeRhs)
					{
						continue;
					}
					recordPrepackUse(subgraph, matmul->rhsStorage, matmul->params);
				}
				else if (const auto* grouped = std::get_if<GroupedQuantizedMatMulNode>(&entry.node))
				{
					if (!grouped->transposeRhs || grouped->rhsStorages.size() != grouped->outputWidths.size())
					{
						continue;
					}
					if (grouped->rhsStorages.size() != grouped->projectionParams.size())
					{
						continue;
					}
					const auto hasMixedFormats =
					    std::ranges::any_of(grouped->projectionParams, [&](const auto& params) {
						    return params.blockFormat != grouped->projectionParams.front().blockFormat;
					    });
					if (hasMixedFormats &&
					    options.cpuAOTGGMLPrepackedWeightLayout != CPUAOTGGMLPrepackedWeightLayout::FieldInterleavedV4)
					{
						continue;
					}
					const auto prepackGroup = std::ranges::any_of(grouped->projectionParams, [&](const auto& params) {
						return ShouldPrepackCPUAOTGGMLFormat(params.blockFormat, options);
					});
					if (!prepackGroup)
					{
						continue;
					}
					for (std::size_t i = 0; i < grouped->rhsStorages.size(); ++i)
					{
						recordPrepackUse(subgraph, grouped->rhsStorages[i], grouped->projectionParams[i], true);
					}
				}
			}
		}

		for (std::size_t i = 0; i < plans.size(); ++i)
		{
			if (!plans[i] || rejected[i] || variableRefCounts[i] != prepackUseCounts[i] ||
			    !graph.GetVariable(i)->Quantization() ||
			    !GGMLPrepackedStorageShape(*plans[i], options.cpuAOTGGMLPrepackedWeightLayout))
			{
				plans[i] = std::nullopt;
			}
			else
			{
				plans[i]->storageLayout = GGMLPrepackedQuantizedStorageLayout(options.cpuAOTGGMLPrepackedWeightLayout);
			}
		}

		// A grouped helper has one physical-layout contract for all projections. Shared variables can make an
		// otherwise eligible member fall back during the per-variable validation above, so repeatedly clear the
		// remaining members of any partially selected group until all groups are layout-consistent.
		bool changed;
		do
		{
			changed = false;
			for (SubgraphId subgraphId = 0; subgraphId < graph.SubgraphCount(); ++subgraphId)
			{
				const auto& subgraph = graph.GetSubgraph(subgraphId);
				for (const auto& entry : subgraph.Nodes())
				{
					const auto* grouped = std::get_if<GroupedQuantizedMatMulNode>(&entry.node);
					if (!grouped)
					{
						continue;
					}
					std::vector<std::size_t> variables;
					variables.reserve(grouped->rhsStorages.size());
					for (const auto storage : grouped->rhsStorages)
					{
						const auto variable = TryGetVariableRefIndex(subgraph, storage);
						if (!variable || *variable >= plans.size())
						{
							variables.clear();
							break;
						}
						variables.push_back(*variable);
					}
					const auto selected = std::ranges::count_if(
					    variables, [&](const auto variable) { return plans[variable].has_value(); });
					if (selected == 0 || selected == variables.size())
					{
						continue;
					}
					for (const auto variable : variables)
					{
						changed = plans[variable].has_value() || changed;
						plans[variable] = std::nullopt;
					}
				}
			}
		} while (changed);
		return plans;
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

	bool IsPromotedConstantVariableName(std::string_view name)
	{
		return name.starts_with("constant.");
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
			    else if constexpr (std::same_as<T, GroupedQuantizedMatMulNode>)
			    {
				    copy.lhs = remap(copy.lhs);
				    for (auto& rhsStorage : copy.rhsStorages)
				    {
					    rhsStorage = remap(rhsStorage);
				    }
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
			    else if constexpr (std::same_as<T, ActivePrefixAttentionNode>)
			    {
				    copy.query = remap(copy.query);
				    copy.keys = remap(copy.keys);
				    copy.values = remap(copy.values);
				    copy.currentPosition = remap(copy.currentPosition);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, GroupedActivePrefixAttentionNode>)
			    {
				    copy.queries = remap(copy.queries);
				    copy.keys = remap(copy.keys);
				    copy.values = remap(copy.values);
				    copy.currentPosition = remap(copy.currentPosition);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, GroupedPagedAttentionNode>)
			    {
				    copy.queries = remap(copy.queries);
				    copy.kvState = remap(copy.kvState);
				    copy.pageTable = remap(copy.pageTable);
				    copy.pageDescriptors = remap(copy.pageDescriptors);
				    copy.activeLength = remap(copy.activeLength);
				    return copy;
			    }
			    else if constexpr (std::same_as<T, PagedKVAppendNode>)
			    {
				    copy.kvState = remap(copy.kvState);
				    copy.pageTable = remap(copy.pageTable);
				    copy.pageDescriptors = remap(copy.pageDescriptors);
				    copy.activeLength = remap(copy.activeLength);
				    copy.keys = remap(copy.keys);
				    copy.values = remap(copy.values);
				    copy.position = remap(copy.position);
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
		const auto prepackedVariablePlans = BuildCPUAOTPrepackedGGMLVariablePlans(graph, options);
		std::uint64_t projectedWeightBytes = 0;
		for (std::size_t variableIndex = 0; variableIndex < graph.VariableCount(); ++variableIndex)
		{
			const auto& data = graph.GetVariable(variableIndex)->Data();
			const auto variableBytes =
			    prepackedVariablePlans[variableIndex] &&
			            GGMLPrepackedStorageShape(*prepackedVariablePlans[variableIndex],
			                                      options.cpuAOTGGMLPrepackedWeightLayout)
			        ? static_cast<std::uint64_t>((*GGMLPrepackedStorageShape(
			              *prepackedVariablePlans[variableIndex], options.cpuAOTGGMLPrepackedWeightLayout))[0])
			        : static_cast<std::uint64_t>(data.NumElements()) * ElementByteSize(data.DType());
			projectedWeightBytes = AlignUpU64(projectedWeightBytes, 64);
			projectedWeightBytes += variableBytes;
		}
		if (projectedWeightBytes > std::numeric_limits<std::size_t>::max())
		{
			throw std::runtime_error("CPU MLIR external weight region exceeds the host address space");
		}
		result.weights.reserve(static_cast<std::size_t>(projectedWeightBytes));
		std::unordered_map<std::size_t, std::size_t> variableExternalIdMap;
		std::vector<std::optional<std::size_t>> inlineVariableMap(graph.VariableCount());
		std::vector<std::vector<std::optional<std::size_t>>> directExternalByNode(graph.SubgraphCount());
		std::vector<std::vector<std::size_t>> externalDepsBySubgraph(graph.SubgraphCount());
		const auto ensureInlineVariable = [&](std::size_t variableIndex) -> std::size_t {
			if (variableIndex >= graph.VariableCount())
			{
				throw std::runtime_error("CPU MLIR externalization references an unknown inline variable");
			}
			if (!inlineVariableMap[variableIndex])
			{
				const auto newIndex = result.graph.AddVariable(graph.GetVariable(variableIndex));
				result.graph.SetVariableName(newIndex, graph.VariableName(variableIndex));
				inlineVariableMap[variableIndex] = newIndex;
			}
			return *inlineVariableMap[variableIndex];
		};

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
					const auto name = graph.VariableName(variable->variableIndex);
					const auto byteSize = TensorByteSizeForShape(output.dtype, output.shape);
					if (IsPromotedConstantVariableName(name) && byteSize < options.cpuAOTExternalConstantMinBytes)
					{
						ensureInlineVariable(variable->variableIndex);
						continue;
					}
					auto [it, inserted] = variableExternalIdMap.emplace(variable->variableIndex, 0);
					if (inserted)
					{
						constexpr std::uint64_t kAlignment = 64;
						const auto& variableData = graph.GetVariable(variable->variableIndex)->Data();
						const auto& prepackedPlan = prepackedVariablePlans[variable->variableIndex];
						const auto prepackedShape =
						    prepackedPlan
						        ? GGMLPrepackedStorageShape(*prepackedPlan, options.cpuAOTGGMLPrepackedWeightLayout)
						        : std::nullopt;
						const auto offset =
						    prepackedPlan && prepackedShape
						        ? AppendGGMLPrepackedPayloadBytes(result.weights, variableData, *prepackedPlan,
						                                          options.cpuAOTGGMLPrepackedWeightLayout, kAlignment)
						        : AppendTensorPayloadBytes(result.weights, variableData, output.dtype, output.shape,
						                                   kAlignment);
						if (!offset)
						{
							return std::nullopt;
						}
						it->second = result.externalTensorInfos.size();
						const auto externalShape =
						    prepackedShape
						        ? std::span<const std::size_t>{ prepackedShape->data(), prepackedShape->size() }
						        : std::span<const std::size_t>{ output.shape.data(), output.shape.size() };
						const auto externalByteSize = prepackedShape
						                                  ? static_cast<std::uint64_t>((*prepackedShape)[0])
						                                  : TensorByteSizeForShape(output.dtype, output.shape);
						result.externalTensorInfos.push_back(MakeExternalTensorInfo(
						    prepackedShape
						        ? std::format("{}.prepacked.{}.{}", name,
						                      GGMLPrepackedLayoutName(options.cpuAOTGGMLPrepackedWeightLayout),
						                      QuantizedBlockFormatName(prepackedPlan->blockFormat))
						        : name,
						    kWeightsRegionName, output.dtype, result.weights, externalShape, *offset, externalByteSize,
						    kAlignment));
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
				if (auto* quantizedMatMul = std::get_if<QuantizedMatMulNode>(&remappedNode))
				{
					const auto& originalNode = std::get<QuantizedMatMulNode>(entry.node);
					if (const auto variableIndex = TryGetVariableRefIndex(original, originalNode.rhsStorage);
					    variableIndex && *variableIndex < prepackedVariablePlans.size() &&
					    prepackedVariablePlans[*variableIndex])
					{
						quantizedMatMul->params.storageLayout = prepackedVariablePlans[*variableIndex]->storageLayout;
					}
				}
				else if (auto* groupedMatMul = std::get_if<GroupedQuantizedMatMulNode>(&remappedNode))
				{
					const auto& originalNode = std::get<GroupedQuantizedMatMulNode>(entry.node);
					if (groupedMatMul->projectionParams.size() != originalNode.rhsStorages.size())
					{
						return std::nullopt;
					}
					for (std::size_t i = 0; i < originalNode.rhsStorages.size(); ++i)
					{
						const auto variableIndex = TryGetVariableRefIndex(original, originalNode.rhsStorages[i]);
						if (!variableIndex || *variableIndex >= prepackedVariablePlans.size() ||
						    !prepackedVariablePlans[*variableIndex])
						{
							continue;
						}
						groupedMatMul->projectionParams[i].storageLayout =
						    prepackedVariablePlans[*variableIndex]->storageLayout;
					}
				}
				if (auto* variable = std::get_if<VariableRefNode>(&remappedNode))
				{
					if (variable->variableIndex >= inlineVariableMap.size() ||
					    !inlineVariableMap[variable->variableIndex])
					{
						return std::nullopt;
					}
					variable->variableIndex = *inlineVariableMap[variable->variableIndex];
				}
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
		const auto reject = [&](std::string_view reason) -> std::optional<CompiledArtifactParts> {
			LogCompileDiagnostic(options, std::string("cpu-parallel linear-chain rejected: ") + std::string(reason));
			return std::nullopt;
		};
		const auto threadCount = ResolveCPUAOTThreadCount(options);
		if (threadCount <= 1)
		{
			return reject("thread_count<=1");
		}

		if (graph.Backward().has_value() || graph.ActivationSlotCount() != 0 || graph.TapeSlotCount() != 0 ||
		    graph.SubgraphCount() == 0)
		{
			return reject("graph has backward/tape/activation state or no forward subgraph");
		}

		const auto& subgraph = graph.GetSubgraph(graph.Forward());
		if (subgraph.Results().size() != 1)
		{
			return reject("requires exactly one forward result");
		}
		const auto finalResult = subgraph.Results()[0];
		if (finalResult.port != 0 || finalResult.node >= subgraph.NodeCount())
		{
			return reject("forward result is not a single node output");
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
				return reject("all chain nodes must have exactly one output");
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
					return reject("variable reference is out of range");
				}
				auto constantData = CopyF32TensorData(graph.GetVariable(variable->variableIndex)->Data(), output.shape);
				if (!constantData)
				{
					return reject("variable is not a static f32 tensor");
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
					return reject("constant is not a static f32 tensor");
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
				return reject("encountered a non MatMulBiasAdd/ReLU fused node");
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
				return reject("fused layer dtype/shape/bias contract is unsupported");
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
				return reject("m>256 keeps packed MLIR fallback");
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
			                     builder.getInt64(EncodeCPUAOTSchedulingPolicy(options.cpuAOTAffinityPolicy,
			                                                                   options.cpuAOTWorkerWaitPolicy)),
			                     builder.getInt1(fused->pattern == FusionPattern::MatMulBiasAddReLU) });
			values[nodeId] = ValueRef{ .ptr = outPtr, .dtype = output.dtype, .shape = output.shape };
			++fusedLayerCount;
		}

		if (fusedLayerCount == 0 || !hasParallelEligibleLayer || !values[finalResult.node] ||
		    totalFlops < options.cpuAOTParallelMinFlops)
		{
			if (fusedLayerCount == 0)
			{
				return reject("no fused MatMulBiasAdd/ReLU layers");
			}
			if (!hasParallelEligibleLayer)
			{
				return reject("no layer selected more than one helper thread");
			}
			if (!values[finalResult.node])
			{
				return reject("final result was not materialized by the chain");
			}
			return reject("total_flops below cpuAOTParallelMinFlops");
		}
		for (auto it = heapAllocations.rbegin(); it != heapAllocations.rend(); ++it)
		{
			builder.CreateCall(freeFn, { *it });
		}
		builder.CreateRetVoid();

		const auto inputSpecs = BuildInputSpecs(graph);
		const auto outputSpecs = BuildOutputSpecs(graph);
		LogCompileDiagnostic(options, std::format("cpu-parallel linear-chain selected: fused_layers={} total_flops={} "
		                                          "threads={}",
		                                          fusedLayerCount, totalFlops, threadCount));
		auto config = CreateNativeTargetMachine();
		ConfigureForNativeObject(*module, config);
		OptimizeLLVMModule(*module, *config.targetMachine, options.cpuAOTLLVMOptLevel);
		auto rodata = SerializeRodata(inputSpecs, outputSpecs, config.triple, CompiledModuleBackend::CPUNative,
		                              CPUAOTRequiredRuntimeFeatures(options));
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

	void CopyDescriptorToArraySlot(llvm::IRBuilder<>& builder, llvm::Value* descriptor, llvm::Value* array,
	                               std::size_t index, const CompiledTensorSpec& spec)
	{
		auto& ctx = builder.getContext();
		auto* ptrTy = llvm::PointerType::get(ctx, 0);
		auto* outputSlot = builder.CreateGEP(ptrTy, array, builder.getInt64(index));
		auto* outputData = builder.CreateLoad(ptrTy, outputSlot);
		auto* descTy = llvm::cast<llvm::StructType>(descriptor->getType());
		const unsigned dataField = descTy->getNumElements() == 5 ? 1 : 0;
		auto* sourceData = builder.CreateExtractValue(descriptor, { dataField });
		const auto byteCount = NumElements(spec) * LiteNN::ElementByteSize(spec.type.dtype);
		builder.CreateMemCpy(outputData, llvm::Align(1), sourceData, llvm::Align(1), builder.getInt64(byteCount));
	}

	void CopyDescriptorToOutput(llvm::IRBuilder<>& builder, llvm::Value* descriptor, llvm::Value* outputArray,
	                            std::size_t outputIndex, const CompiledTensorSpec& spec)
	{
		CopyDescriptorToArraySlot(builder, descriptor, outputArray, outputIndex, spec);
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

	std::optional<std::size_t> FindPublicOutputSlot(const Runtime::RuntimeScheduleOutputProjection& projection,
	                                                std::size_t functionalOutputIndex)
	{
		for (std::size_t slot = 0; slot < projection.publicOutputIndices.size(); ++slot)
		{
			if (projection.publicOutputIndices[slot] == functionalOutputIndex)
			{
				return slot;
			}
		}
		return std::nullopt;
	}

	const Runtime::RuntimeStateOutputAlias*
	FindStateOutputAlias(const Runtime::RuntimeScheduleOutputProjection& projection, std::size_t functionalOutputIndex)
	{
		const auto it = std::ranges::find_if(
		    projection.stateAliases, [&](const auto& alias) { return alias.outputIndex == functionalOutputIndex; });
		return it == projection.stateAliases.end() ? nullptr : &*it;
	}

	llvm::AllocaInst* CreateOutputScratch(llvm::IRBuilder<>& builder, const CompiledTensorSpec& spec)
	{
		auto& ctx = builder.getContext();
		auto* scratch = builder.CreateAlloca(GetElementType(ctx, spec.type.dtype), builder.getInt64(NumElements(spec)));
		scratch->setAlignment(llvm::Align(64));
		return scratch;
	}

	void AddUniformEntryWrapper(llvm::Module& module, std::string_view calleeName,
	                            std::span<const CompiledTensorSpec> inputs, std::span<const CompiledTensorSpec> outputs,
	                            std::span<const CompiledModuleExternalTensorInfo> externalInputs = {},
	                            const Runtime::RuntimeScheduleOutputProjection* outputProjection = nullptr)
	{
		auto* callee = module.getFunction(calleeName);
		if (!callee)
		{
			throw std::runtime_error("Compiled subgraph function was not found in LLVM module");
		}
		if (outputProjection && outputProjection->functionalOutputCount != outputs.size())
		{
			throw std::runtime_error("Compiled subgraph output projection does not match functional outputs");
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
			llvm::Value* outputData = nullptr;
			if (outputProjection)
			{
				if (const auto publicSlot = FindPublicOutputSlot(*outputProjection, i))
				{
					auto* outputSlot = builder.CreateGEP(ptrTy, outputArray, builder.getInt64(*publicSlot));
					outputData = builder.CreateLoad(ptrTy, outputSlot);
				}
				else if (const auto* alias = FindStateOutputAlias(*outputProjection, i))
				{
					if (alias->inputIndex >= inputs.size())
					{
						throw std::runtime_error("Compiled subgraph output projection references an unknown input");
					}
					if (inputs[alias->inputIndex].type != outputs[i].type)
					{
						throw std::runtime_error("Compiled subgraph state-output projection type mismatch");
					}
					auto* inputSlot = builder.CreateGEP(ptrTy, inputArray, builder.getInt64(alias->inputIndex));
					outputData = builder.CreateLoad(ptrTy, inputSlot);
				}
				else
				{
					throw std::runtime_error("Compiled subgraph output projection is missing a functional output");
				}
			}
			else
			{
				auto* outputSlot = builder.CreateGEP(ptrTy, outputArray, builder.getInt64(i));
				outputData = builder.CreateLoad(ptrTy, outputSlot);
			}
			auto* descriptor = BuildMemRefDescriptor(builder, outputData, outputs[i]);
			outputDescriptors.push_back(descriptor);
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
				if (outputProjection)
				{
					if (const auto publicSlot = FindPublicOutputSlot(*outputProjection, 0))
					{
						CopyDescriptorToOutput(builder, descriptor, outputArray, *publicSlot, outputs[0]);
					}
					else if (const auto* alias = FindStateOutputAlias(*outputProjection, 0))
					{
						CopyDescriptorToArraySlot(builder, descriptor, inputArray, alias->inputIndex, outputs[0]);
					}
					else
					{
						throw std::runtime_error("Compiled subgraph output projection is missing a returned output");
					}
				}
				else
				{
					CopyDescriptorToOutput(builder, descriptor, outputArray, 0, outputs[0]);
				}
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
				if (outputProjection)
				{
					if (const auto publicSlot = FindPublicOutputSlot(*outputProjection, i))
					{
						CopyDescriptorToOutput(builder, descriptor, outputArray, *publicSlot, outputs[i]);
					}
					else if (const auto* alias = FindStateOutputAlias(*outputProjection, i))
					{
						CopyDescriptorToArraySlot(builder, descriptor, inputArray, alias->inputIndex, outputs[i]);
					}
					else
					{
						throw std::runtime_error("Compiled subgraph output projection is missing a returned output");
					}
				}
				else
				{
					CopyDescriptorToOutput(builder, descriptor, outputArray, i, outputs[i]);
				}
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
				if (outputProjection)
				{
					if (const auto publicSlot = FindPublicOutputSlot(*outputProjection, 0))
					{
						CopyDescriptorToOutput(builder, call, outputArray, *publicSlot, outputs[0]);
					}
					else if (const auto* alias = FindStateOutputAlias(*outputProjection, 0))
					{
						CopyDescriptorToArraySlot(builder, call, inputArray, alias->inputIndex, outputs[0]);
					}
					else
					{
						throw std::runtime_error("Compiled subgraph output projection is missing a returned output");
					}
				}
				else
				{
					CopyDescriptorToOutput(builder, call, outputArray, 0, outputs[0]);
				}
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
			if (outputProjection)
			{
				if (const auto publicSlot = FindPublicOutputSlot(*outputProjection, i))
				{
					CopyDescriptorToOutput(builder, descriptor, outputArray, *publicSlot, outputs[i]);
				}
				else if (const auto* alias = FindStateOutputAlias(*outputProjection, i))
				{
					CopyDescriptorToArraySlot(builder, descriptor, inputArray, alias->inputIndex, outputs[i]);
				}
				else
				{
					throw std::runtime_error("Compiled subgraph output projection is missing a returned output");
				}
			}
			else
			{
				CopyDescriptorToOutput(builder, descriptor, outputArray, i, outputs[i]);
			}
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
		RegisterJITRuntimeSymbol("litenn_cpu_profile_node_begin",
		                         reinterpret_cast<void*>(&litenn_cpu_profile_node_begin));
		RegisterJITRuntimeSymbol("litenn_cpu_profile_node_end", reinterpret_cast<void*>(&litenn_cpu_profile_node_end));
		RegisterJITRuntimeSymbol("litenn_cpu_matmul_bias_relu_parallel_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_matmul_bias_relu_parallel_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_swiglu_f32", reinterpret_cast<void*>(&litenn_cpu_swiglu_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_rope_at_positions_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_rope_at_positions_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_rms_norm_f32", reinterpret_cast<void*>(&litenn_cpu_rms_norm_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_active_prefix_attention_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_active_prefix_attention_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_active_prefix_attention_f32_rank3",
		                         reinterpret_cast<void*>(&litenn_cpu_active_prefix_attention_f32_rank3));
		RegisterJITRuntimeSymbol("litenn_cpu_active_prefix_attention_f32_rank3_grouped",
		                         reinterpret_cast<void*>(&litenn_cpu_active_prefix_attention_f32_rank3_grouped));
		RegisterJITRuntimeSymbol("litenn_cpu_grouped_paged_attention_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_grouped_paged_attention_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_scatter_update_axis0_f32_rank3",
		                         reinterpret_cast<void*>(&litenn_cpu_scatter_update_axis0_f32_rank3));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_matmul_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_matmul_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_matmul_q8k_staged_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_matmul_q8k_staged_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_compact_interleaved_bytes",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_compact_interleaved_bytes));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_prepack_compact_interleaved",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_prepack_compact_interleaved));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_matmul_compact_q8k_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_matmul_compact_q8k_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_field_interleaved_v4_bytes",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_field_interleaved_v4_bytes));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_prepack_field_interleaved_v4",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_prepack_field_interleaved_v4));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_matmul_field_interleaved_v4_q8k_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_matmul_field_interleaved_v4_q8k_f32));
		RegisterJITRuntimeSymbol(
		    "litenn_cpu_swiglu_ggml_block_matmul_field_interleaved_v4_q8k_f32",
		    reinterpret_cast<void*>(&litenn_cpu_swiglu_ggml_block_matmul_field_interleaved_v4_q8k_f32));
		RegisterJITRuntimeSymbol(
		    "litenn_cpu_ggml_block_grouped_matmul2_field_interleaved_v4_q8k_f32",
		    reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul2_field_interleaved_v4_q8k_f32));
		RegisterJITRuntimeSymbol(
		    "litenn_cpu_ggml_block_grouped_matmul3_field_interleaved_v4_q8k_f32",
		    reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul3_field_interleaved_v4_q8k_f32));
		RegisterJITRuntimeSymbol(
		    "litenn_cpu_ggml_block_grouped_matmul2_mixed_field_interleaved_v4_q8k_f32",
		    reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul2_mixed_field_interleaved_v4_q8k_f32));
		RegisterJITRuntimeSymbol(
		    "litenn_cpu_ggml_block_grouped_matmul3_mixed_field_interleaved_v4_q8k_f32",
		    reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul3_mixed_field_interleaved_v4_q8k_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_grouped_matmul2_compact_q8k_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul2_compact_q8k_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_grouped_matmul3_compact_q8k_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul3_compact_q8k_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_q8k_activation_block_bytes",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_q8k_activation_block_bytes));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_prepare_q8k_activation_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_prepare_q8k_activation_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_matmul_q8k_prepared_activation_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_matmul_q8k_prepared_activation_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_q4k_prepacked_block_bytes",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_q4k_prepacked_block_bytes));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_prepack_q4k_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_prepack_q4k_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_matmul_q4k_prepacked_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_matmul_q4k_prepacked_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_q6k_prepacked_block_bytes",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_q6k_prepacked_block_bytes));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_prepack_q6k_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_prepack_q6k_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_matmul_q6k_prepacked_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_matmul_q6k_prepacked_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_grouped_matmul2_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul2_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_grouped_matmul2_mixed_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul2_mixed_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_grouped_matmul2_q8k_staged_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul2_q8k_staged_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_grouped_matmul2_mixed_q8k_staged_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul2_mixed_q8k_staged_f32));
		RegisterJITRuntimeSymbol(
		    "litenn_cpu_ggml_block_grouped_matmul2_q8k_prepared_activation_f32",
		    reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul2_q8k_prepared_activation_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_grouped_matmul2_q4k_prepacked_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul2_q4k_prepacked_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_grouped_matmul2_q6k_prepacked_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul2_q6k_prepacked_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_grouped_matmul3_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul3_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_grouped_matmul3_mixed_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul3_mixed_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_grouped_matmul3_q8k_staged_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul3_q8k_staged_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_grouped_matmul3_mixed_q8k_staged_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul3_mixed_q8k_staged_f32));
		RegisterJITRuntimeSymbol(
		    "litenn_cpu_ggml_block_grouped_matmul3_q8k_prepared_activation_f32",
		    reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul3_q8k_prepared_activation_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_grouped_matmul3_q4k_prepacked_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul3_q4k_prepacked_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_grouped_matmul3_q6k_prepacked_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_grouped_matmul3_q6k_prepacked_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_get_rows_i32_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_get_rows_i32_f32));
		RegisterJITRuntimeSymbol("litenn_cpu_ggml_block_get_rows_i64_f32",
		                         reinterpret_cast<void*>(&litenn_cpu_ggml_block_get_rows_i64_f32));
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
			return litenn::translateExecutablePlanToMLIR(Detail::BuildExecutablePlanFromGraph(graph), ctx,
			                                             { .enableNodeProfiling = options.enableCPUAOTNodeProfiling });
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
		LogMLIRModuleStats(options, "cpu-mlir after lower LiteNN dialect", *module);
		TimedCompileDiagnostic(options, "cpu-mlir bufferize", [&] {
			mlir::PassManager pm(&ctx);
			litenn::addBufferizationPipeline(pm);
			if (mlir::failed(pm.run(*module)))
			{
				throw std::runtime_error("LiteNN bufferization pipeline failed");
			}
		});
		LogMLIRModuleStats(options, "cpu-mlir after bufferize", *module);
		TimedCompileDiagnostic(options, "cpu-mlir lower LLVM dialect", [&] {
			mlir::PassManager pm(&ctx);
			litenn::addLLVMCodegenPipeline(
			    pm, litenn::LLVMCodegenOptions{
			            .cpuAOTThreadCount = static_cast<std::uint64_t>(options.cpuAOTThreadCount),
			            .cpuAOTSchedulingPolicy =
			                EncodeCPUAOTSchedulingPolicy(options.cpuAOTAffinityPolicy, options.cpuAOTWorkerWaitPolicy),
			            .enableGGMLQ8KStagedMatMul = options.enableCPUAOTGGMLQ8KStagedMatMul,
			            .enableGGMLPrepackedWeights =
			                options.enableCPUAOTGGMLPrepackedWeights ||
			                options.cpuAOTGGMLPrepackedWeightPolicy != CPUAOTGGMLPrepackedWeightPolicy::Disabled,
			            .enableBoundedActivationMath =
			                options.cpuAOTActivationMathPolicy == CPUAOTActivationMathPolicy::Bounded,
			        });
			if (mlir::failed(pm.run(*module)))
			{
				throw std::runtime_error("LiteNN LLVM codegen pipeline failed");
			}
		});
		LogMLIRModuleStats(options, "cpu-mlir after lower LLVM dialect", *module);
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

	std::optional<CompiledArtifactParts>
	TryCompileCPUMLIRExternalRegions(const Graph& graph, const CompilerOptions& options,
	                                 const Runtime::RuntimeScheduleOutputProjection* outputProjection = nullptr)
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
		LogLLVMModuleStats(options, "cpu-llvm after translate", *llvmModule);

		auto config = TimedCompileDiagnostic(options, "cpu-aot create target machine",
		                                     [&] { return CreateNativeTargetMachine(); });
		TimedCompileDiagnostic(options, "cpu-aot configure module",
		                       [&] { ConfigureForNativeObject(*llvmModule, config); });

		auto inputSpecs =
		    TimedCompileDiagnostic(options, "cpu-aot build input specs", [&] { return BuildInputSpecs(graph); });
		auto functionalOutputSpecs =
		    TimedCompileDiagnostic(options, "cpu-aot build output specs", [&] { return BuildOutputSpecs(graph); });
		auto entryOutputSpecs = TimedCompileDiagnostic(options, "cpu-aot build entry output specs", [&] {
			return BuildEntryOutputSpecs(functionalOutputSpecs, outputProjection);
		});
		TimedCompileDiagnostic(options, "cpu-aot add uniform entry wrapper", [&] {
			AddUniformEntryWrapper(*llvmModule, "subgraph_" + std::to_string(graph.Forward()), inputSpecs,
			                       functionalOutputSpecs, externalized->entryExternalTensorInfos, outputProjection);
		});
		if (outputProjection && !outputProjection->stateAliases.empty())
		{
			TimedCompileDiagnostic(options, "cpu-aot strip state-alias unsafe attributes",
			                       [&] { StripStateAliasUnsafeAttributes(*llvmModule); });
		}
		LogLLVMModuleStats(options, "cpu-llvm after entry wrapper", *llvmModule);
		const auto effectiveOptLevel = EffectiveCPUAOTLLVMOptLevel(options, outputProjection);
		TimedCompileDiagnostic(options, std::format("cpu-aot optimize LLVM module O{}", effectiveOptLevel),
		                       [&] { OptimizeLLVMModule(*llvmModule, *config.targetMachine, effectiveOptLevel); });
		LogLLVMModuleStats(options, "cpu-llvm after optimize", *llvmModule);

		auto rodata = TimedCompileDiagnostic(options, "cpu-aot serialize rodata", [&] {
			return SerializeRodata(inputSpecs, entryOutputSpecs, config.triple, CompiledModuleBackend::CPUNative,
			                       CPUAOTRequiredRuntimeFeatures(options));
		});
		auto instructions =
		    TimedCompileDiagnostic(options, "cpu-aot emit object file", [&] { return EmitObjectFile(*llvmModule); });
		LogCompileDiagnostic(options, std::format("cpu-aot object file bytes={}", instructions.size()));
		return CompiledArtifactParts{ std::move(rodata),
			                          std::move(instructions),
			                          std::move(externalized->constants),
			                          std::move(externalized->weights),
			                          std::move(externalized->externalTensorInfos),
			                          std::move(inputSpecs),
			                          std::move(entryOutputSpecs) };
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
	std::vector<std::shared_ptr<const void>> borrowedExternalOwners;
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

CompiledModuleCPUHelperProfiler::CompiledModuleCPUHelperProfiler() : impl_(std::make_unique<Impl>())
{
	impl_->previous = CompiledModuleCPUHelperProfilerAccess::current;
	CompiledModuleCPUHelperProfilerAccess::current = impl_.get();
}

CompiledModuleCPUHelperProfiler::~CompiledModuleCPUHelperProfiler()
{
	if (CompiledModuleCPUHelperProfilerAccess::current == impl_.get())
	{
		CompiledModuleCPUHelperProfilerAccess::current = impl_->previous;
	}
}

std::vector<CompiledModuleCPUHelperProfileEvent> CompiledModuleCPUHelperProfiler::Snapshot() const
{
	std::vector<CompiledModuleCPUHelperProfileEvent> events;
	events.reserve(impl_->events.size());
	for (const auto& [_, event] : impl_->events)
	{
		events.push_back(event);
	}
	std::ranges::sort(events, [](const auto& lhs, const auto& rhs) {
		if (lhs.totalMilliseconds != rhs.totalMilliseconds)
		{
			return lhs.totalMilliseconds > rhs.totalMilliseconds;
		}
		return lhs.helper < rhs.helper;
	});
	return events;
}

std::vector<CompiledModuleCPUNodeProfileEvent> CompiledModuleCPUHelperProfiler::SnapshotNodes() const
{
	std::vector<CompiledModuleCPUNodeProfileEvent> events;
	events.reserve(impl_->nodeEvents.size());
	const auto schemas = DefaultOpSchemaRegistry().Schemas();
	for (const auto& [_, event] : impl_->nodeEvents)
	{
		auto snapshot = event;
		snapshot.opKind = snapshot.schemaId < schemas.size() ? schemas[snapshot.schemaId].kind
		                                                     : std::format("schema_{}", snapshot.schemaId);
		events.push_back(std::move(snapshot));
	}
	std::ranges::sort(events, [](const auto& lhs, const auto& rhs) {
		if (lhs.selfMilliseconds != rhs.selfMilliseconds)
		{
			return lhs.selfMilliseconds > rhs.selfMilliseconds;
		}
		if (lhs.subgraphId != rhs.subgraphId)
		{
			return lhs.subgraphId < rhs.subgraphId;
		}
		return lhs.nodeId < rhs.nodeId;
	});
	return events;
}

std::vector<CompiledModuleCPUParallelProfileEvent> CompiledModuleCPUHelperProfiler::SnapshotParallel() const
{
	return impl_->parallelEvents;
}

double CompiledModuleCPUHelperProfiler::NodeInstrumentationMilliseconds() const
{
	return impl_->nodeInstrumentationMilliseconds;
}

CompiledModule<CPU>::CompiledModule() = default;

CPUAOTActivationMathCapabilities LiteNN::QueryCPUAOTActivationMathCapabilities() noexcept
{
	return {};
}

bool LiteNN::IsCPUAOTActivationMathPolicySupported(CPUAOTActivationMathPolicy policy) noexcept
{
	const auto capabilities = QueryCPUAOTActivationMathCapabilities();
	switch (policy)
	{
	case CPUAOTActivationMathPolicy::Strict:
		return capabilities.strictSupported;
	case CPUAOTActivationMathPolicy::Bounded:
		return capabilities.boundedSupported;
	default:
		return false;
	}
}

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

CompiledModuleSeparatedArtifact::CompiledModuleSeparatedArtifact(
    std::vector<std::byte> metadata, std::vector<std::byte> constants, CompiledModuleRegion borrowedWeights,
    std::shared_ptr<const void> borrowedWeightsOwner, std::vector<std::byte> instructions,
    std::vector<CompiledTensorSpec> inputSpecs, std::vector<CompiledTensorSpec> outputSpecs,
    CompiledModuleBackend backend)
    : metadata_(std::move(metadata)), constants_(std::move(constants)), borrowedWeights_(borrowedWeights),
      borrowedWeightsOwner_(std::move(borrowedWeightsOwner)), instructions_(std::move(instructions)),
      inputSpecs_(std::move(inputSpecs)), outputSpecs_(std::move(outputSpecs)), backend_(backend)
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

CompiledModuleSeparatedArtifact CompiledModuleSeparatedArtifact::FromOwnedRegionsWithBorrowedWeights(
    std::vector<std::byte> metadata, std::vector<std::byte> constants, CompiledModuleRegion weights,
    std::shared_ptr<const void> weightsOwner, std::vector<std::byte> instructions)
{
	if (weights.size != 0 && !weightsOwner)
	{
		throw std::runtime_error("Borrowed separated artifact weights require an owner");
	}
	auto separatedMetadata = ValidateSeparatedImage({
	    .metadata = { .data = metadata.data(), .size = metadata.size() },
	    .constants = { .data = constants.data(), .size = constants.size() },
	    .weights = weights,
	    .instructions = { .data = instructions.data(), .size = instructions.size() },
	});
	return CompiledModuleSeparatedArtifact(
	    std::move(metadata), std::move(constants), weights, std::move(weightsOwner), std::move(instructions),
	    std::move(separatedMetadata.legacyMetadata.inputSpecs), std::move(separatedMetadata.legacyMetadata.outputSpecs),
	    separatedMetadata.legacyMetadata.backend);
}

CompiledModuleSeparatedArtifact CompiledModuleSeparatedArtifact::FromOwnedRegionsWithTrustedBorrowedWeights(
    std::vector<std::byte> metadata, std::vector<std::byte> constants, CompiledModuleRegion weights,
    std::shared_ptr<const void> weightsOwner, std::vector<std::byte> instructions)
{
	if (weights.size != 0 && !weightsOwner)
	{
		throw std::runtime_error("Trusted borrowed separated artifact weights require an owner");
	}
	auto separatedMetadata = ValidateSeparatedImage(
	    {
	        .metadata = { .data = metadata.data(), .size = metadata.size() },
	        .constants = { .data = constants.data(), .size = constants.size() },
	        .weights = weights,
	        .instructions = { .data = instructions.data(), .size = instructions.size() },
	    },
	    false);
	return CompiledModuleSeparatedArtifact(
	    std::move(metadata), std::move(constants), weights, std::move(weightsOwner), std::move(instructions),
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

CompiledModule<CPU> CompiledModuleSeparatedArtifact::Load() &&
{
	auto metadata = ValidateSeparatedImage(Image());
	auto constants = RegionBytes({ .data = constants_.data(), .size = constants_.size() }, kConstantsRegionName);
	auto weights = RegionBytes(WeightsRegion(), kWeightsRegionName);
	auto instructions = RestoreLegacyInstructionsFromSeparated(
	    metadata.legacyMetadata.backend,
	    RegionBytes({ .data = instructions_.data(), .size = instructions_.size() }, kInstructionsRegionName),
	    constants);
	auto module = CompiledModule<CPU>::Load({
	    .rodata = metadata.legacyRodata.data(),
	    .rodataSize = metadata.legacyRodata.size(),
	    .instructions = instructions.data(),
	    .instructionSize = instructions.size(),
	});
	module.impl_->externalConstants = std::move(constants_);
	module.impl_->externalWeights.assign(weights.begin(), weights.end());
	return module;
}

CompiledModule<CPU> CompiledModuleSeparatedArtifact::LoadBorrowedExternalRegions() const
{
	return CompiledModule<CPU>::LoadBorrowedExternalRegions(Image());
}

CompiledModule<CPU> CompiledModuleSeparatedArtifact::LoadBorrowedExternalRegions() &&
{
	auto owner = std::make_shared<CompiledModuleSeparatedArtifact>(std::move(*this));
	auto metadata = DeserializeSeparatedMetadata(owner->metadata_);
	auto constants =
	    RegionBytes({ .data = owner->constants_.data(), .size = owner->constants_.size() }, kConstantsRegionName);
	auto weights = RegionBytes(owner->WeightsRegion(), kWeightsRegionName);
	auto instructions = RestoreLegacyInstructionsFromSeparated(
	    metadata.legacyMetadata.backend,
	    RegionBytes({ .data = owner->instructions_.data(), .size = owner->instructions_.size() },
	                kInstructionsRegionName),
	    constants);
	auto module = CompiledModule<CPU>::Load({
	    .rodata = metadata.legacyRodata.data(),
	    .rodataSize = metadata.legacyRodata.size(),
	    .instructions = instructions.data(),
	    .instructionSize = instructions.size(),
	});
	module.impl_->borrowedExternalConstants = constants;
	module.impl_->borrowedExternalWeights = weights;
	module.impl_->borrowedExternalOwners.push_back(std::move(owner));
	return module;
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
	if (borrowedWeightsOwner_)
	{
		return CompiledModuleSeparatedArtifact(metadata_, ToByteVector(constants, kConstantsRegionName),
		                                       borrowedWeights_, borrowedWeightsOwner_, instructions_, inputSpecs_,
		                                       outputSpecs_, backend_);
	}
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
	const auto weights = WeightsRegion();
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
		    .data = weights.data,
		    .size = weights.size,
		},
		.instructions = {
		    .data = instructions_.data(),
		    .size = instructions_.size(),
		},
	};
}

CompiledModuleRegion CompiledModuleSeparatedArtifact::WeightsRegion() const
{
	if (borrowedWeightsOwner_)
	{
		return borrowedWeights_;
	}
	return { .data = weights_.data(), .size = weights_.size() };
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
	return RegionBytes(WeightsRegion(), kWeightsRegionName);
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
	const auto objectBytes = EmitSeparatedCarrierObject(metadata_, constants_, Weights(), instructions_, symbolPrefix);
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
	              EmitSingleRegionCarrierObject(Weights(), symbolPrefix, kWeightsRegionName, ".litenn_weights"));
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
	WriteAllBytes(directory / (prefix + ".weights.bin"), Weights());
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

	CompiledArtifactParts
	CompileCPUArtifactPartsFromGraph(const Graph& graph, const CompilerOptions& options,
	                                 const Runtime::RuntimeScheduleOutputProjection* outputProjection = nullptr)
	{
		ValidateCPUAOTCompilerOptions(options);
		Validation::ValidateGraph(graph);
		if (!outputProjection)
		{
			if (auto parallelParts = TryCompileCPUParallelLinearChainF32WithExternalRegionFusion(graph, options))
			{
				return MakeCompiledArtifactParts(
				    std::move(parallelParts->rodata), std::move(parallelParts->instructions),
				    std::move(parallelParts->inputSpecs), std::move(parallelParts->outputSpecs),
				    CompiledModuleBackend::CPUNative, std::move(parallelParts->constants),
				    std::move(parallelParts->weights), std::move(parallelParts->externalTensorInfos));
			}
		}
		if (auto externalParts = TryCompileCPUMLIRExternalRegions(graph, options, outputProjection))
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
		const auto functionalOutputSpecs = BuildOutputSpecs(graph);
		const auto entryOutputSpecs = BuildEntryOutputSpecs(functionalOutputSpecs, outputProjection);
		AddUniformEntryWrapper(*llvmModule, "subgraph_" + std::to_string(graph.Forward()), inputSpecs,
		                       functionalOutputSpecs, {}, outputProjection);
		if (outputProjection && !outputProjection->stateAliases.empty())
		{
			StripStateAliasUnsafeAttributes(*llvmModule);
		}
		OptimizeLLVMModule(*llvmModule, *config.targetMachine, EffectiveCPUAOTLLVMOptLevel(options, outputProjection));

		auto rodata = SerializeRodata(inputSpecs, entryOutputSpecs, config.triple, CompiledModuleBackend::CPUNative,
		                              CPUAOTRequiredRuntimeFeatures(options));
		auto instructions = EmitObjectFile(*llvmModule);
		return MakeCompiledArtifactParts(std::move(rodata), std::move(instructions), inputSpecs, entryOutputSpecs,
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

CompiledModuleArtifact Compiler<CPU>::CompileArtifact(const Runtime::RuntimeSchedule& schedule)
{
	return CompileArtifact(schedule, CompilerOptions::Defaults());
}

CompiledModuleArtifact Compiler<CPU>::CompileArtifact(const Runtime::RuntimeSchedule& schedule,
                                                      const CompilerOptions& options)
{
	Runtime::ValidateRuntimeSchedule(schedule);
	auto scheduleOptions = options;
	if (!schedule.module.plan.variables.empty())
	{
		scheduleOptions.enableCPUAOTExternalRegions = true;
	}
	const auto projection = Runtime::RuntimeScheduleOutputProjectionForFunction(schedule, schedule.module.plan.forward);
	auto parts = CompileCPUArtifactPartsFromGraph(BuildCompilerGraphFromPlan(schedule.module.plan), scheduleOptions,
	                                              &projection);
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

CompiledModule<CPU> Compiler<CPU>::Compile(const Runtime::RuntimeSchedule& schedule)
{
	return Compile(schedule, CompilerOptions::Defaults());
}

CompiledModule<CPU> Compiler<CPU>::Compile(const Runtime::RuntimeSchedule& schedule, const CompilerOptions& options)
{
	auto artifact = CompileArtifact(schedule, options);
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
