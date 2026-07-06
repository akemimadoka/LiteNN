#include <LiteNN/Device.h>
#ifdef LITENN_ENABLE_CUDA
#include <LiteNN/Device/CUDA.h>
#endif
#ifdef LITENN_ENABLE_VULKAN
#include <LiteNN/Device/Vulkan.h>
#endif
#include <LiteNN/ExecutablePlan.h>
#include <LiteNN/Graph.h>
#include <LiteNN/Runtime/Scheduler.h>
#include <LiteNN/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#ifndef LITENN_COMPILER_COMPILEDMODULE_H
#define LITENN_COMPILER_COMPILEDMODULE_H

namespace LiteNN
{
	enum class CompiledModuleBackend : std::uint32_t
	{
		CPUNative = 1,
		CUDANative = 2,
		VulkanNative = 3,
	};

	struct CompiledModuleCPUHelperProfileEvent
	{
		std::string helper;
		std::string detail;
		std::uint64_t calls{};
		double totalMilliseconds{};
	};

	class CompiledModuleCPUHelperProfiler
	{
	public:
		CompiledModuleCPUHelperProfiler();
		CompiledModuleCPUHelperProfiler(const CompiledModuleCPUHelperProfiler&) = delete;
		CompiledModuleCPUHelperProfiler& operator=(const CompiledModuleCPUHelperProfiler&) = delete;
		~CompiledModuleCPUHelperProfiler();

		std::vector<CompiledModuleCPUHelperProfileEvent> Snapshot() const;

	private:
		friend struct CompiledModuleCPUHelperProfilerAccess;

		struct Impl;
		std::unique_ptr<Impl> impl_;
	};

	struct CompiledTensorSpec
	{
		TensorType type;
		std::string name;
		std::optional<QuantizationParams> quantization;

		TensorType Type() const
		{
			return type;
		}

		static CompiledTensorSpec FromType(std::string name, const TensorType& type,
		                                   std::optional<QuantizationParams> quantization = std::nullopt)
		{
			return { .type = type, .name = std::move(name), .quantization = std::move(quantization) };
		}
	};

	struct CompiledModuleImage
	{
		const void* rodata{};
		std::size_t rodataSize{};
		const void* instructions{};
		std::size_t instructionSize{};
	};

	struct CompiledModuleRegion
	{
		const void* data{};
		std::size_t size{};
	};

	struct CompiledModuleSeparatedImage
	{
		CompiledModuleRegion metadata;
		CompiledModuleRegion constants;
		CompiledModuleRegion weights;
		CompiledModuleRegion instructions;
	};

	struct CompiledModuleExportedSymbols
	{
		const void* rodata{};
		const void* rodataSize{};
		const void* instructions{};
		const void* instructionSize{};
	};

	struct CompiledModuleSeparatedExportedSymbols
	{
		const void* metadata{};
		const void* metadataSize{};
		const void* constants{};
		const void* constantsSize{};
		const void* weights{};
		const void* weightsSize{};
		const void* instructions{};
		const void* instructionsSize{};
	};

	struct CompiledModuleRegionInfo
	{
		std::string name;
		std::uint64_t size{};
		std::uint64_t alignment{ 1 };
		std::uint64_t checksum{};
	};

	enum class CompiledModuleExternalTensorRebindPolicy : std::uint32_t
	{
		ExactChecksum = 1,
	};

	enum class CPUAOTAffinityPolicy : std::uint32_t
	{
		None = 0,
		Compact = 1,
	};

	struct CompiledModuleExternalTensorInfo
	{
		std::string name;
		std::string region;
		TensorType type;
		std::uint64_t byteOffset{};
		std::uint64_t byteSize{};
		std::uint64_t alignment{ 1 };
		std::uint64_t checksum{};
		CompiledModuleExternalTensorRebindPolicy rebindPolicy{
			CompiledModuleExternalTensorRebindPolicy::ExactChecksum
		};
	};

	struct CompilerOptions
	{
		/// CPU AOT worker count. Zero means use hardware_concurrency().
		std::size_t cpuAOTThreadCount{};
		/// Optional CPU AOT worker affinity policy. Defaults to no pinning.
		CPUAOTAffinityPolicy cpuAOTAffinityPolicy{ CPUAOTAffinityPolicy::None };
		/// Minimum f32 linear-chain FLOPs before the CPU parallel AOT path is used.
		std::uint64_t cpuAOTParallelMinFlops{ 1ull << 28 };
		/// Store CPU AOT constants/variable weights in separated artifact regions when supported.
		bool enableCPUAOTExternalRegions{};
		/// Minimum generic CPU MLIR ConstantNode byte size before externalizing it.
		/// VariableRefNode weights are externalized independently when external regions are enabled.
		std::uint64_t cpuAOTExternalConstantMinBytes{ 64 };
		/// Retry the CPU f32 linear-chain external-region path after an internal FusionPass.
		bool enableCPUAOTExternalRegionFusion{ true };
		/// Opt in to Q8_K activation-staged GGML block MatMul helpers for formats where small numeric deltas are
		/// allowed.
		bool enableCPUAOTGGMLQ8KStagedMatMul{};
		/// Opt in to compile-time prepared GGML_Q4_K/GGML_Q6_K weight payloads in separated CPU AOT artifacts.
		bool enableCPUAOTGGMLPrepackedWeights{};
		/// Prefer CUDA native AOT kernels before falling back to CPU AOT bridge.
		bool enableCUDANativeAOT{ true };
		/// Prefer Vulkan native AOT kernels before falling back to CPU AOT bridge.
		bool enableVulkanNativeAOT{ true };
		/// LLVM optimization level for CPU AOT codegen. Values above 3 are clamped to 3.
		std::uint8_t cpuAOTLLVMOptLevel{ 3 };
		/// Print coarse compiler phase timing diagnostics to stderr.
		bool enableCompileDiagnostics{};

		static CompilerOptions Defaults();
	};

	struct CompileBudgetEstimate
	{
		std::size_t subgraphCount{};
		std::size_t nodeCount{};
		std::size_t variableCount{};
		std::size_t variableRefNodeCount{};
		std::size_t constantNodeCount{};
		std::size_t quantizedConstantNodeCount{};
		std::uint64_t variablePayloadBytes{};
		std::uint64_t constantPayloadBytes{};
		std::uint64_t quantizedConstantPayloadBytes{};
		std::uint64_t projectedInlineMLIRPayloadBytes{};
		std::uint64_t projectedExternalConstantBytes{};
		std::uint64_t projectedExternalWeightBytes{};
		bool cpuAOTExternalRegionsEnabled{};
	};

	/// Estimate tensor payload pressure before invoking MLIR/LLVM code generation.
	CompileBudgetEstimate EstimateCompileBudget(const ExecutablePlan& plan, const CompilerOptions& options);
	CompileBudgetEstimate EstimateCompileBudget(const Graph& graph, const CompilerOptions& options);

	struct CompiledModuleTensorInvocation
	{
		std::span<const Tensor<CPU>> inputs;
		std::span<Tensor<CPU>> outputs;
	};

	struct CompiledTensorBinding
	{
		void* data{};
		TensorType type;
		std::string name;
		std::optional<QuantizationParams> quantization;
	};

	struct CompiledModuleBindingInvocation
	{
		std::span<const CompiledTensorBinding> inputs;
		std::span<const CompiledTensorBinding> outputs;
	};

	template <Device D>
	class CompiledModule;

	template <Device D>
	class Compiler;

	class CompiledModuleSeparatedArtifact
	{
	public:
		CompiledModuleSeparatedArtifact() = default;
		CompiledModuleSeparatedArtifact(const CompiledModuleSeparatedArtifact&) = default;
		CompiledModuleSeparatedArtifact(CompiledModuleSeparatedArtifact&&) noexcept = default;
		CompiledModuleSeparatedArtifact& operator=(const CompiledModuleSeparatedArtifact&) = default;
		CompiledModuleSeparatedArtifact& operator=(CompiledModuleSeparatedArtifact&&) noexcept = default;
		~CompiledModuleSeparatedArtifact() = default;

		static CompiledModuleSeparatedArtifact CopyFromImage(CompiledModuleSeparatedImage image);
		static CompiledModuleSeparatedArtifact FromOwnedRegions(std::vector<std::byte> metadata,
		                                                        std::vector<std::byte> constants,
		                                                        std::vector<std::byte> weights,
		                                                        std::vector<std::byte> instructions);
		static CompiledModuleSeparatedArtifact
		FromOwnedRegionsWithBorrowedWeights(std::vector<std::byte> metadata, std::vector<std::byte> constants,
		                                    CompiledModuleRegion weights, std::shared_ptr<const void> weightsOwner,
		                                    std::vector<std::byte> instructions);
		static CompiledModuleSeparatedArtifact FromExportedSymbols(CompiledModuleSeparatedExportedSymbols symbols);

		CompiledModule<CPU> Load() const;
		CompiledModule<CPU> Load() &&;
		/// Loads instructions into a CPU module while borrowing constants/weights from this artifact.
		/// The artifact must outlive every run of the returned module.
		CompiledModule<CPU> LoadBorrowedExternalRegions() const;
		/// Moves this artifact into the returned CPU module so external regions can be borrowed without an extra copy.
		CompiledModule<CPU> LoadBorrowedExternalRegions() &&;
#ifdef LITENN_ENABLE_CUDA
		CompiledModule<CUDA> Load(CUDA device) const;
		/// Loads a CUDA module from separated regions. CUDA-native constants are copied
		/// to device memory during load; CPU-bridge artifacts borrow constants/weights
		/// through the embedded CPU module and require this artifact to outlive runs.
		CompiledModule<CUDA> LoadBorrowedExternalRegions(CUDA device) const;
#endif
#ifdef LITENN_ENABLE_VULKAN
		CompiledModule<Vulkan> Load(Vulkan device) const;
		CompiledModule<Vulkan> LoadBorrowedExternalRegions(Vulkan device) const;
#endif

		CompiledModuleSeparatedArtifact WithReboundConstants(CompiledModuleRegion constants) const;
		CompiledModuleSeparatedArtifact WithReboundWeights(CompiledModuleRegion weights) const;

		CompiledModuleSeparatedImage Image() const;
		std::span<const std::byte> Metadata() const;
		std::span<const std::byte> Constants() const;
		std::span<const std::byte> Weights() const;
		std::span<const std::byte> Instructions() const;
		std::vector<CompiledModuleRegionInfo> RegionInfos() const;
		std::vector<CompiledModuleExternalTensorInfo> ExternalTensorInfos() const;
		std::span<const CompiledTensorSpec> InputSpecs() const;
		std::span<const CompiledTensorSpec> OutputSpecs() const;
		CompiledModuleBackend Backend() const;
		std::optional<std::size_t> FindInput(std::string_view name) const;
		std::optional<std::size_t> FindOutput(std::string_view name) const;

		void WriteObjectFile(const std::filesystem::path& path, std::string_view symbolPrefix = "litenn_module") const;
		void WriteObjectFiles(const std::filesystem::path& directory,
		                      std::string_view symbolPrefix = "litenn_module") const;
		void WriteRegionFiles(const std::filesystem::path& directory,
		                      std::string_view filePrefix = "litenn_module") const;

	private:
		friend class CompiledModuleArtifact;

		CompiledModuleSeparatedArtifact(std::vector<std::byte> metadata, std::vector<std::byte> constants,
		                                std::vector<std::byte> weights, std::vector<std::byte> instructions,
		                                std::vector<CompiledTensorSpec> inputSpecs,
		                                std::vector<CompiledTensorSpec> outputSpecs, CompiledModuleBackend backend);
		CompiledModuleSeparatedArtifact(std::vector<std::byte> metadata, std::vector<std::byte> constants,
		                                CompiledModuleRegion borrowedWeights,
		                                std::shared_ptr<const void> borrowedWeightsOwner,
		                                std::vector<std::byte> instructions, std::vector<CompiledTensorSpec> inputSpecs,
		                                std::vector<CompiledTensorSpec> outputSpecs, CompiledModuleBackend backend);

		CompiledModuleRegion WeightsRegion() const;

		std::vector<std::byte> metadata_;
		std::vector<std::byte> constants_;
		std::vector<std::byte> weights_;
		CompiledModuleRegion borrowedWeights_;
		std::shared_ptr<const void> borrowedWeightsOwner_;
		std::vector<std::byte> instructions_;
		std::vector<CompiledTensorSpec> inputSpecs_;
		std::vector<CompiledTensorSpec> outputSpecs_;
		CompiledModuleBackend backend_{ CompiledModuleBackend::CPUNative };
	};

	class CompiledModuleArtifact
	{
	public:
		CompiledModuleArtifact() = default;
		CompiledModuleArtifact(const CompiledModuleArtifact&) = default;
		CompiledModuleArtifact(CompiledModuleArtifact&&) noexcept = default;
		CompiledModuleArtifact& operator=(const CompiledModuleArtifact&) = default;
		CompiledModuleArtifact& operator=(CompiledModuleArtifact&&) noexcept = default;
		~CompiledModuleArtifact() = default;

		static CompiledModuleArtifact CopyFromImage(CompiledModuleImage image);
		static CompiledModuleArtifact FromExportedSymbols(CompiledModuleExportedSymbols symbols);

		/// Loads the artifact into a runnable module. The artifact remains valid after loading.
		CompiledModule<CPU> Load() const;
		/// Loads the artifact and transfers owned external regions into the runnable module.
		CompiledModule<CPU> Load() &&;
#ifdef LITENN_ENABLE_CUDA
		/// Loads the artifact into a CUDA module. CPU-native artifacts bridge through CPU AOT;
		/// CUDA-native artifacts load their embedded CUDA instruction payload.
		CompiledModule<CUDA> Load(CUDA device) const;
#endif
#ifdef LITENN_ENABLE_VULKAN
		/// Loads the artifact into a Vulkan module. CPU-native artifacts require explicit host fallback;
		/// Vulkan-native artifacts load their embedded SPIR-V instruction payload.
		CompiledModule<Vulkan> Load(Vulkan device) const;
#endif

		CompiledModuleImage Image() const;
		std::span<const std::byte> Rodata() const;
		std::span<const std::byte> Instructions() const;
		std::span<const std::byte> Constants() const;
		std::span<const std::byte> Weights() const;
		std::span<const CompiledTensorSpec> InputSpecs() const;
		std::span<const CompiledTensorSpec> OutputSpecs() const;
		CompiledModuleBackend Backend() const;
		std::span<const CompiledModuleExternalTensorInfo> ExternalTensorInfos() const;
		std::optional<std::size_t> FindInput(std::string_view name) const;
		std::optional<std::size_t> FindOutput(std::string_view name) const;
		/// Builds separated-image metadata for the artifact's current constant/weight regions.
		std::vector<std::byte> BuildSeparatedMetadata() const;
		CompiledModuleSeparatedArtifact SeparateRodata() const;

		void WriteObjectFile(const std::filesystem::path& path, std::string_view symbolPrefix = "litenn_module") const;

	private:
		friend class Compiler<CPU>;
#ifdef LITENN_ENABLE_CUDA
		friend class Compiler<CUDA>;
#endif
#ifdef LITENN_ENABLE_VULKAN
		friend class Compiler<Vulkan>;
#endif

		CompiledModuleArtifact(std::vector<std::byte> rodata, std::vector<std::byte> instructions,
		                       std::vector<CompiledTensorSpec> inputSpecs, std::vector<CompiledTensorSpec> outputSpecs,
		                       CompiledModuleBackend backend, std::vector<std::byte> constants = {},
		                       std::vector<std::byte> weights = {},
		                       std::vector<CompiledModuleExternalTensorInfo> externalTensorInfos = {});

		std::vector<std::byte> rodata_;
		std::vector<std::byte> instructions_;
		std::vector<std::byte> constants_;
		std::vector<std::byte> weights_;
		std::vector<CompiledModuleExternalTensorInfo> externalTensorInfos_;
		std::vector<CompiledTensorSpec> inputSpecs_;
		std::vector<CompiledTensorSpec> outputSpecs_;
		CompiledModuleBackend backend_{ CompiledModuleBackend::CPUNative };
	};

	template <>
	class CompiledModule<CPU>
	{
	public:
		CompiledModule();
		CompiledModule(const CompiledModule&);
		CompiledModule(CompiledModule&&) noexcept;
		CompiledModule& operator=(const CompiledModule&);
		CompiledModule& operator=(CompiledModule&&) noexcept;
		~CompiledModule();

		/// Loads a borrowed image by copying rodata/instruction bytes into module-owned storage.
		/// The caller may release the original image memory after this returns.
		static CompiledModule Load(CompiledModuleImage image);
		static CompiledModule Load(CompiledModuleSeparatedImage image);

		/// Loads a separated image while borrowing constants/weights from the caller-provided regions.
		/// The caller must keep image.constants and image.weights stable for every run of the returned module.
		static CompiledModule LoadBorrowedExternalRegions(CompiledModuleSeparatedImage image);

		/// Adapter helper: runs with Tensor inputs and returns newly allocated output tensors.
		std::vector<Tensor<CPU>> RunTensors(std::span<const Tensor<CPU>> inputs) const;

		/// Adapter helper: runs with Tensor inputs into caller-provided output tensors.
		void RunTensorsInto(std::span<const Tensor<CPU>> inputs, std::span<Tensor<CPU>> outputs) const;

		/// Runs the compiled entry point with explicit typed input/output buffer bindings.
		void RunIntoBindings(std::span<const CompiledTensorBinding> inputs,
		                     std::span<const CompiledTensorBinding> outputs) const;

		/// Runs independent invocations concurrently when threadCount > 1.
		/// Concurrent binding or Tensor-adapter calls are supported when each call uses
		/// independent input/output buffers.
		void RunManyTensorsInto(std::span<const CompiledModuleTensorInvocation> invocations,
		                        std::size_t threadCount = 0) const;
		void RunManyIntoBindings(std::span<const CompiledModuleBindingInvocation> invocations,
		                         std::size_t threadCount = 0) const;

		CompiledModuleImage Image() const;
		std::span<const std::byte> Rodata() const;
		std::span<const std::byte> Instructions() const;
		std::span<const CompiledTensorSpec> InputSpecs() const;
		std::span<const CompiledTensorSpec> OutputSpecs() const;
		CompiledModuleBackend Backend() const;
		std::optional<std::size_t> FindInput(std::string_view name) const;
		std::optional<std::size_t> FindOutput(std::string_view name) const;

		void WriteObjectFile(const std::filesystem::path& path, std::string_view symbolPrefix = "litenn_module") const;

	private:
		friend class CompiledModuleArtifact;
		friend class CompiledModuleSeparatedArtifact;

		struct Impl;

		explicit CompiledModule(std::shared_ptr<Impl> impl);

		std::shared_ptr<Impl> impl_;
	};

	template <>
	class Compiler<CPU>
	{
	public:
		static CompiledModuleArtifact CompileArtifact(const ExecutablePlan& plan);
		static CompiledModuleArtifact CompileArtifact(const ExecutablePlan& plan, const CompilerOptions& options);
		static CompiledModuleArtifact CompileArtifact(const Runtime::RuntimeSchedule& schedule);
		static CompiledModuleArtifact CompileArtifact(const Runtime::RuntimeSchedule& schedule,
		                                              const CompilerOptions& options);
		static CompiledModule<CPU> Compile(const ExecutablePlan& plan);
		static CompiledModule<CPU> Compile(const ExecutablePlan& plan, const CompilerOptions& options);
		static CompiledModule<CPU> Compile(const Runtime::RuntimeSchedule& schedule);
		static CompiledModule<CPU> Compile(const Runtime::RuntimeSchedule& schedule, const CompilerOptions& options);
	};

#ifdef LITENN_ENABLE_CUDA
	enum class CUDAGraphReplayMode
	{
		Disabled,
		Enabled,
	};

	struct CompiledModuleCUDARunOptions
	{
		void* stream{};
		bool synchronize{ true };
		CUDAGraphReplayMode graphReplay{ CUDAGraphReplayMode::Disabled };
		bool enableCUBLASLt{ true };

		static CompiledModuleCUDARunOptions GraphReplay()
		{
			return CompiledModuleCUDARunOptions{ .graphReplay = CUDAGraphReplayMode::Enabled };
		}
	};

	struct CompiledModuleCUDATensorInvocation
	{
		std::span<const Tensor<CUDA>> inputs;
		std::span<Tensor<CUDA>> outputs;
		CompiledModuleCUDARunOptions options;
	};

	template <>
	class CompiledModule<CUDA>
	{
	public:
		CompiledModule();
		CompiledModule(const CompiledModule&);
		CompiledModule(CompiledModule&&) noexcept;
		CompiledModule& operator=(const CompiledModule&);
		CompiledModule& operator=(CompiledModule&&) noexcept;
		~CompiledModule();

		/// Loads a borrowed CUDA module image. CPU-native images bridge through CPU AOT.
		static CompiledModule Load(CompiledModuleImage image, CUDA device = CUDA{});
		static CompiledModule Load(CompiledModuleSeparatedImage image, CUDA device = CUDA{});
		static CompiledModule LoadBorrowedExternalRegions(CompiledModuleSeparatedImage image, CUDA device = CUDA{});

		std::vector<Tensor<CUDA>> RunTensors(std::span<const Tensor<CUDA>> inputs) const;
		std::vector<Tensor<CUDA>> RunTensors(std::span<const Tensor<CUDA>> inputs,
		                                     CompiledModuleCUDARunOptions options) const;
		void RunTensorsInto(std::span<const Tensor<CUDA>> inputs, std::span<Tensor<CUDA>> outputs) const;
		void RunTensorsInto(std::span<const Tensor<CUDA>> inputs, std::span<Tensor<CUDA>> outputs,
		                    CompiledModuleCUDARunOptions options) const;
		void RunManyTensorsInto(std::span<const CompiledModuleCUDATensorInvocation> invocations,
		                        std::size_t threadCount = 0) const;

		CompiledModuleImage Image() const;
		std::span<const std::byte> Rodata() const;
		std::span<const std::byte> Instructions() const;
		std::span<const CompiledTensorSpec> InputSpecs() const;
		std::span<const CompiledTensorSpec> OutputSpecs() const;
		CompiledModuleBackend Backend() const;
		std::optional<std::size_t> FindInput(std::string_view name) const;
		std::optional<std::size_t> FindOutput(std::string_view name) const;

		void WriteObjectFile(const std::filesystem::path& path, std::string_view symbolPrefix = "litenn_module") const;

	private:
		struct Impl;

		explicit CompiledModule(std::shared_ptr<Impl> impl);

		std::shared_ptr<Impl> impl_;
	};

	template <>
	class Compiler<CUDA>
	{
	public:
		static CompiledModuleArtifact CompileArtifact(const ExecutablePlan& plan);
		static CompiledModuleArtifact CompileArtifact(const ExecutablePlan& plan, const CompilerOptions& options);
		static CompiledModule<CUDA> Compile(const ExecutablePlan& plan, CUDA device = CUDA{});
		static CompiledModule<CUDA> Compile(const ExecutablePlan& plan, const CompilerOptions& options);
		static CompiledModule<CUDA> Compile(const ExecutablePlan& plan, CUDA device, const CompilerOptions& options);
	};
#endif

#ifdef LITENN_ENABLE_VULKAN
	struct VulkanNativeSupportReport
	{
		bool supported{};
		std::string capability;
		std::string reason;
	};

	struct CompiledModuleVulkanProfileEvent
	{
		std::size_t kernelIndex{};
		std::string entryPoint;
		VulkanDispatchDim groups;
		VulkanDispatchDim localSize;
		std::uint32_t descriptorCount{};
		double moduleCreationWallMs{};
		double dispatchWallMs{};
		bool gpuTimestampAvailable{};
		double gpuElapsedMs{};
	};

	struct CompiledModuleVulkanRunOptions
	{
		bool synchronize{ true };
		std::vector<CompiledModuleVulkanProfileEvent>* profileEvents{};
	};

	struct CompiledModuleVulkanTensorInvocation
	{
		std::span<const Tensor<Vulkan>> inputs;
		std::span<Tensor<Vulkan>> outputs;
		CompiledModuleVulkanRunOptions options;
	};

	class CompiledModuleVulkanRunWorkspace
	{
	public:
		std::span<Tensor<Vulkan>> Outputs()
		{
			return outputs_;
		}

		std::span<const Tensor<Vulkan>> Outputs() const
		{
			return outputs_;
		}

	private:
		std::vector<Tensor<Vulkan>> outputs_;
		std::vector<Tensor<CPU>> cpuInputs_;
		std::vector<Tensor<CPU>> cpuOutputs_;

		friend class CompiledModule<Vulkan>;
	};

	template <>
	class CompiledModule<Vulkan>
	{
	public:
		CompiledModule();
		CompiledModule(const CompiledModule&);
		CompiledModule(CompiledModule&&) noexcept;
		CompiledModule& operator=(const CompiledModule&);
		CompiledModule& operator=(CompiledModule&&) noexcept;
		~CompiledModule();

		static CompiledModule Load(CompiledModuleImage image, Vulkan device = Vulkan{});
		static CompiledModule Load(CompiledModuleSeparatedImage image, Vulkan device = Vulkan{});
		static CompiledModule LoadBorrowedExternalRegions(CompiledModuleSeparatedImage image, Vulkan device = Vulkan{});

		std::vector<Tensor<Vulkan>> RunTensors(std::span<const Tensor<Vulkan>> inputs) const;
		std::vector<Tensor<Vulkan>> RunTensors(std::span<const Tensor<Vulkan>> inputs,
		                                       CompiledModuleVulkanRunOptions options) const;
		std::vector<Tensor<Vulkan>> AllocateOutputTensors() const;
		CompiledModuleVulkanRunWorkspace CreateRunWorkspace() const;
		std::span<Tensor<Vulkan>> RunTensors(std::span<const Tensor<Vulkan>> inputs,
		                                     CompiledModuleVulkanRunWorkspace& workspace) const;
		std::span<Tensor<Vulkan>> RunTensors(std::span<const Tensor<Vulkan>> inputs,
		                                     CompiledModuleVulkanRunWorkspace& workspace,
		                                     CompiledModuleVulkanRunOptions options) const;
		void RunTensorsInto(std::span<const Tensor<Vulkan>> inputs, std::span<Tensor<Vulkan>> outputs) const;
		void RunTensorsInto(std::span<const Tensor<Vulkan>> inputs, std::span<Tensor<Vulkan>> outputs,
		                    CompiledModuleVulkanRunOptions options) const;
		void RunManyTensorsInto(std::span<const CompiledModuleVulkanTensorInvocation> invocations,
		                        std::size_t threadCount = 0) const;

		CompiledModuleImage Image() const;
		std::span<const std::byte> Rodata() const;
		std::span<const std::byte> Instructions() const;
		std::span<const CompiledTensorSpec> InputSpecs() const;
		std::span<const CompiledTensorSpec> OutputSpecs() const;
		CompiledModuleBackend Backend() const;
		std::optional<std::size_t> FindInput(std::string_view name) const;
		std::optional<std::size_t> FindOutput(std::string_view name) const;

		void WriteObjectFile(const std::filesystem::path& path, std::string_view symbolPrefix = "litenn_module") const;

	private:
		struct Impl;

		explicit CompiledModule(std::shared_ptr<Impl> impl);

		std::shared_ptr<Impl> impl_;
	};

	template <>
	class Compiler<Vulkan>
	{
	public:
		static VulkanNativeSupportReport QueryNativeSupport(const ExecutablePlan& plan);
		static CompiledModuleArtifact CompileArtifact(const ExecutablePlan& plan);
		static CompiledModuleArtifact CompileArtifact(const ExecutablePlan& plan, const CompilerOptions& options);
		static CompiledModule<Vulkan> Compile(const ExecutablePlan& plan, Vulkan device = Vulkan{});
		static CompiledModule<Vulkan> Compile(const ExecutablePlan& plan, const CompilerOptions& options);
		static CompiledModule<Vulkan> Compile(const ExecutablePlan& plan, Vulkan device,
		                                      const CompilerOptions& options);
	};
#endif
} // namespace LiteNN

#endif
