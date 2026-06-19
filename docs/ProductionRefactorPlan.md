# LiteNN Production Refactor Plan

This document captures the current refactor direction after the vNext break-window work. It is intentionally narrower
than `Roadmap.md`: the goal is to turn LiteNN from a broad capability prototype into a smaller set of production-grade
paths.

## Direction

The next phase should prioritize deployable execution paths over more feature breadth.

The latest direction-setting benchmark snapshot is recorded in
[`PerformanceBenchmark_2026-06-18.md`](PerformanceBenchmark_2026-06-18.md). The short version is: CPU AOT has already
removed most graph/runtime overhead, CUDA Graph replay is the only competitive CUDA model path, and Vulkan needs
device-local memory plus MatMul/schedule work before it can be treated as a broad performance backend.

Production profile:

- Core graph construction through `ModelGraph` / `ModelBuilder`.
- Executable-plan validation as the stable runtime/compiler contract.
- vNext model packages with separated tensor/artifact regions.
- CPU reference execution for correctness and diagnostics.
- CPU AOT as the first production package/load path.
- CUDA and Vulkan as optional native backends with explicit fallback policy.
- Importer-owned manifests for GGUF, Torch, safetensors, and LoRA.
- Benchmark/profile output for every production-supported backend path.

Non-production profile:

- Experimental SDXL full generation.
- Broad llama.cpp operator compatibility beyond selected model families.
- Full AOT training.
- Complete heterogeneous graph execution.
- Native int4/fp4/block-quantized kernels.
- Wide Vulkan mobile device matrix.

These are still valuable, but they should not block stabilizing the production profile unless a concrete product target
requires them.

## Phase 1: Stabilize the Production Surface

- [x] Write a short production support matrix covering runtime, compiler, importer, packaging, benchmark, and mobile
      support levels. `LiteNN/ProductionSupport.h` exposes the same matrix to tools and CLIs.
- [x] Make every supported path name its package/runtime ABI: inputs, outputs, mutable state, external tensors,
      ownership, alignment, checksum, and fallback policy. `LiteNN/ProductionSupport.h` now exposes
      `QueryProductionPathABIs()` and ABI diagnostics for tooling, examples, and CI gates.
- [x] Ensure examples only use vNext packages, separated artifacts, or importer manifests for production flows.
      `G14PublicApiGuard.ProductionExamplesUseVNextPackagesAndManifests` scans `example/` recursively and the SDXL
      CLI now exposes package-only import/load wording.
- [x] Keep all old graph-archive or raw-Graph convenience paths out of public production APIs. The G14 guard covers
      removed graph archive names, raw `Graph&` overloads, unsafe graph access naming, and production example drift.
- [x] Add CI profiles for minimal runtime, importers, compiler/AOT, tools/examples, and optional GPU-enabled builds.
      `CMakePresets.json` defines the production profiles, `.github/workflows/ci.yml` runs the CPU/default matrix, and
      `ci-optional-gpu` is available for self-hosted CUDA/Vulkan runners.

Exit criteria:

- A user can identify the supported LiteNN deployment profile without reading the entire roadmap.
- CI can catch accidental reintroduction of old APIs or deployment-specific dependencies in the core runtime.
- CPU AOT package load/run is the reference production deployment path.

## Phase 2: Runtime and Artifact ABI Convergence

- [x] Treat runtime schedule, compiled artifact metadata, vNext package manifests, and external tensor binding as one ABI
      family rather than separate backend-specific conventions. `DescribeVNextABIFamily()` now exposes a shared summary
      over package versions, functions, runtime state/buffer bindings, tensor bindings, and artifact entries/regions.
- [x] Add named entry-point metadata for future multi-entry artifacts: `forward`, `loss`, `backward`, `optimizer_step`,
      and backend-specific entry names. vNext artifact entries now carry an explicit entry kind in memory, package JSON,
      ABI summaries, and manifest validation.
- [x] Normalize state binding for KV cache, diffusion latent/state, training activations, optimizer state, and LoRA
      adapters. Runtime state bindings now round-trip through vNext package JSON and are validated against persistent
      memory buffers.
- [x] Make fallback/transfer/profile records available from the same schedule representation across CPU, CUDA, Vulkan,
      and heterogeneous execution. `RuntimeScheduleProfileRecord` now derives profile rows from runtime schedule steps,
      and vNext packages preserve fallback backend metadata in runtime step JSON.
- [x] Define an ABI version bump rule for changes to tensor binding, external regions, backend requirements, and runtime
      state. `VNextABIVersionBumpRuleFor()` maps production ABI changes to the required version component; tensor
      binding, external regions, backend requirements, runtime state, runtime schedule, and artifact entries require an
      `artifactABI` bump.

Exit criteria:

- Adding a backend or training entry point extends the shared ABI instead of inventing a parallel binding path.
- Separated rodata/weights/instructions and vNext packages describe the same logical object model.

## Phase 3: Backend Production Choices

CPU:

- [x] Keep CPU interpreter as the reference path. `ProductionBackendProfile::CPUReferenceInterpreter` is the explicit
      reference correctness profile.
- [x] Decide whether CPU production kernels use an external library backend or a small maintained native kernel set.
      `QueryProductionCPUKernelStrategy()` records the policy: prefer an explicit external-library backend while allowing
      a small maintained native kernel set.
- [x] Avoid expanding hand-written CPU GEMM/Conv kernels without a clear backend strategy. New handwritten CPU
      GEMM/Conv kernels are gated on measured gaps, workload ownership, and benchmark evidence.

CUDA:

- [x] Keep native CUDA support capability-gated. `ProductionSupport.h` now exposes
      `QueryProductionCUDANativeCapabilities()` so CUDA native is reported as per-feature capability gates rather than
      a blanket backend label.
- [x] Make CUDA Graph replay an explicit production fast-path policy instead of a loose boolean switch; callers request
      it through `CompiledModuleCUDARunOptions::GraphReplay()`, and unsupported stream/synchronization combinations fail
      loudly instead of silently falling back to non-graph launch.
- [x] Prioritize high-value kernels: Linear/MatMul, normalization, reductions, attention, and quantized projection.
      The production capability table marks MatMul/Linear/reductions/low-precision MatMul as verified or experimental
      according to current support, while normalization, attention, and quantized projection remain explicit deferred
      high-priority capabilities until implementation and benchmark/parity evidence land.
- [x] Keep host fallback explicit and visible in schedules/profile output.

Vulkan/mobile:

- [ ] Finish graph partitioning and device-local memory planning before claiming broad mobile GPU production support.
      `QueryMobileVulkanProductionGateStatuses()` now records graph partitioning, explicit fallback visibility, and
      separated artifact regions as available gates, while device-local memory planning and mobile device-matrix
      validation remain constrained blockers for broad mobile GPU production claims.
- [x] Keep desktop Vulkan and mobile Vulkan as separate support profiles.
      `ProductionBackendProfile::VulkanDesktopNative` and `VulkanMobileConstrained` report different scope, fallback,
      and capability policies.
- [x] Require explicit skip/failure behavior when a device lacks required storage, subgroup, timestamp, or alignment
      capabilities. Backend profiles now expose the skip/failure policy that Vulkan tests and device probes must honor.

Exit criteria:

- Backend support is stated as capability plus verified workload, not as a blanket backend label.
- Benchmark rows cannot silently compare native execution with hidden CPU fallback.

## Phase 4: Model Import Boundaries

GGUF / llama.cpp:

- [x] Support selected LLaMA-family model signatures as named profiles.
      `QueryLLaMACompatibilityProfiles()` exposes tiny-fixture, LLaMA2-like causal LM, and LLaMA3-like causal LM
      profile descriptors for GGUF tooling without promising broad llama.cpp parity.
- [x] Reject unsupported llama.cpp ops and layouts with actionable diagnostics instead of chasing full operator parity.
      `AnalyzeLLaMACompatibility()` reports blocking diagnostics for missing required tensors, unsupported RoPE variants,
      invalid head/cache layout, and metadata that current lowering cannot execute safely.
- [x] Keep external llama.cpp golden validation as the acceptance criterion for new profiles.
      Production-selected LLaMA profiles explicitly require external llama.cpp golden logits before claiming model-family
      acceptance; internal tiny fixtures remain regression coverage only.

Torch / safetensors:

- [x] Keep safetensors as tensor storage, not graph architecture discovery.
      `ImportSafetensorsVariablesManifest()` now emits an importer-owned manifest with an explicit
      tensor-storage-only diagnostic instead of implying that safetensors can discover architecture.
- [x] Require explicit manifest/config information for graph construction.
      Safetensors variable import records a `MissingMetadata` diagnostic for production graph construction; Torch
      graph construction remains manifest-driven through `ImportTorchManifest`.
- [x] Preserve layout, dtype, quantization, and LoRA mapping diagnostics in importer reports.
      Safetensors manifest weight mappings now record source/graph tensor types, layout conversion,
      quantization mapping, and LoRA binding; Torch/LoRA import paths continue to expose their detailed reports.

SDXL:

- [ ] Treat SDXL as a large-model importer and memory-policy stress target until tokenizer, text encoders, UNet, VAE,
      scheduler, and reference-image validation are all production-gated.
- [ ] Do not let SDXL completeness block the vNext production profile.

Exit criteria:

- Importers produce explicit manifests/packages with diagnostics.
- No importer silently guesses architecture-critical semantics that should come from config or source manifests.

## Phase 5: Quantization and Low Precision

- [x] Keep scalar low precision and block-quantized storage as separate concepts.
      `DataType` remains the byte-addressable scalar dtype surface, while `QuantizationParams` describes affine, block,
      packed-nibble, and external storage semantics.
- [x] Do not add byte-addressable fake `DataType` values for int4/fp4.
      Int4, UInt4, FP4E2M1, and FP4E3M0 are represented as `PackedNibbleFormat` storage metadata instead of scalar
      dtype values.
- [x] Define packed 4-bit storage through quantization/storage metadata first.
      Packed nibble format, nibble order, scale layout, storage dtype, and logical element count are queryable through
      quantization metadata and preserved by vNext package metadata.
- [x] Implement CPU reference pack/unpack/dequantize before native kernels.
      CPU reference helpers and graph/const-fold dequantization tests cover integer and float packed-nibble storage.
- [ ] Add native quantized Linear/MatMul only after storage, package, and parity contracts are stable.
      `QueryProductionQuantizationCapabilities()` now reports `NativeQuantizedLinearMatMul` as a deferred native-kernel
      capability that requires parity and benchmark evidence.

Exit criteria:

- A quantized model can be inspected, packaged, rebound, and dequantized deterministically before optimized kernels exist.
- Native int4/fp4 kernels are performance improvements, not semantic foundations.

## Things to Remove or Demote

Remove or keep out of production paths:

- Pre-vNext graph archive compatibility.
- Raw `Graph&` public runtime/compiler APIs.
- Hidden CPU fallback inside GPU paths.
- Environment-variable-only behavior switches in library internals.
- Ad hoc JSON/parsing utilities when an approved dependency exists.
- Broad compatibility promises such as "all llama.cpp ops" or "full SDXL generation" without fixture-backed gates.

Demote to non-blocking or long-term:

- Full SDXL image generation quality parity.
- Exact RWKV6/7, gated attention, and uncommon ggml model-family signatures.
- Full compiled AOT training.
- Broad Vulkan mobile device farms.
- Hand-written CPU production GEMM/Conv unless chosen as the explicit CPU backend strategy.

## Suggested Immediate Order

1. Add a production support matrix and package/runtime ABI spec.
2. Split roadmap history away from active checklist work.
3. Add CI/build profiles for the supported production surface.
4. Make CPU AOT vNext package deployment the default example path.
5. Pick the next performance project: either CPU production backend integration or native quantized Linear/MatMul.
