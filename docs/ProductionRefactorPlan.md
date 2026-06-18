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
- [ ] Add named entry-point metadata for future multi-entry artifacts: `forward`, `loss`, `backward`, `optimizer_step`,
      and backend-specific entry names.
- [ ] Normalize state binding for KV cache, diffusion latent/state, training activations, optimizer state, and LoRA
      adapters.
- [ ] Make fallback/transfer/profile records available from the same schedule representation across CPU, CUDA, Vulkan,
      and heterogeneous execution.
- [ ] Define an ABI version bump rule for changes to tensor binding, external regions, backend requirements, and runtime
      state.

Exit criteria:

- Adding a backend or training entry point extends the shared ABI instead of inventing a parallel binding path.
- Separated rodata/weights/instructions and vNext packages describe the same logical object model.

## Phase 3: Backend Production Choices

CPU:

- [ ] Keep CPU interpreter as the reference path.
- [ ] Decide whether CPU production kernels use an external library backend or a small maintained native kernel set.
- [ ] Avoid expanding hand-written CPU GEMM/Conv kernels without a clear backend strategy.

CUDA:

- [ ] Keep native CUDA support capability-gated.
- [x] Make CUDA Graph replay an explicit production fast-path policy instead of a loose boolean switch; callers request
      it through `CompiledModuleCUDARunOptions::GraphReplay()`, and unsupported stream/synchronization combinations fail
      loudly instead of silently falling back to non-graph launch.
- [ ] Prioritize high-value kernels: Linear/MatMul, normalization, reductions, attention, and quantized projection.
- [ ] Keep host fallback explicit and visible in schedules/profile output.

Vulkan/mobile:

- [ ] Finish graph partitioning and device-local memory planning before claiming broad mobile GPU production support.
- [ ] Keep desktop Vulkan and mobile Vulkan as separate support profiles.
- [ ] Require explicit skip/failure behavior when a device lacks required storage, subgroup, timestamp, or alignment
      capabilities.

Exit criteria:

- Backend support is stated as capability plus verified workload, not as a blanket backend label.
- Benchmark rows cannot silently compare native execution with hidden CPU fallback.

## Phase 4: Model Import Boundaries

GGUF / llama.cpp:

- [ ] Support selected LLaMA-family model signatures as named profiles.
- [ ] Reject unsupported llama.cpp ops and layouts with actionable diagnostics instead of chasing full operator parity.
- [ ] Keep external llama.cpp golden validation as the acceptance criterion for new profiles.

Torch / safetensors:

- [ ] Keep safetensors as tensor storage, not graph architecture discovery.
- [ ] Require explicit manifest/config information for graph construction.
- [ ] Preserve layout, dtype, quantization, and LoRA mapping diagnostics in importer reports.

SDXL:

- [ ] Treat SDXL as a large-model importer and memory-policy stress target until tokenizer, text encoders, UNet, VAE,
      scheduler, and reference-image validation are all production-gated.
- [ ] Do not let SDXL completeness block the vNext production profile.

Exit criteria:

- Importers produce explicit manifests/packages with diagnostics.
- No importer silently guesses architecture-critical semantics that should come from config or source manifests.

## Phase 5: Quantization and Low Precision

- [ ] Keep scalar low precision and block-quantized storage as separate concepts.
- [ ] Do not add byte-addressable fake `DataType` values for int4/fp4.
- [ ] Define packed 4-bit storage through quantization/storage metadata first.
- [ ] Implement CPU reference pack/unpack/dequantize before native kernels.
- [ ] Add native quantized Linear/MatMul only after storage, package, and parity contracts are stable.

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
