# LiteNN Performance Optimization Roadmap

This roadmap tracks the performance work derived from `PerformanceAnalysis_2026-05-16.md` and the current
CUDA AOT implementation state. It is intentionally separate from `Architecture.md` and `CUDAAOTRoadmap.md`:
those documents describe capability coverage, while this one tracks benchmark-driven performance work.

## Baseline

- Benchmark sources:
  - `benchmark/results/backend_pytorch_comparison_cpu_threads_2026-05-16.md`
  - `docs/PerformanceAnalysis_2026-05-19.md`
- CPU finding: default CPU AOT already emits packed AVX-512/zmm FMA kernels for the tested MNIST-like Linear/MLP
  objects. The old scalar CPU fast path was retired on 2026-05-19. A guarded large-static-f32 intra-op path has landed,
  but it is currently a modest improvement for the largest MLP case rather than the final CPU kernel strategy.
- CUDA finding: native non-graph execution is still dominated by launch/library scheduling on small graphs. CUDA Graph
  replay is the current fast path; on the local RTX 4090 run it matches PyTorch CUDA for
  `MLP(784->512->256->10)/batch:512` and remains slower mainly on tiny workloads with a fixed `0.03-0.05 ms` floor.

## Cross-Cutting Profiling Tooling

Goal: make slow paths diagnosable from one reproducible bundle instead of manually stitching together benchmark output,
object disassembly, backend CSVs, and platform profilers.

Status: first slice implemented. The canonical checklist lives in `docs/Roadmap.md` under G6.

- [ ] Add a whole-process profile bundle command that combines existing `litenn_profile` evidence with waterfall
  timeline output and optional platform sampling.
  - [x] First slice: `benchmark/profile_bundle.py` wraps `litenn_profile` or an arbitrary command, captures
    stdout/stderr, writes `manifest.json`, `trace.json`, and `summary.md`, and redacts user-specified sensitive paths
    from recorded artifacts.
  - [ ] Timeline output: fine-grained Chrome Trace / Perfetto JSON for import, conversion, lowering, MLIR/LLVM compile,
    object load, runtime schedule, transfers, synchronization, GPU dispatches, and decode-loop token phases.
    - [x] Qwen smoke slice: direct GGUF/Qwen smoke logs are streamed to disk during execution, the wrapper emits
          `qwen_smoke_trace.json` and `qwen_smoke_waterfall.md`, and the GGUF AOT path reports separated-cache
          population, artifact separation, cache read, and JIT/load timing explicitly.
    - [x] Cache-population visibility/fix: GGUF decode AOT cache writes now stream large regions in chunks with progress
          diagnostics and build separated metadata directly from the compiled artifact, avoiding an extra multi-GB
          `SeparateRodata()` weight copy during cache population. Borrowing/mapping original GGUF weight regions remains
          the G16.7 production target.
  - [x] Sampling raw capture: optional Linux `perf record` wrapper captures raw `perf.data` beside the bundle when
    requested.
  - [x] Sampling normalization: collapsed-stack inputs are merged and converted to `speedscope.json`.
  - [ ] Platform-native sampling import: optional Windows ETW/xperf, Linux `perf`, and macOS Instruments import
    adapters normalize their own raw formats into the collapsed-stack path.
  - [x] Flame graph output: collapsed-stack inputs render a simple built-in SVG/HTML flame graph; external renderers can
    still be added later for richer presentation.
  - Bundle output: raw logs, `trace.json`, `speedscope.json`, collapsed stacks, benchmark/profile CSVs, anonymized
    model/backend metadata, and a short Markdown bottleneck summary.

## P0: CUDA Native Hot-Path Fixed Costs

Goal: remove per-call host overhead that makes native CUDA MatMul appear 80-170x slower than PyTorch CUDA.

Status: implemented on 2026-05-16 for the current CUDA native runtime hot path.

- [x] Persist cuBLAS handles instead of creating/destroying one handle per MatMul call.
  - Implementation: `TryCUBLASMatMul` now uses a thread-local per-device handle cache and rebinds the stream per call.
  - Validation: `CUDANativeMatMul/batch:1/width:128` dropped from millisecond-level timing to `0.028 ms`.
- [x] Cache `CUfunction` lookup results.
  - Implementation: `CUDADriverModule` now owns a guarded function cache and supports eager `CacheFunction` at load time.
  - Validation: PTX function lookup is no longer a per-launch driver call.
- [x] Reuse CUDA native workspace allocations.
  - Implementation: `CompiledModule<CUDA>::Impl` owns one max-sized workspace buffer for the loaded native payload.
  - Validation: stable-shape payload execution no longer allocates/frees workspace inside each `RunInto`.
  - Safety boundary: asynchronous `RunInto` with a non-empty shared workspace is rejected until a workspace pool or event-owned lifetime model is added.
- [x] Rename CUDA bridge benchmark entries to CPU fallback terminology.
  - Implementation: benchmark entries now use `CUDACPUFallbackRunInto`.
  - Validation: benchmark list output makes native CUDA and CPU fallback paths visually distinct.

P0 validation run:

| Benchmark | Real time |
| --- | ---: |
| `CUDANativeMatMul/batch:1/width:128` | `0.028 ms` |
| `CUDANativeMatMul/batch:32/width:128` | `0.031 ms` |
| `CUDANativeMatMul/batch:128/width:128` | `0.029 ms` |
| `CUDANativeMatMul/batch:512/width:128` | `0.028 ms` |

## P1: CUDA Native Whole-Graph Scheduling

Goal: make real model benchmarks run through CUDA native instead of falling back to CPU AOT.

Status: implemented for fused inference Linear/MLP chains on 2026-05-16. Optional CUDA Graph replay is now
available for pointer-stable `RunInto` invocations through `CompiledModuleCUDARunOptions::enableGraphReplay`.

- [x] Add a static launch scheduler for single-subgraph CUDA native graphs.
  - Implementation: `CUDANativeInstructionPayload` launch tables now support mixed library-call and PTX kernels.
- [x] Allocate hidden activations from payload workspace.
  - Implementation: non-final fused layer outputs use native payload workspace and loaded modules reuse one workspace buffer.
- [x] Compile Linear and MLP chains into mixed launch payloads: cuBLAS MatMul plus MLIR/NVPTX epilogues.
  - Implementation: optimized `FusedOpNode(MatMulBiasAdd/ReLU)` chains with graph variables/constants lower to
    `litenn_cublas_matmul_f32` plus generated epilogue kernels.
  - Payload ABI: added constant tensor storage so static/shared-library loaded artifacts can carry model weights.
- [x] Add artifact inspection and CUDA runtime tests for multi-layer native MLP graphs.
  - Validation: `CompiledModuleCUDATest.CompilerArtifactsExposeNativeLinearChainPayload` and
    `CompiledModuleCUDATest.RunsNativeLinearChainWithConstantsAndWorkspace`.
- [x] Add optional CUDA Graph capture/replay after the launch table scheduler is stable.
  - Implementation: `CompiledModule<CUDA>` captures and caches `cudaGraphExec_t` per input/output pointer binding
    for synchronized default-stream CUDA-native `RunInto`; capture does a non-captured warm-up first so cuBLAS
    handles are initialized outside stream capture.
  - Validation: `CompiledModuleCUDATest.RunsNativeLinearChainWithCUDAGraphReplay`.

P1 validation spot check:

| Benchmark | AOT RunInto | CUDA Native RunInto |
| --- | ---: | ---: |
| `Linear(784->10)/batch:512` | `0.053 ms` | `0.060 ms` |
| `MLP(784->128->10)/batch:512` | `0.336 ms` | `0.118 ms` |
| `MLP(784->512->256->10)/batch:512` | `1.76 ms` | `0.234 ms` |

P1 CUDA Graph replay spot check:

| Benchmark | CUDA Native RunInto | CUDA Graph RunInto |
| --- | ---: | ---: |
| `Linear(784->10)/batch:512` | `0.060 ms` | `0.031 ms` |
| `MLP(784->128->10)/batch:512` | `0.118 ms` | `0.054 ms` |
| `MLP(784->512->256->10)/batch:512` | `0.234 ms` | `0.069 ms` |

## P2: CPU AOT Intra-Op Parallelism

Goal: close the gap with PyTorch CPU 16T on large batch and large hidden sizes.

Status: active again as of 2026-06-19. The initial implementation landed on 2026-05-19. The old 2026-05-16 experimental Linear/MLP runtime fast path was
removed after instruction-level profiling and focused benchmark validation. It lowered fused linear chains into calls
to a scalar C++ row kernel plus per-call thread creation, bypassing the MLIR-generated packed/zmm FMA kernel. The new
path keeps the small/medium default MLIR path and only tries a persistent-pool sidecar helper for large static f32 fused
Linear/MLP chains. Recent internal T1/T16 benchmark runs show that the sidecar path still needs production tuning: it can
beat simple rows, but it can also lose to the packed MLIR fallback on wide MLP shapes.

- [x] Profile the default CPU AOT object path at instruction level.
  - Result: generated objects use packed `zmm` FMA instructions and have no gather/scatter in the tested MNIST-like
    Linear/MLP cases.
- [x] Investigate and retire the experimental fast path.
  - Result: removed the extra runtime ABI, env controls, benchmark entries, and correctness test tied to the retired path.
- [x] Add a persistent worker pool for CPU AOT helper kernels.
  - Implementation: the pool is process-local, reuses worker threads, and only waits for workers participating in the
    current operation.
- [x] Add a guarded large-static-f32 fused Linear/MLP parallel path.
  - Implementation: `TryCompileCPUParallelLinearChainF32` emits an object that calls
    `litenn_cpu_matmul_bias_relu_parallel_f32`.
  - Gating: `LITENN_CPU_AOT_THREADS=1` falls back to MLIR; `LITENN_CPU_AOT_PARALLEL_MIN_FLOPS` defaults to `1 << 28`.
    Benchmark/profile entry points also honor `LITENN_COMPILE_DIAGNOSTICS=1` and forward it into
    `CompilerOptions::enableCompileDiagnostics` so actual sidecar selection/rejection reasons can be compared with
    `litenn_profile`'s predicted/object columns.
- [x] Improve the helper's local kernel quality enough for the large benchmark to benefit.
  - Implementation: row-bias initialization uses `memcpy`; helper pointers carry restrict semantics; GCC is given
    ivdep hints for the inner contiguous column loops.
- [x] Add the first cache-reuse microkernel improvement for the sidecar helper.
  - Implementation: the helper now processes rows in blocks of four so each RHS row load is reused across multiple
    output rows before advancing `k`.
- [x] Add benchmark labels for CPU AOT thread-policy comparison.
  - Implementation: `AOTRunIntoT1`, `AOTRunIntoT16`, `TrainCPUAOTT1`, and `TrainCPUAOTT16`.
- [x] Add correctness coverage for the new branch.
  - Validation: `CompiledModuleTest.CPUParallelLinearChainMatchesInterpreter` forces the branch and compares with the
    interpreter.
- [x] Add a shape-aware gate for the sidecar path.
  - Requirement: use the helper only when `m/k/n`, total FLOPs, and estimated thread overhead predict a win over the
    packed MLIR fallback; narrow final projections should not drag a whole fused chain into a slower path.
  - Implementation: the current conservative gate keeps very large row counts on the packed MLIR fallback and only
    enables multithread helper calls for sufficiently wide/high-FLOP layers; narrow tail projections use one helper
    thread inside an otherwise eligible chain. Tests and diagnostic runs can still force the sidecar path by setting
    `CompilerOptions::cpuAOTParallelMinFlops` to `1`.
- [x] Add a configurable worker-affinity policy for multithread experiments.
  - Implementation: `CompilerOptions::cpuAOTAffinityPolicy` defaults to `None`; `Compact` pins the CPU AOT helper's
    persistent worker threads to low-numbered CPUs while the policy remains active and restores their prior affinity
    when the policy is disabled or workers exit. Benchmark/profile entry points can set it through
    `LITENN_CPU_AOT_AFFINITY=compact`.
  - Evaluation: local Windows `MLP(784->512->256->10)/batch:128` sidecar-helper runs with
    `LITENN_CPU_AOT_THREADS=16` and `LITENN_CPU_AOT_PARALLEL_MIN_FLOPS=1` did not improve with compact affinity; the
    measured real time regressed versus no affinity. Keep affinity opt-in until topology-aware policies are available.
- [x] Add profile counters for helper layer shapes and selected grain/thread counts.
  - Implementation: `litenn_profile` now prints a CPU AOT parallel-selection table with fused layer count, selected
    parallel layer count, total FLOPs, per-layer `m/k/n`, selected helper threads, gate reason, and an emitted-object
    symbol check showing whether `litenn_cpu_matmul_bias_relu_parallel_f32` was actually used.
  - Boundary: this closes observability for the current sidecar helper; moving the work into optimized MLIR/LLVM
    lowering or a production GEMM backend remains open below.
- [ ] Move the parallel work into the optimized MLIR/LLVM lowering path or a production GEMM backend.
  - Requirement: the sidecar helper is acceptable as a first intra-op landing, but it does not preserve the MLIR
    packed/zmm microkernel and should not become the long-term CPU kernel architecture.

P2 retirement validation for the removed fast path:

| Benchmark | Default AOT RunInto | FastPath 1T | FastPath 16T |
| --- | ---: | ---: | ---: |
| `Linear(784->10)/batch:512` | `0.061 ms` | `1.53 ms` | `7.56 ms` |
| `MLP(784->128->10)/batch:512` | `0.393 ms` | `31.9 ms` | `16.1 ms` |
| `MLP(784->512->256->10)/batch:512` | `2.42 ms` | `356 ms` | `40.5 ms` |

Conclusion: default AOT already emits the better instruction stream. CPU multi-thread work should continue as a new
optimized-lowering task, not as the retired fast path.

P2 current validation:

| Benchmark | T1 / MLIR fallback | Default hardware-thread AOT | T16 |
| --- | ---: | ---: | ---: |
| `Linear(784->10)/batch:512` | `0.054 ms` | `0.053 ms` | `0.053 ms` |
| `MLP(784->128->10)/batch:512` | `0.337 ms` | `0.336 ms` | `0.357 ms` |
| `MLP(784->512->256->10)/batch:512` | `2.52 ms` | `1.76 ms` | `2.42 ms` |

Conclusion: intra-op parallelism is now implemented and guarded, but the present sidecar helper only helps the largest
local CPU case modestly. The next CPU performance step should parallelize the optimized lowering itself or call a
production GEMM backend.

Quantized validation note: `litenn_bench` now separates direct per-element weight decode,
`PreparedQuantizedLinear/<format>` execution, and dequantized reference execution. This makes repeated int4/fp4 weight
decode visible before investing in native packed microkernels.

## P3: CPU Kernel Refinement

Goal: continue smaller single-thread improvements after parallelism lands.

Status: implemented for static f32 constant-RHS MatMul shapes on 2026-05-16. Training-loss fusion is
deferred until P5 adds a training benchmark that can separate forward, backward, and loss costs.

- [x] Add K-block panel packing for large-K MatMul.
  - Implementation: `LowerNarrowMatMulPass` can materialize a K-panel packed RHS constant for wide static
    `MatMul` shapes and lower the row-tile kernel through the panelized layout.
  - Validation: `CompiledModuleTest.KPanelPackedWideMatMulMatchesReference`.
- [x] Replace fixed RHS packing thresholds with a static cost model.
  - Implementation: RHS packing now checks static `m/k/n`, output row reuse, constant f32 RHS storage, and
    estimated FLOPs versus packing bytes before selecting the packed path.
  - Scope: the new K-panel path is tried before the older packed wide-row path; the older path still keeps its
    conservative minimum-width guard to avoid destabilizing previously measured shapes.
- [x] Explore final-layer MatMulBiasAdd plus Softmax/CrossEntropy fusion for training workloads.
  - Result: deferred to P5. The current benchmark suite is inference-oriented, so adding this fusion now would
    create a correctness and API surface without a stable training performance signal.

P3 validation spot check:

| Validation | Result |
| --- | --- |
| `CompiledModuleTest.KPanelPackedWideMatMulMatchesReference` | Passed |
| Default `AOTRunInto/MLP(784->512->256->10)/batch:512` | `2.16 ms` |

## P4: CUDA Kernel Quality

Goal: improve native CUDA throughput once host overhead and graph scheduling are no longer dominant.

Status: target selection, optional cuBLASLt support, and optional CUDA Graph replay landed on 2026-05-16.
cuBLASLt remains opt-in because the local benchmark showed regressions on current small/medium inference shapes
when it was used by default. CUDA Graph replay is opt-in rather than default because it is pointer-binding
sensitive. Tensor Core codegen is evaluated and deferred until explicit fp16/bf16 tensor support exists.

- [x] Replace default `sm_30` target with `native` or a more modern baseline such as `sm_75`.
  - Implementation: the default NVPTX target is now `sm_75`; callers can pass `native` explicitly to query the
    current CUDA device and emit its `sm_<major><minor>` target.
  - Validation: `CompiledModuleTest.CUDANativeDefaultTargetUsesModernBaseline`.
- [x] Add cuBLASLt path and cache algorithm selection.
  - Implementation: when LiteNN is built with `CUDA::cublasLt`, the CUDA device runtime exposes an opt-in
    `CUDAExecutionOptions::enableCUBLASLt` MatMul path with per-device handle reuse and per-shape heuristic caching.
  - Default policy: disabled unless explicitly requested, because the current benchmark favors the existing
    cuBLAS path for the covered inference sizes.
- [x] Evaluate CUDA Graph replay as the default for static-shape inference payloads.
  - Result: implemented as opt-in, not default. The measured static-shape `RunInto` path benefits strongly,
    but graph executables capture raw input/output pointers, so default enablement should wait for an explicit
    pointer-stability contract or graph-node parameter update support.
- [x] Evaluate Tensor Core / WMMA generation only after cuBLASLt and scheduler measurements are stable.
  - Result: deferred. Float32 MNIST-style inference remains dominated by launch and library-call policy at the
    measured sizes; Tensor Core work should start from explicit fp16/bf16 tensor types and cuBLASLt policy
    measurements instead of hand-written WMMA in the current f32 path.

P4 validation spot check:

| Benchmark | Real time |
| --- | ---: |
| `CUDANativeMatMul/batch:1/width:128` | `0.036 ms` |
| `CUDANativeMatMul/batch:32/width:128` | `0.047 ms` |
| `CUDANativeMatMul/batch:128/width:128` | `0.038 ms` |
| `CUDANativeMatMul/batch:512/width:128` | `0.031 ms` |
| `CUDANativeGraphRunInto/Linear(784->10)/batch:512` | `0.031 ms` |
| `CUDANativeGraphRunInto/MLP(784->128->10)/batch:512` | `0.054 ms` |
| `CUDANativeGraphRunInto/MLP(784->512->256->10)/batch:512` | `0.069 ms` |

## P5: Training Benchmark Baseline

Goal: make training bottlenecks visible before optimizing them.

Status: implemented on 2026-05-16.

- [x] Add `bench_train.cpp` for MNIST Linear, MLP-128, and MLP-512.
  - Implementation: `litenn_bench_train` covers synthetic MNIST-shaped Linear, MLP-128, and MLP-512 batches.
- [x] Report forward, backward, optimizer step, and full step timings separately.
  - Implementation: `TrainCPUInterpreter/{Forward,Backward,OptimizerStep,FullStep}` benchmark families.
- [x] Track CPU AOT T1/T16, CUDA CPU fallback, and CUDA native variants independently.
  - Implementation: training forward baselines are registered as `TrainCPUAOT`, `TrainCPUAOTT1`,
    `TrainCPUAOTT16`, `TrainCUDACPUFallback`, and `TrainCUDANative`; CUDA CPU fallback uses
    `LITENN_CUDA_DISABLE_NATIVE_AOT=1` to keep the bridge measurable after native coverage expanded.
- [x] Add a CPU AOT trainer full-step row for the current compiled-training subset.
  - Implementation: `TrainCPUAOT/FullStep/MNIST-Linear` and `TrainCPUAOT/FullStep/MNIST-MLP128` run through
    `Trainer` with `TrainExecutionPolicy::AOT`; forward saved activations are passed into the compiled backward entry
    as explicit ABI params.
  - Metric split: the row reports `compile_ms` as a setup counter, while the benchmark timer measures train-step latency.

P5 validation spot check:

| Benchmark | Real time |
| --- | ---: |
| `TrainCPUAOT/FullStep/MNIST-Linear/batch:32` | `156 ms` Debug smoke; `compile_ms=232` |
| `TrainCPUAOT/FullStep/MNIST-MLP128/batch:32` | `290 ms` Debug smoke; `compile_ms=438` |
| `TrainCPUInterpreter/FullStep/MNIST-MLP128/batch:512` | `17.84 ms` |
| `TrainCPUInterpreter/FullStep/MNIST-MLP512/batch:512` | `86.18 ms` |
| `TrainCUDANative/Forward/MNIST-MLP128/batch:512` | `0.188 ms` |
| `TrainCUDANative/Forward/MNIST-MLP512/batch:512` | `0.905 ms` |

## Validation Checklist

- Build focused targets with `cmd /c cmake --build ...`.
- Run `CompiledModuleTest`, `CompiledModuleCUDATest`, and `CUDADeviceTest` after CUDA runtime changes.
- Run `litenn_bench --benchmark_filter=CUDANativeMatMul` before and after P0 changes.
- Keep raw benchmark output under `benchmark/results/` and summarize cross-backend numbers in Markdown. PyTorch
  comparison runs should use `python311 benchmark/bench.py`.
