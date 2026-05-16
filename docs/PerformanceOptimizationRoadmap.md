# LiteNN Performance Optimization Roadmap

This roadmap tracks the performance work derived from `PerformanceAnalysis_2026-05-16.md` and the current
CUDA AOT implementation state. It is intentionally separate from `Architecture.md` and `CUDAAOTRoadmap.md`:
those documents describe capability coverage, while this one tracks benchmark-driven performance work.

## Baseline

- Benchmark source: `benchmark/results/backend_pytorch_comparison_2026-05-16.md`
- Main finding: CPU AOT single-thread performance is already competitive with PyTorch CPU 1T, but trails
  PyTorch CPU 16T on larger MLP batches because LiteNN CPU kernels are still single-threaded.
- Main CUDA finding: CUDA native MatMul is dominated by per-call host overhead; CUDA CPU bridge fallback
  is not GPU execution and should not be interpreted as CUDA backend throughput.

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

Status: implemented for fused inference Linear/MLP chains on 2026-05-16. CUDA Graph replay remains a later
optional launch-overhead reduction after more native graph shapes are covered.

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
- [ ] Add optional CUDA Graph capture/replay after the launch table scheduler is stable.
  - Deferred: current shared-workspace native payloads reject asynchronous execution until a workspace pool or event-owned
    lifetime model is added.

P1 validation spot check:

| Benchmark | AOT RunInto | CUDA Native RunInto |
| --- | ---: | ---: |
| `MLP(784->128->10)/batch:512` | `0.348 ms` | `0.142 ms` |
| `MLP(784->512->256->10)/batch:512` | `2.34 ms` | `0.217 ms` |

## P2: CPU AOT Intra-Op Parallelism

Goal: close the gap with PyTorch CPU 16T on large batch and large hidden sizes.

Status: runtime ABI and an experimental fused Linear/MLP fast path landed on 2026-05-16, but the new path is
opt-in via `LITENN_CPU_AOT_LINEAR_CHAIN_FASTPATH=1` because the current row kernel is a correctness/ABI proving
step and does not yet reuse the existing MLIR micro-kernel pipeline. Default CPU AOT still uses the previous
optimized object path to avoid benchmark regressions.

- [x] Add a small runtime `ParallelFor` ABI for compiled CPU modules.
  - Implementation: exported `litenn_cpu_parallel_for_u64` plus JIT symbol registration.
- [x] Split MatMul along the output row/batch dimension, keeping K serial to avoid partial-sum reduction.
  - Implementation: experimental `litenn_cpu_matmul_bias_relu_f32` splits row ranges and calls the same serial row kernel
    per range.
- [x] Gate experimental parallel lowering by static FLOP and output-size thresholds.
  - Implementation: fused Linear/MLP lowering is opt-in and only matches static-shape f32 chains; runtime execution stays
    serial below `1 << 20` estimated FLOPs or when `LITENN_CPU_AOT_THREADS=1`.
- [x] Verify small-batch serial performance does not regress.
  - Validation: `CompiledModuleTest.CPULinearChainFastPathMatchesInterpreter` covers correctness with the opt-in path;
    benchmark validation showed the naive row kernel is slower than the existing default CPU AOT path, so default
    enablement is deferred.

P2 validation spot check with the experimental path disabled by default:

| Benchmark | Default AOT RunInto |
| --- | ---: |
| `Linear(784->10)/batch:1` | `0.001 ms` |
| `MLP(784->512->256->10)/batch:512` | `2.34 ms` |

## P3: CPU Kernel Refinement

Goal: continue smaller single-thread improvements after parallelism lands.

- [ ] Add K-block panel packing for large-K MatMul.
- [ ] Replace fixed RHS packing thresholds with a static cost model.
- [ ] Explore final-layer MatMulBiasAdd plus Softmax/CrossEntropy fusion for training workloads.

## P4: CUDA Kernel Quality

Goal: improve native CUDA throughput once host overhead and graph scheduling are no longer dominant.

- [ ] Replace default `sm_30` target with `native` or a more modern baseline such as `sm_75`.
- [ ] Add cuBLASLt path and cache algorithm selection.
- [ ] Evaluate CUDA Graph replay as the default for static-shape inference payloads.
- [ ] Evaluate Tensor Core / WMMA generation only after cuBLASLt and scheduler measurements are stable.

## P5: Training Benchmark Baseline

Goal: make training bottlenecks visible before optimizing them.

- [ ] Add `bench_train.cpp` for MNIST MLP-128 and MLP-512.
- [ ] Report forward, backward, optimizer step, and full step timings separately.
- [ ] Track CPU AOT, CUDA CPU fallback, and CUDA native variants independently.

## Validation Checklist

- Build focused targets with `cmd /c cmake --build ...`.
- Run `CompiledModuleTest`, `CompiledModuleCUDATest`, and `CUDADeviceTest` after CUDA runtime changes.
- Run `litenn_bench --benchmark_filter=CUDANativeMatMul` before and after P0 changes.
- Keep raw benchmark output under `benchmark/results/` and summarize cross-backend numbers in Markdown.
