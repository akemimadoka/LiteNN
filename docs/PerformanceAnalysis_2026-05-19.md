# LiteNN Instruction-Level Performance Analysis - 2026-05-19

This report follows `docs/PerformanceAnalysis_2026-05-16.md` and focuses on the CPU AOT instruction stream,
the local profile tool output, CUDA native/profile behavior, and the current CPU intra-op parallelism landing.

## Inputs

- `benchmark/results/backend_ggml_pytorch_comparison_2026-05-19.md`
- `benchmark/results/profile_default_2026-05-19.txt`
- `benchmark/results/profile_after_cpu_parallel_2026-05-19.txt`
- `benchmark/results/cpu_aot_intraop_batch512_2026-05-19.txt`
- `benchmark/results/cuda_native_batch512_2026-05-19.txt`
- `benchmark/results/cuda_native_matmul_2026-05-19.txt`
- `benchmark/results/pytorch_cuda_2026-05-19.txt`
- Object files emitted by `build-release/benchmark/litenn_profile.exe`
- Disassembly stats from `benchmark/analyze_asm.ps1`

Representative commands:

```powershell
cmd /c "set PATH=C:\msys64\mingw64\bin;%PATH% && cmake --build build-release --target litenn_profile litenn_bench --parallel"
cmd /c "set PATH=C:\msys64\mingw64\bin;%PATH% && build-release\benchmark\litenn_profile.exe benchmark\results\profile_after_cpu_parallel_2026-05-19 > benchmark\results\profile_after_cpu_parallel_2026-05-19.txt"
cmd /c "set PATH=C:\msys64\mingw64\bin;%PATH% && build-release\benchmark\litenn_bench.exe --benchmark_filter=AOT.*batch:512 --benchmark_min_time=0.05s"
cmd /c "set PATH=C:\msys64\mingw64\bin;%PATH% && build-release\benchmark\litenn_bench.exe --benchmark_filter=CUDANative.*batch:512 --benchmark_min_time=0.1s"
python311 benchmark\bench.py --device cuda
```

## Summary

The default CPU AOT path is not scalar. The generated hot kernels use packed AVX-512/zmm FMA instructions, and the
profiled MNIST-like objects contain no gather/scatter operations. The original CPU "fast path" was removed because it
bypassed that packed MLIR kernel and replaced it with a scalar sidecar kernel.

CPU intra-op parallelism now exists as a conservative large-static-f32 fused Linear/MLP chain path. It uses a persistent
worker pool and the runtime symbol `litenn_cpu_matmul_bias_relu_parallel_f32`, but it is gated by thread count and FLOPs:
`LITENN_CPU_AOT_THREADS=1` falls back to the MLIR AOT path, and `LITENN_CPU_AOT_PARALLEL_MIN_FLOPS` defaults to
`1 << 28`. This avoids polluting small/medium models.

Current CPU result: small Linear/MLP128 stay on the MLIR path; the large `MLP(784->512->256->10)/batch:512` benchmark
improves from the single-thread MLIR label at about `2.52 ms` to the default hardware-thread AOT path at about `1.76 ms`.
This is a real but still partial gain. The T16 variant is slower than the default hardware-thread policy on this local
32-thread machine, so the current path should be treated as a first intra-op landing, not the final CPU kernel strategy.

CUDA is much clearer: without CUDA Graph replay, native whole-graph execution is dominated by launch/library scheduling.
With graph replay, `MLP(784->512->256->10)/batch:512` reaches `0.069 ms`, matching the local PyTorch CUDA result. The
remaining CUDA gap is concentrated in tiny workloads where the fixed execution floor is around `0.03-0.05 ms`.

## CPU Instruction-Level Findings

Disassembly stats for `subgraph_0` in default CPU AOT objects:

| Object | Lines | Packed FMA | zmm FMA | ymm FMA | xmm FMA | Gather | Scatter | Vector loads | Broadcasts | Stack vector ops |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `linear_b512.o` | 293 | 32 | 32 | 0 | 0 | 0 | 0 | 40 | 0 | 0 |
| `mlp128_b512.o` | 684 | 160 | 160 | 0 | 0 | 0 | 0 | 88 | 64 | 0 |
| `mlp512_b512.o` | 1076 | 288 | 288 | 0 | 0 | 0 | 0 | 136 | 128 | 16 |

Representative instructions in `mlp512_b512.s` include packed `vfmadd231ps` on zmm registers.

Interpretation:

- SIMD codegen is active and targeting AVX-512 width on this machine.
- There are no gather/scatter instructions in these cases, so the core loop uses regular contiguous vector memory access.
- No prefetch instructions were emitted. That may matter for larger hidden sizes, but the larger practical gap was
  missing intra-op scheduling.

## CPU Profile Results

Default CPU AOT profile before the new intra-op path:

| Case | Compile | Run | RunInto | Run - RunInto |
| --- | ---: | ---: | ---: | ---: |
| `linear_b512` | 39.36 ms | 0.0577 ms | 0.0569 ms | 0.80 us |
| `mlp128_b512` | 81.90 ms | 0.3632 ms | 0.3764 ms | -13.26 us |
| `mlp512_b512` | 293.33 ms | 3.6425 ms | 2.9543 ms | 688.20 us |

After CPU intra-op landing:

| Case | Compile | Run | RunInto | Run - RunInto |
| --- | ---: | ---: | ---: | ---: |
| `linear_b512` | 38.15 ms | 0.0532 ms | 0.0529 ms | 0.22 us |
| `mlp128_b512` | 81.80 ms | 0.3404 ms | 0.3409 ms | -0.57 us |
| `mlp512_b512` | 42.95 ms | 1.6940 ms | 1.7430 ms | -48.98 us |

Focused benchmark after the landing:

| Benchmark | Real time |
| --- | ---: |
| `AOTRunInto/Linear(784->10)/batch:512` | 0.053 ms |
| `AOTRunIntoT1/Linear(784->10)/batch:512` | 0.054 ms |
| `AOTRunInto/MLP(784->128->10)/batch:512` | 0.336 ms |
| `AOTRunIntoT1/MLP(784->128->10)/batch:512` | 0.337 ms |
| `AOTRunInto/MLP(784->512->256->10)/batch:512` | 1.76 ms |
| `AOTRunIntoT1/MLP(784->512->256->10)/batch:512` | 2.52 ms |
| `AOTRunIntoT16/MLP(784->512->256->10)/batch:512` | 2.42 ms |

The profile tool and benchmark harness do not measure exactly the same surface, so the absolute numbers should not be
mixed as one time series. The consistent signal is that only the large static MLP benefits, and the gain is not yet large
enough to close the PyTorch CPU 16T gap.

## CUDA Profile Results

Sequential CUDA native and PyTorch CUDA batch-512 results:

| Model | LiteNN CUDA Native | LiteNN CUDA Graph | PyTorch CUDA | Graph vs PyTorch |
| --- | ---: | ---: | ---: | ---: |
| `Linear(784->10)` | 0.060 ms | 0.031 ms | 0.016 ms | +94% |
| `MLP(784->128->10)` | 0.118 ms | 0.054 ms | 0.045 ms | +20% |
| `MLP(784->512->256->10)` | 0.234 ms | 0.069 ms | 0.069 ms | ~0% |

Standalone CUDA native MatMul floor:

| Benchmark | Real time |
| --- | ---: |
| `CUDANativeMatMul/batch:1/width:128` | 0.036 ms |
| `CUDANativeMatMul/batch:32/width:128` | 0.047 ms |
| `CUDANativeMatMul/batch:128/width:128` | 0.038 ms |
| `CUDANativeMatMul/batch:512/width:128` | 0.031 ms |

Interpretation:

- CUDA Graph replay is the most important CUDA optimization currently in tree. It removes most host-side launch table
  overhead for pointer-stable inference.
- Non-graph native execution is still launch/library-call limited for these small graphs.
- The large MLP batch-512 path is no longer meaningfully slower than PyTorch CUDA once graph replay is enabled.
- The tiny Linear case cannot amortize LiteNN's fixed CUDA execution floor yet; this is an overhead-policy issue more
  than a math-kernel issue.

## What "Fast Path" Was

The removed fastpath consisted of:

- Env gate: `LITENN_CPU_AOT_LINEAR_CHAIN_FASTPATH=1`
- Thread count override: `LITENN_CPU_AOT_THREADS`
- Compiler shortcut: `TryCompileCPULinearChainF32`
- Runtime ABI symbols: `litenn_cpu_parallel_for_u64` and `litenn_cpu_matmul_bias_relu_f32`
- Benchmark labels: `AOTFastPathRunIntoT1`, `AOTFastPathRunIntoT16`, `TrainCPUAOTFastPathT1`, `TrainCPUAOTFastPathT16`

It was not an optimized MLIR path. It materialized constants into an LLVM module, allocated intermediates with
`malloc/free`, and called a C++ helper that computed MatMul/Bias/ReLU using scalar nested loops. The 16-thread version
created and joined worker threads during the operation, which was expensive for these matrix sizes.

Focused validation before removal:

| Benchmark | Default AOT RunInto | FastPath 1T | FastPath 16T |
| --- | ---: | ---: | ---: |
| `Linear(784->10)/batch:512` | 0.061 ms | 1.53 ms | 7.56 ms |
| `MLP(784->128->10)/batch:512` | 0.393 ms | 31.9 ms | 16.1 ms |
| `MLP(784->512->256->10)/batch:512` | 2.42 ms | 356 ms | 40.5 ms |

Decision: remove it. Keeping the code made benchmarks harder to read and risked future accidental enablement.

## Changes Made

- Removed the old CPU linear-chain fastpath compiler branch and benchmark labels.
- Added a persistent CPU worker pool for AOT runtime helpers.
- Added a guarded large-static-f32 fused Linear/MLP compiler branch using
  `litenn_cpu_matmul_bias_relu_parallel_f32`.
- Added restrict/vectorization hints and row-bias `memcpy` initialization for the helper kernel.
- Added `AOTRunIntoT1`, `AOTRunIntoT16`, `TrainCPUAOTT1`, and `TrainCPUAOTT16` labels.
- Added a compiled-module correctness test that forces the new parallel path and compares against the interpreter.
- Refreshed CUDA profile notes around native execution, graph replay, and the small-workload fixed floor.

## Next Optimization Direction

1. Move CPU intra-op from the sidecar C++ helper into the optimized MLIR/LLVM lowering path, or delegate larger static
   MatMul/Linear workloads to a production CPU GEMM backend such as BLAS/oneDNN.
2. Preserve the single-thread MLIR packed/zmm path as the fallback; do not route `LITENN_CPU_AOT_THREADS=1` through
   sidecar kernels.
3. Keep CUDA Graph replay as the recommended fast path for pointer-stable static-shape inference, and investigate
   graph-node parameter update support before making it default.
4. Extend `litenn_profile` so it can report instruction stats and CUDA launch breakdowns directly after object/payload
   emission.
