# LiteNN Performance Benchmark Snapshot - 2026-06-18

This snapshot records the current inference benchmark state before the next production-refactor implementation pass. It
is a direction-finding run, not a publication-grade benchmark suite: each LiteNN row used `--benchmark_min_time=0.02s`
to keep the full backend matrix practical during iteration.

## Environment

- Build: `build-release`, target `litenn_bench`.
- Host: Google Benchmark reported `32 X 4300 MHz CPU`, L3 `32768 KiB (x2)`.
- CUDA device observed by PyTorch: `NVIDIA GeForce RTX 4090`.
- PyTorch: `2.9.1+cu128`.
- PyTorch CPU default: `16/16` threads.
- PyTorch CPU single-thread comparison: `--threads 1`.

## Commands

```powershell
cmd /c cmake --build build-release --target litenn_bench --parallel

.\build-release\benchmark\litenn_bench.exe `
  --benchmark_filter='^(Interpreter|LlamaCppGGMLT1|LlamaCppGGMLT16|AOTRun|AOTRunInto|EGraphAOTRunInto|AOTRunIntoT1|AOTRunIntoT16|CUDACPUFallbackRunInto|CUDANativeRunInto|CUDANativeGraphRunInto|VulkanNativeRunInto|VulkanNativeGraphRunInto|VulkanNativeGraphDeviceLocalRunInto|VulkanNativeManualPipeline)/(Linear|MLP).*' `
  --benchmark_min_time=0.02s `
  --benchmark_repetitions=1 `
  --benchmark_out=benchmark\results\litenn_standard_backends_2026-06-18.json `
  --benchmark_out_format=json

python311 benchmark\bench.py --device all
python311 benchmark\bench.py --device cpu --threads 1

python311 benchmark\compare_backends.py `
  --litenn-json benchmark\results\litenn_standard_backends_2026-06-18.json `
  --pytorch-text benchmark\results\pytorch_all_default_2026-06-18.txt `
  --pytorch-cpu1-text benchmark\results\pytorch_cpu_threads1_2026-06-18.txt `
  --out-md benchmark\results\backend_comparison_2026-06-18.md `
  --out-csv benchmark\results\backend_comparison_2026-06-18.csv
```

Raw benchmark outputs are under `benchmark/results`, which is ignored by git. The important summary is copied below.

## Headline Table

Values are `ms/batch`; lower is better.

| Model | Batch | PyTorch CPU | PyTorch CUDA | LiteNN CPU AOT | LiteNN CUDA Graph | LiteNN Vulkan DeviceLocal | Best |
|---|---:|---:|---:|---:|---:|---:|---|
| Linear(784->10) | 1 | 0.008 | 0.025 | 0.001 | 0.038 |  | AOTRunIntoT16 0.001 |
| Linear(784->10) | 32 | 0.019 | 0.027 | 0.004 | 0.036 |  | AOTRunIntoT1 0.004 |
| Linear(784->10) | 128 | 0.036 | 0.025 | 0.015 | 0.031 |  | AOTRunIntoT16 0.015 |
| Linear(784->10) | 512 | 0.114 | 0.028 | 0.062 | 0.060 |  | PyTorch CUDA 0.028 |
| MLP(784->128->10) | 1 | 0.025 | 0.066 | 0.003 | 0.052 | 0.085 | EGraphAOTRunInto 0.003 |
| MLP(784->128->10) | 32 | 0.046 | 0.065 | 0.025 | 0.094 | 0.105 | AOTRunIntoT1 0.025 |
| MLP(784->128->10) | 128 | 0.085 | 0.062 | 0.098 | 0.090 | 0.154 | PyTorch CUDA 0.062 |
| MLP(784->128->10) | 512 | 0.237 | 0.060 | 0.391 | 0.081 | 0.172 | PyTorch CUDA 0.060 |
| MLP(784->512->256->10) | 1 | 0.080 | 0.092 | 0.018 | 0.098 | 0.117 | LiteNN CPU AOT Run 0.017 |
| MLP(784->512->256->10) | 32 | 0.133 | 0.101 | 0.135 | 0.249 | 0.167 | PyTorch CUDA 0.101 |
| MLP(784->512->256->10) | 128 | 0.309 | 0.097 | 0.490 | 0.182 | 0.191 | PyTorch CUDA 0.097 |
| MLP(784->512->256->10) | 512 | 0.906 | 0.097 | 1.524 | 0.106 | 0.365 | PyTorch CUDA 0.097 |

## Key Findings

1. CPU AOT is already a strong single-thread baseline for these small dense graphs.
   - Against PyTorch CPU single-thread, LiteNN CPU AOT is faster across all measured rows: about `0.14x` to `0.41x`
     on small/medium MLP and about `0.33x` on the largest batch-512 MLP.
   - Against PyTorch default CPU, LiteNN remains faster for Linear and small MLPs, but loses on larger batched MLP:
     `MLP(784->128->10)/batch:512` is `1.65x` PyTorch CPU, and
     `MLP(784->512->256->10)/batch:512` is `1.68x` PyTorch CPU.
   - This points at CPU kernel strategy and intra-op policy, not graph/runtime overhead, as the next CPU bottleneck.

2. CUDA Graph replay is the only competitive CUDA path for model execution.
   - Non-graph CUDA native is launch/dispatch dominated on multi-layer models. For
     `MLP(784->512->256->10)/batch:512`, CUDA Graph is `0.106 ms` while CUDA Native is `1.24 ms`.
   - CUDA Graph is close to PyTorch CUDA on the largest dense MLP: `0.106 ms` vs `0.097 ms` (`1.09x`).
   - Smaller GPU rows are still overhead-bound: Linear batch 1 to 512 is `1.22x` to `2.16x` PyTorch CUDA.

3. Vulkan DeviceLocal helps, but desktop Vulkan is still behind CUDA Graph.
   - DeviceLocal is consistently better than host-visible graph buffers in the MLP rows where both are present.
   - The best large MLP Vulkan row is `0.365 ms` for batch 512, about `3.76x` PyTorch CUDA and `3.44x` LiteNN CUDA
     Graph.
   - This keeps Vulkan important for mobile/portability, but the current bottleneck is still command/memory planning
     and MatMul kernel quality rather than frontend overhead.

4. Interpreter is correctly positioned as a validation and constant-evaluation path.
   - Interpreter is roughly `12x` to `53x` slower than CPU AOT in the measured rows.
   - It should not be part of production inference performance goals except as a correctness oracle.

5. ggml is not the main baseline for these small MLPs.
   - `GGML 1T` is close on tiny Linear batch 1, but LiteNN AOT is generally faster for the measured MNIST-sized dense
     graphs.
   - `GGML 16T` is dominated by thread overhead here. It should be compared again on LLaMA-like larger layers, not used
     to judge this small-model benchmark.

## Next Performance Decisions

- CPU: decide whether production CPU should use an external BLAS/oneDNN-like backend, or whether LiteNN should invest in
  a maintained packed-GEMM/Conv kernel family. The current AOT path is good enough for control overhead, but not enough
  to beat PyTorch default CPU on larger dense batches.
- CUDA: make graph replay the production CUDA model path. Keep non-graph native rows as diagnostics and fallback, not as
  the expected fast path.
- Vulkan: prioritize whole-graph scheduling, device-local memory planning, and MatMul kernel work before broader op
  breadth claims.
- Benchmarking: keep `python311` for PyTorch comparison, keep the benchmark output table as a required artifact for
  future backend changes, and add larger LLaMA-like GEMM/attention rows before drawing conclusions about ggml parity.
