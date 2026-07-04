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

Priority classes for the GGUF/Qwen decode work:

- P0, step-latency reducers: optimize the capacity-independent generated-token path. This includes production-shaped
  GGML helper benchmarks, activation-side reuse/staging, Q8_K activation-staged vec-dot kernels, shape-aware
  thread/grain policy, grouped QKV projection, and grouped SwiGLU gate/up projection.
- P1, long-context blockers: remove capacity-shaped work that prevents 32K/128K/1M contexts. This includes verified
  in-place KV append, paged KV-cache state, grouped active-prefix attention, and long-context attention plans.
- P2, observability and validation gates: add per-helper/per-node timing, long-context benchmark rows, and golden
  validation. These do not directly speed up execution, but they keep P0/P1 changes measurable and regression-safe.

- [ ] Add a whole-process profile bundle command that combines existing `litenn_profile` evidence with waterfall
  timeline output and optional platform sampling.
  - [x] First slice: `benchmark/profile_bundle.py` wraps `litenn_profile` or an arbitrary command, captures
    stdout/stderr, writes `manifest.json`, `trace.json`, and `summary.md`, and redacts user-specified sensitive paths
    from recorded artifacts.
  - [x] GGUF decode parser slice: the bundle now converts `--stream-stats` and helper diagnostics into
    `gguf_decode_summary.json`, `gguf_decode_summary.md`, and `gguf_decode_trace.json` for token-step and helper
    attribution.
  - [x] Existing-run import slice: `benchmark/profile_bundle.py --qwen-smoke-report <qwen_smoke_report.json>` can
    rebuild the same decode helper/step attribution from a completed Qwen smoke directory, linking the original
    `qwen_smoke_trace.json` and waterfall into the bundle manifest so large-model evidence can be re-summarized without
    re-running the model.
  - [x] Reproducible A/B metadata slice: Qwen smoke reports now record CPU AOT opt level, helper thread count, affinity,
    parallelism gate, Q8_K-staged mode, and compile-diagnostics mode; `benchmark/gguf_decode_compare.py` includes the
    compact configuration string in comparison tables.
  - [x] Profile-summary comparison slice: `benchmark/gguf_decode_compare.py --litenn-profile-summary
    <gguf_decode_summary.json>` can compare already-bundled decode runs directly and carries top-helper/helper-share
    columns into Markdown/CSV/JSON outputs.
  - [ ] Timeline output: fine-grained Chrome Trace / Perfetto JSON for import, conversion, lowering, MLIR/LLVM compile,
    object load, runtime schedule, transfers, synchronization, GPU dispatches, and decode-loop token phases.
    - [x] Qwen smoke slice: direct GGUF/Qwen smoke logs are streamed to disk during execution, the wrapper emits
          `qwen_smoke_trace.json` and `qwen_smoke_waterfall.md`, and the GGUF AOT path reports separated-cache
          population, artifact separation, cache read, and JIT/load timing explicitly.
    - [x] Cache-population visibility/fix: GGUF decode AOT cache writes now stream large regions in chunks with progress
          diagnostics and build separated metadata directly from the compiled artifact, avoiding an extra multi-GB
          `SeparateRodata()` weight copy during cache population. Borrowing/mapping original GGUF weight regions remains
          the G16.7 production target.
    - [x] Runtime memory pressure slice: GGUF decode now releases the imported archive before token execution, and
          separated-artifact cache hits can move owned constants/weights into the CPU module instead of copying the
          multi-GB weights region during load.
    - [x] Qwen decode smoke diagnosis: on 2026-07-01, `max_cache_length=10` measured about `0.54-0.62 s` per
          prompt/generation step, while the user's `max_cache_length=2049` run measured about `6 s` per generated step.
          The current capacity-decode graph masks inactive cache suffixes but still scans the full static capacity each
          step.
    - [x] Replace the immediate full-capacity decode attention hot path with active-prefix execution. On 2026-07-02,
          `ActivePrefixAttentionNode` gained CPU AOT lowering plus a rank-3 KV-cache path that reads
          `[capacity, kvHeads, headDim]` directly instead of materializing per-head `Slice+Reshape` tensors. A real
          Qwen2.5-Coder-14B Q4_K_M `--stateful --max-tokens 1 --max-cache-length 2048` smoke improved from about
          `4.3-5.0 s/step` to about `0.62-0.72 s/step`, and the lowered LLVM instruction count dropped from about
          `1.67M` to about `0.52M`.
    - [x] Add a head-aware RoPE helper/node for decode. Correct query RoPE must use head-local dimensions; applying RoPE
          to the full concatenated query vector is numerically wrong for Qwen/LLaMA. On 2026-07-02, runtime-position
          `RoPENode` gained a CPU AOT helper rewrite (`litenn_cpu_rope_at_positions_f32`) so per-head decode RoPE stays
          numerically correct without expanding into Reshape/Slice/Cos/Sin/Concat IR. A real
          Qwen2.5-Coder-14B Q4_K_M `--stateful --max-tokens 1 --max-cache-length 2048 --no-aot-cache-write` smoke
          completed in about 57s wall time, compiled the stateful decode artifact in about 22.1s, produced about 389k
          LLVM instructions after the entry wrapper, ran prompt/generation steps in about `0.63-0.72 s/step`, and
          generated token `9707` / `Hello`.
    - [x] Profile the remaining Qwen CPU AOT decode gap against a user's llama.cpp CPU reference run with GPU-style
          accelerators disabled. The 2026-07-02 analysis in `docs/PerformanceAnalysis_2026-07-02.md` shows that
          `max_cache_length=10` still takes about `522 ms` for the generated token, `max_cache_length=2048` adds about
          `200 ms`, and forcing `LITENN_CPU_AOT_THREADS=16` regresses to about `815 ms`. This rules out "threading is
          simply off" and "full-capacity attention is still the only bottleneck" as sufficient explanations.
    - [ ] P2: Add operator-level and helper-level timing for stateful GGUF CPU AOT decode.
          Required output: per-layer/per-node duration, helper symbol name, GGML block format, input/output shape,
          thread count, cache length, and generated-token phase. The current per-step waterfall is enough to find the
          broad gap but not enough to rank RMSNorm, quantized projections, active-prefix attention, state copies,
          logits projection, and sampler work.
          - [x] Add explicit CPU AOT helper profiling scopes and GGUF decode diagnostics output for helper symbol,
                shape/format/thread detail, call count, total time, and average time per decode step. This covers
                sidecar/helper attribution for quantized projections, get-rows, RoPE, KV scatter, and active-prefix
                attention; per-layer/per-node attribution remains open.
          - [x] Bundle helper shares: `gguf_decode_summary.json` and `.md` now report helper time as a percentage of
                total step time and per-step helper totals/top helper, making it clearer whether the next bottleneck is
                quantized projection, attention, KV state movement, or non-helper residual work.
    - [x] P0: Add production-shaped GGML helper benchmark rows for the real Qwen decode dimensions:
          `5120->5120`, `5120->1024`, `5120->13824`, `13824->5120`, and `5120->152064`.
          The current `4096->4096` row is useful but under-specifies the 337-projection full-step workload.
          Completed on 2026-07-02: `litenn_bench` now registers named `baseline4096`, `qwen_hidden`, `qwen_kv`,
          `qwen_ffn_up`, `qwen_ffn_down`, and `qwen_logits` rows for every supported GGML helper format and `T1/T16`.
    - [x] P0: Cache Q4_K/Q5_K activation subblock sums inside `litenn_cpu_ggml_block_matmul_f32`.
          This removes repeated `lhsSum` accumulation for every output column while preserving the direct Float32
          reference-style dot path. Validation slice: `GGMLBlockMatMulHelper/Q4_K/qwen_kv/T1` measured about `3.03 ms`
          before the slice and about `1.91 ms` after the split hot-path implementation; `Q4_K/baseline4096/T1` measured
          about `6.09 ms` after the change.
    - [x] P0: Tile the Q4_K direct CPU helper across four output columns.
          `litenn_cpu_ggml_block_matmul_f32` now shares each activation-block scan across a four-column Q4_K tile while
          keeping the existing Float32 accumulation semantics. Short validation on 2026-07-02 passed the stateful GGUF
          logits parity test and measured `GGMLBlockMatMulHelper/Q4_K/qwen_kv/T1` at about `1.73 ms` CPU and
          `Q4_K/baseline4096/T1` at about `5.79 ms` CPU; the same `qwen_kv` helper row measured about `0.33 ms` CPU at
          `T16`, so the remaining thread-model work is full-decode scheduling and grain selection rather than a blanket
          rejection of helper-level parallelism.
    - [x] P0: Tile the Q6_K direct CPU helper across four output columns.
          This extends activation-scan reuse to FFN-down-style Q6_K projections while preserving the current Float32
          accumulation semantics. Short validation on 2026-07-02 passed the stateful GGUF logits parity test and measured
          `GGMLBlockMatMulHelper/Q6_K/qwen_ffn_down/T1` at about `49.9 ms` CPU and the matching `T16` row at about
          `4.58 ms` CPU. This remains a major single-thread hotspot until Q8_K activation-staged vec-dot kernels land.
    - [x] P0: Tile the Q8_0 and Q5_K direct CPU helper paths across four output columns.
          The grouped-output helper now covers all currently supported GGML direct MatMul formats. Validation on
          2026-07-02 passed `GGUFLLaMAQuantizedExecution.*`; a short helper run measured `Q8_0/qwen_kv/T1` at about
          `2.03 ms` CPU and `Q5_K/qwen_kv/T1` at about `3.67 ms` CPU.
    - [x] P0: Implement and evaluate a llama.cpp-style activation-staged GGML block MatMul kernel family. The current
          production helper still computes GGML_Q4_K/Q5_K/Q6_K/Q8_0 blocks directly against Float32 activations unless a
          format-specific staged route is explicitly enabled. llama.cpp stages Float32 activations into Q8_K work buffers
          and calls Q4_K/Q6_K x Q8_K vec-dot kernels with tiled scheduling and architecture-specific packed/repacked
          variants, so the staged family remains a targeted optimization track rather than the global default.
          - [x] Add a scalar Q8_K-staged helper prototype and benchmark rows without switching the default AOT helper.
                Validation on 2026-07-03 passed exact-activation parity for Q4_K/Q5_K/Q6_K. Short helper measurements
                showed the scalar staged path is not a default-switch candidate yet: Q4_K `qwen_kv/T1` was slower
                (`~2.78 ms` CPU staged vs `~2.60 ms` CPU direct in that run), while Q6_K `qwen_ffn_down/T1` only improved
                modestly (`~66.4 ms` CPU staged vs `~70.3 ms` CPU direct) and carries activation-quantization deltas.
          - [x] Add a guarded AVX2 16-lane dot primitive for the Q8_K-staged Q4_K/Q5_K/Q6_K path and keep it behind
                runtime CPU feature detection. Validation on 2026-07-03 passed
                `GGUFLLaMAQuantizedExecution.Q8KStagedHelperMatchesDirectHelperForExactActivationRows`; short helper
                measurements showed the AVX2 staged path helps Q6_K single-thread (`qwen_ffn_down/T1` about `41.7 ms`
                CPU staged vs `44.3 ms` CPU direct in that run) but Q4_K/Q5_K remained slower than the direct helper.
                Do not switch the default globally until the policy is format-specific and accuracy-aware.
          - [x] Add an explicit CPU AOT opt-in route from GGML_Q6_K matmul placeholders to the Q8_K-staged sidecar,
                while keeping the default path on the numerically stricter direct helper. Q4_K/Q5_K/Q8_0 remain direct
                until they have measured production-shape wins. The GGUF decode CLI exposes this as
                `--cpu-aot-q8k-staged-matmul` for A/B profiling.
          - [x] Expose GGUF decode CPU AOT tuning as CLI flags for reproducible A/B runs:
                `--cpu-aot-threads`, `--cpu-aot-affinity`, `--cpu-aot-llvm-opt-level`,
                `--cpu-aot-parallel-min-flops`, and `--compile-diagnostics` / `--no-compile-diagnostics`.
                Artifact-affecting options are part of the decode cache key.
          - [x] Add VNNI, repacked-weight, or other architecture-specific vec-dot kernels for the Q8_K-staged path, then
                re-run the direct-vs-staged helper table before changing the compiler/runtime default. The current AVX2
                implementation uses a u8*s8 `maddubs` pairwise dot for Q4_K/Q5_K/Q6_K staged lanes. Validation passed
                `GGUFLLaMAQuantizedExecution.Q8KStagedHelperMatchesDirectHelperForExactActivationRows`; a short
                2026-07-04 helper run still rejected a global default switch (`Q4_K/qwen_kv/T1` direct `~3.12 ms` CPU vs
                staged `~15.6 ms` CPU, `Q6_K/qwen_ffn_down/T1` direct/staged both about `46.9 ms` CPU but staged worse
                in real time). Keep staged routing explicit/format-gated until a later packed-weight or VNNI kernel wins.
    - [x] P0: Add a measured thread/grain policy for decode-shaped quantized projections.
          `requestedThreadCount == 0` now takes an auto policy instead of blindly using every hardware thread: it caps
          GGML block MatMul helpers at 16 workers, applies smaller caps to tiny output-group counts, and keeps explicit
          `T1/T2/T4/T8/T16/T32` requests unchanged. The helper benchmark matrix now includes `T0/T1/T2/T4/T8/T16/T32`
          rows. A short 2026-07-03 run showed `T0` tracks the conservative cap (`Q4_K/qwen_kv` about `0.30 ms` real,
          `Q6_K/qwen_ffn_down` about `3.28 ms` real) while explicit `T32` remains available for isolated helper cases
          where it wins.
    - [ ] P1: Stop using monolithic max-cache-length-shaped CPU AOT decode artifacts as the default long-context path.
          - [x] Make the GGUF decode-loop CLI default to the stateful runtime-schedule path. Completed on 2026-07-04:
                `--run-llama-*-decode-loop` now builds/loads the logits-only public-output stateful schedule unless
                `--functional` is passed explicitly for compatibility or diagnostics.
          - [ ] Replace the remaining max-cache-length-shaped stateful function signature with per-layer/per-block
                reusable decode artifacts or page-table state bindings.
    - [ ] P1: Replace dense full-capacity KV state with paged-KV execution. Active-prefix attention removes inactive suffix
          scans from attention, but the 1M-context target still requires page tables, active-length metadata, and
          capacity-independent artifact shapes.
          - [x] Add the paged-KV runtime-state ABI/manifest/planner contract. Completed on 2026-07-04:
                `RuntimeStateBinding` can carry `PagedKVCache` layout metadata, vNext packages round-trip it, and dynamic
                GGUF decode planning marks KV cache states with page size, logical capacity, resident page count,
                plane offsets, and token/page strides. This is the contract step only; the current CPU decode graph still
                uses the dense capacity-shaped fallback signature until paged lowering lands.
          - [ ] Replace the dense fallback decode signature with page-table/page-descriptor state bindings so cache-hit
                artifacts stop scaling with max context length.
    - [ ] P1: Decouple persistent AOT instruction cache from model-weight storage.
          - [x] Deduplicate GGUF decode AOT cache weights across instruction-cache variants for the same source model.
                Completed on 2026-07-04: cache entries now write `weights.path.txt` pointing to a model-level shared
                weight blob under the cache root, while metadata/constants/instructions remain per artifact. Legacy
                per-cache `weights.bin` entries still load. This removes repeated multi-GB weight writes when tuning AOT
                flags or thread policy.
          - [ ] Replace the shared copied blob with direct mapped/borrowed GGUF/package regions after separated metadata
                can encode source file offsets and stable source checksums.
    - [x] P1: Add a verified in-place KV append sidecar before the full paged-KV migration.
          `litenn_cpu_scatter_update_axis0_f32_rank3` now has direct regression coverage for both same-buffer in-place
          append and distinct-output copy semantics, and stateful decode schedule coverage confirms projected cache
          outputs alias their input buffers. The `KVScatterUpdateHelper` benchmark records the cost boundary: on
          2026-07-03, Qwen-shaped alias append rounded to `0.000 ms`, while copy mode measured about `0.210 ms` for a
          2048-token cache and about `1.71 ms` real / `1.41 ms` CPU for an 8192-token cache.
    - [x] P0/P1: Add grouped LLM decode helpers after operator timing is available: fused/concatenated QKV projection,
          fused gate/up projection for SwiGLU, and grouped active-prefix attention per KV head. Projection grouping is
          P0; attention grouping is P1 because it scales with active context length.
          - [x] Add grouped-projection benchmark rows that compare separate Q/K/V or gate/up helper calls against
                concatenated output-major GGML weights using the existing helper ABI. A short 2026-07-03 Q4_K run
                validated `max_abs_delta=0`; `qwen_qkv/T0` improved from about `1.37 ms` real separate to `1.07 ms`
                concatenated, while `qwen_gate_up/T0` improved from about `3.49 ms` to `3.17 ms`.
          - [x] Add AOT lowering that recognizes same-input compatible projection groups and emits a concatenated
                helper call or a multi-output sidecar without copying model weights at runtime. Completed on
                2026-07-04 with `GroupedQuantizedMatMulNode`, Q/K/V and gate/up layer helpers, executable-plan
                round-trip support, CPU MLIR lowering to `litenn_cpu_ggml_block_grouped_matmul2_f32` /
                `litenn_cpu_ggml_block_grouped_matmul3_f32`, and a projection-span sidecar that accepts independent
                output-major GGML rhs memrefs. Validation passed the grouped Q4_K AOT regression, the quantized
                projection storage preservation regression, and `GGUFLLaMAQuantizedExecution.*`.
          - [x] Add active-prefix attention helper benchmark rows for Qwen-shaped rank-3 KV caches. A short 2026-07-03
                run measured one KV-head helper call at about `0.022 ms` for 128 active rows, `0.470 ms` for 2048 rows,
                and `2.53 ms` for 8192 rows, which makes grouped KV-head/online-softmax work measurable before adding a
                new kernel.
          - [x] Cache per-row attention scores inside the CPU active-prefix helper so max, denominator, and value
                aggregation no longer recompute the query-key dot product. Validation on 2026-07-03 passed the CPU AOT
                decode parity tests; helper timing improved to about `0.006 ms` for 128 rows, `0.099 ms` for 2048 rows,
                and `0.651 ms` for 8192 rows.
          - [x] Add a grouped active-prefix attention CPU sidecar ABI and benchmark rows that compare Qwen-shaped GQA
                grouped execution against repeated per-query-head rank-3 helper calls. A short 2026-07-03 run validated
                `max_abs_delta=0`; 128 active rows stayed roughly neutral (`0.397 ms` grouped vs `0.410 ms` repeated),
                while 2048 active rows improved from about `9.30 ms` repeated to `8.31 ms` grouped.
          - [x] Route GGUF capacity decode graphs through `GroupedActivePrefixAttentionNode` and lower that node to the
                grouped CPU AOT sidecar. The builder still applies RoPE per query/KV head before grouping, and the
                grouped helper remains a conservative CPU sidecar rather than a final KV-head-tiled attention kernel.
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
