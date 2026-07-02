# LiteNN Qwen CPU Decode Performance Analysis - 2026-07-02

This report follows `PerformanceAnalysis.md`, `PerformanceAnalysis_2026-05-16.md`, and
`PerformanceAnalysis_2026-05-19.md`, but focuses on the current GGUF/Qwen2.5 stateful CPU AOT decode path.

## Scope

The user reported that the same Qwen2.5-Coder-14B Q4_K_M model runs at about `0.1-0.2 s/step` in llama.cpp with
FlashAttention and CUDA-style accelerators disabled. The current LiteNN CPU AOT stateful decode path is slower even
after active-prefix attention and dynamic RoPE helper lowering. This report records the local evidence gathered in this
round and explains why "some LiteNN low-level kernels are fast" does not yet imply full LLM decode parity.

No code changes were made for this analysis.

## Evidence

### End-to-End Decode Runs

The prompt was the Qwen chat-template form of `hello`, producing nine prompt tokens and one generated token. The output
token remained `9707` / `Hello` in all successful runs.

| Run | Threads | Max cache length | Build ms | Run ms | Generation step ms | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `qwen_smoke.py`, stateful CPU AOT | default hardware threads | 2048 | 22212.6 | 6218.8 | 722.1 | Successful end-to-end smoke |
| direct token-id decode | default hardware threads | 10 | 25379.8 | 4489.9 | 521.5 | Performance data completed; process then failed only because the output directory did not exist |
| direct token-id decode | 16 | 10 | 25256.6 | 7319.0 | 815.4 | Successful; slower than default hardware-thread policy |

The capacity comparison is important: active-prefix attention removed the earlier full-capacity scan as the dominant
problem, but `max_cache_length=2048` is still about `200 ms` slower per generated token than `max_cache_length=10`.
The larger remaining gap is capacity-independent: even at `max_cache_length=10`, the generation step is about `522 ms`.

### Compile/Artifact Shape

Both `max_cache_length=10` and `2048` now produce the same CPU AOT instruction surface after the latest active-prefix
and RoPE helper work:

- External weights: about `8.98 GB`, `579` tensors.
- Lowered LLVM after entry wrapper: `51` functions, `9696` blocks, about `389k` instructions.
- Object file: about `2.17 MB`.
- Stateful schedule: `49` states, `194` bindings, `98` inputs, `98` functional outputs, one public logits output.

This is no longer the earlier multi-million-instruction/full-capacity IR explosion. The current runtime gap is not an
Interpreter fallback and not primarily a cold-compile issue.

### Isolated GGML Block MatMul Helper

Current LiteNN benchmark rows for a decode-shaped `batch=1, in=4096, out=4096` helper:

| Helper | T1 | T16 |
| --- | ---: | ---: |
| `GGMLBlockMatMulHelper/Q8_0` | 9.65 ms | 0.887 ms |
| `GGMLBlockMatMulHelper/Q4_K` | 9.52 ms | 0.938 ms |
| `GGMLBlockMatMulHelper/Q6_K` | 10.6 ms | 1.21 ms |

These rows explain both the progress and the remaining gap. The helper is now fast enough to make full-model smoke
possible, but a Qwen2.5 decode step contains hundreds of quantized projection calls plus normalization, RoPE, attention,
state writes, the final full-vocabulary projection, and sampling/logit handling. A `~1 ms` helper is not sufficient for
`0.1-0.2 s/step` once multiplied across the whole decoder.

## Difference From llama.cpp

The local `third_party/llama.cpp` source shows a concrete kernel-organization difference, not just a generic "needs
optimization" gap.

### Activation Quantization and Vec-Dot Kernels

For Q4_K/Q6_K CPU matmul, llama.cpp uses the type table:

- `GGML_TYPE_Q4_K` -> `ggml_vec_dot_q4_K_q8_K`
- `GGML_TYPE_Q6_K` -> `ggml_vec_dot_q6_K_q8_K`
- `vec_dot_type` is `GGML_TYPE_Q8_K`

When the activation input is Float32, `ggml_compute_forward_mul_mat` first converts it into the `vec_dot_type` work
buffer, then runs the quantized `vec_dot` kernel. The work is chunked in 16-by-16 tiles and distributed through the
ggml threadpool.

LiteNN's current `litenn_cpu_ggml_block_matmul_f32` is different: it walks output elements/columns, reads each GGML
block, and computes `DotGGMLBlockF32(block, lhs)` directly against Float32 activations. The Q4_K/Q5_K/Q6_K paths parse
scale/min/quant fields per output column and accumulate Float32 products. This is correctness-oriented and portable,
but it misses llama.cpp's Q8_K activation staging and architecture-specific vector-dot/repacked multi-column kernels.

This is the strongest current explanation for why isolated LiteNN helper rows do not translate into llama.cpp-level
full-step latency.

### Thread and Chunk Policy

The default LiteNN path already allows hardware-thread execution. For this workload, forcing `LITENN_CPU_AOT_THREADS=16`
made the `max_cache_length=10` generation step worse (`521.5 ms` -> `815.4 ms`). Therefore the immediate issue is not
"threads are off". The issue is thread granularity and kernel organization: LiteNN currently parallelizes the helper
over output elements, while llama.cpp uses graph-wide task scheduling plus matmul chunking that is aware of rows,
columns, and the `vec_dot` kernel shape.

### Capacity-Dependent Overhead Remains

The `max_cache_length=10` vs `2048` comparison leaves about `200 ms` attributable to remaining capacity-shaped state or
attention/cache work. Active-prefix attention fixed the worst full-suffix scanning behavior, but the runtime ABI still
uses dense full-capacity KV tensors and 98 explicit state bindings/outputs. This matters for 2K context and becomes a
hard blocker for the 1M-context target.

## Conclusions

1. The old reports' "LiteNN low-level kernels are fast" mostly covered static Float32 Linear/MLP AOT kernels and some
   standalone helper microbenchmarks. It does not prove that the current GGUF quantized full-decoder step is faster than
   llama.cpp.
2. The current full-step gap is evidence-backed:
   - `~522 ms` remains even at tiny cache capacity.
   - `~200 ms` additional overhead appears when raising cache capacity from 10 to 2048.
   - T16 is slower than the default hardware-thread policy, so a simple thread-count setting is not the fix.
3. The main confirmed kernel gap is the quantized projection implementation: LiteNN computes Q4_K/Q6_K blocks against
   Float32 activations, while llama.cpp stages activations into Q8_K and calls optimized Q4_K/Q6_K x Q8_K vec-dot
   kernels with tiled scheduling and architecture-specific repacked variants.
4. The current profiling surface is still too coarse to rank every per-layer cost. The next implementation work should
   first add per-node/per-helper timing for the stateful decode schedule, then replace the confirmed quantized matmul
   kernel mismatch and remeasure.

## Roadmap Implications

The next high-ROI work should be concrete and evidence-driven:

1. Add GGUF decode operator-level profiling around the compiled stateful runtime schedule and sidecar helper calls.
2. Replace `litenn_cpu_ggml_block_matmul_f32` with a Q8_K activation staging + Q4_K/Q5_K/Q6_K/Q8_0 vec-dot kernel
   family, with architecture-specific packed/repacked variants where available.
3. Add a thread/cost model for decode-shaped quantized projections; default hardware threads are better than T16 here,
   but per-projection scheduling still needs measured gates.
4. Continue reducing capacity-shaped overhead through paged KV/cache descriptors, because `2048` still costs about
   `200 ms` more than `10` after active-prefix attention.
5. Only after per-op timing is available should grouped QKV projection, MLP gate/up fusion, or logits/sampler fusion be
   prioritized; today those are plausible but not yet ranked by direct evidence.
