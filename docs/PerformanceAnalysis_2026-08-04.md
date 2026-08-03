# LiteNN Qwen CPU Decode Stage Analysis - 2026-08-04

This report extends the July Qwen CPU decode analyses with a same-host, stage-level comparison against a CPU-only
llama.cpp control. It focuses on steady state stateful decode after the field-interleaved-v4 prepared-weight path,
Q8_K activation staging, grouped projections, and shape-aware thread policy landed.

The private local GGUF model path is intentionally omitted. No project source changes were made while collecting this
evidence. The one-off llama.cpp callback profiler and raw logs live under the ignored
`build/qwen_profile_20260804` directory.

## Configuration

The compared model was Qwen2.5-Coder-14B-Instruct Q4_K_M. Both paths ran CPU-only with eight worker threads and no
GPU offload or Flash Attention.

LiteNN configuration:

- stateful CPU AOT decode, no Interpreter fallback;
- LLVM optimization level 0;
- `field-interleaved-v4`, prepared-weight policy `all`;
- adaptive worker wait policy and no explicit affinity policy;
- nine prompt tokens, eight prompt-replay steps, and 32 generated tokens;
- separated AOT artifact and weights loaded from an existing cache.

llama.cpp control configuration:

- `n_gpu_layers=0`, memory mapping enabled, Flash Attention disabled;
- `n_batch=1`, `n_ubatch=1`, eight compute and batch threads;
- eight warm-up decode calls followed by 32 recorded calls;
- the stage profiler used the public graph-evaluation callback to insert boundaries after selected tensors.

The llama.cpp stage profiler repeatedly decoded one valid token rather than sampling the LiteNN token sequence. This
keeps the short-context projection shapes equivalent, but it is not a logits or generated-text parity test. Existing
golden tests remain the correctness gate.

## End-to-End Baseline

Adjacent no-profile runs on the same host produced:

| Runtime | Latency | Throughput | Relative latency | Relative throughput |
| --- | ---: | ---: | ---: | ---: |
| LiteNN CPU AOT | `256.616 ms/token` | `3.89687 token/s` | `1.269x` | `0.788x` |
| llama.cpp CPU | `202.224 ms/token` | `4.94502 token/s` | `1.000x` | `1.000x` |

LiteNN is therefore about `26.9%` slower by latency and `21.2%` lower by throughput in this sample. Host frequency and
cache state caused visible run-to-run variation, so future acceptance must use alternating repeated runs rather than a
single absolute number.

## Profiling Method

LiteNN helper profiling measured `281.703 ms/token` while its adjacent no-profile run measured
`256.616 ms/token`. The llama.cpp FFN-boundary profile measured `222.810 ms/token` while its adjacent baseline measured
`202.224 ms/token`. The profilers therefore introduced about `9-10%` overhead.

To compare stages, each runtime's stage values below are uniformly scaled by its own
`no-profile latency / profiled latency` ratio:

- LiteNN scale: `0.910945`;
- llama.cpp scale: `0.907607`.

These normalized values are attribution estimates, not hardware-counter measurements. LiteNN rows marked as a helper
lower bound exclude normalization, residual, and generated-code self time. llama.cpp block rows include work between
the selected graph boundaries.

## Corresponding Stage Comparison

| Stage | LiteNN normalized | llama.cpp normalized | LiteNN gap | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Complete FFN evidence | `168.48 ms` helper lower bound | `121.94 ms` full block | at least `+46.54 ms` | Dominant gap |
| Gate + Up | `81.52 ms` grouped helper | `76.55 ms` prefix, gate, and up | at least `+4.97 ms` | Near parity relative to Down |
| Q4_K activation + Down | `39.19 ms` | `19.17 ms` | `+20.03 ms`, `2.04x` | Major shape-specific gap |
| Q6_K activation + Down | `47.77 ms` | `26.23 ms` | `+21.54 ms`, `1.82x` | Major shape-specific gap |
| Attention | `63.83 ms` helper lower bound | `69.74 ms` full block | inconclusive / near parity | Not the immediate owner |
| Final logits | `11.35 ms` helper | `10.25 ms` full boundary | `+1.10 ms` | Secondary |

The normalized FFN difference explains about `87%` of the `53.39 ms/token` total baseline gap. Q4_K and Q6_K
activation-plus-Down account for about `41.57 ms`, or `78%` of the total gap. Optimizing Attention first cannot recover
enough time to reach CPU parity at the tested short context.

### Raw LiteNN Helper Attribution

Generation-phase helper means before normalization were:

| LiteNN helper bucket | Raw time per token |
| --- | ---: |
| Q4_K grouped Gate/Up, 48 calls | `89.49 ms` |
| Q6_K FFN Down, 24 calls | `47.29 ms` |
| Q4_K FFN Down, 24 calls | `37.88 ms` |
| SwiGLU, 48 calls | `10.29 ms` |
| QKV Q6_K + Q4_K | `37.03 ms` |
| Attention output | `30.30 ms` |
| Grouped active-prefix attention | `2.46 ms` |
| RoPE | `0.23 ms` |
| Steady logits, excluding one outlier | `12.46 ms` |

### Raw llama.cpp FFN-Boundary Attribution

The low-boundary profiler inserted 194 selected boundaries per token and measured:

| llama.cpp boundary bucket | Raw time per token |
| --- | ---: |
| Attention block | `76.836 ms` |
| Q4_K FFN activation and Down | `21.118 ms` |
| Q6_K FFN activation and Down | `28.895 ms` |
| FFN prefix and Gate | `44.185 ms` |
| FFN Up | `40.154 ms` |
| Final logits | `11.289 ms` |
| Graph trailing work | `0.182 ms` |

A finer llama.cpp profile with hundreds of boundaries per token added about `25%` overhead. It was useful only for
directional source inspection and is not used in the normalized table.

## Weight-Stream Evidence

The 24 Q4_K Down matrices contain about `0.956 GB` of quantized blocks and the 24 Q6_K Down matrices contain about
`1.393 GB`. Using raw profile times gives the following conservative effective read rates:

| Down weight family | LiteNN projection-only rate | llama.cpp activation-plus-projection rate |
| --- | ---: | ---: |
| Q4_K | about `25.2 GB/s` | at least `45.2 GB/s` |
| Q6_K | about `29.5 GB/s` | at least `48.2 GB/s` |

The llama.cpp denominator also includes the activation work preceding Down, so its pure projection rate is higher than
shown. LiteNN's grouped Gate/Up path reaches roughly `43 GB/s`, while ordinary single projections in the real model
typically sustain only about `23-30 GB/s`. The gap is therefore correlated with the single-projection execution shape,
not merely total model bytes.

## Hot-Cache Cross-Check

The existing helper benchmark repeatedly reads one prepared matrix, so it mainly measures a cache-hot ceiling. With
T8 and the same field-interleaved-v4 helper:

| Format | `5120 -> 13824` Up | `13824 -> 5120` Down |
| --- | ---: | ---: |
| Q4_K | `0.308 ms` | `0.328 ms` |
| Q6_K | `0.631 ms` | `0.623 ms` |

Swapping the matrix direction while keeping weight bytes approximately equal does not reproduce the full-model Down
regression. Q4_K and Q6_K hot rows are effectively direction-neutral. The current microbenchmark can therefore prove
kernel correctness and cache-hot instruction throughput, but it cannot predict the 8.5 GiB streaming decode path.

Thread sweeps also plateau after the low-thread region on cache-hot Down rows. Adding workers alone is not a credible
fix; memory concurrency and dispatch behavior must be measured in a cold-stream workload.

## Source-Level Interpretation

The current LiteNN field-interleaved helper in `CompiledModule.cpp`:

- stages or reuses a Q8_K activation;
- partitions output groups into about four dynamic tasks per resolved worker;
- uses Q4_K AVX2 x8 for the tested Down shape;
- uses the runtime-gated Q6_K AVX-512 x16 kernel where available;
- invokes `LiteNNCPUParallelFor` independently for each projection helper.

llama.cpp combines a persistent graph thread pool with architecture-specific kernels. Its current local build uses
`-O3 -march=native`; Q4_K uses the x86 8x8 repacked GEMV path and Q6_K uses its mature AVX2 Q6_K x Q8_K vector dot.
LiteNN builds the containing translation unit with `-O3` and uses per-function ISA target attributes for its
microkernels. The compile-flag difference remains worth auditing, but the strong cache-hot LiteNN rows make it a lower
priority than the measured cold-stream Down behavior.

The evidence supports the following explanation:

1. LiteNN's quantized arithmetic is competitive when repeatedly reading one matrix.
2. The real model loses memory-level parallelism or pays disproportionate dispatch/barrier cost on ordinary
   `13824 -> 5120` single projections.
3. Grouped Gate/Up amortizes this behavior and is close to llama.cpp, while both Q4_K and Q6_K Down remain roughly
   twice as expensive.
4. The root cause is not yet proven at the hardware-counter level. Cache misses, memory stalls, worker idle time, and
   effective concurrent streams must be measured before committing to a final kernel rewrite.

Windows system-wide sampling was unavailable in this run because the current process could not enable the required
performance-profile policy. Internal timers cover almost all module time, but they do not expose PMU cache or stall
counters.

## Optimization Direction

The next tranche should proceed in this order:

1. Add a cold-stream benchmark that rotates through a working set larger than LLC and can replay the 48-layer Qwen
   projection order. Report warm/cold ratio, bytes per call, effective GB/s, worker count, and grouped/single mode.
2. Add low-overhead worker diagnostics around FFN Down to measure dispatch, useful work, barrier wait, and per-worker
   bytes. Use this to distinguish fixed helper overhead from insufficient memory-level parallelism.
3. Evaluate Down-specific execution changes: multiple independent output-group streams per worker, software prefetch,
   Q4_K x16 AVX2 under a cold-stream-only evidence gate, and Q6_K AVX2 versus AVX-512 selection.
4. Fuse SwiGLU output production with Q8_K preparation for FFN Down so the intermediate Float32 activation is not
   traversed and dispatched twice.
5. Accept a change only after alternating full-decode LiteNN/llama.cpp runs. Cache-hot helper wins alone are not enough.

Acceptance for CPU parity work:

- no fallback and unchanged generated-token parity;
- separated prepared weights remain no more than `1.03x` the source quantized payload;
- Q4_K and Q6_K Down each sustain at least `40 GB/s` in the new cold-stream benchmark;
- normalized FFN latency is within `10%` of the same-run llama.cpp FFN block;
- LiteNN generated-token latency is within `5%` of the alternating CPU-only llama.cpp T8 control median.

