# Qwen CPU Decode Performance Evidence - 2026-08-04

> Historical stage evidence: the measurements below remain valid for the recorded binaries, but their cross-runtime
> ranking predates the controlled compiler/OpenMP matrix. The current performance conclusion and next evidence gate
> are owned by `QwenCPUDecodeBuildControl_2026-08-04.md`.

> Current decision baseline: the later alternating Clang/no-OpenMP control measured LiteNN `5.49%` behind by the
> preferred median per-pair statistic. The older `26.9%` result below is retained as historical stage evidence, not as
> the current project gap. The residual closure and scheduling controls near the end of this document use the current
> post-quantizer implementation.

This document is the evidence record for the current Qwen CPU decode optimization tranche. It separates measured
results and supported conclusions from the implementation checklist in `PerformanceOptimizationRoadmap.md`. The more
detailed profiling narrative remains in `PerformanceAnalysis_2026-08-04.md`.

## Scope

The target workload is Qwen2.5-Coder-14B-Instruct Q4_K_M, stateful one-token CPU AOT decode. The compared paths use
eight CPU workers, no GPU offload, and no Flash Attention. LiteNN uses separated cached AOT artifacts and prepared
`field-interleaved-v4` weights. Private model and local build paths are intentionally omitted.

The comparison answers three questions:

1. Which major model stage owns the remaining LiteNN versus llama.cpp latency gap?
2. Can the gap be reproduced outside the complete model with production-sized data?
3. Which implementation change has the largest evidence-backed upper bound?

## Measurement Boundaries

- End-to-end values are adjacent no-profile runs on the same host. They remain sensitive to host frequency and cache
  state and are not release acceptance by themselves.
- Stage values are normalized by each profiler's adjacent no-profile/profile latency ratio. They are attribution
  estimates, not PMU measurements.
- LiteNN helper buckets are lower bounds because generated-code work between helpers is excluded. llama.cpp boundary
  buckets include all work between selected graph tensors.
- The llama.cpp stage run repeatedly decodes one valid token. Shapes and short-context execution are comparable, but
  this is not a generated-text parity test.
- The cold-stream benchmark reproduces the observed 48-layer Down weight order and activation turnover, but excludes
  surrounding normalization, residual, and scheduler work.

## End-to-End Result

| Runtime | Latency | Throughput | Latency relative to llama.cpp | Throughput relative to llama.cpp |
| --- | ---: | ---: | ---: | ---: |
| LiteNN CPU AOT | `256.616 ms/token` | `3.89687 token/s` | `1.269x` | `0.788x` |
| llama.cpp CPU | `202.224 ms/token` | `4.94502 token/s` | `1.000x` | `1.000x` |

The observed gap is `54.392 ms/token`. LiteNN is `26.9%` slower relative to the llama.cpp latency denominator and
delivers `21.2%` less throughput in this run.

## Stage Comparison

The LiteNN helper profile measured `281.703 ms/token` beside a `256.616 ms/token` no-profile run. The llama.cpp
boundary profile measured `222.810 ms/token` beside a `202.224 ms/token` no-profile run. The corresponding scale
factors are `0.910945` and `0.907607`.

| Stage | LiteNN normalized | llama.cpp normalized | LiteNN gap | Result |
| --- | ---: | ---: | ---: | --- |
| Complete FFN evidence | `168.48 ms` lower bound | `121.94 ms` full block | at least `46.54 ms` | Primary owner |
| Gate + Up | `81.52 ms` | `76.55 ms` | at least `4.97 ms` | Near parity relative to Down |
| Q4_K activation + Down | `39.19 ms` | `19.17 ms` | `20.03 ms`, `2.04x` | Primary target |
| Q6_K activation + Down | `47.77 ms` | `26.23 ms` | `21.54 ms`, `1.82x` | Primary target |
| Attention | `63.83 ms` lower bound | `69.74 ms` full block | not comparable as a deficit | Not the short-context owner |
| Final logits | `11.35 ms` | `10.25 ms` | `1.10 ms` | Secondary |

The FFN attribution explains at least `85.6%` of the end-to-end gap. Q4_K and Q6_K activation-plus-Down account for
about `41.56 ms`, or `76.4%` of the end-to-end gap. Attention-first work cannot close CPU parity for this short-context
workload.

## Raw Helper and Boundary Data

| LiteNN generation helper bucket | Raw time per token |
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

| llama.cpp low-boundary bucket | Raw time per token |
| --- | ---: |
| Attention block | `76.836 ms` |
| Q4_K FFN activation and Down | `21.118 ms` |
| Q6_K FFN activation and Down | `28.895 ms` |
| FFN prefix and Gate | `44.185 ms` |
| FFN Up | `40.154 ms` |
| Final logits | `11.289 ms` |
| Graph trailing work | `0.182 ms` |

The low-boundary llama.cpp profile inserted 194 boundaries per token. A finer profile added about `25%` overhead and
is excluded from quantitative conclusions.

## Cache-Hot Kernel Control

The existing helper benchmark repeatedly reads one prepared matrix and therefore measures an instruction-throughput
ceiling rather than model-sized streaming behavior.

| Format | `5120 -> 13824` Up | `13824 -> 5120` Down |
| --- | ---: | ---: |
| Q4_K | `0.308 ms` | `0.328 ms` |
| Q6_K | `0.631 ms` | `0.623 ms` |

Direction reversal does not reproduce the complete-model Down deficit. The arithmetic kernel is competitive when the
same matrix remains hot, so a broad kernel rewrite is not the first evidence-backed action.

## Cache-Cold Stream Reproduction

`GGMLFieldInterleavedV4ColdProjectionStream` rotates through prepared weights larger than the host's aggregate LLC.
The production row follows the observed 24 Q4_K plus 24 Q6_K Down order. Distinct-activation rows force the same Q8_K
activation cache miss and preparation path as decode; the shared-activation control changes activation identity and
all prepared-block caching implied by that reuse.

Release T8, three-repetition medians:

| Stream | Sequence | Per call | Prepared-byte rate | Cold/hot | Prepared/source |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q4_K Down x24, distinct activation | `41.175 ms` | `1.716 ms` | `23.85 GB/s` | `2.00x` | `1.0278x` |
| Q6_K Down x24, distinct activation | `53.221 ms` | `2.218 ms` | `26.18 GB/s` | `1.80x` | `1.0000x` |
| Real Q4_K_M Down x48, distinct activation | `96.726 ms` | `2.015 ms` | `24.56 GB/s` | `1.96x` | `1.0113x` |
| Same x48 weights, shared activation | `64.065 ms` | `1.335 ms` | `37.08 GB/s` | `1.31x` | `1.0113x` |

The isolated Q4_K and Q6_K sequences are respectively `8.7%` and `12.5%` above the complete-model helper totals.
This is sufficiently close to use the cold stream as the immediate optimization gate. Reference comparison reported
maximum absolute deltas of `8.55e-4` for Q4_K and `4.15e-3` for Q6_K.

## Controlled SwiGLU Fusion A/B

The distinct/shared control above does not isolate activation materialization. Reusing one activation also reuses its
prepared Q8_K blocks and changes the cache and access pattern of all 48 calls. The `32.661 ms` difference therefore
measures the combined effect of activation identity and prepared-block reuse; it is not a valid estimate of the cost
that a SwiGLU-to-Down fusion can remove.

The compiler now marks only rank-2 Float32 SwiGLU values with one non-public consumer and lowers compatible
field-interleaved-v4 Q4_K/Q6_K Down projections to a fused runtime helper. Runtime, object import/load, and AOT execution
parity pass for both formats. A paired benchmark alternates the materialized and fused sequences in each iteration over
the same 48 prepared weights, gate values, and up values:

| Paired T8 run | Materialized mean | Fused mean | Materialized - fused | Result |
| --- | ---: | ---: | ---: | --- |
| 5 repetitions | `83.3715 ms` | `83.0530 ms` | `+0.3185 ms` (`+0.38%`) | Small apparent fused win |
| 7 repetitions | `80.9137 ms` | `80.9833 ms` | `-0.0696 ms` (`-0.09%`) | Sign reversal |

The sequence means are stable within each run, but the paired delta is smaller than run-to-run noise and changes sign.
The maximum output delta against the staged reference was about `6.71e-4`. The supported conclusion is that the fusion
is performance-neutral at this scale. It remains useful for a cleaner compiler-owned dataflow and removes redundant
materialization checks, but it does not explain or close the FFN-Down gap. The previous `223.955 ms/token` optimistic
bound is invalidated.

## Current Native Residual Closure

The current cache-hit artifact was profiled with helper, stream, and native-node aggregate instrumentation over eight
prompt-replay and 16 generation steps. The private model path and raw local artifact paths are intentionally omitted.
The generation phase produced the following complete ledger:

| Bucket | Mean ms/token | Share of module | Interpretation |
| --- | ---: | ---: | --- |
| CPU AOT module | `205.435` | `100.00%` | Intrusively profiled module call; not a throughput baseline |
| Timed helpers | `185.885` | `90.48%` | Includes projection dispatch and barriers |
| Module non-helper | `19.550` | `9.52%` | Closed by the rows below |
| Native node self | `13.499` | `6.57%` | Generated operations outside helper timers |
| Node-marker instrumentation | `5.392` | `2.62%` | Profiling cost, absent from production |
| Module unattributed | `0.658` | `0.32%` | Remaining timer/accounting residual |

The categorized ledger reconciled to module non-helper time with effectively zero arithmetic closure error. Marker
instrumentation was `2.62%` of module time, so the run passes the predefined `2%` closure and `3%` perturbation gates.

| Non-helper category | Mean ms/token | Share of non-helper | Actionability |
| --- | ---: | ---: | --- |
| Node-marker instrumentation | `5.392` | `27.58%` | Measurement-only; not a production optimization |
| Call/control | `5.347` | `27.35%` | Non-profile always-inline A/B already rejected |
| Projection wrapper | `3.124` | `15.98%` | Largest unresolved generated-code category |
| Normalization | `1.956` | `10.01%` | Secondary generated-code category |
| Unemitted node self | `1.384` | `7.08%` | Aggregate coverage follow-up, not necessarily missing work |
| Attention/position/state | `1.025` | `5.24%` | Small at the tested short context |
| Module unattributed | `0.658` | `3.37%` | Too small to own the cross-runtime gap |
| Elementwise | `0.569` | `2.91%` | Small individually |
| Views/data movement and remaining rows | `0.095` | `0.49%` | Not a current target |

The `CallNode` category is not an optimization estimate. Forced always-inlining previously reduced the module from 52
functions to two and cut LLVM IR instruction count, yet increased compile-artifact time by `22.7%` and changed the
no-profile decode median from `263.721` to `264.153 ms/token` (`0.16%` slower). Marker cost is likewise absent from
normal execution. Removing both from the ledger leaves projection wrappers plus normalization as the largest plausible
generated-code cluster, about `5.08 ms` in this intrusive run; it requires a non-profile A/B before promotion.

## Dispatch Controls

Stable post-quantizer profiling measured `10.365 ms/token` of worker dispatch across 97 ordinary parallel projections,
or about `0.107 ms/call`. Dispatch is already included in parallel-wall and helper time and must not be added to the
non-helper ledger.

Three controls constrain the implementation route:

| Control | Dispatch result | End-to-end/parallel result | Decision |
| --- | ---: | --- | --- |
| `atomic::wait/notify_one` | `21.7%` lower | Parallel wall/barrier rose; profiled token regressed `0.54%` | Rejected |
| Worker wait `Latency` versus `Adaptive` | Not isolated | Median `252.650` versus `255.594 ms`, `1.16%` better | Keep explicit option; not the default |
| Signal only sleeping workers | `10.365 -> 10.139 ms`, `2.2%` lower | Diagnostic parallel wall `66.225 -> 76.543 ms`; below gate | Rejected and removed |

The last comparison was not an alternating throughput batch, so its wall-time regression is directional. It is still
enough to reject the implementation because dispatch improvement was far below the required `50%` and no compensating
whole-token gain was observed. The combined evidence says that swapping wake primitives or suppressing semaphore
signals cannot close the gap. Any further P0 dispatch work needs a sequence-level contract that amortizes multiple
helper submissions while preserving useful-worker arrival and memory bandwidth.

## Current Conclusions

### Established by measurement

1. The current accepted cross-runtime gap is `5.49%` against the alternating Clang/no-OpenMP control, not the older
   `26.9%` historical stage result.
2. Cache-hot Up and Down rows are nearly direction-neutral, and the cold stream reproduces real Down helper totals
   within `12.5%`; hot helper rank alone cannot select a kernel rewrite.
3. The controlled SwiGLU fusion is performance-neutral within noise. The `32.661 ms` distinct/shared difference is not
   a removable materialization estimate.
4. The native non-helper ledger now closes within the `2%` requirement with `2.62%` marker overhead. Only `0.658 ms`
   remains unattributed.
5. `CallNode` inlining and marker removal cannot provide production gain. Projection wrapper plus normalization is the
   largest unresolved generated-code cluster, but its measured upper bound still requires non-profile validation.
6. Dispatch remains a material helper-side owner. Two wake-path implementations failed, and the latency polling policy
   produced only a `1.16%` median advantage, so the next attempt must amortize submissions across a helper sequence.

### Supported but not yet proven

1. A sequence-level dispatch contract can remove enough of the `10.365 ms/token` floor without delaying useful worker
   arrival or reducing memory bandwidth.
2. Projection wrapper and normalization work contains a removable generated-code cost rather than unavoidable ABI,
   shape, or profile-boundary overhead.
3. The remaining `23-30 GB/s` cold projection rate reflects a kernel or memory-scheduling deficit against the stronger
   reference; matched low-overhead reference-stage counters are still required before choosing that route.

### Not established

1. Fine llama.cpp callback differencing remains rejected because stage CV ranged from `10.53%` to `82.51%`.
2. PMU evidence does not yet distinguish LLC/DRAM stalls from dispatch imbalance on Windows.
3. Wider SIMD, more threads, prefetch, or a new prepared layout has not shown both a cold-stream and full-model win.

## Decision Record

| Priority | Decision | Evidence gate |
| --- | --- | --- |
| Done | Close native module residual accounting | Closure within `2%`, instrumentation within `3%` |
| Done | Reject wake-primitive-only dispatch changes | Dispatch plus parallel wall plus full-token gate |
| P0 | Add sequence-level projection dispatch amortization | At least `50%` less dispatch, `3%` less token latency, no wall/barrier regression |
| P1 | Prove or reject projection-wrapper plus normalization removal | Non-profile paired A/B; at least `2%` full-token gain |
| P1 | Add low-overhead matched reference-stage counters | Below `3%` total overhead and below `15%` stage CV |
| P1 | Tune projection scheduling or kernels only after attribution | Improvement in cold stream and complete decode, not hot rows alone |
| P0 | Re-run full-model no-profile controls after accepted changes | Identical generated tokens and alternating medians |
| P2 | Add optional PMU evidence | Clean fallback on hosts without profiling privileges |

The implementation order and completion state are maintained in `PerformanceOptimizationRoadmap.md`; the high-level
project gate is mirrored in `Roadmap.md`.
