# Qwen CPU Decode Performance Evidence - 2026-08-04

> Historical stage evidence: the measurements below remain valid for the recorded binaries, but their cross-runtime
> ranking predates the controlled compiler/OpenMP matrix. The current performance conclusion and next evidence gate
> are owned by `QwenCPUDecodeBuildControl_2026-08-04.md`.

> Current decision baseline: the later alternating Clang/no-OpenMP control measured LiteNN `5.49%` behind by the
> preferred median per-pair statistic. A subsequent exact-token, non-synchronizing aggregate profile passed all
> measurement gates and selected FFN activation + Down as the largest accepted stage deficit. The older `26.9%`
> result below is retained as historical stage evidence, not as the current project gap. The residual closure and
> scheduling controls near the end of this document use the current post-quantizer implementation.

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
- The historical synchronized llama.cpp boundary run repeatedly decoded one valid token. Its shapes and short-context
  execution are comparable, but it is not a generated-text parity test. The accepted aggregate control instead
  replays the exact prompt and 15 generated token ids used by the LiteNN profile.
- The cold-stream benchmark reproduces the observed 48-layer Down weight order and activation turnover, but excludes
  surrounding normalization, residual, and scheduler work.

## End-to-End Result

| Runtime | Latency | Throughput | Latency relative to llama.cpp | Throughput relative to llama.cpp |
| --- | ---: | ---: | ---: | ---: |
| LiteNN CPU AOT | `256.616 ms/token` | `3.89687 token/s` | `1.269x` | `0.788x` |
| llama.cpp CPU | `202.224 ms/token` | `4.94502 token/s` | `1.000x` | `1.000x` |

The observed gap is `54.392 ms/token`. LiteNN is `26.9%` slower relative to the llama.cpp latency denominator and
delivers `21.2%` less throughput in this run.

## Accepted Exact-Token Aggregate Control

The repository-owned benchmark patch records coarse llama.cpp CPU stages on graph thread zero and reuses existing
ggml graph barriers. It adds no callback synchronization. A clean executable and a separately instrumented executable
were built with the same pinned source, Clang target/sysroot, no-OpenMP configuration, and optimization settings.
Three exact-token repetitions at T8 passed every acceptance gate:

| Gate | Result |
| --- | ---: |
| Clean baseline median | `155.541 ms/token` |
| Instrumented median | `154.745 ms/token` |
| Median measurement overhead | `-0.21%` |
| Clean / instrumented whole-run CV | `0.23%` / `0.39%` |
| Accounted stage coverage | `98.75%` |
| Stage CV range | `0.16-0.98%` |

The accepted stage medians and matched LiteNN helper-only lower bounds are:

| Stage | LiteNN lower bound | llama.cpp aggregate | Conservative LiteNN deficit |
| --- | ---: | ---: | ---: |
| Attention | `36.226 ms` | `34.169 ms` | at least `+2.057 ms` (`+6.02%`) |
| FFN Gate/Up | `65.048 ms` | `67.038 ms` | no deficit established |
| FFN activation + Down | `54.057 ms` | `41.165 ms` | at least `+12.892 ms` (`+31.32%`) |
| Final logits | `10.995 ms` | `10.943 ms` | `+0.052 ms` (`+0.48%`) |

FFN activation + Down is the only accepted double-digit stage deficit. It is about `6.3x` the Attention difference
and explains at least `63%` of the profiled module-versus-clean-reference difference. This result opens the Down
cold-stream kernel and memory-scheduling tranche; it does not reopen Gate/Up, logits, or broad dispatch tuning.

## Historical Synchronized Stage Comparison

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

### Selective Q4_K x16 Down Control

After the non-synchronizing reference profile selected FFN Down, the existing AVX2 x16 Q4_K tile was re-evaluated only
for contraction projections whose input width is at least twice the output width. This excludes the previously
regressed Gate/Up, square hidden, and vocabulary projection classes. Separate baseline and experimental executables
were alternated in three process pairs; each process reported the median of three T8 cold-stream repetitions.

| Stream | Pair 1 | Pair 2 | Pair 3 | Paired median | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| Q4_K Down x24 | `+7.35%` | `-10.86%` | `+1.51%` | `+1.51%` | Below gate |
| Unchanged Q6_K control x24 | `-5.55%` | `+1.16%` | `+0.81%` | `+0.81%` | Host-drift control |
| Real mixed Q4_K_M Down x48 | `+4.45%` | `-3.41%` | `-9.09%` | `-3.41%` | Regressed |

Positive values mean the x16 executable was faster. The Q4_K-only effect is small and unstable, while the production
mixed sequence regresses. The prototype was removed without a full-model run. A direct x16 reuse is therefore closed;
Q4_K needs a different decomposition or memory-scheduling change.

### Bounded Software Prefetch Control

A second contraction-only experiment prefetched four cache lines from each future field-v4 weight block into the
shared cache hierarchy. Q4_K x8 used one stream and Q6_K x16 used both independent packed-group streams. Distances of
2, 4, and 8 quantization blocks were tested without changing arithmetic, scheduling, or the prepared layout ABI.

| Prefetch distance | Q4_K x24 median | Q6_K x24 median | Mixed x48 median |
| ---: | ---: | ---: | ---: |
| 2 blocks | `21.263 ms` | `31.916 ms` | `52.019 ms` |
| 4 blocks | `19.283 ms` | `29.685 ms` | `47.574 ms` |
| 8 blocks | `20.112 ms` | `28.220 ms` | `47.873 ms` |
| Clean alternating-run range | `18.72-20.67 ms` | `26.06-27.76 ms` | `42.87-45.87 ms` |

No distance beats the clean mixed-stream range, and Q6_K regresses most consistently. The prefetch code was removed.
The current sequential block access is already served effectively by hardware prefetch; adding sparse software hints
increases instruction and cache pressure instead of closing the cold-stream deficit.

### Pair-Sum-Folded Q4_K x16 Control

The two-stream Q4_K x16 tile was then rebuilt around the accepted x8 reduction: each stream accumulates safe Int16
pair sums and folds pair reduction plus scale into one `vpmaddwd`, instead of expanding every chunk to Int32 before a
separate scale multiply. Only contraction-shaped Down rows entered the new tile. Five focused correctness tests passed,
but three alternating baseline/variant process pairs rejected it:

| Stream | Pair 1 | Pair 2 | Pair 3 | Paired median |
| --- | ---: | ---: | ---: | ---: |
| Q4_K Down x24 | `-6.91%` | `-18.30%` | `-4.59%` | `-6.91%` |
| Unchanged Q6_K control x24 | `+7.63%` | `-5.18%` | `-6.36%` | `-5.18%` |
| Real mixed Q4_K_M Down x48 | `-2.80%` | `-1.53%` | `-5.23%` | `-2.80%` |

All Q4_K and mixed pairs regress despite the shorter arithmetic chain and a lower cache-hot call time. The prototype
was removed. Interleaving two packed output-group streams increases cold-stream pressure enough to outweigh shared
activation loads and instruction savings on this host.

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

## RMSNorm Lowering Control

The first cluster experiment replaced the six-stage Float32 RMSNorm lowering with a strict rank-2 CPU AOT helper for
last-axis normalization with `[1, hidden]` scale and no bias. Other dtypes, axes, affine forms, and normalization modes
keep the generic lowering. A real cache-miss compile and profiled decode verified 97 helper calls per token at
`0.36-0.45 ms/token`, compared with the earlier `1.956 ms/token` intrusive normalization row.

The structural compile comparison used the same model, decode schedule, O0 LLVM setting, T8 helper policy, and
field-interleaved-v4 weights:

| Compile metric | Materialized RMSNorm | Native helper | Change |
| --- | ---: | ---: | ---: |
| Post-LiteNN MLIR operations | `19,337` | `16,668` | `-13.80%` |
| Post-bufferization operations | `29,640` | `26,351` | `-11.10%` |
| Post-LLVM-lowering operations | `360,190` | `332,496` | `-7.69%` |
| LLVM IR instructions | `244,443` | `228,166` | `-6.66%` |
| Object emission | `8,591.008 ms` | `7,712.264 ms` | `-10.23%` |
| Object bytes | `1,046,614` | `1,015,284` | `-2.99%` |
| Complete AOT artifact compile | `22,742.512 ms` | `22,394.102 ms` | `-1.53%` |

The no-profile cache-hit control alternated materialized/helper order across three pairs. Per-pair full-generation
latencies were `177.810/177.240`, `180.350/168.760`, and `176.300/173.700 ms/token`, corresponding to helper gains of
`0.32%`, `6.42%`, and `1.47%`. Generated token ids matched in every pair. The preferred paired-median gain is `1.47%`,
below the predefined `2%` full-token gate. The standalone helper is therefore retained for its measured compiler-size
benefit, but it does not close the runtime cluster target.

The follow-up fused RMSNorm directly into Q8_K activation staging for a single grouped field-v4 projection consumer.
Loaded AOT and public-result fallback parity passed, but three alternating cache-hit pairs measured `-3.23%`, `+4.14%`,
and `-0.19%`, for a `-0.19%` paired median with identical tokens and no fallback. LLVM instructions fell `1.26%` and
object bytes fell `1.36%`, but the runtime gate failed and the experimental lowering and two helper ABIs were removed.
Projection-wrapper self remains an instrumented upper bound because the node timer surrounds external-helper calls;
it is not evidence for broad ABI rewrites.

## Dispatch Controls

Stable post-quantizer profiling measured `10.365 ms/token` of worker dispatch across 97 ordinary parallel projections,
or about `0.107 ms/call`. Dispatch is already included in parallel-wall and helper time and must not be added to the
non-helper ledger.

Five controls constrain the implementation route:

| Control | Dispatch result | End-to-end/parallel result | Decision |
| --- | ---: | --- | --- |
| `atomic::wait/notify_one` | `21.7%` lower | Parallel wall/barrier rose; profiled token regressed `0.54%` | Rejected |
| Worker wait `Latency` versus `Adaptive` | Not isolated | Median `252.650` versus `255.594 ms`, `1.16%` better | Keep explicit option; not the default |
| Signal only sleeping workers | `10.365 -> 10.139 ms`, `2.2%` lower | Diagnostic parallel wall `66.225 -> 76.543 ms`; below gate | Rejected and removed |
| Full module sequence standby | About `10.365 -> 0.05 ms`, over `99%` lower | Parallel wall `66.225 -> 71.595 ms`; paired median token gain `2.81%` | Rejected and removed |
| Current-width module standby | Ordinary projection dispatch remained about `0.05 ms` | Parallel wall `68.795 ms`; paired median token gain `1.46%` | Rejected and removed |

The sleeping-worker signal-elision comparison was not an alternating throughput batch, so its wall-time regression is
directional. The two module-sequence variants were then tested against a same-interface baseline in alternating order.
Full standby produced per-pair median gains of `-0.40%`, `+2.81%`, and `+3.90%`; current-width standby, which parks the
extra four workers across T4 hidden projections, produced `+1.46%`, `-1.73%`, and `+2.95%`. Generated token ids matched
within every pair. Neither variant met the `3%` paired-median gate, and both retained a parallel-wall regression in the
structured profile. Both implementations were removed.

The combined evidence now closes dispatch-only work: the measured dispatch counter can be almost eliminated, but the
saved producer-side signaling cost is exchanged for worker residency, wake arrival, or parallel-wall cost and does not
produce a stable token-level win. The profiler retains `signaledWorkerCount` so a future helper fusion can demonstrate
fewer submissions, but another thread-pool wait/standby policy is not a current optimization target.

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
5. `CallNode` inlining and marker removal cannot provide production gain. Standalone native RMSNorm reduced compiler
   IR/object size and delivered a `1.47%` paired-median token gain, below the `2%` runtime gate. RMSNorm-to-Q8_K staging
   fusion then produced a `-0.19%` paired median and was removed; projection-wrapper self remains an instrumented upper
   bound.
6. Dispatch-only optimization is closed. Two wake paths and two sequence-standby variants failed token and/or
   parallel-wall gates even when the dispatch counter fell by more than `99%`.

### Supported but not yet proven

1. The accepted FFN activation + Down deficit and `23-30 GB/s` cold projection rate support a kernel or
   memory-scheduling deficit. The next experiment must still separate interleaving, prefetch, and format-specific SIMD
   effects rather than assuming one of them is the owner.

### Not established

1. Fine llama.cpp callback differencing remains rejected because stage CV ranged from `10.53%` to `82.51%`; the
   accepted replacement is the non-synchronizing aggregate mode above.
2. PMU evidence does not yet distinguish LLC/DRAM stalls from dispatch imbalance on Windows.
3. Wider SIMD, more threads, prefetch, or a new prepared layout has not shown both a cold-stream and full-model win.

## Decision Record

| Priority | Decision | Evidence gate |
| --- | --- | --- |
| Done | Close native module residual accounting | Closure within `2%`, instrumentation within `3%` |
| Done | Reject wake-primitive and sequence-standby dispatch changes | Dispatch plus parallel wall plus full-token gate |
| Done | Reject RMSNorm-to-Q8_K staging fusion and broad wrapper ABI work | Exact-token paired median `-0.19%`; experiment removed |
| Done | Add low-overhead matched reference-stage counters | `-0.21%` overhead, `98.75%` coverage, `0.16-0.98%` stage CV |
| P0 | Raise FFN-Down cold-stream throughput | Improvement in cold stream and complete decode, not hot rows alone |
| P0 | Re-run full-model no-profile controls after accepted changes | Identical generated tokens and alternating medians |
| P2 | Add optional PMU evidence | Clean fallback on hosts without profiling privileges |

The implementation order and completion state are maintained in `PerformanceOptimizationRoadmap.md`; the high-level
project gate is mirrored in `Roadmap.md`.
