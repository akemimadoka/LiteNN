# Qwen CPU Decode Performance Evidence - 2026-08-04

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
activation cache miss and preparation path as decode; the shared-activation control changes only activation reuse.

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

## Isolated Activation-Handoff Cost

The real-order distinct-activation stream is `32.661 ms`, or `51.0%` of the shared-control time, slower than the same
weight stream with a reused activation. The changed path performs a Float32 activation comparison and copy and then
regenerates Q8_K blocks for each layer before entering the same prepared projection kernel.

If the complete `32.661 ms` transferred to end-to-end decode, eliminating it would produce an optimistic bound of
`223.955 ms/token` or `4.465 token/s`. That would close `60.0%` of the observed end-to-end gap but would still leave
LiteNN about `10.7%` slower by latency. This is a prioritization bound, not a forecast: AOT codegen, cache behavior, and
the fused implementation must be measured in the full model.

## Conclusions

### Established by measurement

1. FFN is the dominant short-context CPU decode gap; Attention and final logits are not the first owners.
2. The deficit is concentrated in Q4_K/Q6_K activation-plus-Down, while grouped Gate/Up is comparatively close.
3. Cache-hot Up and Down rows are nearly direction-neutral, so hot helper throughput does not predict model latency.
4. The production-sized cold stream reproduces the real Down helper totals within `12.5%`.
5. Activation turnover adds `32.661 ms` to an otherwise identical 48-matrix stream. This is the largest directly
   isolated and currently actionable surcharge.

### Supported but not yet proven

1. Producing Q8_K activation blocks at the single-consumer SwiGLU boundary and consuming them directly in Down should
   remove more latency than speculative kernel tuning.
2. The remaining shared-activation rate of `37.08 GB/s` versus the stage-control lower bounds suggests additional
   worker scheduling, memory concurrency, or kernel issues after activation fusion.

### Not established

1. PMU evidence does not yet distinguish LLC/DRAM stalls from dispatch imbalance on Windows.
2. The full `32.661 ms` isolated surcharge is not guaranteed to disappear in end-to-end execution.
3. Wider SIMD, more threads, prefetch, or a new prepared layout has not shown a cold-stream and full-model win yet.

## Decision Record

| Priority | Decision | Evidence gate |
| --- | --- | --- |
| P0 | Fuse single-consumer SwiGLU preparation with field-v4 Q4_K/Q6_K Down | Runtime parity, AOT artifact routing, cold-stream A/B |
| P0 | Re-run full-model stage and no-profile controls | Identical generated tokens and alternating T8 medians |
| P1 | Attribute residual shared-activation time | Activation, dispatch, useful work, and barrier phases |
| P1 | Tune projection scheduling or kernels only after attribution | Improvement in cold stream and complete decode, not hot rows alone |
| P2 | Add optional PMU evidence | Clean fallback on hosts without profiling privileges |

The implementation order and completion state are maintained in `PerformanceOptimizationRoadmap.md`; the high-level
project gate is mirrored in `Roadmap.md`.
