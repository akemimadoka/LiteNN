# Qwen CPU Decode FFN Activation/Down Decision - 2026-08-04

## Scope

This document records the measurements that close the first FFN-Down microkernel tranche and re-rank the remaining
Qwen CPU decode work. It separates accepted measurements from source-level hypotheses. The implementation checklist
lives in `PerformanceOptimizationRoadmap.md`; this file is the immutable decision evidence for that checklist.

The compared workload is the same short-window, exact-token Qwen CPU decode control used by the other 2026-08-04
reports. Private model and executable paths are intentionally omitted. LiteNN uses a cache-hit CPU AOT artifact and
the field-interleaved-v4 prepared-weight policy at T8. The reference is the controlled Clang/no-OpenMP llama.cpp build.

## Measurement Quality

The reference aggregate profiler was accepted before using its stages for ranking:

| Gate | Measured | Required |
| --- | ---: | ---: |
| Clean reference median | `155.541 ms/token` | control |
| Instrumented reference median | `154.745 ms/token` | control |
| Paired instrumentation overhead | `-0.21%` | absolute value at most `3%` |
| Aggregate stage coverage | `98.75%` | `95-102%` |
| Clean/instrumented whole-run CV | `0.23% / 0.39%` | at most `3%` |
| Reference stage CV range | `0.16-0.98%` | at most `15%` |

The LiteNN helper decomposition uses stable generated-token positions 10-24. It is suitable for partitioning the
LiteNN stage, but helper medians are not additive with perfect precision and omit inline work.

## Accepted Stage Data

| Matched stage | LiteNN helper lower bound | LiteNN CV | Reference complete stage | Apparent difference |
| --- | ---: | ---: | ---: | ---: |
| Attention | `36.226 ms` | `4.71%` | `34.169 ms` | at least `+2.057 ms` |
| FFN Gate/Up | `65.048 ms` | `3.33%` | `67.038 ms` | no deficit established |
| FFN activation + Down | `54.057 ms` | `2.97%` | `41.165 ms` | at least `+12.892 ms` |
| Final logits | `10.995 ms` | `4.67%` | `10.943 ms` | `+0.052 ms` |

The accepted comparison proves that the composite FFN activation + Down stage is the largest remaining matched-stage
deficit. It does not by itself prove that the Down projection kernels own the complete `12.892 ms` difference.

### LiteNN FFN Activation/Down Decomposition

| LiteNN helper | Median ms/token | Calls/token |
| --- | ---: | ---: |
| Q6_K Down | `25.602` | 24 |
| Q4_K Down | `17.121` | 24 |
| Projection subtotal | `42.723` | 48 |
| Standalone SwiGLU | `11.043` | 48 |
| Sum of component medians | `53.766` | - |

The `0.291 ms` difference between the component sum and the `54.057 ms` aggregate is expected from taking medians at
different aggregation levels. The important comparison is:

```text
LiteNN projection-only subtotal      42.723 ms
reference activation + Down total    41.165 ms
minimum visible projection excess     1.558 ms (3.78%)
```

The reference activation cost has not yet been split from its Down projection cost. If that unknown cost is `A`, the
projection difference is `1.558 ms + A` and the activation difference is approximately `11.043 ms - A`. Therefore:

- assigning the complete `12.892 ms` composite difference to projection kernels is invalid;
- assigning it all to SwiGLU is also invalid;
- a reference-side activation/Down split is required before another projection rewrite can be evidence-selected.

## Completed Down Experiments

Every candidate below was built against the same source baseline, exercised through the production-shaped cache-cold
benchmark, and removed when it missed the gate. Positive percentages mean the candidate was faster.

| Candidate | Q4_K or Q6_K result | Real mixed Q4_K_M result | Decision |
| --- | --- | --- | --- |
| Existing Q4_K AVX2 x16, contraction-only | `+7.35% / -10.86% / +1.51%`; median `+1.51%` | `+4.45% / -3.41% / -9.09%`; median `-3.41%` | Rejected and removed |
| Q4_K pair-sum-folded dual x8 output streams | `-6.91% / -18.30% / -4.59%`; median `-6.91%` | `-2.80% / -1.53% / -5.23%`; median `-2.80%` | Rejected and removed |
| Software prefetch distance 2/4/8 | Q6_K regressed consistently | `52.019 / 47.574 / 47.873 ms` versus clean `42.87-45.87 ms` | Rejected and removed |
| Q6_K AVX2 x8 versus AVX-512 x16 | median `-3.98%` | median `+1.76%` | Rejected and removed |
| Q6_K AVX2 x16 versus AVX-512 x16 | median `+2.31%` | median `-2.11%` | Rejected and removed |

The production Q6_K AVX-512 x16 path remains selected. The production Q4_K x8 path also remains selected for this
shape. These results close the current x16 reuse, output-stream, software-prefetch, and Q6 ISA-selection branches.
None earned an exact-token full-model run because the cold mixed-stream gate failed first.

### Earlier Controlled Fusion Result

The existing materialized-versus-fused mixed-format cold-stream benchmark changed sign across independent samples:

| Sample | Materialized minus fused | Interpretation |
| --- | ---: | --- |
| Five-run sample | `+0.3185 ms` (`+0.38%`) | apparent fused win |
| Seven-run sample | `-0.0696 ms` (`-0.09%`) | apparent fused loss |

This proves that merely replacing a materialized SwiGLU followed by Down with the current fused helper is neutral
within noise in that benchmark. The fused helper still evaluates SwiGLU and stages Q8_K activation; it is not a fused
vector-math/projection kernel.

## Production Artifact Audit

The accepted real Qwen helper profile reports 48 calls to `litenn_cpu_swiglu_f32` per token and no calls to
`litenn_cpu_swiglu_ggml_block_matmul_field_interleaved_v4_q8k_f32`. Thus the profiled production artifact did not use
the already implemented fusion helper. This is an artifact fact, not yet a root-cause proof.

The source audit found a likely contract mismatch:

- `getFusableSwiGLUDownConsumer` requires the plan node's source `storageLayout` to already be
  `GGMLFieldInterleavedV4`;
- the selected prepared layout is attached later while emitting the quantized MatMul;
- imported graphs may retain source GGML storage in the plan and select field-interleaved-v4 only during AOT
  preparation.

The next implementation must first add a real imported-plan/artifact regression that proves this condition is the
reason fusion is absent. The fix must key fusion eligibility to the prepared layout actually emitted, not assume the
source layout. A symbol/IR assertion must prevent a synthetic graph from being mistaken for production coverage.

## Source-Level Activation Candidate

Both the standalone and current fused helpers evaluate SwiGLU lane by lane with scalar `std::exp`:

```text
gate / (1 + exp(-gate)) * up
```

This makes vectorized sigmoid/SwiGLU math the largest visible, independently replaceable helper candidate at
`11.043 ms/token`. It is not yet a measured cross-runtime deficit because llama.cpp's activation time is still inside
the `41.165 ms` composite reference stage. A SIMD or bounded fast-exp implementation must therefore pass both a
standalone speed gate and exact-token full-model validation; source inspection alone is not an acceptance argument.

## Decision

1. Close the current FFN-Down cold-stream microkernel tranche. Five bounded variants failed the production-shaped gate,
   and the `12.892 ms` composite deficit does not isolate projection arithmetic.
2. Keep field-interleaved-v4 and the current Q4_K x8/Q6_K AVX-512 x16 routes as production baselines.
3. Promote an FFN activation/Down attribution tranche to P0:
   - split reference SwiGLU from Q4_K/Q6_K Down using non-synchronizing aggregate counters;
   - prove and repair production imported-artifact fusion eligibility;
   - benchmark scalar, exact SIMD, and explicitly bounded fast-math SwiGLU candidates;
   - retain only a candidate that improves the exact-token full model.
4. Do not reopen Down prefetch, x16 reuse, dual-output streams, or Q6 AVX2 selection without new PMU or reference-split
   evidence that specifically selects one of those mechanisms.

## Acceptance Gates

### Attribution and Artifact Gates

- Reference clean/instrumented overhead at most `3%`, coverage `95-102%`, whole-run CV at most `3%`, and every promoted
  activation/Down substage CV at most `15%`.
- Real imported Qwen artifact, not only a synthetic graph, must import the fused helper when field-interleaved-v4 is
  the selected prepared layout and must omit it for incompatible/public/multi-consumer cases.
- No interpreter or fallback execution; generated token ids and decoded text must match the baseline.

### SwiGLU Candidate Gates

- Preserve finite, signed-zero, NaN, and infinity behavior under a separately named exact path. Any approximate path
  must be explicit in compiler/runtime options and report its numerical contract.
- Pass focused scalar-versus-SIMD tests over adversarial and representative gate ranges for contiguous and strided
  tensors, plus Q4_K/Q6_K fused-helper parity and loaded-AOT execution.
- Improve the production-shaped 48-layer SwiGLU sequence by at least `2x` or save at least `5 ms/token` before a
  full-model run.
- Improve alternating cache-hit exact-token full-decode median by at least `3%` across three pairs, with no pair showing
  a correctness or fallback regression.
- Re-run the accepted reference split; final FFN activation + Down latency must be within `10%` and whole-token latency
  within `5%` of the adjacent Clang/no-OpenMP reference.

## Evidence Owners

- Cross-runtime and stage gates: `QwenCPUDecodeStageControl_2026-08-04.md`.
- Earlier projection and activation-preparation history: `QwenCPUDecodeProjectionProfile_2026-08-04.md`.
- Cache-cold benchmark and first fusion result: `PerformanceAnalysis_2026-08-04.md`.
- Ordered implementation checklist: `PerformanceOptimizationRoadmap.md`.
