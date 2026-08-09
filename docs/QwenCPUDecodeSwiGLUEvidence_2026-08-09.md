# Qwen CPU Decode SwiGLU Evidence - 2026-08-09

## Scope

This report isolates the remaining Qwen CPU decode SwiGLU work after the 2026-08-04 activation/Down decision. It
records accepted measurements, resolves why the profiled artifact still called standalone SwiGLU, and defines the
next implementation gate. The ordered checklist lives in `PerformanceOptimizationRoadmap.md`.

The workload is the 48-layer Qwen2.5-Coder-14B decode shape with FFN width 13824. Private model and executable paths
are intentionally omitted. All standalone rows were built in Release mode and use the production
`litenn_cpu_swiglu_f32` helper with its strict scalar `std::exp` implementation.

## Existing Cross-Runtime Control

The accepted low-overhead comparison remains the selection boundary:

| Stage | LiteNN | Controlled CPU-only reference | Difference |
| --- | ---: | ---: | ---: |
| Attention | `36.226 ms` | `34.169 ms` | at least `+2.057 ms` |
| FFN Gate/Up | `65.048 ms` | `67.038 ms` | no LiteNN deficit established |
| FFN activation + Down | `54.057 ms` | `41.165 ms` | at least `+12.892 ms` |
| Final logits | `10.995 ms` | `10.943 ms` | `+0.052 ms` |

The strongest controlled reference clean median was `155.541 ms/token`, and the paired control placed LiteNN `5.49%`
behind it. Reference instrumentation overhead was `-0.21%`, aggregate coverage was `98.75%`, whole-run CV was below
`0.4%`, and reference stage CV was `0.16-0.98%`.

LiteNN's helper decomposition measured Q4_K/Q6_K Down projections at `42.723 ms/token` and standalone SwiGLU at
`11.043 ms/token`. The reference still combines activation and Down, so these numbers do not prove how much of the
`12.892 ms` composite deficit belongs to either component.

## Artifact Root Cause

The previous profile imported `litenn_cpu_swiglu_f32` 48 times per token and never imported the fused
field-interleaved-v4 SwiGLU+Down helper. The earlier source audit proposed that source-layout gating might block
fusion when field-interleaved-v4 is selected only during AOT preparation. A fresh importer regression disproved that
hypothesis.

The actual sequence was:

1. CPU AOT cache format version 2 was introduced before SwiGLU+Down AOT fusion existed.
2. The fusion lowering was added without changing that cache version.
3. Existing Qwen cache entries therefore remained valid hits and continued loading their older standalone-SwiGLU
   object code.

CPU AOT cache version 3 now invalidates those stale objects. A compile-time regression prevents this feature boundary
from returning to a pre-fusion cache version.

The new imported-model regression exercises a mixed Q4_K/Q6_K LLaMA archive rather than a hand-built fusion graph. It
proves all of the following in one fresh compile:

- the executable plan contains seven output-major quantized projections whose source storage layout is still
  `Source`;
- AOT preparation emits seven field-interleaved-v4 external weights;
- the generated object imports the fused SwiGLU+Down helper and does not import standalone `litenn_cpu_swiglu_f32`;
- the loaded artifact executes and returns finite public logits.

That regression initially exposed a separate `FusionPass` remap failure for `GroupedQuantizedMatMulNode`. Consumer
accounting and node remapping now support grouped quantized projections, and the focused FusionPass and GGUF importer
suites pass.

## Standalone Production-Shape Benchmark

`SwiGLUF32ProductionSequence` evaluates distinct gate/up tensors so the 48-call row represents a layer sequence rather
than repeatedly touching one small hot buffer. Every row checks a scalar reference, signed zero, infinities, and NaN.
Five aggregate repetitions produced:

| Layout | Calls | Width | Mean | Median | Mean throughput | CV | Max abs/relative error | Special mismatches |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Contiguous | 1 | 13824 | `0.253 ms` | `0.234 ms` | `55.14 M elements/s` | `11.70%` | `0 / 0` | `0` |
| Stride 2 | 1 | 13824 | `0.277 ms` | `0.274 ms` | `49.66 M elements/s` | `4.25%` | `0 / 0` | `0` |
| Contiguous | 48 | 13824 | `13.3 ms` | `13.5 ms` | `50.20 M elements/s` | `2.81%` | `0 / 0` | `0` |
| Stride 2 | 48 | 13824 | `13.5 ms` | `13.4 ms` | `49.09 M elements/s` | `5.79%` | `0 / 0` | `0` |

The 48-call contiguous row is within `2.26 ms` of the real helper profile's `11.043 ms/token`, so the standalone
benchmark reproduces the correct cost class. The real run can be faster because its activation distribution, cache
state, and surrounding helper schedule differ from independently randomized benchmark streams.

## Conclusions

1. Strict scalar exponential math is the dominant independently isolated SwiGLU cost. Moving from contiguous to
   stride-2 inputs changes the 48-call mean by only about `1.5%`; layout and call dispatch are not the primary owner.
2. Imported-plan fusion eligibility is working. The observed standalone calls came from a stale pre-fusion AOT cache,
   not from source-layout gating. Future code-generation changes that alter imported helpers must bump the AOT cache
   version or encode a stronger feature identity.
3. Existing SwiGLU+Down fusion does not remove sigmoid math. Earlier materialized-versus-fused cold-stream pairs were
   neutral within noise, so regenerating the artifact is a correctness/structure fix rather than a predicted 11 ms
   speedup.
4. A vector-math implementation is the highest-value bounded LiteNN-side experiment. It must first save at least
   `5 ms/token` or deliver `2x` on the 48-call row before consuming full-model benchmark time.
5. No additional Down-kernel rewrite is selected. The controlled reference must still split activation from Down, or
   PMU evidence must identify a specific projection mechanism, before that work can return to P0.

## Next Decision Gates

- Split reference activation and Down with the already accepted non-synchronizing aggregate method.
- Inventory available exact vector exponential support and benchmark it without changing strict special-value
  behavior.
- If exact SIMD cannot clear the gate, define a separately named bounded fast-exp policy with explicit error,
  saturation, and special-value contracts; never make approximation an implicit compiler side effect.
- For a candidate that clears the standalone gate, regenerate a cache-version-3 Qwen artifact and run three
  alternating exact-token pairs. Retention requires at least `3%` median full-token gain, unchanged tokens/text, and no
  fallback.

## Reproduction

```powershell
cmd /d /c "cmake --build build-release --target litenn_bench --parallel"
build-release\benchmark\litenn_bench.exe `
  --benchmark_filter='SwiGLUF32ProductionSequence.*' `
  --benchmark_min_time=0.5s `
  --benchmark_repetitions=5 `
  --benchmark_report_aggregates_only=true `
  --benchmark_out=build\swiglu_strict_baseline_20260809.json `
  --benchmark_out_format=json
```

Focused artifact validation:

```powershell
build-release\tests\GGUFImporterTest.exe `
  --gtest_filter=GGUFLLaMAQuantizedExecution.ImportedMixedKQuantizedPlanEmitsAndRunsFreshSwiGLUDownFusion
build-release\tests\CompiledModuleTest.exe
```
