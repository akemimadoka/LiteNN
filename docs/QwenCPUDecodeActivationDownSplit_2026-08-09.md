# Qwen CPU Decode Activation/Down Stage Split - 2026-08-09

## Scope

This report closes the measurement question left by the accepted 2026-08-04 Qwen CPU decode stage control: whether
LiteNN's largest observed FFN deficit belongs to SwiGLU activation or to the Q4_K/Q6_K Down projections. It records a
fresh low-overhead reference split and combines it with the existing LiteNN helper accounting only at matching stage
boundaries. Raw process artifacts remain in the ignored `build/qwen_stage_activation_control_20260809` directory.

No production runtime links to llama.cpp. The reference instrumentation is a benchmark-only patch applied to a
detached worktree.

## Measurement Identity

- Workload: Qwen2.5-Coder-14B-Instruct Q4_K_M, `8,988,110,272`-byte GGUF.
- Host: AMD Ryzen 9 9950X, Windows, balanced power policy.
- Reference source revision: `b81c2cdd748dc2704d5989cf03936325554c12d3`.
- Build: Release, Clang `22.1.8`, native ISA enabled, OpenMP disabled, identical clean/profile configuration.
- Clean binary SHA-256: `7b6de6e50524aebe98568563021a686419ae7bbf89080b740f6352c8d406e43d`.
- Instrumented binary SHA-256: `79e857a4b3209e944fe2009cba5dabfd6dfb0a76694ed7c21748b2904def0043`.
- Decode policy: 8 threads, 9-token exact prefill, 15 fixed decode tokens, no warmup, three alternating clean/profile
  pairs.
- Prefill token digest: `c283080bdc6c1a7c05bb55f4d175d15bf4876f4e2cb63ecca6d50361b7e46917`.
- Decode token digest: `f9b59f675fe201894801db31226c6058c6f3b2cceac6df2e7bafab8ddfd7edde`.

The aggregate counter records transitions on worker thread 0 after ggml's existing graph-node barriers. Separating
activation from Down adds no new scheduler synchronization: activation closes immediately before the named
`ffn_down` MatMul begins.

## Acceptance Gate

| Gate | Measured | Required | Result |
| --- | ---: | ---: | --- |
| Clean median | `165.299 ms/token` | informational | pass |
| Clean whole-run CV | `2.66%` | at most `3%` | pass |
| Instrumented median | `164.495 ms/token` | informational | pass |
| Instrumented whole-run CV | `0.25%` | at most `3%` | pass |
| Median instrumentation overhead | `-0.79%` | at most `3%` | pass |
| Median stage coverage | `98.44%` | `95-102%` | pass |
| Maximum stage CV | `5.61%` | at most `15%` | pass |
| Clean/profile actual frequency | `4965/4962 MHz` | no material skew | pass |

All gates passed. The negative measured overhead is noise, not a speedup claim; its magnitude confirms that the
counter does not perturb the workload enough to invalidate stage ranking.

## Reference Stage Results

Values are normalized to the paired clean runtime before aggregation.

| Reference stage | Median ms/token | CV | Calls/token |
| --- | ---: | ---: | ---: |
| Attention | `36.846` | `2.87%` | 49 |
| FFN Gate/Up | `70.916` | `2.52%` | 48 |
| FFN activation | `0.210` | `5.61%` | 48 |
| FFN Down | `43.045` | `2.69%` | 48 |
| Final logits | `11.920` | `1.73%` | 1 |

The activation boundary is therefore real and stable enough for ranking. It accounts for only about `0.13%` of the
reference token time, while Down accounts for about `26%`.

## Bounded Cross-Runtime Attribution

The matching LiteNN values come from stable generated-token steps 10-24 of the structured cache-hit helper profile:
standalone SwiGLU is `11.043 ms/token`, and the Q4_K plus Q6_K Down projections sum to `42.723 ms/token`. Those helper
timers are lower bounds for their containing generated stages, but their activation and projection boundaries match
the new reference split.

| Component | LiteNN helper lower bound | Reference full stage | Difference | Ratio |
| --- | ---: | ---: | ---: | ---: |
| SwiGLU activation | `11.043 ms` | `0.210 ms` | `+10.833 ms` | `52.5x` |
| Q4_K/Q6_K Down | `42.723 ms` | `43.045 ms` | `-0.322 ms` | `0.993x` |
| Activation + Down helper sum | `53.766 ms` | `43.255 ms` | `+10.511 ms` | `1.243x` |

The Down difference is smaller than run-to-run variation and does not establish a LiteNN Down deficit. By contrast,
the activation difference alone is about `84%` of the previously accepted `12.892 ms` combined-stage gap. Distinct
profile campaigns cannot be added as if they were one adjacent end-to-end pair, so `10.833 ms` is an attribution
signal and optimization ceiling, not a promised whole-token gain.

Independent production-shape evidence points to the same owner. For 48 calls at width 13824, LiteNN's strict scalar
`std::exp` path measured `15.8 ms`, while the already vendored bounded GGML implementation measured `0.526 ms`, about
`30x` faster with maximum observed absolute/relative error `9.54e-7/3.47e-7` and no signed-zero, infinity, or NaN
mismatches. Release disassembly also shows one scalar `expf` call per LiteNN lane and no vector body.

## Conclusions

1. The accepted FFN activation/Down gap is activation-owned. SwiGLU is the only measured component with a
   double-digit LiteNN deficit; the Down projection is already at reference scale.
2. Additional Q4_K/Q6_K Down rewrites are removed from CPU P0 until a new accepted profile identifies a specific
   projection deficit. Existing rejected x16, prefetch, and stream variants remain closed.
3. The highest-value next implementation is an explicit bounded vector activation-math policy used by both standalone
   and fused SwiGLU. Strict scalar math remains the default/reference behavior.
4. Vector-math ownership must be decided before production integration. The preferred dependency option is a
   maintained cross-platform provider such as SLEEF; the alternative is an attributed compact kernel adapted from the
   pinned GGML implementation. Linking the complete GGML runtime for one primitive is not acceptable.
5. A candidate must first save at least `5 ms/token` or reach `2x` in the 48-call production-shape benchmark. It is
   retained only after three alternating exact-token Qwen cache-hit pairs show at least `3%` median full-token gain,
   identical token ids/text, no fallback, the intended helper import, and no regression beyond the accepted reference
   stage envelope.

## Reproduction

Prepare and build the benchmark-only instrumented reference as documented in
`benchmark/llama_cpp_stage_profile/README.md`, then run:

```powershell
python311 benchmark\run_llama_cpp_stage_control.py `
  --model <model.gguf> `
  --baseline-binary clang-noopenmp=<clean-profiler.exe> `
  --binary clang-noopenmp=<instrumented-profiler.exe> `
  --mode aggregate --threads 8 --warmup 0 --steps 15 --repetitions 3 `
  --prefill-token-ids <nine-fixed-token-ids> `
  --decode-token-ids <fifteen-fixed-token-ids> `
  --overhead-threshold-percent 3 --stage-variance-threshold-percent 15 `
  --output-json build\qwen_stage_activation_control_20260809\control.json `
  --output-md build\qwen_stage_activation_control_20260809\control.md
```
