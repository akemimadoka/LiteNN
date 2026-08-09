# Qwen CPU Decode Sustained 128-Token Control - 2026-08-09

## Scope

This report records the first accepted sustained CPU decode comparison between LiteNN CPU AOT and the strongest
locally reproduced CPU-only llama.cpp control. It extends the earlier 15-token stage campaign to 128 generated tokens,
keeps both runtimes on the same token trajectory, and separates throughput evidence from natural greedy-output drift.

The fixed-cost stage evidence remains in `docs/QwenCPUDecodeStageComparison_2026-08-09.md`. This report owns the
context-growth and sustained-throughput decisions derived from the longer run.

## Measurement Boundary

- Workload: Qwen2.5-Coder-14B-Instruct Q4_K_M, `8,988,110,272`-byte GGUF.
- Host: AMD Ryzen 9 9950X, 32 logical CPUs, Windows.
- Reference: llama.cpp `b81c2cdd7`, Clang 22.1.8 Release, native ISA, OpenMP/CUDA/BLAS disabled.
- Runtime policy: llama.cpp T2; LiteNN CPU AOT T8, adaptive wait, strict activation math, all eligible weights in
  field-interleaved-v4 layout, LLVM optimization level 0.
- Prompt: 9 chat-template tokens. The run generated 128 tokens and compared the 127 recurrent decode evaluations after
  the prompt-produced first token.
- Controls: three alternating adjacent pairs, required LiteNN AOT-cache hit, no fallback, 3% per-runtime CV gate, and
  sampled actual frequency.
- Private model and executable paths are omitted. Raw artifacts remain under ignored `build` directories.

## Why Fixed Replay Is Required

Natural greedy generation matched for the first 23 generated tokens, then diverged at zero-based generated index 23.
Once the next token differs, all later contexts differ, so a 128-token natural-output timing comparison no longer
measures equivalent work.

The paired runner now performs these steps:

1. Capture one unmeasured llama.cpp greedy trajectory.
2. Recover and hash its 128 token IDs, preserving UTF-8 and undoing Windows stdout CRLF translation.
3. Let LiteNN compute its natural argmax at every step, but feed the captured reference token into the next step.
4. Require exact generated-token identity for the forced trajectory while reporting natural argmax mismatches
   separately.

The first Windows capture exposed a real text-boundary issue: stdout CRLF expansion changed the round-trip token count
from 128 to 129. Normalizing the platform newline back to model LF bytes restores exactly 128 tokens. The runner has
unit coverage for generated-token suffix extraction, replay diagnostics, token identity, and newline normalization; the
adapter's UTF-8 file transport was also verified with a non-ASCII round trip.

## Paired Throughput

All three pairs passed forced-trajectory identity, no-fallback, cache-hit, and variance gates.

| Pair | Order | llama.cpp ms/token | llama.cpp token/s | LiteNN ms/token | LiteNN token/s | LiteNN delta |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | llama.cpp -> LiteNN | `174.000` | `5.750` | `212.474` | `4.706` | `-18.15%` |
| 2 | LiteNN -> llama.cpp | `178.410` | `5.610` | `209.742` | `4.768` | `-15.01%` |
| 3 | llama.cpp -> LiteNN | `175.420` | `5.700` | `204.450` | `4.891` | `-14.19%` |

| Aggregate | llama.cpp | LiteNN |
| --- | ---: | ---: |
| Median throughput | `5.700 token/s` | `4.768 token/s` |
| Throughput CV | `1.25%` | `1.96%` |
| Throughput range | `5.610-5.750 token/s` | `4.706-4.891 token/s` |

The preferred median adjacent-pair difference is `-15.01%`. The independent medians differ by about `34.3 ms/token`.
Sampled weighted actual frequencies were `4995-5151 MHz` for llama.cpp and `5159-5206 MHz` for LiteNN, so a lower
LiteNN clock does not explain the deficit. One reference run observed a high-performance-to-balanced power-policy
transition before process exit; the throughput CV still passed, but future acceptance tooling should reject an
in-process policy transition rather than only recording it.

The earlier 15-token control placed LiteNN `5.49%` behind. The two campaigns use different generated trajectories and
power-policy observations, so their totals must not be subtracted as if they were one paired experiment. They do,
however, establish that the short-window result is not a sufficient sustained-throughput claim.

## LiteNN Position Growth

LiteNN stream statistics show that almost all per-step time is inside the compiled module. Sampling averaged only
`0.13-0.17 ms`; input preparation, state update, logits publication, and other host overhead were each near zero at the
reported precision.

The following bins aggregate all three runs. They include all 128 generation invocations so the position trend is
visible; the paired whole-runtime number above uses the aligned 127 recurrent-decode boundary.

| Generated positions | Samples | Mean step ms | Mean module ms | Mean sampling ms |
| --- | ---: | ---: | ---: | ---: |
| 1-16 | 48 | `198.210` | `198.080` | `0.130` |
| 17-48 | 96 | `203.280` | `203.120` | `0.150` |
| 49-80 | 96 | `207.710` | `207.550` | `0.160` |
| 81-112 | 96 | `215.780` | `215.610` | `0.170` |
| 113-128 | 48 | `221.200` | `221.030` | `0.160` |

The last 16-token module mean is `22.95 ms` (`11.6%`) above the first 16-token mean. Per-run linear fits measured
`0.27`, `0.20`, and `0.15 ms` of additional module latency per generated position. These fits are descriptive, not an
attention attribution: cache state, host variation, and all capacity-sensitive nodes are included. A matched
position-binned stage profile is required before selecting an attention or KV implementation.

## Natural Greedy Drift

Forced replay does not hide natural-output behavior. Every run reported exactly 14 natural argmax mismatches, with the
first mismatch at generated index 23.

At that first divergence:

- llama.cpp ranked token 498 first at `20.9231` and token 3151 second at `20.4133`, a `0.5099` margin;
- LiteNN ranked token 3151 first at `19.0398`, token 358 second at `18.8532`, and token 498 third at `18.2418`;
- across the full vocabulary, LiteNN versus reference measured MAE `0.4944`, RMSE `0.6117`, maximum absolute delta
  `3.4293`, and cosine similarity `0.986888`.

Two controls narrow ownership:

- Stateful and functional LiteNN decode produced bit-identical logits at the divergence, excluding in-place KV/state
  aliasing as the cause.
- Source-layout and field-interleaved-v4 logits differed by only MAE `0.05848` with cosine `0.999616`; the source path
  was slightly farther from the reference. Weight packing is therefore not the primary owner.

The drift does not grow monotonically over the inspected prefix, and a low-margin top-two flip is compatible with
different floating-point/quantized reduction orders. It still requires a layer-localized comparison before LiteNN can
claim reference-level model fidelity. Exact natural-token parity is the wrong sustained performance gate; fixed token
replay is the performance gate, while logit/perplexity/model-quality checks must own numerical acceptance.

## Decisions

1. The accepted sustained CPU gap is `15.01%`, not the earlier short-window `5.49%`.
2. Bounded vector SwiGLU remains a confirmed fixed-cost P0: its measured `10.833 ms/token` deficit is real. At the
   128-token median it can explain only about one third of the `31-34 ms/token` gap, so it cannot close sustained decode
   by itself.
3. The additional P0 is the context-dependent compiled-module slope. The next profile must compare position bins and
   split Attention into projection, RoPE/KV append, score-softmax-value, and output projection before choosing a kernel.
4. Sampling and Python/CLI orchestration are not steady-state optimization owners; they are outside the aligned module
   timing or below `0.2 ms/token`.
5. Natural logit drift is a correctness workstream, not a reason to compare different generated trajectories. Add
   layer checkpoints and model-quality metrics rather than requiring long exact greedy text.
6. A 512-token run is premature until process-memory/power-policy gates and position-binned attribution are available;
   otherwise it would confirm growth without identifying its owner.

## Ordered Follow-Up

- [ ] P0 fixed cost: select the bounded vector-math provider, implement standalone/fused SwiGLU, and pass the existing
  numerical, artifact, microbenchmark, and full-model promotion gates.
- [ ] P0 growth cost: add low-overhead position-binned counters to both runtimes and compare 1-16, 17-48, 49-80,
  81-112, and 113-128 over the same forced trajectory.
- [ ] P0 growth cost: split the context-sensitive path into QKV projection, RoPE/KV append, score-softmax-value,
  attention output, FFN, logits, and residual; require counter coverage and instrumentation overhead gates before
  selecting work.
- [ ] P0 correctness: add per-layer hidden-state checkpoints at prompt end, first divergence, and final position, then
  localize the first material numerical drift independently of stateful/functional and source/prepacked controls.
- [ ] P1 harness: reject power-policy changes during a measured process and capture peak working set/private bytes,
  mapped weight/artifact residency, and KV-cache bytes without adding a mandatory third-party dependency.
- [ ] P1 validation: rerun 512 fixed tokens only after the above gates pass; then extend to paged 2K/32K/128K/1M
  context tiers.
- [ ] P1 quality: add corpus perplexity/logit-distribution and task-level output checks so acceptable kernel-order
  differences are evaluated by model quality rather than exact long greedy text.
