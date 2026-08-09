# Qwen CPU Decode Position-Binned Stage Control - 2026-08-09

## Scope

This report records the first position-binned stage campaign for the strongest locally reproduced CPU-only llama.cpp
reference. It uses the same 128-token fixed trajectory as
`docs/QwenCPUDecodeSustained128Control_2026-08-09.md` and asks a narrower question: does the reference runtime show a
measurable context-sensitive stage increase over positions 1-128 that can explain LiteNN's accepted `22.95 ms`
compiled-module increase?

The campaign also validates the new benchmark-only per-step stage counters and the in-process power-policy gate. The
result is diagnostic rather than accepted: stage coverage is strong, but clean/profile overhead and several
position-bin variance gates narrowly or materially fail. Raw artifacts remain under ignored `build` directories, and
private model and executable paths are omitted.

## Measurement Boundary

- Workload: Qwen2.5-Coder-14B-Instruct Q4_K_M, `8,988,110,272`-byte GGUF.
- Host: AMD Ryzen 9 9950X, 32 logical CPUs, Windows high-performance power policy.
- Reference: llama.cpp `b81c2cdd7`, Clang 22.1.8 Release, native ISA, OpenMP/CUDA/BLAS disabled.
- Runtime: two CPU threads, 9 fixed prefill tokens, 128 fixed decode tokens, no warmup exclusion.
- Pairing: three alternating adjacent clean/instrumented process pairs.
- Bins: `1-16`, `17-48`, `49-80`, `81-112`, and `113-128`.
- Instrumentation: benchmark-only non-synchronizing counters snapshot after each `llama_decode`, outside the measured
  decode interval. Each token records Attention, FFN Gate/Up, SwiGLU activation, FFN Down, and logits.
- Gates: whole-runtime and per-bin CV at most `3%`, stage CV at most `15%`, clean/profile overhead within `3%`, stage
  coverage `95-102%`, exact call shape, and no in-process power-policy transition.

Every instrumented token reported exactly `49/48/48/48/1` calls for Attention/Gate-Up/activation/Down/logits. The
power policy remained high-performance before and after every measured process.

## Acceptance Result

The campaign is **not accepted** as an absolute position-stage comparison.

| Gate | Result | Evidence |
| --- | --- | --- |
| Whole-runtime variance | Pass | clean/profile CV `1.16%/1.62%` |
| Whole-stage variance | Pass | maximum stage CV `7.99%` |
| Whole clean/profile overhead | Fail | median `-3.08%`, just outside the `+/-3%` gate |
| Aggregate call shape | Pass | exact `49/48/48/48/1` calls per token |
| Whole-stage coverage | Pass | median `99.53%` |
| Position-bin runtime variance | Fail | clean bin CV reaches `7.06%`; profile bin CV reaches `3.16%` |
| Position-bin stage variance | Fail | the `0.19 ms` activation stage reaches `17.01%` CV in one bin |
| Position-bin clean/profile overhead | Fail | median bin overhead reaches `-4.20%`; pair extrema reach `-10.44%` |
| Position-bin stage coverage | Pass | median coverage remains within the configured range |
| Power-policy stability | Pass | all processes remain on high-performance |

The negative overhead does not indicate that counters improve performance. It exposes process-to-process host noise:
individual clean runs contain isolated latency peaks in different bins, while the paired instrumented processes do not
contain the same peaks. Applying a clean/profile normalization factor independently to every bin therefore transfers
that noise into every normalized stage in the bin.

## Whole-Window Stage Data

The following normalized medians are useful for checking aggregate shape and coverage. Because the whole overhead gate
misses by `0.08` percentage points, they are retained as diagnostic values and do not replace the accepted short-window
stage data.

| Stage | Median ms/token | CV |
| --- | ---: | ---: |
| Attention | `45.224` | `2.04%` |
| FFN Gate/Up | `85.613` | `2.65%` |
| SwiGLU activation | `0.189` | `7.99%` |
| FFN Down | `55.890` | `1.54%` |
| Logits | `14.595` | `2.90%` |

The clean and instrumented whole-window medians are `202.315` and `197.511 ms/token`. Instrumented stage coverage is
`99.53%`; the unclassified remainder is therefore below `0.5%` at the median.

## Position-Binned Data

### Clean/Profile Process Timing

| Positions | Clean median ms/token | Profile median ms/token | Median overhead | Coverage |
| --- | ---: | ---: | ---: | ---: |
| 1-16 | `202.521` | `194.213` | `-4.20%` | `99.57%` |
| 17-48 | `202.998` | `196.453` | `-2.90%` | `99.58%` |
| 49-80 | `195.319` | `196.484` | `+0.93%` | `99.59%` |
| 81-112 | `200.660` | `199.664` | `-0.23%` | `99.52%` |
| 113-128 | `197.419` | `199.381` | `+1.70%` | `99.63%` |

The clean series is not monotonic and does not reproduce LiteNN's accepted position trend. Across the three clean
runs, different bins contain isolated increases of roughly `10%`; three repetitions are insufficient to make the
clean/profile bin normalization stable.

### Raw Instrumented Stage Medians

These values are not normalized against a separate clean process. They avoid the noisiest cross-process correction and
are useful for detecting relative movement among stages inside the same instrumented run. They remain diagnostic
because the instrumented process itself has no same-process clean counterfactual.

| Positions | Attention | Gate/Up | Activation | Down | Logits | Profile total |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1-16 | `43.410` | `81.658` | `0.181` | `54.721` | `13.960` | `194.213` |
| 17-48 | `43.887` | `82.803` | `0.184` | `54.884` | `13.907` | `196.453` |
| 49-80 | `44.156` | `82.983` | `0.200` | `55.944` | `14.018` | `196.484` |
| 81-112 | `44.593` | `84.161` | `0.193` | `55.078` | `14.451` | `199.664` |
| 113-128 | `44.643` | `84.253` | `0.190` | `55.412` | `14.444` | `199.381` |

From the first to final bin, the profile total rises `5.17 ms` (`2.66%`). The stage changes are broad rather than
Attention-specific: Attention `+2.84%`, Gate/Up `+3.18%`, activation `+4.58%`, Down `+1.26%`, and logits `+3.46%`.
Gate/Up and logits have no algorithmic dependency on active KV length, so their similar movement is evidence of a
common runtime effect rather than a context-growth owner.

The within-profile Attention/Gate-Up ratio is `0.5316`, `0.5300`, `0.5321`, `0.5299`, and `0.5299` across the five
bins. Attention's share of the five measured stages remains between `22.38%` and `22.47%`. This campaign therefore
provides no evidence that reference Attention grows materially relative to context-independent projections over the
first 128 generated positions.

## Cross-Runtime Interpretation

The accepted LiteNN run measures compiled-module means of `198.080 ms` at positions 1-16 and `221.030 ms` at
positions 113-128: `+22.95 ms` or `+11.6%`. The reference campaign does not pass the absolute bin gates, so subtracting
its first and last medians from LiteNN as though they were paired stage measurements would be invalid.

The relative-stage evidence still changes the next decision:

1. Do not promote a generic Attention or KV rewrite from the LiteNN module slope alone. The reference does not show a
   comparable Attention-share increase over this range.
2. The next P0 attribution must be inside LiteNN. It must separate QKV projection, RoPE/KV append,
   score-softmax-value, attention output projection, Gate/Up, activation, Down, logits, and generated-code residual by
   position.
3. Compare context-sensitive stages to same-process context-independent anchors before applying cross-process
   normalization. A common movement across projections and logits is host drift, not an Attention algorithm signal.
4. The fixed `10.833 ms/token` strict SwiGLU deficit remains independently accepted. This rejected position campaign
   neither weakens nor strengthens that result.
5. A 512-token campaign remains premature. Longer duration would amplify memory and thermal effects without first
   identifying which LiteNN stage owns the accepted 128-token slope.

## Ordered Follow-Up

- [x] Emit benchmark-only per-token reference stage counters and position-bin summaries.
- [x] Reject measured processes whose power policy changes during execution.
- [ ] Add equivalent LiteNN clean/profile position-bin accounting over the fixed trajectory, preserving AOT-cache-hit,
  no-fallback, exact-token, stage-coverage, and overhead gates.
- [ ] Split LiteNN Attention into QKV projection, RoPE/KV append, score-softmax-value, and output projection; keep FFN,
  logits, and residual as same-process anchors.
- [ ] Add a dual absolute/relative variance rule for sub-millisecond stages, then repeat at least five alternating
  reference pairs. Do not relax the whole-token or multi-millisecond stage gates.
- [ ] Capture peak working set/private bytes, mapped weight/artifact residency, and KV-cache bytes before advancing to
  512 fixed tokens.
- [ ] Promote an implementation only when an accepted LiteNN bin profile identifies a stage-specific increase and an
  adjacent full-model A/B passes correctness, no-fallback, cache, variance, and end-to-end latency gates.
