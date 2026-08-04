# Qwen CPU Decode Stage Control - 2026-08-04

This report owns the low-overhead stage-attribution experiment that follows
`QwenCPUDecodeBuildControl_2026-08-04.md`. It records accepted end-to-end facts, rejected stage measurements, and the
resulting implementation priorities separately so that noisy callback data cannot select a production optimization.

## Scope

- Host: AMD Ryzen 9 9950X, Windows, balanced power policy.
- Workload: CPU-only Qwen2.5-Coder-14B Q4_K_M generated-token decode, two threads, flash attention disabled, F16 KV,
  mmap enabled, 9 warmup calls and 15 measured calls.
- References: Clang 22.1.8/no-OpenMP and GNU 16.1/no-OpenMP llama.cpp builds from the same source commit.
- LiteNN: cache-hit CPU AOT, field-interleaved-v4 prepared weights, structured helper and worker profiling enabled.
- Privacy: model and executable paths are redacted from repository artifacts. Raw run output remains under ignored
  build directories.

The out-of-tree profiler is `benchmark/llama_cpp_stage_profile`; the paired driver is
`benchmark/run_llama_cpp_stage_control.py`. Neither links llama.cpp into a LiteNN production target.

## Measurement Design

Three callback strategies were evaluated:

1. Coarse/FFN callbacks synchronized at every layer boundary. They added `21.14-29.74%` to full-token latency and were
   rejected by the perturbation gate.
2. Selected-layer callbacks synchronized six boundaries in one layer. Full-token overhead fell to `0.10-3.01%`, but
   each reported local interval still included a synchronization boundary. Extrapolating the intervals exceeded the
   uninstrumented token total, so these values were rejected as locally biased.
3. Cumulative cut scans created a separate context for each of five cuts and synchronized only once per token. A stage
   is the difference between adjacent cumulative prefixes. Full-token drift stayed low, but subtracting two roughly
   `20 ms` prefixes to recover a roughly `1 ms` stage amplified run-to-run noise.

The runner requires three alternating repetitions, at most `3%` total-latency CV, at most `15%` absolute profile
drift, and at most `15%` CV for every derived stage. The cumulative scan failed the total/stage variance gates and is
therefore diagnostic evidence only.

## llama.cpp Results

| Build | Scan | Baseline ms/token | Baseline CV | Scan baseline ms/token | Scan CV | Drift | Actual MHz |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Clang/no-OpenMP | layer 4 | 165.840 | 1.99% | 166.330 | 3.67% | +0.26% | 5125 |
| Clang/no-OpenMP | layer 5 | 166.160 | 3.35% | 165.630 | 1.59% | -0.08% | 5124 |
| GNU/no-OpenMP | layer 4 | 168.740 | 2.91% | 169.080 | 0.66% | +0.20% | 5114 |
| GNU/no-OpenMP | layer 5 | 168.030 | 1.16% | 171.930 | 2.88% | +2.32% | 5107 |

The total-latency rows reproduce the build-control ordering: Clang/no-OpenMP remains the strongest local reference at
about `166 ms/token`, and measured frequency does not explain the compiler difference.

The following values are shown to document why the stage gate failed. They must not be used as kernel acceptance data.

| Build / layer | Attention ms (CV) | Gate ms (CV) | Up ms (CV) | Activation + Down ms (CV) |
| --- | ---: | ---: | ---: | ---: |
| Clang / 4 | 0.823 (15.84%) | 0.698 (82.51%) | 0.798 (12.78%) | 1.191 (30.21%) |
| Clang / 5 | 0.612 (38.23%) | 0.853 (52.96%) | 1.098 (53.77%) | 0.974 (60.57%) |
| GNU / 4 | 0.837 (17.57%) | 0.795 (22.20%) | 1.090 (51.31%) | 1.477 (41.53%) |
| GNU / 5 | 0.826 (10.53%) | 0.751 (34.26%) | 1.295 (58.20%) | 1.487 (36.40%) |

The stage CV range is `10.53-82.51%`. The compiler-to-compiler stage sums also move much more than the corresponding
whole-token baselines. Both observations reject fine callback differencing as the next attribution mechanism.

## Accepted Non-Synchronizing Aggregate Control

The replacement profiler applies a benchmark-only patch to a detached llama.cpp worktree. Thread 0 records coarse
stage transitions after ggml's existing node barriers; it neither uses `cb_eval` nor adds a scheduler synchronization.
The strict overhead control compares a completely clean profiler binary with the instrumented binary, so the result
includes both enabled-counter cost and any static code-layout or atomic-read effect.

The final run replayed the exact nine-token Qwen chat prompt as one prefill and then the 15 generated tokens that
correspond to the actual-completion eval window. A current exact-replay sweep selected T8 for this profiler workload;
single-run clean baselines at T2/T4/T8/T16 were `182.207/170.301/163.729/169.525 ms/token`. This supersedes the old
repeated-token T2 choice for stage attribution, but not the separately accepted actual-completion throughput result.

| Gate metric | Result | Gate |
| --- | ---: | ---: |
| Clean baseline median | `155.541 ms/token` | - |
| Instrumented median | `154.745 ms/token` | - |
| Paired measurement overhead median | `-0.21%` | absolute value at most `3%` |
| Aggregate stage coverage | `98.75%` | `95-102%` |
| Clean/instrumented whole-run CV | `0.23% / 0.39%` | at most `3%` |
| Stage CV range | `0.16-0.98%` | at most `15%` |

All gates passed. The normalized reference stages are therefore accepted for ranking:

| Reference stage | Median ms/token | CV | Calls/token |
| --- | ---: | ---: | ---: |
| Attention | `34.169` | `0.37%` | 49 segments |
| FFN Gate/Up | `67.038` | `0.20%` | 48 segments |
| FFN activation + Down | `41.165` | `0.16%` | 48 segments |
| Final logits | `10.943` | `0.98%` | 1 segment |

The same generated-token positions 10-24 in LiteNN's structured T8 cache-hit profile have the following helper-only
lower bounds. Attention includes QKV, output projection, active-prefix attention, RoPE, and KV update. Down includes
both Q4_K/Q6_K Down projections and SwiGLU. Inline normalization, residual, and view work remains outside these rows,
so a positive LiteNN difference is a conservative deficit rather than timer-boundary inflation.

| Matched stage | LiteNN helper lower bound | LiteNN CV | Reference full stage | LiteNN difference |
| --- | ---: | ---: | ---: | ---: |
| Attention | `36.226 ms` | `4.71%` | `34.169 ms` | at least `+2.057 ms` (`+6.02%`) |
| FFN Gate/Up | `65.048 ms` | `3.33%` | `67.038 ms` | no deficit established |
| FFN activation + Down | `54.057 ms` | `2.97%` | `41.165 ms` | at least `+12.892 ms` (`+31.32%`) |
| Final logits | `10.995 ms` | `4.67%` | `10.943 ms` | `+0.052 ms` (`+0.48%`) |

FFN activation + Down is the only double-digit accepted deficit and is about 6.3 times the Attention difference. It
also explains at least 63% of the approximately `20.5 ms` difference between the profiled module and clean reference
medians. Down kernel/cache-stream scheduling is therefore promoted to the next CPU P0. Gate/Up, logits, another broad
ABI rewrite, and dispatch-only work remain closed until a later accepted profile changes the ranking.

## LiteNN Accounting

The following medians are from stable generated-token steps 10-24 of one structured cache-hit accounting run. This run
is suitable for partitioning LiteNN's own module time, but it is not an adjacent cross-runtime throughput comparison.

| Bucket | Median ms/token | Notes |
| --- | ---: | --- |
| CPU AOT module | 176.063 | Complete compiled module call |
| Timed helpers | 164.155 | 2690 helper calls per token |
| Module non-helper residual | 11.408 | Inline AOT work, untimed operations, and instrumentation residual |
| Parallel wall | 66.225 | Included in helper time |
| Worker dispatch | 10.365 | 97 parallel projection calls; included in parallel wall/helper time |
| Final barrier wait | 3.263 | Included in parallel wall/helper time |
| Activation quantization | 1.183 | No longer a P0 owner |

| Helper group | Median ms/token | Calls/token |
| --- | ---: | ---: |
| Grouped Q4_K Gate/Up | 65.048 | 48 |
| Q6_K Down | 25.602 | 24 |
| Q4_K Down | 17.121 | 24 |
| Q4_K hidden/output | 13.977 | 48 |
| SwiGLU | 11.043 | 48 |
| Q6_K logits | 10.995 | 1 |
| Mixed Q6_K QKV | 10.228 | 24 |
| Q4_K QKV | 9.439 | 24 |
| Active-prefix attention | 2.372 | 48 |

The `11.408 ms` non-helper residual is large enough to explain the roughly `9-10 ms/token` absolute deficit implied by
the accepted `-5.49%` paired result against the approximately `166 ms/token` Clang reference. This is a size comparison,
not proof that all residual work is removable: the bucket includes generated inline normalization, bias/residual/view
and state operations as well as instrumentation error.

Worker dispatch is similarly material at `10.365 ms/token`, or about `0.107 ms` per parallel projection call, but it is
already inside helper time. Eliminating it would improve the helper total; it must not be added again to the non-helper
residual. The earlier `atomic::wait` experiment reduced dispatch but regressed parallel wall time, so queue wake-up
tuning alone is not justified.

## Conclusions

1. The accepted end-to-end conclusion remains LiteNN `5.49%` behind the strongest paired Clang/no-OpenMP reference.
   This experiment reproduces the reference near `166 ms/token` and again rules out frequency as the explanation.
2. Fine llama.cpp callback timing remains rejected, but the non-synchronizing aggregate counters pass all overhead,
   coverage, whole-run variance, and stage-variance gates.
3. FFN activation + Down is now the accepted largest cross-runtime deficit: LiteNN's helper-only lower bound is
   `54.057 ms/token` versus the reference's complete `41.165 ms/token` stage.
4. The highest-value confirmed unknown is LiteNN's `11.408 ms/token` module non-helper residual. It must be reconciled
   into low-cardinality operation groups before selecting a generated-code optimization.
5. The next independent candidate is dispatch amortization across the 97 parallel projection calls. It should use a
   persistent sequence/batch contract, not another worker wake-up primitive, and must reduce whole-token latency rather
   than only the dispatch counter.
6. Future llama.cpp stage attribution should use the repository-owned aggregate-counter worktree. The cumulative-cut
   facility remains useful only as a rejection gate and coarse directional diagnostic.

## Acceptance Gates for the Next Tranche

- Residual accounting overhead at or below `3%`, with categorized time plus helper time reconciling to module time
  within `2%`.
- No fallback, byte-identical generated text, cache hit, and the same 9-prompt/15-eval window.
- Dispatch batching must reduce dispatch by at least `50%`, full-token latency by at least `3%`, and must not increase
  parallel wall or barrier time.
- Down kernel experiments may now proceed because the matched stage deficit and both runtime stage CVs pass the `15%`
  gate. Retention still requires cache-cold and complete-decode improvement, not a cache-hot-only win.
- Final closure requires two alternating paired batches within `5%` of the same-run Clang/no-OpenMP median.
