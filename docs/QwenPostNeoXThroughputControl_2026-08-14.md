# Qwen Post-NeoX Throughput Control (2026-08-14)

## Scope

This report attempts the post-NeoX-RoPE normal CPU AOT cache-hit throughput acceptance gate for Qwen2.5-Coder 14B
Q4_K_M. It uses a production logits-only artifact with capacity 256, LLVM O0, T8 adaptive workers, strict activation
math, and field-interleaved-v4/all prepared GGML weights. The reference is the strongest local Clang/no-OpenMP
llama.cpp build at T2. No machine-specific model, cache, or executable path is retained.

All measured batches alternate runtime order, replay one fixed llama.cpp trajectory, require an AOT cache hit, and
retain exact trajectory, natural sampler, fallback, power-policy, frequency, and process evidence. The variance limit
was fixed at 3% per runtime. No failed sample was deleted.

## Artifact Control

The paired harness previously derived LiteNN's cache capacity from `prompt + predict`, which prevented reuse of a
shape-stable production artifact. It now accepts and records an explicit `--litenn-max-cache-length`. A fresh
post-NeoX capacity-256 artifact with the production prepared-weight policy was built in 63.4 seconds. Its probe ran 21
prompt/decode steps without fallback; subsequent campaigns required a cache hit.

The harness also now rejects power-policy changes both within one process and between the two runtimes in a pair. The
first attempted batch exposed why this matters: Windows switched repeatedly between High Performance and Balanced in
pairs 2-4. That five-pair batch is rejected despite exact trajectories and reports LiteNN/reference medians of
`4.263/4.590 token/s`, a `-6.75%` paired median, LiteNN CV `1.88%`, and reference CV `4.10%`.

A fail-fast retry stopped after pair 1 when each runtime crossed the same two policies in opposite directions. Later
batches explicitly selected Balanced before launching the harness and retained that scheme across every process.

## Stable-Power Results

| Batch | Window | Reference median / CV | LiteNN median / CV | Paired median | Gate owner | Result |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| A | 128 tokens | 5.060 t/s / 4.03% | 4.506 t/s / 2.63% | -9.10% | Reference variance | REJECT |
| B | 128 tokens | 4.750 t/s / 2.76% | 4.247 t/s / 3.98% | -9.83% | LiteNN variance | REJECT |
| C | 16 tokens (15 eval) | 4.960 t/s / 1.57% | 4.383 t/s / 4.80% | -11.64% | LiteNN variance | REJECT |

Every stable-power batch has exact fixed-trajectory parity, zero natural sampler mismatches, zero fallback, and stable
power policy. Weighted actual frequency remains around 5.1 GHz and does not explain the rejected variation.

### Batch A Pairs

| Pair | Order | Reference | LiteNN | Paired delta |
| ---: | --- | ---: | ---: | ---: |
| 1 | Reference -> LiteNN | 5.12 | 4.65 | -9.10% |
| 2 | LiteNN -> Reference | 5.06 | 4.36 | -13.81% |
| 3 | Reference -> LiteNN | 5.11 | 4.53 | -11.29% |
| 4 | LiteNN -> Reference | 4.73 | 4.39 | -7.19% |
| 5 | Reference -> LiteNN | 4.74 | 4.51 | -4.94% |

### Batch B Pairs

| Pair | Order | Reference | LiteNN | Paired delta |
| ---: | --- | ---: | ---: | ---: |
| 1 | Reference -> LiteNN | 4.86 | 4.16 | -14.39% |
| 2 | LiteNN -> Reference | 4.73 | 4.44 | -6.17% |
| 3 | Reference -> LiteNN | 4.75 | 4.52 | -4.76% |
| 4 | LiteNN -> Reference | 4.71 | 4.25 | -9.83% |
| 5 | Reference -> LiteNN | 5.03 | 4.14 | -17.71% |

The two 128-token batches move the failing CV owner between runtimes. Batch C makes the reference stable but leaves
LiteNN at `4.80%` CV. This pattern does not support removing a single outlier or blaming clock frequency.

## Conclusions

1. Post-NeoX cache identity, cache-hit execution, exact fixed trajectory, natural sampling, and fallback gates pass.
2. Throughput acceptance remains open because no complete batch satisfies the predeclared 3% variance rule for both
   runtimes. The roadmap item must not be marked complete.
3. Three stable-power medians consistently place LiteNN `9.10%-11.64%` behind the Clang/no-OpenMP reference. This is
   strong directional evidence of a remaining performance deficit, but not an accepted point estimate.
4. The process-per-sample harness includes fresh executable startup, weight mapping, thread creation, and page/cache
   state before each measured decode window. These costs are excluded from reported token time but can change the
   following steady window and are now the first variance-attribution target.
5. RoPE correctness work is not a plausible direct owner of a roughly 10% decode deficit; its arithmetic share is
   small, and the same directional gap existed before the semantic fix. Kernel selection still requires matched stage
   evidence under an accepted measurement design.

## Next Gate

1. Add process-internal repeated decode windows after one warmup/mapping phase for both runtimes, retaining every
   window and reset boundary. Use within-process CV to separate kernel variability from process startup/residency.
2. Preserve a process-level campaign around the in-process medians so production startup/mapping variation remains
   visible rather than hidden.
3. Only after both runtime CVs pass 3%, report an accepted paired throughput delta and select the largest matched stage
   deficit for optimization.
4. Keep the 128-token position-aware window; the 16-token control alone cannot represent context-growth costs.
