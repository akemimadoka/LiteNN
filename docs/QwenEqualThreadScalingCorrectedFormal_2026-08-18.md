# Qwen Corrected Equal-Thread CPU Decode Control (2026-08-18)

## Scope

This report records the corrected formal equal-thread control for Qwen2.5-Coder 14B Q4_K_M CPU AOT decode after
fixing process-affinity inheritance and affinity-domain host telemetry. It is separate from the first formal campaign
because the earlier T8 result escaped the requested CPU set and its host gate observed unrelated CPUs.

Both runtimes use the same thread count, logical CPUs 0-7, fixed 63-token replay trajectory, CPU-only execution, and
alternating process order. Each thread point requests five process pairs, with one warmup and three measured windows
per process. Thread-specific LiteNN AOT cache preparation is outside timed windows. The gates require exact trajectory
and natural-sampler parity, zero fallback, stable power policy, complete telemetry, exact process affinity, stable host
activity/frequency, and at most 3% process and in-process window CV. Private model, executable, cache, and output paths
are omitted.

The host was an AMD Ryzen 9 9950X with 32 logical CPUs on Windows. The Balanced power policy remained unchanged.

## Completed Measurements

T1, T2, and T4 completed all five pairs and 15 measured windows per runtime. T8 is intentionally absent from the
performance table: the first attempt was externally terminated during pair 2, and the clean retry correctly timed out
at host admission before running a timed pair because of sustained unrelated all-core load.

| Runtime | Threads | Median t/s | Process CV | Wall ms/token | CPU ms/token | Tokens/CPU-s | T1 speedup | Parallel efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | 1 | 3.872 | 1.30% | 259.850 | 250.992 | 3.984 | 1.000x | 100.0% |
| LiteNN | 1 | 3.131 | 0.95% | 317.290 | 307.044 | 3.257 | 1.000x | 100.0% |
| Reference | 2 | 4.909 | 2.74% | 205.705 | 359.127 | 2.785 | 1.268x | 63.4% |
| LiteNN | 2 | 4.177 | 8.62% | 239.429 | 466.022 | 2.146 | 1.334x | 66.7% |
| Reference | 4 | 4.671 | 1.92% | 213.726 | 666.419 | 1.501 | 1.206x | 30.2% |
| LiteNN | 4 | 4.299 | 4.23% | 232.639 | 898.065 | 1.114 | 1.373x | 34.3% |

| Threads | LiteNN wall-throughput delta | LiteNN/reference CPU-time ratio | Extra LiteNN CPU ms/token | Formal child accepted |
| ---: | ---: | ---: | ---: | --- |
| 1 | -19.14% | 1.223x | +56.052 | No |
| 2 | -14.91% | 1.298x | +106.895 | No |
| 4 | -7.98% | 1.348x | +231.647 | No |

The five paired wall-throughput deltas preserve the same direction at every completed thread point:

| Threads | Pair 1 | Pair 2 | Pair 3 | Pair 4 | Pair 5 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | -16.40% | -17.52% | -19.45% | -20.42% | -19.87% |
| 2 | -11.78% | -13.38% | -15.40% | -24.54% | -19.42% |
| 4 | -13.51% | -8.99% | -9.36% | -5.20% | -7.11% |

## Gate Audit

All completed T1/T2/T4 processes pass exact fixed-trajectory parity, natural-sampler parity, zero-fallback,
power-policy, telemetry-coverage, and process-affinity gates. Every measured process remains within logical CPUs 0-7,
confirming the affinity repair under the formal workload.

| Threads | Process CV | In-process CV | Host stability | Telemetry | Affinity | Accepted |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | Pass | Fail | Pass | Pass | Pass | No |
| 2 | Fail | Fail | Fail | Pass | Pass | No |
| 4 | Fail | Fail | Pass | Pass | Pass | No |

- T1 fails only one in-process gate: LiteNN pair 5 reaches `6.387%` window CV. The other nine process-window CVs are
  between `0.247%` and `1.358%`.
- T2 has LiteNN process CV `8.615%`; LiteNN pairs 4 and 5 reach `14.257%` and `6.290%` window CV. Reference pair 1 also
  fails host stability because affinity-domain activity rises by `2.651x`; its frequency ratio remains `0.983` and
  passes the frequency gate.
- T4 has LiteNN process CV `4.228%` and LiteNN pair 1 reaches `10.135%` window CV. The remaining nine process-window
  CVs are at most `2.819%`, and all host and affinity gates pass.
- The T8 retry waited 180 seconds without accepting a process. Excluding two warmup samples, 715 admission samples over
  CPUs 0-7 had `35.205%` minimum, `118.367%` median, and `119.824%` maximum activity against a 35% limit. Rejecting
  this run is expected behavior; it is not a LiteNN performance result.

## Conclusions

1. LiteNN has a reproducible low-thread core-efficiency deficit. All five T1 pairs are slower, with a `19.14%` median
   throughput deficit and `22.3%` more process CPU time per token. The failed T1 gate is one LiteNN window outlier, not
   a sign reversal, so T1 is the highest-value diagnostic point while a fully accepted campaign remains pending.
2. More workers hide wall latency but do not remove work. The wall deficit narrows from `19.14%` at T1 to `7.98%` at
   T4, while LiteNN's CPU-time ratio worsens from `1.223x` to `1.348x` and its extra CPU cost grows from `56.052` to
   `231.647 ms/token`. The next optimization should reduce instructions, memory traffic, or runtime work before tuning
   for a higher thread count.
3. LiteNN scales better in wall time from its lower T1 baseline (`1.373x` at T4 versus `1.206x` for the reference), but
   both runtimes lose CPU efficiency after T2. This evidence does not justify selecting T4 or T8 as a production
   default; that decision requires a complete accepted T1/T2/T4/T8 curve.
4. Variance is concentrated rather than universal. T1 has one failing LiteNN process, T4 has one, and T2 has two plus
   one reference host excursion. Outliers must remain in the evidence, but the next attribution run should capture
   stage timing and PMU data in the same process so the slow windows can be distinguished from steady extra work.
5. The first implementation owner must come from matched T1 stage and PMU evidence. QKV, RoPE/KV append, attention,
   attention output, Gate/Up, activation, Down, logits, residual/dispatch, and unclassified runtime overhead must cover
   the full token. A kernel or scheduler change is not promoted merely because a local LiteNN bucket is large.

## Planned Acceptance Sequence

1. Complete a fresh T8 five-pair run after host admission succeeds, then aggregate T1/T2/T4/T8 without reusing either
   partial T8 artifact.
2. Run matched T1 stage controls and collect cycles, instructions, IPC, cache misses, and memory-bandwidth evidence for
   both runtimes. Repeat at T4 to identify the source of the growing CPU-time ratio.
3. Promote only the largest measured cross-runtime owner. It must explain a material share of the `56.052 ms/token`
   T1 excess and improve whole-token latency without changing trajectory, fallback, peak memory, or variance gates.
4. Rerun the complete formal curve after the optimization and select a production default from accepted wall latency,
   CPU ms/token, and parallel-efficiency evidence.

## Raw Evidence Identity

Raw reports remain in ignored local build artifacts. Their SHA-256 identities are retained here:

- T1 complete child: `9d9619c3dddee1ca1e09f2fcddbfde50e5e7f10252326a7ef0a6950d5a7d19c3`
- T2 complete child: `60188596f3633e6d99310e41e0d13b678e9366fb4497c240cce139e13c4187ec`
- T4 complete child: `9f3c7bde763c3c00935efc9749bbe11f2afa3781bfd5b086ef8c65f22b2c7a21`
- Interrupted T8 child, excluded: `65855bb146b0cc63475093b858c56a7bb367f2a6580f5538e693ed909d6fe7fe`
- Host-admission-rejected T8 retry, excluded: `6ebd941cf16ceacfba2f652b467832de53321d0bec7546f058a046f9a5aa45a7`

