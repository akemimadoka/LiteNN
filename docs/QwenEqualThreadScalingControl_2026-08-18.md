# Qwen Equal-Thread CPU Decode Scaling Control (2026-08-18)

## Scope

This report evaluates the CPU work efficiency hidden by the earlier throughput-optimized T8 LiteNN versus T2
reference control. It uses the same Qwen2.5-Coder 14B Q4_K_M fixed trajectory, CPU AOT artifact policy, strict
correctness gates, and mapped in-process windows, but assigns both runtimes the same thread count and the same OS
process-affinity domain. No private model, executable, or cache path is retained.

The first real campaign is intentionally a short infrastructure control, not an acceptance result:

- thread counts: T1 and T2;
- shared process CPU set: logical CPUs 0-7;
- three alternating process pairs per thread count;
- one warmup and two measured windows per process;
- 15 post-prefill decode calls per window;
- 3% process and in-process CV thresholds;
- exact fixed trajectory, zero fallback, stable power policy, complete window telemetry, and verified process affinity;
- host admission at three consecutive activity samples no greater than 35, with two monitor warmup samples;
- 2-second runtime and 5-second pair cooldowns.

Thread count participates in the LiteNN AOT cache identity. The controller therefore performs an explicit unmeasured
cache-preparation phase for each thread count. Every timed LiteNN process still requires an AOT cache hit. Compilation
or first publication cannot enter the measured windows.

## Results

| Runtime | Threads | Process median | Process CV | Window wall | Process CPU | Tokens/CPU-s | T1 speedup | Parallel efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | 1 | 3.225 t/s | 3.082% | 307.555 ms/token | 290.104 ms/token | 3.447 | 1.000x | 100.0% |
| LiteNN | 1 | 2.537 t/s | 7.081% | 388.555 ms/token | 372.396 ms/token | 2.685 | 1.000x | 100.0% |
| Reference | 2 | 4.328 t/s | 0.612% | 232.158 ms/token | 341.146 ms/token | 2.931 | 1.342x | 67.10% |
| LiteNN | 2 | 4.209 t/s | 1.072% | 237.592 ms/token | 407.813 ms/token | 2.452 | 1.659x | 82.97% |

| Threads | LiteNN wall-throughput difference | LiteNN/reference process CPU-time ratio |
| ---: | ---: | ---: |
| 1 | -21.35% | 1.284x |
| 2 | -2.75% | 1.195x |

The top-level equal-thread, shared-affinity, and CPU-telemetry gates pass. Both child controls are rejected by the
unchanged variance rule:

- T1 process CV is `3.082%/7.081%` for reference/LiteNN. Pair 2 reference window CV is `7.356%`; pair 3 reaches
  `3.511%/4.003%`.
- T2 process CV passes at `0.612%/1.072%`, but pair 3 reference window CV is `13.345%` across the two short windows.
- Every correctness, no-fallback, power-policy, host-stability, telemetry-coverage, and process-affinity gate passes.

Host admission was active rather than ceremonial. Most processes admitted after approximately 1 second. Three T2
processes observed transient activity around 116-117 and waited about 3.0-8.5 seconds for the required quiet streak
before launch. No timed process started during those excursions.

The raw top-level report SHA-256 is
`15c3965e632c19913d01e9a9e25283c39f7f6a1bedff1c3bea96f70c8a938d33`. Raw child reports remain in ignored local
build artifacts.

## Conclusions

1. LiteNN has a material single-thread core-efficiency deficit on this workload. At T1 it is `21.35%` slower in wall
   throughput and consumes `28.4%` more process CPU time per token.
2. LiteNN scales better from T1 to T2: `1.659x` versus `1.342x`. This closes most of the wall-time difference at T2,
   but does not close work efficiency; LiteNN still consumes `19.5%` more CPU time per token.
3. The earlier T8 LiteNN versus T2 reference direction primarily demonstrates successful parallel scaling. It does
   not establish equal-resource kernel efficiency.
4. Adding more workers is no longer the first optimization choice. The highest-value CPU performance question is the
   accepted T1/T2 stage owner for extra work, cache misses, vector utilization, or generated-code overhead.
5. The short campaign proves the control path and reveals a strong direction, but cannot be quoted as an accepted
   performance result. Fifteen-call/two-window controls remain too sensitive for the 3% gate.

## Next Gate

1. Run T1/T2/T4/T8 with five alternating pairs and three measured 63-call windows, retaining the same host admission,
   cooldown, affinity, correctness, and 3% gates.
2. Report speedup, parallel efficiency, process CPU ms/token, and tokens/CPU-second at every thread count. Select the
   production default from both latency and CPU-cost curves.
3. At the first accepted equal-thread point with a material deficit, capture matched QKV, attention output, Gate/Up,
   activation, Down, logits, dispatch, and residual stages.
4. Add PMU evidence for cycles, instructions, IPC, cache misses, and memory bandwidth at T1 before selecting a kernel
   rewrite. Optimize the largest accepted cross-runtime stage and instruction-level deficit.
