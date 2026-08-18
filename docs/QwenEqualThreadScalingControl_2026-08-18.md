# Qwen Equal-Thread CPU Decode Scaling Control (2026-08-18)

## Scope

This report separates CPU core efficiency from parallel scaling for Qwen2.5-Coder 14B Q4_K_M decode. LiteNN and the
Clang/no-OpenMP reference use the same thread count, the same logical CPU 0-7 process-affinity domain, the same fixed
trajectory, and equivalent CPU-only runtime settings. Private model, executable, cache, and output paths are omitted.

The formal campaign uses T1/T2/T4/T8, five alternating process pairs, one warmup plus three measured 63-call windows,
a 3% process and in-process CV limit, exact token trajectory, zero fallback, a stable power policy, complete telemetry,
and pre-generated thread-specific AOT caches. Compilation and first cache publication are outside timed windows.

## Formal Results

| Runtime | Threads | Median | Process CV | Wall ms/token | CPU ms/token | Tokens/CPU-s | T1 speedup | Parallel efficiency |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Reference | 1 | 3.761 t/s | 4.00% | 260.791 | 253.472 | 3.945 | 1.000x | 100.0% |
| LiteNN | 1 | 3.107 t/s | 1.74% | 320.217 | 312.004 | 3.205 | 1.000x | 100.0% |
| Reference | 2 | 5.360 t/s | 0.59% | 186.843 | 338.046 | 2.958 | 1.425x | 71.2% |
| LiteNN | 2 | 4.499 t/s | 1.46% | 222.035 | 386.905 | 2.585 | 1.448x | 72.4% |
| Reference | 4 | 5.332 t/s | 0.86% | 187.543 | 612.351 | 1.633 | 1.418x | 35.4% |
| LiteNN | 4 | 4.947 t/s | 0.86% | 201.652 | 712.054 | 1.404 | 1.592x | 39.8% |
| Reference | 8 | 4.849 t/s | 2.81% | 208.266 | 1164.931 | 0.858 | 1.289x | 16.1% |
| LiteNN | 8 | 4.901 t/s | 5.87% | 204.042 | 1532.738 | 0.652 | 1.578x | 19.7% |

| Threads | LiteNN wall-throughput difference | LiteNN/reference CPU-time ratio | Formal child accepted |
| ---: | ---: | ---: | --- |
| 1 | -17.41% | 1.231x | No |
| 2 | -16.05% | 1.145x | No |
| 4 | -7.22% | 1.163x | No |
| 8 | +1.06% | 1.316x | No |

All correctness, fixed-trajectory, natural-sampler, no-fallback, power-policy, and telemetry-coverage gates pass at
every thread count. The campaign is nevertheless rejected rather than selectively filtered:

- T1 fails the reference process CV at `4.00%`, one reference in-process CV at `7.62%`, and one host-stability gate.
- T2 passes every process and in-process variance gate (`0.59%/1.46%` process CV) but fails the then host-wide
  stability gate.
- T4 passes process CV (`0.86%/0.86%`) but three reference in-process controls and host-wide stability fail.
- T8 fails LiteNN process CV (`5.87%`), four in-process controls, host stability, and process affinity.

The T8 affinity failure was deterministic: every reference window remained on logical CPUs 0-7, while every LiteNN
window reported CPUs 0-31. The Windows runtime enumerated the machine topology and applied per-thread group affinity
without intersecting it with the externally restricted process mask. T1/T2/T4 did not reach enough compact targets to
expose the defect. Host stability was also evaluated over all 32 logical CPUs even though both processes were confined
to CPUs 0-7, so unrelated work outside the benchmark domain could reject an otherwise stable run.

The raw formal scaling report SHA-256 is
`72b94bb61fc6630c06e76a1acb9e0c7219161c96eadbcefb0138c43c9bc06f9e`. Raw reports remain in ignored local build
artifacts.

## Control Repairs

Two measurement defects were fixed before drawing an implementation priority from the formal sweep:

1. CPU AOT affinity now intersects Compact/Spread targets with an externally restricted process or thread affinity
   domain. An unrestricted Windows process retains the existing multi-processor-group behavior.
2. Windows PDH telemetry retains processor group and logical CPU identity. Admission and post-window stability prefer
   affinity-domain utilization and weighted frequency, with the previous host-wide metrics retained as a fallback.

A short post-fix T8 validation used three alternating pairs and two measured 31-call windows. Every LiteNN and
reference window reported exactly CPUs 0-7; all admission and window checks used
`affinityDomainUtilityPercentMean`; correctness, process CV, host stability, telemetry, and affinity gates passed.
The medians were `4.413 t/s` reference and `4.607 t/s` LiteNN (`+4.39%` paired median). The run remains rejected because
two-window in-process CV exceeds 3%, including `13.46%` in one LiteNN process and `8.72%` in one reference process.
This validates the repairs but is not accepted performance evidence. Its raw report SHA-256 is
`5600229b66d4c3e7959a1d07b5316626581180df0aedeb86671a84fa52fedfd5`.

## Conclusions

1. LiteNN has a real core-efficiency deficit at low thread counts. The formal T1/T2 directions are `17.41%/16.05%`
   slower and consume `23.1%/14.5%` more process CPU time per token. T2 has especially low process and window variance,
   but must be rerun because the original host gate observed the wrong CPU domain.
2. Parallel scaling closes wall time without closing work efficiency. LiteNN reaches wall parity around T8 while using
   `31.6%` more process CPU time, so adding workers is not a substitute for reducing instructions, cache traffic, or
   projection/activation work.
3. T4 is the best observed LiteNN throughput point in the formal sweep (`4.947 t/s`) and T2 is the best reference point
   (`5.360 t/s`). Both runtimes lose efficiency beyond two threads; a production default must be selected from an
   accepted latency-versus-CPU-cost curve, not maximum hardware concurrency.
4. The next implementation owner should come from accepted T1 and T2 matched-stage evidence. QKV, attention output,
   Gate/Up, activation, Down, logits, dispatch, and residual stages must be compared under equal resource controls;
   T1 PMU cycles, instructions, IPC, cache misses, and memory bandwidth should decide whether the owner is arithmetic,
   generated-code overhead, or memory traffic.
5. No new kernel or scheduler rewrite is justified by the rejected T8 lead. The immediate P0 is one corrected formal
   rerun, followed by profiling the largest accepted low-thread stage deficit.

## Next Gate

1. Rerun the complete T1/T2/T4/T8 five-pair, three-window, 63-call campaign with the affinity repair and domain-scoped
   host telemetry. Require all existing 3% variance, correctness, host, telemetry, and affinity gates.
2. Select the production CPU thread default from accepted wall latency and process CPU ms/token curves.
3. Capture matched stage and PMU evidence at the first accepted low-thread point with a material deficit, then promote
   only the largest measured cross-runtime owner to implementation P0.
