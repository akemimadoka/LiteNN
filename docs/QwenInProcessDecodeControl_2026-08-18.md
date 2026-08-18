# Qwen In-Process Decode Control (2026-08-18)

## Scope

This report evaluates steady CPU AOT decode throughput for Qwen2.5-Coder 14B Q4_K_M after the NeoX RoPE fix. It
supersedes the process-per-sample interpretation in `docs/QwenPostNeoXThroughputControl_2026-08-14.md` without
discarding those rejected observations. No machine-specific model, cache, or executable path is retained.

The LiteNN configuration is the production logits-only capacity-256 artifact, LLVM O0, T8 adaptive workers, strict
activation math, and field-interleaved-v4/all prepared GGML weights. The reference is the local Clang/no-OpenMP
llama.cpp CPU build at T2. Both runtimes consume the same nine-token chat prompt and the same fixed 64-token reference
trajectory; each measured window therefore contains 63 post-prefill decode calls.

## Measurement Design

Both runtime adapters now expose process-internal decode-window controls. One process maps the model or compiled
module and creates its runtime resources once. Each window then resets state, replays the prompt outside the timed
region, and measures the same forced post-prefill token sequence. One warmup window precedes three retained measured
windows. The outer harness still alternates fresh processes, so it reports two independent variance levels:

- in-process CV across the three decode windows;
- process-level CV across the per-process medians.

The predeclared acceptance threshold is 3% for every runtime at both levels. Exact trajectory, natural sampler parity,
zero fallback, and stable power policy are mandatory. Failed windows and pairs are retained. The paired-delta CV is
not an acceptance metric because its mean can cross zero.

The implementation also freezes prompt token ids once through a combined chat-template/tokenization endpoint. This
removes repeated tokenizer model mappings from the alternating loop without placing tokenization inside decode timing.

## Campaign Results

| Campaign | Outer pairs | Decode calls/window | Reference median / process CV | LiteNN median / process CV | Paired median | Maximum within-process CV (reference / LiteNN) | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Short-window control | 3 | 15 | 4.956 t/s / 0.628% | 5.220 t/s / 0.802% | +5.476% | 2.814% / 4.533% | REJECT |
| 63-call preliminary | 3 | 63 | 4.949 t/s / 1.540% | 5.009 t/s / 1.148% | +1.626% | 1.265% / 1.750% | PASS, insufficient pair count |
| 63-call acceptance attempt | 5 | 63 | 4.966 t/s / 2.601% | 5.106 t/s / 0.745% | +2.983% | 4.546% / 13.727% | REJECT |
| 63-call allocation/telemetry rerun | 5 | 63 | 4.651 t/s / 3.723% | 4.998 t/s / 2.316% | +9.350% | 1.663% / 6.729% | REJECT |

The short-window control proves that 15 decode calls are too sensitive to one slow window for the 3% rule. The
three-pair 63-call campaign passes every implemented gate and places the two runtimes near parity, but it is only a
preliminary control because the roadmap requires at least five alternating pairs.

### Five-Pair Attempt

| Pair | Order | Reference median | LiteNN median | Paired delta | Within-process CV (reference / LiteNN) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Reference -> LiteNN | 4.974 t/s | 5.079 t/s | +2.115% | 0.802% / 0.582% |
| 2 | LiteNN -> Reference | 4.894 t/s | 5.102 t/s | +4.265% | 0.943% / 2.251% |
| 3 | Reference -> LiteNN | 4.934 t/s | 5.106 t/s | +3.492% | 1.060% / 1.828% |
| 4 | LiteNN -> Reference | 4.966 t/s | 5.114 t/s | +2.983% | 0.811% / 1.453% |
| 5 | Reference -> LiteNN | 5.224 t/s | 5.181 t/s | -0.825% | 4.546% / 13.727% |

Pairs 1-4 pass both runtime window gates. Pair 5 fails both. Its raw measured windows move in opposite directions:

| Pair-5 window | Reference | LiteNN | LiteNN prefill |
| ---: | ---: | ---: | ---: |
| 1 | 4.972 t/s | 5.633 t/s | 1432 ms |
| 2 | 5.224 t/s | 5.181 t/s | 1431 ms |
| 3 | 5.445 t/s | 4.277 t/s | 1923 ms |

The power scheme remains Balanced. Process-wide weighted-frequency medians are about 4.986 GHz for the reference and
5.069 GHz for LiteNN, but those aggregates cannot explain or reject the opposite within-process trends. The final
LiteNN window slows in both prefill and decode, while the reference improves across its three windows. This is direct
evidence that process-wide frequency and policy records are not sufficient temporal controls.

All three campaigns preserve exact forced trajectories, zero natural sampler mismatches, and zero fallback. Raw report
SHA-256 digests are retained for local evidence verification:

- short-window control: `9b1a2da3e0e074f4a9675ccb530fe5619f8d62a48bf043619145f96b3087dc61`;
- three-pair 63-call control: `6b7e0c7f0031327e37d28d63fcb17771bf6f690670ae7cfc9064dc8d47338c98`;
- five-pair 63-call attempt: `b8476b79959e7158d761396d00d8b341e58e58992ed871f41b2e64f3df4312ea`.

## Allocation Lifecycle Finding

The first timestamp-instrumented LiteNN smoke exposed a separate correctness-of-ownership problem before another
acceptance run was meaningful. Generated CPU AOT entries allocate result memrefs through the JIT-registered `malloc`.
The uniform wrapper copies those results into caller-owned output and state-alias buffers, but allocations left live by
the generated entry were not reclaimed when the wrapper returned. Replaying windows in one mapped process therefore
retained approximately `1.7 GiB` per window. A short pre-fix run grew to about `16.03 GiB` RSS and `7.43 GiB` private
bytes, and the earlier five-pair attempt's final process exceeded `25 GiB` during repeated windows.

Commit `86f02ab` scopes generated allocations to one CPU entry invocation. Generated `free` calls still release their
own allocations immediately; scope exit releases only outstanding allocations created through that invocation, and
never borrowed inputs, external weights, caller-owned outputs, or state aliases. Focused compiled-module/stateful/GGUF
tests, a full parallel build, and all `636` CTest cases pass after the change.

The post-fix smoke remains within about `9.12-9.22 GiB` RSS and `0.50-0.60 GiB` private bytes. In the formal rerun, all
15 LiteNN measured windows report `8.585-8.586 GiB` RSS and `0.559 GiB` private bytes. There is no per-window residency
growth, so retained generated return allocations are closed as the owner of the former memory and late-window drift.

## Timestamp-Aligned Rerun

The rerun uses the same fixed 63-call trajectory, one warmup plus three measured windows, five alternating process
pairs, Balanced power policy, exact trajectory, no fallback, and the unchanged 3% gate. It additionally joins every
native decode interval to raw host-frequency and process-resource samples. `host utility` below is the mean of the
Windows PDH per-logical-processor utility counters; it is useful for relative interference detection and may exceed
100 under boost, so it is not a conventional whole-machine utilization percentage.

| Pair | Order | Reference median | LiteNN median | Paired delta | Within-process CV (reference / LiteNN) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | Reference -> LiteNN | 4.933 t/s | 4.877 t/s | -1.135% | 1.075% / 6.729% |
| 2 | LiteNN -> Reference | 4.696 t/s | 4.809 t/s | +2.410% | 1.470% / 1.315% |
| 3 | Reference -> LiteNN | 4.651 t/s | 5.086 t/s | +9.350% | 0.423% / 1.275% |
| 4 | LiteNN -> Reference | 4.470 t/s | 5.037 t/s | +12.676% | 1.663% / 0.924% |
| 5 | Reference -> LiteNN | 4.567 t/s | 4.998 t/s | +9.423% | 1.451% / 0.519% |

Every correctness, trajectory, fallback, power-policy, and telemetry-coverage gate passes. The run is rejected by two
variance owners: reference process medians have `3.723%` CV, and pair 1 LiteNN windows have `6.729%` CV. LiteNN process
medians have `2.316%` CV; the other four LiteNN processes have at most `1.315%` within-process CV.

Pair 1 localizes its LiteNN failure to the third measured window:

| Window | Throughput | Weighted frequency | Host utility | LiteNN process CPU | RSS / private |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4.905 t/s | 5179 MHz | 38.6 | 662.4% | 8.59 / 0.56 GiB |
| 2 | 4.877 t/s | 5172 MHz | 38.9 | 674.8% | 8.59 / 0.56 GiB |
| 3 | 4.343 t/s | 5040 MHz | 116.6 | 654.2% | 8.59 / 0.56 GiB |

The `-11.46%` first-to-last throughput movement accompanies only a `-2.55%` frequency movement, stable LiteNN CPU
consumption, stable residency, and a threefold host-utility excursion. This is direct evidence of a host interference
interval; it is not evidence of an accumulating LiteNN allocation or progressively idle worker pool. The other four
LiteNN processes show first-to-last movement between `-2.48%` and `+2.14%`.

### Throughput Versus CPU Work

The two runtimes intentionally use their previously selected local throughput configurations: LiteNN T8 and the
reference T2. The wall-time result must therefore be separated from CPU-efficiency claims:

| Measured-window median | Reference T2 | LiteNN T8 | LiteNN / reference |
| --- | ---: | ---: | ---: |
| Wall time | 216.291 ms/token | 200.197 ms/token | 0.926x |
| Process CPU time | 353.919 ms/token | 1275.298 ms/token | 3.603x |
| Process CPU utilization | 168.7% | 646.5% | 3.832x |
| Weighted actual frequency | 4979 MHz | 5102 MHz | 1.025x |
| Measured-window RSS | 14.031 GiB | 8.586 GiB | 0.612x |
| Measured-window private bytes | 6.339 GiB | 0.559 GiB | 0.088x |

The directional wall-throughput lead is purchased with roughly four times the configured worker budget and `3.60x`
the measured process CPU time. This campaign therefore supports neither an accepted LiteNN throughput lead nor CPU
efficiency parity. It does show that the mapped LiteNN artifact has lower measured residency on this Windows control
and that allocation growth no longer contaminates the timing series.

The post-fix raw report SHA-256 is
`e15c3d46f75dee76437a05959d75210a6a30b5cc223348daf4cabdf6bd64cc57`. The report remains in ignored local build
artifacts; no machine-specific model or executable path is retained here.

## Conclusions

1. The earlier `-9.10%` to `-11.64%` fresh-process direction is not representative of mapped steady-state decode.
   Reusing one mapped runtime removes that deficit in every campaign median.
2. Generated CPU AOT return allocations were a real process-lifetime defect. The scoped-allocation repair removes the
   repeated-window memory growth and the prior residency-driven late-window degradation.
3. The post-fix five-pair median direction favors LiteNN by `9.350%`, but the result remains rejected. Reference
   process CV is `3.723%`, and one externally disturbed LiteNN window raises its process CV to `6.729%`.
4. Correctness, cache identity, fallback, power policy, and telemetry coverage are closed. The immediate measurement
   owner is host-state admission and isolation, not another speculative kernel rewrite.
5. T8 LiteNN consumes `3.60x` the process CPU time of the T2 reference. CPU efficiency and scaling are the largest
   unclosed performance questions even if a later wall-throughput campaign passes.
6. Existing matched-stage evidence does not currently select a projection rewrite. New implementation work must be
   chosen from accepted same-budget scaling data or a fresh matched-stage deficit.

## Next Gate

1. Add an explicit pre-process and pre-window host-state admission gate. Require a quiet rolling utility baseline,
   bounded frequency drift, and no external-load excursion; retain rejected windows without silently retrying them.
2. Add a fixed configurable cooldown between runtimes and pairs, report admission wait/cooldown time, and repeat the
   five-pair campaign without changing the 3% variance threshold.
3. Run adjacent T1/T2/T4/T8 sweeps for both runtimes with the same affinity domain. Report wall time, process CPU
   ms/token, speedup, parallel efficiency, and throughput per CPU-second; choose production defaults from that curve.
4. Require two accepted five-pair batches before claiming wall-throughput parity or a lead.
5. Only then capture matched QKV, attention-output, Gate/Up, Down, logits, dispatch, and residual stages at the selected
   equal-budget point and optimize the largest accepted cross-runtime deficit.
