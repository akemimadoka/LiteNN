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

## Conclusions

1. The earlier `-9.10%` to `-11.64%` fresh-process direction is not representative of mapped steady-state decode.
   Reusing one mapped runtime removes that deficit in every campaign median.
2. The accepted three-pair control supports CPU decode parity under this configuration, not a stable LiteNN lead. Its
   paired range crosses zero (`-2.507%` to `+2.197%`).
3. The five-pair median favors LiteNN by `2.983%`, but the result is rejected. It must not be used as an accepted point
   estimate because pair 5 violates both in-process variance gates.
4. Correctness, cache identity, fallback, and power-policy gates are closed for this experiment. The remaining owner is
   temporal host/runtime drift inside a mapped process, now localized to individual windows.
5. The next kernel must be selected from matched stage evidence after the window-level stability gate passes. The
   projection-heavy LiteNN budget remains a profiling priority, not proof of a cross-runtime deficit.

## Next Gate

1. Timestamp each decode window and join it to per-window CPU frequency, utility, process CPU time, working/private
   bytes, active CPU set, and system-load samples. Preserve raw samples, not only process-wide aggregates.
2. Add a temporal drift report for monotonic window trends and reject a pair when host load or runtime telemetry cannot
   account for a threshold breach. Keep the existing 3% CV rule unchanged.
3. Repeat at least five alternating 63-call pairs under the same fixed trajectory and stable power policy. Acceptance
   requires every in-process CV and both process-level CVs to pass.
4. After acceptance, capture matched QKV, attention-output, Gate/Up, Down, and logits stages in the same window design
   and optimize the largest measured cross-runtime deficit.
