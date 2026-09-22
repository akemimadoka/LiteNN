# Qwen T1 Decode Window and Stage Audit (2026-09-22)

## Decision

The next diagnostic targets are FFN Down and the vocabulary projection, not Gate/Up merely because it is large.
Five clean/profile pairs per runtime reproduce large **directional** differences in those two buckets. Neither
campaign passes the acceptance gate: power policy changes during execution, clean-run variance exceeds 3%, and
some position-bin gates fail. No production kernel, default thread count, or throughput-parity claim is promoted.

Before another kernel experiment, resolve the remaining runtime-identity differences and acquire finer, matched
stage/PMU evidence. This report does not supersede an accepted historical control with a rejected one.

## Measurement Repair

The position profilers previously accepted superficially identical prompt/generated-token lists with different
measured inputs:

- LiteNN `generation`: prefill prefix `P[:-1]`; measured inputs `[P[-1], *G[:-1]]`, N calls.
- Reference exact decode: prefill `P`; measured inputs `G`, N calls.

They were shifted by one position. In particular, the last prompt call produces the first logits and can include
cold output-weight access. The retained T8 bounded run had a `1350.15 ms` first generation module call; its remaining
31 calls averaged `166.06 ms`. Other clean first-generation calls were `188.48/205.84 ms`. This establishes that the
first-call boundary is material; it does not prove the cause of every outlier is paging or exclude host interference.

The new explicit LiteNN `--measurement-window decode` uses inputs `G[:-1]` after prefix `P`, measures N-1 calls,
validates the native runtime-step indices, and retains the first generation call and its full stage profile separately.
This is a predefined semantic boundary, not removal of whichever sample happens to be slow. The default generation
window remains available for generation latency. First-generation-call latency is not complete startup/prefill TTFT.
The existing paired throughput controller already excluded the first generated token; this fix concerns the separate
position-stage controller, not that controller's previously accepted throughput windows.

Both stage controllers now share the same tiny-stage exception: at most 15% CV, or at most 0.05 ms standard deviation
when the stage mean is at most 1 ms. Whole-token/bin 3% CV and overhead gates remain unchanged. Power policy must match
throughout the campaign, not merely within each process. An optional OS process CPU set is applied by the common
launcher; native per-window affinity/host gates in the formal controller remain a separate requirement.

## Current Control

- Host: Windows, Ryzen 9 9950X; T1 restricted to logical CPU 0 for both controllers.
- Same Qwen2.5-Coder 14B Q4_K_M file: 8,988,110,272 bytes. The model path is intentionally not recorded here.
- Nine prompt tokens; 32 forced generated outputs in LiteNN; 31 measured decode inputs in both runtimes.
- Position bins: 1-15 and 16-31; five alternating clean/profile pairs per runtime, no measured-window retries.
- LiteNN: current GCC 16.2 Release binary, CPU AOT O0, bounded activation, v4/all prepared weights, capacity 41,
  cache-hit-only execution. No Interpreter fallback. Ten runs preserve the replay and report zero natural-token
  mismatches and zero fallback. This is not a new external golden-logit validation.
- Reference: pinned llama.cpp `b81c2cdd7`, Clang 23.1.1, MinGW target/sysroot, Release, native ISA, no OpenMP or GPU,
  flash attention disabled. Clean/instrumented libraries share build options; instrumentation adds no node barriers.
- Prefix digest: `c283080bdc6c1a7c05bb55f4d175d15bf4876f4e2cb63ecca6d50361b7e46917`.
- Measured-input digest: `d2652273a7ac29e1da25544bb1f0d9a9c4049d453ddabf945a580e825b8f7b72`.

Counts and both digests match between reports, as do model size and the requested/applied CPU domain. The campaigns
run sequentially, not as alternating cross-runtime pairs. The launcher CPU mask is not continuous affinity telemetry.
Do not label this as completion of the formal corrected T1/T2/T4/T8 control.

Reference defaults are still F16 KV and at least 64 context slots. Source inspection shows LiteNN derives the KV type
from the quantized embedding's F32 expressed type (`GGUFImporter.cpp`, `LLaMABuilder.cpp`); its artifact capacity is 41.
These controls therefore match token positions, not every runtime numerical/layout setting. Attention comparisons
must first match KV dtype/capacity explicitly. The reference's coarse Attention bucket also includes several norms,
residual boundaries, and embedding work that LiteNN reports separately.

## Measurements

All values below are retained rejected-campaign observations, in ms per measured token.

| Runtime/boundary | Clean median | Profile median | Clean CV | Profile CV | Paired median overhead |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reference decode call | 233.200 | 227.144 | 4.51% | 0.92% | -2.73% |
| LiteNN compiled module | 279.304 | 282.495 | 6.16% | 1.01% | +1.14% |

| Raw profile stage | Reference median | LiteNN median | Difference | Interpretation |
| --- | ---: | ---: | ---: | --- |
| Gate/Up | 91.737 | 93.924 | +2.187 | Large local share does not establish a large deficit |
| Activation | 0.179 | 1.062 | +0.883 | Different Q8_K staging boundaries; not pure exp time |
| Down | 69.877 | 94.602 | +24.725 | Highest-priority projection diagnostic candidate |
| Logits | 14.834 | 34.481 | +19.647 | Second high-value candidate, especially large-output dispatch |
| Generated module residual | n/a | 10.645 | n/a | Reference residual scope is not equivalent |

LiteNN raw Down/logits stage CV is `0.98/1.15%`. Reference raw Down spans `69.206-71.443 ms` across five profiles.
Reference normalized Down/logits medians are `71.835/14.940 ms`; LiteNN normalized medians are `93.056/34.037 ms`.
The direction survives normalization, but normalization cannot repair unstable clean runs or unmatched policies.
Stage medians do not sum to the whole-token median, so their differences must not be turned into an exact percentage
of the full deficit or a promised speedup. Q8_K preparation is charged to LiteNN's activation timer, whereas the
reference Down MatMul includes its activation conversion. Gate/Up includes extra normalization work on the reference.

| Gate | Reference | LiteNN |
| --- | --- | --- |
| Whole clean/profile variance | Fail | Fail |
| Whole profile overhead | Pass | Pass |
| Whole stage variance | Pass | Pass |
| Stage coverage | 99.93%, pass | 100%, pass, includes computed residual |
| Position-bin latency variance | Fail | Fail |
| Position-bin stage variance | Fail | Pass |
| Position-bin profile overhead | Fail | Fail |
| Campaign power policy | Fail | Fail |

The observed power schemes switch between High Performance (`8c5e7fda-...`) and Balanced (`381b4222-...`) during
multiple processes. No power policy was changed by this experiment. LiteNN first-generation module calls are retained
separately at `399.608-440.743 ms` clean and `414.976-426.417 ms` profiled. All samples remain in the raw bundle.

## Next Acceptance Work

1. Explicitly configure/report KV dtype and context capacity in both controls, and reject mismatched identities.
   Preserve a separate reference-default F16 row when evaluating user-facing deployment choices.
2. Run cross-runtime alternating T1 windows after a stable-power/host-admission check. Keep the current 3% thresholds,
   all outliers, exact input replay, cache-hit, no-fallback, and per-window affinity/host telemetry gates.
3. Split Down by Q4_K/Q6_K and isolate logits' real 5120-to-152064 projection. Profile the actual full-model helper
   dispatch, then compare cache-cold shape-matched streams before changing SIMD layout, prefetch, or thread policy.
4. Collect cycles/instructions and cache/stall evidence in aligned windows. `xperf -pmcsources` exposes TotalCycles,
   InstructionRetired, DcacheMisses and related sources on this host; this turn collected no PMU counters. The shell
   is not elevated and WPR was not recording. Tool discovery is not evidence of IPC or bandwidth.
5. Promote a change only after at least 5% whole-token gain without correctness, memory, or variance regression.

## Artifacts and Reuse

Raw reports: `build/cpu_activation_validation/t1_reference/stages.json` and
`build/cpu_activation_validation/t1_litenn/stages.json`. Rejected T8 generation data remains in
`build/cpu_activation_validation/bounded/stages.json`. Invocation tokens/paths are redacted in controller reports;
native child reports remain under ignored build storage. Reproduction conventions and window mapping are documented
in `benchmark/llama_cpp_stage_profile/README.md`.

Binary SHA-256:

- Reference clean: `a3f4b48d0b131e7469fb9667c16e4f58788737d923b5e4d5b7eb0b4f69656c6c`.
- Reference profiled: `1f16196d3225ec587e539742b7f4225698cb8a420712d48c9153e656c1bf393c`.
- LiteNN: `5c9e4a8b2aabf9e23972e601067a15d340275df5f4eb4851d925e27a5d65395e`.

The T1 artifact reused the existing 9.16 GB shared payload and added about 1 MiB of instruction/metadata storage.
Both reference builds and the detached source fit in 196.82 MiB. Final `build` logical size is 8.75 GiB, with
29.04 MiB of current validation evidence and one 8.53 GiB shared-weight store. The 32 GiB/10,000-file budget check passes.
