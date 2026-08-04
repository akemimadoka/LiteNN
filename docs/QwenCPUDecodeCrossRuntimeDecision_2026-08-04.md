# Qwen CPU Decode Cross-Runtime Decision - 2026-08-04

This document freezes the measured data and optimization decisions for the current Qwen CPU AOT decode tranche. It
is intentionally separate from the experiment narrative and implementation checklist:

- `QwenCPUDecodeBuildControl_2026-08-04.md` owns build and runtime controls.
- `QwenCPUDecodeStageControl_2026-08-04.md` owns the rejected fine-stage measurement.
- `QwenCPUDecodePerformanceEvidence_2026-08-04.md` owns detailed LiteNN profiling and experiment history.
- `PerformanceOptimizationRoadmap.md` owns implementation order.

Private model, executable, cache, and local reference-build paths are excluded from this record.

## Workload and Acceptance Boundary

The target is CPU-only stateful generated-token decode for a Qwen2.5-Coder-14B-Instruct Q4_K_M model. The controlled
comparison uses the same nine-token chat prompt, 15 steady eval/module calls, greedy sampling, ignored EOS, F16 KV,
no GPU offload, and no Flash Attention. Cross-runtime results require:

- alternating adjacent runs;
- byte-identical generated text and the same token window;
- an AOT cache hit and no LiteNN fallback;
- coefficient of variation at or below `3%` for each runtime;
- captured binary/build/runtime identity and sampled host frequency;
- no private absolute path in repository artifacts.

Profiled helper and node timings are attribution evidence, not end-to-end acceptance measurements. A candidate is
retained as a runtime optimization only after a non-profile paired A/B.

## Accepted End-to-End Comparison

The strongest reproducible local reference is the Clang 22.1.8, no-OpenMP llama.cpp build. All builds in the compiler
matrix use the same source commit and native ISA policy.

| Runtime/build | Median latency | Median throughput | CV | Decision use |
| --- | ---: | ---: | ---: | --- |
| llama.cpp GNU/OpenMP | `197.64 ms/token` | `5.06 token/s` | `0.41%` | Historical harness control |
| llama.cpp GNU/no-OpenMP | `170.02 ms/token` | `5.88 token/s` | `0.85%` | Compiler control |
| llama.cpp Clang/no-OpenMP | `165.93 ms/token` | `6.03 token/s` | `0.38%` | Strongest local reference |
| LiteNN CPU AOT, paired batch | about `181.2 ms/token` from `5.518 token/s` | `5.518 token/s` | `1.92%` | Current production candidate |

The direct three-pair control measured llama.cpp at `5.970 token/s` median and LiteNN at `5.518 token/s` median. The
preferred median of adjacent per-pair differences places LiteNN `5.49%` behind; the ratio of independent medians would
place it `7.57%` behind. Both runtimes passed output, fallback, variance, and frequency gates.

Disabling OpenMP improves the GNU reference by `16.21%`, and replacing GNU with Clang while keeping OpenMP disabled
adds `2.55%`. Weighted actual frequency differed by less than `0.18%` across the build matrix. Therefore the earlier
GNU/OpenMP comparison materially understated the reference implementation and cannot be used as the current parity
claim.

An independently observed `6.85 token/s` reference remains unclosed provenance. It is `13.60%` above the strongest
local llama.cpp sweep median. It is a reproduction target, not an accepted baseline, until its exact build/runtime
configuration passes two paired batches.

## Cross-Runtime Stage Evidence

The old synchronized stage profile compared LiteNN with the slower GNU/OpenMP reference. Its normalized result was:

| Historical stage | LiteNN | llama.cpp GNU/OpenMP | Historical difference |
| --- | ---: | ---: | ---: |
| Complete FFN evidence | `168.48 ms` lower bound | `121.94 ms` full block | at least `46.54 ms` |
| Gate + Up | `81.52 ms` | `76.55 ms` | at least `4.97 ms` |
| Q4_K activation + Down | `39.19 ms` | `19.17 ms` | `20.03 ms` |
| Q6_K activation + Down | `47.77 ms` | `26.23 ms` | `21.54 ms` |
| Attention | `63.83 ms` lower bound | `69.74 ms` full block | no LiteNN deficit established |
| Final logits | `11.35 ms` | `10.25 ms` | `1.10 ms` |

This remains useful mechanism evidence: it motivated the cold-stream and activation-quantization controls. It does not
select the next cross-runtime kernel because the reference build is now known to be materially slower.

The replacement cumulative-cut profile used Clang/no-OpenMP and GNU/no-OpenMP references over the matched token
window. Whole-token drift stayed between `-0.08%` and `+2.32%`, but derived stage CV ranged from `10.53%` to `82.51%`.
Subtracting adjacent synchronized prefixes amplified noise enough to fail the `15%` stage-CV gate. No Attention,
Gate/Up, Down, or logits difference from that run is accepted for optimization ranking.

## LiteNN Internal Attribution

LiteNN's low-cardinality native ledger closes the generated module independently of the rejected cross-runtime stage
subtraction:

| Bucket | Mean ms/token | Share of module | Interpretation |
| --- | ---: | ---: | --- |
| CPU AOT module | `205.435` | `100.00%` | Intrusively profiled module call |
| Timed helpers | `185.885` | `90.48%` | Includes projection dispatch and barriers |
| Module non-helper | `19.550` | `9.52%` | Closed by native-node and marker ledgers |
| Native node self | `13.499` | `6.57%` | Generated work outside helper timers |
| Marker instrumentation | `5.392` | `2.62%` | Measurement-only cost |
| Unattributed | `0.658` | `0.32%` | Too small to own the observed gap |

The largest non-helper categories were `5.347 ms/token` Call/control, `3.124 ms/token` projection wrappers, and
`1.956 ms/token` normalization. Broad CallNode inlining reduced IR but regressed full-token latency by `0.16%` and
increased compile-artifact time by `22.7%`; it is rejected. Marker cost does not exist in production. Projection
wrappers plus normalization are therefore the largest plausible generated-code cluster, but their intrusive timing is
an upper bound rather than an expected gain.

## Mechanism Controls

| Control | Measured result | Supported conclusion |
| --- | --- | --- |
| Q8_K activation preparation | `235/640 us` to `5.59/15.3 us` for 5120/13824 values; production `45.2007 -> 1.1865 ms/token` | Accepted; removed the former P0 bottleneck |
| Mixed Q4_K_M cold Down stream | `96.726 ms` for 48 distinct activations versus `64.065 ms` for one shared activation | Model-sized weight/activation turnover matters; not a pure materialization estimate |
| Direct SwiGLU-to-Down fusion | Paired deltas `+0.38%` and `-0.09%` | Runtime-neutral; retained only for compiler dataflow |
| Worker sequence standby | Dispatch reduced by over `99%`, but full/current-width paired gains were `2.81%` and `1.46%` | Dispatch-only optimization rejected |
| Standalone RMSNorm helper | 97 calls total `0.36-0.45 ms/token`; paired median gain `1.47%` | Below runtime gate; retained for compile structure and fusion |

The standalone RMSNorm helper nevertheless produced structural compiler gains:

| Compile metric | Change |
| --- | ---: |
| Post-LiteNN MLIR operations | `-13.80%` |
| Post-bufferization operations | `-11.10%` |
| Post-LLVM operations | `-7.69%` |
| LLVM IR instructions | `-6.66%` |
| Object emission time | `-10.23%` |
| Object bytes | `-2.99%` |
| Complete artifact compile time | `-1.53%` |

## Conclusions

1. The accepted local performance gap is approximately `5.49%`, not the historical `26.9%`. LiteNN is close enough
   that optimization decisions must survive adjacent paired runs; cache-hot microbenchmarks alone are insufficient.
2. The old FFN-Down comparison identified real mechanisms but cannot rank current cross-runtime work. A low-overhead
   matched-stage reference profile is the highest-value evidence task.
3. Q8_K preparation and worker dispatch are closed directions for now. Their large isolated counters were either
   removed or failed to translate into the required full-token gain.
4. RMSNorm-to-grouped-Q8_K staging fusion is the best bounded implementation experiment: it can remove normalized
   Float32 materialization and staging comparison/copy while building on a parity-proven helper. It still requires the
   `2%` non-profile full-token gate.
5. FFN-Down kernel or prefetch work is conditional. It should proceed only if accepted matched-stage counters or a
   cache-cold full-model-correlated experiment identifies Down as the largest remaining deficit.
6. The external `6.85 token/s` result affects the size of the target, but not implementation selection until its
   provenance is reproduced.

## Ordered Decision Gates

| Order | Work | Acceptance gate | Resulting action |
| ---: | --- | --- | --- |
| 1 | Add non-synchronizing reference-stage aggregate counters | Below `3%` total overhead and below `15%` CV per stage | Promote only the largest accepted cross-runtime deficit |
| 2 | Fuse single-consumer RMSNorm into grouped field-v4 Q8_K staging | Exact tokens, no fallback, compile growth at most `5%`, paired median gain at least `2%` | Retain as runtime optimization or keep only the standalone helper |
| 3 | Reproduce the external `6.85 token/s` provenance | Two accepted paired batches with exact configuration | Update the controlled parity target |
| 4 | Tune FFN-Down cold-stream/kernel behavior if selected | Cache-cold and full-decode gain; no cache-hot-only acceptance | Retain the measured variant only |
| 5 | Re-run sustained and long-context controls | 128/512 generated tokens, then 2K through 1M context tiers | Validate that short-window parity survives production scale |
