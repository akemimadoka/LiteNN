# LiteNN Performance Optimization Roadmap

This roadmap tracks the performance work derived from `PerformanceAnalysis_2026-05-16.md` and the current
CUDA AOT implementation state. It is intentionally separate from `Architecture.md` and `CUDAAOTRoadmap.md`:
those documents describe capability coverage, while this one tracks benchmark-driven performance work.

## Baseline

- Benchmark sources:
  - `benchmark/results/backend_pytorch_comparison_cpu_threads_2026-05-16.md`
  - `docs/PerformanceAnalysis_2026-05-19.md`
- CPU finding: default CPU AOT already emits packed AVX-512/zmm FMA kernels for the tested MNIST-like Linear/MLP
  objects. The old scalar CPU fast path was retired on 2026-05-19. A guarded large-static-f32 intra-op path has landed,
  but it is currently a modest improvement for the largest MLP case rather than the final CPU kernel strategy.
- CUDA finding: native non-graph execution is still dominated by launch/library scheduling on small graphs. CUDA Graph
  replay is the current fast path; on the local RTX 4090 run it matches PyTorch CUDA for
  `MLP(784->512->256->10)/batch:512` and remains slower mainly on tiny workloads with a fixed `0.03-0.05 ms` floor.

## Cross-Cutting Profiling Tooling

Goal: make slow paths diagnosable from one reproducible bundle instead of manually stitching together benchmark output,
object disassembly, backend CSVs, and platform profilers.

Status: first slice implemented. The canonical checklist lives in `docs/Roadmap.md` under G6.

Priority classes for the GGUF/Qwen decode work:

- P0, step-latency reducers: optimize the capacity-independent generated-token path. This includes production-shaped
  GGML helper benchmarks, activation-side reuse/staging, Q8_K activation-staged vec-dot kernels, shape-aware
  thread/grain policy, grouped QKV projection, and grouped SwiGLU gate/up projection.
- P1, long-context blockers: remove capacity-shaped work that prevents 32K/128K/1M contexts. This includes verified
  in-place KV append, paged KV-cache state, grouped active-prefix attention, and long-context attention plans.
- P2, observability and validation gates: add per-helper/per-node timing, long-context benchmark rows, and golden
  validation. These do not directly speed up execution, but they keep P0/P1 changes measurable and regression-safe.

### 2026-08-04 CPU Decode Gap Closure Tranche

Decision summary and ordered gates: `docs/QwenCPUDecodeCrossRuntimeDecision_2026-08-04.md`; detailed evidence owner:
`docs/QwenCPUDecodePerformanceEvidence_2026-08-04.md`; detailed profiling narrative:
`docs/PerformanceAnalysis_2026-08-04.md`; projection phase evidence:
`docs/QwenCPUDecodeProjectionProfile_2026-08-04.md`; stronger build/runtime control:
`docs/QwenCPUDecodeBuildControl_2026-08-04.md`; low-overhead stage-control evidence:
`docs/QwenCPUDecodeStageControl_2026-08-04.md`. The earlier GNU/OpenMP T8 stage attribution is retained as historical
profiling evidence but no longer selects implementation work. The accepted stronger paired control places LiteNN
`5.49%` behind Clang/no-OpenMP. A matched cumulative-cut profile reproduced the Clang reference near `166 ms/token`,
but its `10.53-82.51%` derived-stage CV rejected fine attribution. LiteNN's stable internal accounting measured
`176.063 ms/token` module time, `164.155 ms` helper time, and an `11.408 ms` non-helper residual.

P0 implementation order:

- [ ] Execute the decision gates in the recorded order; do not select work from rejected synchronized stage deltas.
  - [ ] First, replace reference callback differencing with non-synchronizing aggregate counters and accept a stage
    only below `3%` whole-token overhead and `15%` stage CV.
  - [ ] In parallel with that evidence work, complete the bounded single-consumer RMSNorm-to-grouped-Q8_K staging A/B;
    retain it as a runtime optimization only at or above the `2%` paired full-token gate.
  - [ ] Reproduce the remaining external `6.85 token/s` provenance with two accepted paired batches.
  - [ ] Open FFN-Down kernel/prefetch work only when the new stage evidence selects it, and require cache-cold plus
    full-decode improvement rather than a cache-hot-only win.
  - [ ] After short-window closure, extend the same correctness and variance gates to 128/512 generated tokens and
    2K/32K/128K/1M context tiers.

- [x] Build a cache-cold GGML projection-stream benchmark.
  - `GGMLFieldInterleavedV4ColdProjectionStream` covers isolated Q4_K/Q6_K `13824 -> 5120` Down streams, the observed
    48-layer Q4_K_M order, and an otherwise identical shared-activation control. It reports source/prepared/allocation
    bytes, effective GB/s, weighted hot/cold ratio, grouped/single mode, requested/resolved threads, unique activation
    count, per-call/full-sequence latency, and reference delta.
  - The 2026-08-04 T8 medians were `41.175 ms` for Q4_K x24, `53.221 ms` for Q6_K x24, `96.726 ms` for the real mixed
    x48 distinct-activation stream, and `64.065 ms` for its shared-activation control. Prepared/source was `1.0113x`
    for the mixed stream. Existing cache-hot rows remain separately labelled instruction-throughput evidence.
- [x] Implement and evaluate direct single-consumer SwiGLU-to-Down fusion.
  - The original distinct/shared activation A/B differed by `32.661 ms`, but it also changed prepared Q8_K reuse and the
    48-call cache/access pattern. The controlled fusion A/B invalidated its interpretation as activation-handoff tax.
  - The fused helper lets Down skip a second Float32 materialization check while preserving standalone SwiGLU behavior.
  - Preserve standalone SwiGLU semantics for non-quantized and non-decode graphs; validate Q4_K/Q6_K runtime parity,
    object import/load, AOT execution, and the mixed-format 48-layer schedule.
  - [x] Establish an explicit single-consumer, non-public-result fusion contract before bufferization; do not infer
    ownership from post-bufferization memref aliases.
  - [x] Lower the marked pair to the fused field-v4 helper and verify the generated object imports the fused symbol.
  - [x] Add runtime, object-import, and loaded-execution parity for both Q4_K and Q6_K.
  - [x] Add a paired materialized/fused mixed-format cold stream. Independent paired runs measured `+0.3185 ms`
    (`+0.38%`) and `-0.0696 ms` (`-0.09%`) materialized-minus-fused, so the optimization is neutral within noise and
    is retained for compiler dataflow cleanliness rather than counted toward CPU parity.
- [x] Attribute the residual single-projection bandwidth loss before changing the kernel.
  - The opt-in structured profile now separates activation lookup/copy/quantization, helper dispatch, per-participant
    task claims/useful work, parallel wall time, and final barrier wait without instrumenting the default path.
  - Stable generated-token steps 10-24 measured `45.2007 ms/step` in Q8_K activation quantization, `14.9425 ms` in
    dispatch, `77.7852 ms` parallel wall time, `3.8463 ms` final barrier wait, and effectively zero cache lookup,
    Float32 copy, or lock contention. Dispatch and barrier are contained in parallel wall time.
  - FFN-Down owns `32.893 ms/step` of the measured quantization. Exact-size microbenchmarks (`235 us` at 5120 and
    `640 us` at 13824) agree with production per-call measurements within `7.1%`, making them the first acceptance gate.
  - The 97 structured events cover ordinary projections; grouped Gate/Up preparation remains outside this table, so
    `45.2007 ms/step` is a lower bound for complete-schedule Q8_K preparation.
- [x] Accelerate Q8_K activation preparation before further thread-policy work.
  - [x] Replace scalar `std::nearbyint` in the Q8_K quantizer with a parity-proven nearest-integer implementation and
    cover signed maxima, ties, clamping, block sums, strided inputs, and multiple blocks with byte-exact tests.
    Exact 5120/13824-element medians improved from `235/640 us` to `5.59/15.3 us` (`41.8-42.0x`). All 19 focused
    quantized-execution tests pass, including the new byte-level reference test.
  - [x] Re-run the cache-hit production profile and require a corresponding reduction from the `45.2007 ms/step`
    baseline with unchanged tokens and no fallback. Ordinary-projection quantization fell to `1.1865 ms/step`
    (`-97.37%`), profiled stable latency fell from `266.887` to `184.275 ms/token`, and identical tokens were generated.
    Three no-helper-profile stable averages were `197.748`, `195.874`, and `200.156 ms/token`; their median is `22.94%`
    below the preceding `256.616 ms/token` baseline.
  - [ ] Extend structured activation timing to grouped Gate/Up helpers only if a later profile needs to separate its
    remaining `~67.8 ms/step` compute from the now-small quantization cost.
  - [ ] P2: Add manual AVX2/AVX-512 preparation only if a future profile makes the remaining `1.1865 ms/step` material.
    The parity-safe scalar source now permits effective compiler optimization and has removed the P0 bottleneck.
- [x] Re-rank residual projection work against a fresh CPU-only llama.cpp stage control.
  - The post-quantizer helper profile is led by grouped Q4_K Gate/Up (`~67.8 ms/step`), followed by Q6_K Down,
    Q4_K Down, hidden/output projections, and logits. Absolute helper rank alone is insufficient: optimize the largest
    measured cross-runtime stage gap, not merely the largest LiteNN stage.
  - Use alternating cache-warm T8 stage controls and preserve exact token/no-fallback gates before selecting the next
    kernel or scheduling change.
  - The first controls found no gap against the bundled GNU/OpenMP llama.cpp build, but the stronger build matrix
    invalidated that conclusion. Disabling OpenMP raises the same GNU build from `5.06` to `5.88 t/s`; Clang/no-OpenMP
    reaches `6.03 t/s`. A three-pair Clang/no-OpenMP control places LiteNN `5.49%` behind by paired median with both
    runtimes below 2% CV and no frequency disadvantage for LiteNN.
  - The fresh llama.cpp T2 stage control measured Attention `65.158 ms`, complete FFN `140.013 ms`, and logits
    `12.893 ms/token`, but that profile used the slower GNU/OpenMP reference. It remains historical evidence only and
    cannot choose the next implementation target.
- [ ] Reproduce the stronger external `6.85 t/s` CPU-only control before opening another kernel P0.
  - Capture llama.cpp commit, compiler, ISA flags, thread count, CPU mask/strictness, polling, mmap, priority, KV dtype,
    context, prompt/decode length, and the exact completion command in a redacted control artifact.
  - Compare actual completion decode, not only `llama-bench`, and run an adjacent LiteNN cache-hit control with the same
    generated-token window. Promote only the measured stage difference to P0.
  - [x] Add a repository-owned actual-completion control runner. `benchmark/run_llama_cpp_completion_control.py`
    captures binary/host/build/runtime fingerprints, alternates thread order, emits live progress, parses per-run eval
    timing, and redacts model, executable, prompt-by-default, extra absolute paths, and raw stdout/stderr artifacts.
  - [x] Establish the clean local raw-prompt actual-completion baseline. Two balanced repetitions measured T2/T4/T8
    at `5.080/4.810/4.115 t/s`; LiteNN's earlier stable-window `5.057 t/s` is within `0.45%`, but the boundaries are
    not identical and this row is no longer the primary cross-runtime conclusion.
  - [x] Align prompt formatting and decode windows. Chat mode produces the same 9-token Qwen template; llama.cpp's 15
    eval calls are compared with LiteNN's 15 post-first-generation module calls. Bracketing llama.cpp controls both
    measured `5.500 t/s`; fresh LiteNN runs measured `5.640/5.112/5.718 t/s`, median `5.640 t/s` (`+2.54%`) with a
    material slow-run variance. The first 9 generated tokens match before the runtimes' ignore-EOS policies diverge.
  - [x] Match llama.cpp `--ignore-eos` semantics by suppressing EOS during LiteNN greedy/random sampling. The post-fix
    16-token Qwen run produced byte-identical text, no fallback, `5.685 t/s` full generation, and `5.765 t/s` aligned
    steady-module throughput versus the local llama.cpp `5.500 t/s` median.
  - [x] Automate paired alternating LiteNN/llama.cpp controls and capture host frequency/power-state evidence. The new
    `benchmark/run_paired_gguf_decode_control.py` enforces prompt/eval-window parity, byte-identical text, no fallback,
    alternating order, binary identity, path redaction, and a per-runtime CV gate. Two independent three-pair batches
    passed the 3% gate. Those GNU/OpenMP results are retained as harness evidence, not the current performance
    conclusion. Evidence: `docs/QwenCPUDecodePairedControl_2026-08-04.md`.
  - [x] Build and compare controlled local compiler/OpenMP variants. On the same commit and workload, GNU/OpenMP,
    GNU/no-OpenMP, and Clang/no-OpenMP measured `5.06`, `5.88`, and `6.03 t/s`; actual frequency was `5077-5086 MHz`.
    OpenMP build choice explains most of the former local-to-external discrepancy. Evidence:
    `docs/QwenCPUDecodeBuildControl_2026-08-04.md`.
  - [x] Sweep local runtime strategy controls. T2 was best; priority and polling changes were below 1%, cross-CCD T2
    regressed `2.15%`, and same-core SMT regressed `36.27%`, confirming that affinity was applied but is not a missing
    positive gain.
  - [x] Run a stronger paired LiteNN control. Clang/no-OpenMP measured `5.970 t/s` median versus LiteNN `5.518 t/s`;
    the preferred median per-pair LiteNN difference is `-5.49%`, with byte-identical output, no fallback, and CV below
    2% for each runtime.
  - [ ] Close the remaining exact external build/runtime provenance. The observed `6.85 t/s` is now `13.60%` above
    the strongest local llama.cpp sweep median rather than `25.23%` above the old GNU/OpenMP median.
    - [x] Capture local commit, compiler, native ISA policy, OpenMP, thread affinity/strictness, polling, priority,
      mmap/repack/warmup, KV dtype, prompt template, context, command, binary hashes, and actual frequency in redacted
      artifacts.
    - [ ] Obtain the exact external binary or the remaining source/build/runtime differences that produced `6.85 t/s`.
    - [ ] Replay it through the paired runner with `--require-variance-gate`; require two accepted batches before
      treating the result as a stable cross-runtime gap.
  - [ ] P0 evidence gate: profile matched Attention, FFN Gate/Up, FFN Down, logits, dispatch, and residual boundaries
    against Clang/no-OpenMP using the same 9-prompt/15-eval window. Repeat against GNU/no-OpenMP to separate compiler
    code generation from OpenMP runtime cost. Promote only the largest statistically accepted LiteNN deficit.
    - [x] Add an out-of-tree llama.cpp stage profiler and paired multi-build runner without linking llama.cpp into
      LiteNN production targets. It supports coarse, FFN, selected-layer, and one-sync cumulative-cut modes and gates
      total variance, absolute drift, and derived-stage variance.
    - [x] Execute Clang/no-OpenMP and GNU/no-OpenMP controls. Cumulative scans kept whole-token drift between `-0.08%`
      and `+2.32%`, but derived-stage CV was `10.53-82.51%`; the result is correctly rejected for fine attribution.
    - [ ] Replace callback differencing with non-synchronizing sampling or counters inside reference kernels. Require
      every promoted stage below `15%` CV and total measurement overhead below `3%`.
- [x] P0: reconcile LiteNN's module non-helper residual before another projection-kernel rewrite.
  - Stable steps 10-24 measured `176.063 ms/token` module time, `164.155 ms` timed helpers, and `11.408 ms` residual.
    The residual is large enough to explain the accepted `~9-10 ms/token` absolute cross-runtime deficit, but currently
    combines inline AOT operations, untimed work, and instrumentation error.
  - Completed on 2026-08-04 with low-cardinality ledgers for call/control, projection wrappers, normalization,
    elementwise work, attention/position/state, views/data movement, embedding, unemitted node self, markers, and
    unattributed module work. A current 16-step generation profile measured `205.435 ms` module, `185.885 ms` helpers,
    and `19.550 ms` non-helper; the ledger closed with effectively zero arithmetic error and marker overhead was
    `2.62%` of module time, passing both gates.
  - The largest rows were marker instrumentation `5.392 ms`, call/control `5.347 ms`, projection wrapper `3.124 ms`,
    and normalization `1.956 ms`. Marker time is measurement-only, and non-profile CallNode inlining already regressed
    median latency by `0.16%` while increasing compile time `22.7%`; neither is a production optimization estimate.
    Only `0.658 ms` remained unattributed. Evidence: `docs/QwenCPUDecodePerformanceEvidence_2026-08-04.md`.
- [x] Evaluate grouped Q4_K x16 specifically on Qwen Gate/Up. Rejected: `1.14 -> 1.11 ms` was below the `~4%` run
  noise and CPU time slightly regressed, so the uncommitted route was removed.
- [x] Evaluate `atomic::wait/notify_one` for the measured worker-dispatch floor. Rejected: dispatch improved `21.7%`,
  but later worker arrival raised parallel wall/barrier time and total profiled latency regressed `0.54%`; the
  semaphore implementation remains.
- [x] P0 candidate, evidence-gated: reduce the measured per-helper dispatch floor. Closed as rejected on 2026-08-04.
  - The post-quantizer stable median is `10.365 ms/step` across 97 ordinary projections, about `0.107 ms/call` and large
    enough to affect the remaining gap. It is already included in helper/parallel-wall time and must not be added to
    the non-helper residual.
  - Batch compatible projection sequences or retain workers across a layer sequence before changing lock policy.
  - Require at least `50%` lower dispatch, at least `3%` lower full-token latency, and no parallel-wall/barrier
    regression. The rejected `atomic::wait` route demonstrates that the dispatch counter alone is not an acceptance
    metric.
  - [x] Reject signal elision for workers observed polling. A diagnostic profile changed dispatch only
    `10.365 -> 10.139 ms` (`2.2%`) and raised parallel wall from `66.225` to `76.543 ms`; it missed the dispatch gate by
    a wide margin and was removed. Together with the rejected atomic-wait path and the prior `1.16%` Latency-policy
    median advantage, this closes wake-primitive-only work. The remaining route is sequence-level submission
    amortization.
  - [x] Evaluate and reject module-level sequence standby. Keeping all participating workers resident reduced ordinary
    projection dispatch from about `10.365` to `0.05 ms/token`, but parallel wall changed `66.225 -> 71.595 ms`; three
    alternating same-interface pairs produced `-0.40%`, `+2.81%`, and `+3.90%` token-median gains, for a `2.81%`
    paired median below the gate.
  - [x] Evaluate and reject current-width sequence standby. Parking the extra four workers during T4 hidden projections
    reduced profiled parallel wall to `68.795 ms`, but three alternating pairs produced `+1.46%`, `-1.73%`, and
    `+2.95%`, for only `1.46%` paired median gain. Token ids matched in every pair. The production implementation and
    mechanism test were removed; only a low-cost `signaledWorkerCount` profiler field remains.
  - Do not schedule another wait primitive, polling budget, or module-standby policy as dispatch P0. Future helper
    fusion may still reduce submission count, but it must be justified by its operation/dataflow benefit rather than
    this closed thread-pool hypothesis.
- [ ] P1 candidate, evidence-gated: reduce the projection-wrapper plus normalization cluster.
  - The accepted residual ledger measured `3.124 ms/token` projection-wrapper and `1.956 ms/token` normalization self
    time under intrusive profiling. Inspect generated IR and ABI boundaries, then use a non-profile paired A/B to
    distinguish removable wrapper/shape work from timer boundary effects.
  - Do not revive broad CallNode inlining. Require unchanged tokens, no fallback, no compile-time increase above 5%,
    and at least `2%` lower full-token median before retaining a combined generated-code change.
  - [x] Add a strict Float32 last-axis RMSNorm CPU AOT helper and evaluate it on the real 97-call decode schedule.
    Profiled helper time is `0.36-0.45 ms/token` versus the prior `1.956 ms/token` normalization row. LLVM IR
    instructions fell `6.66%`, object emission fell `10.23%`, object bytes fell `2.99%`, and complete artifact compile
    fell `1.53%`. Three no-profile alternating pairs preserved exact tokens and produced `0.32%`, `6.42%`, and `1.47%`
    gains; the `1.47%` paired median does not close the `2%` runtime gate. Retain the helper as a compiler-size win and
    fusion primitive, not as completion evidence.
  - [ ] Fuse a single-consumer RMSNorm into grouped field-v4 Q8_K activation staging. Skip the normalized Float32
    materialization and staging cache comparison/copy, keep standalone RMSNorm semantics elsewhere, and require the
    same exact-token/no-fallback/compile/full-token gates.
  - [ ] Re-profile projection-wrapper self only after fused staging. Node timers currently surround external helper
    calls, so the `3.124 ms/token` row is an upper bound contaminated by profile boundaries rather than proof that C ABI
    wrappers are expensive.
- [ ] P1 candidate, evidence-gated: raise FFN-Down cold-stream throughput.
  - Evaluate multiple independent output-group streams per worker and bounded software prefetch distances.
  - Re-evaluate Q4_K AVX2 x16 only for the measured cold-stream Down shape; the previously rejected broad x16 routing
    remains rejected.
  - Compare Q6_K AVX2 x8/x16 and AVX-512 x16 under the same cold-stream/full-decode gate instead of assuming wider ISA
    is faster.
  - Preserve the field-interleaved-v4 layout ABI unless a replacement proves both higher full-decode throughput and a
    prepared-size ratio no greater than `1.03x`.
- [ ] Close with controlled full-model evidence.
  - Alternate at least three LiteNN and three CPU-only llama.cpp runs using each runtime's measured production thread
    policy; capture actual frequency to control host state and filesystem-cache variance.
  - Record no-profile total latency plus low-boundary FFN/Attention/logits attribution; reject fine callbacks whose
    overhead exceeds `15%`.
  - Require no fallback, unchanged generated tokens, Q4_K/Q6_K Down cold-stream throughput of at least `40 GB/s`, FFN
    latency within `10%` of the corresponding llama.cpp block, and total latency within `5%` of the same-run llama.cpp
    median.
  - Do not reuse the invalidated activation-fusion bound. Forecasts must be derived from a controlled paired benchmark
    whose changed variable matches the proposed implementation.

P1 follow-up after the P0 gate:

- [ ] Add optional PMU/platform sampling for LLC misses, memory stalls, and effective bandwidth. The profile bundle must
  degrade cleanly when Windows policy or CI privileges do not permit system profiling. Windows PDH actual-frequency
  and utility sampling is complete and rules out a clock explanation for the current build/paired controls; PMU cache,
  stall, bandwidth, and residency evidence remains open.
- [ ] Extend paired actual-decode controls from the current 15-call window to sustained 128- and 512-token windows, then
  cover 2K/32K/128K/1M context tiers as paged-KV capacity becomes production-ready. Preserve byte-identical text,
  no-fallback, AOT-cache-hit, alternating-order, and variance gates.
- [ ] Preserve a reproducible out-of-tree llama.cpp stage-control recipe without adding llama.cpp runtime linkage to
  LiteNN production targets.
- [ ] Add non-gating warm/cold benchmark trend output and alert when a cache-hot win regresses the cold-stream or
  full-decode row.

- [ ] Add a whole-process profile bundle command that combines existing `litenn_profile` evidence with waterfall
  timeline output and optional platform sampling.
  - [x] First slice: `benchmark/profile_bundle.py` wraps `litenn_profile` or an arbitrary command, captures
    stdout/stderr, writes `manifest.json`, `trace.json`, and `summary.md`, and redacts user-specified sensitive paths
    from recorded artifacts.
  - [x] GGUF decode parser slice: the bundle now converts `--stream-stats` and helper diagnostics into
    `gguf_decode_summary.json`, `gguf_decode_summary.md`, and `gguf_decode_trace.json` for token-step and helper
    attribution, including residual/non-helper share.
  - [x] Low-overhead measurement slice: `--stream-stats` records step/runtime buckets without enabling helper timing;
    detailed helper attribution is now an explicit `--profile-helpers` opt-in. Smoke reports record that choice and the
    bundle reports helper/residual attribution as unavailable instead of manufacturing zero-valued shares.
  - [x] Existing-run import slice: `benchmark/profile_bundle.py --qwen-smoke-report <qwen_smoke_report.json>` can
    rebuild the same decode helper/step attribution from a completed Qwen smoke directory, linking the original
    `qwen_smoke_trace.json` and waterfall into the bundle manifest so large-model evidence can be re-summarized without
    re-running the model.
  - [x] Reproducible A/B metadata slice: Qwen smoke reports now record CPU AOT opt level, helper thread count, affinity,
    parallelism gate, Q8_K-staged mode, and compile-diagnostics mode; `benchmark/gguf_decode_compare.py` includes the
    compact configuration string in comparison tables.
  - [x] Profile-summary comparison slice: `benchmark/gguf_decode_compare.py --litenn-profile-summary
    <gguf_decode_summary.json>` can compare already-bundled decode runs directly and carries top-helper/helper-share plus
    top-operator/operator-share and residual-share columns into Markdown/CSV/JSON outputs. Updated on 2026-07-07: the
    comparison rows also surface top-node format, activation path, resolved helper threads, compiled-module runtime, and
    host-overhead totals when the source summary contains GGUF decode runtime buckets.
  - [x] Timeline output: fine-grained Chrome Trace / Perfetto JSON for import, conversion, lowering, MLIR/LLVM compile,
    object load, runtime schedule, transfers, synchronization, GPU dispatches, and decode-loop token phases.
    - [x] Qwen smoke slice: direct GGUF/Qwen smoke logs are streamed to disk during execution, the wrapper emits
          `qwen_smoke_trace.json` and `qwen_smoke_waterfall.md`, and the GGUF AOT path reports separated-cache
          population, artifact separation, cache read, and JIT/load timing explicitly.
    - [x] Bundle merge slice: `benchmark/profile_bundle.py --qwen-smoke-report` now imports the referenced
          `qwen_smoke_trace.json` into the top-level `trace.json`, and `--trace-json` can merge arbitrary
          Chrome Trace / Perfetto JSON evidence into the same bundle.
    - [x] Cache-population visibility/fix: GGUF decode AOT cache writes now stream large regions in chunks with progress
          diagnostics and build separated metadata directly from the compiled artifact, avoiding an extra multi-GB
          `SeparateRodata()` weight copy during cache population. Borrowing/mapping original GGUF weight regions remains
          the G16.7 production target.
    - [x] Runtime memory pressure slice: GGUF decode now releases the imported archive before token execution, and
          separated-artifact cache hits can move owned constants/weights into the CPU module instead of copying the
          multi-GB weights region during load.
    - [x] Qwen decode smoke diagnosis: on 2026-07-01, `max_cache_length=10` measured about `0.54-0.62 s` per
          prompt/generation step, while the user's `max_cache_length=2049` run measured about `6 s` per generated step.
          The current capacity-decode graph masks inactive cache suffixes but still scans the full static capacity each
          step.
    - [x] Replace the immediate full-capacity decode attention hot path with active-prefix execution. On 2026-07-02,
          `ActivePrefixAttentionNode` gained CPU AOT lowering plus a rank-3 KV-cache path that reads
          `[capacity, kvHeads, headDim]` directly instead of materializing per-head `Slice+Reshape` tensors. A real
          Qwen2.5-Coder-14B Q4_K_M `--stateful --max-tokens 1 --max-cache-length 2048` smoke improved from about
          `4.3-5.0 s/step` to about `0.62-0.72 s/step`, and the lowered LLVM instruction count dropped from about
          `1.67M` to about `0.52M`.
    - [x] Add a head-aware RoPE helper/node for decode. Correct query RoPE must use head-local dimensions; applying RoPE
          to the full concatenated query vector is numerically wrong for Qwen/LLaMA. On 2026-07-02, runtime-position
          `RoPENode` gained a CPU AOT helper rewrite (`litenn_cpu_rope_at_positions_f32`) so per-head decode RoPE stays
          numerically correct without expanding into Reshape/Slice/Cos/Sin/Concat IR. A real
          Qwen2.5-Coder-14B Q4_K_M `--stateful --max-tokens 1 --max-cache-length 2048 --no-aot-cache-write` smoke
          completed in about 57s wall time, compiled the stateful decode artifact in about 22.1s, produced about 389k
          LLVM instructions after the entry wrapper, ran prompt/generation steps in about `0.63-0.72 s/step`, and
          generated token `9707` / `Hello`.
    - [x] Profile the remaining Qwen CPU AOT decode gap against a user's llama.cpp CPU reference run with GPU-style
          accelerators disabled. The 2026-07-02 analysis in `docs/PerformanceAnalysis_2026-07-02.md` shows that
          `max_cache_length=10` still takes about `522 ms` for the generated token, `max_cache_length=2048` adds about
          `200 ms`, and forcing `LITENN_CPU_AOT_THREADS=16` regresses to about `815 ms`. This rules out "threading is
          simply off" and "full-capacity attention is still the only bottleneck" as sufficient explanations.
    - [x] P2: Add operator-level and helper-level timing for stateful GGUF CPU AOT decode.
          Required output: per-layer/per-node duration, helper symbol name, GGML block format, input/output shape,
          thread count, cache length, and generated-token phase. The current per-step waterfall is enough to find the
          broad gap but not enough to rank RMSNorm, quantized projections, active-prefix attention, state copies,
          logits projection, and sampler work.
          - [x] Add explicit CPU AOT helper profiling scopes and GGUF decode diagnostics output for helper symbol,
                shape/format/thread detail, call count, total time, and average time per decode step. This covers
                sidecar/helper attribution for quantized projections, get-rows, RoPE, KV scatter, and active-prefix
                attention; per-layer/per-node attribution remains open.
          - [x] Bundle helper shares: `gguf_decode_summary.json` and `.md` now report helper time as a percentage of
                total step time and per-step helper totals/top helper, making it clearer whether the next bottleneck is
                quantized projection, attention, KV state movement, or non-helper residual work.
          - [x] Add operator/role attribution in profile bundles. Completed on 2026-07-05:
                `profile_bundle.py` classifies helper events into projection, attention, position_encoding, kv_update,
                embedding, normalization, and other roles; writes ranked `operators`, per-step `top_operator`, and
                operator/role trace args. This keeps the next report grounded while deeper per-layer/per-node timing is
                still pending.
          - [x] Add helper-derived node timing rows. Completed on 2026-07-05:
                `gguf_decode_summary.json` now emits `node_timings` with node kind, node name, helper symbol, GGML
                format, input/output shape, thread counts, calls, total/average milliseconds, and explicit
                `attribution=helper-derived`; comparison tables expose the top node. Stable runtime layer ids remain a
                later metadata enhancement rather than a P2 blocker.
          - [x] Add a measured Qwen2.5-Coder-14B Q4_K_M stateful decode attribution report. Completed on 2026-07-05:
                `docs/PerformanceAnalysis_2026-07-05.md` records the user's `~819 ms/token` LiteNN run against a
                `6.85 tokens/s` llama.cpp CPU reference. The repaired profile bundle attributes about `82.5%` of step
                time to timed helpers and about `80.5%` to quantized projection helpers; active-prefix attention is only
                about `0.16%` and KV append about `0.01%` in the 2K-context run.
          - [x] P0: Replace the current direct/staged GGML projection helpers with production packed Q4_K/Q6_K kernels.
                Target the measured top rows first: grouped gate/up `1x5120 -> 1x27648`, hidden/output
                `1x5120 -> 1x5120`, FFN-down `1x13824 -> 1x5120`, KV `1x5120 -> 1x1024`, and logits
                `1x5120 -> 152064`. The current Q8_K-staged prototype is not a default-switch candidate on the July 5
                production-shaped T16 rows, so this work needs packed/repacked weight layout, architecture-specific
                vector-dot kernels, activation staging reuse where it actually wins, and a format-specific policy.
                Progress on 2026-07-06: direct Q4_K/Q6_K helpers now use a full-column-group fast path that avoids
                repeated tail-validity checks for the common complete `x4` output group. A wider `x8` grouping
                experiment was rejected because Qwen-shaped helper rows regressed. Short helper validation after the
                retained fast path measured Q4_K default rows at about `0.70 ms` for `5120->5120`, `1.64 ms` for
                `5120->13824`, `17.5 ms` for logits, and grouped gate/up concatenated at about `3.32 ms`; a real
                Qwen2.5-Coder-14B Q4_K_M `--stateful --max_cache=11` smoke measured about `506 ms/generated token`.
                Deep profile on 2026-07-06: `build/qwen_perf_profile_default_20260706` captured 24 decode steps with
                `--stream-stats --compile-diagnostics --ignore-eos`. Total step time was about `10244 ms`; timed
                helpers accounted for about `9607 ms` (`93.8%`) and residual/non-helper time for about `637 ms`
                (`6.2%`). Generated-token steps averaged about `430.7 ms`, of which helpers averaged about
                `400.3 ms` and residual averaged about `30.4 ms`. Projection helpers alone accounted for about
                `90.2%` of all step time: grouped gate/up `34.5%`, FFN-down `23.9%`, hidden/output `16.2%`, KV
                `8.7%`, and logits `7.0%`. Attention was only about `0.37%`, KV append about `0.01%`, and token lookup
                rounded to zero. This makes projection kernel replacement the only current path with enough headroom to
                close the llama.cpp CPU gap.
                Implementation route:
                - [ ] Add a production CPU packed-weight format for GGML_Q4_K/GGML_Q6_K decode projections, stored in
                      separated compiled weights with an explicit layout/version tag and a fallback to source GGML
                      layout. The first target is output-major Qwen gate/up, FFN-down, hidden/output, KV, and logits
                      projections.
                      Progress on 2026-07-07: added a Q6_K prepacked sidecar ABI and `litenn_bench`
                      `GGMLBlockMatMulQ6KPrepackedHelper/...` rows. The sidecar expands each Q6_K block into 16 prepared
                      float scales plus 256 signed quant lanes and validates against the direct helper, but it remains
                      benchmark-only until full Qwen-shaped measurements justify routing AOT weights through this format.
                      Short validation on 2026-07-07 passed
                      `GGUFLLaMAQuantizedExecution.Q6KPrepackedHelperMatchesDirectHelper`; `qwen_ffn_down/T1` improved
                      from about `32.2 ms` direct to `15.1 ms` prepacked, and `T8` improved from about `4.77 ms` to
                      `2.28 ms`, both with `max_abs_delta=0`.
                      Progress on 2026-07-07: extended the sidecar family to Q4_K and renamed the benchmark surface to
                      `GGMLBlockMatMulPrepackedHelper/{Q4_K,Q6_K}/...`, plus added
                      `GGMLBlockMatMulPrepackWeight/...` rows to isolate one-time conversion cost. Validation passed
                      `GGUFLLaMAQuantizedExecution.Q4KPrepackedHelperMatchesDirectHelper` and the existing Q6_K parity
                      test. Short Qwen-shaped rows measured Q4_K `qwen_ffn_up/T1` from about `30.0 ms` direct to
                      `22.7 ms` prepacked, and `T8` from about `3.94 ms` to `3.36 ms`; Q6_K `qwen_ffn_down/T1`
                      remained about `33.4 ms` direct to `15.0 ms` prepacked, and `T8` about `4.68 ms` to `2.21 ms`,
                      all with `max_abs_delta=0`. One-time prepack cost measured about `6.94 ms` for Q4_K
                      `5120->13824` and `14.9 ms` for Q6_K `13824->5120`, reinforcing that prepared weights should be
                      stored in the shared separated weight cache rather than regenerated per step.
                      Progress on 2026-07-07: the prepared path is now available in separated CPU AOT artifacts for
                      ordinary Q4_K/Q6_K `QuantizedMatMulNode` weights whose variable storage is used only as that RHS.
                      `CompilerOptions::enableCPUAOTGGMLPrepackedWeights` externalizes those weights in prepared layout,
                      GraphToMLIR marks the prepared RHS placeholder, and LLVM lowering calls
                      `litenn_cpu_ggml_block_matmul_{q4k,q6k}_prepacked_f32`. The GGUF decode CLI and smoke tooling
                      expose this as `--cpu-aot-ggml-prepacked-weights`, include it in the AOT cache key, and force
                      external regions because the option changes weight layout. Validation passed the Q4/Q6 helper
                      parity tests and the output-major Q4_K/Q6_K/Q8_0 AOT regression with prepared artifacts. Remaining
                      production work: grouped Q/K/V and gate/up prepared routing, full-decode A/B, and default-policy
                      selection.
                      Progress on 2026-07-07: grouped Q/K/V and gate/up prepared routing is now implemented for
                      `GroupedQuantizedMatMulNode` as well. The CPU AOT prepack planner records projection-local
                      expressed shapes for each grouped RHS variable, GraphToMLIR marks consistent prepared grouped
                      storage, and LLVM lowering calls
                      `litenn_cpu_ggml_block_grouped_matmul{2,3}_{q4k,q6k}_prepacked_f32`. Validation passed the grouped
                      Q4_K AOT regression with prepacked external weights, plus the ordinary Q4/Q6 prepared-helper and
                      output-major quantized-matmul regressions. Remaining production work: full-decode A/B and
                      default-policy selection.
                      Progress on 2026-07-07: added `GGMLGroupedProjectionPrepackedHelper/{Q4_K,Q6_K}/...` benchmark
                      rows for Qwen Q/K/V and gate/up projection shapes. A short gate/up `T8` smoke measured about
                      `4.78 ms` real for Q4_K and `4.93 ms` real for Q6_K, both with `max_abs_delta=0`, giving the next
                      full-decode A/B run a direct grouped-prepared helper surface to compare against direct and
                      Q8_K-staged grouped rows.
                      Progress on 2026-07-07: added an explicit prepared-weight policy
                      `disabled|profitable|all` to `CompilerOptions`, the GGUF decode CLI, `qwen_smoke.py`, and the
                      thread-matrix harness. The legacy `--cpu-aot-ggml-prepacked-weights` switch remains an all-format
                      experiment, while `profitable` currently routes only Q6_K through prepared separated weights
                      because isolated helper evidence shows the clearest win there; Q4_K remains direct until
                      full-decode A/B proves it should be promoted. Validation locks this behavior with
                      `GGUFLLaMAQuantizedExecution.CPUAOTPrepackedWeightPolicyRoutesOnlyProfitableFormats`, and
                      `gguf_decode_compare.py` now records the policy in its config column.
                      Progress on 2026-07-07: `benchmark/gguf_decode_thread_matrix.py` can now run
                      `--cpu-aot-ggml-prepacked-weight-policies disabled,profitable,all` in one pass, writing
                      policy-separated work directories and a Policy column in the summary. This turns the next real
                      full-decode run into a single controlled thread x prepared-weight-policy matrix.
                      Profile on 2026-07-07: a local Qwen2.5-Coder 14B Q4_K_M `T8`, `max_tokens=8`,
                      stateful CPU AOT run compared prepared-weight policies against a CPU-only `llama-bench`
                      `T8`, `ngl=0`, `flash_attn=0` control. Generation-phase results were: disabled
                      `1232.2 ms/token` (`0.812 tok/s`), profitable `1063.1 ms/token` (`0.941 tok/s`), all
                      `861.6 ms/token` (`1.161 tok/s`), and llama.cpp `235.1 ms/token` (`4.254 tok/s`). The
                      policy-aware cache fix was required because shared weights differ sharply by layout:
                      raw `~8.98 GB`, profitable `~10.10 GB`, all `~17.93 GB`. Conclusion: prepared weights are real
                      full-decode wins (`all` is about `30%` faster than disabled), but even the all-prepared path is
                      only about `27%` of the llama.cpp CPU decode throughput; the remaining top row is Q4_K prepared
                      FFN gate/up (`~45.5%` of generation step time), so the next high-yield work is a production
                      packed/repacked vector-dot microkernel rather than more scheduler plumbing.
                      Follow-up analysis on 2026-07-07: `docs/PerformanceAnalysis_2026-07-07.md` adds a cache-hit
                      phase profile and a llama.cpp source audit. A cache-hit `T8` policy matrix measured disabled
                      `1188.6 ms/token`, profitable `1033.0 ms/token`, and a short all-prepared sample at
                      `816.9 ms/token`; the CPU-only llama.cpp `T8` control remained `235.1 ms/token`. The evidence
                      keeps the top generated-token gap inside Q4_K/Q6_K projection helpers, not host overhead,
                      active-prefix attention, KV append, or Interpreter fallback.
                      Source audit target: llama.cpp routes Q4_K/Q6_K through Q8_K activation staging and compact
                      repacked GEMV/vec-dot kernels (`ggml_vec_dot_q4_K_q8_K`, `ggml_vec_dot_q6_K_q8_K`,
                      `ggml_gemv_q4_K_8x*_q8_K`, `ggml_gemv_q6_K_8x*_q8_K`, and `block_q4_Kx8/x16` /
                      `block_q6_Kx8` layouts). LiteNN's current prepared layout is a useful proof that layout matters,
                      but it is too expanded to be the production answer.
                      Follow-up kernel slice completed on 2026-07-08: prepared Q4_K/Q6_K helpers now use a
                      runtime-gated AVX2 `lhsColumnStride == 1` fast path for complete x4 output-column groups while
                      preserving the scalar tail path. Validation passed the Q4/Q6 prepacked helper parity tests and
                      the prepared-weight policy AOT regression. Short Qwen-shaped `T8` helper smokes measured grouped
                      gate/up at about `3.67 ms` for Q4_K and `4.23 ms` for Q6_K, with direct-helper deltas in the
                      expected float-ordering range (`~3.7e-4` to `~1.6e-3`). This improves the current expanded
                      prepared layout but does not close the compact-layout requirement; v3 repack still needs to keep
                      shared weight size near raw GGUF size.
                      Layout-tagging slice completed on 2026-07-08: GraphToMLIR now emits an integer prepared-layout
                      id for the current expanded-F32-scale layout, LLVM lowering rejects unknown prepared layout ids
                      instead of accepting any marker attribute, and separated weight names include
                      `.prepacked.expanded_f32_scales_v1.<format>`. This gives compact/v3 repacks an explicit ABI split
                      point and prevents future layout/helper mismatches from being diagnosed only by byte size.
                      Cache/report isolation slice completed on 2026-07-08: GGUF decode AOT cache keys, shared-weight
                      cache keys, `qwen_smoke.py` reports, `gguf_decode_compare.py` config labels, and
                      `gguf_decode_thread_matrix.py` Markdown rows now carry the prepared layout token. Compact/v3 can
                      therefore be benchmarked without silently reusing expanded-v1 shared weights or mixing result rows.
                      AVX2 reduction slice completed on 2026-07-12: the expanded-v1 Q4_K/Q6_K kernels now reduce
                      eight Float32 lanes entirely in registers instead of spilling each accumulator to a temporary
                      array. Q4_K remained neutral; short Qwen-shaped T8 smokes moved Q6_K hidden from about `0.757 ms`
                      CPU to `0.558 ms`, FFN-up from about `1.98 ms` to `1.74 ms`, and grouped gate/up from about
                      `4.23 ms` to `3.91 ms`. This is retained as a production-v1 improvement while compact-v3 is built.
                - [x] Add a compact prepared-weight layout v3 for GGML_Q4_K/GGML_Q6_K decode projections.
                      This should store interleaved/repacked blocks rather than expanding every block into float scale
                      metadata plus wide quant lanes. Acceptance: shared weight size stays close to the raw GGUF
                      footprint while preserving or improving the all-prepared full-decode speed.
                      Runtime prototype completed on 2026-07-25: added a versioned 64-byte header, x4 output-row
                      block-grouped Q4_K/Q6_K payload, prepack/runtime C ABI, JIT symbol registration, parity and
                      padded-tail coverage, and production-shape benchmark rows. Storage ratio is effectively `1.00`
                      on Qwen shapes and compact output exactly matches the Q8_K-staged helper. Dedicated AVX2 x4
                      kernels removed the initial ~2x strided-decoder regression. A follow-up block-grouped repack lets
                      the mature byte-stride-1 vec-dot consume four contiguous raw blocks directly: Q6_K T8 now beats
                      staged on hidden/FFN-up/FFN-down (`1.32 vs 1.48 ms`, `2.85 vs 3.63 ms`, `3.33 vs 3.77 ms`);
                      Q4_K FFN-down also wins (`2.67 vs 2.78 ms`) while hidden/FFN-up remain about 10% slower. Keep v3
                      experimental until Q4_K and expanded-v1 meet the acceptance gate.
                      Single-projection AOT slice completed on 2026-08-01: added the explicit
                      `CPUAOTGGMLPrepackedWeightLayout::{ExpandedF32ScalesV1,CompactBlockGroupedV3}` compiler option,
                      layout-aware separated-weight generation, compact layout id `3` in GraphToMLIR, and matching
                      LLVM helper dispatch. Q4_K/Q6_K compact artifacts execute with Q8_K-staged parity and are smaller
                      than expanded-v1 in regression coverage. Expanded-v1 remained the default for the acceptance phase.
                      Evaluation-surface slice completed on 2026-08-01: the GGUF decode CLI, environment-backed CLI
                      configuration, `qwen_smoke.py`, reports, decode artifact keys, and shared-weight keys now use the
                      selected layout instead of a hardcoded expanded-v1 token. The thread-matrix harness can run
                      `--cpu-aot-ggml-prepacked-weight-layouts expanded-v1,compact-v3` together with policy and thread
                      axes, with isolated work directories and canonical layout labels. This enables a controlled
                      cache-hit full-decode acceptance run without cross-layout cache reuse.
                      Grouped AOT slice completed on 2026-08-01: compact-v3 now routes two- and three-projection
                      `GroupedQuantizedMatMulNode` operations through
                      `litenn_cpu_ggml_block_grouped_matmul{2,3}_compact_q8k_f32`, stages each activation row once, and
                      shares that workspace across Q/K/V- and gate/up-style projections. A new explicit
                      `QuantizedStorageLayout` field is preserved by executable plans, vNext model packages, and rodata
                      v5; GraphToMLIR validates bytes against that semantic tag instead of inferring layout from size.
                      This fixes the real Q4_K width-2 collision where compact-v3 and expanded-v1 are both 640 bytes.
                      Q4_K/Q6_K two- and three-way AOT execution, the complete 80-test GGUF suite, and the affected
                      package/rodata regression suites pass.
                      Controlled 14B acceptance matrix completed on 2026-08-01: for Qwen2.5-Coder 14B Q4_K_M,
                      stateful CPU AOT, O0, T8, all-prepared, and an eight-token cache-hit decode, expanded-v1 measured
                      `482.108 ms/token` (`2.074 tok/s`) while the first compact-v3 kernel measured `626.098 ms/token`
                      (`1.597 tok/s`). Compact-v3 reduces separated weights from `17,929,588,736` to
                      `8,982,164,544` bytes (`-49.9%`), cache-population work from `91.433 s` to `49.335 s` (`-46.0%`),
                      and sampled peak working set from about `33.8` to `20.4 GiB`, but its initial decode latency was
                      `29.9%` higher. Compact-v3 must therefore remain opt-in.
                      Register-decode slice completed on 2026-08-01: the compact Q4_K/Q6_K x4 AVX2 path now extracts
                      nibbles/bitplanes and performs integer horizontal reduction in SIMD registers instead of building
                      byte arrays and spilling lane sums. The same cache-hit decode improved to `567.607 ms/token`
                      (`1.762 tok/s`), a `9.34%` latency reduction and `10.3%` throughput gain; step-16 helper time fell
                      from `592.755` to `528.082 ms`, with identical generated tokens and no fallback. The remaining
                      expanded-v1 latency gap was `17.7%` after this slice.
                      Paired-dot slice completed on 2026-08-01: complete x4 compact blocks now evaluate two output
                      columns per 256-bit AVX2 dot, sharing the Q8_K load, multiply-add, zero-point correction, and
                      horizontal reduction. The same real cache-hit run improved again to `505.917 ms/token`
                      (`1.977 tok/s`), another `10.87%` latency reduction and a cumulative `19.2%` reduction from the
                      first compact kernel; step-16 helper time fell to `475.246 ms`. Compact-v3 is now only `4.94%`
                      slower than expanded-v1 while using `49.9%` fewer prepared-weight bytes. It remains opt-in until
                      repeated acceptance runs close or reverse that final gap; the next kernel slice is x8
                      field-interleaved scale/quant reuse.
                      Field-interleaved v4 kernel prototype completed on 2026-08-01: a separate version-4 header and
                      prepack/runtime ABI interleave four-byte quant chunks across eight output rows. Q4_K stores
                      unpacked per-subblock scale/min vectors at `1.0278x` raw block bytes; Q6_K remains `1.00x` raw.
                      Portable full/tail execution and AVX2 x8 execution both match Q8_K-staged output exactly. Against
                      the paired-dot v3 T8 rows, v4 measured Q4 hidden `0.506 vs 0.801 ms`, Q4 FFN-down
                      `1.36 vs 2.16 ms`, Q6 FFN-down `1.74 vs 2.76 ms`, and Q6 logits `14.3 vs 20.0 ms`. JIT symbols and
                      benchmark rows are available.
                      AOT wiring completed on 2026-08-01: explicit layout id `4` now propagates through quantization
                      metadata, Plan-to-MLIR validation, LLVM helper selection, externalized prepared-weight payloads,
                      cache identity, and the `field-interleaved-v4` CLI/script token. Single projection and grouped
                      2/3-projection helpers load the v4 payload directly; grouped helpers share one Q8_K activation
                      staging pass while preserving each projection's x8 boundary. Full/tail, Q4_K/Q6_K, single/grouped
                      AOT load-and-run coverage passes in the complete 81-test GGUF importer suite.
                      Real 14B acceptance completed on 2026-08-01 under the same stateful O0/T8/all-prepared eight-token
                      cache-hit conditions as v1/v3. The v4 separated weights are `9,160,094,784` bytes: only `1.98%`
                      above compact-v3 and `48.91%` below expanded-v1. Two no-fallback runs with identical generated tokens
                      measured `403.054` and `449.863 ms/token` (`2.481` and `2.223 tok/s`), beating compact-v3 by
                      `20.3%` and `11.1%` and expanded-v1 by `16.4%` and `6.7%`. The first v4 cache population measured
                      `27.306 s` compile, `7.199 s` metadata construction, and `19.840 s` weight writing. Field-interleaved-v4
                      is now the compiler and smoke-driver default whenever prepared GGML weights are enabled; v1/v3 remain
                      explicit comparison/compatibility selections.
                      AVX512VL/VNNI experiment rejected on 2026-08-01: replacing the AVX2 `maddubs + madd` sequence with
                      256-bit `vpdpbusd` on Ryzen 9 9950X did not improve Q4_K and regressed Q6_K FFN-down/logits medians
                      from `1.86/13.4 ms` to `2.01/15.7 ms`; it also introduced up to about `9.8e-4` extra Float32
                      difference on the production-shaped benchmark. The VNNI path was removed. The safe CPU-feature
                      check hoist remains; future AVX512 work must evaluate a true x16 layout/kernel rather than merely
                      substituting the x8 dot instruction.
                - [x] Add a decode-step Q8_K activation workspace keyed by the normalized hidden vector.
                      Quantize each hidden vector once per layer/step and pass the staged activation to all compatible
                      projection helpers. Do not route the existing per-helper staged prototype by default; it has
                      already lost on production-shaped rows without compact repack + vec-dot kernels.
                      First implementation slice completed on 2026-07-08: added the C ABI
                      `litenn_cpu_ggml_prepare_q8k_activation_f32`,
                      `litenn_cpu_ggml_q8k_activation_block_bytes`, and
                      `litenn_cpu_ggml_block_matmul_q8k_prepared_activation_f32`, plus
                      `GGMLBlockMatMulQ8KPreparedActivationHelper/...` benchmark rows. The new helper consumes a
                      caller-provided Q8_K activation workspace and matches the internal staged helper for Q4_K/Q5_K/Q6_K
                      in `GGUFLLaMAQuantizedExecution.Q8KPreparedActivationHelperMatchesInternalStaging`. A short
                      Q6_K `qwen_ffn_down/T8` smoke measured about `6.90 ms` real for prepared activation versus
                      `8.14 ms` for internal per-helper staging, showing the reuse path is worth wiring into graph/AOT
                      lowering once compact vec-dot kernels land.
                      Second implementation slice completed on 2026-07-08: added grouped prepared-activation helper
                      ABI for 2-way and 3-way projection groups
                      (`litenn_cpu_ggml_block_grouped_matmul{2,3}_q8k_prepared_activation_f32`), parity coverage in
                      `GGUFLLaMAQuantizedExecution.Q8KPreparedActivationGroupedHelperMatchesInternalStaging`, and
                      `GGMLGroupedProjectionQ8KPreparedActivationHelper/...` benchmark rows. A short Q6_K
                      `qwen_gate_up/grouped/T8` smoke measured `17.1 ms` prepared versus `16.5 ms` internal staged,
                      which confirms the already-grouped helper only stages lhs once; the high-return work is now
                      eliminating any remaining separate-projection fallback and replacing the current dot loop with
                      compact llama.cpp-class vec-dot kernels.
                      AOT-visible slice completed on 2026-07-08: LLVM lowering now routes non-prepacked grouped
                      GGML_Q6_K projection groups to
                      `litenn_cpu_ggml_block_grouped_matmul{2,3}_q8k_staged_f32` when
                      `enableCPUAOTGGMLQ8KStagedMatMul` is enabled. Validation is covered by
                      `GGUFLLaMAQuantizedExecution.CompilesGroupedQ6KProjectionToQ8KStagedHelper`, including both
                      2-way gate/up-style and 3-way QKV-style groups.
                      Kernel slice completed on 2026-07-08: added a Q6_K x Q8_K AVX2 all-valid x4 output-column
                      fast path for the common no-tail decode rows. Short aggregate smokes measured Q6_K
                      `qwen_gate_up/grouped/T8` staged at `6.72 ms` mean versus `15.8 ms` direct grouped, and
                      `qwen_qkv/grouped/T8` staged at `1.85 ms` mean versus `4.02 ms` direct grouped.
                      Follow-up slice completed on 2026-07-08: extended the all-valid AVX2 fast path to Q4_K/Q5_K
                      and widened AOT staged lowering from Q6_K-only to Q4_K/Q5_K/Q6_K for both single and grouped
                      projection helpers. Aggregate smokes measured Q4_K `qwen_gate_up/grouped/T8` staged at
                      `4.52 ms` versus `7.62 ms` direct, Q5_K at `5.33 ms` versus `14.1 ms`, and single
                      `qwen_ffn_down/T8` staged wins for Q4_K/Q5_K/Q6_K (`2.08/3.82/3.69 ms` versus
                      `2.79/5.97/4.67 ms`). Validation is covered by
                      `GGUFLLaMAQuantizedExecution.CompilesOutputMajorKQuantAndQ8_0MatMulWithoutMaterializingWeight`
                      and `GGUFLLaMAQuantizedExecution.CompilesGroupedKQuantProjectionToQ8KStagedHelper`.
                      Completed for the production field-interleaved-v4 path on 2026-08-01: single-projection helpers
                      share a bounded thread-local Q8_K workspace containing both the exact Float32 source snapshot and
                      staged blocks. Reuse requires a bitwise source match, so arena-buffer reuse and in-place mutation
                      invalidate safely; grouped helpers that already stage once do not pay the snapshot cost. Steady
                      T8 helper medians moved Q4_K/Q6_K hidden rows from about `0.52/0.70 ms` to `0.28/0.42 ms` with
                      `max_abs_delta=0`. Two real 14B cache-hit runs produced the same token sequence with no fallback;
                      the step-16 Q4_K/Q6_K KV rows fell from about `38.35/13.14 ms` before reuse to
                      `25.13/9.21 ms` and `11.84/4.67 ms` in the two runs. Generated-token latency ranged from
                      `341.854` to `427.627 ms/token`, versus the earlier v4 two-run range of
                      `403.054-449.863 ms/token`; projection-frequency noise remains visible, but the reused KV
                      staging reduction is consistent.
                      A true AVX2 x16 execution tile was evaluated on 2026-08-01 without changing the v4 payload ABI:
                      two adjacent x8 packed groups share each Q8_K load/broadcast while retaining separate vector
                      accumulators. Production-shaped T8 medians improved Q4_K logits from about `10.3` to `8.69 ms`
                      and Q6_K logits from about `15.1` to `13.3 ms`, with exact staged-helper parity. Applying x16 to
                      every aligned projection was rejected by the real 14B profile because 1024-column KV rows and
                      ordinary Q4_K rows regressed enough to erase the logits gain. The production dispatch therefore
                      uses x16 only for single projections with at least 32768 output columns; grouped and narrower
                      projections remain on x8. A selective no-fallback acceptance run reproduced the exact token
                      sequence at `360.743 ms/token` (`2.772 tok/s`) and reduced the step-16 Q6_K logits row from
                      `16.486` to `14.049 ms`; overall decode remains within the observed host-frequency variance.
                      F16 scale conversion follow-up completed on 2026-08-01: v4 AVX2 kernels now require and
                      runtime-check F16C, then convert all eight packed FP16 block scales with one `vcvtph2ps` instead
                      of eight calls through the portable scalar decoder. CPUs without AVX2+F16C keep the existing
                      portable path. Exact-parity T8 medians improved Q4_K hidden/up/down from `0.355/0.709/0.715` to
                      `0.283/0.549/0.508 ms` and Q6_K from `0.413/1.16/1.15` to `0.338/0.970/1.00 ms`; T1 gains were
                      larger. The real 14B cache-hit run reproduced identical tokens with no fallback at
                      `314.807 ms/token` (`3.177 tok/s`), reducing latency by `12.7%` from the immediately preceding
                      selective-x16 run and by `7.9%` from the prior best `341.854 ms/token` scalar-conversion run.
                      Two follow-up instruction experiments were rejected on 2026-08-01 after production-shape and
                      full-decode validation. A runtime-gated AVX-512 VNNI x8 path reduced the dot instruction count
                      but regressed T8 Q4_K hidden/up/down from the established `0.283/0.549/0.508 ms` range to about
                      `0.299/0.642/0.709 ms` on the Ryzen 9 9950X; Q6_K also broadly regressed. Sharing one Q4_K byte
                      load across low/high nibbles produced an isolated `0.208 ms` hidden-row sample, but two real
                      decode runs raised the step-16 `5120->5120` helper total from the recent `32.8-37.3 ms` range to
                      `42.8-43.6 ms` and reduced generated-token throughput. Both implementations were removed. Future
                      ISA/layout changes must pass repeated full-decode evidence rather than instruction-count or one
                      microbenchmark evidence alone.
                - [x] Implement production Q4_K/Q6_K x Q8_K GEMV/vec-dot kernels for the top Qwen decode rows.
                      Start with the measured rows: gate/up `1x5120 -> 1x27648`, hidden/output `1x5120 -> 1x5120`,
                      FFN-down `1x13824 -> 1x5120`, KV `1x5120 -> 1x1024`, and logits `1x5120 -> 152064`. Prefer a
                      portable compact kernel first, then add x86 AVX2/AVX512/VNNI variants behind runtime feature
                      checks.
                      AVX2 register-decode slice completed on 2026-08-01: compact x4 kernels now decode Q4_K/Q6_K
                      directly from contiguous blocks and reduce Q8_K dot products without temporary arrays. The real
                      T8 14B cache-hit result improved by `9.34%`, but production acceptance remains open because
                      expanded-v1 was still `17.7%` faster after that slice.
                      Paired-dot follow-up completed on 2026-08-01: two compact columns now share one 256-bit Q8_K dot
                      sequence. Real decode improved from `567.607` to `505.917 ms/token`; the expanded-v1 gap is down
                      to `4.94%`. Next implement an x8 field-interleaved prepared layout and GEMV loop that shares
                      decoded scales/lanes across more output rows, then evaluate AVX512/VNNI.
                      x8 prototype completed on 2026-08-01: field-interleaved-v4 groups eight output rows and converts
                      each four-byte quant chunk into eight Int32 partial sums per AVX2 load. Production-shaped T8
                      single-projection rows improve by roughly `6.6%` to `37%` over paired-dot v3 with exact parity.
                      AOT/grouped wiring completed on 2026-08-01: externalized v4 weights, id-4 MLIR lowering,
                      single-projection and shared-staging grouped 2/3 helpers, CLI/cache isolation, and load-and-run
                      regression coverage are complete. Full 14B decode acceptance subsequently passed, and v4 replaced
                      v3 as the prepared-weight default while v1/v3 remain explicit comparison layouts.
                      Wide-output x16 follow-up completed on 2026-08-01: adjacent x8 v4 groups share Q8_K source
                      broadcasts in one AVX2 tile. Microbenchmarks and real decode rejected blanket x16 routing, so the
                      measured production gate is output width >=32768 for single projections only. This accelerates
                      vocabulary projections while preserving x8 scheduling for the dominant gate/up, hidden, down,
                      and KV rows. This P0 remains open for new kernels targeting those dominant rows.
                      The grouped-v4 benchmark follow-up exposed why the stricter threshold matters: Q4_K gate/up x8
                      evaluates both 13824-column projections in about `2.26 ms` at T1, while a single 13824-column
                      projection routed through x16 took about `2.21 ms`. The threshold was raised from 8192 to 32768
                      so FFN-width projections cannot enter the slower tile; the new grouped benchmark covers Q/K/V and
                      gate/up across the complete T0/T1/T2/T4/T8/T16/T32 matrix with exact reference parity. After the
                      correction, the single Q4_K FFN-up row fell to `1.34/0.359 ms` at T1/T8, about `39%/35%` faster
                      than its former x16 route.
                      The next dominant-row slice replaced scalar FP16 scale decoding inside all v4 AVX2 x8/x16
                      kernels with runtime-gated F16C vector conversion. This produced broad T1/T8 gains and a measured
                      `314.807 ms/token` real decode result, but the remaining projection helpers still account for
                      roughly `288 ms` of the final profiled step and keep this P0 open.
                      A 2026-08-02 scheduling follow-up reduced v4 dynamic task fan-out from roughly eight to four tasks
                      per requested worker. Exact-parity helper tests and all 42 targeted GGUF quantized/decode tests
                      passed. Under the same cache-hit 14B O0/T8/all-prepared profile, generated-token latency improved
                      from `307.082` to `299.370 ms/token` (`-2.51%`); step-16 grouped gate/up, Q6_K down, Q4_K down,
                      and hidden helper totals improved by about `5.8%`, `4.9%`, `2.7%`, and `4.5%` respectively.
                      A separate attempt to keep x8 accumulators in YMM registers across the complete K loop was
                      rejected despite promising short microbenchmarks: real decode regressed to `342.163 ms/token`
                      and every dominant projection family slowed, indicating harmful register pressure/codegen.
                      A static contiguous worker-partition experiment was rejected on 2026-08-02. Replacing the v4
                      helpers' roughly four dynamic tasks per thread with one grain-aligned range per participant
                      regressed T8 grouped Q4_K gate/up from `1.31` to `1.42 ms`, Q4_K FFN-up/down from
                      `0.330/0.321` to `0.388/0.375 ms`, and Q6_K hidden from `0.173` to `0.226 ms`. The code was
                      removed without a full-decode run; dynamic stealing remains useful for cross-domain balance.
                      True AVX-512 x16 Q6_K progress completed on 2026-08-02: the v4 kernel now combines two adjacent
                      x8 groups in ZMM registers and shares each Q8_K broadcast, behind an AVX512F+BW+VL+F16C runtime
                      gate. Exact helper parity and all 42 targeted quantized/decode tests pass. T8 medians improved
                      hidden `0.249 -> 0.173 ms`, FFN-up `0.841 -> 0.746 ms`, FFN-down `1.03 -> 0.730 ms`, and logits
                      `13.5 -> 12.3 ms`; unlike the rejected 256-bit VNNI substitution, the true x16 tile wins across
                      both narrow and wide production rows. A cache-hit 14B run reproduced the exact token sequence
                      with no fallback at `260.166 ms/token` (`3.844 tok/s`), another `1.57%` latency reduction from
                      the immediately preceding `264.313 ms/token` split-kernel baseline. AVX2 x8 remains the fallback
                      on unsupported CPUs, and Q4_K dispatch is unchanged.
                      A matching true AVX-512 x16 Q4_K tile was evaluated and rejected on 2026-08-02. Isolated square
                      hidden microbenchmarks looked faster, but FFN-up/down and logits regressed when broadly routed.
                      After restricting the prototype to square projections, the real 14B helper profile still raised
                      the 48-call Q4_K `5120x5120` bucket from the preceding roughly `25.9` to `28.65 ms/step`, while
                      step-16 helper time rose from about `234.4` to `254.6 ms`. A superficially faster unprofiled run
                      was therefore attributed to host/cache variance, and the Q4_K AVX-512 code and dispatch were
                      removed. Q4_K needs a different decomposition rather than a direct copy of the Q6_K x16 strategy.
                      A MinGW `sysv_abi` experiment also removed the Win64 nonvolatile-XMM save/restore sequence from
                      the private SIMD helpers exactly as intended, but worse whole-kernel register allocation raised
                      the same 14B run from `299.370` to `345.568 ms/token` (`+15.4%`). The native ABI remains the
                      measured production choice; prologue instruction counts alone are not an acceptance signal.
                      Block-level integer reduction completed on 2026-08-02: the Q4_K and Q6_K v4 x8/x16 kernels now
                      accumulate `dot * subblock_scale` (and Q4_K minimum correction) in Int32 for the complete
                      256-element quantization block, matching the production x86 kernel structure before one final
                      Float32 conversion and `d * lhs.d` scale. All 42 targeted GGUF quantized/decode tests passed. The
                      same cache-hit 14B O0/T8/all/v4 run preserved the exact token sequence with no fallback and
                      improved from `299.370` to `277.218 ms/token` (`-7.40%`, `3.607 tok/s`); step-16 helper time fell
                      from about `288` to `243.026 ms`, with grouped gate/up `95.197 -> 85.156 ms`, Q6_K FFN-down
                      `50.331 -> 45.163 ms`, Q4_K FFN-down `40.532 -> 35.745 ms`, and hidden/output
                      `30.062 -> 27.824 ms`. An odd/even Q4_K subblock-template experiment was separately rejected:
                      eliminating the nibble branch expanded the x8 kernel and regressed production-shaped hidden,
                      FFN-down, and grouped gate/up microbenchmarks by roughly `17-40%`.
                      The whole-K x8 accumulator design was re-evaluated after block-level integer reduction. All 42
                      targeted GGUF quantized/decode tests still passed, but production-shaped Q4_K T8 hidden,
                      FFN-up, and FFN-down medians regressed to `0.216/0.399/0.431 ms`; Q6_K measured
                      `0.241/1.27/1.24 ms`. The repeated block-call prologues are cheaper than the resulting register
                      pressure and code-generation loss, so this variant was rejected before an unnecessary 14B run.
                      A production decode without helper profiling measured `278.863 ms/token`, close to the profiled
                      `277.218 ms/token`; helper instrumentation is not the remaining performance gap.
                      A Q6_K x8 paired-segment decode experiment was rejected on 2026-08-02. Sharing each `ql/qh` load
                      across segments 0/2 and 1/3 improved the T8 hidden median from `0.265` to `0.212 ms`, but increased
                      the higher-weight FFN-up/down medians from `0.802/0.896` to `1.12/1.17 ms`; T1 FFN rows were close
                      to twice as slow. The added Q8 broadcasts and live vectors outweigh saved weight loads on large
                      streaming rows, so the production x8 kernel keeps one segment live at a time.
                      Pair-sum/scale folding completed on 2026-08-02: Q4_K x8 now accumulates safe Int16 pair dots and
                      folds pair reduction plus unsigned subblock scaling into one `vpmaddwd`. Q6_K x8 and AVX-512 x16
                      use two Int16 partial sums so the wider six-bit range cannot overflow, then fold each signed scale
                      into the same reduction instruction. An immediate same-host helper A/B reduced step-16 helper
                      time from `244.051` to `238.362 ms`, grouped Q4_K gate/up from `87.409` to `84.598 ms`, and the
                      hidden/output bucket from `27.500` to `26.348 ms`. Production-shaped Q6_K x16 T8 medians improved
                      hidden from about `0.173` to `0.112 ms` and FFN-up/down from `0.746/0.730` to `0.714/0.711 ms`,
                      while logits remained neutral at about `12.3 ms`. Three no-fallback 14B cache-hit runs produced
                      identical text at `259.299`, `249.367`, and `254.126 ms/token`; the `254.126 ms/token` median is
                      `2.32%` below the preceding stable `260.166 ms/token` result. A separate Q4_K paired-nibble load
                      experiment was removed: its cache-hot microbenchmarks improved, but real helper time regressed to
                      `240.044 ms` because extra live vectors and scheduling pressure outweighed duplicate L1-load removal.
                      Shape-aware v4 decode thread caps completed on 2026-08-02. A repeated T4/T8/T16 matrix showed
                      that no single thread count wins all production shapes: Q4_K hidden favored T4
                      (`0.116 ms` median versus `0.173/0.247`), Q4_K FFN-down favored T8 (`0.311 ms` versus
                      `0.465/0.338`), while grouped gate/up, Q6_K FFN-down, and Q6_K logits favored T16
                      (`1.19/0.597/11.4 ms`). Small 1024-column Q4_K/Q6_K projections favored T2. The v4 runtime now
                      treats the configured count as a hard upper bound and applies decode-only limits of T2 for small
                      outputs and T4 for square Q4_K hidden projections. Automatic decode caps all remaining projections
                      at T8; an explicit configured count remains a hard upper bound and may select T16. Batched/prefill
                      and sub-1M-operation work retain the generic policy. Helper profile
                      details report the actual resolved count, and a regression locks the Q4_K square `T8 -> T4`
                      decision. A real 14B profile confirmed `resolved_threads=4` and reduced the 48-call hidden bucket
                      from `26.348` to `25.186 ms`. Three no-profile cache-hit runs generated identical text at
                      `245.097`, `245.177`, and `249.406 ms/token`; the `245.177 ms/token` median is `3.52%` below the
                      preceding `254.126 ms/token` median and within `4.3%` of the `235.1 ms/token` CPU control result.
                      A refreshed five-repetition helper matrix on 2026-08-02 found local T16 wins over T8 for grouped
                      Q4_K gate/up (`1.16 vs 1.32 ms`), Q6_K FFN-down (`0.60 vs 0.73 ms`), and Q6_K logits
                      (`11.49 vs 12.16 ms`), but the matching 16-token whole-model run regressed from T8
                      `255.927/255.594 ms` to T16 `274.278/275.618 ms` mean/median (`+7.17%` mean). Exact output hashes
                      matched. Cross-CCD bandwidth contention therefore outweighs isolated-row gains; the automatic
                      T8 ceiling and Q4_K hidden T4 specialization remain unchanged.
                - [x] Re-run the cache-hit policy matrix after compact Q8_K kernels and only then retune thread/grain
                      defaults. Completed on 2026-08-02: the automatic T16 ceiling regressed the profiled 14B run to
                      `299.960 ms/token`; capping automatic decode at T8 reduced it to `285.087 ms/token`. Alternating
                      unprofiled auto/explicit-T8 runs measured `279-289` and `268-274 ms/token` under visible host
                      variance, so the production default stops at T8 while explicit T16 remains an opt-in experiment.
                - [x] Split CPU GGML sidecars and architecture-specific microkernels out of `CompiledModule.cpp` into
                      focused translation units. Completed on 2026-08-02: the Q4_K/Q6_K v4 AVX2/F16C x8/x16 kernels
                      and their POD layout ABI now live in the internal `Runtime/CPUGGMLV4Microkernels` unit; MLIR,
                      helper ABI, profiling, scalar fallback, and scheduling remain in `CompiledModule.cpp`. All 42
                      targeted quantized/decode tests passed. Touching only the microkernel rebuilt its object and three
                      affected links in `12.996 s`, without compiling `CompiledModule.cpp.obj`, versus `169.5 s` for a
                      monolithic kernel edit (`-92.3%`). The internal layout header is excluded from installation.
                - [x] Move Q8_K activation staging from per-helper temporary work into a decode-step activation-staging
                      cache so the same normalized hidden vector can be quantized once and reused across Q/K/V/O,
                      gate/up/down, and logits projections where shapes and tolerances permit.
                      Progress on 2026-07-06: grouped Q8_K-staged helper wrappers and benchmark rows were added for
                      two-/three-projection GGML helpers. A short Q4_K gate/up T8 smoke showed direct grouped output
                      matches the separate-reference path (`max_abs_delta=0`) and is slightly faster than separate
                      helper calls, while Q8_K-staged grouped rows remained slower and had about `1.36` max absolute
                      delta. Keep staged grouped routing benchmark-only until packed kernels or step-level activation
                      reuse make it win.
                      Progress on 2026-07-08: the first reusable-activation ABI landed for non-grouped helpers, so this
                      item is now blocked on graph/AOT ownership of the staged workspace rather than raw helper
                      availability.
                      Completed on 2026-08-01 by the content-validated field-interleaved-v4 thread workspace described
                      above. This preserves the existing artifact ABI while sharing Q's staged hidden vector with the
                      separate mixed-format K/V helpers. Memory is bounded by the largest activation seen by each
                      calling thread and does not scale with token count or context capacity.
                - [x] Implement low-thread packed GEMV microkernels for Q4_K/Q6_K x Q8_K before retuning thread policy.
                      Completed on 2026-08-02: field-interleaved-v4 provides compact Q8_K activation staging, AVX2 x8
                      Q4_K/Q6_K kernels, a runtime-gated AVX-512 x16 Q6_K kernel, grouped staging reuse, block-level
                      integer reduction, pair-sum/scale folding, and shape-aware low-thread dispatch. All 43 targeted
                      quantized/decode tests pass with exact output and no fallback. Three stable no-profile 14B runs
                      measured a `245.177 ms/token` median, within `4.3%` of the `235.1 ms/token` CPU control result.
                - [x] Add Qwen-shaped packed-kernel benchmark rows for the exact top profile rows and require
                      full-decode profile evidence before switching the default route. Completed the benchmark-surface
                      portion on 2026-07-06: grouped projection helper rows now use the full
                      `T0/T1/T2/T4/T8/T16/T32` GGML thread matrix, so gate/up, hidden/output, KV, and logits rows can be
                      compared around the low-thread region where the CPU-only llama.cpp control run peaks. Acceptance
                      smoke: `GGMLGroupedProjectionHelper/Q4_K/qwen_gate_up/(separate|concatenated)/T2/T4/T8` executed
                      successfully with `max_abs_delta=0` for concatenated rows.
                      target for the actual packed-kernel tranche remains: bring default stateful CPU AOT below
                      `300 ms/generated token` on the local Qwen2.5 14B Q4_K_M control run without increasing
                      residual/fallback share.
          - [x] P0: Run full-decode thread/grain A/B instead of extrapolating from isolated helpers.
                The July 5 helper rows show Q4_K grouped gate/up improving from about `3.60 ms` at T16 to `2.96 ms` at
                T32, while the real decode path resolves default helpers to T16. Validate default/T8/T16/T32 in full
                stateful decode with helper share, residual share, and generated-token TPS before changing defaults.
                First implementation slice completed on 2026-07-05: large GGML block MatMul auto-policy now permits up
                to 32 workers instead of capping at 16, while tiny output-unit counts remain capped at 4/8 workers. The
                same slice hoists grouped-projection column/projection resolution out of the K-block loop. Short
                Qwen-shaped helper validation after the change measured T0/default at about `0.58 ms` for Q4_K
                `5120->5120`, `1.72 ms` for Q4_K `5120->13824`, `17.9 ms` for Q4_K logits, `3.02 ms` for Q6_K
                `5120->13824`, and `33.0 ms` for Q6_K logits. Full decode A/B remains required before closing this
                item.
                CPU-only llama.cpp control run on 2026-07-06: an out-of-tree `llama-bench` build from
                `third_party/llama.cpp` was configured with CUDA/Vulkan/BLAS disabled and MinGW Windows 10 API flags.
                On the same Qwen2.5-Coder-14B Q4_K_M GGUF, TG-only `ngl=0, flash_attn=0` measured about `5.07 t/s`
                at T4, `4.60 t/s` at T8, `3.52 t/s` at T16 with `b=1/ub=1`, `3.34 t/s` at T16 with default
                `b=2048/ub=512`, and `2.62 t/s` at T32. LiteNN stateful CPU AOT on the same prompt measured about
                `0.49 t/s` with `--cpu-aot-threads 4` and about `2.33 t/s` with the default thread policy. This rules
                out simply copying llama.cpp's low thread count as the fix: LiteNN first needs llama.cpp-class
                low-thread quantized projection efficiency and graph-wide task scheduling.
                Follow-up profile on 2026-07-06: `build/qwen_perf_profile_t4_20260706` used explicit
                `--cpu-aot-threads 4` and captured 12 steps. Step time averaged about `2007 ms/generated token`; timed
                helpers accounted for about `98.7%` of total step time. Projection helpers were even more dominant:
                grouped gate/up `41.2%`, FFN-down `27.6%`, hidden/output `16.2%`, logits `8.6%`, and KV `4.5%`.
                Residual/non-helper time did not explode; the kernel itself became the limiter. A fresh CPU-only
                llama.cpp `llama-bench` matrix on the same machine measured about `4.65 t/s` at T2, `4.64 t/s` at T4,
                `4.29 t/s` at T8, `3.47 t/s` at T16, and `2.35 t/s` at T32 with `ngl=0` and `flash_attn=0`. Therefore
                the next thread-policy work should wait for packed kernels; otherwise T4/T8 retuning makes LiteNN worse.
                Matrix tooling completed on 2026-07-06: `benchmark/gguf_decode_thread_matrix.py` runs LiteNN stateful
                CPU AOT decode across auto/T2/T4/T8/T16/T32, supports cache-hit and profile-bundle modes, and redacts the
                model path in saved command manifests. Updated on 2026-07-07: profile-bundle matrix runs also invoke
                      `gguf_decode_compare.py` over the generated summaries and write `profile_summary_compare/` next to the
                      matrix artifacts; the matrix can also forward `--cpu-aot-q8k-staged-matmul` so direct-vs-staged
                      activation paths are captured by the same full-decode A/B harness. Use it for the next full-decode
                      acceptance run before changing default thread policy.
                Completed measurement slice on 2026-08-01 with the v4 all-prepared stateful artifact and identical
                generated tokens. The first cache-hit pass measured T4 `339.890 ms/token`, T16 `305.097`, and T32
                `319.923`; an immediate repeat measured T8 `314.078`, T16 `347.432`, and T32 `326.426`. Ordering,
                temperature, and memory-frequency effects were large enough to reverse the T8/T16 ranking, while T4
                was consistently weak and T32 did not win. Keep explicit T8/current defaults for now; hardware-aware
                retuning remains under the separate post-kernel policy item instead of reopening this measurement task.
                Graph-wide scheduling follow-up completed on 2026-08-01: the persistent CPU AOT pool now signals only
                the workers selected for the current helper, uses a spin barrier for the calling thread, and lets
                participating workers poll briefly for the next helper before sleeping. This replaces per-helper
                `notify_all` wakeups of all 31 workers on the 32-thread host and avoids a kernel sleep/wakeup at each
                short operator. Q4_K T8 hidden/FFN-up/FFN-down medians measured `0.118/0.425/0.466 ms` versus the
                established `0.283/0.549/0.508 ms`; Q6_K hidden measured `0.206 ms` versus `0.338 ms`, all with exact
                helper parity. A cache-hit 14B run preserved the generated token sequence, reduced step-16 helper time
                from `338.182` to `302.936 ms` (`-10.4%`), and reduced the eight-token generation mean from `355.983`
                to `342.513 ms/token` (`-3.8%`). Wide Q6_K/logits rows remain bandwidth/kernel limited.
                A production-profiled v4 grain follow-up on 2026-08-02 reduced dynamic task claims without adopting the
                rejected one-static-task-per-worker policy. Four tasks per worker improved the same 14B T8 run from
                `307.082` to `299.370 ms/token` with identical tokens and no fallback, while preserving work stealing.
          - [x] P0: Add a repository-owned CPU-only llama.cpp control harness.
                Completed on 2026-07-06. `benchmark/run_llama_cpp_control.py` accepts the GGUF path at runtime, locates
                or accepts a `llama-bench` executable, runs a CPU-only TG matrix for T2/T4/T8/T16/T32 by default, and
                redacts `model_filename` in the saved JSON unless `--keep-model-filename` is requested. Example:
                `python311 benchmark/run_llama_cpp_control.py --model <model.gguf> --llama-bench <llama-bench> --output-json build/llama_control.json --output-md build/llama_control.md`.
          - [x] P0: Make `gguf_decode_compare.py` phase-aware for profile-summary inputs.
                Completed on 2026-07-06. `--litenn-profile-summary` now emits separate `profile-summary-all`,
                `profile-summary-prompt_replay`, and `profile-summary-generation` rows when the input bundle contains
                matching steps, and recomputes top helper/operator/node attribution for each selected phase. The default
                Qwen profile now reports the full mixed replay+generation path at about `1.56 t/s`, while the steady
                generated-token phase is visible as about `430.7 ms/token` (`~2.32 t/s`). Use the generation row for
                steady decode acceptance and the all row for end-to-end prompt replay plus decode regressions.
          - [x] P0: Split the current `~143 ms/step` residual into ranked runtime buckets.
                - [x] Profile-bundle residual buckets. Completed on 2026-07-06: `benchmark/profile_bundle.py` now emits
                      `residual_buckets`, `top_residual_steps`, Markdown residual tables, and Chrome trace residual
                      events using the stable `step_ms - helper_total_ms` attribution. This makes prompt-replay,
                      generation, and per-step residual drift visible in comparison artifacts.
                - [x] Decode-loop runtime buckets. Completed on 2026-07-07: GGUF decode `--stream-stats` now reports
                      input preparation, compiled-module runtime, helper-profile emission, logits output, sampling,
                      state update, and unattributed host overhead per token. `benchmark/profile_bundle.py` imports those
                      fields into `runtime_buckets`, per-step JSON, Markdown tables, and Chrome trace events so measured
                      runs can separate module time from CLI/runtime shell costs.
                - [x] Module helper/non-helper split. Completed on 2026-07-07: GGUF decode stream stats now include
                      `helper_total_ms` and `module_non_helper_ms`, and the profile bundle plus decode comparison tools
                      preserve those fields as runtime buckets and comparison columns. This separates sidecar helper
                      time from generated-code/runtime-entry work before full per-node non-helper instrumentation lands.
                - [x] Add stable per-layer/per-node timing for non-helper generated code. Completed on 2026-08-02:
                      opt-in `CompilerOptions::enableCPUAOTNodeProfiling` inserts MLIR markers around every
                      `ExecutablePlan` node and reports stable subgraph/node/schema identity, operation kind, inclusive
                      time, helper time, and exclusive self time. `litenn_gguf_convert --profile-nodes`,
                      `qwen_smoke.py`, and `profile_bundle.py` preserve these events and emit ranked native node-kind
                      totals, per-node JSON/Markdown rows, and trace events. The normal compile/cache path remains
                      uninstrumented, and profile artifacts have distinct cache identities.
                      A three-step cache-hit 14B Q4_K_M O3 diagnostic run measured warm module/helper/non-helper times of
                      `648.992/621.482/27.510 ms` and `583.455/556.285/27.170 ms`. After excluding nested marker callback
                      overhead, total warm native self time was `20.411-20.723 ms`: `UnaryOpNode` led at
                      `9.718-10.430 ms`, nested `CallNode` frames fell to `2.687-2.843 ms`, binary elementwise work was
                      `1.855-1.924 ms`, and grouped plus ordinary quantized wrapper work was `2.611-2.890 ms`.
                      Filtering sub-microsecond rows reduced profile emission to `13.0-15.2 ms/step`; the summary retains
                      full self/helper totals while detailed rows retain active stable ids. Profiling changes helper
                      timing, so these runs are attribution evidence rather than representative throughput measurements.
                - [x] P0: Validate and remove the measured decode wrapper/elementwise overhead. Use focused
                      non-profiled microbenchmarks to separate remaining marker cost from the `~2.7 ms` nested-call
                      bucket, account for the `~6.5-7.1 ms` gap between module non-helper and native self totals, then
                      fuse the `~10 ms` SiLU/unary path with adjacent gate multiplication where legality and numerics
                      permit.
                    - [x] Fuse exact SwiGLU activation and gate multiplication. Completed on 2026-08-02: `BinaryOp::SwiGLU`
                          is now a first-class floating-point operation across shape/type validation, Interpreter,
                          Autograd, LiteNN dialect lowering, and CPU AOT. The Qwen MLP builder emits one node instead of
                          `negate/exp/add/divide/multiply`; a strided rank-2 CPU helper handles grouped-projection column
                          views without materialization. The profiled 48-layer decode schedule fell from `8366` to
                          `8078` node calls (`-288`, `-3.44%`) and removed `UnaryOpNode` from the decode graph. The
                          original inline fused region cost `13.08-15.76 ms/step`; helper plus residual node self time
                          measured `11.4-12.5 ms/step`. An unprofiled 16-step fresh-artifact run averaged
                          `277.804 ms/token`; an immediate cache-hit run, excluding its first page-fault-heavy step,
                          averaged `273.341 ms/token` with a `269.672 ms` median and exact token parity. The historical
                          `245.177 ms` median remains the best recorded result, so this is a structural and modest
                          measured improvement rather than a new throughput record.
                    - [x] Close the module/native-self attribution gap and remove repeated output allocation. Completed
                          on 2026-08-02: the CPU node profiler now measures marker callback time independently and the
                          profile bundle preserves node-self, marker, and module-unattributed buckets. A four-step
                          cache-hit run measured `17.76-19.79 ms` module non-helper time as `11.54-12.51 ms` node self,
                          `5.66-6.28 ms` marker callbacks, and only `0.56-1.11 ms` unattributed module time. The former
                          `~6.5-7.1 ms` gap was therefore predominantly intrusive instrumentation, not production
                          wrapper work. Stateful decode now preallocates its static logits output and calls
                          `RunTensorsInto`, removing one `608256`-byte output allocation per generated token. A noisy
                          warm 15-step sample measured `287.708/290.187 ms` mean/median with exact token parity; it does
                          not establish a throughput gain over the historical best.
                - [x] P0: Validate selective decoder-block call inlining with a non-profiled A/B artifact. Rejected on
                      2026-08-02 after a real 14B Q4_K_M interleaved comparison: forced LLVM always-inlining reduced
                      the module from 52 functions and 245883 instructions to 2 functions and 140672 instructions,
                      but increased the object from 1054155 to 1080114 bytes and compile-artifact time from
                      `22435.718 ms` to `27538.612 ms` (`+22.7%`). Exact generated token ids matched, while steady
                      no-inline versus inline decode measured `264.743/263.721 ms` versus `265.704/264.153 ms`
                      mean/median (`+0.36%/+0.16%`). The apparent `3.59 ms/step` CallNode self bucket was therefore
                      instrumentation cost rather than removable production overhead. The experimental option and
                      pass were removed; optimization returns to the dominant quantized helpers.
                - [x] Remove full-sort and repeated-history-scan overhead from greedy sampling. Completed on
                      2026-08-01: greedy and zero-temperature sampling now perform one stable argmax pass over logits,
                      build repeat-penalty membership once only when enabled, and consume the last Tensor logits row
                      through a view instead of copying it. A cache-hit Qwen2.5-Coder-14B Q4_K_M T8 run preserved the
                      exact generated token sequence while reducing generation sampling from the previous
                      `~9-11 ms/token` to `0.119-0.221 ms/token` (`0.162 ms` mean over eight tokens). Sampling is no
                      longer a material share of steady decode latency.
          - [x] P1: Skip full-vocabulary logits projection for prompt replay steps that cannot be sampled.
                The July 5 run spends about `53 ms` per logits projection and executes one on every prompt replay step.
                Skipping all but the last replay logits improves prompt/prefill latency, though it is not a steady-state
                generated-token TPS fix.
                Completed on 2026-08-02: stateful dense and paged-reference schedules accept an `emit_logits` Bool and
                place the lm-head behind a compiled `CondNode`. The CLI passes false for replay-only tokens and true for
                the last prompt token plus generated tokens; KV and position state aliases update on both branches and
                the public output remains `logits`. The artifact cache version advanced to v7 while shared physical
                weights remain separately reusable. A six-token 14B Q4_K_M profile showed five replay steps with
                `2641` helper calls and no vocabulary projection; the generation step had `2642` calls and exactly one
                `1x5120 -> 152064` Q6_K logits helper taking `13.997 ms`. All 91 `GGUFImporterTest` cases pass,
                including compiled false/true branch parity and paged schedule validation.
          - [ ] P1: Add a sampler-only logits path for text generation when public logits are not requested.
                This is deferred behind dominant projection work after a 2026-08-02 upper-bound audit. The current
                Q6_K lm-head scans about 638 MB of quantized weights and measures `12.0/11.9 ms` mean/median at T16,
                about `5%` of a `245-270 ms` step. Exact greedy or top-k sampling cannot avoid that weight scan; a
                fused projection+sampler would mainly remove the 608256-byte logits write and about `0.16 ms` of host
                sampling. Revisit after gate/up/down and QKV projection costs fall, or when approximate/hierarchical
                vocabulary selection becomes an explicit model contract.
          - [x] P1: Eliminate repeated per-head RoPE transcendental work after projection work moves.
                RoPE is not the current top bottleneck, but the default profile still shows `55296` helper calls over
                24 steps and about `13-14 ms` per decode step. Once projection kernels are reduced, convert per-head
                RoPE calls into a batched per-layer helper or fuse it into the Q/K layout path.
                Progress on 2026-08-01: the CPU helper now keeps one bounded thread-local frequency/angle table keyed by
                `(headDim, base, frequencyScale, position)`. Stateful decode therefore computes `pow/sin/cos` once for
                the current token instead of repeating them across all 2304 Q/K head calls; the cache retains only one
                head-sized entry and does not grow with the requested context length. A cache-hit Qwen2.5-Coder 14B
                Q4_K_M T8 run reduced the 2304-call RoPE row from about `15.9 ms/step` to `0.287 ms/step` (`-98.2%`)
                with identical generated tokens and no fallback. The residual dispatch cost is now below `0.3 ms/step`,
                so structural batching/fusion is no longer justified without a future profile regression.
    - [x] P2: Add context-extension validation gates before reporting long-context readiness. Completed on 2026-07-05:
          `ValidateLLaMAContextRequest` rejects requests beyond model context, requires explicit RoPE scaling metadata
          when exceeding the original trained context, accepts implemented linear scaling within its factor-derived
          limit, and blocks YaRN/LongRoPE long-context execution until their runtime formulas have golden coverage.
    - [x] P2: Add a repeatable long-context matrix harness. Completed on 2026-07-05:
          `benchmark/gguf_context_matrix.py` drives qwen smoke rows for `2k,32k,128k,1m`, supports dry-run command
          inspection, paged-reference/cache controls, writes JSON/Markdown summaries, and can attach per-target profile
          bundles with `--profile-bundles`. The remaining work is to run the matrix on a real model and attach the
          resulting measurements.
    - [x] P0: Add production-shaped GGML helper benchmark rows for the real Qwen decode dimensions:
          `5120->5120`, `5120->1024`, `5120->13824`, `13824->5120`, and `5120->152064`.
          The current `4096->4096` row is useful but under-specifies the 337-projection full-step workload.
          Completed on 2026-07-02: `litenn_bench` now registers named `baseline4096`, `qwen_hidden`, `qwen_kv`,
          `qwen_ffn_up`, `qwen_ffn_down`, and `qwen_logits` rows for every supported GGML helper format and `T1/T16`.
    - [x] P0: Cache Q4_K/Q5_K activation subblock sums inside `litenn_cpu_ggml_block_matmul_f32`.
          This removes repeated `lhsSum` accumulation for every output column while preserving the direct Float32
          reference-style dot path. Validation slice: `GGMLBlockMatMulHelper/Q4_K/qwen_kv/T1` measured about `3.03 ms`
          before the slice and about `1.91 ms` after the split hot-path implementation; `Q4_K/baseline4096/T1` measured
          about `6.09 ms` after the change.
    - [x] P0: Tile the Q4_K direct CPU helper across four output columns.
          `litenn_cpu_ggml_block_matmul_f32` now shares each activation-block scan across a four-column Q4_K tile while
          keeping the existing Float32 accumulation semantics. Short validation on 2026-07-02 passed the stateful GGUF
          logits parity test and measured `GGMLBlockMatMulHelper/Q4_K/qwen_kv/T1` at about `1.73 ms` CPU and
          `Q4_K/baseline4096/T1` at about `5.79 ms` CPU; the same `qwen_kv` helper row measured about `0.33 ms` CPU at
          `T16`, so the remaining thread-model work is full-decode scheduling and grain selection rather than a blanket
          rejection of helper-level parallelism.
    - [x] P0: Tile the Q6_K direct CPU helper across four output columns.
          This extends activation-scan reuse to FFN-down-style Q6_K projections while preserving the current Float32
          accumulation semantics. Short validation on 2026-07-02 passed the stateful GGUF logits parity test and measured
          `GGMLBlockMatMulHelper/Q6_K/qwen_ffn_down/T1` at about `49.9 ms` CPU and the matching `T16` row at about
          `4.58 ms` CPU. This remains a major single-thread hotspot until Q8_K activation-staged vec-dot kernels land.
    - [x] P0: Tile the Q8_0 and Q5_K direct CPU helper paths across four output columns.
          The grouped-output helper now covers all currently supported GGML direct MatMul formats. Validation on
          2026-07-02 passed `GGUFLLaMAQuantizedExecution.*`; a short helper run measured `Q8_0/qwen_kv/T1` at about
          `2.03 ms` CPU and `Q5_K/qwen_kv/T1` at about `3.67 ms` CPU.
    - [x] P0: Implement and evaluate a llama.cpp-style activation-staged GGML block MatMul kernel family. The current
          production helper still computes GGML_Q4_K/Q5_K/Q6_K/Q8_0 blocks directly against Float32 activations unless a
          format-specific staged route is explicitly enabled. llama.cpp stages Float32 activations into Q8_K work buffers
          and calls Q4_K/Q6_K x Q8_K vec-dot kernels with tiled scheduling and architecture-specific packed/repacked
          variants, so the staged family remains a targeted optimization track rather than the global default.
          - [x] Add a scalar Q8_K-staged helper prototype and benchmark rows without switching the default AOT helper.
                Validation on 2026-07-03 passed exact-activation parity for Q4_K/Q5_K/Q6_K. Short helper measurements
                showed the scalar staged path is not a default-switch candidate yet: Q4_K `qwen_kv/T1` was slower
                (`~2.78 ms` CPU staged vs `~2.60 ms` CPU direct in that run), while Q6_K `qwen_ffn_down/T1` only improved
                modestly (`~66.4 ms` CPU staged vs `~70.3 ms` CPU direct) and carries activation-quantization deltas.
          - [x] Add a guarded AVX2 16-lane dot primitive for the Q8_K-staged Q4_K/Q5_K/Q6_K path and keep it behind
                runtime CPU feature detection. Validation on 2026-07-03 passed
                `GGUFLLaMAQuantizedExecution.Q8KStagedHelperMatchesDirectHelperForExactActivationRows`; short helper
                measurements showed the AVX2 staged path helps Q6_K single-thread (`qwen_ffn_down/T1` about `41.7 ms`
                CPU staged vs `44.3 ms` CPU direct in that run) but Q4_K/Q5_K remained slower than the direct helper.
                Do not switch the default globally until the policy is format-specific and accuracy-aware.
          - [x] Add an explicit CPU AOT opt-in route from GGML_Q6_K matmul placeholders to the Q8_K-staged sidecar,
                while keeping the default path on the numerically stricter direct helper. Q4_K/Q5_K/Q8_0 remain direct
                until they have measured production-shape wins. The GGUF decode CLI exposes this as
                `--cpu-aot-q8k-staged-matmul` for A/B profiling.
          - [x] Expose GGUF decode CPU AOT tuning as CLI flags for reproducible A/B runs:
                `--cpu-aot-threads`, `--cpu-aot-affinity`, `--cpu-aot-llvm-opt-level`,
                `--cpu-aot-parallel-min-flops`, and `--compile-diagnostics` / `--no-compile-diagnostics`.
                Artifact-affecting options are part of the decode cache key.
          - [x] Guard CPU AOT O3 for state-alias decode schedules. Completed on 2026-07-05: Qwen stateful decode with
                `--cpu-aot-llvm-opt-level 3` reproduced a Windows access violation on the first decode step, while O1/O2
                completed. CPU AOT now strips alias-sensitive LLVM attributes around state-alias entry wrappers and
                downgrades only state-alias schedules requested as O3 to effective O2 with an explicit compile
                diagnostic. Non-state-alias CPU AOT artifacts can still use O3. True O3 alias-safety proof remains a
                follow-up before reenabling O3 for mutable-state decode entries.
          - [x] Add VNNI, repacked-weight, or other architecture-specific vec-dot kernels for the Q8_K-staged path, then
                re-run the direct-vs-staged helper table before changing the compiler/runtime default. The current AVX2
                implementation uses a u8*s8 `maddubs` pairwise dot for Q4_K/Q5_K/Q6_K staged lanes. Validation passed
                `GGUFLLaMAQuantizedExecution.Q8KStagedHelperMatchesDirectHelperForExactActivationRows`; a short
                2026-07-04 helper run still rejected a global default switch (`Q4_K/qwen_kv/T1` direct `~3.12 ms` CPU vs
                staged `~15.6 ms` CPU, `Q6_K/qwen_ffn_down/T1` direct/staged both about `46.9 ms` CPU but staged worse
                in real time). Keep staged routing explicit/format-gated until a later packed-weight or VNNI kernel wins.
    - [x] P0: Add a measured thread/grain policy for decode-shaped quantized projections.
          `requestedThreadCount == 0` now takes an auto policy instead of blindly using every hardware thread: it caps
          GGML block MatMul helpers at 16 workers, applies smaller caps to tiny output-group counts, and keeps explicit
          `T1/T2/T4/T8/T16/T32` requests unchanged. The helper benchmark matrix now includes `T0/T1/T2/T4/T8/T16/T32`
          rows. A short 2026-07-03 run showed `T0` tracks the conservative cap (`Q4_K/qwen_kv` about `0.30 ms` real,
          `Q6_K/qwen_ffn_down` about `3.28 ms` real) while explicit `T32` remains available for isolated helper cases
          where it wins.
          - [x] Make worker waiting an explicit CPU AOT scheduling policy. Completed on 2026-08-02: the compiler API,
                runtime helpers, GGUF CLI, Qwen driver, cache key, reports, comparison table, and thread-matrix driver
                carry `Adaptive`, `LowPower`, or `Latency` without adding a helper ABI argument. Adaptive adjusts a
                bounded poll window from observed arrivals, LowPower blocks immediately, and Latency uses the longest
                bounded poll. A real 14B Q4_K_M T8/all/v4/O0 16-token run measured mean/median Adaptive
                `255.927/255.594 ms`, LowPower `277.163/276.340 ms`, and Latency `252.958/252.650 ms`; generated-text
                hashes matched. The 1.16% Latency advantage is too small to justify its sustained polling as the
                library default, so Adaptive remains the balanced policy.
    - [x] P1: Stop using monolithic max-cache-length-shaped CPU AOT decode artifacts as the default long-context path.
          - [x] Make the GGUF decode-loop CLI default to the stateful runtime-schedule path. Completed on 2026-07-04:
                `--run-llama-*-decode-loop` now builds/loads the logits-only public-output stateful schedule unless
                `--functional` is passed explicitly for compatibility or diagnostics.
          - [x] Replace the remaining max-cache-length-shaped paged-reference stateful function signature with
                page-table state bindings. Completed on 2026-07-05: `--paged-reference-decode` accepts
                `--paged-resident-pages` and can run after initializing page metadata from compiled input specs;
                resident KV backing shape is `[2,residentPages,pageSize,kvHeads,headDim]` while logical capacity
                remains in page-table metadata/cache key.
    - [x] P1: Replace dense full-capacity KV state with paged-KV execution. Active-prefix attention removes inactive suffix
          scans from attention, but the 1M-context target still requires page tables, active-length metadata, and
          capacity-independent artifact shapes.
          - [x] Add the paged-KV runtime-state ABI/manifest/planner contract. Completed on 2026-07-04:
                `RuntimeStateBinding` can carry `PagedKVCache` layout metadata, vNext packages round-trip it, and dynamic
                GGUF decode planning marks KV cache states with page size, logical capacity, resident page count,
                plane offsets, and token/page strides. This is the contract step only; the current CPU decode graph still
                uses the dense capacity-shaped fallback signature until paged lowering lands.
                Updated on 2026-07-05: the dynamic decode backing state shape is now true paged
                `[2, residentPages, pageSize, kvHeads, headDim]`; only the temporary function input/output binding
                remains dense.
          - [x] Publish explicit page-table/page-descriptor/active-length runtime states. Completed on 2026-07-04:
                dynamic GGUF decode schedules now allocate visible auxiliary `RuntimeStateBinding`s derived from the
                paged KV layout, and vNext manifests persist the descriptor-state name. Lowering still needs to consume
                these states before the dense fallback signature can disappear.
          - [x] Attach runtime-state requirements to stateful artifact entries. Completed on 2026-07-04:
                vNext manifest construction now fills empty artifact-entry `requiredStateBindings` from the runtime
                schedule, so paged decode packages expose the KV/page-table/page-descriptor/active-length/position
                dependencies needed by future cache-hit loaders and dispatchers.
          - [x] Define host-side paged KV initialization semantics. Completed on 2026-07-04:
                runtime helpers initialize invalid page-table entries, fixed four-column page descriptors, active
                lengths, and checked prefix mapping into resident pages so future paged attention/state kernels share a
                stable state format.
          - [x] Publish structured long-context attention execution plans. Completed on 2026-07-04:
                LLaMA artifact planning now names the implemented CPU active-prefix path and planned CPU/CUDA/Vulkan
                paged-attention paths, including page size, max context, full-mask avoidance, streaming decode, and
                required paged KV runtime states for diagnostics and lowering.
          - [x] Add the CPU paged-attention reference execution path. Completed on 2026-07-05:
                `GroupedPagedAttentionNode` executes against explicit KV/page-table/page-descriptor/active-length state
                in the interpreter and is validated against dense grouped active-prefix attention. This closes the
                semantic reference for paged lowering; dynamic decode still needs to emit the paged node instead of the
                dense fallback signature.
          - [x] Replace the dense fallback paged-reference decode signature with page-table/page-descriptor state
                bindings so cache-hit artifacts stop scaling directly with max context length. Completed on 2026-07-05:
                `GroupedPagedAttentionNode` has CPU AOT sidecar lowering and `pagedResidentPageCount` lets resident
                backing capacity stay independent from logical `max-cache-length`.
          - [x] P2: Add graph-side paged KV writeback for full decode-loop execution. Completed on 2026-07-05:
                `PagedKVAppendNode` appends current K/V into explicit paged KV/page-table/page-descriptor/active-length
                graph state, GGUF paged-reference decode now returns those updated states, and the runtime schedule
                aliases them back to the input state buffers while keeping logits as the only public output. The current
                implementation is a correctness/reference lowering; in-place and evicting kernels remain the performance
                follow-up.
    - [x] P1: Decouple persistent AOT instruction cache from model-weight storage.
          - [x] Deduplicate GGUF decode AOT cache weights across instruction-cache variants for the same source model.
                Completed on 2026-07-04: cache entries now write `weights.path.txt` pointing to a model-level shared
                weight blob under the cache root, while metadata/constants/instructions remain per artifact. Legacy
                per-cache `weights.bin` entries still load. This removes repeated multi-GB weight writes when tuning AOT
                flags or thread policy.
          - [x] Load decode AOT cache hits through borrowed separated regions. Completed on 2026-07-04:
                rvalue separated-artifact borrowed loading keeps the cache artifact alive inside the CPU module, avoiding
                a second constants/weights copy after reading the shared cache blob.
                Updated on 2026-07-05: cache hits map the shared weight store as a borrowed separated-artifact weights
                region instead of reading the multi-GB blob into a temporary vector.
          - [x] Make shared-weight cache identity include the compiled external tensor layout and content checksums.
                Completed on 2026-08-01 and refined on 2026-08-02: shared weights are nested under a deterministic
                physical payload identity derived from total bytes plus sorted offset/size/content-checksum tuples.
                Tensor names, alignment declarations, metadata ordering, and compiler flags no longer duplicate the
                same payload; changed ranges or bytes remain isolated and artifact metadata retains strict ABI checks.
                Population writes a complete unique staging directory and atomically renames it, so concurrent writers
                publish exactly one payload and reuse it without exposing a partial multi-GB file.
          - [x] Remove graph materialization and repeated multi-GB validation from trusted cache hits. Completed on
                2026-08-01: the importer has a metadata-only path (`35.6 ms` on the 14B control), stateful cache hits
                create inputs from the compiled module ABI, and full tensor payload import occurs only on cache miss.
                Generic separated-artifact APIs retain full checksums; the internal content-addressed cache uses an
                explicit trusted-borrowed factory after complete-marker, size, layout-identity, metadata, constants,
                instruction, alignment, and tensor-range validation. Cache `build_ms` fell from about `31.4 s` to
                `18.8 ms`; artifact read/load fell to `4.5/5.5 ms`, sampled peak working set fell from about `26.8` to
                `8.4-8.8 GiB`, and generated tokens remained identical. The skipped sequential read appears as normal
                demand paging in the cold first step rather than blocking every process startup.
          - [x] P2: Replace repeated cache-local weight blobs with borrowed/mapped shared weight regions. Completed on
                2026-07-05: cache hits mmap the model-level shared weight store through borrowed separated regions.
                Directly borrowing GGUF/source-package tensor offsets is deferred until separated-artifact metadata can
                encode source offsets and stable checksums instead of only compiled-weight-region offsets.
    - [x] P1: Add a verified in-place KV append sidecar before the full paged-KV migration.
          `litenn_cpu_scatter_update_axis0_f32_rank3` now has direct regression coverage for both same-buffer in-place
          append and distinct-output copy semantics, and stateful decode schedule coverage confirms projected cache
          outputs alias their input buffers. The `KVScatterUpdateHelper` benchmark records the cost boundary: on
          2026-07-03, Qwen-shaped alias append rounded to `0.000 ms`, while copy mode measured about `0.210 ms` for a
          2048-token cache and about `1.71 ms` real / `1.41 ms` CPU for an 8192-token cache.
    - [x] P0/P1: Add grouped LLM decode helpers after operator timing is available: fused/concatenated QKV projection,
          fused gate/up projection for SwiGLU, and grouped active-prefix attention per KV head. Projection grouping is
          P0; attention grouping is P1 because it scales with active context length.
          - [x] Add grouped-projection benchmark rows that compare separate Q/K/V or gate/up helper calls against
                concatenated output-major GGML weights using the existing helper ABI. A short 2026-07-03 Q4_K run
                validated `max_abs_delta=0`; `qwen_qkv/T0` improved from about `1.37 ms` real separate to `1.07 ms`
                concatenated, while `qwen_gate_up/T0` improved from about `3.49 ms` to `3.17 ms`.
          - [x] Add AOT lowering that recognizes same-input compatible projection groups and emits a concatenated
                helper call or a multi-output sidecar without copying model weights at runtime. Completed on
                2026-07-04 with `GroupedQuantizedMatMulNode`, Q/K/V and gate/up layer helpers, executable-plan
                round-trip support, CPU MLIR lowering to `litenn_cpu_ggml_block_grouped_matmul2_f32` /
                `litenn_cpu_ggml_block_grouped_matmul3_f32`, and a projection-span sidecar that accepts independent
                output-major GGML rhs memrefs. Validation passed the grouped Q4_K AOT regression, the quantized
                projection storage preservation regression, and `GGUFLLaMAQuantizedExecution.*`. On 2026-07-07 this
                grouped route gained opt-in prepared Q4_K/Q6_K external-weight support without concatenating weights;
                the grouped Q4_K AOT regression now also validates
                `litenn_cpu_ggml_block_grouped_matmul3_q4k_prepacked_f32`.
          - [x] Add active-prefix attention helper benchmark rows for Qwen-shaped rank-3 KV caches. A short 2026-07-03
                run measured one KV-head helper call at about `0.022 ms` for 128 active rows, `0.470 ms` for 2048 rows,
                and `2.53 ms` for 8192 rows, which makes grouped KV-head/online-softmax work measurable before adding a
                new kernel.
          - [x] Cache per-row attention scores inside the CPU active-prefix helper so max, denominator, and value
                aggregation no longer recompute the query-key dot product. Validation on 2026-07-03 passed the CPU AOT
                decode parity tests; helper timing improved to about `0.006 ms` for 128 rows, `0.099 ms` for 2048 rows,
                and `0.651 ms` for 8192 rows.
          - [x] Add a grouped active-prefix attention CPU sidecar ABI and benchmark rows that compare Qwen-shaped GQA
                grouped execution against repeated per-query-head rank-3 helper calls. A short 2026-07-03 run validated
                `max_abs_delta=0`; 128 active rows stayed roughly neutral (`0.397 ms` grouped vs `0.410 ms` repeated),
                while 2048 active rows improved from about `9.30 ms` repeated to `8.31 ms` grouped.
          - [x] Route GGUF capacity decode graphs through `GroupedActivePrefixAttentionNode` and lower that node to the
                grouped CPU AOT sidecar. The builder still applies RoPE per query/KV head before grouping, and the
                grouped helper remains a conservative CPU sidecar rather than a final KV-head-tiled attention kernel.
  - [x] Sampling raw capture: optional Linux `perf record` and Windows `xperf` ETW wrappers capture raw platform
    evidence beside the bundle when requested.
  - [x] Sampling normalization: collapsed-stack inputs are merged and converted to `speedscope.json`.
  - [ ] Platform-native sampling import adapters normalize raw formats into the collapsed-stack path.
  - [x] Linux `perf`: `--sampler linux-perf` now runs `perf script` automatically after capture, records the redacted
          script/diagnostics, folds repeated callchains into collapsed stacks, and emits Speedscope plus SVG/HTML flame
          graphs in the same bundle. Individual samples retain monotonic timestamp, PID/TID, CPU, command, and full
          stack as `platform.sampling` instant events in the merged `trace.json`, allowing Linux samples to align with
          command and imported decode spans. `--skip-sampler-import` retains raw-only capture when conversion cost is
          unwanted.
    - [ ] Windows ETW/xperf and macOS Instruments import remain pending; raw Windows ETL capture is already available.
  - [x] Flame graph output: collapsed-stack inputs render a simple built-in SVG/HTML flame graph; external renderers can
    still be added later for richer presentation.
  - Bundle output: raw logs, `trace.json`, `speedscope.json`, collapsed stacks, benchmark/profile CSVs, anonymized
    model/backend metadata, and a short Markdown bottleneck summary.

## P0: CUDA Native Hot-Path Fixed Costs

Goal: remove per-call host overhead that makes native CUDA MatMul appear 80-170x slower than PyTorch CUDA.

Status: implemented on 2026-05-16 for the current CUDA native runtime hot path.

- [x] Persist cuBLAS handles instead of creating/destroying one handle per MatMul call.
  - Implementation: `TryCUBLASMatMul` now uses a thread-local per-device handle cache and rebinds the stream per call.
  - Validation: `CUDANativeMatMul/batch:1/width:128` dropped from millisecond-level timing to `0.028 ms`.
- [x] Cache `CUfunction` lookup results.
  - Implementation: `CUDADriverModule` now owns a guarded function cache and supports eager `CacheFunction` at load time.
  - Validation: PTX function lookup is no longer a per-launch driver call.
- [x] Reuse CUDA native workspace allocations.
  - Implementation: `CompiledModule<CUDA>::Impl` owns one max-sized workspace buffer for the loaded native payload.
  - Validation: stable-shape payload execution no longer allocates/frees workspace inside each `RunInto`.
  - Safety boundary: asynchronous `RunInto` with a non-empty shared workspace is rejected until a workspace pool or event-owned lifetime model is added.
- [x] Rename CUDA bridge benchmark entries to CPU fallback terminology.
  - Implementation: benchmark entries now use `CUDACPUFallbackRunInto`.
  - Validation: benchmark list output makes native CUDA and CPU fallback paths visually distinct.

P0 validation run:

| Benchmark | Real time |
| --- | ---: |
| `CUDANativeMatMul/batch:1/width:128` | `0.028 ms` |
| `CUDANativeMatMul/batch:32/width:128` | `0.031 ms` |
| `CUDANativeMatMul/batch:128/width:128` | `0.029 ms` |
| `CUDANativeMatMul/batch:512/width:128` | `0.028 ms` |

## P1: CUDA Native Whole-Graph Scheduling

Goal: make real model benchmarks run through CUDA native instead of falling back to CPU AOT.

Status: implemented for fused inference Linear/MLP chains on 2026-05-16. Optional CUDA Graph replay is now
available for pointer-stable `RunInto` invocations through `CompiledModuleCUDARunOptions::enableGraphReplay`.

- [x] Add a static launch scheduler for single-subgraph CUDA native graphs.
  - Implementation: `CUDANativeInstructionPayload` launch tables now support mixed library-call and PTX kernels.
- [x] Allocate hidden activations from payload workspace.
  - Implementation: non-final fused layer outputs use native payload workspace and loaded modules reuse one workspace buffer.
- [x] Compile Linear and MLP chains into mixed launch payloads: cuBLAS MatMul plus MLIR/NVPTX epilogues.
  - Implementation: optimized `FusedOpNode(MatMulBiasAdd/ReLU)` chains with graph variables/constants lower to
    `litenn_cublas_matmul_f32` plus generated epilogue kernels.
  - Payload ABI: added constant tensor storage so static/shared-library loaded artifacts can carry model weights.
- [x] Add artifact inspection and CUDA runtime tests for multi-layer native MLP graphs.
  - Validation: `CompiledModuleCUDATest.CompilerArtifactsExposeNativeLinearChainPayload` and
    `CompiledModuleCUDATest.RunsNativeLinearChainWithConstantsAndWorkspace`.
- [x] Add optional CUDA Graph capture/replay after the launch table scheduler is stable.
  - Implementation: `CompiledModule<CUDA>` captures and caches `cudaGraphExec_t` per input/output pointer binding
    for synchronized default-stream CUDA-native `RunInto`; capture does a non-captured warm-up first so cuBLAS
    handles are initialized outside stream capture.
  - Validation: `CompiledModuleCUDATest.RunsNativeLinearChainWithCUDAGraphReplay`.

P1 validation spot check:

| Benchmark | AOT RunInto | CUDA Native RunInto |
| --- | ---: | ---: |
| `Linear(784->10)/batch:512` | `0.053 ms` | `0.060 ms` |
| `MLP(784->128->10)/batch:512` | `0.336 ms` | `0.118 ms` |
| `MLP(784->512->256->10)/batch:512` | `1.76 ms` | `0.234 ms` |

P1 CUDA Graph replay spot check:

| Benchmark | CUDA Native RunInto | CUDA Graph RunInto |
| --- | ---: | ---: |
| `Linear(784->10)/batch:512` | `0.060 ms` | `0.031 ms` |
| `MLP(784->128->10)/batch:512` | `0.118 ms` | `0.054 ms` |
| `MLP(784->512->256->10)/batch:512` | `0.234 ms` | `0.069 ms` |

## P2: CPU AOT Intra-Op Parallelism

Goal: close the gap with PyTorch CPU 16T on large batch and large hidden sizes.

Status: active again as of 2026-06-19. The initial implementation landed on 2026-05-19. The old 2026-05-16 experimental Linear/MLP runtime fast path was
removed after instruction-level profiling and focused benchmark validation. It lowered fused linear chains into calls
to a scalar C++ row kernel plus per-call thread creation, bypassing the MLIR-generated packed/zmm FMA kernel. The new
path keeps the small/medium default MLIR path and only tries a persistent-pool sidecar helper for large static f32 fused
Linear/MLP chains. Recent internal T1/T16 benchmark runs show that the sidecar path still needs production tuning: it can
beat simple rows, but it can also lose to the packed MLIR fallback on wide MLP shapes.

- [x] Profile the default CPU AOT object path at instruction level.
  - Result: generated objects use packed `zmm` FMA instructions and have no gather/scatter in the tested MNIST-like
    Linear/MLP cases.
- [x] Investigate and retire the experimental fast path.
  - Result: removed the extra runtime ABI, env controls, benchmark entries, and correctness test tied to the retired path.
- [x] Add a persistent worker pool for CPU AOT helper kernels.
  - Implementation: the pool is process-local, reuses worker threads, and only waits for workers participating in the
    current operation.
- [x] Add a guarded large-static-f32 fused Linear/MLP parallel path.
  - Implementation: `TryCompileCPUParallelLinearChainF32` emits an object that calls
    `litenn_cpu_matmul_bias_relu_parallel_f32`.
  - Gating: `LITENN_CPU_AOT_THREADS=1` falls back to MLIR; `LITENN_CPU_AOT_PARALLEL_MIN_FLOPS` defaults to `1 << 28`.
    Benchmark/profile entry points also honor `LITENN_COMPILE_DIAGNOSTICS=1` and forward it into
    `CompilerOptions::enableCompileDiagnostics` so actual sidecar selection/rejection reasons can be compared with
    `litenn_profile`'s predicted/object columns.
- [x] Improve the helper's local kernel quality enough for the large benchmark to benefit.
  - Implementation: row-bias initialization uses `memcpy`; helper pointers carry restrict semantics; GCC is given
    ivdep hints for the inner contiguous column loops.
- [x] Add the first cache-reuse microkernel improvement for the sidecar helper.
  - Implementation: the helper now processes rows in blocks of four so each RHS row load is reused across multiple
    output rows before advancing `k`.
- [x] Add benchmark labels for CPU AOT thread-policy comparison.
  - Implementation: `AOTRunIntoT1`, `AOTRunIntoT16`, `TrainCPUAOTT1`, and `TrainCPUAOTT16`.
- [x] Add correctness coverage for the new branch.
  - Validation: `CompiledModuleTest.CPUParallelLinearChainMatchesInterpreter` forces the branch and compares with the
    interpreter.
- [x] Add a shape-aware gate for the sidecar path.
  - Requirement: use the helper only when `m/k/n`, total FLOPs, and estimated thread overhead predict a win over the
    packed MLIR fallback; narrow final projections should not drag a whole fused chain into a slower path.
  - Implementation: the current conservative gate keeps very large row counts on the packed MLIR fallback and only
    enables multithread helper calls for sufficiently wide/high-FLOP layers; narrow tail projections use one helper
    thread inside an otherwise eligible chain. Tests and diagnostic runs can still force the sidecar path by setting
    `CompilerOptions::cpuAOTParallelMinFlops` to `1`.
- [x] Add a configurable worker-affinity policy for multithread experiments.
  - Implementation: `CompilerOptions::cpuAOTAffinityPolicy` defaults to `None`; `Compact` builds a topology-aware
    physical-core-first target list, pins the caller plus persistent workers while the policy remains active, and
    restores their prior processor-group/CPU-set affinity when the policy is disabled or the threads exit. Windows
    uses processor-core relationships and full group affinity; Linux respects the process's allowed CPU set and groups
    logical CPUs by package/core before placing SMT siblings. Benchmark/profile entry points can set it explicitly.
  - Evaluation: local Windows `MLP(784->512->256->10)/batch:128` sidecar-helper runs with
    `LITENN_CPU_AOT_THREADS=16` and `LITENN_CPU_AOT_PARALLEL_MIN_FLOPS=1` did not improve with compact affinity; the
    measured real time regressed versus no affinity. A 2026-08-02 audit found that the original logical-index mapping
    assigned slots 1 through 7 to only four physical cores on the 16-core/32-thread Windows reference host. The fixed
    physical-core mapping reduced a cache-hit 14B O0/T8/all/v4 decode from the broken Compact result of
    `765.725 ms/token` to `454.803 ms/token`, but it remained slower than scheduler-managed `None` at
    `277.218 ms/token`: one compact 32 MiB L3 domain cannot supply this streaming-weight workload as effectively as
    the scheduler's cross-domain placement. Keep `None` as the decode default; a future bandwidth-aware scatter policy
    requires an independent acceptance matrix rather than changing `Compact` semantics.
  - Bandwidth-oriented follow-up completed on 2026-08-02: added an explicit `Spread` policy and propagated
    `none|compact|spread` through every CPU helper, compiler option, GGUF CLI, Qwen smoke driver, benchmark environment,
    and thread-matrix entry point. Spread interleaves the lower/upper halves of the physical-core-first topology before
    SMT siblings, preserving Compact semantics and prior affinity restoration. Correctness passes for both policies.
    Current 14B T8 cache-hit samples were Spread `240.774/260.167/285.250 ms/token` (median `260.167`), None
    `257.998/260.166`, and Compact `300.594`; Spread is therefore an explicit experiment rather than the default.
    Automatic bandwidth placement remains open until the runtime enumerates actual LLC/NUMA domains and a longer
    repeated matrix demonstrates a stable win on symmetric and asymmetric hosts.
- [x] Add profile counters for helper layer shapes and selected grain/thread counts.
  - Implementation: `litenn_profile` now prints a CPU AOT parallel-selection table with fused layer count, selected
    parallel layer count, total FLOPs, per-layer `m/k/n`, selected helper threads, gate reason, and an emitted-object
    symbol check showing whether `litenn_cpu_matmul_bias_relu_parallel_f32` was actually used.
  - Boundary: this closes observability for the current sidecar helper; moving the work into optimized MLIR/LLVM
    lowering or a production GEMM backend remains open below.
- [ ] Move the parallel work into the optimized MLIR/LLVM lowering path or a production GEMM backend.
  - Requirement: the sidecar helper is acceptable as a first intra-op landing, but it does not preserve the MLIR
    packed/zmm microkernel and should not become the long-term CPU kernel architecture.

P2 retirement validation for the removed fast path:

| Benchmark | Default AOT RunInto | FastPath 1T | FastPath 16T |
| --- | ---: | ---: | ---: |
| `Linear(784->10)/batch:512` | `0.061 ms` | `1.53 ms` | `7.56 ms` |
| `MLP(784->128->10)/batch:512` | `0.393 ms` | `31.9 ms` | `16.1 ms` |
| `MLP(784->512->256->10)/batch:512` | `2.42 ms` | `356 ms` | `40.5 ms` |

Conclusion: default AOT already emits the better instruction stream. CPU multi-thread work should continue as a new
optimized-lowering task, not as the retired fast path.

P2 current validation:

| Benchmark | T1 / MLIR fallback | Default hardware-thread AOT | T16 |
| --- | ---: | ---: | ---: |
| `Linear(784->10)/batch:512` | `0.054 ms` | `0.053 ms` | `0.053 ms` |
| `MLP(784->128->10)/batch:512` | `0.337 ms` | `0.336 ms` | `0.357 ms` |
| `MLP(784->512->256->10)/batch:512` | `2.52 ms` | `1.76 ms` | `2.42 ms` |

Conclusion: intra-op parallelism is now implemented and guarded, but the present sidecar helper only helps the largest
local CPU case modestly. The next CPU performance step should parallelize the optimized lowering itself or call a
production GEMM backend.

Quantized validation note: `litenn_bench` now separates direct per-element weight decode,
`PreparedQuantizedLinear/<format>` execution, and dequantized reference execution. This makes repeated int4/fp4 weight
decode visible before investing in native packed microkernels.

## P3: CPU Kernel Refinement

Goal: continue smaller single-thread improvements after parallelism lands.

Status: implemented for static f32 constant-RHS MatMul shapes on 2026-05-16. Training-loss fusion is
deferred until P5 adds a training benchmark that can separate forward, backward, and loss costs.

- [x] Add K-block panel packing for large-K MatMul.
  - Implementation: `LowerNarrowMatMulPass` can materialize a K-panel packed RHS constant for wide static
    `MatMul` shapes and lower the row-tile kernel through the panelized layout.
  - Validation: `CompiledModuleTest.KPanelPackedWideMatMulMatchesReference`.
- [x] Replace fixed RHS packing thresholds with a static cost model.
  - Implementation: RHS packing now checks static `m/k/n`, output row reuse, constant f32 RHS storage, and
    estimated FLOPs versus packing bytes before selecting the packed path.
  - Scope: the new K-panel path is tried before the older packed wide-row path; the older path still keeps its
    conservative minimum-width guard to avoid destabilizing previously measured shapes.
- [x] Explore final-layer MatMulBiasAdd plus Softmax/CrossEntropy fusion for training workloads.
  - Result: deferred to P5. The current benchmark suite is inference-oriented, so adding this fusion now would
    create a correctness and API surface without a stable training performance signal.

P3 validation spot check:

| Validation | Result |
| --- | --- |
| `CompiledModuleTest.KPanelPackedWideMatMulMatchesReference` | Passed |
| Default `AOTRunInto/MLP(784->512->256->10)/batch:512` | `2.16 ms` |

## P4: CUDA Kernel Quality

Goal: improve native CUDA throughput once host overhead and graph scheduling are no longer dominant.

Status: target selection, optional cuBLASLt support, and optional CUDA Graph replay landed on 2026-05-16.
cuBLASLt remains opt-in because the local benchmark showed regressions on current small/medium inference shapes
when it was used by default. CUDA Graph replay is opt-in rather than default because it is pointer-binding
sensitive. Tensor Core codegen is evaluated and deferred until explicit fp16/bf16 tensor support exists.

- [x] Replace default `sm_30` target with `native` or a more modern baseline such as `sm_75`.
  - Implementation: the default NVPTX target is now `sm_75`; callers can pass `native` explicitly to query the
    current CUDA device and emit its `sm_<major><minor>` target.
  - Validation: `CompiledModuleTest.CUDANativeDefaultTargetUsesModernBaseline`.
- [x] Add cuBLASLt path and cache algorithm selection.
  - Implementation: when LiteNN is built with `CUDA::cublasLt`, the CUDA device runtime exposes an opt-in
    `CUDAExecutionOptions::enableCUBLASLt` MatMul path with per-device handle reuse and per-shape heuristic caching.
  - Default policy: disabled unless explicitly requested, because the current benchmark favors the existing
    cuBLAS path for the covered inference sizes.
- [x] Evaluate CUDA Graph replay as the default for static-shape inference payloads.
  - Result: implemented as opt-in, not default. The measured static-shape `RunInto` path benefits strongly,
    but graph executables capture raw input/output pointers, so default enablement should wait for an explicit
    pointer-stability contract or graph-node parameter update support.
- [x] Evaluate Tensor Core / WMMA generation only after cuBLASLt and scheduler measurements are stable.
  - Result: deferred. Float32 MNIST-style inference remains dominated by launch and library-call policy at the
    measured sizes; Tensor Core work should start from explicit fp16/bf16 tensor types and cuBLASLt policy
    measurements instead of hand-written WMMA in the current f32 path.

P4 validation spot check:

| Benchmark | Real time |
| --- | ---: |
| `CUDANativeMatMul/batch:1/width:128` | `0.036 ms` |
| `CUDANativeMatMul/batch:32/width:128` | `0.047 ms` |
| `CUDANativeMatMul/batch:128/width:128` | `0.038 ms` |
| `CUDANativeMatMul/batch:512/width:128` | `0.031 ms` |
| `CUDANativeGraphRunInto/Linear(784->10)/batch:512` | `0.031 ms` |
| `CUDANativeGraphRunInto/MLP(784->128->10)/batch:512` | `0.054 ms` |
| `CUDANativeGraphRunInto/MLP(784->512->256->10)/batch:512` | `0.069 ms` |

## P5: Training Benchmark Baseline

Goal: make training bottlenecks visible before optimizing them.

Status: implemented on 2026-05-16.

- [x] Add `bench_train.cpp` for MNIST Linear, MLP-128, and MLP-512.
  - Implementation: `litenn_bench_train` covers synthetic MNIST-shaped Linear, MLP-128, and MLP-512 batches.
- [x] Report forward, backward, optimizer step, and full step timings separately.
  - Implementation: `TrainCPUInterpreter/{Forward,Backward,OptimizerStep,FullStep}` benchmark families.
- [x] Track CPU AOT T1/T16, CUDA CPU fallback, and CUDA native variants independently.
  - Implementation: training forward baselines are registered as `TrainCPUAOT`, `TrainCPUAOTT1`,
    `TrainCPUAOTT16`, `TrainCUDACPUFallback`, and `TrainCUDANative`; CUDA CPU fallback uses
    `LITENN_CUDA_DISABLE_NATIVE_AOT=1` to keep the bridge measurable after native coverage expanded.
- [x] Add a CPU AOT trainer full-step row for the current compiled-training subset.
  - Implementation: `TrainCPUAOT/FullStep/MNIST-Linear` and `TrainCPUAOT/FullStep/MNIST-MLP128` run through
    `Trainer` with `TrainExecutionPolicy::AOT`; forward saved activations are passed into the compiled backward entry
    as explicit ABI params.
  - Metric split: the row reports `compile_ms` as a setup counter, while the benchmark timer measures train-step latency.

P5 validation spot check:

| Benchmark | Real time |
| --- | ---: |
| `TrainCPUAOT/FullStep/MNIST-Linear/batch:32` | `156 ms` Debug smoke; `compile_ms=232` |
| `TrainCPUAOT/FullStep/MNIST-MLP128/batch:32` | `290 ms` Debug smoke; `compile_ms=438` |
| `TrainCPUInterpreter/FullStep/MNIST-MLP128/batch:512` | `17.84 ms` |
| `TrainCPUInterpreter/FullStep/MNIST-MLP512/batch:512` | `86.18 ms` |
| `TrainCUDANative/Forward/MNIST-MLP128/batch:512` | `0.188 ms` |
| `TrainCUDANative/Forward/MNIST-MLP512/batch:512` | `0.905 ms` |

## Validation Checklist

- Build focused targets with `cmd /c cmake --build ...`.
- Run `CompiledModuleTest`, `CompiledModuleCUDATest`, and `CUDADeviceTest` after CUDA runtime changes.
- Run `litenn_bench --benchmark_filter=CUDANativeMatMul` before and after P0 changes.
- Keep raw benchmark output under `benchmark/results/` and summarize cross-backend numbers in Markdown. PyTorch
  comparison runs should use `python311 benchmark/bench.py`.
