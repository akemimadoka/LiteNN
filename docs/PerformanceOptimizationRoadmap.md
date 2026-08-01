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

- [ ] Add a whole-process profile bundle command that combines existing `litenn_profile` evidence with waterfall
  timeline output and optional platform sampling.
  - [x] First slice: `benchmark/profile_bundle.py` wraps `litenn_profile` or an arbitrary command, captures
    stdout/stderr, writes `manifest.json`, `trace.json`, and `summary.md`, and redacts user-specified sensitive paths
    from recorded artifacts.
  - [x] GGUF decode parser slice: the bundle now converts `--stream-stats` and helper diagnostics into
    `gguf_decode_summary.json`, `gguf_decode_summary.md`, and `gguf_decode_trace.json` for token-step and helper
    attribution, including residual/non-helper share.
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
          - [ ] P0: Replace the current direct/staged GGML projection helpers with production packed Q4_K/Q6_K kernels.
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
                - [ ] Add a compact prepared-weight layout v3 for GGML_Q4_K/GGML_Q6_K decode projections.
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
                      than expanded-v1 in regression coverage. Existing callers retain expanded-v1 by default.
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
                      AOT load-and-run coverage passes in the complete 81-test GGUF importer suite. Real 14B cache-hit
                      acceptance remains open before v4 becomes the default.
                - [ ] Add a decode-step Q8_K activation workspace keyed by the normalized hidden vector.
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
                - [ ] Implement production Q4_K/Q6_K x Q8_K GEMV/vec-dot kernels for the top Qwen decode rows.
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
                      regression coverage are complete. Full 14B decode acceptance remains the final gate before v4
                      may replace v3.
                - [ ] Re-run the cache-hit policy matrix after compact Q8_K kernels and only then retune thread/grain
                      defaults. The current data says thread retuning without llama.cpp-class low-thread kernels is a
                      secondary lever.
                - [ ] Move Q8_K activation staging from per-helper temporary work into a decode-step activation-staging
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
                - [ ] Implement low-thread packed GEMV microkernels for Q4_K/Q6_K x Q8_K before retuning thread policy.
                      The llama.cpp control source uses `q8_K` activation dot kernels, `gemv_q4_K/q6_K_*_q8_K`, and
                      repacked/VNNI/AMX-oriented paths; LiteNN's current hot path still performs direct Float32 x GGML
                      block accumulation in `litenn_cpu_ggml_block_matmul_f32` with `x4` output grouping.
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
          - [ ] P0: Run full-decode thread/grain A/B instead of extrapolating from isolated helpers.
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
          - [ ] P0: Split the current `~143 ms/step` residual into ranked runtime buckets.
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
                - [ ] Add stable per-layer/per-node timing for non-helper generated code and expose RMSNorm, SwiGLU,
                      residual adds, logits/sampler handling, state aliasing, and runtime entry overhead separately.
                      This is the next blocker once quantized projection time is reduced.
          - [ ] P1: Skip full-vocabulary logits projection for prompt replay steps that cannot be sampled.
                The July 5 run spends about `53 ms` per logits projection and executes one on every prompt replay step.
                Skipping all but the last replay logits improves prompt/prefill latency, though it is not a steady-state
                generated-token TPS fix.
          - [ ] P1: Add a sampler-only logits path for text generation when public logits are not requested.
                The July 6 profile shows the final vocabulary projection costs about `7%` of total decode time even
                after grouped projection work. Golden-logit and API runs still need full public logits, but text-only
                decode can use a projection+sampler path that keeps only the selected token/top-k candidates.
          - [ ] P1: Batch or fuse per-head RoPE helper calls after projection work moves.
                RoPE is not the current top bottleneck, but the default profile still shows `55296` helper calls over
                24 steps and about `13-14 ms` per decode step. Once projection kernels are reduced, convert per-head
                RoPE calls into a batched per-layer helper or fuse it into the Q/K layout path.
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
  - [ ] Platform-native sampling import: optional Windows ETW/xperf, Linux `perf`, and macOS Instruments import
    adapters normalize their own raw formats into the collapsed-stack path.
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
  - Implementation: `CompilerOptions::cpuAOTAffinityPolicy` defaults to `None`; `Compact` pins the CPU AOT helper's
    persistent worker threads to low-numbered CPUs while the policy remains active and restores their prior affinity
    when the policy is disabled or workers exit. Benchmark/profile entry points can set it through
    `LITENN_CPU_AOT_AFFINITY=compact`.
  - Evaluation: local Windows `MLP(784->512->256->10)/batch:128` sidecar-helper runs with
    `LITENN_CPU_AOT_THREADS=16` and `LITENN_CPU_AOT_PARALLEL_MIN_FLOPS=1` did not improve with compact affinity; the
    measured real time regressed versus no affinity. Keep affinity opt-in until topology-aware policies are available.
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
