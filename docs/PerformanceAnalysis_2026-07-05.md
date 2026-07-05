# LiteNN Qwen CPU Decode Performance Analysis - 2026-07-05

This report follows `PerformanceAnalysis.md`, `PerformanceAnalysis_2026-05-16.md`,
`PerformanceAnalysis_2026-05-19.md`, and `PerformanceAnalysis_2026-07-02.md`. It uses a real
Qwen2.5-Coder-14B Q4_K_M stateful CPU AOT smoke run rather than extrapolating from small helper rows.

The user's current reference point is about `6.85 tokens/s` in llama.cpp with GPU offload and major accelerators
disabled. That is about `146 ms/token`. The LiteNN run measured about `819 ms/token`, or `1.22 tokens/s`.
LiteNN is therefore about `5.6x` slower on this workload.

## Evidence

The input was a private local GGUF model path and the prompt `hello`. The path is intentionally not recorded here.
The relevant LiteNN evidence came from the completed `qwen_smoke` work directory and a profile bundle regenerated from
its report:

- `build/qwen_test/litenn_decode_token_ids.stdout.txt`
- `build/qwen_test/litenn_decode_token_ids.stderr.txt`
- `build/qwen_test/qwen_smoke_report.json`
- `build/qwen_test_profile_bundle/gguf_decode_summary.json`
- `build/qwen_test_profile_bundle/gguf_decode_summary.md`

The bundle was regenerated with:

```powershell
python311 benchmark\profile_bundle.py --skip-litenn-profile `
  --out-dir build\qwen_test_profile_bundle `
  --qwen-smoke-report build\qwen_test\qwen_smoke_report.json
```

The profile importer needed three fixes while preparing this report:

- Existing-run reports can contain relative trace/log paths that are already workspace-relative, so path resolution must
  first try the path as-is before prefixing the report directory.
- Runs without `--stream-stats` still emit `decode step N ok X ms`; the GGUF parser now uses those lines as a fallback
  for step timing.
- `litenn_cpu_ggml_block_grouped_matmul2_f32` is now classified as a projection helper instead of `unknown`.

## End-to-End Timing

| Metric | Value |
| --- | ---: |
| Build time | `20754.9 ms` |
| Runtime decode time | `14737.4 ms` |
| Executed steps | `18` |
| Prompt replay steps | `8` |
| Generation steps | `10` |
| Average step time | `818.602 ms` |
| Generated-token latency | `819.139 ms/token` |
| Generated-token throughput | `1.22079 tokens/s` |
| llama.cpp reference | `6.85 tokens/s` |
| LiteNN vs llama.cpp | `0.178x` throughput, about `5.6x` slower |
| Fallback count | `0` |

Cold compile is still expensive (`20.6 s` for the stateful AOT artifact), but it is not the step-latency bottleneck.
The measured runtime gap is in the generated token path.

## Compile Phase

| Phase | Time |
| --- | ---: |
| Import GGUF archive | `4328.150 ms` |
| Build stateful decode runtime schedule | `117.903 ms` |
| Compile stateful AOT artifact | `20623.834 ms` |
| Externalize graph | `8037.419 ms` |
| Bufferize | `2003.910 ms` |
| Lower LLVM dialect | `2283.526 ms` |
| Emit object file | `7293.260 ms` |
| Object size | `1071816 bytes` |
| External weight region | `8982142976 bytes`, `579` tensors |

This remains an important usability issue, but it is separate from the `~819 ms/token` steady-state runtime gap.

## Runtime Attribution

The repaired profile bundle parsed `18` steps, `198` helper events, and the following totals:

| Bucket | Total | Average per step | Share |
| --- | ---: | ---: | ---: |
| Step wall time | `14734.834 ms` | `818.602 ms` | `100.00%` |
| Timed helper bodies | `12152.507 ms` | `675.139 ms` | `82.47%` |
| Residual/non-helper | `2582.327 ms` | `143.463 ms` | `17.53%` |

The full gap to the llama.cpp reference is about `819 - 146 = 673 ms/token`. Timed projection helpers alone average
about `662 ms/step`, and all timed helpers average `675 ms/step`. That makes quantized projection the dominant,
evidence-backed optimization target.

### Operator Ranking

| Operator role | Calls | Total ms | Average per step | Share |
| --- | ---: | ---: | ---: | ---: |
| `projection/ffn_gate_up_grouped` | `864` | `4545.657` | `252.537 ms` | `30.85%` |
| `projection/ffn_down` | `864` | `3328.974` | `184.943 ms` | `22.59%` |
| `projection/hidden_or_output` | `1728` | `2195.193` | `121.955 ms` | `14.90%` |
| `projection/logits` | `18` | `959.671` | `53.315 ms` | `6.51%` |
| `projection/kv` | `1728` | `832.523` | `46.251 ms` | `5.65%` |
| `position_encoding/rope` | `41472` | `266.234` | `14.791 ms` | `1.81%` |
| `attention/active_prefix` | `864` | `22.943` | `1.275 ms` | `0.16%` |
| `kv_update/append` | `1728` | `1.190` | `0.066 ms` | `0.01%` |
| `embedding/token_lookup` | `18` | `0.122` | `0.007 ms` | `0.00%` |

Projection helpers account for about `80.5%` of the full step. Active-prefix attention and KV append are not the
current 2K-context bottleneck.

### Top Helpers

| Helper shape | Calls | Total ms | Average per step | Share |
| --- | ---: | ---: | ---: | ---: |
| Q4_K grouped gate/up, `1x5120 -> 1x27648` | `864` | `4545.657` | `252.537 ms` | `30.85%` |
| Q4_K hidden/output, `1x5120 -> 1x5120` | `1728` | `2195.193` | `121.955 ms` | `14.90%` |
| Q6_K FFN down, `1x13824 -> 1x5120` | `432` | `2142.374` | `119.021 ms` | `14.54%` |
| Q4_K FFN down, `1x13824 -> 1x5120` | `432` | `1186.600` | `65.922 ms` | `8.05%` |
| Q6_K logits, `1x5120 -> 1x152064` | `18` | `959.671` | `53.315 ms` | `6.51%` |
| Q4_K KV, `1x5120 -> 1x1024` | `1296` | `565.858` | `31.437 ms` | `3.84%` |
| Q6_K KV, `1x5120 -> 1x1024` | `432` | `266.665` | `14.815 ms` | `1.81%` |
| RoPE helper, `1x128 -> 1x128` | `41472` | `266.234` | `14.791 ms` | `1.81%` |

The helper details report `requested_threads=0 resolved_threads=16` for the major GGML matmul helpers.

## Helper Benchmark Cross-Check

Short same-build helper benchmarks were run to compare direct and Q8_K-staged paths on Qwen-shaped rows. These are
microbenchmarks, not full-model replacements, but they help rule out easy switches.

### Direct Q4_K/Q6_K T16

| Row | Real time |
| --- | ---: |
| Q4_K `5120 -> 5120` | `0.797 ms` |
| Q4_K `5120 -> 1024` | `0.289 ms` |
| Q4_K `5120 -> 13824` | `1.73 ms` |
| Q4_K `13824 -> 5120` | `1.94 ms` |
| Q4_K `5120 -> 152064` | `16.9 ms` |
| Q6_K `5120 -> 5120` | `1.29 ms` |
| Q6_K `5120 -> 1024` | `0.402 ms` |
| Q6_K `5120 -> 13824` | `3.23 ms` |
| Q6_K `13824 -> 5120` | `3.06 ms` |
| Q6_K `5120 -> 152064` | `36.3 ms` |

### Q8_K-Staged Q4_K/Q6_K T16

| Row | Real time |
| --- | ---: |
| Q4_K `5120 -> 5120` | `1.23 ms` |
| Q4_K `5120 -> 1024` | `0.596 ms` |
| Q4_K `5120 -> 13824` | `2.64 ms` |
| Q4_K `13824 -> 5120` | `3.01 ms` |
| Q4_K `5120 -> 152064` | `25.2 ms` |
| Q6_K `5120 -> 5120` | `1.64 ms` |
| Q6_K `5120 -> 1024` | `0.671 ms` |
| Q6_K `5120 -> 13824` | `3.64 ms` |
| Q6_K `13824 -> 5120` | `4.29 ms` |
| Q6_K `5120 -> 152064` | `37.0 ms` |

The current staged prototype is not a default-switch candidate. It is slower on these production-shaped T16 rows.
The next kernel track must therefore be a real packed/repacked vector-dot implementation, not merely flipping the
existing staged flag.

### Grouped Gate/Up

| Row | Real time |
| --- | ---: |
| Q4_K gate/up separate T16 | `3.84 ms` |
| Q4_K gate/up concatenated T16 | `3.60 ms` |
| Q4_K gate/up separate T32 | `3.11 ms` |
| Q4_K gate/up concatenated T32 | `2.96 ms` |
| Q6_K gate/up separate T16 | `8.13 ms` |
| Q6_K gate/up concatenated T16 | `6.84 ms` |
| Q6_K gate/up separate T32 | `5.88 ms` |
| Q6_K gate/up concatenated T32 | `5.68 ms` |

The real model averages about `5.26 ms` for each Q4_K grouped gate/up helper call (`252.5 ms / 48 layers`). The isolated
T16 benchmark is lower (`3.60 ms`), and T32 is lower again (`2.96 ms`). This suggests a low-risk thread-policy A/B may
recover some time, but the maximum plausible win is far smaller than the `~673 ms/token` gap. The high-ROI fix remains
the projection kernel organization itself.

## Interpretation

The July 2 analysis correctly identified quantized projection as the likely root. The July 5 helper-attributed run
turns that into a measured conclusion:

- The runtime gap to `6.85 tokens/s` is almost exactly the amount of time spent inside quantized projection helpers.
- The current Q8_K-staged prototype does not win on production-shaped rows, so it is not enough to copy the staging
  concept without also implementing the packed/repacked vector-dot kernel family and a format-specific policy.
- Active-prefix attention, paged append, and token lookup are already too small in this 2K run to explain the gap.
- The residual `143 ms/step` is still large: if projection were magically reduced to llama.cpp-like levels, residual
  overhead would become the next blocker. It needs deeper per-layer timing around RMSNorm, SwiGLU, residual adds,
  sampler/logit handling, state aliasing, and runtime entry overhead.

## Highest-ROI Optimization Targets

1. **Production packed quantized projection kernels.** Target Q4_K and Q6_K first because they dominate this model.
   The implementation should use model-load-time packed/repacked weight layout, activation-side staging/reuse where it
   wins, AVX-512/VNNI or AVX2 vector-dot kernels, and output-column tiling that avoids reparsing block metadata for
   adjacent outputs. Success metric: reduce projection helper average from about `662 ms/step` to below `120 ms/step`
   on this run shape.
2. **Full-step thread and grain policy.** The current helpers resolve to 16 threads by default. Isolated grouped
   gate/up improves at T32, but this must be validated in full decode to avoid oversubscription or cache regressions.
   Success metric: full stateful decode A/B table for T8/T16/T32/default with helper share and residual share.
3. **Residual attribution and fusion.** Add stable per-layer/per-node timing for non-helper generated code and expose
   RMSNorm/SwiGLU/residual/logits-sampler buckets. Success metric: split the current `143 ms/step` residual into ranked
   operator groups before optimizing it.
4. **Prompt replay logits skip.** The profile shows one full-vocabulary logits projection on every step. During prompt
   replay, logits are only needed for the final replay token before sampling. Skipping earlier replay logits would save
   about `53 ms` for each skipped prompt token. This improves prefill/replay latency, not steady generated-token TPS.
5. **Compile-time usability.** The 20s stateful compile and 8GB externalization path remain worth optimizing, but they
   should not distract from the steady-state projection gap.

## Roadmap Update

`PerformanceOptimizationRoadmap.md` now treats the July 5 run as the current Qwen decode baseline. The next performance
work should prioritize production packed Q4_K/Q6_K projection kernels and full-step thread policy A/B before spending
time on attention, KV append, or generic compile-time polish.
