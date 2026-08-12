# Qwen Natural Decode Layer Drift Evidence (2026-08-12)

## Decision

The natural greedy mismatch at generated index 23 is not a first-nonzero-error event. LiteNN and llama.cpp already
produce different Float32 hidden values at block 0 while following the same token trajectory, as expected from
independent quantized kernels and accumulation orders. A zero-tolerance "first failing layer" therefore selects block 0
at every measured position and is not a useful localization rule.

The useful signal is relative amplification against a neighborhood, not one adjacent token. With generated indices
16-22 as controls, index 23 exceeds the control maximum for both NRMSE and cosine distance throughout blocks 38-46.
Block 46 is the strongest robust outlier: NRMSE is `0.185127` versus a `0.140524` control median and `0.155339`
maximum, producing modified-z `33.42`; cosine-distance modified-z is `15.44`. Blocks 44 and 32 are secondary windows.
Block 47 is not an NRMSE outlier against this neighborhood, invalidating the earlier broad 39-47 interpretation.

The completed sub-layer campaign further rejects attention as the point that creates the index-23 anomaly. In block
46, rotated Q/K, attention context, and attention output are not positive joint outliers; attention-output NRMSE is
only `1.030x` its control median. The inherited residual is already exceptional, and the FFN path then separates
again: SwiGLU is `1.374x` and Q6_K Down output is `1.496x` the control median. The post-FFN residual has the strongest
joint modified-z (`13.27/12.91`) because its controls are unusually tight. Blocks 43-47 show the same sustained
Down-stage pattern. The subsequent identical-activation gate closes Q6_K Down: LiteNN field-v4/Q8_K matches captured
llama.cpp Down within `3.67e-7` NRMSE across blocks 43-47. Same-input Q4_K Gate/Up and SwiGLU are now the next gate.

No model path or private artifact location is part of this evidence record.

## Controlled Setup

| Dimension | LiteNN | Reference |
| --- | --- | --- |
| Model class | Qwen2.5-Coder 14B Instruct, Q4_K_M | Same GGUF payload |
| Device | CPU | CPU, `n_gpu_layers=0` |
| Decode schedule | Stateful CPU AOT, one token per invocation | One token per `llama_decode` invocation |
| Prompt replay | Nine single-token invocations | Nine single-token invocations |
| Attention | Active-prefix stateful attention | Flash attention disabled |
| Math/runtime | LLVM O0, strict activation math, T8 | Vendored llama.cpp/ggml commit `b81c2cdd7` |
| LiteNN weights | `all`, field-interleaved-v4 | llama.cpp CPU repack |
| Cache capacity | 137 tokens | llama.cpp rounds the requested context to 256 |
| Trajectory | Fixed 24 generated tokens, SHA-256 `2712ed2bc3907397db81e977be51a3d3e8fd38a17b92d53dfe97fd6092a0cd5f` |
| Selected positions | Generated indices 0, 22, and 23 | Same |
| Hidden boundary | Post-FFN residual `layer_hidden_N` | Qwen2 `l_out-N`, after `build_cvec` |

The generated-index timeline is aligned explicitly: index 0 is the forward pass for the final prompt token, and index
`N > 0` is the forward pass after feeding forced generated token `N-1`. This avoids the one-step offset in the older
reference logits helper.

## Data Integrity

- Both runtimes emitted 48 ordered Float32 tensors of shape `[1,5120]` at each selected index.
- Each selected-position bundle is `983,040` bytes with contiguous, non-overlapping manifest ranges.
- No selected tensor contains a non-finite value.
- The shared comparator accepted coordinates, names, dtypes, shapes, and payload ranges for all 96 index-22/23 rows.
- A zero-tolerance LiteNN self-compare passes 48/48 rows; the cross-runtime comparison fails all rows, beginning at
  block 0, which is why nonzero-error onset is rejected as a localization criterion.
- Replacing a batched llama.cpp prompt with single-token prompt replay leaves the block-0 index-0 metrics unchanged:
  mean absolute error `0.0519471`, RMS error `0.0675342`, and cosine similarity `0.983681`. Prompt batching is not the
  source of the observed baseline difference.

## Layer Data

`NRMSE` is cross-runtime RMS error divided by the llama.cpp hidden-state RMS for the same block and generated index.
`Delta NRMSE` and `Delta cosine` are index 23 minus index 22. The two indices consume different forced input tokens,
so these deltas rank diagnostic windows but are not by themselves causal proof.

| Block | RMS error @22 | RMS error @23 | RMS change | NRMSE @22 | NRMSE @23 | Delta NRMSE | Cosine @22 | Cosine @23 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 0.169267 | 0.088223 | -47.88% | 0.30653 | 0.14920 | -15.73 pp | 0.97835 | 0.99222 |
| 5 | 0.334896 | 0.320540 | -4.29% | 0.37786 | 0.35427 | -2.36 pp | 0.94176 | 0.93805 |
| 9 | 0.458161 | 0.484501 | +5.75% | 0.35250 | 0.36599 | +1.35 pp | 0.94897 | 0.93888 |
| 15 | 0.691976 | 0.728359 | +5.26% | 0.33631 | 0.33321 | -0.31 pp | 0.95745 | 0.95666 |
| 20 | 0.796478 | 0.879554 | +10.43% | 0.35775 | 0.36238 | +0.46 pp | 0.94673 | 0.94731 |
| 25 | 0.817916 | 0.918783 | +12.33% | 0.34073 | 0.36303 | +2.23 pp | 0.94773 | 0.93817 |
| 30 | 1.042003 | 1.129802 | +8.43% | 0.32064 | 0.34997 | +2.93 pp | 0.95195 | 0.93857 |
| 35 | 1.572173 | 1.668117 | +6.10% | 0.30223 | 0.30071 | -0.15 pp | 0.95594 | 0.95486 |
| 38 | 1.949583 | 2.127754 | +9.14% | 0.27549 | 0.28696 | +1.15 pp | 0.96302 | 0.95856 |
| 39 | 2.078691 | 2.401465 | +15.53% | 0.26556 | 0.29309 | +2.75 pp | 0.96537 | 0.95668 |
| 42 | 2.278519 | 2.720774 | +19.41% | 0.22091 | 0.25968 | +3.88 pp | 0.97628 | 0.96601 |
| 43 | 2.386506 | 2.880789 | +20.71% | 0.21098 | 0.25296 | +4.20 pp | 0.97848 | 0.96772 |
| 44 | 2.529663 | 3.006763 | +18.86% | 0.19440 | 0.23683 | +4.24 pp | 0.98190 | 0.97170 |
| 47 | 3.039928 | 3.670059 | +20.73% | 0.14379 | 0.19474 | +5.10 pp | 0.98962 | 0.98178 |

The raw RMS delta first turns positive at block 6, but it is small and not monotonic. NRMSE first crosses above the
index-22 control at block 9, falls below it again in blocks 11-18, remains mostly positive through blocks 19-34, and
reaches the strongest sustained region in blocks 39-47. The evidence supports a distributed accumulation model more
strongly than a single abrupt bad block.

## Index 16-23 Neighborhood

The second campaign captured all 48 post-FFN residuals at generated indices 16-23 in both runtimes, for 384 matched
rows. For each block, index 23 is compared with the median and median absolute deviation of indices 16-22. Modified-z
uses `0.67448975 * (target - median) / MAD`; `above max` is retained because a very small MAD can magnify the score.

| Block | Target NRMSE | Control median | Control max | Delta vs median | Modified-z | Cosine-distance z | Above both maxima |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 32 | 0.337416 | 0.292819 | 0.312752 | +0.044597 | 2.01 | 3.83 | yes |
| 38 | 0.286960 | 0.245513 | 0.275490 | +0.041446 | 1.22 | 1.23 | yes |
| 39 | 0.293088 | 0.236570 | 0.265561 | +0.056519 | 2.92 | 2.65 | yes |
| 40 | 0.278207 | 0.221787 | 0.246782 | +0.056420 | 3.06 | 2.81 | yes |
| 41 | 0.268713 | 0.217028 | 0.235822 | +0.051685 | 2.82 | 2.71 | yes |
| 42 | 0.259678 | 0.203705 | 0.220914 | +0.055973 | 2.97 | 3.02 | yes |
| 43 | 0.252962 | 0.195341 | 0.210975 | +0.057621 | 3.26 | 3.34 | yes |
| 44 | 0.236826 | 0.183635 | 0.194398 | +0.053191 | 3.33 | 4.33 | yes |
| 45 | 0.213361 | 0.163759 | 0.178446 | +0.049602 | 2.76 | 3.10 | yes |
| 46 | 0.185127 | 0.140524 | 0.155339 | +0.044603 | 33.42 | 15.44 | yes |
| 47 | 0.194739 | 0.162983 | 0.208631 | +0.031756 | 0.83 | 1.43 | no (NRMSE) |

The block-46 score is not merely a large absolute residual: its target NRMSE is `31.74%` above the control median and
`19.18%` above the largest control. Its unusually small control MAD (`0.00090015`) explains the very large z-score, but
the independent cosine-distance signal and both above-maximum checks preserve the ranking. Block 47 then partially
returns to the neighborhood envelope, so final post-FFN residual magnitude alone cannot identify the responsible
operation.

## Runtime Observations

The LiteNN diagnostic run reproduced exactly one forced-token argmax mismatch, at generated index 23. It reported:

- AOT build: `30,998.7 ms` with cache publication disabled.
- Decode run: `6,422.85 ms` for 32 total single-token forwards.
- Generated-token module average: `204.787 ms`, or `4.883 token/s` in this diagnostic run.
- Checkpoint writes: `3.297 ms` at index 22 and `3.010 ms` at index 23.
- Runtime outputs: 49 per invocation, consisting of public logits plus 48 diagnostic hidden states.

These timings validate that checkpoint output is a small selected-step cost. They are not accepted throughput numbers:
the diagnostic ABI changes output liveness and the llama.cpp callback synchronizes selected graph nodes.

## Sub-Layer Index 16-23 Neighborhood

The third campaign exposed 13 internal boundaries only for blocks `31,32,33,43,44,45,46,47`. Both runtimes emitted
the same v1 manifests and Float32 shapes: `[40,128]` for rotated queries/attention context, `[8,128]` for rotated keys
and values, `[1,5120]` for hidden/residual/Down boundaries, and `[1,13824]` for Gate/Up/SwiGLU. This produced 832
matched coordinates (`13 boundaries * 8 blocks * 8 generated indices`) without changing the normal decode ABI.

The table reports the most useful block-46 sequence. `Ratio` is target-index NRMSE divided by the indices-16-22
control median. `Joint max` requires both NRMSE and cosine distance to exceed all seven controls.

| Boundary | NRMSE ratio | NRMSE z | Cosine-distance z | Joint max | Interpretation |
| --- | ---: | ---: | ---: | --- | --- |
| Attention norm | 1.250 | 1.74 | 2.30 | yes | anomaly already present at block input |
| Rotated Q | 0.943 | -4.99 | -2.25 | no | large baseline kernel delta, not index-23 amplification |
| Rotated K | 0.922 | -2.91 | -2.68 | no | large baseline kernel delta, not index-23 amplification |
| Value | 1.185 | 1.36 | 1.51 | no | within neighborhood maximum |
| Attention context | 0.941 | -2.83 | -2.97 | no | attention reduces relative separation |
| Attention output | 1.030 | 0.77 | -0.02 | no | Wo does not create a target outlier |
| Attention residual | 1.288 | 5.25 | 7.43 | yes | inherited block-input error remains after residual add |
| FFN norm | 1.287 | 2.12 | 2.38 | yes | no material ratio change across normalization |
| FFN Gate | 1.200 | 4.13 | 9.11 | yes | activation-dependent projection separation |
| FFN Up | 1.265 | 2.80 | 3.63 | yes | activation-dependent projection separation |
| SwiGLU | 1.374 | 4.07 | 4.91 | yes | nonlinear combination amplifies separation |
| FFN Down | 1.496 | 2.70 | 3.76 | yes | largest block-46 ratio; Q6_K projection is the causal test target |
| Post-FFN | 1.325 | 13.27 | 12.91 | yes | strongest robust score due to tight control MAD |

Spatial controls support the same reading. Attention-output NRMSE ratios for blocks `43,44,45,46,47` are
`0.997,1.104,1.016,1.030,0.914`, while Down ratios are `1.396,1.698,1.496,1.496,1.463`. The late-window separation
therefore persists across FFN Down but not across attention output. This evidence does not yet distinguish three
possibilities: inherited SwiGLU input error, a source-Q6_K execution error, or normal but quality-relevant independent
quantized accumulation. The next experiment must feed the same captured SwiGLU activation into an exact-dequantized
Down reference and the native LiteNN Q6_K path.

The selected-block LiteNN run used T8 because one default-thread diagnostic build completed three steps normally and
then remained in step 4 for more than five minutes, executing on one core; its working set transiently reached about
`9.4 GB` before returning below `1 GB`. The T8 reruns completed all 32 forwards. This anomalous run produced no
checkpoint data and is excluded from numerical results, but it establishes a separate deterministic-runtime gate for
the default thread policy.

## Conclusions And Next Gates

1. Do not change model math based on zero-tolerance hidden-state mismatch. Independent Q4_K/Q6_K kernels differ from
   block 0, while the generated sequence remains equal through index 22.
2. The index 16-23 block and sub-layer neighborhood gates are complete. Selectable outputs cover all 13 boundaries in
   blocks 31-33 and 43-47 without expanding every block's diagnostic ABI. Attention does not create the late-window
   target separation; FFN SwiGLU/Down is now the first causal test window.
3. Identical-activation Q6_K Down verification is complete for blocks 43-47. LiteNN production field-v4/Q8_K matches
   captured llama.cpp output within `3.67e-7` NRMSE, while both share the expected Q8_K distance from exact
   dequantization. Do not rewrite Down; apply the same-input gate to Q4_K Gate/Up and strict SwiGLU next. Full data:
   `docs/QwenSubLayerDriftAnalysis_2026-08-12.md`.
4. Reproduce and localize the default-thread diagnostic step-4 long loop. Diagnostic output selection must not alter
   deterministic completion or state progression.
5. Capture final RMSNorm and aligned logits, including top candidates, the expected-vs-selected token margin, and the
   contribution of output projection error. An argmax mismatch can be benign when the reference margin is tiny.
6. After localization, validate a candidate with the same forced trajectory, natural greedy parity beyond 128 tokens,
   corpus perplexity delta, and unchanged cache-hit performance before promoting it.
