# Qwen Sub-Layer Drift Analysis (2026-08-12)

## Scope

This report isolates the sub-layer evidence collected after the natural Qwen decode mismatch at generated index 23.
It records measured data and the conclusions that can be drawn from it. Implementation status and future work remain
in `docs/Roadmap.md` and `docs/PerformanceOptimizationRoadmap.md`.

No private model path or generated artifact location is recorded here.

## Question

LiteNN and llama.cpp follow the same forced token trajectory through generated index 23, but natural greedy decoding
selects a different token at that index. Their hidden states already differ at block 0 because the two runtimes use
independent quantized kernels and accumulation orders. The useful question is therefore not where the first nonzero
difference appears, but where the index-23 difference grows abnormally relative to nearby generated positions.

The experiment tests whether the late-layer anomaly is created by attention, by the FFN activation path, by the Q6_K
Down projection, or only by the final residual/logit margin.

## Controlled Setup

| Dimension | Value |
| --- | --- |
| Model class | Qwen2.5-Coder 14B Instruct, Q4_K_M |
| Candidate runtime | LiteNN stateful CPU AOT, LLVM O0, strict activation math, explicit T8 |
| Reference runtime | Vendored llama.cpp/ggml `b81c2cdd7`, CPU only, flash attention disabled |
| Decode schedule | Nine prompt tokens replayed one at a time, followed by a fixed 24-token trajectory |
| Trajectory identity | SHA-256 `2712ed2bc3907397db81e977be51a3d3e8fd38a17b92d53dfe97fd6092a0cd5f` |
| Target | Generated index 23 |
| Controls | Generated indices 16-22 |
| Blocks | 31, 32, 33, 43, 44, 45, 46, 47 |
| Boundaries | 13 per block, from attention norm through post-FFN residual |
| Matched coordinates | 832 (`8 indices * 8 blocks * 13 boundaries`) |

Both runtimes emitted the same v1 checkpoint manifests. Shapes were `[40,128]` for rotated queries and attention
context, `[8,128]` for rotated keys and values, `[1,5120]` for hidden/residual/Down boundaries, and `[1,13824]` for
Gate, Up, and SwiGLU. All compared values were finite Float32 values.

Generated index 0 is the forward pass for the final prompt token. Generated index `N > 0` is the pass after feeding
forced generated token `N-1`. This alignment avoids the one-token offset present in older logits-only diagnostics.

## Metrics

For each boundary and block, cross-runtime NRMSE is the RMS difference divided by the llama.cpp tensor RMS. The target
index is compared with the median and median absolute deviation of the seven control indices. Modified-z is
`0.67448975 * (target - median) / MAD`. Cosine distance is analyzed independently.

`Ratio` below is target NRMSE divided by control-median NRMSE. A joint maximum requires both target NRMSE and target
cosine distance to exceed every control. Modified-z is a ranking aid rather than a standalone proof because a very
small control MAD can make it large.

## Block 46 Boundary Data

Block 46 is the strongest post-FFN neighborhood outlier. The complete boundary sequence shows where its separation is
inherited, reduced, or amplified.

| Boundary | NRMSE ratio | NRMSE z | Cosine-distance z | Joint maximum | Reading |
| --- | ---: | ---: | ---: | --- | --- |
| Attention norm | 1.250 | 1.74 | 2.30 | yes | block input is already anomalous |
| Rotated Q | 0.943 | -4.99 | -2.25 | no | does not amplify the target |
| Rotated K | 0.922 | -2.91 | -2.68 | no | does not amplify the target |
| Value | 1.185 | 1.36 | 1.51 | no | remains inside the control envelope |
| Attention context | 0.941 | -2.83 | -2.97 | no | relative separation decreases |
| Attention output | 1.030 | 0.77 | -0.02 | no | output projection does not create an outlier |
| Attention residual | 1.288 | 5.25 | 7.43 | yes | inherited residual difference remains |
| FFN norm | 1.287 | 2.12 | 2.38 | yes | normalization does not materially change the ratio |
| FFN Gate | 1.200 | 4.13 | 9.11 | yes | projection output is target-dependent |
| FFN Up | 1.265 | 2.80 | 3.63 | yes | projection output is target-dependent |
| SwiGLU | 1.374 | 4.07 | 4.91 | yes | nonlinear combination increases separation |
| FFN Down | 1.496 | 2.70 | 3.76 | yes | largest block-46 ratio and next causal test point |
| Post-FFN | 1.325 | 13.27 | 12.91 | yes | strongest robust score because controls are tight |

Attention cannot be the operation that creates the block-46 target anomaly: Q, K, context, and attention output all
fail the positive joint-outlier gate. The attention residual is exceptional because its input is already exceptional,
not because the measured attention output becomes exceptional.

## Spatial Controls

The late-block panel checks whether the block-46 sequence is an isolated artifact. Ratios are target NRMSE divided by
the seven-control median.

| Block | Attention output | SwiGLU | FFN Down | Post-FFN | Strongest joint boundary and score |
| ---: | ---: | ---: | ---: | ---: | --- |
| 31 | 1.221 | 0.991 | 0.981 | 1.163 | attention norm, 3.25 |
| 32 | 1.149 | 1.150 | 1.138 | 1.142 | FFN Up, 2.77 |
| 33 | 0.999 | 1.079 | 1.074 | 1.124 | rotated Q, 3.05 |
| 43 | 0.997 | 1.432 | 1.396 | 1.304 | FFN Down, 5.55 |
| 44 | 1.104 | 1.780 | 1.698 | 1.301 | post-FFN, 3.64 |
| 45 | 1.016 | 1.542 | 1.496 | 1.320 | FFN Down, 3.66 |
| 46 | 1.030 | 1.374 | 1.496 | 1.325 | post-FFN, 12.91 |
| 47 | 0.914 | 1.322 | 1.463 | 1.223 | attention residual, 3.97 |

Blocks 43-47 provide the important spatial control. Their attention-output ratios are
`0.997, 1.104, 1.016, 1.030, 0.914`, while their Down ratios are
`1.396, 1.698, 1.496, 1.496, 1.463`. The target-specific late-window separation is therefore sustained through the FFN
activation and Down stages, but is absent from attention output.

The first joint outlier in every selected block is attention norm because each block receives an already different
residual stream. That fact identifies inheritance, not causality; choosing the first joint outlier would repeat the
same mistake as choosing the first bitwise mismatch.

## Runtime Observations

The numerical panel used explicit T8. The three-block T8 diagnostic completed AOT construction in about `58.3 s` and
execution in `38.4 s`; the eight-block adjacent panel completed construction in about `60.3 s` and execution in
`40.0 s`. These are diagnostic timings: selected internal outputs change liveness and callback synchronization, so
they are not accepted throughput measurements.

A default-thread diagnostic run completed its first three steps in about `0.43-0.49 s` each, then remained in step 4
for more than five minutes on one CPU thread. Its working set transiently reached about `9.4 GB` before falling below
`1 GB`. Explicit T8 completed the same 32-forward trajectory. The stalled run produced no selected checkpoint and is
excluded from the numerical tables, but it establishes an independent determinism and thread-policy defect.

## Conclusions

1. Bitwise onset is not a valid localization method. Independent quantized execution differs from block 0 while the
   generated trajectory remains equal through index 22.
2. Attention does not create the index-23 late-layer amplification. The strongest attention evidence is inherited in
   the residual stream, while block-46 attention output is only `1.030x` its control median.
3. The first actionable causal window is FFN Gate/Up -> SwiGLU -> Q6_K Down in blocks 43-47. The data does not yet say
   whether the owner is inherited activation drift, an implementation error, or an expected accumulation-order
   difference.
4. Q6_K math must not be changed from this cross-runtime comparison alone. The next experiment must feed one captured
   SwiGLU activation to exact-dequantized, LiteNN source-Q6_K, and llama.cpp Down paths.
5. The final user-visible mismatch is still unproven until final RMSNorm/logits are aligned and the expected versus
   selected top-token margin is measured.
6. The default-thread long loop is separate from numerical drift and must be reproduced and localized independently.

## Decision Gates

| Identical-activation result | Decision |
| --- | --- |
| LiteNN source-Q6_K departs from exact dequantization while llama.cpp remains near it | fix and regression-test the LiteNN Q6_K path |
| Both native paths remain within the same exact-reference error envelope | close Down-kernel correctness and test same-input Gate/Up plus SwiGLU semantics |
| Down explains only a small fraction of the post-FFN amplification | prioritize inherited residual/activation analysis and final-logit margin |
| A numerical change improves the selected token but worsens the control distribution | reject it as overfitting |

Any accepted correction must preserve the fixed trajectory, pass at least 128 tokens of natural decode, improve or
preserve corpus perplexity, avoid fallback, and retain cache-hit throughput.
