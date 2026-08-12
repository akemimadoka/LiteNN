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

## Identical-Activation Q6_K Down Result

The causal gate was executed after the initial sub-layer report. For each block 43-47, the llama.cpp-captured SwiGLU
tensor at generated index 23 was fed unchanged to four projection paths using the same GGUF Down weight:

1. Per-weight-row Q6_K dequantization with Float64 dot accumulation, used as the exact represented-weight reference.
2. Vendored ggml Q6_K x Q8_K `vec_dot`.
3. LiteNN's source-layout Q6_K x Float32 helper.
4. LiteNN's production field-interleaved-v4 Q6_K x Q8_K helper.

The captured llama.cpp Down output is the fifth comparison tensor. The exact implementation dequantizes only one
weight row at a time, avoiding a roughly 283 MiB Float32 materialization for each `5120 x 13824` Down weight.

| Block | Production vs captured NRMSE | Production vs captured max abs | Captured vs exact NRMSE | Source-F32 vs exact NRMSE |
| ---: | ---: | ---: | ---: | ---: |
| 43 | 3.667e-7 | 5.722e-6 | 2.4225% | 5.259e-7 |
| 44 | 3.386e-7 | 1.144e-5 | 1.6732% | 4.531e-7 |
| 45 | 3.325e-7 | 7.629e-6 | 1.9145% | 4.885e-7 |
| 46 | 2.731e-7 | 5.722e-6 | 1.2129% | 6.674e-7 |
| 47 | 3.184e-7 | 1.144e-5 | 1.4082% | 5.374e-7 |

The ggml vec-dot path also matches the captured output within `2.80e-7` to `3.71e-7` NRMSE. The `1.21%-2.42%`
distance from exact dequantization is therefore the shared Q8_K activation-quantization policy, not a LiteNN-only
error. LiteNN's source-Float32 helper is closest to the exact represented-weight result, as expected, but it is not the
production decode contract.

This experiment closes Q6_K Down correctness for the observed mismatch. The earlier `1.396x-1.698x` Down-stage
target/control ratios result from feeding different SwiGLU activations to equivalent production projection semantics.
The same-input experiment therefore moved upstream to Q4_K Gate/Up and strict SwiGLU.

## Identical-Input Gate, Up, And SwiGLU Result

For each block 43-47, the llama.cpp-captured `ffn_norm` tensor at generated index 23 was fed unchanged to exact
represented-weight Q4_K projection, ggml vec-dot, LiteNN source-Float32 projection, and LiteNN's production grouped
field-v4/Q8_K Gate/Up helper. Every candidate pair was then passed through the real `litenn_cpu_swiglu_f32` strict
activation helper. A fifth candidate applies only LiteNN strict SwiGLU to the captured llama.cpp Gate/Up values, which
isolates activation math from projection math.

| Block | Production Gate vs captured NRMSE | Production Up vs captured NRMSE | Production pipeline SwiGLU NRMSE | Captured Gate/Up + LiteNN SwiGLU NRMSE | Pipeline max abs |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 43 | 1.296e-7 | 1.944e-7 | 2.580e-7 | 3.756e-8 | 1.144e-5 |
| 44 | 1.442e-7 | 2.207e-7 | 3.620e-7 | 6.044e-8 | 7.629e-6 |
| 45 | 1.408e-7 | 2.171e-7 | 3.161e-7 | 4.394e-8 | 7.629e-6 |
| 46 | 1.206e-7 | 1.587e-7 | 3.134e-7 | 5.273e-8 | 9.537e-6 |
| 47 | 1.055e-7 | 1.657e-7 | 2.911e-7 | 4.942e-8 | 5.722e-6 |

The maximum production Gate, Up, and complete SwiGLU NRMSE values are `1.44e-7`, `2.21e-7`, and `3.62e-7`.
Recomputing only strict SwiGLU from captured Gate/Up stays below `6.05e-8`. These are floating-point ordering and
elementary-function noise, not a model-semantic discrepancy. Together with the Down result, this closes the complete
FFN implementation in blocks 43-47 as the owner of the index-23 token mismatch.

The cross-runtime SwiGLU and Down outlier ratios are caused by different `ffn_norm` inputs entering equivalent FFN
semantics. Since attention output also failed the target-outlier gate, the remaining useful question is no longer
which late-layer kernel is wrong. It is whether accumulated residual-stream differences cross a small final-logit
margin. Final RMSNorm, output projection, top-k candidates, and the expected-versus-selected margin are now the next
causal gate.

## Conclusions

1. Bitwise onset is not a valid localization method. Independent quantized execution differs from block 0 while the
   generated trajectory remains equal through index 22.
2. Attention does not create the index-23 late-layer amplification. The strongest attention evidence is inherited in
   the residual stream, while block-46 attention output is only `1.030x` its control median.
3. Identical-input verification closes Q4_K Gate/Up, strict SwiGLU, and Q6_K Down in blocks 43-47 as
   implementation-error candidates.
4. Q6_K math must not be changed from the cross-runtime sub-layer ratios. On the same captured SwiGLU activation,
   LiteNN's production field-v4/Q8_K path matches the llama.cpp Down result within `3.67e-7` NRMSE across blocks 43-47.
5. The final user-visible mismatch is now best explained as accumulated residual-stream drift, but this remains
   unproven until final RMSNorm/logits are aligned and the expected versus selected top-token margin is measured.
6. The default-thread long loop is separate from numerical drift and must be reproduced and localized independently.

## Decision Gates

| Identical-activation result | Decision |
| --- | --- |
| Observed: LiteNN production Q6_K matches captured llama.cpp Down | close Down-kernel correctness; do not rewrite it |
| Observed: Gate/Up and strict SwiGLU match on the same `ffn_norm` input | close complete FFN kernel correctness |
| Final logits show a narrow expected-versus-selected margin consistent with measured residual drift | validate quality statistically; do not force bitwise kernel parity |
| A numerical change improves the selected token but worsens the control distribution | reject it as overfitting |

Any accepted correction must preserve the fixed trajectory, pass at least 128 tokens of natural decode, improve or
preserve corpus perplexity, avoid fallback, and retain cache-hit throughput.
