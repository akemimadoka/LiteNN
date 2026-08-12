# Qwen Natural Decode Layer Drift Evidence (2026-08-12)

## Decision

The natural greedy mismatch at generated index 23 is not a first-nonzero-error event. LiteNN and llama.cpp already
produce different Float32 hidden values at block 0 while following the same token trajectory, as expected from
independent quantized kernels and accumulation orders. A zero-tolerance "first failing layer" therefore selects block 0
at every measured position and is not a useful localization rule.

The useful new signal is relative amplification. Compared with generated index 22, index 23 has lower cross-runtime
error through the first blocks, crosses above the index-22 normalized error around block 9, and shows its strongest,
most consistent additional error in blocks 39-47. Block 47 RMS error rises from `3.03993` to `3.67006` (`+20.73%`),
while reference-RMS-normalized error rises by `5.10` percentage points. This makes the late decoder blocks and final
normalization/logits path the first diagnostic window, but it does not yet prove that one late-block operator is wrong.

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

## Runtime Observations

The LiteNN diagnostic run reproduced exactly one forced-token argmax mismatch, at generated index 23. It reported:

- AOT build: `30,998.7 ms` with cache publication disabled.
- Decode run: `6,422.85 ms` for 32 total single-token forwards.
- Generated-token module average: `204.787 ms`, or `4.883 token/s` in this diagnostic run.
- Checkpoint writes: `3.297 ms` at index 22 and `3.010 ms` at index 23.
- Runtime outputs: 49 per invocation, consisting of public logits plus 48 diagnostic hidden states.

These timings validate that checkpoint output is a small selected-step cost. They are not accepted throughput numbers:
the diagnostic ABI changes output liveness and the llama.cpp callback synchronizes selected graph nodes.

## Conclusions And Next Gates

1. Do not change model math based on zero-tolerance hidden-state mismatch. Independent Q4_K/Q6_K kernels differ from
   block 0, while the generated sequence remains equal through index 22.
2. Measure indices 16-23 (preferably 0-31) to build a per-block neighborhood distribution. Index 23 should be treated
   as exceptional only where its NRMSE/cosine delta exceeds normal token-to-token variation.
3. Add selectable sub-layer checkpoints for the late-block window first: attention norm, Q/K/V after bias and RoPE,
   attention output, post-attention residual, FFN norm, Gate/Up, SwiGLU, Down, and post-FFN residual. Use blocks 9,
   19-25, and 38-47 as crossover, sustained-growth, and strongest-growth windows.
4. Capture final RMSNorm and aligned logits, including top candidates, the expected-vs-selected token margin, and the
   contribution of output projection error. An argmax mismatch can be benign when the reference margin is tiny.
5. After localization, validate a candidate with the same forced trajectory, natural greedy parity beyond 128 tokens,
   corpus perplexity delta, and unchanged cache-hit performance before promoting it.
