# Qwen CPU Decode Bounded Activation Evidence (2026-08-19)

## Decision

LiteNN now owns a compact bounded Float32 exponential/SwiGLU kernel derived from the repository's pinned ggml
implementation. This avoids adding a second vector-math dependency or linking the complete ggml runtime into
`LiteNNCompiler`. Strict scalar `std::exp` remains the default and reference policy; bounded math is explicit in the
compiler options, helper ABI, rodata feature set, and GGUF AOT cache identity.

The bounded contract is:

- maximum advertised exponential error: 2 ULP;
- overflow input: `88.3762626647949F`;
- underflow input: `-103.972084045410F`;
- preserved signed zero, NaN, and infinity behavior for SwiGLU;
- AVX2+FMA dispatch on supported x86 hosts, bounded scalar tail handling, and strict scalar fallback for strided rows.

## Production-Shape Microbenchmark

The benchmark shape is 48 calls at width 13824. Seven aggregate-only repetitions were used. Host state changed
substantially between the two strict measurements (`11.6 ms` in the initial run and `51.9 ms` in the repeated run),
while the built-in bounded row remained stable at `0.182-0.184 ms`. The conservative initial comparison is therefore
used for the promotion ratio.

| Path | 48-call median | Versus strict | Maximum abs/rel error | Special mismatches |
|---|---:|---:|---:|---:|
| strict `std::exp`, contiguous | 11.6 ms | baseline | 0 / 0 | 0 |
| built-in bounded, contiguous | 0.182 ms | 63.7x faster, 11.4 ms saved | 9.54e-7 / 3.47e-7 | 0 |
| pinned ggml control, contiguous | 0.512 ms | 22.7x faster | 9.54e-7 / 3.47e-7 | 0 |

The strided bounded route intentionally executes strict scalar math because gather/scatter plus approximate scalar
math regressed. It therefore preserves strict results without making non-contiguous workloads slower.

## Full-Model A/B

The production control is a 14B Q4_K_M model, stateful CPU AOT, T8/adaptive, O0, all profitable prepacking enabled,
field-interleaved-v4, 32 generated tokens, cache-hit-only loading, and no fallback. Strict and bounded runs reuse one
central 9.16 GB shared-weight payload; only their small instruction artifacts differ.

| Pair order | Strict ms/token | Bounded ms/token | Improvement | Strict / bounded token/s |
|---|---:|---:|---:|---:|
| strict then bounded | 179.832 | 170.476 | 5.20% | 5.561 / 5.866 |
| bounded then strict | 175.558 | 170.202 | 3.05% | 5.696 / 5.875 |
| strict then bounded | 176.209 | 160.348 | 9.00% | 5.675 / 6.236 |

The paired median improvement is `5.20%`; all three pairs are positive. Every run produced the same token ids and
decoded text, loaded the expected AOT artifact, and reported `fallback_count=0`.

## Stage Attribution

An additional cache-hit helper profile used the same prompt and eight generation tokens. Generation-step medians are:

| Stage | Strict | Bounded | Strict minus bounded |
|---|---:|---:|---:|
| FFN activation | 12.338 ms | 1.038 ms | 11.301 ms |
| FFN Down | 41.969 ms | 42.832 ms | -0.863 ms |
| helper total | 167.702 ms | 159.861 ms | 7.841 ms |
| module residual | 11.779 ms | 12.668 ms | -0.889 ms |
| whole step | 179.850 ms | 172.944 ms | 6.907 ms |

The activation reduction is directly observed and is not transferred to Down. Gate/Up and residual variation absorb
part of the local saving, so the unprofiled alternating full-model result remains the throughput authority.

## Verification And Next Boundary

Capability, special-value/tail, standalone/fused artifact, rodata load, imported GGUF fusion, shared-weight-cache, and
profile-classification tests pass. The implementation clears both promotion gates: more than `2x` or `5 ms` in the
48-layer activation sequence and at least `3%` median in three exact-token cache-hit pairs.

This closes the confirmed activation implementation deficit. Remaining cross-runtime work is controlled end-to-end
closure: reproduce the strongest reference provenance under matched host state, then re-rank the now smaller
Gate/Up, logits, attention-output, and module-residual differences rather than reopening FFN Down.
