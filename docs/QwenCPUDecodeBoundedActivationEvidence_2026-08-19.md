# Qwen CPU Decode Bounded Activation Evidence (2026-08-19)

## Decision

LiteNN now owns a compact bounded Float32 exponential/SwiGLU kernel derived from the repository's pinned ggml
implementation. This avoids adding a second vector-math dependency or linking the complete ggml runtime into
`LiteNNCompiler`. Strict scalar `std::exp` remains the default and reference policy; bounded math is explicit in the
compiler options, helper ABI, rodata feature set, and GGUF AOT cache identity.

The bounded contract is:

- maximum advertised exponential error: 2 ULP;
- largest finite exponential input: `0x1.62e42ep+6F` (`88.7228317`, corrected on 2026-09-22);
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

## Range Correction (2026-09-22)

The original scalar tail clamped `exp` to infinity above `88.3762627`, earlier than the Float32 overflow boundary.
The vector and strided paths did not have that clamp. A regression reproduced `gate=-88.5, up=1e38` returning `-0`
in scalar lanes while the strict result was `-32.4998665`. The previous absolute-only tolerance hid the unamplified
error. The tail now delegates large exponents and non-finite inputs to `std::exp`; the normal SIMD polynomial is
unchanged. Capability metadata reports the corrected largest finite input, verified against its next representable
Float32 neighbor.

Range tests cover widths 1/7/8/9/15/16/17, contiguous and stride-2 storage, amplified upstream values, signed zero,
NaN, infinity, and overflow neighbors. The test failed before the fix and passed after it. The focused compiler/GGUF
CTest set passed 196/196. On the current GCC 16.2 Release build, five 48-call repetitions measured `7.19 ms` strict,
`0.144 ms` built-in bounded, and `0.392 ms` pinned ggml. Contiguous strict/bounded wall CV was `0.49/0.37%`; maximum
absolute/relative delta stayed `9.54e-7/3.47e-7` with zero special-value mismatches. Strided timing was noisy and is not
used for a speedup claim. These are current same-binary controls, not a speedup against the August toolchain.

## Post-Correction Position Control (2026-09-22)

Three alternating clean/profile pairs used 32 forced generated tokens, T8/adaptive/O0, bounded math, v4 prepacked
weights, and the existing 41-position cache. The raw bundle is `build/cpu_activation_validation/bounded/stages.json`.
No model payload was copied. Both position bins and all measured runs preserved the fixed trajectory, expected helper
shape, cache-hit-only execution, and zero fallback. The whole-window clean/profile module medians were
`164.022/160.822 ms/token`; paired median module overhead was `-1.29%` with `100%` accounting coverage.

Raw activation/Down/Gate-Up medians were `1.027/39.885/63.532 ms/token`. These are diagnostic, not accepted stage deltas:
whole-window variance and first-bin stage variance failed (first-bin major stages reached approximately `24%` CV).
Every sample is retained. The aggregate's `100%` coverage includes an explicitly computed module residual and therefore
does not independently prove all execution was attributed. No speedup against August or llama.cpp follows from this
campaign; fresh equal-thread controls remain required.
