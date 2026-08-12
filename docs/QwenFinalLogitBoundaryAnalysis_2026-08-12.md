# Qwen Final Logit Boundary Analysis (2026-08-12)

## Scope

This report closes the final numerical boundary for the first natural Qwen decode rank mismatch observed at generated
index 23. It compares LiteNN and llama.cpp at the same input token and decode coordinate, reconstructs each runtime's
logits from its captured final residual, and separates final-kernel behavior from residual-stream drift.

The report contains no machine-specific model or artifact paths. Implementation status and follow-up work remain in
`docs/Roadmap.md` and `docs/PerformanceOptimizationRoadmap.md`.

## Controlled Setup

| Dimension | Value |
| --- | --- |
| Model class | Qwen2.5-Coder 14B Instruct, Q4_K_M |
| Candidate runtime | LiteNN stateful CPU AOT, paged-reference decode, LLVM O0, strict activation math, explicit T8 |
| Candidate weights | all eligible projections prepacked as field-interleaved-v4 |
| Reference runtime | vendored llama.cpp/ggml `b81c2cdd7`, CPU only, flash attention disabled |
| Prompt | Qwen chat-template tokenization of `hello`, 9 tokens |
| Decode trajectory | fixed 24-token reference trajectory, captured before selecting generated token 24 |
| Compared coordinate | generated index 23, absolute step 32, position 31, input token 4113 |
| Final residual | block 47 `post_ffn`, Float32 `[1,5120]` |
| LM head | tied/output-major Q6_K, vocabulary size 152064 |

Both checkpoint manifests identify the same generated index, absolute step, position, input token, layer, dtype, and
shape. This is a same-session comparison: each runtime's logits and block-47 residual were written synchronously from
the run used by the verifier. Older logits captured from different sessions are not used.

## Verification Method

For each runtime, the verifier performs the following sequence:

1. Load the block-47 `post_ffn` residual from the validated checkpoint manifest.
2. Apply LiteNN's production Float32 RMSNorm helper with the imported `output_norm.weight` and model epsilon.
3. Apply LiteNN's production field-interleaved-v4/Q8_K Q6_K projection helper with the imported LM-head weight.
4. Compare the reconstructed logits with the logits emitted by that runtime in the same session.
5. Compare candidate and reference residuals, normalized values, complete logits, top-k ranks, and the two disputed
   token margins.

The diagnostic command is reproducible without embedding a local model path:

```text
litenn_gguf_convert --verify-llama-final-logits <model.gguf> \
  <litenn-checkpoints> <llama-checkpoints> 23 \
  <litenn-logits.txt> <llama-logits.txt> <report.json> 8 10
```

## Reconstruction Closure

| Comparison | Max abs | RMS error | NRMSE | Cosine similarity |
| --- | ---: | ---: | ---: | ---: |
| LiteNN reconstructed vs LiteNN actual logits | 0 | 0 | 0 | 1 |
| llama residual reconstructed vs llama actual logits | 7.629e-6 | 1.250e-6 | 4.064e-7 | 0.99999999999985 |

LiteNN's actual logits are reproduced bit-for-bit. More importantly, feeding llama.cpp's captured residual through
LiteNN's production final RMSNorm and Q6_K LM head reproduces llama.cpp's actual logits to ordinary Float32 ordering
noise. Therefore neither final RMSNorm semantics nor the production LM-head projection is the unique owner of the
rank mismatch.

## Cross-Runtime Drift

| Boundary | Max abs | Mean abs | RMS error | Reference RMS | NRMSE | Cosine similarity |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| block-47 `post_ffn` | 67.0148 | 2.74885 | 3.67006 | 18.8460 | 0.194739 | 0.981777 |
| final normalized hidden | 4.15072 | 0.314704 | 0.407503 | 1.51285 | 0.269361 | 0.964942 |
| actual logits | 3.47967 | 0.505332 | 0.624026 | 3.07611 | 0.202862 | 0.986768 |

The final kernels preserve, rather than create, the distinction already present in the block-47 residual. The change
in NRMSE across boundaries is not by itself an implementation defect because the normalization and projection applied
to the reference residual reproduce the reference logits almost exactly.

## Top-K And Margin

| Runtime | Rank 1 | Rank 2 | Rank 3 | Top-1 margin |
| --- | --- | --- | --- | ---: |
| LiteNN | 3151: 19.039795 | 358: 18.853163 | 498: 18.241762 | 0.186632 |
| llama.cpp | 498: 20.956869 | 3151: 20.477192 | 389: 19.896915 | 0.479677 |

The decisive paired margin is defined as `logit(reference token 498) - logit(candidate token 3151)`:

| Runtime | Paired margin |
| --- | ---: |
| LiteNN | -0.798033 |
| llama.cpp | +0.479677 |
| LiteNN minus llama.cpp | -1.277710 |

The ordinary LiteNN top-1 margin is only `0.186632` because token 358, not token 498, is its runner-up. Keeping the
ordinary top-1 margin separate from the disputed-token paired margin prevents an incorrect claim that only 0.186632
logit units explain the cross-runtime rank reversal.

## Diagnostic Runtime

The instrumented LiteNN capture completed without fallback in `52.9 s`, including `43.732 s` of diagnostic AOT build.
The 32 executed steps took `8.397 s`; the 24 generated steps took `6.506 s`, or `271.09 ms/generated token`
(`3.689 token/s`). The artifact exposed 62 outputs per step and used LLVM O0, so this row is diagnostic evidence, not a
production throughput benchmark.

## Conclusions

1. Final RMSNorm and the Q6_K LM head are closed as unique implementation-error candidates. The same LiteNN helpers
   reproduce llama.cpp logits from llama.cpp's residual with `4.064e-7` NRMSE.
2. The index-23 rank reversal is caused by the residual stream presented to the final normalization, not by a new
   discrepancy introduced after block 47.
3. Late Q4_K Gate/Up, strict SwiGLU, Q6_K Down, final RMSNorm, and the LM head are all closed by identical-input
   evidence. Cross-runtime ratios measured with different inputs must not be used to rewrite these kernels.
4. One rank mismatch after 23 matching generated tokens is insufficient evidence of a model-quality regression.
   Independent quantized execution is not expected to be bitwise identical, especially around modest top-token
   margins.
5. No model-math change should be accepted solely to select token 498 at this coordinate. The next correctness gate is
   distributional: natural-decode agreement, corpus perplexity, task quality, and throughput must be evaluated before
   any deeper numerical intervention.
6. The default-thread step-4 stall is independent of this numerical result and remains the immediate runtime
   correctness investigation.

## Decision Gates

| Result | Decision |
| --- | --- |
| Observed: both runtimes' logits reconstruct from their own residuals | close final RMSNorm and LM-head correctness |
| Observed: the rank reversal is already encoded in `post_ffn` | classify index 23 as accumulated residual drift |
| Quality gates remain statistically equivalent | accept non-bitwise quantized execution; do not alter model math |
| Quality gates show a repeatable regression | add same-input whole-block attribution, including attention and KV state |
| A proposed fix changes one token but worsens perplexity, task quality, or throughput | reject it as overfitting |
