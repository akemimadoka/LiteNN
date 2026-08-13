# Qwen NeoX RoPE Fix Evidence (2026-08-13)

## Scope

This report validates the explicit NeoX RoPE implementation against the same Qwen2.5-Coder 14B Q4_K_M evidence
campaign that first identified the adjacent-pair layout defect. It records measured before/after correctness,
separates Q/K projection error from RoPE error with new pre-RoPE checkpoints, and defines the next acceptance work.
No machine-specific model or artifact path is retained.

This is a correctness report. The diagnostic graphs expose additional outputs and are not throughput evidence.

## Implementation Under Test

The graph and executable-plan contracts now carry a required `RoPELayout` value with no implicit legacy default.
`Normal` uses adjacent pairs and `NeoX` uses head-local half-split pairs. The value participates in validation,
hashing, diagnostics, graph transforms, and vNext serialization. Interpreter, CPU AOT, generic MLIR, and CUDA-native
lowering implement the same layouts. GGUF lowering maps `llama` to `Normal`, maps `qwen2` to `NeoX`, and rejects an
unknown architecture instead of guessing.

Independent formula tests cover static and runtime positions. CPU AOT and CUDA-native numerical tests compare both
layouts against independently calculated expected values. The Qwen builder continues to apply RoPE separately to
each attention head, so NeoX's half split is local to `headDim`, not the concatenated hidden width.

## Controlled Method

The explanation, C++, and reasoning inputs replayed llama.cpp's exact token history at the three original first-
divergence decisions: decode step 10, prefill decision 0, and decode step 3. LiteNN used stateful CPU AOT, LLVM O0,
a 256-token cache capacity, and block-0 internal capture. llama.cpp supplied independently computed checkpoints.

The follow-up run captured all 48 post-block residuals and 15 block-0 boundaries. Two new boundaries,
`query_pre_rope` and `key_pre_rope`, have head-major shapes matching the post-RoPE checkpoints. The complete
diagnostic run took 380.3 seconds.

## Whole-Layer Recovery

| Prompt class | Block 0 before | Block 0 after | Reduction | Peak before | Peak after | Reduction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Explanation | 0.245696 | 0.018264 | 92.57% | 0.448772 (block 14) | 0.049465 (block 5) | 88.98% |
| C++ | 0.182373 | 0.013156 | 92.79% | 0.341429 (block 5) | 0.028683 (block 5) | 91.60% |
| Reasoning | 0.289230 | 0.012876 | 95.55% | 0.441850 (block 30) | 0.026350 (block 5) | 94.04% |

The post-fix block-47 NRMSE is `0.009855`, `0.010468`, and `0.007950` respectively. The former model-wide
`0.34-0.45` peak is gone; the remaining whole-network drift stays near the scale expected from repeated quantized
projections and nonlinear amplification.

## Block-0 Boundary Recovery

| Boundary | Explanation | C++ | Reasoning |
| --- | ---: | ---: | ---: |
| Attention norm | 3.561e-7 | 4.713e-7 | 9.139e-7 |
| Query before RoPE | 0.001653141 | 0.002074694 | 0.001559316 |
| Query after RoPE | 0.001653158 | 0.002074728 | 0.001559297 |
| Key before RoPE | 0.000630113 | 0.000711678 | 0.000556859 |
| Key after RoPE | 0.000630113 | 0.000711676 | 0.000556855 |
| Value | 0.004560 | 0.010771 | 0.008886 |
| Attention context | 0.004323 | 0.005543 | 0.004539 |
| Attention output | 0.015795 | 0.016193 | 0.019162 |
| FFN norm | 0.021874 | 0.034300 | 0.037041 |
| FFN down | 0.023123 | 0.018880 | 0.014151 |
| Post-FFN | 0.018264 | 0.013156 | 0.012876 |

The largest absolute pre/post RoPE NRMSE change across all six Q/K comparisons is `3.43e-8`. RoPE therefore adds no
measurable error beyond the incoming projection residual. Query error is `0.156%-0.207%`; key error is
`0.0557%-0.0712%`; both are below the independently observed value-projection error of `0.456%-1.077%`.

This closes RoPE as the remaining mismatch owner. The first threshold crossing now occurs at the quantized Q
projection, while the largest block-0 amplification occurs later around attention output and FFN normalization/down.
Those later values do not by themselves prove a defective kernel because each consumes already different inputs.

## Natural-Generation Recovery

The exact 3-prompt, 192-token campaign was rerun after the fix.

| Metric | Before | After |
| --- | ---: | ---: |
| Weighted exact-prefix agreement | 6.7708% | 71.875% |
| Same-context top-10 overlap, mean | 83.125% | 98.849% |
| Same-context top-10 overlap, minimum | 70% | 90% |
| Natural-trajectory top-10 overlap, mean | 13.698% | 72.396% |
| Median first divergence | Step 3 | Step 10 |
| Cases with no divergence in 64 tokens | 0 / 3 | 2 / 3 |
| Fallback / non-finite cases | 0 / 0 | 0 / 0 |

The C++ and reasoning cases match all 64 greedy tokens. The explanation case first differs at decision 10, where
both runtimes have exactly the same top-10 token set and only swap the top two candidates. llama.cpp prefers token
1576 over 369 by `0.0602742`; LiteNN prefers 369 over 1576 by `0.0548668`. This is a narrow quantized numerical
tie, not the broad semantic distribution failure measured before the fix.

The current strict quality gate still fails because its exact-prefix threshold is intentionally conservative. It
must not be weakened merely to turn this campaign green. Exact greedy prefix is highly discontinuous near a top-2
tie, so acceptance now needs a fixed-reference distributional metric and corpus loss evidence before deciding
whether the residual requires kernel changes.

## Conclusions

1. The explicit NeoX implementation removes the identified Qwen2 RoPE semantic defect in both prefill and decode.
2. Pre/post checkpoints prove that the repaired RoPE contributes at most `3.43e-8` additional NRMSE in this campaign.
3. Whole-network peak error falls by `88.98%-94.04%`, and two of three 64-token greedy generations become exact.
4. The only remaining natural first divergence is a `0.055-0.060` logit-margin top-2 swap with identical top-10
   membership. A tolerance-based argmax or token-specific correction would hide evidence and is rejected.
5. The next correctness priority is distributional evaluation under a fixed reference trajectory, followed by
   public-corpus perplexity/cross-entropy. Projection-kernel changes should be justified by those results.
6. Production throughput acceptance remains open because diagnostic-output runs are not comparable to the normal
   cache-hit decode benchmark.

## Next Acceptance Gates

1. Replay the reference token trajectory for all evaluated positions and compare distributions before natural paths
   diverge. Retain top-k overlap, selected-token margin, rank correlation, and a stable logit-distance metric.
2. Measure perplexity/cross-entropy delta on a fixed public evaluation slice, with aggregate and per-sample results.
3. If those gates show a material regression, isolate Q/K/V/O and Gate/Up/Down projections with identical captured
   inputs and represented-weight references before changing quantized accumulation.
4. Run the normal cache-hit throughput campaign and enforce the existing variance gate; checkpoint builds remain
   diagnostic-only artifacts.
5. Add a small task-quality panel only after distributional and corpus-loss gates are reproducible.

