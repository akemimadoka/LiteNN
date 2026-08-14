# Qwen Fixed-Reference Trajectory Analysis (2026-08-14)

## Scope

This report compares LiteNN CPU AOT and llama.cpp logits while both runtimes consume the same reference token
trajectory. It evaluates the post-NeoX-RoPE Qwen2.5-Coder 14B Q4_K_M implementation without allowing an early greedy
token difference to change later inputs. No machine-specific model or artifact path is retained.

The campaign covers three prompt classes and 64 decisions per prompt, for 192 paired full-vocabulary distributions.
LiteNN used the normal stateful decode artifact with an AOT cache hit, no fallback, and finite logits throughout.

## Predeclared Gate

The acceptance thresholds were set before inspecting this run:

| Check | Required | Measured | Result |
| --- | ---: | ---: | --- |
| Fixed-trajectory top-1 agreement | >= 95% | 98.9583% | PASS |
| Mean top-10 overlap | >= 95% | 99.0104% | PASS |
| Worst centered-logit cosine | >= 0.999 | 0.998840426 | FAIL |
| Worst Jensen-Shannon divergence | <= 0.001 nats | 0.001292896 | FAIL |

The aggregate result is therefore **FAIL**. The thresholds are not relaxed after measurement.

## Aggregate Results

| Metric | Mean | Worst |
| --- | ---: | ---: |
| Top-1 agreement | 98.9583% (190/192) | n/a |
| Top-10 overlap | 99.0104% | 80% |
| Centered-logit NRMSE | 0.0262642 | 0.0481669 |
| Centered-logit cosine | 0.999652735 | 0.998840426 |
| KL(reference || LiteNN) | 0.000254584 nats | 0.00517479 nats |
| Jensen-Shannon divergence | 0.0000637318 nats | 0.00129290 nats |
| Total variation distance | 0.00466768 | 0.0481971 |
| Reference top-token rank in LiteNN | 1.0104 | 2 |

All 192 forced context tokens agree by construction and were validated before comparison. The reference top token is
never ranked below second by LiteNN.

## Per-Case Results

| Case | Top-1 | Top-10 mean / min | NRMSE mean / max | Cosine min | JS mean / max | TV max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Explanation | 96.875% | 99.531% / 80% | 0.026383 / 0.031384 | 0.999511475 | 0.000123181 / 0.00129290 | 0.0481971 |
| C++ | 100% | 98.906% / 90% | 0.026335 / 0.034048 | 0.999420705 | 0.000027407 / 0.000465697 | 0.0241122 |
| Reasoning | 100% | 98.594% / 90% | 0.026075 / 0.048167 | 0.998840426 | 0.000040607 / 0.000398180 | 0.0221666 |

## Top-1 Mismatches

Only two decisions change top-1, both in the explanation case. In both decisions the reference token remains LiteNN's
second-ranked token and the top-10 set is identical.

| Step | Reference / LiteNN token | Reference / LiteNN margin | Cosine | JS | TV |
| --- | --- | --- | ---: | ---: | ---: |
| 10 | 1576 / 369 | 0.0602742 / 0.0548668 | 0.999593990 | 0.000411357 | 0.0285706 |
| 33 | 3321 / 7292 | 0.1757546 / 0.0147706 | 0.999577166 | 0.001292896 | 0.0481971 |

Step 33 owns the campaign's maximum KL, Jensen-Shannon, and total-variation values. It is the first probability-level
outlier to investigate if corpus loss confirms a material quality regression.

## Metric Interpretation

The worst centered-logit cosine occurs at reasoning step 61, not at either top-1 mismatch. Both runtimes select token
17, top-10 overlap is 100%, and their top margins are 8.1374 and 8.0686. Although cosine falls to 0.998840426, the
Jensen-Shannon divergence is only 7.39e-7 nats and total variation is 6.72e-5. The cosine failure is therefore driven
mainly by low-probability vocabulary-tail differences rather than a changed predictive distribution near the top.

Centered full-vocabulary cosine remains a useful numerical diagnostic, but it must not independently authorize model
math or quantized-kernel changes. Probability-space divergence and corpus next-token loss are the stronger acceptance
owners. The existing strict threshold remains recorded until corpus evidence supports changing the policy.

## Artifact Cost

The LiteNN side alone writes 192 per-position text files totaling 561,285,622 bytes (0.523 GiB). Reusing the reference
artifacts still leaves a complete comparison near the previously measured 384-file, 1.07-GiB scale. Re-evaluating the
existing artifacts with full probability metrics took about 77 seconds. This makes a compact indexed Float32 logit
container a practical prerequisite for routine regression use, not merely a storage cleanup.

## Conclusions

1. The repaired NeoX path is distributionally close under identical contexts: 190/192 top-1 decisions match, mean
   top-10 overlap is 99.01%, and the reference winner never falls below rank 2.
2. The predeclared gate still fails honestly on one centered-cosine tail outlier and one Jensen-Shannon outlier.
3. The cosine and probability failures have different owners. Reasoning step 61 is tail-sensitive but prediction-
   stable; explanation step 33 is the meaningful probability-level outlier.
4. There is not yet evidence that the residual changes aggregate language-model quality. A fixed public-corpus
   cross-entropy/perplexity comparison is the next P0 correctness gate.
5. If corpus loss shows a material regression, the next localization target is explanation step 33 under identical
   inputs, starting with projection and sub-layer boundaries. If corpus loss is equivalent, the residual should remain
   documented rather than trigger speculative kernel rewrites.
6. Normal cache-hit throughput acceptance remains separate and open; this full-logit diagnostic run is not a valid
   decode-speed measurement.

## Next Gates

1. Add a reproducible public text-slice manifest, tokenize it through the model tokenizer, and compare per-token and
   per-sample NLL, aggregate cross-entropy, and perplexity under an identical teacher-forced trajectory. The existing
   llama.cpp `decode-logits` path captures after feeding each supplied token, so the corpus path must expose an explicit
   pre-target decision boundary and test the one-token shift rather than reusing those files by name alone.
2. Preserve explicit absolute and relative regression thresholds, token coverage, finite values, cache-hit state, and
   fallback status in the corpus report.
3. Investigate explanation step 33 with same-input layer/sub-layer checkpoints only if the corpus gate identifies a
   material loss delta.
4. Replace text logits with a compact indexed Float32 container before promoting the distribution and corpus campaigns
   into routine CI.
5. Run the normal cache-hit throughput campaign separately with the existing variance and trajectory gates.
