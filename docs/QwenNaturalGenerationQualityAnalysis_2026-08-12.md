# Qwen Natural Generation Quality Analysis (2026-08-12)

## Scope

This report evaluates whether LiteNN preserves natural greedy generation quality against an independently executed
llama.cpp reference. It is deliberately separate from forced-token numerical parity and throughput reports. No
machine-specific model or artifact path is retained.

## Method

The controlled model was Qwen2.5-Coder 14B Instruct Q4_K_M. Three public English prompts covered explanation, C++, and
short quantitative reasoning. Both runtimes consumed the exact same chat-template token ids, used greedy sampling,
ran 64 natural generated tokens per prompt, and emitted the full 152,064-entry decision distribution at every step.
LiteNN used stateful CPU AOT, strict activation math, LLVM O0, a 256-token cache capacity, and the default 32-logical-
processor policy. Both manifests explicitly reported no fallback, and the gate rejected sparse or non-finite logits.

Decision step zero is the prefill distribution that selects the first generated token. The first divergent decision
still has an identical input context, so it is included in same-context top-k metrics. Later steps consume different
token histories and are reported only as trajectory-level evidence.

The campaign was executed twice. Each run covered 192 reference and 192 candidate tokens. A stable projection of the
two reports, including all selected token ids, first-divergence diagnostics, and aggregate metrics, had the identical
SHA-256 digest `2ff6220874dfad708ec5e86a4eb729d8460788a2039a4ecc05326bf37c3a02bf`.

## Results

| Metric | Campaign A | Campaign B |
| --- | ---: | ---: |
| Valid cases | 3/3 | 3/3 |
| Reference/candidate tokens | 192/192 | 192/192 |
| Weighted token-prefix agreement | 6.7708% | 6.7708% |
| Same-context top-10 overlap, mean | 83.125% | 83.125% |
| Same-context top-10 overlap, minimum | 70.0% | 70.0% |
| Trajectory top-10 overlap, mean | 13.6979% | 13.6979% |
| Median first-divergence decision | 3 | 3 |
| Fallback/non-finite cases | 0/0 | 0/0 |

| Prompt class | Common prefix | First divergence | Same-context top-10 mean/min | Trajectory top-10 mean |
| --- | ---: | ---: | ---: | ---: |
| Explanation | 10/64 | 10 | 81.82% / 70.0% | 17.97% |
| C++ | 0/64 | 0 | 90.0% / 90.0% | 8.59% |
| Reasoning | 3/64 | 3 | 85.0% / 80.0% | 14.53% |

The first-divergence distributions disagree on the top two tokens rather than failing through random sampling:

| Prompt class | Reference/candidate token | Reference preference | Candidate preference |
| --- | --- | ---: | ---: |
| Explanation | 1576 / 369 | token 1576 by `+0.0602742` | token 369 by `+0.516001` |
| C++ | 95456 / 8420 | token 95456 by `+0.379282` | token 8420 by `+0.668711` |
| Reasoning | 13940 / 1372 | token 13940 by `+1.557903` | token 1372 by `+0.122990` |

The configured quality gate required at least two cases, 128 reference tokens, 95% weighted prefix agreement, and
90% mean same-context top-10 overlap. Coverage and integrity passed; both quality thresholds failed in both campaigns.

## Runtime Observation

The first campaign also disproved the initial generation-only atomic-wait thread-pool repair. Its third LiteNN case
stalled at forward step 45 with `desiredWorkers=31`, `workersDone=30`, the caller spinning in `ParallelFor`, and every
worker blocked through libwinpthread. A mutex/condition-variable predicate superseded the atomic blocking path. The
same case then completed 103 total prompt-plus-generation forwards, and the full repeat campaign completed under the
default thread policy. Five 4096-call stress processes, 8 CPU-parallel tests, and 22 quantized/attention tests passed.

Full-vocabulary text capture is diagnostic rather than production-shaped: 384 files occupied 1.07 GiB for one paired
campaign, and offline comparison took about 39 seconds. A compact indexed Float32 container is therefore required
before this gate is practical in routine CI.

## Conclusions

1. LiteNN has a deterministic, repeatable natural-generation quality regression on this Q4_K_M model. It is not an
   EOS, fallback, non-finite, tokenizer, prompt-template, or random-sampler artifact.
2. The regression is visible at identical-context distributions, including a prefill decision, so post-divergence
   trajectory differences are not the only signal.
3. Previous single-prompt forced-token and late-layer diagnostics were too narrow to establish model-level quality.
   Their kernel-local conclusions remain useful, but they do not close the end-to-end correctness objective.
4. The next correctness P0 is same-input whole-block attribution at the three first-divergence boundaries, including
   prefill capture and representative early, middle, and late blocks. Throughput-oriented model-math changes remain
   secondary until the earliest drift owner is measured.

## Reproduction Shape

Use `example/gguf/qwen_quality_campaign.py` with a local model, `litenn_gguf_convert`, and
`litenn_llamacpp_adapter`. The default prompt set contains three prompts and 64 tokens each. The driver writes paired
manifests, JSON/Markdown reports, supports separated AOT cache reuse, and returns nonzero when a configured quality
threshold fails.
