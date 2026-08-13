# Qwen First-Divergence RoPE Analysis (2026-08-12)

> Follow-up: `docs/QwenNeoXRoPEFixEvidence_2026-08-13.md` records the implemented fix, pre/post-RoPE attribution,
> whole-network recovery, and the post-fix 192-token quality campaign.

## Scope

This report localizes LiteNN's deterministic Qwen2.5-Coder 14B Q4_K_M natural-generation regression at the first
same-context decision where each of three prompts diverges from llama.cpp. It combines measured whole-layer and
block-internal checkpoints with a source-level RoPE semantic audit. No machine-specific model or artifact path is
retained.

The purpose is correctness attribution, not throughput comparison. The earlier natural-generation campaign remains
the model-level gate; this report identifies the first implementation contract that must be repaired before further
model-math or performance tuning can be trusted.

## Controlled Method

The explanation, C++, and reasoning prompts used the exact reference token history through their first divergent
decision. This keeps prompt tokens, generated-prefix tokens, current input token, absolute position, and KV history
identical between runtimes. It also covers both execution forms: the C++ case diverges at the prefill decision, while
the other two cases diverge during stateful decode.

LiteNN used stateful CPU AOT, strict activation math, LLVM O0, a 256-token cache capacity, and the default thread
policy. llama.cpp provided independently computed reference checkpoints. Both manifests agreed on generated index,
absolute step, position, and current input token before any tensor comparison. The campaign captured all 48
post-block residuals and 13 internal boundaries in block 0. The complete run took 263.6 seconds with diagnostic
capture enabled.

## Whole-Layer Results

Every first-divergence case already exceeds 1% NRMSE after block 0. The error does not appear first in a late FFN or
the final logits head.

| Prompt class | Prompt tokens | First divergence | Block 0 NRMSE | Peak layer / NRMSE | Block 47 NRMSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Explanation | 21 | 10 | 0.245696 | 14 / 0.448772 | 0.175762 |
| C++ | 26 | 0 (prefill) | 0.182373 | 5 / 0.341429 | 0.121301 |
| Reasoning | 40 | 3 | 0.289230 | 30 / 0.441850 | 0.139108 |

Whole-layer data alone establishes an early error but cannot distinguish projection, RoPE, attention, or residual
composition. The block-internal capture provides that boundary.

## Block 0 Boundary Results

| Boundary | Explanation NRMSE | C++ NRMSE | Reasoning NRMSE |
| --- | ---: | ---: | ---: |
| Attention norm | 3.561e-7 | 4.713e-7 | 9.139e-7 |
| Rotated query | 1.000884 | 0.854149 | 0.966437 |
| Rotated key | 0.357427 | 0.276941 | 0.354332 |
| Value | 0.004560 | 0.010771 | 0.008886 |
| Attention context | 0.310061 | 0.300423 | 0.302404 |
| Attention output | 0.257036 | 0.209937 | 0.244833 |
| Attention residual | 0.253896 | 0.209433 | 0.241812 |
| FFN norm | 0.359993 | 0.429737 | 0.470522 |
| FFN gate | 0.303589 | 0.388258 | 0.460706 |
| FFN up | 0.319404 | 0.419370 | 0.501850 |
| FFN SwiGLU | 0.287125 | 0.496512 | 0.611469 |
| FFN down | 0.286342 | 0.267404 | 0.376526 |
| Post-FFN | 0.245696 | 0.182373 | 0.289230 |

The discriminating boundaries show a categorical change rather than gradual quantized-kernel drift:

| Boundary | NRMSE range | Cosine-distance range | Maximum absolute error range |
| --- | ---: | ---: | ---: |
| Attention norm | 3.561e-7 to 9.139e-7 | below 1.0e-14 | 2.980e-7 to 1.073e-6 |
| Rotated query | 0.854149 to 1.000884 | 0.364767 to 0.500817 | 8.993 to 12.241 |
| Rotated key | 0.276941 to 0.357427 | 0.038349 to 0.063876 | 8.988 to 11.184 |
| Value | 0.004560 to 0.010771 | 1.024e-5 to 5.800e-5 | 0.004016 to 0.005264 |

Attention norm matching below `9.14e-7` NRMSE closes token history, embedding selection, block input, and first
RMSNorm as meaningful owners. Value projection does not pass through RoPE and remains within `1.08%` NRMSE, which is
consistent with expected quantized projection-order differences. Rotated Q and K instead jump immediately to
`27.7%-100.1%` NRMSE and then contaminate attention and the rest of the network.

The current checkpoints expose only post-RoPE Q/K. Pre-RoPE Q/K capture is still required to quantify the smaller
projection residual independently, but it cannot explain away the separate layout mismatch established below.

## Source-Level Semantic Audit

llama.cpp classifies `LLM_ARCH_QWEN2` as `LLAMA_ROPE_TYPE_NEOX` in
`third_party/llama.cpp/src/llama-model.cpp`. Its documented NeoX layout pairs values offset by half the rotated head
dimension: an eight-element head is arranged as `[ccccssss]`. Normal RoPE instead uses adjacent pairs and is
documented as `[cscscscs]` in `third_party/llama.cpp/ggml/include/ggml.h`.

LiteNN has no RoPE layout field. `RoPENode` contains only input, optional positions, base, frequency scale, and
position offset, while its public contract explicitly says that rotation is over adjacent pairs. The CPU AOT helper
reads `pair * 2` and `pair * 2 + 1`; generic MLIR reshapes the feature dimension to `[halfDim, 2]`. The Qwen GGUF
builder calls this generic adjacent-pair operation without carrying architecture-specific layout metadata.

Therefore the measured Q/K discontinuity has a direct semantic cause: Qwen2 requires half-split NeoX pairing, while
LiteNN executes adjacent-pair normal RoPE. This is a contract defect shared by graph construction and lowering, not a
small floating-point accumulation difference.

## Why Existing Tests Missed It

Current layer tests encode adjacent-pair values as the expected semantics. The dynamic AOT test compares compiled
execution against LiteNN's own Interpreter, so it verifies consistency between two implementations of the same
incomplete contract. It does not compare against an external NeoX formula or an architecture carrying explicit RoPE
layout metadata.

The earlier whole-layer and late-block reports remain valid for the exact boundaries they tested. In particular,
same-input late FFN and final-head checks still close those kernels locally. They could not identify the initial
model-wide drift because their inputs already contained the block-0 attention error. This finer-grained evidence
supersedes generic quantized accumulation as the explanation for the first block's error onset.

## Conclusions

1. The natural-generation regression has a measured first owner: block-0 rotated Q/K, immediately after an almost
   exact attention-normalized input.
2. Source inspection independently proves that LiteNN applies normal adjacent-pair RoPE to a Qwen2 model that
   requires NeoX half-split pairing.
3. The issue affects prefill and stateful decode, so it is not specific to KV-cache growth or post-prefill state.
4. Performance optimization against the current Qwen output is secondary until the RoPE contract is corrected and
   the same distributional gate is rerun.
5. A RoPE fix is strongly expected to remove the measured Q/K discontinuity, but restored end-to-end quality remains
   an acceptance result, not an assumption.

## Required Acceptance Gates

1. Add an explicit RoPE layout enum to the graph and executable-plan contract; include it in validation, hashing,
   serialization, diagnostics, cloning, and transforms.
2. Import Qwen2 as NeoX from GGUF architecture metadata. Unknown or unsupported layouts must fail explicitly instead
   of silently selecting adjacent pairs.
3. Implement the layout in the Interpreter, CPU AOT helper, and generic MLIR path. Other enabled backends must either
   implement identical semantics or reject the layout before execution.
4. Add independent normal and NeoX formula fixtures at position zero and nonzero positions, including multi-head
   head-local pairing and static/dynamic position paths. Add pre-RoPE Q/K checkpoints to isolate projection residual.
5. Repeat the exact three-prompt first-divergence attribution and 192-token natural-generation campaign. Require the
   attention-norm baseline to remain stable, remove the rotated-Q/K semantic jump, retain finite/no-fallback status,
   and demonstrate a material quality improvement without exceeding the accepted throughput variance.

## Reproduction Shape

Use `example/gguf/qwen_first_divergence_attribution.py` with a local `model.gguf`, a completed natural-generation
quality report, `litenn_gguf_convert`, and `litenn_llamacpp_adapter`. The driver replays the exact llama.cpp token
prefix at each first divergence, supports prefill decision zero, captures whole layers and selected sub-layer blocks,
and writes a structured JSON summary. Machine-local model and output paths belong only in untracked build artifacts.
