# LiteNN GGML Compatibility Notes

This note separates general-purpose LiteNN operators from compatibility-only surfaces that intentionally preserve ggml or llama.cpp layout contracts.

## Global Layout Rule

- LiteNN tensors use row-major semantic shapes.
- GGML tensors arrive in `ne[]` order and are converted during GGUF import.
- If a builder helper accepts imported GGUF weights in ggml-oriented `[outFeatures, inFeatures]` shape, it must transpose once at import time and materialize LiteNN's normal `[inFeatures, outFeatures]` shape before execution.

## General-Purpose LiteNN Surfaces Used By GGUF Lowering

These operators and helpers keep ordinary LiteNN semantics even when they appear in llama.cpp lowering:

- `GetRowsNode`: row lookup over LiteNN axis 0. Used for token embedding lookup after imported weights are already in LiteNN row-major shape.
- `ArgsortNode` and `Layer::AddTopK`: normal LiteNN sort/select surfaces with an explicit `axis` field. They are not ggml-axis-0 special cases anymore.
- `UnaryOp::Transpose`: standard 2D transpose over LiteNN shapes only.
- `Layer::LinearLayer` / `MakeLinearFromArchive`: execution always uses LiteNN `[inFeatures, outFeatures]` weights. Imported GGUF `[outFeatures, inFeatures]` tensors are normalized into that layout during lowering.
- `Layer::FlashAttnExt`: current single-head 2D attention helper uses LiteNN row-major query/key/value matrices plus explicit options for causal mask, additive mask, softcap, sinks, and scale.

## Compatibility-Only Surfaces

These surfaces intentionally preserve ggml contracts and should be treated as importer-lowering utilities, not as generic LiteNN math building blocks.

### `Layer::AddMulMatId` and `MulMatIdNode`

- Purpose: represent ggml `MUL_MAT_ID` for MoE routing.
- Shape contract: `as=[k, m, expertCount]`, `b=[k, usedExpertSlots, tokenCount]`, `ids=[usedExperts, tokenCount]`.
- Result contract: always Float32 `[m, usedExperts, tokenCount]`.
- Reason it is compatibility-only: the dimension order is ggml-specific and does not follow LiteNN's otherwise normal matrix conventions.

### `Layer::AddId`

- Purpose: match ggml `ADD_ID` semantics by adding expert-selected bias rows into `a[:, usedExpert, token]`.
- Internals: it rewrites through LiteNN `Transpose`, `GetRows`, `Reshape`, and `Add`, but the public shape contract remains tied to ggml expert/token ordering.
- Reason it is compatibility-only: the helper is shaped around imported MoE routing layout rather than a general-purpose bias-add abstraction.

## Practical Rule For New Lowering Work

- If an operator can be expressed with normal LiteNN row-major tensors plus explicit axes or options, keep it as a general-purpose surface.
- If an operator has to preserve ggml-specific dimension order or accumulator rules to stay compatible, document it as compatibility-only and keep that contract local to GGUF or llama.cpp lowering.