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

### `Layer::AddRepeat`

- Purpose: represent ggml `REPEAT` tiling without adding a dedicated repeat node.
- Shape contract: target rank must be at least the input rank, and each target dimension must be an integer multiple of the corresponding input dimension after leading-one padding.
- Internals: it rewrites through `Reshape + BroadcastTo + Reshape`, so non-singleton dimensions tile in row-major order.

### `Layer::AddWindowPartition` and `Layer::AddWindowUnpartition`

- Purpose: represent ggml `WIN_PART` / `WIN_UNPART` for vision-style local attention paths.
- Shape contract: inputs use LiteNN row-major semantic `[channels, width, height, batch]`; partition output is `[channels, window, window, windows * batch]` with zero padding to a multiple of the window size.
- Reason it is compatibility-only: the helper preserves the ggml/SAM-style channel-width-height-batch convention locally instead of introducing it as a general LiteNN image layout.

### `Layer::AddGetRelativePosition` and `Layer::AddRelativePositionBias2D`

- Purpose: represent ggml `GET_REL_POS` / `ADD_REL_POS` in vision attention lowering.
- Shape contract: `AddGetRelativePosition` gathers from `[2 * size - 1, channels]` into `[query, key, channels]` for the ggml-compatible `query == key` case. `AddRelativePositionBias2D` adds width and height relative-position bias to scores shaped `[qH, qW, kH, kW, heads]`.
- Internals: these helpers rewrite through constant `Gather`, `Reshape`, `BroadcastTo`, and `Add`.

### `Layer::AddSSMConv`

- Purpose: represent ggml `SSM_CONV` for Mamba-style depthwise causal convolution input buffers.
- Shape contract: `convInput=[kernel - 1 + tokens, channels, batch]`, `weight=[kernel, channels]`, result `[channels, tokens, batch]`.
- Internals: it rewrites through permutation, grouped `Conv2DNode`, and reshaping, keeping the operation on the existing convolution substrate.

## Practical Rule For New Lowering Work

- If an operator can be expressed with normal LiteNN row-major tensors plus explicit axes or options, keep it as a general-purpose surface.
- If an operator has to preserve ggml-specific dimension order or accumulator rules to stay compatible, document it as compatibility-only and keep that contract local to GGUF or llama.cpp lowering.