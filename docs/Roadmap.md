# LiteNN Roadmap

This document is the planning entry for LiteNN. It is organized as a goal tree.
Date notes are kept at the end as historical hints, while the checklists below
are the source of truth for current planning and completion status.

## Goal Tree

### G1: Low Precision and Quantization Foundation

Purpose: make dtype and quantization metadata a stable cross-cutting contract
for tensors, graph validation, serialization, CPU/CUDA runtime paths, compiler
lowering, and compiled artifacts.

#### G1.1 Scalar DType Foundation

Status: completed for scalar storage/reference paths on 2026-05-17.

- [x] Add scalar dtypes: fp16, bf16, fp8e4m3, fp8e5m2, int8, uint8.
- [x] Centralize dtype name, byte size, category, and max-valid-value checks.
- [x] Support CPU allocation, zero fill, dtype conversion, tensor initialization, model serialization, and compiled module metadata.
- [x] Add MLIR type/constant lowering for scalar low precision. FP8 is initially represented as one-byte storage until arithmetic lowering is implemented.
- [x] Add tests for dtype metadata, CPU conversion roundtrip, initializer support, and model serialization.

Completed notes:

- Scalar dtype metadata now lives in one header and is consumed by CPU/CUDA allocation, validation, serialization, optimizer utilities, compiled module metadata, and MLIR translation.
- CPU reference conversion and basic arithmetic are available for fp16/bf16/fp8 storage wrappers; FP8 MLIR lowering currently uses one-byte storage attributes rather than native FP8 arithmetic.
- Model format version was bumped so new dtype values are accepted by current loaders while older files remain readable.

#### G1.2 Quantized Tensor Storage

Status: completed for graph/runtime/storage metadata paths on 2026-05-17.

- [x] Introduce quantized tensor metadata: scale, zero point, group size, axis, and block format.
- [x] Support int8/uint8 affine quantization for per-tensor, per-axis, and grouped weight-style storage in CPU reference helpers.
- [x] Keep scalar int8/uint8 tensors distinct from quantized tensors by requiring explicit `QuantizationParams` on `QuantizedTensor` or `Variable`.
- [x] Define variable quantization serialization payloads and preserve them through ModelIO.
- [x] Add roundtrip tests for quantization metadata, CPU reference quantize/dequantize, and ModelIO.
- [x] Add graph-level Dequantize/Quantize nodes and interpreter/const-fold support.
- [x] Add quantized constant payloads if GGUF conversion needs constants rather than variables.
- [x] Add non-scalar block formats for GGUF/ggml quantized weights.

Completed notes:

- `QuantizationParams` now records affine scheme, granularity, block format, storage dtype, expressed dtype, axis, group size, scales, and zero points.
- CPU reference helpers support `QuantizeAffine` / `DequantizeAffine` for int8 and uint8 storage.
- Model format version was bumped so variable quantization metadata can be stored after each variable tensor while older model versions remain readable.
- `QuantizeNode` and `DequantizeNode` are validated, interpreted, serialized, and const-folded for scalar affine int8/uint8 paths.
- `QuantizedConstantNode` preserves raw storage payloads plus `QuantizationParams`, allowing GGUF-style weights to be represented as constants instead of only variables.
- `QuantizationScheme::Block` and GGML block format metadata now distinguish raw UInt8 payload shape from logical `expressedShape`; ModelIO version 5 persists that extra logical-shape metadata.

#### G1.3 CUDA Low Precision Kernels

Status: completed for capability reporting, CPU-bridge conversion coverage, fp16/bf16 GEMM attempts, and benchmark registration on 2026-05-17.

- [x] Add CUDA capability detection for fp16, bf16, fp8, and int8 storage / matmul policy.
- [x] Use cuBLAS/cuBLASLt for supported GEMM cases and explicit fallback paths for unsupported devices.
- [x] Add conversion coverage for f32 <-> fp16/bf16/fp8/int8 through the existing synchronous CPU bridge.
- [x] Add benchmark coverage per dtype for CUDA device MatMul, exposing native and fallback behavior.
- [ ] Add native CUDA conversion kernels for low-precision dtype conversion.
- [ ] Define and implement FP8/int8 native GEMM accumulation and output policy through cuBLASLt.

Completed notes:

- `CUDALowPrecisionCapabilities` reports compute capability, cuBLASLt build support, storage support, and native MatMul support for low-precision dtypes.
- `DeviceTraits<CUDA>::DoBinaryOp(MatMul)` still keeps the existing fp32/fp64 cuBLAS path and now attempts fp16/bf16 `cublasGemmEx` when the build and device report support.
- FP8 and int8 native GEMM are intentionally not marked as implemented yet. They need explicit accumulation/output semantics and cuBLASLt kernel policy before they are safe for production inference.
- CUDA dtype conversion remains synchronous through CPU reference conversion. Dedicated CUDA conversion kernels are a later performance task, not a correctness blocker.

### G2: llama.cpp / GGUF Compatibility

Purpose: support practical llama.cpp model import, lowering, validation,
execution, and later AOT compilation. This is more than a GGUF container reader:
it includes layout semantics, tokenizer/config handling, operator coverage,
runtime decode behavior, and golden validation against llama.cpp.

#### G2.1 GGUF Reader and Archive Import

Status: completed for standalone GGUF-to-LiteNN archive conversion on 2026-05-17.

- [x] Read GGUF metadata, tensor directory, tensor payloads, and ggml quantized block formats from `third_party/llama.cpp`.
- [x] Map GGUF tensors to LiteNN variables with stable names and shape validation.
- [x] Import tokenizer/config metadata needed by LLaMA-like models.
- [x] Emit LiteNN model files that can be loaded without linking llama.cpp at runtime.
- [x] Add real-GGUF layout tests that use non-symmetric matrices and payloads written in ggml `ne[]` order.
- [x] Define one explicit imported tensor layout policy: convert ggml `ne[]` order to LiteNN row-major semantic shape on import.
- [x] Apply the same layout policy to quantized block dequantization output, not only plain tensors.

Completed notes:

- `tools/gguf/litenn_gguf_convert` now converts GGUF files into LiteNN `.ltnn` archives through a small vendored ggml/gguf support library instead of linking the full llama.cpp runtime.
- GGUF K/V metadata is preserved in graph metadata with scalar/array widening into LiteNN's metadata value model, so tokenizer/config keys survive conversion.
- GGUF tensor names are preserved as graph variable names, and supported ggml block-quantized payloads are archived as LiteNN variables with block quantization metadata.
- Model format version 6 now persists graph variable names plus metadata, which is the storage substrate the GGUF converter relies on.
- The first converter intentionally emits a weight-archive graph with an empty forward subgraph; executable LLaMA-family graph lowering is tracked separately under G2.3.

Known risks from review:

- Completed on 2026-05-17: GGUF import now reverses ggml `ne[]` dimensions into LiteNN row-major semantic shape. The non-square token embedding fixture validates real ggml payload order, and GGML block dequantization now treats the last LiteNN dimension as the ggml row width.

#### G2.2 llama.cpp Operator Coverage

Audit source: `third_party/llama.cpp/ggml/include/ggml.h` `ggml_op`, `ggml_unary_op`, and `ggml_glu_op` enums, plus llama.cpp model graph builders. This list tracks operators that matter beyond the GGUF container format itself.

P0: required for common LLaMA-family decode/inference:

- [x] Embedding / row lookup: `GET_ROWS`, `GET_ROWS_BACK` lowering or a dedicated embedding node.
- [x] RMSNorm: `RMS_NORM` plus epsilon metadata; backward can remain deferred for inference-only import.
- [x] RoPE: `ROPE` with mode, base, scale, and position handling compatible with llama.cpp metadata.
- [x] Attention mask and softmax: `SOFT_MAX`, `DIAG_MASK_INF`, `TRI`, scale, and causal masking behavior.
- [x] Quantized weight MatMul: `MUL_MAT` over supported GGML block formats now lowers by dequantizing archive weights during import.
- [x] KV cache updates/views: `VIEW`, `CPY`, `SET`, `CONT`, `RESHAPE`, `PERMUTE`, `TRANSPOSE`, and slicing semantics, or a higher-level KV cache op.
- [x] MLP activation path: `SILU`, `GLU` / `SWIGLU`, `MUL`, `ADD`, `SCALE`, and broadcast helpers such as `REPEAT` / `ADD1`.
- [ ] Re-audit P0 completion against real decode graphs, because helper presence does not yet guarantee llama.cpp-equivalent layout, cache, RoPE, or axis semantics.

P1: needed by popular variants, MoE models, or efficient attention:

- [x] MoE routing: `MUL_MAT_ID`, `ADD_ID`, `TOP_K`, `ARGSORT`, and gather/scatter style row selection.
- [x] Existing additional normalization coverage: LayerNorm-style `NORM` and `L2_NORM` helper paths are now present alongside the already-completed `RMS_NORM` helper path.
- [x] `GROUP_NORM` helper coverage is now present for the remaining normalization gap in common GGML import paths.
- [x] `FLASH_ATTN_EXT` now lowers through a LiteNN attention-helper rewrite for the current single-head 2D path, including scale, causal/additive mask, softcap, and sinks semantics.
- [x] Activation coverage: `GELU`, `GELU_ERF`, `GELU_QUICK`, `SIGMOID`, `TANH`, `RELU`, `LEAKY_RELU`, `CLAMP`, `HARDSWISH`, `HARDSIGMOID`.
- [x] Existing shape/data movement coverage already includes `CONCAT`, `RESHAPE`, slicing/view patterns, `TRANSPOSE`, `GET_ROWS`, and broadcast-based rewrites used by current LLaMA lowering.
- [x] `PAD` and `CUMSUM` helper coverage closes the remaining shape/data movement gaps used by current llama.cpp-style lowering.
- [x] Add explicit `axis` semantics to `TOP_K` and `ARGSORT`, or document them as ggml-axis-0 compatibility helpers.
- [x] Clarify whether `MUL_MAT_ID` is a ggml-layout compatibility op or a normal LiteNN semantic op, then add reference tests with non-square expert/token dimensions.

P2: architecture-specific model families and multimodal support:

- [ ] SSM/Mamba style ops: `SSM_CONV`, `SSM_SCAN`.
- [ ] RWKV and gated attention: `RWKV_WKV6`, `RWKV_WKV7`, `GATED_LINEAR_ATTN`, `GATED_DELTA_NET`.
- [ ] Vision/multimodal ops: `CONV_1D/2D/3D` equivalents, `CONV_TRANSPOSE_*`, `IM2COL`, `POOL_*`, `UPSCALE`, `WIN_PART`, `WIN_UNPART`, `GET_REL_POS`, `ADD_REL_POS`.
- [ ] Loss/training/backward ops only if converted models need training or fine-tuning: `*_BACK`, `CROSS_ENTROPY_LOSS`, optimizer ops.

P3: unsupported in the first converter unless a real model requires them:

- [ ] Custom callback ops: `MAP_CUSTOM*`, `CUSTOM`.
- [ ] Optimizer-only graph ops: `OPT_STEP_ADAMW`, `OPT_STEP_SGD`.
- [ ] Rare numerical helpers with no first target model dependency: `SOLVE_TRI`, `OUT_PROD`, `TIMESTEP_EMBEDDING`.

#### G2.3 LLaMA Graph Lowering

Status: completed for CPU-runnable small LLaMA-family prefill and static-shape decode graphs on 2026-05-17. External llama.cpp parity fixtures remain tracked under broader compatibility validation.

Completed note: the end-to-end LLaMA-family forward graph now accepts token ids as input and lowers token embedding through `GetRowsNode` over `token_embd.weight^T`. Supported GGML block-quantized `MUL_MAT` weights are now dequantized during import/lowering, keeping the executable target graph on LiteNN's existing floating-point runtime path.

Layer and graph helper checklist:

- [x] Add RMSNorm helper and focused LayerTest coverage.
- [x] Add RoPE helper and focused LayerTest coverage.
- [x] Add causal masking helper and focused LayerTest coverage.
- [x] Add attention KV cache helper(s) for append/view/update and focused tests.
- [x] Add SwiGLU/MLP helper(s) covering gate/up/down projections and focused tests.

LLaMA graph lowering checklist:

- [x] Map GGUF hyperparameters needed for LLaMA-family graph construction.
- [x] Lower one decoder block from GGUF metadata and tensor names into LiteNN Graph.
- [x] Lower token embedding, final norm, and LM head around decoder blocks.
- [x] Emit a runnable forward graph for at least one common LLaMA-family architecture.
- [x] Add a CLI command or option that emits the lowered executable LLaMA graph, separate from raw GGUF archive import.
- [x] Make converter stage boundaries explicit: `import archive`, `lower causal lm`, `compile`, and `run`.
- [x] Fail with actionable diagnostics when tokenizer, layout, RoPE mode, KV-cache behavior, or unsupported ops block conversion.

CPU correctness checklist:

- [x] Add CPU interpreter smoke coverage for the first lowered decoder block.
- [x] Add CPU interpreter smoke coverage for the first fully lowered small LLaMA-family graph.
- [x] Keep the lowering path validated on CPU before relying on CUDA or AOT-only checks.
- [x] Add deterministic golden logits tests for tiny GGUF fixtures on the LiteNN CPU interpreter path.
- [x] Add tolerance policy by dtype and quantization format for llama.cpp parity tests.

Known risks from review:

- Completed on 2026-05-17: `litenn_gguf_convert` now has explicit `--import` and `--lower-llama` modes. The default two-argument form remains an archive import alias.
- Completed on 2026-05-17: LLaMA decode lowering now exposes static-shape KV-cache inputs and updated-cache outputs for autoregressive execution.
- Completed on 2026-05-17: G2.3 now has deterministic prefill/decode golden-logit tests, parity tolerance policy helpers, and actionable diagnostics for unsupported RoPE scale and decode position/cache mismatch.
- Current lowering intentionally accepts token ids rather than importing tokenizer runtime logic. Validation against external llama.cpp golden outputs is still tracked under G4 regression integration.

#### G2.4 RoPE and Position Semantics

Status: completed for default and linear-scaled 2D LLaMA RoPE on 2026-05-17. YaRN/LongRoPE metadata is preserved and fails with explicit diagnostics until the full numerical variant is implemented.

- [x] Provide a basic RoPE helper with `rope.freq_base`.
- [x] Parse `rope.freq_base` metadata.
- [x] Parse and validate core llama.cpp RoPE metadata for dimension count and frequency scale.
- [x] Parse and preserve advanced RoPE scaling type and model-family-specific parameters.
- [x] Add per-token position input or explicit position-offset input for decode.
- [x] Support non-default RoPE variants used by common GGUF models, including context-extension/scaling modes where applicable.
- [x] Add golden tests for RoPE on non-zero offsets and non-default scaling.

Known risks from review:

- Completed on 2026-05-17: LLaMA hyperparameter parsing now reads `rope.dimension_count` and `rope.freq_scale`; lowering rotates the configured RoPE prefix and reports unsupported scaling variants explicitly instead of silently generating wrong math.
- Completed on 2026-05-17: fixed-length LLaMA lowering now accepts an explicit `positionOffset`, passing it through RoPE and causal masking for non-zero-position segment prefill.
- Completed on 2026-05-17: RoPE helper now accepts `frequencyScale`; GGUF parsing handles `rope.scaling.type`, `rope.scaling.factor`, original context length, finetune flags, and YaRN/LongRoPE metadata. LLaMA lowering executes `none` and `linear` scaling, including partial `rope.dimension_count` prefix rotation with an unrotated tail.
- Current RoPE helper still assumes a narrow 2D sequence-by-feature input with llama.cpp pair layout. YaRN/LongRoPE numerical formulas are intentionally not approximated; conversion preserves metadata and emits actionable diagnostics.

#### G2.5 Attention Decode and KV Cache Semantics

Status: completed for static-shape interpreter decode semantics on 2026-05-17. CUDA/AOT cache-buffer ABI remains tracked under G3.

- [x] Provide attention helper coverage for single-head 2D path with causal/additive masks, scale, softcap, and sinks.
- [x] Add KV cache helper scaffolding for append/view/update scenarios.
- [x] Support rectangular causal attention where `queryLength != keyLength`.
- [x] Add a past-length or absolute-position-aware causal mask rule.
- [x] Expose cache inputs and outputs in the lowered LLaMA graph for decode.
- [x] Validate prefill-plus-decode logits against a deterministic golden fixture.
- [x] Decide how CUDA/AOT backends represent and update KV-cache buffers without hidden interpreter-only state.

Known risks from review:

- Completed on 2026-05-17: `CausalMask` and `FlashAttnExt` now support rectangular causal score matrices with explicit query/key position offsets.
- Completed on 2026-05-17: `LowerLLaMACausalLMDecode` uses explicit past key/value cache inputs, appends current rotated keys/values, and returns updated cache tensors alongside logits.
- Completed on 2026-05-17: prefill-then-decode is now covered by a deterministic fixture that compares the second decode logits against the equivalent full-prefill logits, validating cache append, RoPE offset, and rectangular causal mask interaction.
- CUDA/AOT backends should use the same explicit cache ABI: `past_key_N`/`past_value_N` inputs and `updated_key_N`/`updated_value_N` outputs. In-place or paged cache mutation is a later optimization, not hidden interpreter state.

#### G2.6 Axis, Shape, and Layout Compatibility

Status: open hardening item.

- [x] Define global conventions for LiteNN semantic shape order versus ggml `ne[]` order.
- [x] Add import-time conversion utilities for ggml tensor layouts if LiteNN keeps row-major semantic tensors.
- [x] Add explicit axis fields to ops that currently assume axis 0 but may be used as general LiteNN layers.
- [ ] Add tests using non-square dimensions for `TopK`, `Argsort`, `GetRows`, `MulMatId`, transposition, and imported linear weights.
- [ ] Document compatibility-only operators separately from general-purpose LiteNN operators.

Known risks from review:

- Completed on 2026-05-17: `ArgsortNode` now carries an explicit axis through validation, dumping, serialization, pass cloning, and interpreter execution. `TopK` exposes the same axis and has last-axis coverage.
- Completed on 2026-05-17: `MulMatId` is now documented as a ggml-compatible helper with ggml shape order and Float32 accumulator/output semantics. Existing non-square tests cover the interpreter path.

### G3: AOT LLM Artifacts

Purpose: compile converted models to embeddable CPU/CUDA artifacts while
preserving rodata/instruction separation and metadata needed by static/shared
library loading.

- [ ] Compile converted models to CPU/CUDA AOT artifacts with rodata/instruction separation.
- [ ] Preserve quantized and low-precision metadata in compiled signatures.
- [ ] Add runtime loader examples for static/shared library embedding.
- [ ] Support CUDA backend selection for lowered LLaMA graphs once G2 decode semantics are stable.
- [ ] Validate that AOT artifacts can consume externally provided weights/cache buffers without interpreter-only assumptions.

### G4: Validation and Benchmarks

Purpose: make correctness and performance claims traceable across CPU
single-thread, CPU multithread, CUDA, AOT, PyTorch, and llama.cpp baselines.

- [ ] Add golden tests against llama.cpp or PyTorch for small fixtures.
- [ ] Track CPU single-thread, CPU multithread, CUDA, and AOT performance in one horizontal benchmark table.
- [ ] Add numerical tolerance policy per dtype and quantization format.
- [ ] Add real GGUF fixture coverage for layout, RoPE, prefill, decode, and quantized weights.
- [ ] Keep `bench.py` execution notes explicit for Windows/Python 3.11 environments.
- [x] Add a self-contained GGUF conversion example that creates a tiny GGUF fixture, imports it, lowers it, saves `.ltnn` artifacts, and runs CPU prefill/decode.

## Hidden Requirements

- Low precision support is not only an enum addition. Tensor allocation, CPU conversion, serialization, graph validation, compiler type lowering, compiled artifact metadata, tests, and debugging output all need one source of dtype truth.
- FP8 and int quantization need explicit storage semantics. Some paths should treat them as scalar element dtypes, while GGUF quantized weights are usually block formats that need separate quantized tensor metadata.
- GGUF conversion implies model format stability, tensor-name mapping, tokenizer/config import, graph construction helpers for transformer blocks, and enough compiler/runtime ops for LLM inference.
- CUDA support needs capability detection and fallback rules. FP16/BF16/FP8 kernels depend on device architecture, CUDA version, and cuBLAS/cuBLASLt availability.
- AOT support must preserve dtype metadata in rodata/instruction-loaded modules so static/shared library embedding can validate buffers before execution.
- GGUF conversion is also an operator-coverage project. llama.cpp can express more graph ops than LiteNN currently owns, so the converter must either lower them to existing primitives, add LiteNN ops, or reject models with actionable diagnostics.
- llama.cpp compatibility is a semantic compatibility target, not only an operator-count target. Shape layout, axis order, RoPE variants, cache mutation, tokenizer/config metadata, and golden logits must be validated together.

## Date Notes

### 2026-05-17

- Added the low precision, quantization, and GGUF import roadmap.
- Completed scalar dtype storage/reference paths.
- Completed graph/runtime/storage metadata paths for affine and block quantization.
- Completed CUDA low-precision capability reporting, CPU-bridge conversion coverage, fp16/bf16 GEMM attempts, and dtype benchmark registration.
- Completed standalone GGUF-to-LiteNN archive conversion.
- Completed the first static CPU-runnable LLaMA-family forward graph path.
- Reviewed the llama.cpp operator additions and recorded hardening work for real GGUF layout, decode/KV-cache, RoPE metadata, axis semantics, and CLI stage separation.
- Added the self-contained `example/gguf` conversion example, optional LLaMA lowering `positionOffset`, and static-shape decode graph lowering with explicit KV-cache inputs/outputs.
