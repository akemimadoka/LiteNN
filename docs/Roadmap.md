# LiteNN Roadmap

This document is the planning entry for LiteNN. It is organized as a goal tree.
Date notes are kept at the end as historical hints, while the checklists below
are the source of truth for current planning and completion status.

## Production Refactor Focus

The current active direction is documented in `docs/ProductionRefactorPlan.md`. Near-term work should prefer stabilizing
the vNext production profile over adding more isolated feature breadth.

Current production profile target:

- vNext model packages and separated compiled artifacts are the durable deployment formats.
- `ExecutablePlan` plus runtime schedule metadata is the stable runtime/compiler boundary.
- CPU interpreter is the correctness/debug reference; CPU AOT is the first production package/load path.
- CUDA and Vulkan are optional native backends with explicit capability checks and visible fallback policy.
- Importers should produce manifests/packages with diagnostics rather than leaking raw graph construction details.

Current demotion/removal policy:

- Do not reintroduce pre-vNext graph archives or raw `Graph&` public runtime/compiler entry points.
- Do not allow hidden CPU fallback inside GPU paths.
- Do not block vNext production stabilization on full SDXL generation, full llama.cpp operator parity, full AOT training,
  or broad mobile Vulkan device coverage.
- Do not add fake byte-addressable int4/fp4 `DataType` values; packed 4-bit support should start from storage and
  quantization metadata.

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

Status: completed for runtime/device coverage plus formal CUDA AOT low-precision cast, MatMul, MatMulBias epilogue, linear-chain lowering, and dtype benchmark registration on 2026-05-20.

- [x] Add CUDA capability detection for fp16, bf16, fp8, and int8 storage / matmul policy.
- [x] Use cuBLAS/cuBLASLt for supported GEMM cases and explicit fallback paths for unsupported devices.
- [x] Add conversion coverage for f32 <-> fp16/bf16/fp8/int8 through the existing synchronous CPU bridge.
- [x] Add benchmark coverage per dtype for CUDA device MatMul, exposing native and fallback behavior.
- [x] Add native CUDA conversion kernels for low-precision dtype conversion.
- [x] Define and implement FP8/int8 native GEMM accumulation and output policy through cuBLASLt.
- [x] Add formal CUDA AOT cast kernels for supported low-precision scalar dtype pairs, including bf16/fp8 payload generation.
- [x] Add formal CUDA AOT native low-precision MatMul payloads and dtype-aware MatMulBias / MatMulBiasAddReLU epilogues.
- [x] Lower low-precision fused linear chains through the formal CUDA AOT path and expose benchmark coverage for native MatMul per dtype.

Completed notes:

- `CUDALowPrecisionCapabilities` reports compute capability, cuBLASLt build support, storage support, and native MatMul support for low-precision dtypes.
- `DeviceTraits<CUDA>::DoBinaryOp(MatMul)` still keeps the existing fp32/fp64 cuBLAS path and now attempts fp16/bf16 `cublasGemmEx` when the build and device report support.
- `DeviceTraits<CUDA>::CopyFromCPU`, `CopyToCPU`, and `ConvertTo` now use an NVRTC-compiled generic CUDA cast kernel when the CUDA driver and NVRTC are available, and otherwise retain the existing CPU bridge fallback.
- `TryCUBLASLtMatMul` now defines explicit native policy for FP8 and int8/uint8: accumulate into Float32 or Int32 scratch output with cuBLASLt, then convert back to the public MatMul result dtype through the native CUDA conversion path.
- `tests/CUDADeviceTest.cpp` now covers low-precision conversion entry points and validates supported FP8/int8 native MatMul against a high-precision accumulate-then-quantize CPU reference.
- `Compiler<CUDA>::CompileArtifact` now emits formal CUDA native payloads for scalar cast kernels, pure low-precision MatMul library calls, and dtype-aware MatMulBias / MatMulBiasAddReLU epilogues used by low-precision linear chains.
- `tests/CompiledModuleCUDATest.cpp` now covers compile-time payload exposure for bf16/fp8 cast plus runtime execution of native bf16 cast, low-precision MatMulBias payloads, and low-precision linear-chain payloads.
- FP8 CUDA-native cast payload generation is available in the formal AOT path, but runtime execution is now gated by actual device native-conversion support; the current NVVM/PTX lowering requires SM90+ for FP8 convert instructions.
- `benchmark/bench.cpp` now registers both `CUDADeviceMatMul/<dtype>` and formal `CUDANativeMatMul/<dtype>` benchmark families, so low-precision CUDA behavior is visible at both the device helper and formal AOT layers.

#### G1.4 Int4 / FP4 and Native Quantized Execution

Status: in progress. LiteNN currently has scalar fp16/bf16/fp8/int8/uint8 dtypes plus GGML-style 4-bit block-quantized
metadata, but does not yet expose scalar int4/fp4 element dtypes or native 4-bit execution kernels.

- [x] Add packed scalar storage descriptors for int4/uint4 and fp4 variants without pretending they are byte-addressable
      `DataType` values; decide whether they live as `QuantizationParams` storage kinds, `TensorStorageRef` formats, or a
      separate packed dtype set. Completed by `PackedNibbleFormat` metadata on `QuantizationParams`.
- [x] Define fp4 variant identifiers explicitly, starting with common e2m1/e3m0-style formats required by target
      hardware/importers.
- [x] Define fp4 NaN/Inf/subnormal/rounding policy and implement numeric conversion golden tests.
      Current CPU reference policy is finite-only, round-to-nearest with ties to the larger magnitude,
      saturation for overflow/non-finite inputs, E2M1 subnormal 0.5, and E3M0 power-of-two grid.
- [x] Extend quantization metadata with packed nibble layout, signedness, block scale layout, and byte-order rules so
      GGUF, safetensors-like extensions, and future vendor formats can share one contract.
- [x] Add CPU reference pack/unpack, dequantize, and optional quantize helpers for int4/uint4/fp4 with golden tests.
      `PackInteger4`, `UnpackInteger4`, `PackFloat4`, and `DequantizePackedNibble` cover the current CPU reference path.
- [x] Add graph/runtime conversion nodes or typed lowering rules that can materialize int4/fp4/block-quantized weights
      into supported compute dtypes when a backend has no native 4-bit kernel. The existing `DequantizeNode` now
      materializes packed-nibble block quantization in the interpreter and ConstFold path, and CPU AOT now has smoke
      coverage for compiling ConstFolded packed-nibble dequantized constants. CPU AOT also lowers dynamic affine
      per-tensor/per-axis/grouped `DequantizeNode`; grouped lowering is a correctness-first full-parameter expansion,
      not a native quantized kernel. CPU AOT lowers dynamic affine `QuantizeNode` with exact round-and-clamp semantics
      for int8/uint8 storage; packed 4-bit quantization remains on the ConstFold/native-kernel path.
- [x] Add the first native quantized MatMul/Linear path for CPU direct execution over affine and packed-nibble weights.
      `EvalQuantizedMatMul` and `EvalQuantizedLinear` compute directly from quantized storage and are covered by parity
      tests against dequantize-plus-float execution.
- [x] Add a reusable CPU prepared-weight path for quantized Linear workloads.
      `PrepareQuantizedLinearWeight` materializes affine or packed-nibble weights once into a reusable float32 payload,
      and `EvalPreparedQuantizedLinear` avoids repeated per-run weight decode while preserving parity with the direct
      quantized reference path.
- [ ] Add accelerated native quantized MatMul/Linear paths in priority order: CUDA/cuBLASLt or custom kernels, Vulkan
      shader path, then CPU AOT lowering.
      Direct dynamic `QuantizeNode` / `DequantizeNode` MLIR lowering remains part of this backend work; the current
      production route for constant packed quantized weights is ConstFold-before-compile, GraphToMLIR now lowers dynamic
      affine per-tensor/per-axis/grouped `QuantizeNode` and `DequantizeNode`, and diagnostics point users to
      `ConstFoldPass` for packed cases that are not yet dynamically lowered.
- [x] Preserve packed 4-bit storage and quantization metadata across vNext packages, separated rodata/weights, compiled
      signatures, and dump/diagnostic output.
- [x] Add benchmark rows and parity tolerances for int4/fp4/block-quantized Linear/MLP/LLM projection workloads.
      `litenn_bench` now registers `NativeQuantizedLinear/<format>` and
      `PreparedQuantizedLinear/<format>` / `DequantizedQuantizedLinearReference/<format>` rows for affine int8, packed
      int4, and packed FP4E2M1 Linear/MLP workloads. Each row reports `max_abs_error` against the opposite execution
      path so benchmark output carries the parity signal needed before choosing CUDA/Vulkan native quantized kernels.

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
- Completed on 2026-05-17: real tiny LLaMA decode graphs are now re-audited across both the CPU interpreter path and CPU `CompileArtifact().Load()` artifact path, covering token-id embedding lookup, explicit KV-cache ABI, RoPE offset handling, and decode-logit parity against the interpreter.

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
- [x] Re-audit P0 completion against real decode graphs, because helper presence does not yet guarantee llama.cpp-equivalent layout, cache, RoPE, or axis semantics.

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

- [x] SSM/Mamba style ops: `SSM_CONV`, `SSM_SCAN`.
- [x] RWKV and gated attention substrate: `RWKVWKVNode` provides the current token-recurrence baseline; exact
      `RWKV_WKV6`, `RWKV_WKV7`, `GATED_LINEAR_ATTN`, and `GATED_DELTA_NET` model-signature mappings are deferred
      to the long-term compatibility queue below.
- [x] Vision/multimodal ops: `CONV_1D/2D/3D` equivalents, `CONV_TRANSPOSE_*`, `IM2COL`, `POOL_*`, `UPSCALE`, `WIN_PART`, `WIN_UNPART`, `GET_REL_POS`, `ADD_REL_POS`.
- [x] Loss/training/backward seed ops: `CROSS_ENTROPY_LOSS` and `CROSS_ENTROPY_LOSS_BACK`.
      （已完成：新增 `CrossEntropyLossNode` 与 `CrossEntropyLossBackwardNode`，对齐 ggml soft-label 语义；
      支持 Float32 CPU reference、解释器、AutogradPass logits 梯度生成、验证、序列化 v19、dump、
      常量折叠、pass clone/dependency、Layer 包装、MLIR 显式 stub 与 `LossNodeTest` 覆盖。）

P3: unsupported in the first converter unless a real model requires them:

- [x] Custom callback ops: `MAP_CUSTOM*`, `CUSTOM` are intentionally not accepted by the portable converter;
      they remain a long-term host-callback extension, because serializable LiteNN graphs cannot safely preserve
      arbitrary llama.cpp callback pointers.
- [x] Optimizer-only graph ops: `OPT_STEP_ADAMW`, `OPT_STEP_SGD`.
      （已完成：新增 `SGDStepNode` 与 `AdamWStepNode`，优化器状态显式作为输入/输出流动；
      支持 Float32 CPU reference、解释器、多输出结果、验证、序列化 v18、dump、pass clone/dependency、
      MLIR 显式 stub 与 `OptimizerGraphOpTest` 覆盖。）
- [x] Rare numerical helpers with no first target model dependency: `SOLVE_TRI`, `OUT_PROD`, `TIMESTEP_EMBEDDING`.
      （已完成：新增 `OutProdNode`、`TimestepEmbeddingNode`、`SolveTriNode` 的 row-major CPU reference，
      包含 Layer 包装、验证、解释器、序列化 v17、常量折叠、pass clone/dependency、dump/validator、
      MLIR 显式 stub 与 `RareNumericalNodeTest` 覆盖。`SolveTriNode` 当前对齐 ggml 已支持的
      lower/non-unit left solve；其他变体留给真实模型需求驱动。）

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

Status: completed for axis/layout hardening coverage and compatibility-only operator documentation on 2026-05-17.

- [x] Define global conventions for LiteNN semantic shape order versus ggml `ne[]` order.
- [x] Add import-time conversion utilities for ggml tensor layouts if LiteNN keeps row-major semantic tensors.
- [x] Add explicit axis fields to ops that currently assume axis 0 but may be used as general LiteNN layers.
- [x] Add tests using non-square dimensions for `TopK`, `Argsort`, `GetRows`, `MulMatId`, transposition, and imported linear weights.
- [x] Document compatibility-only operators separately from general-purpose LiteNN operators.

Known risks from review:

- Completed on 2026-05-17: `ArgsortNode` now carries an explicit axis through validation, dumping, serialization, pass cloning, and interpreter execution. `TopK` exposes the same axis and has last-axis coverage.
- Completed on 2026-05-17: `MulMatId` is now documented as a ggml-compatible helper with ggml shape order and Float32 accumulator/output semantics. Existing non-square tests cover the interpreter path.
- Completed on 2026-05-17: focused tests now explicitly cover non-square `GetRows`, `Argsort`, `TopK`, `MulMatId`, 2D `Transpose`, and imported GGUF linear-weight transposition into LiteNN layout.
- Completed on 2026-05-17: [GGMLCompatibility.md](GGMLCompatibility.md) separates compatibility-only ggml surfaces such as `AddId` and `MulMatId` from general-purpose LiteNN operators used by GGUF lowering.

### G3: AOT LLM Artifacts

Purpose: compile converted models to embeddable CPU/CUDA artifacts while
preserving rodata/instruction separation and metadata needed by static/shared
library loading.

Status: completed for the current embeddable CPU/CUDA artifact tranche. On 2026-05-17, LiteNN's CPU AOT path covered tiny token-id LLaMA-family artifacts end-to-end for static decode graphs and for a minimal single-token full-graph prefill run; two-token full-graph prefill is additionally covered through artifact compile/load smoke. On 2026-05-20, the GGUF conversion CLI gained CPU/CUDA carrier-object export for converted `.ltnn` graphs, and compiled signatures preserve low-precision dtype plus quantization metadata. Broader CUDA LLaMA artifact parity is deferred to the long-term validation queue.

- [x] Compile converted models to CPU/CUDA AOT artifacts with rodata/instruction separation.
- [x] Preserve quantized and low-precision metadata in compiled signatures.
- [x] Add runtime loader examples for static/shared library embedding.
- [x] Support CUDA backend selection for lowered LLaMA graphs once G2 decode semantics are stable.
      （已完成：`litenn_gguf_convert --compile-cuda <input.ltnn> <output.o> [symbol-prefix]`
      直接加载 lowered `.ltnn` 图并通过 `Compiler<CUDA>::CompileArtifact` 生成 CUDA backend carrier object；
      README 与 smoke 覆盖已同步到 G3 artifact 流程。）
- [x] Validate that AOT artifacts can consume externally provided weights/cache buffers without interpreter-only assumptions.

Known progress from review:

- Completed on 2026-05-17: `GraphToMLIR` now lowers `GetRowsNode`, unblocking token-id embedding lookup for CPU AOT compilation of lowered LLaMA graphs.
- Completed on 2026-05-17: tiny LLaMA full-graph CPU artifacts now have compile/load smoke coverage for 2-token prefill, and 1-token prefill is validated end-to-end against the CPU interpreter after `CompileArtifact().Load()`.
- Completed on 2026-05-17: tiny static decode graphs now execute through CPU `CompileArtifact().Load()` with explicit `past_key_N`/`past_value_N` inputs and updated-cache outputs, matching the interpreter without hidden interpreter-only cache state.
- Completed on 2026-05-20: `example/carrier` demonstrates object carrier generation plus static/shared-library style loading through rodata/instruction addresses.
- Completed on 2026-05-20: `litenn_gguf_convert --compile-cpu/--compile-cuda` loads converted `.ltnn` graphs and writes carrier objects that preserve rodata/instruction separation.
- Completed on 2026-05-20: compiled module rodata v4 persists `CompiledTensorSpec::quantization`, preserving quantized output signatures across artifact copy/load and dump output while continuing to expose scalar dtype for low-precision signatures.
- Completed on 2026-05-20: CPU artifact tests cover external input/output buffers and explicit KV-cache inputs/outputs; broader CUDA artifact parity for lowered LLaMA graphs is deferred until real external fixtures are available.

### G4: Validation and Benchmarks

Purpose: make correctness and performance claims traceable across CPU
single-thread, CPU multithread, CUDA, AOT, PyTorch, and llama.cpp baselines.

- [x] Add golden tests against llama.cpp or PyTorch for small fixtures.
- [x] Track CPU single-thread, CPU multithread, CUDA, and AOT performance in one horizontal benchmark table.
- [x] Add numerical tolerance policy per dtype and quantization format.
- [x] Add real GGUF fixture coverage for layout, RoPE, prefill, decode, and quantized weights.
- [x] Keep `bench.py` execution notes explicit for Windows/Python 3.11 environments.
- [x] Add a self-contained GGUF conversion example that creates a tiny GGUF fixture, imports it, lowers it, saves `.ltnn` artifacts, and runs CPU prefill/decode.

Completed notes:

- `benchmark/results/backend_ggml_pytorch_comparison_2026-05-19.md` records a horizontal CPU/AOT/CUDA/PyTorch/ggml comparison table.
- `benchmark/bench.py` documents Windows execution through `python311`, including CUDA and CPU-thread variants.
- `LLaMAParityTolerance` defines dtype/quantization-aware tolerance policy for LLaMA logits validation.
- `GGUFImporterTest` covers non-square GGUF layout, quantized block payload handling, RoPE/position metadata, prefill/decode fixtures, and CPU artifact parity for tiny LLaMA graphs.
- Completed on 2026-05-20: `PyTorchGoldenTest` adds a small `torch.relu(x @ w + b)` fixture against PyTorch golden values for both the CPU interpreter and CPU AOT artifact path.
- Deferred long-term item: broader external llama.cpp golden-output comparison for LLaMA-family fixtures.

### G5: Core Node Expansion

Purpose: 补全 LiteNN 作为通用神经网络框架仍缺失的基础原语 Node，使 G2.2 P2
所列 ggml 算子能以"Node 组合 + Layer 包装"的健康方式落地，而非以兼容专用的
catch-all 桩节点伪装"已支持"。同时也把一部分明显应当作为 Node 的常用热点路径
（Softmax、归一化、批量 MatMul 等）从纯 Layer 实现升级为 Node，便于后端原生
优化与 MLIR 端到端编译。

设计原则：

- 任何 P2/兼容算子若可由通用 Node 组合实现，就以 Layer 实现，不新增 Node。
- 真正需要新增 Node 的两类情况：
  1. 现有原语在表达上**无法**干净表达（如任意维度转置、关联扫描、Im2Col 窗口、稀疏 gather/scatter）；
  2. 用 Layer 可以表达但**性能/数值/可微性收益足够高**（如 Softmax、归一化、批量 MatMul）。
- 新增 Node 必须一次完成完整的 14 个 touch points，否则视为污染（参考 G5.0）。
- 仅在 ggml 内部出现、无任何通用 NN 含义的工件（如 `WIN_PART`、`GET_REL_POS`），
  以 Layer 形式落地在 ggml 兼容路径，并明确标注为 compatibility-only。

#### G5.0 Add-a-Node Touch Point Checklist

每新增一个 Node 必须同时完成以下 14 项；缺失任何一项即视为"半成品 Node"，应拒绝合入。

本 checklist 在 `PermuteNode` 落地时被首次端到端走通，作为 reference implementation 固化；
后续每个 G5 Node 都必须复用同一流程。每条目下注明 PermuteNode 中对应位置以便对照。

- [x] `src/LiteNN/Graph.h`: struct 定义并加入 `Node` 命名空间（自动进入 `NodeVariant`）。
      Ref: `PermuteNode { NodeOutput input; std::vector<std::size_t> permutation; }`。
- [x] `src/LiteNN/Debug/Dump.cpp`: 格式化文本与 node-kind 名称。
- [x] `src/LiteNN/Validation/GraphValidator.h`: name 映射、输入引用校验、输出 shape/dtype 校验。
      Ref: 校验 permutation 是 input rank 的合法排列，`output.shape[d] == input.shape[permutation[d]]`。
- [x] `src/LiteNN/Serialization/ModelIO.h`: 写入/读取分支；如新字段不兼容旧版本需 bump 文件版本。
      Ref: `kModelVersion` bump 至 11；新增 `NodeKind::Permute`。
- [x] `src/LiteNN/Pass/ConstFoldPass.h`: clone + 可选常量折叠求值（3 处分支）。
      Ref: clone / markInput / Eval 三处 + `EvalPermute` helper。
- [x] `src/LiteNN/Pass/ForwardOnlyPass.h`: clone 分支（注意用 `remapOutput` 而非 `remap`）。
- [x] `src/LiteNN/Pass/FusionPass.h`: clone + 候选模式分类（2 处分支）。
- [x] `src/LiteNN/Pass/InlinePass.h`: clone 分支。
- [x] `src/LiteNN/Pass/AutogradPass.h`: 输入依赖、clone、反向梯度（3 处分支）。
      Ref: `EmitPermuteGrad` 用逆置换 `inverse[permutation[d]] = d`。
- [x] `src/LiteNN/Runtime/Interpreter.h`: `Execute` 重载，至少覆盖 CPU 参考实现。
- [x] `src/LiteNN/Compiler/Translation/GraphToMLIR.cpp`(+ `LiteNNDialect.td` 若需要新 op): MLIR 下沉。
      Ref: PermuteNode 暂为显式 stub（抛 "not supported"），复用解释器路径——
      允许 native lowering 滞后于 Node 主体，但必须显式 stub，不得静默通过。
- [x] `src/LiteNN/Device/*` (CUDA 等): 后端原生路径或显式 fallback。
      Ref: CPU 写 native kernel；CUDA 非热点走 host fallback
      （`MakeHostBuffer + CopyToCPU + CPU traits + CopyFromCPU`），热点 Node 必须写 native kernel。
- [x] `src/LiteNN/Layer/`: 至少一个 Layer 包装，确保上层调用方走 Layer 而非裸 Node。
      Ref: `Layer/Permute.h` (`AddPermute` / `BuildPermute` / `AddTranspose`)。
- [x] `tests/`: 形状/dtype/数值正确性测试；若可微，需 AutogradPass 单测；若可编译，需 AOT smoke。
      Ref: `tests/PermuteNodeTest.cpp` 6 用例覆盖 forward (2D/3D/identity) + Layer 包装 + backward + ConstFold。

当前 G5.4 audit 结果（2026-05-18）：

- [x] `Im2ColNode`、`Conv2DNode`、`ConvTranspose2DNode`、`Pool2DNode`、`UpsampleNode`
      均已对照上述 14 个 touch points 收口：Graph/Debug/Validator/ModelIO/ConstFold/
      ForwardOnly/Fusion/Inline/Autograd/Runtime/GraphToMLIR/Device fallback/Layer/tests。
- [x] 对尚未实现 native lowering 或反向传播的路径，均采用显式 stub 或 host fallback，
      不允许静默宣称可编译/可微。

#### G5.1 Foundation Data Movement Nodes

- [x] `PermuteNode`: 任意维度转置，输入 + permutation；替代仅 2D 的 `UnaryOp::Transpose`，
      `UnaryOp::Transpose` 保留作为兼容别名直到调用方迁移完成。
      解锁：multi-head 批量注意力、`PERMUTE`/`TRANSPOSE` 通用语义、`WIN_PART`/`WIN_UNPART` 的 Layer 实现。
      （已完成：CPU 内核 + CUDA host fallback + Validator/Dump/ModelIO v11/Interpreter +
      全部 Pass clone + ConstFold + AutogradPass `EmitPermuteGrad`（逆置换） +
      `Layer/Permute.h`（含 `AddTranspose`）+ `PermuteNodeTest` 6 个用例全部通过；
      GraphToMLIR 现已直接展开为 `linalg.generic` 置换拷贝。）
- [x] `BroadcastToNode`: 显式将某些维度从 1 扩到指定大小（含插入前导单位维），
      不复制数据但暴露明确的 shape 推导；替代 BinaryOp 隐式广播作为前置步骤。
      解锁：`REPEAT`、`UPSCALE-Nearest` 的 Layer 实现。
      （已完成：Graph/Validator/Dump/ModelIO v12/Interpreter CPU reference + non-CPU host fallback
      through interpreter、ConstFold、ForwardOnly/Fusion/Inline/Autograd clone/dependency 接入、
      `Layer/BroadcastTo.h`、`DataMovementNodeTest` 覆盖前导维插入与 singleton dim 扩展；
      GraphToMLIR 现已直接展开为带 affine broadcast map 的 `linalg.generic`。Autograd
      differentiation 暂为显式 stub。）
- [x] `PadNode`: 任意轴前/后填充，模式 = constant/reflect/replicate，含填充值；
      取代 `Layer::AddPad` 走 zero-Constant + 多次 Concat 的低效路径。
      解锁：`PAD` 与 Conv 边界、注意力 mask 边界。
      （已完成：旧 `Layer::AddPad(input, paddings)` 保持兼容并改走 `PadNode`；
      新增 low/high/mode/value 接口；CPU reference 支持 constant/reflect/replicate；
      ModelIO v12、ConstFold、pass clone/dependency、`DataMovementNodeTest` 已覆盖三种模式；
      MLIR lowering 与 Autograd differentiation 暂为显式 stub。）
- [x] `GatherNode`: 任意轴 gather，indices 任意 rank；`GetRowsNode` 成为 axis=0 的特例。
      解锁：`GET_REL_POS`、稀疏 KV 访问、tokenwise routing 的真实路径。
      （已完成：CPU reference 支持任意 axis 与 indices 任意 rank，indices dtype 为 Int32/Int64；
      新增 `Layer/Gather.h`；ModelIO v12、ConstFold、pass clone/dependency、`DataMovementNodeTest`
      已覆盖 axis=1 gather；MLIR lowering 与 Autograd differentiation 暂为显式 stub。）
- [x] `ScatterNode`: 任意轴 scatter（加性/替换两种模式），对应 `GET_ROWS_BACK`、KV-cache `SET`。
      （已完成：CPU reference 支持 update/add 两种模式，重复 index 在 update 模式下后写覆盖、
      add 模式下累加；Bool add 被显式拒绝；新增 `Layer/Scatter.h`；
      ModelIO v12、ConstFold、pass clone/dependency、`DataMovementNodeTest` 已覆盖 update/add
      与序列化 roundtrip；MLIR lowering 与 Autograd differentiation 暂为显式 stub。）

#### G5.2 Scan and Recurrence Nodes

- [x] `ScanNode`: 沿指定轴的关联扫描（先支持 sum/max；预留 prod、logsumexp 接口）。
      取代 `Layer::AddCumsum` 的 O(N) 串行 slice+add+concat 路径。
      解锁：高效 `CUMSUM`、`SSM_SCAN` 的可向量化基线。
      （已完成：新增 `ScanOp` 与 `ScanNode`，CPU reference 支持 sum/max/prod/logsumexp；
      `Layer::AddCumsum` 改走 `ScanNode`；Validator/Dump/ModelIO v13/Interpreter/ConstFold/
      ForwardOnly/Fusion/Inline/Autograd clone-dependency 接入；`ScanHotPathNodeTest`
      覆盖解释器、常量折叠、序列化与 dump。MLIR lowering、CUDA kernel 与 Autograd
      differentiation 暂为显式 stub。）
- [x] `SSMScanNode`: Mamba 风格 selective scan（state, dt, A, B, C 五元 + 可选 D）。
      （已完成：先落地 rank-2 `[steps, channels]` 的最小 CPU reference，用广播参数表达
      `dt/A/B/C/D`，用于真实 Mamba/SSM 模型接入前的语义锚定；Validator/Dump/ModelIO v13/
      Interpreter/ConstFold/pass clone-dependency 接入。CUDA/MLIR 与真实模型签名细化留 TODO。）
- [x] `RWKVWKVNode`: 抽象 RWKV/GLA/GatedDeltaNet 的 token-by-token 递推核。
      （已完成：先落地 RWKV 风格 `key/value/receptance/timeDecay/timeFirst` 最小 CPU
      reference，time 参数支持广播；Validator/Dump/ModelIO v13/Interpreter/ConstFold/pass
      clone-dependency 接入。真实 RWKV/GLA/GatedDeltaNet 变体签名、CUDA/MLIR 与数值黄金样例
      留 TODO。）

#### G5.3 Hot-path Fused Nodes

- [x] `SoftmaxNode`: 沿指定轴的数值稳定 softmax；取代 `Layer::AddSoftmax` 的
      max-subtract + exp + reduce + divide 五次访存。CUDA/MLIR 可一次性下沉为 fused kernel。
      （已完成：`Layer::AddSoftmax` 改走 `SoftmaxNode`；CPU reference 使用 max-subtract
      稳定路径；Validator/Dump/ModelIO v13/Interpreter/ConstFold/pass clone-dependency 接入。
      GraphToMLIR 现已展开为 max-reduce/subtract/exp/sum-reduce/divide 组合 lowering；
      CUDA fused lowering 与 Autograd differentiation 暂为显式 stub。）
- [x] `NormalizationNode`: 统一 `LayerNorm` / `RMSNorm` / `GroupNorm` 三种归一化，
      参数 = mode + axis（或 group 数）+ eps + 可选 affine。取代三个 Layer 中各自展开
      的 reduce+broadcast+sqrt+divide 链。
      （已完成：新增 `NormalizationMode` 与 `Layer::AddNormalization`；`LayerNorm`、`RMSNorm`、
      `GroupNorm` 改走统一 Node；CPU reference 支持 affine 广播；Validator/Dump/ModelIO v13/
      Interpreter/ConstFold/pass clone-dependency 接入；GraphToMLIR 已将 `LayerNorm`/`RMSNorm`
      展开到现有 LiteNN dialect primitive，解除 LLaMA CPU AOT 路径阻塞。`GroupNorm` MLIR、
      CUDA fused lowering 与 Autograd differentiation 暂为显式 stub。）
- [x] `BatchMatMulNode`: 显式批量 MatMul，支持 >2D 输入与前导维广播；
      解锁多头注意力的真实表达，并使 cuBLAS strided batched GEMM 可被原生映射。
      在该 Node 落地后，`FlashAttnExt` 与 LLaMA decoder 由 2D 路径升级为 multi-head 批量路径。
      （已完成：新增 `Layer::AddBatchMatMul`；CPU reference 支持前导维 trailing broadcast；
      Validator/Dump/ModelIO v13/Interpreter/ConstFold/pass clone-dependency 接入。cuBLAS/
      cuBLASLt、MLIR lowering、FlashAttnExt/LLaMA decoder 升级与 Autograd differentiation
      留后续优化。）

#### G5.4 Convolution and Pooling Nodes

- [x] `Im2ColNode`: 通用滑窗展开（1D/2D/3D 由 spatial rank 参数决定）。
      `Conv*` 的 Layer 实现 = `Im2Col` + reshape + `MatMul`；首要目标是表达正确性而非性能。
      （已完成：新增 `Layer::AddIm2Col`；CPU reference 支持 channel-first 任意 spatial rank
      滑窗展开与零 padding；Validator/Dump/ModelIO v16/Interpreter/ConstFold/pass
      clone-dependency 接入。MLIR/CUDA lowering 与 Autograd differentiation 暂为显式 stub。）
- [x] `Conv2DNode`: 直接卷积原语，与 Im2Col-Layer 路径互为参考；CUDA 后端绑定 cuDNN。
      （已完成：新增 `Layer::AddConv2D`；CPU reference 支持 channel-first NCHW、stride、
      dilation、padding、可选 bias、grouped convolution；Validator/Dump/ModelIO v16/Interpreter/
      ConstFold/pass clone-dependency 接入。MLIR lowering、CUDA/cuDNN lowering 与 Autograd
      differentiation 暂为显式 stub。）
- [x] `ConvTranspose2DNode`: 转置卷积；表达上可借助 padded Conv，但 CUDA 上 cuDNN 有原生 kernel，
      因此独立成 Node。
      （已完成：新增 `Layer::AddConvTranspose2D`；CPU reference 支持 NCHW、stride、dilation、
      padding、output padding、可选 bias、grouped transposed convolution；Validator/Dump/
      ModelIO v16/Interpreter/ConstFold/pass clone-dependency 接入。MLIR lowering、CUDA/cuDNN
      lowering 与 Autograd differentiation 暂为显式 stub。）
- [x] `Pool2DNode`: max/average 池化，含 1D 退化形式（kernel 高度 = 1 即 Pool1D）。
      （已完成：新增 `Layer::AddMaxPool2D` / `Layer::AddAveragePool2D`；CPU reference 支持
      channel-first 4D 输入、padding、average `countIncludePad` 语义；Validator/Dump/ModelIO v16/
      Interpreter/ConstFold/pass clone-dependency 接入。MLIR/CUDA lowering 与 Autograd
      differentiation 暂为显式 stub。）
- [x] `UpsampleNode`: nearest/bilinear/bicubic 插值；nearest 可由 `BroadcastTo` + `Reshape` 表达
      并作为参考，但 bilinear/bicubic 的数值表达过于复杂，必须以 Node 形式存在。
      （已完成：新增 `Layer::AddUpsample` / `AddNearestUpsample2D` / `AddBilinearUpsample2D` /
      `AddBicubicUpsample2D`；CPU reference 支持 nearest、bilinear、bicubic 与 `alignCorners`；
      Validator/Dump/ModelIO v16/Interpreter/ConstFold/pass clone-dependency 接入。MLIR/CUDA
      lowering 与 Autograd differentiation 暂为显式 stub。）

#### G5.5 P2 Coverage Driven by New Nodes

Status: 已按 G5.1–G5.4 的实际 Node 覆盖标注；这里记录 P2 覆盖关系和仍需真实模型驱动的尾项。

- [x] G2.2 P2 `PAD`、`CUMSUM`：由 G5.1 / G5.2 的新 Node 直接驱动 Layer 重写。
      （已完成：`PadNode` 与 `ScanNode`；`Layer::AddPad` 和 `Layer::AddCumsum` 已迁移。）
- [x] G2.2 P2 `REPEAT`：由 `BroadcastTo + Reshape` 或专用 Layer 收口。
      （已完成：`Compatibility::GGML::AddRepeat` 通过 `Reshape + BroadcastTo + Reshape` 表达 ggml-style tiling，
      覆盖非 singleton 维度重复并有 `LayerRepeat` 测试。）
- [x] G2.2 P2 `WIN_PART`、`WIN_UNPART`、`GET_REL_POS`、`ADD_REL_POS`：以 ggml 兼容 Layer 落地，
      明确标注为 compatibility-only，不引入新 Node。
      （已完成：`Compatibility::GGML::AddWindowPartition` / `AddWindowUnpartition`、`AddGetRelativePosition`、
      `AddRelativePositionBias2D` 均复用现有 Pad/Permute/Reshape/Gather/Broadcast/Add 底座；
      `LayerWindow` 与 `LayerRelativePosition` 覆盖数值行为。）
- [x] G2.2 P2 `CONV_1D/2D/3D`、`CONV_TRANSPOSE_*`、`IM2COL`、`POOL_*`、`UPSCALE`：由 G5.4 驱动。
      （已完成：G5.4 Node substrate 已闭合，包含 generic `Im2ColNode`、NCHW `Conv2DNode`、
      `ConvTranspose2DNode`、`Pool2DNode` 和 `UpsampleNode` 的 CPU/reference 路径、Layer 包装、
      序列化、常量折叠、pass clone/dependency、dump/validator、解释器 host fallback、MLIR 显式 stub
      与测试覆盖。后续 `CONV_1D/3D` 专用 Layer、CUDA/cuDNN native kernel 与 MLIR lowering
      属于性能/后端扩展，不再是 G5.0 checklist 缺口。）
- [x] G2.2 P2 `SSM_SCAN`：由 G5.2 `SSMScanNode` 驱动。
      （已完成：`SSM_SCAN` 的最小 CPU reference Node，含验证、序列化、解释器、常量折叠与 pass 接入。）
- [x] G2.2 P2 `SSM_CONV`：由兼容 Layer 映射到 grouped `Conv2DNode`，真实 Mamba 端到端接入后再扩展参数布局。
      （已完成：`Compatibility::GGML::AddSSMConv` 将 ggml `[kernel - 1 + tokens, channels, batch]` buffer
      与 `[kernel, channels]` depthwise weight 改写为 grouped `Conv2DNode`，并由 `LayerSSMConv` 验证。）
- [x] G2.2 P2 RWKV-style recurrence substrate：由 G5.2 `RWKVWKVNode` 驱动。
      （已完成：`RWKVWKVNode` 最小 CPU reference，含验证、序列化、解释器、常量折叠与 pass 接入。）
- [x] G2.2 P2 `CROSS_ENTROPY_LOSS` / `CROSS_ENTROPY_LOSS_BACK`：由训练/损失 Node tranche 驱动。
      （已完成：`CrossEntropyLossNode` 输出平均 loss，`CrossEntropyLossBackwardNode` 输出 logits 梯度；
      AutogradPass 可由 loss 自动生成 backward helper，常量折叠、序列化、dump、解释器和 Layer 包装已覆盖。）
- [x] G2.2 P2 `RWKV_WKV6/7`、`GATED_LINEAR_ATTN`、`GATED_DELTA_NET` 真实变体映射：延期到长期队列，
      等真实模型权重布局、状态 ABI 和黄金样例一起推进，避免用当前 `RWKVWKVNode` 的简化签名假装兼容。
- [x] G2.2 P2 其他训练/反向相关算子：通用 `*_BACK` 家族延期到长期队列；当前已完成
      `CROSS_ENTROPY_LOSS(_BACK)` 和 optimizer-only `SGDStepNode` / `AdamWStepNode`。

#### G5.6 GraphToMLIR Lowering Priority Queue

Status: completed for the practical generic-lowering tranche on 2026-05-23. Shape/data-movement,
attention/classification, batched GEMM, irregular gather/pad, and cross-entropy loss helpers now
have CPU AOT coverage. Stateful recurrence and convolution/pooling families remain explicitly
deferred to backend-specific lowering projects rather than naive generic loops.

Priority rule: first implement nodes that unlock existing LLaMA/attention/classification AOT graphs
without requiring new dialect ABI; next handle batched GEMM and irregular data movement; leave
stateful recurrence, convolution families, and optimizer-only nodes to backend-specific projects.

- [x] P0 `PermuteNode`: direct `linalg.generic` lowering with affine inverse-permutation map.
- [x] P0 `BroadcastToNode`: direct `linalg.generic` lowering with affine broadcast map, including
      leading-rank insertion and singleton-dimension expansion.
- [x] P0 `SoftmaxNode`: numerically stable GraphToMLIR expansion through existing LiteNN dialect
      primitives (`Reduce(Max)`, `Subtract`, `Exp`, `Reduce(Sum)`, `Divide`) so the normal lowering
      pipeline can reach linalg/math ops without a dedicated softmax dialect op.
- [x] P0 AOT smoke: `CompiledModuleTest.CPUDataMovementSoftmaxArtifactMatchesInterpreter` covers
      `BroadcastTo -> Permute -> Softmax` through CPU artifact compile/load/run against interpreter output.
- [x] P1 `BatchMatMulNode`: direct batched `linalg.generic` lowering with NumPy-style leading-dim
      broadcasting and reduction over K. CPU/CUDA optimized strided-batched GEMM mapping remains a
      performance backend task.
- [x] P1 `GatherNode` / `PadNode`: direct `linalg.generic` lowering for arbitrary-axis gather and
      constant/reflect/replicate pad boundary handling.
- [x] P2 `CrossEntropyLossNode` / `CrossEntropyLossBackwardNode`: expanded through existing LiteNN
      dialect primitives (`Max`, `Exp`, `Log`, `Sum`, elementwise ops) so training loss helper graphs
      can compile through the normal CPU AOT pipeline.
- [x] P2 AOT smoke: `CompiledModuleTest.CPUBatchMatMulArtifactMatchesInterpreter`,
      `CPUGatherPadArtifactMatchesInterpreter`, `CPUCrossEntropyArtifactMatchesInterpreter`, and
      `CPUCrossEntropyBackwardArtifactMatchesInterpreter` compare CPU artifacts against interpreter output;
      `CPURankOneSoftmaxCrossEntropyArtifactMatchesInterpreter` covers the rank-1 reduce-to-`[1]`
      boundary used by vector classification heads.
- [x] P2 `Conv2DNode` / `ConvTranspose2DNode` / `Pool2DNode` / `UpsampleNode`: deferred to
      backend-native lowering plans because naive generic loops are unlikely to be production useful.
- [x] P2 `ScanNode` / `SSMScanNode` / `RWKVWKVNode` / `MulMatIdNode`: deferred until real model
      signatures, state ABI decisions, and golden-output validation are available.

### G6: Performance, Profiling, and Backend Optimization

Purpose: keep performance claims tied to repeatable profile/benchmark evidence across CPU AOT, CPU multithread,
CUDA native, CUDA Graph, PyTorch, and ggml.

Status: current profile/benchmark coverage is in place. CPU AOT multithread optimization is active again because
large-batch MLP measurements show the first sidecar helper path can lose to the packed MLIR fallback on some shapes.

- [x] Profile CPU AOT at instruction level and document whether generated code is scalar or vectorized.
- [x] Remove the misleading old CPU scalar "fast path" benchmark/compiler branch.
- [x] Add CPU AOT intra-op thread-policy benchmarks for default/T1/T16.
- [x] Implement a guarded CPU AOT intra-op path for large static f32 fused Linear/MLP chains.
- [x] Add a persistent CPU worker pool for the current AOT helper path.
- [x] Add CUDA native and CUDA Graph profile/benchmark notes, including comparison with PyTorch CUDA.
- [x] Persist raw CPU/CUDA profile and benchmark outputs under `benchmark/results/`.
- [ ] Productionize CPU AOT multithreading so T16 is not a regression on large-batch Linear/MLP shapes:
      tune helper microkernels, gating, thread grain policy, and eventually move the work into optimized MLIR/LLVM
      lowering or a production GEMM backend.
      - [x] Make the actual CPU parallel linear-chain compiler path report sidecar selection/rejection reasons when
            `CompilerOptions::enableCompileDiagnostics` is enabled, including selected fused layer count, total FLOPs,
            thread count, and thread/shape/FLOP gate rejection reasons. This keeps `litenn_profile` predictions
            auditable against emitted-object behavior while tuning T16 gates.
- [x] Add an explicit CPU AOT worker-affinity policy hook for multithread experiments:
      `CompilerOptions::cpuAOTAffinityPolicy` defaults to no pinning, while benchmark/profile entry points can opt into
      compact worker pinning for local measurements. Correctness follow-up completed on 2026-08-02: Compact now maps
      the caller and workers to distinct physical cores before SMT siblings, handles Windows processor groups, and
      respects Linux allowed CPU sets. This fixed a logical-index mapping that placed T8 on only four physical cores;
      real 14B cache-hit decode improved from the broken Compact result of `765.725` to `454.803 ms/token`, but remained
      slower than scheduler-managed `None` at `277.218 ms/token`. Keep no-pinning as the production decode default;
      bandwidth-aware cross-cache-domain scattering is separate nonblocking work.
- [x] Add an explicit bandwidth-oriented CPU AOT `Spread` affinity policy:
      all CPU helper decoders plus compiler, GGUF CLI, Qwen smoke, benchmark environment, and thread-matrix surfaces
      accept `none|compact|spread`. Spread interleaves the lower and upper halves of the physical-core-first topology
      before SMT siblings without changing Compact semantics, and both pinned policies pass the parallel-chain
      correctness test. Real 14B T8 cache-hit samples were Spread `240.774/260.167/285.250 ms/token` (median
      `260.167`), None `257.998/260.166`, and Compact `300.594`, so None remains the default. Accurate LLC/NUMA-domain
      enumeration and stable automatic policy selection remain nonblocking work rather than being inferred from this
      enumeration-order heuristic.
- [x] Extend `litenn_profile` with first-class CPU AOT instruction stats instead of relying on manual objdump report synthesis.
- [x] Extend `litenn_profile` with CPU AOT parallel-helper selection counters.
      The profile report now prints layer shapes, estimated FLOPs, selected helper thread counts, sidecar-vs-MLIR gate
      reasons, and an emitted-object symbol check for `litenn_cpu_matmul_bias_relu_parallel_f32`.
- [x] Extend `litenn_profile` with CUDA launch breakdowns.
- [ ] Add a whole-process performance profile bundle tool for flame graphs and waterfall timelines.
      Current `litenn_profile` can be used directly for deterministic backend evidence such as emitted-object
      disassembly, CPU AOT instruction statistics, CPU helper-selection counters, CUDA launch breakdowns, and Vulkan
      profile rows. It is not yet a sampling profiler and does not produce a unified timeline across import, lowering,
      compile, load, execution, transfer, and synchronization phases.
      - [x] Add the first `benchmark/profile_bundle.py` command that wraps `litenn_profile` or an arbitrary smoke
            command, captures stdout/stderr, writes `manifest.json`, `summary.md`, and a command-level
            Chrome Trace / Perfetto-compatible `trace.json`.
      - [x] Parse GGUF decode `--stream-stats` and helper diagnostic output into `gguf_decode_summary.json`,
            `gguf_decode_summary.md`, and `gguf_decode_trace.json` so token-step and helper attribution are bundled
            with the command logs.
      - [x] Separate lightweight step statistics from helper instrumentation: `--stream-stats` no longer enables helper
            timers implicitly, `--profile-helpers` opts into detailed attribution, and profile bundles report missing
            attribution as unavailable rather than as zero helper time.
      - [x] Include helper percentage attribution in the GGUF decode bundle summary: total helper share, per-step helper
            share, residual/non-helper share, and per-step top helper are now recorded so large residual/non-helper time
            remains visible.
      - [x] Import completed Qwen smoke reports with `benchmark/profile_bundle.py --qwen-smoke-report <report>` so
            helper/step attribution can be rebuilt from existing large-model evidence, while the bundle manifest links
            the original `qwen_smoke_trace.json` and waterfall outputs.
      - [x] Persist CPU AOT A/B knobs in Qwen smoke reports and comparison tables, including LLVM opt level, helper
            thread count, affinity, parallelism gate, Q8_K-staged mode, and compile-diagnostics mode.
      - [x] Compare completed profile bundles directly: `gguf_decode_compare.py --litenn-profile-summary
            <gguf_decode_summary.json>` now derives throughput and carries top-helper/helper-share plus
            top-operator/operator-share and residual-share columns into the output tables.
      - [x] Emit fine-grained Chrome Trace / Perfetto-compatible waterfall JSON from LiteNN spans, including graph import,
            GGUF/safetensors conversion, MLIR pass pipeline, LLVM/object emission, module loading, runtime schedule
            steps, GPU dispatch, host/device transfer, synchronization, and decode-loop token phases. Completed on
            2026-07-05: `qwen_smoke.py` writes command/span trace artifacts and `profile_bundle.py` can merge
            `qwen_smoke_trace.json` or arbitrary `--trace-json` inputs into the bundle `trace.json`.
      - [x] Add the first optional Linux `perf record` wrapper so local Linux runs can capture raw `perf.data` beside
            the bundle without making platform profilers mandatory.
      - [ ] Add full platform sampling adapters as optional wrappers: Windows ETW/xperf or WPA-export input, Linux `perf`,
            and a macOS Instruments-compatible import path when available.
            Updated on 2026-08-02: the bundle tool has optional Linux `perf record` and Windows `xperf` ETW start/stop
            wrappers. Linux capture now runs `perf script` automatically, records redacted conversion diagnostics,
            folds repeated callchains, and feeds collapsed stacks directly into Speedscope and flame-graph generation.
            Windows WPA/ETW and macOS Instruments stack import remain open.
      - [x] Normalize collapsed-stack inputs into merged collapsed stacks and Speedscope JSON outputs; render a simple
            built-in SVG/HTML flame graph for first-pass local diagnosis.
      - [ ] Convert all platform-native sampling outputs into the collapsed-stack input accepted by the bundle tool.
            - [x] Linux `perf.data` automatic import through `perf script`, with `--skip-sampler-import` for raw-only
                  captures.
            - [ ] Windows ETW/WPA and macOS Instruments import adapters.
      - [ ] Correlate samples with waterfall spans by thread id and timestamp, preserving backend, model, shape,
            token-count, compiler-option, commit, and tool-version metadata in the bundle manifest.
            - [x] Linux `perf script` samples preserve monotonic timestamp, PID/TID, CPU, command, and full stack as
                  `platform.sampling` instant events in the merged Chrome Trace / Perfetto timeline.
            - [ ] Add equivalent Windows/macOS event import and richer span ownership metadata.
      - [x] Add a no-local-path-leak profile mode for GGUF/Qwen smoke runs so large private model paths stay outside
            tracked artifacts while the bundle still records enough anonymized model metadata for comparison.
      - [ ] Acceptance: one command produces a profile directory containing raw tool logs, `trace.json`,
            `speedscope.json` or collapsed stacks, optional flame-graph HTML/SVG, benchmark/profile CSVs, and a short
            Markdown summary of top compile/runtime bottlenecks. Current first slice writes raw logs, `trace.json`,
            `manifest.json`, `summary.md`, `speedscope.json`, and built-in flame graph outputs for collapsed-stack
            inputs; qwen smoke trace/summary import is merged into the main bundle trace, and Linux native capture now
            generates stack outputs in the same command. Windows/macOS stack import and sample/timeline correlation
            remain open.

Completed notes:

- `docs/PerformanceAnalysis_2026-05-19.md` records CPU instruction-level findings, CPU intra-op results, CUDA native/CUDA Graph profile results, and the old fastpath retirement rationale.
- `docs/PerformanceOptimizationRoadmap.md` tracks the performance-specific P0-P5 checklist and current validation numbers.
- CPU AOT now keeps `LITENN_CPU_AOT_THREADS=1` on the MLIR packed/zmm fallback path, while larger static f32 fused chains can call `litenn_cpu_matmul_bias_relu_parallel_f32`.
  The productionization goal is to make that multithread path consistently profitable instead of merely available.
- CUDA Graph replay is currently the best CUDA inference path for pointer-stable static-shape runs; local batch-512 MLP512 graph replay reaches the same reported time as PyTorch CUDA in the 2026-05-19 run.
- Completed on 2026-05-20: `litenn_profile` writes raw `.o` and `.s` files, counts packed/scalar FMA, vector loads, broadcasts, gathers/scatters, stack vector ops, and falls back from `subgraph_0` to the first function for fused helper artifacts.
- Completed on 2026-05-20: `litenn_profile` prints CUDA launch breakdowns with backend kind, binary kind, kernel/library/PTX counts, workspace bytes, compile/load time, first native run, steady native run, first CUDA Graph run, and steady CUDA Graph replay time.
- Completed on 2026-05-20: rare numerical helper substrate landed for `OUT_PROD`, `TIMESTEP_EMBEDDING`, and the currently ggml-supported `SOLVE_TRI` variant, with CPU interpreter/reference tests and serialization coverage.
- Completed on 2026-05-20: optimizer-only graph ops landed as `SGDStepNode` and `AdamWStepNode`, keeping optimizer state explicit and serializable.

### G7: Heterogeneous Execution

Purpose: allow one graph to execute across multiple devices/backends while keeping graph semantics deterministic,
buffer ownership explicit, and compiled artifacts loadable without interpreter-only hidden state.

#### G7.1 Device Placement Contract

- [x] Add graph-level device-placement metadata for params, variables, intermediate values, and results.
      `PlacementOptions` now accepts node/value constraints that higher-level graph metadata can lower into, covering
      params, variables, intermediate values, and results through their executable `NodeOutput` producer.
- [x] Define automatic placement defaults: keep current single-device behavior when no placement metadata is present.
      `PlacementOptions::defaultBackend` keeps unconstrained plans on one requested backend, while an empty default
      preserves the existing cost-model behavior.
- [x] Add explicit copy/transfer edges or runtime transfer plans instead of hiding cross-device moves inside arbitrary nodes.
      `PlacementPlan` now records backend-to-backend transfer steps when a node consumes a value produced by another
      backend, and `RuntimeSchedule` can append those transfer steps with trace/profile labels.
- [x] Validate placement consistency, unsupported device ops, and illegal host/device aliasing with actionable diagnostics.
      Placement option validation rejects missing nodes/values, non-candidate backends, conflicting constraints, and
      unsatisfied node/value placement decisions; memory planning continues to reject hidden memory-space copies.

#### G7.2 Runtime Scheduling

- [x] Split execution into per-device segments with explicit input/output buffer boundaries.
      `Runtime::BuildPlacementSegments` groups contiguous placement decisions by backend and records explicit segment
      input/output buffers; `AppendPlacementSegmentSteps` exposes those segments through schedule trace/profile rows.
- [x] Add a CPU/CUDA mixed-execution smoke test where only a subgraph segment runs on CUDA and the rest remains on CPU.
      `G14Remaining.PlacementSegmentsExposePerBackendBufferBoundaries` now builds a CPU -> CUDA -> CPU segmented
      schedule with explicit transfer and sync steps.
      Schedule-level CPU/CUDA segment smoke coverage exists; true mixed executor run is still pending.
- [x] Track synchronization points and stream/event ownership for CUDA segments.
      Runtime sync steps now carry `streamOwner`, `eventOwner`, and `syncScope`; CUDA transfer-boundary sync records use
      `cuda-default-stream` / `cuda-runtime-event` and round-trip through vNext package metadata.
      `AppendPlacementSyncSteps` can now make transfer-adjacent sync points explicit for CUDA/Vulkan backends; concrete
      CUDA stream/event ownership still needs executor integration.
- [x] Add profiling output that reports per-device time, transfer time, and synchronization overhead.
      `RuntimeScheduleProfileSummary::devices` aggregates dispatch, transfer, sync, fallback, wall-time, and device-time
      buckets per backend.
      First transfer metadata rows are available from runtime schedule profile records; real measured transfer/sync timing
      still requires executor integration. `BuildRuntimeScheduleProfileSummary` now aggregates dispatch/transfer/sync/
      fallback buckets when measured records are supplied.

#### G7.3 AOT and Artifact Support

- [x] Extend compiled artifact metadata with per-segment backend kind, required device capabilities, and transfer ABI.
      Runtime package metadata now preserves `runtimeSegments`, `DispatchSegment` step links, and artifact-level
      `backendRequirements` entries generated from schedules with per-segment backend/capability/transfer-ABI strings.
- [x] Support loading heterogeneous artifacts from separated rodata/instruction regions.
      `LoadVNextArtifactRegions` resolves artifact region files relative to the package path or an explicit base
      directory, verifies declared size/checksum, and exposes named `rodata`/`metadata` plus `instructions` bytes for
      backend-specific module loaders.
- [x] Reject artifacts when a required backend/device capability is unavailable, with fallback policy documented.
      `ValidateVNextArtifactBackendRequirements` gates artifacts against the caller-provided backend/capability/transfer
      ABI set, and requires both artifact and runtime policy to explicitly allow fallback.

### G8: E-Graph Optimization

Purpose: add equality-saturation based graph optimization for algebraic rewrites, layout rewrites, and backend-aware
fusion without baking every pattern into ad-hoc passes.

Status: first conservative e-graph tranche completed on 2026-05-23. LiteNN now has a small
`EGraphPass` substrate for pure single-output tensor expressions, deterministic simplification,
explain/dump reporting, guardrails, and randomized interpreter-parity tests. Backend-aware extraction
and aggressive algebraic saturation remain long-term work. AOT benchmark hooks now include both normal-model
`EGraphAOTRunInto` rows and a redundant pure-graph AOT microbenchmark to expose current rewrite wins separately
from existing const-fold/fusion passes.

#### G8.1 E-Graph Substrate

- [x] Choose or implement a small C++ e-graph/e-class substrate compatible with LiteNN's `Graph`/`NodeOutput` model.
      （已完成：`src/LiteNN/Pass/EGraphPass.h` 内置 `TinyEGraph`，以 term key + union-find
      表达 e-class，并通过 `NodeOutput`/`OutputInfo` 连接 LiteNN 子图。）
- [x] Define canonical e-graph terms for pure tensor ops, constants, shapes, dtypes, and layout annotations.
      （已完成：首批覆盖 param/constant/variable、unary/binary、reshape/permute/broadcast/cast；
      term key 含 dtype、shape 和 layout-like attributes。）
- [x] Add import/export between LiteNN subgraphs and e-graph terms while preserving multi-output nodes where needed.
      （已完成：纯单输出节点可导入并从 rewrite/extraction 重建子图；多输出节点和无法证明纯性的节点作为
      opaque boundary 保留并通过输入重映射保持连通。）
- [x] Keep stateful/runtime-only ops outside the first e-graph tranche unless explicit purity rules exist.
      （已完成：optimizer step、activation/tape、control-flow、call/fused body 等不会进入首批 e-class
      rewrite，只参与普通 dependency remap。）

#### G8.2 Rewrite Rule Sets

- [x] Add safe algebraic rewrites for elementwise identities, associativity/commutativity where numerically acceptable, and constant folding handoff.
      （已完成：`x+0`、`x-0`、`x*1`、`x*0`、`Negate(Negate(x))`；
      commutative canonicalization is guarded by dtype/numerical-safety options. Full associativity remains disabled by default.）
- [x] Add NN-specific rewrites: matmul+bias+activation fusion, reshape/permute/broadcast canonicalization, and redundant copy removal.
      （已完成：reshape no-op/compose、permute identity/compose、broadcast no-op/compose；
      matmul+bias+activation fusion remains owned by `FusionPass` until an e-graph cost model can choose it explicitly.）
- [x] Add backend-aware cost models for CPU AOT, CUDA native, and interpreter fallback.
      （已完成：placement cost model uses backend capability relative cost, transfer/fallback penalties, layout/dtype
      penalties, workspace pressure, and explicit fallback policy; regression coverage ranks CPU AOT, CUDA native, and
      interpreter fallback paths. E-graph fused/decomposed extraction remains future aggressive-saturation work.）
- [x] Add numerical-safety flags so aggressive rewrites can be disabled for strict reproducibility.
      （已完成：`EGraphOptions::allowUnsafeFloatingRewrites` defaults to false; floating commutativity/associativity
      is not broadly applied under strict mode.）

#### G8.3 Validation

- [x] Add golden graph tests comparing original and optimized graphs on randomized inputs.
      （已完成：`tests/EGraphPassTest.cpp` compares original/optimized interpreter outputs across randomized inputs.）
- [x] Add rewrite explain/dump tooling so an optimization result can be reviewed.
      （已完成：`EGraphReport` records rewrite events and `DumpLastReport()` emits a reviewable text trace.）
- [x] Add guardrails for e-graph blow-up: iteration limits, node limits, timeout, and deterministic extraction.
      （已完成：`EGraphOptions` exposes iteration/term/e-class/timeout limits; extraction rebuild is deterministic
      and covered by a hit-limit test.）
- [x] Add AOT benchmark coverage for e-graph optimization.
      （已完成：`benchmark/bench.cpp` registers `EGraphAOTRunInto` for the existing model matrix, plus
      `AOTRedundantRawRunInto` / `EGraphAOTRedundantRunInto` for a pure redundant graph that is intentionally
      sensitive to rewrite/extraction quality.）

### G9: Rodata and Weight Separation

Purpose: make model weights, constants, compiled metadata, and executable instructions separable so static/shared
library embedding, memory mapping, hot weight swapping, and mobile packaging do not depend on one monolithic blob.

#### G9.1 Artifact Layout

- [x] Split compiled artifact storage into instruction bytes, immutable metadata, constant rodata, and external weight references.
- [x] Define stable section names and exported symbols for each region in carrier object output.
- [x] Add alignment, endian, version, and checksum fields per region.
- [x] Preserve dtype, quantization, shape, and variable-name metadata across separated regions.

#### G9.2 Loading and Binding

- [x] Add APIs to load instructions/metadata once and bind rodata/weights from separate addresses.
- [x] Add APIs to rebind compatible weight regions without recompiling instructions.
- [x] Validate region compatibility with detailed mismatch errors.
- [x] Add tests for static object, shared library, memory-mapped file, and in-memory byte span loading.

#### G9.3 Importer Integration

- [x] Let GGUF/LiteNN conversion emit archive-only, executable graph, compiled artifact, and separated-rodata variants.
- [x] Support large-weight packaging without exceeding PE/COFF section/object limits.
- [x] Document recommended packaging layouts for desktop, mobile, static library, and shared library users.

#### G9.4 CPU AOT External Constants and Weights

Purpose: make the already-separated artifact ABI useful for CPU MLIR/AOT graphs whose model weights would otherwise be
embedded into LLVM globals inside the instruction object.

- [x] Add the first CPU AOT context/region binding ABI so generated instructions can read immutable constants and
      variable weights from separated artifact regions instead of embedding every payload into instruction-owned LLVM
      globals.
- [x] Extend separated metadata to v2 with external tensor table entries: graph variable/constant name, region, dtype,
      shape, byte offset, byte size, alignment, checksum, and rebinding compatibility policy.
- [x] Populate the external tensor table for the initial CPU AOT external variable weights and constants path.
- [x] Populate the external tensor table for the first generic CPU MLIR externalized tensors.
- [x] Teach the CPU MLIR compile path to externalize forward-subgraph `VariableRefNode` / byte-addressable
      `ConstantNode` tensors through hidden AOT parameters bound from separated weights/constants regions.
- [x] Add a generic CPU MLIR external constant size-threshold policy and keep tiny scalar constants inline by default.
- [x] Propagate hidden external parameters through nested/control subgraphs.
- [x] Generalize the first external-regions slice beyond the optimized f32 parallel linear-chain AOT path.
- [x] Add an explicit CPU borrowed external-region loader for stable mapped constants/weights memory while keeping the
      default `Load()` API copy-owned.
- [x] Extend borrowed external-region loading to CUDA native host constants and document the exact shared-library/static
      library lifetime contract.
- [x] Keep CUDA CPU-bridge loading compatible with CPU AOT external regions by routing external-region artifacts
      through separated-region loading when `CompiledModuleArtifact::Load(CUDA{})` is used.
- [x] Add focused tests that verify non-empty CPU constants/weights regions, successful rebinding, size-mismatch
      diagnostics, and parity against interpreter output for the initial external-regions AOT path.
- [x] Add focused tests for the initial metadata-table population and public inspection API.
- [x] Add focused tests for external CPU weights rebinding, size-mismatch diagnostics, checksum mismatch diagnostics,
      and explicit external-region option coverage for the initial external-regions path.
- [x] Auto-apply `FusionPass` before the CPU f32 linear-chain external-regions fast path when external regions are
      explicitly enabled, so callers do not need to pre-fuse common Layer-generated linear chains just to externalize
      weights.
- [x] Add focused malformed external tensor metadata-table diagnostics for the initial external-regions path.
- [x] Replace hardwired compiler environment-variable behavior with explicit `CompilerOptions`; benchmark/profile entry
      points that still need environment-variable control parse it locally before calling compiler APIs.
- [x] Add broader malformed metadata-table cases for the current external-regions path.
- [x] Add generic MLIR externalization parity tests against inline AOT outputs.
- [x] Apply the externalization policy to Torch/SDXL imported graphs so full fixed-shape UNet artifacts do not inflate
      CPU instruction objects with multi-GiB weight globals.

Notes:

- Completed on 2026-05-22: `CompiledModuleArtifact::SeparateRodata()` now creates an owning separated artifact with
  metadata, constants, weights, and instructions regions. The metadata region contains a versioned/endian-checked
  manifest with per-region size, alignment, and checksum validation.
- Completed on 2026-05-22: CUDA native AOT constants are physically moved out of the instruction payload into the
  separated constants region; CPU MLIR object constants remain instruction-owned until the compiler grows an external
  global/weight pointer ABI.
- Completed on 2026-05-22: separated artifacts can load from in-memory spans or exported symbol addresses, rebind
  compatible constants/weights, and write either one combined separated carrier object or one object per region to avoid
  PE/COFF large-section pressure. `WriteRegionFiles` additionally supports raw metadata/constants/weights/instructions
  files for memory-mapped packaging.
- Completed on 2026-05-22: `litenn_gguf_convert --compile-cpu-separated/--compile-cuda-separated` emits split carrier
  objects for converted `.ltnn` graphs.
- Completed on 2026-05-25: CPU AOT gained an explicit external-region binding path for the optimized f32 parallel
  linear-chain compiler. It places `VariableRefNode` payload bytes into the separated weights region and `ConstantNode` payload bytes
  into the separated constants region, emits instruction code that obtains the bound base addresses through the LiteNN
  CPU runtime ABI, and covers direct artifact loading, separated-image loading, constants/weights rebinding, and
  interpreter parity in `CompiledModuleTest`.
- Completed on 2026-05-25: CUDA CPU-bridge loading now preserves CPU AOT external regions for both
  `CompiledModuleArtifact::Load(CUDA{})` and `CompiledModule<CUDA>::Load(CompiledModuleSeparatedImage)`, with a CUDA
  bridge regression covering a CPUNative fallback artifact whose variables live in the separated weights region.
- Completed on 2026-05-25: separated metadata v2 now carries external tensor entries and exposes them through
  `CompiledModuleSeparatedArtifact::ExternalTensorInfos()`. The initial CPU external-regions path records variable
  or constant names, constants/weights-region offsets, byte sizes, alignment, dtype/shape, per-entry checksum, and
  exact-checksum rebind policy; image validation checks table ranges and per-entry checksums in addition to whole-region
  checksums.
- Completed on 2026-05-25: the initial CPU external-regions runtime ABI now exposes both
  `litenn_cpu_external_constants()` and `litenn_cpu_external_weights()`, and the CUDA CPU bridge verifies that CPUNative
  fallback artifacts preserve non-empty weights regions and weights-region metadata entries.
- Completed on 2026-05-25: `CompiledModuleTest.CPUParallelLinearChainLoadsExternalRegions` now covers weights rebinding,
  wrong-size weights rejection, and corrupted weights checksum rejection; a separate explicit-options regression covers
  enabling external regions without relying on process environment variables.
- Completed on 2026-05-25: when external regions are enabled, `Compiler<CPU>::CompileArtifact` now retries the CPU f32
  linear-chain external-regions path after an internal `FusionPass`, covering common unfused Layer-generated linear
  graphs without requiring callers to mutate the graph first.
- Completed on 2026-05-25: CPU separated images now have `CompiledModule<CPU>::LoadBorrowedExternalRegions()` and
  `CompiledModuleSeparatedArtifact::LoadBorrowedExternalRegions()` for memory-mapped constants/weights. The default
  `Load()` path still copies external regions; the borrowed API explicitly requires the supplied constants/weights
  memory to outlive every run of the returned module.
- Completed on 2026-05-25: `CompiledModuleTest.CPUParallelLinearChainRejectsMalformedExternalTensorMetadata` corrupts
  an external tensor region name in separated metadata and verifies that validation reports the malformed table before
  load/JIT execution.
- Completed on 2026-05-25 and tightened during vNext: `CompilerOptions` now controls CPU AOT thread count, CPU parallel
  min-FLOPs, CPU external regions, generic MLIR external constant minimum byte size, internal external-region fusion,
  and CUDA native-AOT enablement. `Compiler<CPU/CUDA>` overloads accept options explicitly; command-line style entry
  points that need environment variables parse them outside the core library before calling compiler APIs.
- Completed on 2026-05-25: `Debug::DumpCompiledModuleMetadata` now also accepts
  `CompiledModuleSeparatedArtifact` and prints metadata/constants/weights/instructions sizes, per-region checksums, and
  external tensor table entries. This makes external region packaging and rebind diagnostics inspectable without parsing
  the binary metadata table by hand.
- Completed on 2026-05-25: external tensor metadata validation now has broader corruption coverage for invalid dtype,
  zero byte size, zero alignment, invalid rebind policy, unaligned offsets, and out-of-bounds byte ranges.
- Completed on 2026-05-25: the generic CPU MLIR compile path now has an initial external-regions route. When
  `CompilerOptions::enableCPUAOTExternalRegions` is enabled and the f32 linear-chain path does not apply, the compiler
  rewrites forward-subgraph `VariableRefNode` and eligible byte-addressable `ConstantNode` payloads into hidden MLIR
  function parameters, emits the actual bytes into separated weights/constants regions, and has the uniform CPU entry wrapper
  bind those hidden memrefs from `litenn_cpu_external_weights()` / `litenn_cpu_external_constants()` at run time.
- Completed on 2026-05-26: generic CPU MLIR external constants now honor
  `CompilerOptions::cpuAOTExternalConstantMinBytes` (default 64 bytes; CLI/env opt-in via
  `LITENN_CPU_AOT_EXTERNAL_CONSTANT_MIN_BYTES`). Small scalar/vector constants stay inline by default, while variable
  weights remain externalized when external regions are enabled. Regression coverage now compares generic external AOT
  output against both interpreter and inline AOT output.
- Completed on 2026-05-26: generic CPU MLIR externalization now rebuilds subgraphs so hidden external params are emitted
  before ordinary nodes and can propagate through `CallNode` and `CondNode` without violating topological validation.
  `CompiledModuleTest` covers both a callee-owned weight/constant capture and a conditional whose two branches capture
  different external tensors. `WhileNode` and special fused bodies that would require external captures deliberately
  fall back to normal inline AOT, because extending their hidden state would change loop/fusion ABI semantics.
- Completed on 2026-05-26: CUDA separated loading now has `CompiledModule<CUDA>::LoadBorrowedExternalRegions()` and
  `CompiledModuleSeparatedArtifact::LoadBorrowedExternalRegions(CUDA)`. For CUDA-native payloads, host constants are
  copied into device memory during load and may be released after `Load` returns; for CUDA CPU-bridge artifacts, the
  embedded CPU module uses the same borrowed constants/weights lifetime rule as `CompiledModule<CPU>`.
- Completed on 2026-05-26: Torch/SDXL imported graphs use the same explicit compiler options as other CLI entry points.
  The SDXL example now writes a separated carrier object when external regions are present, loads separated carrier
  symbols from DLL/shared-object exports, and preserves the image-region workflow by writing metadata to the existing
  `.rodata.bin` path plus sibling `.constants.bin`/`.weights.bin` files when external regions are enabled.

### G10: LoRA and Adapter Support

Purpose: support parameter-efficient adaptation without requiring full model rewrites, while preserving AOT compatibility
and clear base-weight versus adapter-weight ownership.

#### G10.1 Graph Representation

- [x] Add a LoRA metadata model: target module/name, rank, alpha, dropout policy, dtype, and merge mode:
      `Layer/LoRA.h` defines `LoRAAdapterMetadata`, `LoRAMergeMode`, and `LinearLoRAAdapter`.
- [x] Add Layer helpers that apply LoRA as `base(x) + scale * (x @ A @ B)` for linear layers:
      `CreateLinearLoRA` plus `AddLinearWithLoRA` now build the unmerged Linear adapter graph.
- [x] Support both unmerged runtime adapters and merged-weight export:
      `AddLinearWithLoRA` builds the runtime adapter path, and `MergeLinearLoRA` materializes a Float32 merged Linear
      layer for export/AOT paths.
- [x] Define compatibility rules for quantized base weights and low-precision adapter weights:
      `ValidateLinearLoRACompatibility` accepts Float32/Float16/BFloat16 unmerged adapters and rejects quantized
      adapters until a dequantized or merged export path is implemented.

#### G10.2 Import and Serialization

- [x] Add LiteNN serialization for adapter tensors and adapter metadata:
      vNext package manifests now carry typed `linear-lora` adapter entries that reference A/B tensor records.
- [x] Add safetensors LoRA import for common naming schemes used by Hugging Face/PEFT-style adapters:
      `ImportLinearLoRAAdapters` scans PEFT `*.lora_A[.adapter].weight` / `*.lora_B[.adapter].weight` tensors,
      materializes LiteNN Linear adapter layout, and returns `LinearLoRAAdapter` handles.
- [x] Add diagnostics for unmatched target names, shape/rank mismatches, and unsupported adapter variants:
      incomplete A/B pairs return diagnostics, duplicate roles and invalid rank/dtype/layout raise actionable errors.
- [x] Add roundtrip tests for saving/loading base model plus one or more adapters:
      `G14VNext.VNextModelPackageRoundTripsLoRAAdapterManifest` verifies adapter metadata and tensor references.

#### G10.3 Runtime and AOT

- [x] Add interpreter execution tests for unmerged LoRA: `LoRALayerTest.AddsUnmergedLinearAdapterDelta` runs the
      adapter graph through the CPU interpreter.
- [x] Add CPU AOT tests for merged LoRA export: `CompiledModuleTest.CPUMergedLoRALinearMatchesInterpreter` compiles
      the merged Linear graph and compares it with interpreter output.
- [x] Add optional runtime adapter binding for AOT when adapter weights are kept separate from base rodata:
      CPU AOT external-region mode now has LoRA-specific coverage proving unmerged A/B adapter variables are emitted as
      rebindable external `weights` tensors and borrowed adapter storage affects execution.
- [x] Add benchmark coverage for merged versus unmerged adapters:
      `litenn_bench` registers Interpreter and CPU AOT `LinearLoRA(784->512,r8)` rows for merged and unmerged paths.

### G11: Mobile Support and Test Matrix

Purpose: make LiteNN usable on mobile targets with constrained memory, predictable binary size, and repeatable device tests.

#### G11.1 Build and Portability

- [x] Define supported first targets: Android arm64-v8a first; iOS arm64 simulator/device remains future work pending
      toolchain availability and a non-Vulkan mobile GPU runtime decision.
- [x] Make compiler/MLIR/CUDA features optional so a minimal interpreter/runtime build is possible.
- [x] Audit C++ standard library, filesystem, reflection, dynamic loading, and thread usage for mobile constraints:
      `MobileSupport.h` exposes `QueryMobileConstraintStatus` / `CollectMobileConstraintDiagnostics` so CLIs can report
      supported, constrained, and unsupported mobile runtime policies without duplicating the audit.
- [x] Add CMake presets or toolchain documentation for mobile builds: `CMakePresets.json` now contains
      `android-arm64-vulkan-runtime`, and `docs/VulkanMobileDeployment.md` documents the preset-based flow.

#### G11.2 Runtime Constraints

- [x] Add allocator hooks or arena-style allocation for predictable memory usage: `CPU` now accepts an optional
      `CPUAllocator`, and `CPULinearArena` provides a resettable bump allocator for CPU/mobile fallback tensors.
- [x] Add binary-size and model-size reporting for mobile builds: `scripts/mobile_size_report.py` scans mobile build
      binaries and LiteNN model/package asset regions, then emits Markdown/JSON reports for CI artifacts or local
      release baselines.
- [x] Add CPU feature detection for ARM NEON and future mobile GPU/NNAPI/CoreML delegation points: `QueryCPUCapabilities`
      reports compile-time x86 SIMD plus ARM NEON/NEON FP16/SVE feature bits for mobile runtime policy and tests.
- [x] Define unsupported features explicitly, such as CUDA-only paths and desktop object loading where unavailable:
      `LiteNN/MobileSupport.h` now reports supported mobile runtime features and diagnostics for desktop-only features
      including CUDA, MLIR compiler, dynamic-library carrier loading, CPU object JIT, and on-device graph compilation.

#### G11.3 Testing

- [ ] Add host-side cross-compile smoke tests in CI once toolchains are available.
- [ ] Add on-device or emulator smoke tests for tensor ops, model loading, and a small inference graph.
- [x] Add mobile package examples using separated rodata/weights from G9: `example/vulkan` now demonstrates both
      metadata/instruction-only Vulkan-native Add and a fused LinearExternalWeights graph with a non-empty separated
      weights region loaded through borrowed external regions.
- [x] Track performance and memory baselines for at least one small MLP/CNN and one tiny transformer block:
      `scripts/mobile_baseline_report.py` combines Google Benchmark JSON rows for MLP/CNN/transformer-like kernels with
      `mobile_size_report.py` JSON totals into Markdown/JSON reports for CI artifacts and local release baselines.

### G12: Torch and Safetensors Import

Purpose: provide a practical bridge from PyTorch/Hugging Face artifacts into LiteNN graph/model formats, starting with
safetensors weights and expanding toward torch-exported graph structure.

Status: active on 2026-05-23. LiteNN now has a native safetensors reader/import path using vendored
`third_party/simdjson` for JSON header parsing, plus a LiteNN-specific Torch manifest importer paired with
safetensors weights. The importer validates expected dtype/shape/layout, materializes common PyTorch weight layouts,
lowers a minimal op set into LiteNN graph nodes, emits a converter report, and can save either variable-only
safetensors imports or manifest-defined graphs through `litenn_safetensors_convert`.

#### G12.1 Safetensors Reader

- [x] Implement a safetensors metadata and tensor-payload reader with bounds checks and dtype/shape validation.
      （已完成：`LiteNN::Serialization::SafetensorsArchive` supports in-memory and file loading, validates header
      length, JSON structure through `third_party/simdjson`, dtype, shape, byte-size, offset bounds, overlap,
      and BOOL payload bytes.）
- [x] Map safetensors dtypes to LiteNN dtypes, including fp16/bf16/fp8/int storage where available.
      （已完成：supported mappings cover F64/F32/F16/BF16/F8_E4M3/F8_E5M2/I64/I32/I8/U8/BOOL; unsupported
      safetensors integer widths are rejected explicitly until LiteNN adds matching storage types.）
- [x] Preserve tensor names and provide rename/transpose hooks for common PyTorch weight layouts.
      （已完成：`ImportSafetensorsVariables` preserves names by default and exposes `SafetensorsImportOptions`
      hooks for renaming and rank-2 transposition.）
- [x] Add tests with minimal safetensors fixtures and corrupted-header/error-path fixtures.
      （已完成：`tests/SafetensorsTest.cpp` covers metadata/payload reading, rename+transpose variable import,
      unsupported dtype, byte-size mismatch, offset overflow, and invalid BOOL payload paths.）

#### G12.2 Torch Weight Import

- [x] Add a CLI/import API to convert safetensors weights into LiteNN variables or separated rodata regions.
      （已完成：library API imports to a variable-only LiteNN graph archive; `tools/torch/litenn_safetensors_convert`
      writes `.ltnn` archives and supports repeated `--rename from=to` and `--transpose name` options. Separated
      rodata export remains owned by G9 artifact packaging once graph manifests are available.）
- [x] Add mapping presets for common module names: Linear, Embedding, LayerNorm/RMSNorm, attention projections, and LoRA adapters.
      （已完成：Torch manifest tensor layouts cover identity/embedding, `torch_linear_weight`,
      `torch_attention_projection_weight`, `torch_bias_1d`, norm weight/bias reshape, and PEFT-style LoRA A/B
      2D transpose layouts.）
- [x] Add diagnostics for missing tensors, extra tensors, dtype mismatch, shape mismatch, and layout mismatch.
      （已完成：manifest import reports missing safetensors sources, unused archive tensors by default,
      expected dtype/source-shape/final-shape mismatches, unsupported layout presets, rank mismatch for transpose/
      bias/norm layouts, duplicate input/tensor/output names, and unsupported ops. Tensor entries can also request
      `target_dtype` so import-time materialization can validate source dtype while storing constants in the compute dtype.）
- [x] Add golden tests against PyTorch for small exported MLP/attention fixtures.
      （已完成：`tests/TorchManifestTest.cpp` imports a PyTorch-style Linear+ReLU fixture from manifest+safetensors,
      checks PyTorch golden outputs through the interpreter, and also checks CPU AOT when MLIR is enabled.
      Attention fixture remains a later coverage expansion.）

#### G12.3 Torch Graph Support

- [x] Decide first graph source: `torch.export`, `torch.fx`, ONNX, or a LiteNN-specific JSON manifest paired with safetensors.
      （决策：首个实现目标采用 LiteNN-specific JSON manifest + safetensors weights。原因是它最容易绑定
      rename/transpose/expected-shape diagnostics，并可作为 torch.export/fx/ONNX 前端的稳定中间层。）
- [x] Define a minimal op mapping table from torch ops to existing LiteNN Layer/Node helpers.
      （已完成：`SupportedTorchManifestOpMappings()` documents the first supported set: Linear/attention projection,
      Embedding, Conv2D/ConvTranspose2D, LayerNorm, RMSNorm, PyTorch-NCHW GroupNorm, timestep embedding,
      residual/feed-forward/GEGLU/attention/VAE decode composites, MatMul/Add/Subtract/Multiply/Divide, scalar scale,
      Concat, Slice, ReLU/GELU/SiLU/Sigmoid/Tanh, Softmax, Pad, Upsample, Clamp, Reshape, Permute, and 2D Transpose.）
- [x] Add a converter report listing lowered ops, folded constants, unsupported ops, and required fallbacks.
      （已完成：`TorchManifestReport` records imported tensors, lowered ops, folded constant layout transforms,
      unsupported ops, fallbacks, and diagnostics; the CLI prints the report after manifest conversion.）
- [x] Add roundtrip examples showing PyTorch weights plus graph manifest imported to LiteNN and compiled through CPU AOT.
      （已完成：`example/torch_manifest` provides a PyTorch fixture exporter that writes safetensors,
      a manifest using Torch layout presets, and a C++ example that loads the manifest, runs the interpreter,
      and also runs CPU AOT when `LiteNNCompiler` is available.）

#### G12.4 SDXL / Diffusion Model Import Facilities

- [x] Add manifest lowering for SDXL UNet/VAE foundation ops: Conv2D, GroupNorm with optional affine scale/bias,
      timestep embedding, Pad, Upsample, and channel-first tensor layout presets.
      （已完成：Torch manifest now lowers `conv2d`, `group_norm`, `timestep_embedding`, `pad`, and `upsample`;
      tensor layouts include PyTorch Conv2D weights and channel/group-norm affine `[1, C, 1, 1]` materialization.
      `tests/TorchManifestTest.cpp` covers a small diffusion-foundation block through the interpreter.）
- [x] Add residual-block and feed-forward graph assembly patterns that combine Conv2D/GroupNorm/SiLU/Linear without
      requiring every PyTorch module to be hand-expanded in manifests.
      （已完成：Torch manifest adds `residual_block` and `feed_forward` composite ops. `residual_block`
      assembles PyTorch-NCHW GroupNorm, activation, Conv2D, optional timestep projection, optional skip Conv2D,
      and residual add; `feed_forward` assembles Linear, activation or gated FFN, output Linear, and optional residual.）
- [x] Add attention-block lowering for fixed-shape self/cross attention: Q/K/V projections, head reshape/permute,
      scaled dot-product attention or FlashAttnExt, output projection, and residual add.
      （已完成：`attention_block` lowers fixed `[tokens, channels]` self/cross attention through Q/K/V Linear
      projections, head reshape/permute, scaled dot-product attention using BatchMatMul + Softmax,
      output projection, optional mask, and optional residual add.）
- [x] Add VAE decode coverage: Conv2D/ConvTranspose2D/Upsample, GroupNorm/SiLU, and final output scaling/clamp policy.
      （已完成：Torch manifest adds primitive `conv_transpose2d`, `scale`, `clamp`, and a `vae_decode`
      composite step list for Conv2D, PyTorch-NCHW GroupNorm, SiLU, Upsample, ConvTranspose2D,
      latent/output scaling, output bias, and clamp/clip.）
- [x] Add scheduler/runtime contract docs for denoise-loop orchestration outside the graph: timestep schedule,
      classifier-free guidance binding, latent scaling, and per-step benchmark boundaries.
      （已完成：`example/sdxl/README.md` documents the denoise-loop runtime contract and keeps scheduler,
      classifier-free guidance, latent scaling, and benchmark boundaries outside the compiled graph.）
- [x] Add SDXL golden/parity fixtures against PyTorch/diffusers for at least one tiny fixed-shape UNet block before
      attempting full SDXL checkpoints.
      （已完成：`TorchManifest.ImportsSDXLCompositePatternsWithTinyParityFixture` imports a tiny fixed-shape
      PyTorch-style UNet/attention/VAE composite fixture from safetensors and checks deterministic golden outputs
      for the residual/feed-forward/VAE paths plus fixed-shape attention execution.）

#### G12.5 SDXL Example and Packaging

- [x] Add an SDXL import experiment example that accepts safetensors plus a manifest, serializes a LiteNN graph,
      compiles a carrier object, and loads it from a DLL/shared object.
      （已完成：`example/sdxl` supports checkpoint inspection, manifest import to `.ltnn`, carrier object emission,
      exported-symbol DLL/shared-object smoke loading, direct in-process AOT smoke running, and raw instruction-object
      dumping for JIT/runtime-symbol diagnostics.）
- [x] Add a manifest template generator that starts from diffusers/original SDXL config files and emits a first
      fixed-shape UNet manifest skeleton with expected tensor names and shapes.
      （已完成：`example/sdxl/sdxl_manifest_probe.py` reads Stability-AI `generative-models` SDXL YAML plus
      a safetensors header, checks key tensor compatibility, and emits importable `unet-stem`,
      `unet-resblock`, `unet-euler-smoke`, and `vae-decode-stem` probe manifests. `unet-resblock` covers
      stem Conv2D plus `input_blocks.1.0` with PyTorch-style GroupNorm, SiLU, timestep projection,
      Conv2D, and residual add. `unet-euler-smoke` maps real SDXL stem, `time_embed`, first ResBlock, and
      output weights to a denoiser-shaped 4-channel `noise_pred` output for sampler smoke tests. The CPU AOT
      smoke path uses manifest `target_dtype` to materialize F16 checkpoint tensors as F32 constants until
      production CPU half-precision lowering is validated.）
- [x] Add a minimal Euler sampler runtime in the SDXL example.
      （已完成：`litenn_sdxl_example --sample-euler` loads a carrier DLL/shared object, initializes the
      `latent` input, fills a `timestep`/`timesteps` input when present, zero-fills other inputs, runs the module
      for each step, and applies an Euler epsilon-prediction update using the `noise_pred` output.）
- [x] Validate a first real-checkpoint end-to-end smoke flow through manifest generation, safetensors import,
      `.ltnn` serialization, CPU AOT compile, carrier DLL load, and Euler sampler execution.
      （已完成：the 64x64 `unet-euler-smoke` path runs from a real SDXL-compatible safetensors checkpoint through
      `--run-model`, `--load-dll`, and `--sample-euler`; generated artifacts remain under build output only.）
- [x] Expand the generator from first probes to broader fixed-shape SDXL skeletons: label/conditioning embedding
      prefix, SpatialTransformer attention blocks, down/up-block traversal, and full VAE decode templates.
      （已完成：`sdxl_manifest_probe.py` now emits `unet-conditioning-smoke` for SDXL vector/label conditioning,
      `spatial-transformer-smoke` for fixed-token middle-block self/cross attention, `vae-decode-full` for fixed-shape
      VAE decoder traversal, and an `--emit-skeleton-plan` JSON that records discovered UNet input/middle/output
      block traversal and skip-join requirements. Torch manifest `concat` now lowers to `ConcatNode` for UNet
      skip-channel joins. `unet-full-fixed` now covers the full fixed-shape UNet traversal with 4D
      SpatialTransformer blocks; production full-image parity still defers tokenizer/text-encoder execution inside
      LiteNN and exact tiled/chunked VAE attention if the current memory-policy fallback is not sufficient.）
- [x] Add an SDXL benchmark path that times import, graph serialization, CPU AOT compile, DLL load, and one denoise-step
      invocation separately.
      （已完成：`example/sdxl/sdxl_bench.py` drives python311 manifest generation, safetensors import,
      `.ltnn` serialization, CPU AOT carrier object emission, shared-library linking, exported-symbol load,
      and one Euler denoise-step invocation, writing Markdown/JSON stage tables under caller-selected build output.）
- [x] Add a reference prompt-generation helper for 1024x1024 SDXL validation.
      （已完成：`example/sdxl/sdxl_generate_reference.py` runs Stability-AI/generative-models with Euler EDM when
      that external Python environment is installed, producing a PNG from a prompt/checkpoint pair for later LiteNN
      parity checks. This remains a reference harness, not a LiteNN full-pipeline frontend.）

#### G12.6 End-to-End SDXL Image Generation

Purpose: move from runnable SDXL subgraph smoke tests to producing a semantic 1024x1024 image from a prompt through
LiteNN-owned execution, while keeping early milestones testable without requiring every text/model component to be
native on day one.

Phase 0: bind external conditioning and make the denoiser graph complete enough to run.

- [x] Add a safetensors runtime input-binding path for compiled SDXL smoke graphs and carrier DLL/shared-object
      execution.
      （已完成：`litenn_sdxl_example --run-model-with-inputs` and `--load-dll-with-inputs` bind compiled input
      signatures by tensor name from safetensors, validating dtype/shape before AOT execution. `--sample-euler`
      also accepts `--inputs` for externally supplied conditioning tensors while still owning latent/timestep updates.）
- [x] Define the initial LiteNN SDXL denoiser ABI for fixed-shape smoke execution: named `latent`, `timestep` or
      `timesteps`, context/vector inputs as named safetensors tensors, and `noise_pred` output.
      （已完成：the example runtime now treats compiled module input names as the ABI, uses strict binding for
      one-shot runs, zero-missing conditioning binding for Euler sampling, and documents the scheduler-owned denoise
      contract in `example/sdxl/README.md`.）
- [x] Add a prompt-to-conditioning export path that can take prompt text through Stability-AI/generative-models or
      diffusers and save the exact LiteNN runtime inputs: `crossattn`, pooled/vector conditioning, original/target
      size embeddings, crop embeddings, negative conditioning, and classifier-free-guidance batch layout.
      （已完成：`example/sdxl/sdxl_export_conditioning.py` loads the Stability-AI/generative-models conditioner,
      builds the SDXL prompt/negative prompt batch, and writes F32/F16/BF16 safetensors bindings for `context`,
      `vector_cond`, negative variants, CFG-concatenated variants, and raw `cond.*` / `uncond.*` tensors.
      The default path now instantiates only the conditioner, patches CLIP/OpenCLIP construction to avoid fetching
      pretrained weights, and loads checkpoint `conditioner.*` tensors directly; `--full-model` remains available for
      parity debugging against the original runtime.）
- [x] Generate a full fixed-shape SDXL UNet manifest from the Stability checkpoint layout: input blocks, middle block,
      output blocks, skip stack, channel-axis `concat`, ResBlocks, downsample/upsample, and transformer blocks.
      （已完成：`example/sdxl/sdxl_manifest_probe.py --probe unet-full-fixed` now emits a batch=1 fixed-shape
      Stability-layout UNet manifest from checkpoint tensor names, including `time_embed`, SDXL vector/label
      conditioning, input/middle/output block traversal, skip-stack channel concat, ResBlocks, conv downsample,
      nearest+conv upsample, and all discovered `spatial_transformer_2d` blocks. The graph still consumes
      externally exported `context`/`vector_cond` tensors while native text encoders remain a separate target.
      `--compute-dtype F16|BF16` now preserves low-precision probe inputs/weights and inserts an explicit cast after
      F32 timestep embedding output so mixed-precision manifest import remains type-correct.）
- [x] Support SDXL Transformer FFN GEGLU combined projection in Torch manifests.
      （已完成：Torch manifest now lowers `slice` and `geglu_feed_forward`; `spatial-transformer-smoke` emits
      norm3 plus `ff.net.0.proj` chunk/GELU/gate/`ff.net.2` for the middle-block transformer smoke path.）
- [x] Add 4D SpatialTransformer lowering templates: NCHW -> token flatten, self/cross attention, GEGLU FFN,
      token -> NCHW restore, and residual/proj_in/proj_out handling.
      （已完成：Torch manifest now has `spatial_transformer_2d`, covering batch=1 NCHW GroupNorm,
      `use_linear_in_transformer=True` proj_in/proj_out, NCHW/token flatten+restore, self-attention,
      cross-attention, GEGLU FFN, and residual add. `spatial-transformer-2d-smoke` emits the SDXL middle-block
      version from real Stability checkpoint tensor names; the 64x64 fixed-shape smoke graph imports and runs through
      CPU AOT `--run-model`.）
- [x] Add parity fixtures for one real SDXL SpatialTransformer block against PyTorch/generative-models tensors,
      including GEGLU and cross-attention.
      （已完成：`example/sdxl/sdxl_export_spatial_transformer_fixture.py` loads a real Stability-AI
      `model.diffusion_model.middle_block.1` module, writes LiteNN `features`/`context` bindings plus PyTorch
      `features_out` reference safetensors, and `sdxl_compare_safetensors.py` compares LiteNN output against the
      reference. This remains an external-env example harness because CI does not carry the SDXL checkpoint or
      generative-models runtime dependencies.）

Phase 1: make a real denoise loop produce a latent.

- [x] Add deterministic Euler sampler output for compiled denoiser artifacts: seed-owned latent init, linear/EDM sigma
      schedules, epsilon prediction update, per-step diagnostics, and final latent safetensors output.
      （已完成：`litenn_sdxl_example --sample-euler` supports `--scheduler linear|edm`, `--rho`,
      `--output-latent`, deterministic random latent initialization, and writes `latent` safetensors output for
      VAE-decode handoff.）
- [x] Complete the production SDXL denoiser runtime contract: latent input scaling, CFG combine, sigma-to-timestep
      mapping parity with Stability-AI/generative-models, and batch convention diagnostics.
      （已完成：`--sample-euler` now supports explicit `epsilon`/`denoised`/`sgm-edm`/`sgm-eps`/`sgm-v`
      denoiser contracts, `auto`/`legacy`/`sigma`/`edm-log`/`zero` timestep modes, SGM-style `c_in` latent
      scaling and `c_noise` timestep binding, plus dual-pass CFG with `negative_*` / `uncond.*` conditioning
      aliases and visible binding diagnostics.）
- [x] Add a CLI flow that accepts exported conditioning and writes the final latent tensor after N denoise steps using
      a compiled LiteNN UNet DLL/shared object.
      （已完成：`litenn_sdxl_example --denoise-latent <module> <inputs.safetensors> <output-latent.safetensors>`
      wraps the sampler with positional conditioning/output paths for pipeline scripts.
      `--denoise-latent-image <rodata.bin> <instructions.obj> ...` provides the same Euler path for separated
      rodata/instruction image-region files. Full checkpoint/config to complete-UNet compilation remains tracked by
      the full fixed-shape UNet manifest item above.）
- [x] Add CPU AOT and CUDA AOT benchmark rows for one full denoise step, with memory use separated from latency.
      （已完成：`litenn_sdxl_example --benchmark-model-with-inputs` reports compile/load/input-bind/upload/run
      timing separately plus rodata/instruction/input/output bytes; `example/sdxl/sdxl_bench.py` now writes
      `cpu-aot-denoise-step` and `cuda-aot-denoise-step` rows into Markdown/JSON. The CUDA row records the actual
      compiled backend, so unsupported full graphs are visible as `cpu_native` bridge/fallback rather than native CUDA.）

Phase 2: decode and write an image.

- [x] Run the full VAE decoder manifest on a saved final latent through a carrier DLL/shared object and write an image
      tensor artifact.
      （已完成：AOT now lowers the nearest `UpsampleNode` path needed by `vae-decode-full`; `--load-dll-with-inputs`
      can write the single `image` output as safetensors with `--output`. The in-process `--run-model-with-inputs`
      path was fixed on Win64 by using the large code model for CPU AOT object emission, avoiding out-of-range
      `.text` to `.rdata` short relocations in large VAE artifacts.）
- [x] Add a PNG writing path or a Python postprocess bridge for `[N, 3, H, W]` Float32 image tensors.
      （已完成：`example/sdxl/sdxl_tensor_to_png.py` reads F32/F16/BF16 image safetensors tensors named `image`,
      `decoded`, `output`, or the first tensor, converts to Float32 for postprocess, and writes an RGB PNG through
      Pillow.）
- [x] Add 1024x1024 memory policy: VAE mid-attention tiling/fallback, workspace sizing, and failure diagnostics when
      CPU-only memory or time would be unreasonable.
      （已完成：`sdxl_manifest_probe.py --probe vae-decode-full` now accepts
      `--vae-mid-attention-policy auto|force|skip` plus `--vae-attention-max-mib`; metadata records token count,
      score/probability bytes, estimated workspace, selected status, and reason. The default `auto` path emits exact
      dense VAE mid-attention for small fixed-shape smoke manifests and uses a recorded skip fallback for 1024x1024
      decodes that exceed the workspace budget; `force` remains available for exact validation on large-memory hosts.
      `sdxl_bench.py` forwards the policy flags so benchmark artifacts carry the same diagnostics.）
- [ ] Validate one generated 1024x1024 image against the reference runtime at fixed seed/prompt, first by tensor stats
      and then by image-level tolerances.
      （当前状态：the prompt-to-image plumbing is validated with prompt `1girl` through the 64x64
      `unet-conditioning-smoke` LiteNN AOT denoiser plus LiteNN AOT VAE decode, producing
      `build-release/sdxl_1girl_script_smoke/1girl_smoke.png` and
      `build-release/sdxl_1girl_regions_smoke/1girl_regions_smoke.png` during local validation. The latter uses direct
      rodata/instruction image-region loading instead of DLL/shared-object loading. The full fixed-shape 64x64 UNet
      manifest imports, but materializes a roughly 10 GiB F32 `.ltnn` graph and CPU AOT compilation did not finish
      within a 15 minute local timeout. F16 manifest import now works and halves the small smoke `.ltnn` size
      (about 38.3 MiB to 19.2 MiB locally), but full 1024x1024 reference parity is still blocked on full-graph compile
      time, external weight/codegen pressure, and native/full CUDA lowering rather than on prompt binding or VAE output
      plumbing.
      2026-05-27 追踪：一次 `unet-full-fixed` 64x64 F16 prompt run stopped在 import 后、compile 前，生成的
      `unet.ltnn` 为约 5.14 GiB，manifest tensor payload 估算约 4.9 GiB；继续 CPU AOT 会因为变量梯度镜像、
      external weights region 复制和 MLIR/LLVM 编译峰值叠加而进入数十 GiB 内存区间。已完成两项止血修复：
      Torch manifest 默认导入 frozen variables，不再为推理权重分配同尺寸 grad storage；CPU AOT external-region
      构建改为直接从 CPU tensor 写入外置 region，避免 payload 临时 vector 的整份二次复制。`sdxl_prompt_to_image.py`
      还新增 `--max-unet-weight-mib` 预检，默认在超大 full UNet manifest 进入 import/compile 前失败，避免再次
      触发长时间高内存运行；大内存主机可显式传 `--max-unet-weight-mib 0` 关闭保护。
      2026-05-28 追踪：full64 BF16 `unet-full-fixed` 已可通过 separated image regions 完成 CPU AOT
      编译、加载、1-step/4-step Euler denoise 和 VAE decode；输出 PNG
      `build\sdxl_bf16_full64\1girl_bf16_64_4step.png` 非空但仍不具备可接受语义质量。F16 full UNet 在
      zero latent 下仍输出 NaN，说明下一步需要 mixed-precision accumulation / per-node finite diagnostics，而不是
      单纯调整初始噪声。
      2026-05-29 追踪：CPU AOT F16 已加入首批 mixed-precision lowering（Softmax、LayerNorm/RMSNorm、
      MatMul、BatchMatMul、Conv2D）并将 `Layer::AddTanh` 改为稳定的 `2*sigmoid(2x)-1` 展开；重新 import
      后的 full64 F16 UNet 可完成 zero-latent 1-step（`pred_rms=0.254`）和 random dual-CFG 4-step
      denoise，不再产生 NaN。输出 `build\sdxl_f16_stable\1girl_f16_64_4step.png` 非空、范围正常，但仍是
      抽象结果，语义质量问题尚未关闭。SDXL CLI 的 sampler / run-with-inputs 路径已默认检查输出 NaN/Inf，
      并提供 `--allow-nonfinite` 作为诊断逃生口，避免再次静默写出坏 latent / output。）

Phase 3: unblock full semantic image generation.

Closure status: Phase 3 is closed for CPU AOT plumbing, diagnostics, and reference-comparison tooling. Remaining native
CUDA full-UNet coverage, 1024x1024 quality parity, and archived cross-dtype reference artifacts are explicitly deferred
to the long-term SDXL parity queue instead of remaining hidden Phase 3 work.

- [x] Add a model-level external-weight format so imported Torch/SDXL `.ltnn` graphs can reference a sibling weight
      payload instead of embedding multi-GiB variable tensors inline.
      - [x] Serialize variable metadata separately from payload bytes, including dtype, shape, quantization metadata,
            grad-storage flag, external filename, byte offset, byte size, and alignment.
      - [x] Load external weights relative to the `.ltnn` file, keep the backing bytes owned by the `Graph`, and expose
            borrowed frozen CPU tensors to existing interpreter/compiler code.
            （已加固：large external weight files are read in bounded chunks; the 64x64 full UNet F16
            `unet.weights.bin` load path now handles a 5,134,927,424 byte sibling weights file without relying on one
            giant `istream::read` call.）
      - [x] Add `--import-external-weights` or an equivalent SDXL import option that writes `<graph>.ltnn` plus a sibling
            weights file without recording private checkpoint paths.
      - [x] Make all SDXL compile/run entry points load both inline and external-weight `.ltnn` files transparently.
      - [x] Add tests proving external-weight models round-trip, run through the interpreter, and compile through CPU AOT
            external regions without re-inlining the payload.
      （已完成：model format v21 adds inline/external variable payload records; `SaveGraphArchiveExternalWeights`
      writes sibling `.weights.bin` payloads with relative paths, `LoadGraphArchive` owns external bytes in `Graph` and
      exposes borrowed CPU tensors, `example/sdxl --import --external-weights` plus
      `sdxl_prompt_to_image.py` default to sibling weight files, and ModelIO/CPU AOT tests cover round-trip,
      interpreter run, and external-region compilation after load.）
- [x] Build the full UNet correctness harness before chasing image quality.
      - [x] Run one `unet-full-fixed` 64x64 single-step denoise through LiteNN with fixed seed,
            timestep/sigma, CFG convention, and exported conditioning.
      - [x] Add an executable reference-runtime UNet step exporter:
            `example/sdxl/sdxl_reference_unet_step.py` consumes the same LiteNN-style safetensors input binding and
            writes reference `noise_pred` via Stability-AI/generative-models.
      - [x] Add tensor-level comparison for `noise_pred`, final latent after one Euler step, and VAE decoded image.
            `example/sdxl/sdxl_compare_artifacts.py` reports mean/max absolute error, RMSE, max relative error,
            tolerance violations, finite-pair counts, and optional JSON metrics.
      - [x] Record tolerances separately for F32, F16, BF16, and image-space comparisons as compare-script presets.
      - [x] Defer archiving fixed seed/prompt reference comparison artifacts for F32/F16/BF16 to a large-memory
            validation-host run. The executable harness is present; artifact curation is not missing Phase 3 code.
- [x] Make full UNet compilation practical for the current CPU AOT path.
      - [x] Profile coarse CPU AOT compile phases after model-level external weights are enabled.
            （已完成首轮阶段定位：full64 F16 O3/O1 均卡在 LLVM module optimization；O0 跳过 module opt 后
            F16 codegen 约 172.6s、object emit 约 120.7s，BF16 codegen 约 115.4s、object emit 约 79.2s。）
      - [x] Avoid MLIR/LLVM IR constants for large weights and keep them as runtime external region arguments through
            lowering.
            （已完成当前 CPU AOT 路径：SDXL example no longer relies on environment variables for this behavior;
            its compile/run/benchmark entry points build explicit `CompilerOptions` with
            `enableCPUAOTExternalRegions=true`, so VariableRef weights are rewritten to hidden runtime parameters
            before GraphToMLIR and enter separated weight regions instead of DenseElementsAttr/LLVM globals.）
      - [x] Add explicit CPU AOT LLVM opt-level control for SDXL-sized compile experiments.
            （已完成：`CompilerOptions::cpuAOTLLVMOptLevel` defaults to O3 for the library, while the SDXL example
            defaults image-region compilation to O0 and exposes `--cpu-aot-llvm-opt-level 0|1|2|3` through the CLI
            and prompt harness.）
      - [x] Remove avoidable multi-GiB copies in separated image-region write/load paths.
            （已完成：`CompiledModuleArtifact::BuildSeparatedMetadata()` plus constants/weights spans let tools write
            separated regions without constructing a second 5GB `CompiledModuleSeparatedArtifact`; image-region loading
            can move owned vectors into `CompiledModuleSeparatedArtifact::FromOwnedRegions(...)` and use borrowed
            external regions for execution. Large reads/writes are chunked.）
      - [x] Defer split/cache full UNet compilation units until profiling shows O0 separated-image compilation is no
            longer sufficient; current full64 BF16/F16 CPU AOT image-region validation completes without splitting.
      - [x] Add compile-time budget diagnostics that report expected model payload, MLIR op count, instruction bytes,
            and external region bytes before codegen starts.
            （已完成首批预算诊断：`EstimateCompileBudget` reports subgraph/node/variable/constant counts,
            variable/constant/qconstant payload bytes, projected inline MLIR payload, and projected external
            constants/weights; SDXL example prints this before compile/run/benchmark. Instruction bytes are reported
            after object emission as before. `--compile-budget` can print the same estimate without invoking MLIR/LLVM.
            Local full64 F16 UNet validation: `unet.ltnn` is 757,293 bytes, sibling `unet.weights.bin` is
            5,134,927,424 bytes, and compile budget reports 7,153 nodes, 1,680 variables, 667 constants,
            4,897.05 MiB projected external weights, and only 1,334 bytes projected inline MLIR payload.）
      - [x] Add mixed-precision accumulation policy for F16 SDXL graphs.
            - [x] Use Float32 intermediates for CPU AOT F16 Softmax and LayerNorm/RMSNorm lowering, then cast results
                  back to F16. Regression coverage: `CompiledModuleTest.CPUFloat16SoftmaxArtifactUsesFloat32Accumulator`
                  and `CompiledModuleTest.CPUFloat16RMSNormArtifactUsesFloat32Accumulator`.
            - [x] Extend the policy to MatMul/BatchMatMul/Conv accumulation. Regression coverage:
                  `CompiledModuleTest.CPUFloat16MatMulArtifactUsesFloat32Accumulator`,
                  `CompiledModuleTest.CPUFloat16BatchMatMulArtifactUsesFloat32Accumulator`, and
                  `CompiledModuleTest.CPUFloat16Conv2DArtifactUsesFloat32Accumulator`.
            - [x] Replace the unstable generated Tanh formula with `2*sigmoid(2x)-1`, avoiding F16 `inf/inf` in
                  GELU/GEGLU paths. Regression coverage:
                  `CompiledModuleTest.CPUFloat16GELUArtifactUsesStableTanh`.
            - [x] Add output-level finite guards to the SDXL sampler and AOT run-with-inputs CLI paths, with
                  `--allow-nonfinite` for collecting broken debug artifacts.
            - [x] Add full-graph finite diagnostics that can stop after the first non-finite SDXL tensor.
                  `litenn_sdxl_example --diagnose-model-with-inputs` runs the `.ltnn` graph through the interpreter,
                  checks every floating node output, supports `--verbose`, `--max-nodes`, and `--allow-nonfinite`,
                  and reports the exact subgraph/node/kind/port that first produces NaN/Inf.
            （已验证当前 CPU AOT F16 路径：重新 import 后的 full64 F16 UNet image-region artifact 可完成
            zero-latent 1-step 和 random dual-CFG 4-step denoise；有限性问题阶段解除。finite diagnostics
            仍保留为后续定位工具，以便处理未来模型或 1024x1024 图中的首个非有限 tensor。）
- [x] Close Phase 3 CPU AOT image-validation plumbing and move native CUDA/full-quality parity to the long-term queue.
      - [x] Keep CUDA/native execution tracked as a separate long-term critical path rather than a Phase 3 blocker:
            full UNet CUDA native/AOT still needs Conv2D, GroupNorm/LayerNorm, Linear/MatMul, BatchMatMul, Softmax,
            SpatialTransformer, GEGLU, Concat, Slice, Reshape, Permute, Upsample, scalar scheduler ops, memory-efficient
            attention, and separated constants/weights ABI validation.
      - [x] Generate 64x64 `1girl` outputs with full UNet, save PNG plus raw safetensors artifacts, and inspect/read the
            resulting image before marking smoke image plumbing complete.
            （进展：已生成并读取 `build\sdxl_bf16_full64\1girl_bf16_64.png` 和
            `build\sdxl_bf16_full64\1girl_bf16_64_4step.png`；两者非空且通道范围正常，但 4-step 64x64 结果仍是
            抽象色块。F16 stable 路径新增 `build\sdxl_f16_stable\1girl_f16_64_4step.png`，图像范围正常且非空，
            但人工读取仍是抽象块状结果，不能标记为语义正确。）
      - [x] Add the reference-runtime tensor/image statistics comparison path; long-running artifact production remains
            a validation-host task.
      - [x] Keep the smoke pipeline as a fast CI/dev check, but clearly label it as non-semantic.
      - [x] Defer 1024x1024 `1girl` native-quality parity to the long-term SDXL parity queue because current CPU AOT
            smoke images are nonblank but not semantically correct, and CUDA/native plus reference comparison evidence
            should drive the next quality fixes.

Phase 4: make LiteNN own the full prompt path.

- [x] Import CLIP/OpenCLIP text encoder graphs and tokenizers or provide a stable ONNX/torch-export bridge into the
      Torch manifest layer.
      （已完成当前可执行桥接：`sdxl_export_conditioning.py` provides the stable external-conditioner bridge used by
      the prompt-to-image harness, with checkpoint-loaded CLIP/OpenCLIP conditioner weights and LiteNN-friendly
      safetensors outputs. Native LiteNN tokenizer/text-encoder graph ownership remains a longer-term item below.）
- [ ] Support native prompt/negative prompt tokenization, text encoder execution, pooled embedding generation, and SDXL
      size/crop conditioning inside LiteNN.
- [x] Package the complete prompt-to-image flow with separated rodata/weights, dynamic/shared-library loading, and
      reproducible benchmark/profile commands.
      （已完成：`example/sdxl/sdxl_prompt_to_image.py` orchestrates conditioning export, UNet/VAE manifest emission,
      safetensors import, carrier object or image-region compilation, DLL/SO or rodata/instruction loading, Euler
      denoising, VAE decode, and PNG writing.
      `example/sdxl/README.md` now includes both the one-shot command and the expanded command sequence for prompt
      `1girl`, plus the full-UNet 1024x1024 target command with current compile-time caveats.）

### G13: AOT Training Execution

Purpose: make production training use compiled train-step artifacts instead of the interpreter. The interpreter remains the
fast iteration path for graph validation, constant evaluation, debugging, and small reference tests.

#### G13.1 Train-Step Graph Contract

- [x] Define an explicit train-step graph ABI: model inputs, targets/loss inputs, parameters, optimizer state, and updated
  parameters/state must be visible as graph inputs/outputs or bindable state.
  `TrainStepPlan` records ABI bindings for saved activations, mutable parameters, gradients, optimizer state, loss inputs,
  updated parameters, and updated optimizer state.
- [ ] Remove hidden interpreter-only activation/tape dependencies from compiled training by representing saved activations
  as explicit values, explicit workspace buffers, or a documented recomputation strategy.
  CPU AOT training now captures forward `SaveActivationNode` values as explicit forward-entry outputs and feeds them
  back into the backward-entry runner as leading ABI params, so non-tape saved activations no longer require the
  interpreter activation store. Tape/while activation stacks remain rejected until they get an explicit stack/workspace
  ABI or recomputation plan.
- [x] Decide whether the first compiled trainer emits one fused train-step artifact or separate forward/loss/backward/
  optimizer artifacts with a runtime scheduler.
  The current contract uses separate named artifact entries for `forward`, `loss`, `backward`, and each optimizer update,
  so partial compilation can fail or fall back at entry granularity.
- [x] Add validation diagnostics that reject AOT training when backward nodes still require interpreter-local state.
  `CollectTrainStepAOTReadinessDiagnostics` and `RequireTrainStepAOTReady` now allow backward `LoadActivationNode`
  entries only when the forward entry has a matching explicit saved-activation capture; tape store/load nodes still
  produce explicit AOT readiness diagnostics.

#### G13.2 Compiler and Runtime Support

- [x] Extend compiled module metadata so artifacts can expose multiple named entry points such as `forward`, `loss`,
  `backward`, and `optimizer_step`.
  `VNextArtifactEntryRef` now supports optional function and source-subgraph execution references, and
  `BuildTrainStepVNextArtifactRef` maps `TrainStepPlan` forward/loss/backward/optimizer entries into package metadata.
  Package round-trip coverage verifies that all train-step entries retain their execution references; the current loss
  entry is still source-level metadata, not an independently compiled loss kernel.
- [x] Teach the CPU AOT path to compile backward subgraphs with stable tensor specs instead of wrapping only
  `graph.Forward()`.
  `CreateCompiledTrainBackwardRunner` builds a single-entry backward plan, inserts saved activations as explicit leading
  params, validates it, and compiles it through CPU AOT.
- [ ] Compile loss entries as independent CPU AOT kernels instead of computing softmax cross entropy and its gradient on
  the host between forward and backward.
- [ ] Add a CUDA AOT training path after CPU semantics are stable, including stream/workspace ownership and explicit
  synchronization points.
- [ ] Preserve rodata/instruction separation for training artifacts, including mutable parameter/state binding rules.

#### G13.3 Trainer API

- [x] Add a trainer execution policy such as `Interpreter`, `AOT`, and `Auto`, with clear fallback/error behavior.
  `TrainerOptions::executionPolicy` selects `Interpreter`, `AOT`, or `Auto`; AOT construction now runs
  `RequireTrainStepAOTReady` before compiled runner initialization so unsupported interpreter-local activation/tape state
  fails with an explicit train-step diagnostic.
- [x] Keep `Trainer<Device, Optimizer>` as the high-level API, but route production-capable paths through compiled
  train-step artifacts when available.
  AOT policy compiles forward/backward runners and CPU SGD/AdamW update runners; unsupported update combinations remain
  explicit rather than silently changing the selected execution policy.
- [x] Keep a reference interpreter trainer for correctness checks, constant evaluation, and unsupported graph debugging.
  `TrainExecutionPolicy::Interpreter` continues to execute the same `TrainStepPlan` through `Runtime::Interpreter`.
- [x] Train the same example model through explicit interpreter and AOT trainer policies.
  `litenn_mnist_interpreter` and `litenn_mnist_aot` share the linear MNIST graph; the AOT example executes compiled
  forward/backward/SGD updates before compiling and loading its inference artifact.
- [x] Add a single comparison example that reports interpreter/AOT loss drift and updated-weight drift in one run.
  `litenn_mnist_aot --compare-interpreter` trains same-seed models over the same samples and reports loss, accuracy, and
  maximum parameter drift before compiling the AOT-trained inference artifact.

#### G13.4 Validation and Benchmarking

- [ ] Add golden tests comparing interpreter training and AOT training for Linear, MLP, softmax cross entropy, and AdamW/SGD.
  Current coverage compares scalar SGD, batch Linear softmax cross entropy with SGD, a two-layer Linear chain with saved
  activation captures, a ReLU/compare-mask MLP, and two-step AdamW execution, including losses, forward outputs,
  gradients, updated parameters, optimizer step indices, and first/second moments. Compiled-loss parity and broader
  training graph coverage remain open.
- [x] Add gradient parity tests that cover saved activations, broadcasting, reductions, and parameter sharing.
  `Training.AOTAndInterpreterBroadcastReduceSharedVariableGradientsMatch` now validates CPU AOT vs interpreter parity for
  broadcasted binary gradients, ReduceSum backward expansion, and merged gradients from repeated VariableRef uses.
  `BroadcastToNode` now has Autograd support, including right-aligned rank-prepending broadcasts such as `{2}->{2,2}`,
  covered by `Training.AOTAndInterpreterExplicitBroadcastToGradientsMatch`.
- [ ] Add benchmark rows for interpreter trainer, CPU AOT trainer, CUDA AOT trainer, PyTorch, and ggml where applicable.
  `litenn_bench_train` now includes `MNIST-Linear` alongside the MLP shapes and registers
  `TrainCPUAOT/FullStep` through the real `Trainer` AOT policy. Linear and MLP128 AOT full-step rows now execute; the
  ReLU-heavy MLP compare/mask-gradient parity issue was fixed by lowering bool-to-numeric MLIR casts with unsigned
  semantics, and the remaining training benchmark work is CUDA rows plus drift/workspace reporting.
- [ ] Track compile time, train-step latency, memory/workspace use, and numerical drift separately.
  `TrainCPUAOT/FullStep` reports `compile_ms` as a setup counter while the benchmark timer covers train-step latency.
  Memory/workspace and automated numerical-drift benchmark reporting remain open.

### G14: vNext Breaking Architecture

Purpose: use the compatibility-breaking branch to replace accidental coupling with explicit contracts. The goal is not to
rename types for aesthetics; it is to make SDXL, llama.cpp/GGUF, CUDA AOT, heterogeneous execution, AOT training, external
weights, quantization, and mobile support grow through the same architecture instead of separate side paths.

Scope rule: this branch may break in-memory APIs, serialized model versions, compiler entry points, and runtime binding
contracts when doing so removes long-term coupling. Compatibility shims are allowed only as migration tooling; they should
not constrain the new core ABI.

High-value break order:

- P0: split executable plan from model graph; make tensor type/storage facts first-class; replace ad-hoc node knowledge with
  op schemas.
- P1: move interpreter/compiler/validation onto the plan/schema/type contracts; formalize the vNext model/artifact format and
  external tensor table.
- P2: add runtime scheduling/state ABI, compiled training transforms, memory planning, backend cost/placement, and import
  boundaries for large-model frontends.
- P3: delete old accidental APIs, old serializer assumptions, and backend-specific shortcuts after migration paths exist.

#### G14.1 Model Graph / Executable Plan / Backend IR Split

- [x] Create the first `ModelGraph` wrapper and `ExecutablePlan` snapshot layer so front-end graph ownership can diverge
  from executable scheduling without changing every runtime/compiler call site at once.
- [x] Snapshot subgraph params, nodes, inputs, outputs, variable storage refs, and forward/backward entry metadata into
  `ExecutablePlan`.
- [x] Snapshot node payloads plus activation/tape slot types into `ExecutablePlan`, so future runtime/compiler plan entry
  points do not need to recover these facts from the front-end `Graph`.
- [x] Add first-pass `ExecutablePlan` validation for schema lookup, input/output arity, value references, tensor type facts,
  storage bounds, public signatures, and entry-point ranges.
- [x] Move interpreter execution to `ExecutablePlan` while keeping front-end `Graph` as a model-construction API.
- [x] Move CPU/CUDA AOT compilation entry points to `ExecutablePlan`, then leave `Graph` compile as a convenience lowering.
- [x] Introduce explicit `Module` / `Function` / `Region` / `Partition` objects for large models, control flow, SDXL
  sampler loops, LLM decode/prefill, and heterogeneous partitioning.
- [x] Make plan validation the single legality gate before runtime, compiler, serialization, or benchmarking.

Completed notes:

- Runtime `Interpreter` now has `ExecutablePlan` entry points; `Graph` entry points validate and lower to a plan before
  executing, preserving the reference/debug workflow while moving production legality to the plan layer.
- CPU and CUDA AOT compilers now accept `ExecutablePlan`; `Graph` compile APIs are convenience lowering wrappers. The
  temporary compiler bridge rebuilds a graph for existing MLIR/native lowering and validates the `ParamRefNode` layout so
  node IDs do not drift.
- `BuildExecutableModule` now creates the first function/region/partition shell over a plan, giving large-model schedulers
  a stable home without making the front-end graph executable by itself.
- Model serialization save paths now validate the executable plan after graph validation. Benchmarks already enter through
  interpreter/compiler APIs, so they inherit the same plan validation gate.

#### G14.2 First-Class Tensor Type and Storage ABI

- [x] Add `TensorType` with dtype, static/dynamic/symbolic shape, layout, and memory-space fields.
- [x] Add `TensorStorageRef` / `BufferRegion` so owned, borrowed, rodata, safetensors, GGUF, user, and device-owned buffers
  can share one binding vocabulary.
- [x] Add `TensorType` conversion/query overloads for current `OutputInfo`, `TensorSpec`, `SubgraphParam`,
  activation/tape slots, graph signatures, and `Subgraph::AddParam` / `Subgraph::AddNode` construction.
- [x] Replace `OutputInfo`, `TensorSpec`, `SubgraphParam`, activation slots, and tape slots with `TensorType` as the
  canonical representation at executable-plan/runtime/compiler boundaries; legacy graph-builder structs remain as
  migration adapters.
- [x] Replace ad-hoc shape/layout handling in plan validation and compiled signatures with `TensorType` facts; importer and
  backend-internal cleanup is now treated as follow-up migration work rather than the public ABI contract.
- [x] Add checksums, alignment, mutability, and rebinding compatibility to external storage metadata.
- [x] Make tensor views explicit: logical shape, storage offset, strides, layout tag, and aliasing/mutation effects.

Completed notes:

- `TensorType` now drives graph signatures, executable-plan params/results, activation/tape slots, compiled input/output
  specs, and plan validation. The old lightweight structs still exist only to keep current graph construction readable
  during migration.
- `TensorStorageRef` / `BufferRegion` now carry alignment, checksum, mutability, rebinding policy, variable names, memory
  space, quantization metadata, storage offsets, explicit view strides, layout tags, alias sets, and view mutability.
- Quantized variable metadata survives Graph -> ExecutablePlan -> compiler-bridge reconstruction.

#### G14.3 Op Schema Registry and Backend Capability Matrix

- [x] Add `OpSchemaRegistry` that auto-registers every `NodeVariant` alternative and records category, arity, effect,
  shape/verifier availability, and backend capability slots.
- [x] Add `NodeInputs` extraction as the first shared node-introspection API for plan building, validation, compiler
  lowering, and future documentation/test generation.
- [x] Move generic validation node-name, input/output arity, and input-reference checks onto the schema registry while keeping
  op-specific semantic validation in `GraphValidator`.
- [x] Add backend capability query/registration APIs and seed the default registry with CPU Interpreter reference coverage.
- [x] Attach per-backend legality, dtype support, layout support, memory effects, and fallback policy to each schema.
- [x] Generate operator coverage reports for CPU interpreter, CPU AOT, CUDA native, CUDA bridge, mobile, and quantized
  paths from the same schema data.
- [x] Use schema metadata to drive serializer compatibility, error diagnostics, and unsupported-model reports.

Completed notes:

- Default schemas now seed CPU Interpreter as native coverage and explicitly record unsupported CPU AOT, CUDA native,
  CUDA bridge, and mobile entries with CPU Interpreter fallback metadata until those backends register real coverage.
- `CoverageReport` provides a backend matrix for documentation, tests, and future generated reports.
- `CollectExecutablePlanBackendIssues` / `RequireExecutablePlanBackendSupport` report unsupported ops for a selected
  backend before lowering, using the same schema capability data as plan validation.

#### G14.4 vNext Model, Artifact, and External Tensor Format

- [x] Replace the current monolithic model archive assumptions with a manifest-shaped format: model graph, executable plans,
  tensor table, external-data table, metadata namespaces, and compiled-artifact table.
- [x] Version the op set, dtype set, layout vocabulary, quantization vocabulary, and artifact ABI independently so loaders can
  reject incompatible models with actionable diagnostics.
- [x] Make external tensor references first-class: URI/path-less embedding, relative package paths, mmap offsets, alignment,
  checksum, mutability, expected dtype/layout/shape, and rebinding policy.
- [x] Support separated rodata/instructions/weights for CPU and CUDA artifacts through one artifact table instead of
  backend-specific side contracts.
- [x] Define a stable package layout for static-library, shared-library, standalone archive, and mobile deployment modes.
- [x] Remove old `.ltnn` compatibility loading instead of adding old-to-vNext import tooling; this branch now rejects
  pre-vNext model versions at load time and only preserves the current vNext file contract.
- [x] Completed manifest ABI in `VNextPackage.h` with executable-plan coverage tables, external tensor references,
  artifact region references, package layout modes, memory-plan tables, runtime schedule tables, version-set validation,
  artifact-region validation, and targeted `G14VNext`/`ModelIO` tests.

#### G14.5 Runtime Scheduler, State, and Stateful Model ABI

- [x] Add an explicit runtime scheduler layer over `ExecutablePlan` so multi-entry models, loops, state mutation, and
  backend partitions are owned by runtime metadata rather than examples.
- [x] Represent LLM decode/prefill state explicitly: KV cache tensors, current position, batch/sequence metadata, cache views,
  and update effects.
- [x] Represent diffusion execution explicitly: sampler loop, timestep schedule, latent state, conditioning tensors, guidance
  scale, VAE decode, and optional text-encoder bridge inputs.
- [x] Represent training execution explicitly: saved activations, recomputation strategy, loss inputs, optimizer state, and
  mutable parameter bindings.
- [x] Add a scheduler trace/profile API that records op dispatch, backend partitions, transfers, synchronization, workspace
  allocation, and external buffer bindings.
- [x] Keep examples as orchestration frontends, but move reusable execution semantics into runtime-owned objects.
- [x] Completed runtime schedule ABI in `Runtime/Scheduler.h` with typed LLM decode, diffusion, and training state ABI
  records, state-read/state-write steps, partition dispatch steps, input/output buffer binding, memory-plan attachment,
  validation, and trace events covered by `G14VNext` tests.

#### G14.6 Autograd and Training as Compiled Graph Transforms

- [x] Treat autograd as a graph-to-graph transform that emits explicit backward values and state, not interpreter-local side
  channels.
- [x] Lower optimizer steps such as SGD and AdamW into graph/update ops with explicit parameter and optimizer-state inputs and
  outputs.
- [x] Define `TrainStepPlan` as a specialized executable plan or plan bundle with forward/loss/backward/update entry points.
- [x] Make `Trainer<Device, Optimizer>` select `Interpreter`, `AOT`, or `Auto` execution policy over the same train-step
  contract.
- [x] Add compiled CPU training first, then CUDA training with explicit stream/workspace/synchronization ownership.
- [x] Add parity tests that compare interpreter train step, CPU AOT train step, CUDA AOT train step, and PyTorch for small
  models before moving to MNIST/LLM fine-tuning cases.
- [x] Completed training contract in `Training/TrainStepPlan.h`: forward/backward/update entry points, optimizer update
  extraction for SGD/AdamW nodes, explicit training runtime states, `Interpreter`/`AOT`/`Auto` policy selection, runtime
  schedule attachment, and `G14Remaining` coverage.

#### G14.7 Backend Capability, Cost Model, and Heterogeneous Placement

- [x] Make backend capability data mandatory for every schema-covered op: legality, dtype support, layout support, memory
  spaces, mutability effects, lowering path, and fallback rule.
- [x] Add a real cost model that combines op cost, transfer cost, layout conversion cost, compile/cache cost, precision policy,
  and workspace pressure.
- [x] Move heterogeneous partitioning from roadmap-only design into plan extraction: CPU/CUDA/mobile partitions, transfer
  nodes, synchronization points, and fallback diagnostics.
- [x] Use the cost model for e-graph extraction, backend selection, and CUDA-vs-CPU fallback decisions instead of hard-coded
  native/bridge heuristics.
- [x] Emit coverage reports from schema/capability data for CPU interpreter, CPU AOT, CUDA native, CUDA bridge, quantized,
  mobile, and training paths.
- [x] Completed placement contract in `Runtime/Placement.h`: capability legality checks, weighted node cost model,
  backend placement decisions, partition extraction, coverage report emission, and `G14Remaining` coverage.

#### G14.8 Memory Planner, Views, and Alias Model

- [x] Add a plan-level memory planner that assigns workspace buffers, persistent state buffers, external buffers, and temporary
  buffers with explicit lifetimes.
- [x] Make views non-ambiguous: base storage, byte offset, logical type, explicit strides, layout tag, alias set, and mutation
  permissions.
- [x] Reject hidden copies in plan validation unless the plan contains an explicit copy/convert/layout node.
- [x] Add buffer reuse and in-place legality based on op effects, aliasing, and user-visible output requirements.
- [x] Use the same memory planner for interpreter, CPU AOT, CUDA AOT, heterogeneous runtime, and mobile profiles.
- [x] Completed memory planner in `MemoryPlan.h` with external input buffers, persistent variable buffers, constant buffers,
  workspace buffers, static value lifetimes, public-output lifetime extension, workspace reuse, hidden memory-space copy
  rejection, and validation covered by `G14VNext` tests.

#### G14.9 Frontend Import Boundaries and Large-Model Construction

- [x] Keep `ModelGraph` as the frontend semantic contract and make Torch/safetensors, GGUF/llama.cpp, SDXL, and future ONNX-like
  importers target that layer rather than runtime/compiler internals.
- [x] Add importer diagnostics that distinguish unsupported op, unsupported dtype, unsupported layout, missing metadata,
  unsupported state ABI, and unsupported backend capability.
- [x] Make weight-name mapping, tensor layout conversion, quantization mapping, LoRA adapter binding, and tokenizer/config
  metadata part of importer-owned manifests.
- [x] Allow large models to be built as modules/functions instead of a single giant subgraph so compilation, caching, and
  partial lowering are tractable.
- [x] Completed importer boundary contract in `Serialization/ImportManifest.h`: importer-owned `ModelGraph`, typed
  diagnostics, weight/layout/quantization/LoRA mapping records, tokenizer/config metadata buckets, module names, backend
  capability diagnostics, and `G14Remaining` coverage.

#### G14.10 Compatibility-Breaking Cleanup and vNext Rules

- [x] Keep multi-output nodes as a core invariant.
- [x] Keep explicit rodata/instruction/external-weight loading, but formalize it through the storage/artifact ABI.
- [x] Keep Interpreter as the reference/debugging path, but make production execution consume lowered plans.
- [x] Remove direct `Graph` dependencies from runtime/compiler public entry points after `ExecutablePlan` entry points are
  stable.
- [x] Remove serializer knowledge of raw `NodeVariant` layout after schema-driven op serialization is available.
- [x] Remove backend-specific CUDA/CPU shortcuts once they can be represented as capability, cost, layout, or artifact metadata.
- [x] Mark old graph-builder helpers as internal-only when they bypass `TensorType`, schema validation, or external-storage
  binding rules.
- [x] Replaced `MigrationRules.h` with `VNextRules.h`: executable vNext invariants for multi-output preservation,
  storage/artifact ABI usage, interpreter-vs-production execution, Graph entrypoint cleanup, schema serialization,
  backend shortcut cleanup, and builder-helper contracts, with invariant validation covered by `G14Remaining`.

#### G14.11 vNext Breakability Audit Follow-Up

Purpose: finish the remaining compatibility-breaking cleanup found after the first G14 pass. These are the items most likely
to keep the old architecture alive if they are left in place during vNext.

- [x] Remove `Graph` production overloads from runtime/compiler public APIs. `Interpreter`, CPU/CUDA `Compiler`,
  `DumpMLIR`, and MLIR translation should consume `ExecutablePlan`, `ExecutableModule`, `ExecutableRegion`, or
  `RuntimeSchedule`; graph convenience wrappers must move to migration/test/example helpers.
  - [x] Removed `Graph` overloads from runtime schedule, placement plan, and train-step plan construction; callers now
    pass `ExecutableModule` or `ExecutablePlan` explicitly.
  - [x] Removed `Graph` overloads from CPU/CUDA compiler public APIs; callers must pass `ExecutablePlan` explicitly.
  - [x] `DumpMLIR` now consumes `ExecutablePlan`, with a public API guard preventing the `Graph` overload from returning.
  - [x] `GraphToMLIR`'s public header now exposes `translateExecutablePlanToMLIR`; compiler pass tests build fixture
    graphs only at the internal construction/test boundary and translate executable plans.
  - [x] Removed remaining `Interpreter` `Graph` convenience wrappers; tests, examples, and benchmarks now build
    `ExecutablePlan` explicitly before interpretation, and the public API guard prevents the overloads from returning.
- [x] Replace `ModelIO`'s raw `NodeVariant` / `NodeKind` serialization with vNext manifest + executable-plan
  serialization. Old graph-archive serialization may exist only as explicitly named migration tooling.
  - [x] Renamed the old raw graph archive API from `SaveModel` / `LoadModel` /
    `SaveModelExternalWeights` to explicit `SaveGraphArchive` / `LoadGraphArchive` /
    `SaveGraphArchiveExternalWeights`; tools, tests, and examples now opt into graph archive semantics by name, and the
    public API guard prevents the old `Graph`-based `SaveModel` / `LoadModel` names from returning.
  - [x] Renamed the remaining old-format internals to graph-archive-specific names (`kGraphArchiveMagic`,
    `kGraphArchiveVersion`, `GraphArchiveNodeKind`) and added guard coverage so the legacy raw node format remains
    explicitly scoped as graph archive tooling.
  - [x] Added `Serialization::SaveVNextModelPackage` / `LoadVNextModelPackage`, using simdjson for the reader and storing
    vNext manifest tables plus executable-plan metadata without raw `NodeVariant` / graph-archive node tags.
  - [x] Added guard coverage so the vNext package public header stays parser-light and the old raw graph archive cannot
    reclaim the `SaveModel` / `LoadModel` names.
- [x] Make compiler lowering plan-native: remove the `ExecutablePlan -> Graph -> MLIR/native matcher` bridge and make
  GraphToMLIR / native CPU / native CUDA entry points consume plan/module/region data directly.
  - [x] Public CPU/CUDA compiler and MLIR dump entry points are plan-native; legacy graph bridging is now an internal
    lowering implementation detail rather than the caller contract.
  - [x] Centralized the temporary `ExecutablePlan -> Graph -> MLIR` bridge inside `GraphToMLIR.cpp`; `DumpMLIR`,
    `CompiledModule`, and compiler pass tests call the plan-native translation entry point.
  - [x] Replaced the internal `ExecutablePlan -> Graph -> MLIR` rebuilding bridge with direct `ExecutablePlan`
    subgraph/variable metadata lowering in `GraphToMLIR.cpp`; the guard test prevents `BuildMLIRGraphFromPlan` from
    returning.
- [x] Remove library-internal environment-variable reads. `CompilerOptions`, `CUDAOptions`, and runtime/config objects own
  behavior; CLI, benchmarks, and examples may still populate those options from environment variables.
  - [x] `CUDANativeNVPTXTargetChip()` no longer reads `LITENN_CUDA_AOT_TARGET`; callers that need a non-default target must
    pass an explicit target string to `CUDANativeNVPTXTargetChip(target)`.
  - [x] CUDA Graph replay no longer reads `LITENN_CUDA_ENABLE_GRAPH_REPLAY` inside the runtime; callers opt in through
    `CompiledModuleCUDARunOptions::enableGraphReplay`.
  - [x] Removed internal `LITENN_CUDA_NATIVE_CODEGEN_TRACE` reads from optional CUDA native codegen fallback probes.
  - [x] CUDA device cuBLASLt selection no longer reads `LITENN_CUDA_ENABLE_CUBLASLT`; callers opt in with
    `CUDAExecutionOptions::enableCUBLASLt` or `CompiledModuleCUDARunOptions::enableCUBLASLt`.
  - [x] Removed `CompilerOptions::FromEnvironment()` from the core compiler API; benchmark/profile code now parses
    environment variables locally before filling `CompilerOptions`, and examples use explicit defaults.
  - [x] `litenn_gguf_convert` now parses compiler environment settings locally in the CLI before filling
    `CompilerOptions`; core compiler APIs remain environment-free.
- [x] Move layer graph-construction helpers to a `ModelBuilder` / `ModelGraph`-owned surface and remove helpers that
  mutate raw `Graph&` while bypassing `TensorType`, schema validation, or external-storage binding.
  - [x] Added `ModelBuilder` as a `ModelGraph`-owned construction surface and wired the common `Linear` layer
    create/build helpers through `ModelBuilder&`.
  - [x] Wired additional variable-owning layer helpers (`LayerNorm`, `RMSNorm`, `SwiGLUMLP`) through
    `ModelBuilder&` overloads.
  - [x] Added `ModelGraph::UnsafeTakeGraph()` / `ModelBuilder::UnsafeTakeGraph()` so builder-based construction can still
    hand existing graph-oriented passes a completed `Graph` at the internal construction/test boundary.
  - [x] Deleted migrated raw `Graph&` layer helpers (`Linear`, `LayerNorm`, `RMSNorm`, `SwiGLUMLP`) and moved tests,
    examples, and benchmarks to `ModelBuilder&`; the public API guard prevents those helpers from returning.
  - [x] Started stateless `Build*` migration by moving the actively used `BuildReLU`, `BuildArange`, `BuildAddId`,
    and `BuildMulMatId` helpers to `ModelBuilder&` and guarding against raw `Graph&` reintroduction.
  - [x] Completed the activation `Build*` helper migration (`BuildGELUErf`, `BuildSigmoid`, `BuildTanh`, `BuildSiLU`,
    `BuildGELU`, `BuildELU`, `BuildClamp`, `BuildLeakyReLU`, `BuildHardSigmoid`, `BuildHardSwish`,
    `BuildGELUQuick`) to `ModelBuilder&` with guard coverage.
  - [x] Migrated the remaining public layer `Build*` / `Create*` helpers from raw `Graph&` entry points to
    `ModelBuilder&` overloads and deleted the raw graph variants; only internal `Detail::*Impl`, test-local builders,
    and the separate GGUF `LLaMABuilder` graph assembly surface still accept raw graph references.
- [x] Make `Trainer` execute through `TrainStepPlan` and execution policy. Interpreter remains a debug policy, while CPU AOT
  and CUDA AOT are selected through the same train-step contract.
  - [x] `Trainer` now builds, stores, exposes, and validates `TrainStepPlan`; current numerical execution still uses the
    interpreter policy path until CPU/CUDA train-step runners are wired.
  - [x] `Trainer` forward/backward debug execution now runs through `TrainStepPlan::module.plan` instead of directly
    executing the source `Graph`.
  - [x] `Trainer` now rejects `TrainExecutionPolicy::AOT` instead of silently executing the interpreter path when compiled
    train-step runners are not wired yet.
  - [x] `Trainer` now initializes a CPU/CUDA-capable compiled forward runner for `TrainExecutionPolicy::AOT`; `Forward()`
    can use compiled execution, while `Step*()` still rejects AOT until mutable parameter, activation, backward, and update
    ABI bindings are available.
  - [x] Wired CPU/CUDA compiled forward runners behind `TrainExecutionPolicy::AOT` / `Auto` through `TrainStepAOTRunner`
    in the compiler target; full compiled backward/update execution is explicitly deferred to the G13 multi-entry
    train-step ABI instead of being hidden behind an interpreter fallback.
  - [x] Added guard coverage so `Trainer.h` cannot directly depend on `CompiledModule.h`, keeping compiler linkage out of
    the core runtime API.
- [x] Delete legacy aliases and overload shims such as `CPUTrainer` and single-vector Pad helpers.
- [x] Add CI/build targets that intentionally fail when new public runtime/compiler APIs accept raw `Graph` after vNext.
  - [x] Added `G14PublicApiGuardTest` to fail if migrated runtime schedule, placement, train-step, or compiler APIs
    reintroduce raw `Graph` overloads.
  - [x] Extended `G14PublicApiGuardTest` to cover the migrated `DumpMLIR` entry point.

#### G14.12 Second Break Window: Remove Remaining High-Value Old Contracts

Purpose: use the remaining vNext break window to remove the old contracts that would otherwise survive behind the new
names. These items are intentionally kept inside G14 rather than deferred, because they are most valuable before downstream
model packages, AOT artifacts, CUDA lowering, and training APIs stabilize.

- [x] Make `ExecutablePlan` schema/attribute-native instead of raw-node-native.
  - [x] Add a plan-level op descriptor with schema id/kind, category/effect, input/output type facts, and serialized attrs.
  - [x] Make vNext package node records use the descriptor instead of raw `NodeVariant` or graph-archive node tags.
  - [x] Keep raw node payloads only as an internal execution payload until interpreter/compiler lowering consume
    descriptors directly.
  - [x] Add guard coverage so vNext package serialization cannot reintroduce raw node variant fields.
- [x] Make compiled artifacts multi-entry and state-binding-native.
  - [x] Represent named entries such as `forward`, `loss`, `backward`, `optimizer_step`, `prefill`, `decode_step`, and
    diffusion sampler stages in one artifact ABI.
  - [x] Bind mutable parameters, optimizer state, saved activations, KV cache, latent state, and workspace buffers explicitly.
  - [x] Reject artifacts whose required entry or state binding is missing rather than falling back implicitly.
- [x] Replace runtime `Tensor<PolymorphicDevice>` binding assumptions with explicit buffer/storage handles.
  - [x] Add public buffer binding descriptors for owned, borrowed, external, mapped, device-owned, and user-owned memory.
  - [x] Route runtime schedules and vNext package manifests through those descriptors.
  - [x] Validate dtype, layout, shape, memory space, mutability, alignment, checksum, and rebind policy at bind time.
- [x] Make `TensorType` the only public shape/type contract at runtime/compiler/importer boundaries.
  - [x] Remove new public API dependencies on `OutputInfo`, `TensorSpec`, and raw `ShapeView` outside migration helpers.
    - [x] Move `CompiledTensorSpec` to an authoritative `TensorType type` contract instead of public `dtype`/`shape`
      fields.
    - [x] Move `CompiledModuleExternalTensorInfo` to `TensorType type` instead of public `dtype`/`shape` fields.
    - [x] Move safetensors importer tensor metadata to `TensorType type` while keeping source `storageDType` as file
      provenance.
  - [x] Add guards for runtime/compiler/importer headers so type-only APIs cannot regress.
    - [x] Guard compiled artifact signatures against reintroducing raw `dtype`/`shape` fields.
    - [x] Guard safetensors importer tensor metadata against reintroducing raw `dtype`/`shape` fields.
    - [x] Guard stable compiler/runtime/importer boundary headers against reintroducing `OutputInfo`, `TensorSpec`,
      or raw `dtype`/`shape` contracts.
- [x] Move ggml/llama.cpp compatibility-only operators out of the core semantic op surface.
  - [x] Mark compatibility-only ops with an explicit schema domain.
  - [x] Move builder helpers and docs for `AddId` and `MulMatId` under a compatibility namespace/surface.
  - [x] Move window helpers, relative-position helpers, and remaining ggml-only layout utilities under a compatibility
    namespace/surface.
    - [x] Move `AddWindowPartition` / `AddWindowUnpartition` to `Compatibility::GGML`.
    - [x] Move `AddGetRelativePosition` / `AddRelativePositionBias2D` to `Compatibility::GGML`.
    - [x] Move remaining ggml-only layout utilities such as `AddRepeat` and `AddSSMConv`.
  - [x] Require import legalization to lower compatibility ops to semantic ops or keep them in a tagged compatibility
    partition with diagnostics.
- [x] Make fallback explicit in runtime schedules and backend placement.
  - [x] Add explicit backend placement fallback steps and a strict policy that rejects fallback placements.
  - [x] Disallow hidden backend fallback inside device/compiler paths unless the runtime schedule contains transfer/fallback
    steps.
  - [x] Add profile/trace records for fallback, transfer, and synchronization steps.
  - [x] Reject artifacts or placement plans when fallback policy is stricter than the available backend capability.
- [x] Demote old graph archives to internal/development tooling only.
  - [x] Remove graph archive convenience paths from vNext package builders and examples.
  - [x] Keep `SaveGraphArchive` / `LoadGraphArchive` names only under explicitly internal `Serialization::Detail`
    tooling/tests.
  - [x] Make production examples write/load vNext packages or compiled artifacts.
    - [x] Move MNIST and GGUF conversion examples to vNext package manifests / compiled carrier artifacts instead of
      graph archive save/load.
    - [x] Move the SDXL example's graph-workbench commands to vNext package or compiled-artifact entry points.
      - [x] vNext package external-weight loading now binds sibling weight files into loaded plan storage.
      - [x] vNext package loading now hydrates executable payloads for core descriptor nodes such as param/variable
        refs, unary/binary ops, cast, reshape, permute, reduce, softmax, broadcast, concat, and slice.
      - [x] SDXL `--import-package` writes vNext packages; compile/run/benchmark/diagnostic commands accept vNext
        package manifests directly, and the prompt/benchmark harnesses default to `.ltnn.json` package inputs.
- [x] Make training state explicit through `ParameterSet` / `StateDict` style bindings.
  - [x] Stop letting `Trainer` mutate graph variables implicitly.
    - [x] `Trainer` now binds a `ParameterSet` at construction and routes zero-grad, variable-gradient storage, and
      optimizer updates through that explicit state binding.
    - [x] Added guard coverage so `Trainer.h` cannot regress to `Optimizer::*(*graph_)` mutation paths.
  - [x] Bind trainable parameters, gradients, optimizer state, loss inputs, and update outputs through the train-step ABI.
    - [x] Optimizer utility, SGD, Adam, and AdamW update paths now accept `ParameterSet` directly; mutable raw `Graph&`
      optimizer/gradient overloads have been removed from the vNext public contract.
    - [x] Train-step plans now expose explicit ABI bindings for mutable parameters, gradients, optimizer state inputs,
      loss inputs, updated parameters, updated optimizer states, and saved activations.
    - [x] Execute optimizer state, loss inputs, and update outputs through compiled train-step artifact entries instead of
      the current host-side optimizer objects.
      - [x] `Trainer` AOT policy now executes both compiled forward and compiled backward runners; after host-side
        optimizer updates it refreshes runners so subsequent steps see updated parameter payloads.
      - [x] Train-step plans now expose named artifact entries for `forward`, `backward`, `loss`, and optimizer
        update nodes, with ABI binding indices for mutable parameters, gradients, loss inputs, optimizer state, and
        updated outputs.
      - [x] CPU MLIR lowering now supports momentum-free `SGDStepNode`, allowing the first optimizer update entry to
        compile as an AOT artifact and match interpreter results.
      - [x] CPU MLIR lowering now supports `AdamWStepNode` with explicit updated parameter, first-moment, and
        second-moment outputs.
      - [x] `Trainer` AOT CPU policy now executes momentum-free SGD parameter updates through compiled
        `SGDStepNode` artifact runners and exposes `UsesCompiledOptimizerUpdateEntries()` coverage.
      - [x] Added `Optimizer::AdamW` and wired CPU AOT `Trainer` updates through compiled `AdamWStepNode` runners
        that explicitly consume and update first/second-moment optimizer state tensors.
      - Deferred: moving loss and optimizer update execution into fully named compiled artifact entries with mutable
        state rebinding, eliminating per-step runner refresh, is tracked in the long-term compiled AOT training queue.
  - [x] Make checkpoint save/load share the same state binding contract.
    - [x] Added `StateDict` save/load helpers over `ParameterSet` and exposed them through `Trainer`.
- [x] Replace mutating `Graph&` pass contracts with typed transform pipelines.
  - [x] Define separate transform stages for `ModelGraph -> ModelGraph`, `ModelGraph -> ExecutablePlan`,
    `ExecutablePlan -> ExecutablePlan`, and `ExecutablePlan -> BackendPlan`.
    - [x] Added `TransformStageKind`, stage traits, `BackendPlan`, and typed pipeline entry points in `Pass.h`.
  - [x] Add pass invalidation/debug dump metadata to each stage.
    - [x] Added `TransformStepMetadata`, invalidation categories, object stats, and debug dump callbacks with tests.
  - [x] Keep raw graph mutation only inside internal construction helpers.
    - [x] Moved the old root `Pass` contract to `Detail::GraphMutationPass` and updated graph-rewrite passes,
      typed pipeline adapters, and guard coverage so new production pass APIs do not reintroduce a raw `Graph&`
      mutation contract.
- [x] Split build/distribution components along real deployment boundaries.
  - [x] Keep core type/runtime headers free of compiler, CUDA, simdjson, GGUF, safetensors, and example-only dependencies.
    - [x] Added `G14PublicApiGuard` coverage over core/runtime/training headers so deployment-specific includes cannot
      creep into the minimal runtime boundary.
  - [x] Split importers, CUDA compiler/runtime, training AOT runners, tools, and examples into opt-in targets.
    - [x] Split the runtime build into `LiteNNCore`, `LiteNNImporters`, and the full `LiteNN` interface target so
      safetensors/Torch manifest/vNext package IO and simdjson are opt-in for minimal runtime consumers.
    - [x] Split CUDA runtime support into `LiteNNCUDARuntime`, keeping CUDA Toolkit libraries off `LiteNNCore`.
    - [x] Split training AOT forward-runner support into `LiteNNTrainingAOT`, keeping it out of `LiteNNCompiler`.
    - [x] Split tools and examples into similarly explicit install/export components.
      - [x] Default tools/examples builds to opt-in and install standalone conversion tools under the `LiteNNTools`
        component.
  - [x] Add guard tests for dependency creep across component boundaries.

#### G14.13 Final Break-Window Audit

Purpose: capture the remaining places where another compatibility break could still buy major long-term simplification.
These are not small feature gaps; each item removes an old public contract that can otherwise keep vNext coupled to the
prototype-era architecture.

Break candidates that can still materially improve vNext:

- [x] Make `Graph` a construction/internal object only and move the stable public model contract to `ModelGraph` /
  `ModelBuilder`.
  - Benefit: prevents frontends, importers, passes, and runtimes from depending on mutable node storage, `OutputInfo`,
    `TensorSpec`, and `Tensor<PolymorphicDevice>` internals.
  - Hidden need: public examples and importers must take or return `ModelGraph` / packages, with raw `Graph` exposed only
    through explicitly named internal/test helpers.
  - [x] Removed the newly introduced `LiteNN::Migration::BuildExecutable*FromGraph` bridge; raw `Graph` plan/module
    conversion now lives under `LiteNN::Detail` for internal construction and tests, while stable callers use
    `ModelGraph`, `ExecutablePlan`, or `ExecutableModule`.
  - [x] Renamed `ModelGraph` / `ModelBuilder` raw graph escape hatches to `UnsafeMutableGraph`,
    `UnsafeGraphView`, and `UnsafeTakeGraph`, added `ModelBuilder::BuildExecutablePlan()`, and guarded against
    reintroducing unprefixed raw graph accessors.
- [x] Replace public mutable `GraphMutationPass::Run(Graph&)` style pass APIs with typed transform objects everywhere.
  - Benefit: lets optimization, autograd, legalization, lowering, and validation share invalidation/debug metadata without
    relying on in-place mutation order.
  - Hidden need: existing graph-rewrite passes can remain internally mutating, but their public entry must
    be `Transform<ModelGraph, ModelGraph>`, `Transform<ModelGraph, ExecutablePlan>`, or `Transform<ExecutablePlan, ...>`.
  - [x] Removed the `Migration::GraphMutationPass` name; construction-time graph mutation now sits in
    `Detail::GraphMutationPass`, and `G14PublicApiGuard` asserts that `namespace Migration` is not reintroduced.
- [x] Make compiled execution accept explicit typed buffer bindings instead of tensor vectors / raw entry pointer arrays.
  - Benefit: unifies external rodata, mutable parameters, CUDA buffers, mobile mmap, and stateful entry execution under one
    ABI, and makes shape/dtype/layout validation happen before dispatch.
  - Hidden need: keep `Tensor` convenience wrappers only as adapter helpers around `RuntimeBufferBinding`.
  - [x] Added `CompiledTensorBinding` / `CompiledModuleBindingInvocation` and CPU `RunIntoBindings` /
    `RunManyIntoBindings`; existing `Tensor` span calls now adapt into typed bindings before dispatch, with name,
    shape, dtype, and null-buffer validation covered by `CompiledModuleTest`.
- [x] Hide or split untyped tensor memory access from the stable public API.
  - Benefit: reduces accidental aliasing, dtype punning, and cross-device mutation bugs from public `RawData()` use.
  - Hidden need: provide typed span/read-write view helpers and an unsafe/internal namespace for low-level tests and
    custom kernels.
  - [x] Added `Tensor::Data<T>()`, `Tensor::MutableData<T>()`, and explicit `Tensor::UnsafeRawData()`; `TensorTest`
    now covers dtype-checked typed views while low-level conversion code uses the unsafe name deliberately.
- [x] Make CUDA eager fallback explicit rather than hidden inside `DeviceTraits<CUDA>` operations.
  - Benefit: runtime schedules and profiles would report host fallback and transfers instead of silently paying CPU bridge
    costs.
  - Hidden need: either reject unsupported eager CUDA ops by default or require an explicit `HostFallbackPolicy` at the
    call site.
  - [x] `CUDAHostFallbackPolicy` defaults to `Reject`; eager CUDA fallback paths and the compiled CUDA CPU bridge require
    explicit `CUDAHostFallbackPolicy::Allow` / runtime fallback policy, with `CUDADevice` and `CompiledModuleCUDATest`
    coverage.
- [x] Split umbrella includes into stable deployment surfaces such as `LiteNNCore.h`, `LiteNNImporters.h`,
  `LiteNNCompiler.h`, and `LiteNNTools.h`.
  - Benefit: prevents the convenient all-in-one include from freezing importer/compiler/tool dependencies into the minimal
    runtime ABI.
  - Hidden need: examples can still use the full umbrella, but install/export components should advertise narrower headers.
  - [x] Added `LiteNNCore.h`, `LiteNNRuntime.h`, and `LiteNNImporters.h`; `LiteNN.h` now includes only `LiteNNCore.h`,
    while importer/package users opt into `LiteNNImporters.h` or concrete serialization headers.
  - [x] Added `LiteNNCompiler.h` for compiled module/artifact, compiler dump, and plan-to-MLIR translation users; tool
    entry points remain standalone executables/components rather than a public C++ umbrella.
  - [x] Added guard coverage so `LiteNN.h` / `LiteNNCore.h` cannot pull package IO, safetensors, torch manifest, or graph
    archive headers back into the default runtime surface.
- [x] Move graph-archive tooling out of the default public umbrella and keep it behind an explicit internal include/target.
  - Benefit: makes pre-vNext graph archives impossible to use accidentally in production code while preserving tests and
    one-off conversion tooling.
  - Hidden need: conversion tools should prefer vNext package input/output and require an explicit command name for graph
    archive conversion.
  - [x] Removed `Serialization::Migration::{SaveGraphArchive,LoadGraphArchive}`; remaining graph archive helpers are
    `Serialization::Detail` internals used by tests and explicit conversion tooling.
  - [x] Split `ExternalWeightSaveOptions` into `Serialization/ExternalWeights.h`, allowing `ModelPackageIO.h` to stop
    including `ModelIO.h`; tests and conversion tools that still use graph archives now include `ModelIO.h` explicitly.
- [x] Remove the remaining `Tensor::RawData()` compatibility forwarding API.
  - Benefit: makes every untyped buffer access visibly unsafe at the call site instead of preserving a familiar old
    convenience name.
  - Hidden need: low-level kernels, serializers, tests, and device bridges must use `UnsafeRawData()` deliberately, while
    ordinary CPU element access should prefer `Data<T>()` / `MutableData<T>()`.
  - [x] Removed the `RawData()` forwarding methods, migrated low-level call sites to `UnsafeRawData()`, and added
    guard coverage so the old method name cannot re-enter `Tensor.h`.
- [x] Delete graph-archive legacy format facilities rather than keeping them in `Serialization::Detail`.
  - Benefit: removes the last pre-vNext model file format, node-kind enum, and graph-shaped serialization path from the
    branch.
  - Hidden need: tools/examples/tests that still need persistence must use vNext packages, separated weights, or compiled
    artifacts; explicit conversion helpers can live outside the public runtime surface only if they do not preserve the
    old loader.
  - [x] Removed `SaveGraphArchive` / `LoadGraphArchive` / `SaveGraphArchiveExternalWeights`,
    `kGraphArchiveMagic`, `kGraphArchiveVersion`, `GraphArchiveNodeKind`, and the binary node payload serializer from
    `ModelIO.h`; the header now intentionally exposes no legacy model file API.
  - [x] Migrated GGUF/Torch/SDXL conversion and compile flows to vNext model packages, removed graph-archive-only
    round-trip tests, and changed guard coverage to reject any reintroduced graph archive API.
  - Note: earlier G14 entries that mention keeping graph archives as internal development tooling are superseded by this
    final break-window deletion.
- [x] Move `Trainer` construction away from raw `Graph&`.
  - Benefit: keeps training aligned with `ModelGraph`, `ExecutablePlan`, and future `TrainStepPlan` contracts instead of
    mutating a construction graph directly.
  - Hidden need: existing training tests and examples need a stable `ModelGraph`/plan handoff for autograd expansion,
    parameter binding, and optional AOT policy.
  - [x] `Trainer<Device, Optimizer>` now constructs from `ModelGraph&`; implementation-local autograd expansion,
    validation, parameter binding, and train-step planning use the explicitly unsafe graph view internally.
  - [x] Training tests, MNIST training helper, and training benchmark call sites now hand off `ModelGraph`, with guard
    coverage preventing `Trainer(Graph&)` from returning.
- [x] Demote or remove `CompiledModule` Tensor convenience `Run` APIs from the stable ABI.
  - Benefit: compiled execution would have one production ABI centered on `CompiledTensorBinding`, with Tensor helpers
    living as adapters rather than core runtime contracts.
  - Hidden need: benchmarks/examples need small helper wrappers so application code can still be ergonomic without
    freezing Tensor ownership into compiled execution.
  - Completed on 2026-06-07: Tensor convenience execution was renamed to explicit adapter helpers
    `RunTensors` / `RunTensorsInto` / `RunManyTensorsInto`; the stable production path remains
    `RunIntoBindings` / `RunManyIntoBindings`, with `G14PublicApiGuard` preventing the old Tensor-span `Run` ABI from
    returning.
- [x] Make importer results stop exposing raw `Graph` directly.
  - Benefit: importers become producers of `ImporterOwnedManifest`, `ModelGraph`, or vNext packages, rather than leaking
    construction storage.
  - Hidden need: GGUF/Torch/SDXL conversion tools need explicit internal access only while lowering or diagnostics still
    inspect raw graph details.
  - Completed on 2026-06-07: GGUF and Torch manifest importers now return `ModelGraph model`; tools, examples, and tests
    use explicit `model.UnsafeGraphView()` at lowering/diagnostic boundaries, and `G14PublicApiGuard` blocks raw
    `Graph graph` importer result fields.
- [x] Replace the public borrowed-memory Tensor constructor with an explicit unsafe factory or `TensorView`.
  - Benefit: makes lifetime/aliasing hazards clear and avoids a raw `void*` constructor that looks like ordinary Tensor
    ownership.
  - Hidden need: compiled bindings, external buffers, and tests need a lightweight borrowed view path that cannot be
    mistaken for owning storage.
  - Completed on 2026-06-07: public borrowed `Tensor(void*, ...)` construction was replaced by
    `Tensor::UnsafeBorrowed(...)`; internal borrowed views now call the explicit factory and `G14PublicApiGuard` prevents
    reintroducing the raw constructor.

Recommendation: treat the first three items as the only remaining break-window candidates that can justify delaying vNext
if the goal is a cleaner long-lived ABI. The other items are valuable but can be staged after vNext if guarded by clear
namespaces, diagnostics, and package/component boundaries.

### G15 Vulkan and Mobile GPU Backend

Purpose: add a Vulkan-native backend as the portable mobile-GPU execution path while keeping compiler, runtime, artifact,
and fallback behavior explicit enough for production deployment.

Status: completed for the current bootstrap/mobile packaging scope on 2026-06-11. G15 now covers the Vulkan runtime ABI,
MLIR-generated SPIR-V elementwise/cast payloads, separated artifact packaging, synchronous execution, benchmark
registration, and first workgroup-size tuning. Broader GPU kernel families and asynchronous/device-local scheduling have
been split into the long-term deferred queue because they require a larger kernel ABI and real Vulkan SDK/device
validation.

#### G15.1 P0 Toolchain, Runtime ABI, and Minimal Execution

Status: completed for the first Vulkan-native Add execution slice on 2026-06-07. The initial bootstrap blob was
removed on 2026-06-10; the current minimal Add kernel is generated through MLIR SPIR-V builder/serialization.

- [x] Add `LITENN_ENABLE_VULKAN` and a separate `LiteNNVulkanRuntime` target so minimal CPU/mobile builds do not link
      Vulkan by default.
- [x] Add `Vulkan` device identity, capability discovery, host-visible storage-buffer allocation, host/device transfer,
      and explicit `VulkanHostFallbackPolicy`.
- [x] Add `CompiledModuleBackend::VulkanNative`, `BackendVulkanNative`, and `BackendVulkanBridge` so capability,
      placement, coverage, and artifacts can name Vulkan without pretending every op is supported.
- [x] Define a `VulkanNativeInstructionPayload` containing SPIR-V words, entry point, descriptor bindings, dispatch
      dimensions, and feature flags.
- [x] Compile and load a same-shape static f32 `BinaryOp::Add` graph as a Vulkan-native SPIR-V payload, then run it
      through `CompiledModule<Vulkan>`.
- [x] Add tests and an example that skip cleanly when Vulkan is not enabled or no Vulkan compute device is available.

#### G15.2 P1 MLIR/SPIR-V Generation Path

Status: completed for the current direct MLIR SPIR-V builder/serializer slice. The minimal Add kernel proves the
direct MLIR SPIR-V builder -> serializer path and no
longer depends on checked-in SPIR-V words.

- [x] Verify the installed LLVM/MLIR tree exposes `LLVMSPIRV*` and `MLIR*ToSPIRV` libraries through CMake, not only
      command-line tools.
- [x] Replace P0 bootstrap SPIR-V blobs with builder-generated MLIR SPIR-V or LLVM SPIR-V codegen for the minimal add
      kernel.
- [x] Keep external `spirv-as` / `glslangValidator` only as test/development fallback, never as a required library
      runtime dependency.
- [x] Add generated MLIR/SPIR-V dumps for debugging for the minimal Add generator.
- [x] Add a validation path that rejects modules with non-Vulkan shader interfaces or unsupported memory models before
      serializing them into `VulkanNativeInstructionPayload`.

#### G15.3 P2 Mobile Packaging and Artifact Shape

- [x] Treat Vulkan instructions as SPIR-V module bytes/words, not desktop object files; keep separated metadata,
      constants, and weights compatible with mobile packaging.
- [x] Document Android/mobile CMake build profiles, loader requirements, validation-layer assumptions, and unsupported
      desktop-only features in `docs/VulkanMobileDeployment.md`.
- [x] Add a mobile-oriented example using separated rodata/instructions/weights with Vulkan-native instructions.

#### G15.4 P3 Coverage and Performance

- [x] Add Vulkan-native same-shape f32 binary arithmetic lowering for Add, Subtract, Multiply, Divide, Max, and Min.
- [x] Add Vulkan-native same-shape f32 unary elementwise lowering for Negate, Abs, Sqrt, Exp, Log, Sin, and Cos.
- [x] Add Vulkan-native same-shape 32-bit cast lowering for Float32 -> Int32 and Int32 -> Float32.
- [x] Add Vulkan-native SPIR-V generation for same-shape low-precision casts covering `Float16`, `Int8`, and `UInt8`
      storage types. Runtime execution still requires target-device 8-bit/16-bit storage-buffer feature enablement before
      these kernels should be selected in production.
- [x] Close the current G15 operator-coverage boundary: reductions, matmul/linear-chain, normalization, softmax, and
      convolution are documented as a follow-on GPU-kernel project instead of being hidden inside the bootstrap scope.
- [x] Add initial descriptor/pipeline cache support: `VulkanContext` owns a `VkPipelineCache`, and
      `VulkanComputeModule` reuses a descriptor pool across synchronous dispatches.
- [x] Add first workgroup-size tuning for same-shape Vulkan elementwise/cast kernels: generated SPIR-V now uses
      `LocalSize = 64`, dispatch uses `ceil(numel / 64)` groups, and each shader guards tail threads with an in-kernel
      `global_id < numel` check.
- [x] Add initial benchmark registration for Vulkan AOT rows. `benchmark/bench.cpp` now registers supported
      `VulkanNativeElementwiseAddRunInto` rows when a Vulkan compute device exists; model-level Vulkan Native rows stay
      deferred until matmul/linear lowering is implemented, so benchmark execution no longer fails on an expected coverage
      gap.
- [x] Filter benchmark rows by device capabilities at registration time. CUDA rows are no longer registered when no CUDA
      device exists, and CUDA MatMul dtype rows are emitted only for dtypes supported by the active device, keeping
      `litenn_bench` output free of expected capability-miss errors.
- [x] Split tiled kernels, memory-planner integration, async queue synchronization, and real Vulkan profile-result tables
      into the long-term deferred queue. These are production GPU-backend projects rather than prerequisites for the
      current G15 bootstrap.

#### G15.5 Active Vulkan Production Gap Closure

Status: active as of 2026-06-12. This section tracks what is still missing for Vulkan to move from a bootstrap/mobile
packaging slice to a useful production backend.

- [x] Add a compiler-facing Vulkan native coverage query. `Compiler<Vulkan>::QueryNativeSupport` now reports whether an
      executable plan can use the current native Vulkan slice and returns a stable reason when it would fall back to CPU.
- [x] Add the first f32 MatMul Vulkan-native slice: rank-2 static `Float32` `BinaryOp::MatMul` now lowers to generated
      MLIR/SPIR-V, emits a `MatMulF32` payload feature bit, runs through `CompiledModule<Vulkan>`, and is registered in
      benchmark as `VulkanNativeMatMul/F32`.
- [x] Add the first f32 MatMulBias Vulkan-native slice: fused `MatMulBiasAdd` / `MatMulBiasAddReLU` now lowers to a
      generated single-kernel SPIR-V baseline with broadcast row bias, and benchmark registers
      `VulkanNativeMatMulBiasAdd/F32`.
- [x] Add Vulkan-native separated external tensor binding for the first model-shaped Linear slice: fused
      `MatMulBiasAdd/ReLU` can now bind graph `VariableRefNode` / `ConstantNode` tensors from separated
      constants/weights metadata, upload them to Vulkan device tensors at load time, and run with only public model
      inputs. `benchmark/bench.cpp` registers `VulkanNativeRunInto/Linear(784->10)` rows when a Vulkan compute device
      exists.
      Validation on 2026-06-12: `CompiledModuleVulkanTest` passed 25/25 after adding load-time capability gating; local
      `litenn_bench --benchmark_filter=VulkanNative --benchmark_min_time=0.001s` reported
      `VulkanNativeRunInto/Linear(784->10)` at roughly 0.154/0.174/0.258/0.260 ms for batch 1/32/128/512.
- [x] Add the first load-time device capability gate for generated Vulkan SPIR-V payloads: `CompiledModule<Vulkan>::Load`
      now checks the payload target environment, Vulkan API version, fixed local workgroup size, storage-buffer range
      limits, and low-precision cast requirements before creating shader modules/pipelines. The runtime reports whether
      matching 8-bit/16-bit storage and fp16/int8 shader features are physically available versus actually enabled on
      LiteNN's logical device.
- [x] Enable optional Vulkan logical-device feature chains for low-precision execution: the runtime enumerates device
      extensions, wires supported `VkPhysicalDevice16BitStorageFeatures`, `VkPhysicalDevice8BitStorageFeatures`, and
      `VkPhysicalDeviceShaderFloat16Int8Features` into `vkCreateDevice`, and exposes both physical availability and
      enabled-state through `QueryVulkanDeviceCapabilities`. `benchmark/bench.cpp` registers
      `VulkanNativeCastRunInto/F32ToFloat16|Int8|UInt8` rows only when the enabled feature set satisfies the artifact
      requirements.
      Validation on 2026-06-12: local Vulkan tests passed 25 and skipped the feature-dependent Float16 runtime test
      because the selected device did not enable the required Float16 storage path; f32 Vulkan-native benchmark rows
      continued to run.
- [x] Expand the Vulkan-native payload ABI with per-kernel requirement metadata. Payload version 2 records descriptor ABI
      version, local workgroup layout, required device feature bits, subgroup-size requirements, and storage-buffer
      offset-alignment requirements. `CompiledModule<Vulkan>::Load` consumes this metadata before creating Vulkan
      pipelines, while version-1 low-precision payloads still keep the old featureSet/spec safety fallback.
      Validation on 2026-06-13: `CompiledModuleVulkanTest` passed 26 tests and skipped the feature-dependent Float16
      runtime case on the local device; `litenn_bench --benchmark_filter=VulkanNative --benchmark_min_time=0.001s`
      continued to run the current f32 Vulkan-native rows.
- [x] Extend Vulkan device capability gating for current descriptor/storage dispatch limits: the runtime now reports
      `maxComputeWorkGroupCount`, storage-buffer descriptor limits, and `maxBoundDescriptorSets`; native artifact load
      rejects kernels whose dispatch groups or storage-buffer descriptor count exceed the selected device.
      Validation on 2026-06-13: `CompiledModuleVulkanTest` passed 30/31 Vulkan tests, with only the local
      feature-dependent Float16 runtime case skipped.
- [x] Extend Vulkan device capability gating for future advanced kernels: descriptor indexing limits, shared-memory and
      push-constant limits, advanced subgroup operation policies, and descriptor-indexing/runtime-descriptor-array
      feature requirements are represented before selecting those kernels.
      Validation on 2026-06-13: `CompiledModuleVulkanTest` passed 36/37 Vulkan tests with only the local
      feature-dependent Float16 runtime case skipped; payload loading rejects disabled advanced device requirement bits
      with the same available/enabled diagnostic style used by low-precision features.
- [ ] Replace the single-kernel whole-graph matcher with a graph partitioner: native-supported islands should run on
      Vulkan, unsupported islands should require an explicit bridge/fallback decision, and tensor movement must be visible
      in the schedule.
- [x] Add the first multi-kernel Vulkan whole-graph slice before the full partitioner: same-shape Float32 chains composed
      of supported `BinaryOp` nodes now compile to multiple Vulkan-native kernels that reuse the final output buffer as
      an accumulator between synchronized dispatches, including mixed chains such as
      `Multiply(Add(lhs, rhs), tail)`.
      Validation on 2026-06-13: `CompiledModuleVulkanTest` passed 35/36 Vulkan tests with only the local
      feature-dependent Float16 runtime case skipped; `example/vulkan` showed `TwoAdd` and `MixedChain` selecting
      `vulkan_native`, while an unsupported diamond graph still requires explicit CPU bridge fallback.
- [ ] Add a Vulkan memory planner with device-local storage plus staging buffers. The current host-visible buffer path is
      good for correctness and small tests, but production kernels need reusable device-local allocations and fewer
      host/device round trips.
      - [x] Add explicit Vulkan buffer residency policy to the public device configuration and allocator, with
            `HostVisibleCoherent` preserving today's default behavior and `DeviceLocal` allocating transfer-capable
            device-local storage buffers.
      - [x] Add staging upload/download helpers for device-local tensors so `CopyFromCPU`, `CopyToCPU`, `ZeroFill`, and
            same-dtype device copies can move through transfer-capable buffers without host-mapping device-local memory.
      - [x] Route `CompiledModule<Vulkan>::RunTensors` output tensors and separated external tensors through the selected
            Vulkan device residency policy; device-local module inputs/outputs now run through staging at host
            boundaries.
      - [x] Add `CompiledModule<Vulkan>::AllocateOutputTensors()` so repeated `RunTensorsInto` callers, benchmarks, and
            profiling tools reuse output lifetimes while preserving the module's selected Vulkan residency policy.
      - [x] Add `CompiledModuleVulkanRunWorkspace` so repeated Vulkan runs can reuse module-policy output tensors and
            CPU-bridge host scratch tensors without changing the existing `RunTensorsInto` ownership model.
      - [x] Add native `WorkspaceTensor` payload/runtime ABI: payload v3 records reusable workspace buffer specs,
            kernels can bind workspace descriptors, and `CompiledModule<Vulkan>` allocates the buffers at load time.
      - [x] Route the existing same-shape binary-chain schedule through `WorkspaceTensor` buffers instead of using the
            public output tensor as the accumulator between kernels.
      - [x] Introduce a reusable Vulkan P0 workspace planner helper and validate one-slot reuse across longer
            same-shape binary chains.
      - [x] Make the Vulkan P0 workspace planner interval-aware so future non-linear multi-kernel schedules can reuse
            buffers only when lifetimes do not overlap.
      - [x] Expose Vulkan workspace tensor count and total bytes in `litenn_profile` console/CSV output so workspace
            planning changes are visible in performance reports.
      - [x] Add Vulkan-only binary-chain profile rows that exercise non-zero `WorkspaceTensor` allocations and persist
            those workspace metrics to `vulkan_profile.csv`.
      - [x] Lower `FusedOpNode(ElementWiseChain)` bodies that still form same-shape f32 binary chains through the
            existing Vulkan workspace-chain planner, so normal graph optimization no longer blocks this native path.
      - [x] Add the first schedule-level lifetime planner slice for same-shape f32 binary DAGs: independent
            intermediate values now receive separate `WorkspaceTensor` slots when lifetimes overlap, covering the
            diamond graph shape.
      - [x] Add `litenn_profile` Vulkan binary-DAG rows so the overlapping-lifetime workspace plan is visible as
            `WS=2` / non-zero `WSBytes` in console and CSV output.
      - [x] Register `litenn_bench` Vulkan binary-chain and binary-DAG rows so workspace-backed native schedules are
            directly visible in the normal backend benchmark matrix.
      - [x] Add support/payload/runtime coverage for branched same-shape f32 binary DAGs with tail reuse, proving the
            current last-use planner handles non-diamond multi-kernel schedules and workspace slot reuse.
      - [x] Add `litenn_profile` and `litenn_bench` rows for branched same-shape f32 binary DAGs so `WS=3` schedules
            and five-dispatch workspace behavior are visible in normal performance runs.
      - [x] Add the first mixed same-shape f32 elementwise DAG schedule: non-fused and fused
            `ElementWiseChain` Unary/Binary DAGs such as `Multiply(Abs(Add(lhs, rhs)), tail)` now lower to one SPIR-V
            module with multiple entry points, explicit `WorkspaceTensor` lifetimes, support/payload/runtime tests, and
            `litenn_bench` / `litenn_profile` mixed-DAG rows.
      - [x] Normalize `litenn_profile` output-directory parsing so both positional paths and `--out-dir` /
            `--out-dir=...` forms write raw objects, assembly, and Vulkan CSV rows to the intended directory.
      - [x] Add representative `litenn_profile` Vulkan rows for Reduce, Softmax, LayerNorm, and RMSNorm native
            kernels so scalar-loop reduction/normalization bottlenecks show up beside elementwise DAG GPU timing.
      - [x] Generalize schedule-level lifetime planning beyond binary DAGs: binary DAG and mixed unary/binary
            elementwise DAG payload emission now share the same schedule-level last-use/workspace allocation helper,
            leaving future operator-family schedulers to feed the common planner instead of duplicating buffer logic.
      - [x] Add graph-level device-local benchmark rows for Vulkan-native MLP128/MLP512 so host-visible and
            device-local residency policies can be compared in the standard benchmark matrix.
- [ ] Implement production operator families in priority order:
      - [x] Add the first static-axis f32 reduction slice: `ReduceOp::Sum`, `ReduceOp::Mean`, `ReduceOp::Max`, and
            `ReduceOp::Min` now
            lower to Vulkan-native SPIR-V with one invocation per output element and a scalar loop over the reduced axis.
            Validation on 2026-06-14: `CompiledModuleVulkanTest` passed 73/74 Vulkan tests with only the local
            feature-dependent Float16 runtime case skipped; `litenn_bench` now registers
            `VulkanNativeReduce/F32/SumAxis1|MeanAxis1|MaxAxis1|MinAxis1` rows and the local
            `MinAxis1/batch:1/width:128` smoke row ran successfully at roughly 0.101 ms.
      - [ ] Replace the static-axis reduce scalar-loop baseline with
            workgroup/subgroup reductions for larger axes.
      - [ ] Complete f32 multi-layer linear-chain lowering with workspace/multi-kernel schedules, then replace the
            current one-output-element-per-thread MatMul/MatMulBias kernels with tiled/shared-memory kernels.
            - [x] Add the first Vulkan-native homogeneous f32 linear-chain slice: same-shape fused
                  `MatMulBiasAdd` chains now compile into one artifact with multiple MatMulBias kernels and
                  `WorkspaceTensor` handoff between layers. Validation on 2026-06-18:
                  `CompiledModuleVulkanTest.WritesVulkanNativePayloadForHomogeneousLinearChain` and
                  `CompiledModuleVulkanTest.RunsHomogeneousLinearChainArithmetic` passed locally; `litenn_bench`
                  registers `VulkanNativeLinearChain/F32/layers:2` microbench rows.
            - [x] Extend linear-chain payloads beyond homogeneous shapes without changing the artifact ABI: generated
                  MatMulBias SPIR-V modules can now contain multiple named entry points, and mixed-shape
                  `MatMulBiasAddReLU -> MatMulBiasAdd` chains run as one Vulkan-native artifact. Validation on
                  2026-06-18: mixed-shape QueryNativeSupport/payload/runtime tests passed; `litenn_bench` registers
                  `VulkanNativeGraphRunInto/MLP(784->128->10)` and smoke-ran batch 1 at roughly 0.266 ms and batch
                  128 at roughly 0.276 ms.
      - [x] Add the first static-axis f32 `SoftmaxNode` slice: Vulkan-native SPIR-V now computes numerically stable
            max-subtracted softmax with one invocation per output element and scalar loops over the softmax axis.
            Validation on 2026-06-13: `CompiledModuleVulkanTest` passed 42/43 Vulkan tests with only the local
            feature-dependent Float16 runtime case skipped; `litenn_bench` now registers
            `VulkanNativeSoftmax/F32/Axis1` rows and the local `batch:1/width:128` smoke row ran successfully.
      - [x] Replace the scalar-loop-per-output Vulkan Softmax baseline with a three-dispatch workspace schedule:
            row max, row sum, and elementwise write now share two row-sized `WorkspaceTensor` buffers. Validation on
            2026-06-17: `CompiledModuleVulkanTest.*Softmax*` passed, and `litenn_profile` reduced
            `softmax_b512` GPU time from roughly 4.93 ms to roughly 0.30 ms on the local Vulkan device.
      - [x] Add the first static-axis f32 normalization slice: non-affine `LayerNorm` and `RMSNorm` now lower to
            Vulkan-native SPIR-V with one invocation per output element and scalar loops over the normalized axis.
            Validation on 2026-06-13: `CompiledModuleVulkanTest` passed 44/45 Vulkan tests with only the local
            feature-dependent Float16 runtime case skipped; `litenn_bench` now registers
            `VulkanNativeNormalization/F32/LayerNormAxis1|RMSNormAxis1` rows and the local
            `RMSNormAxis1/batch:1/width:128` smoke row ran successfully.
      - [x] Replace the Vulkan LayerNorm scalar-loop baseline with a two-dispatch workspace schedule: row stats and
            elementwise write now share row-sized mean/denominator `WorkspaceTensor` buffers, including affine
            scale/bias bindings. Validation on 2026-06-17: `CompiledModuleVulkanTest.*Normalization*` and
            `CompiledModuleVulkanTest.*GroupNorm*` passed, `litenn_profile` reduced `layernorm_b512` GPU time from
            roughly 1.45 ms to roughly 0.21 ms, and `affine_layernorm_b512` profiled at roughly 0.23 ms.
            RMSNorm remains on the single-dispatch path because local profile showed the staged route was slower.
      - [x] Add affine `LayerNorm`/`RMSNorm` for the common rank-2 axis-1 layout: scale/bias tensors with shape
            `[axis]` or `[1,axis]` now bind as Vulkan input/external tensors, including separated variable weights.
            Validation on 2026-06-13: `CompiledModuleVulkanTest` passed 47/48 Vulkan tests with only the local
            feature-dependent Float16 runtime case skipped; `litenn_bench` now registers
            `VulkanNativeNormalizationAffine/F32/LayerNormAxis1|RMSNormAxis1` rows and the local
            `LayerNormAxis1/batch:1/width:128` smoke row ran successfully.
      - [x] Add the first static-shape f32 `GroupNorm` slice: Vulkan-native SPIR-V now follows the existing ggml-style
            LiteNN semantics where rank-4 uses the last dimension as batch and rank<4 uses a single batch lane.
            Validation on 2026-06-13: `CompiledModuleVulkanTest` passed 50/51 Vulkan tests with only the local
            feature-dependent Float16 runtime case skipped; `litenn_bench` now registers `VulkanNativeGroupNorm/F32`
            rows and the local `groups:8/elements:784` smoke row ran successfully.
      - [x] Add affine `GroupNorm` for native/ggml layout: scale/bias tensors with shape `[groupedVolume]` or
            `[1,groupedVolume]` now bind as Vulkan input/external tensors, including separated variable weights.
            Validation on 2026-06-13: `CompiledModuleVulkanTest` passed 52/53 Vulkan tests with only the local
            feature-dependent Float16 runtime case skipped; `litenn_bench` now registers
            `VulkanNativeGroupNormAffine/F32` rows and the local `groups:8/elements:784` smoke row ran successfully.
      - [ ] Add workgroup/subgroup normalization kernels that avoid recomputing the same group or axis statistics per
            output element.
      - [x] Add the first f32 `Pool2DNode` slice: Vulkan-native SPIR-V now supports static rank-4 NCHW
            MaxPool/AveragePool with rank-2 kernel/stride parameters.
            Validation on 2026-06-13: `CompiledModuleVulkanTest` passed 55/56 Vulkan tests with only the local
            feature-dependent Float16 runtime case skipped; `litenn_bench` now registers
            `VulkanNativePool2D/F32/Max|Average` rows and the local `Average/batch:1/channels:8/spatial:16` smoke row
            ran successfully.
      - [x] Add padded `Pool2DNode` coverage: native SPIR-V now supports low/high padding for Max/Average and
            `AveragePool` `countIncludePad` semantics.
            Validation on 2026-06-13: `CompiledModuleVulkanTest` passed 57/58 Vulkan tests with only the local
            feature-dependent Float16 runtime case skipped; `litenn_bench` now registers
            `VulkanNativePool2D/F32/MaxPadded|AveragePadded|AveragePaddedIncludePad` rows and the local
            `AveragePaddedIncludePad/batch:1/channels:8/spatial:16` smoke row ran successfully.
      - [x] Add baseline direct `Conv2DNode` coverage: native SPIR-V now supports static rank-4 NCHW f32 Conv2D with
            stride, dilation, low/high padding, groups, and optional bias, including separated variable/constant
            weight tensors.
            Validation on 2026-06-13: `CompiledModuleVulkanTest` passed 61/62 Vulkan tests with only the local
            feature-dependent Float16 runtime case skipped; `litenn_bench` now registers
            `VulkanNativeConv2D/F32` rows and the local `batch:1/channels:8/outChannels:8/spatial:16` smoke row ran
            successfully.
      - [x] Add baseline nearest `UpsampleNode` coverage: native SPIR-V now supports static rank-4 NCHW f32 nearest
            upsampling with `alignCorners=false`, matching the CPU reference integer source-index mapping.
            Validation on 2026-06-13: `CompiledModuleVulkanTest` passed 64/65 Vulkan tests with only the local
            feature-dependent Float16 runtime case skipped; `litenn_bench` now registers
            `VulkanNativeUpsampleNearest/F32` rows and the local `batch:1/channels:8/spatial:16x32` smoke row ran
            successfully.
      - [x] Add baseline direct `ConvTranspose2DNode` coverage: native SPIR-V now supports static rank-4 NCHW f32
            transposed convolution with stride, dilation, low/high padding, output padding, groups, optional bias, and
            separated variable/constant weight tensors.
            Validation on 2026-06-14: `CompiledModuleVulkanTest` passed 67/68 Vulkan tests with only the local
            feature-dependent Float16 runtime case skipped; `litenn_bench` now registers
            `VulkanNativeConvTranspose2D/F32` rows and the local `batch:1/channels:8/outChannels:8/spatial:16x31`
            smoke row ran successfully.
      - [x] Add baseline `SliceNode` coverage: native SPIR-V now supports static non-empty f32 tensors with arbitrary
            rank, in-range `axis/start/length`, and direct graph-parameter input.
            Validation on 2026-06-14: `CompiledModuleVulkanTest` passed 70/71 Vulkan tests with only the local
            feature-dependent Float16 runtime case skipped; `litenn_bench` now registers `VulkanNativeSlice/F32` rows
            and the local `batch:1/channels:8to4/spatial:16` smoke row ran successfully.
      - [x] Add baseline `ConcatNode` coverage: native SPIR-V now supports two direct graph-parameter f32 inputs with
            compatible static non-empty shapes and an arbitrary concat axis.
            Validation on 2026-06-14: `CompiledModuleVulkanTest` passed 73/74 Vulkan tests with only the local
            feature-dependent Float16 runtime case skipped; `litenn_bench` now registers `VulkanNativeConcat/F32` rows
            and the local `batch:1/channels:4plus4/spatial:16` smoke row ran successfully.
      - [ ] Add tiled/shared-memory convolution and image/latent-processing kernels for mobile vision workloads.
      - [ ] Add low-precision arithmetic kernels beyond casts, including fp16/bf16/int8 paths guarded by device
            capabilities.
            - [x] Add the first fp16 arithmetic slice for Vulkan native same-shape binary kernels: simple Float16
                  Add/Subtract/Multiply/Divide/Min/Max graphs now compile through dtype-aware SPIR-V generation with
                  shaderFloat16/storageBuffer16BitAccess requirements, support diagnostics, tests, and optional
                  `litenn_bench` F16 Add rows when the selected device enables the required features.
            - [x] Add the matching fp16 same-shape unary slice for Vulkan native Negate/Abs/Sqrt/Exp/Log/Sin/Cos,
                  including dtype-aware SPIR-V generation, payload feature bits, device requirements, support
                  diagnostics, and feature-gated runtime coverage.
            - [x] Add load-time capability regression coverage and benchmark visibility for fp16 Vulkan arithmetic:
                  Float16 binary/unary payloads are rejected when required device features are disabled, and
                  `litenn_bench` now registers `VulkanNativeUnaryAbsRunInto` F32/F16 rows when supported.
            - [x] Add the first int8/uint8 arithmetic slice for Vulkan native same-shape binary kernels: Add/Subtract/
                  Multiply/Divide/Min/Max now lower through integer SPIR-V ops for `Int8` and `UInt8`, carry
                  shaderInt8/storageBuffer8BitAccess requirements, and register optional Int8/UInt8 Add benchmark rows
                  when the selected device enables those features.
            - [x] Add the first int8 unary arithmetic slice for Vulkan native same-shape kernels: Negate/Abs now lower
                  through integer SPIR-V ops for `Int8`, carry shaderInt8/storageBuffer8BitAccess requirements, report
                  native support, validate runtime behavior when the selected device enables those features, and register
                  optional Int8 Abs benchmark rows.
- [ ] Add asynchronous execution and synchronization primitives: reusable command buffers, fences/timeline semaphores,
      queue ownership rules, and `RunManyTensorsInto` semantics that do not serialize every dispatch through a full
      wait.
      - [x] Reuse per-module descriptor set, command buffer, fence, and timestamp query pool in the synchronous P0
            runtime path instead of allocating/resetting those Vulkan objects on every dispatch; timeline semaphore
            based async submission remains open.
      - [x] Add synchronous multi-kernel batch submission for Vulkan-native payloads without profile-event sinks:
            `CompiledModule<Vulkan>` now records each kernel's reusable command buffer and submits the command buffers
            through one `vkQueueSubmit` plus one fence wait, while profile mode keeps per-kernel synchronized dispatch so
            timestamp attribution remains stable. Validation on 2026-06-18: `CompiledModuleVulkanTest` passed 109/112
            with the same three local fp16 feature skips; `VulkanNativeGraphRunInto` smoke rows ran for MLP128/MLP512
            batches 1/32/128/512. Public async entry points remain explicitly rejected until timeline-semaphore
            ownership is implemented, covered by `CompiledModuleVulkanTest.RejectsPublicAsyncVulkanNativeRun`.
- [x] Add the first Vulkan profiling tranche: `CompiledModuleVulkanRunOptions` accepts a profile-event sink, native
      dispatch records now include kernel index, entry point, dispatch groups, local workgroup layout, descriptor count,
      module creation wall time, and synchronized CPU-side dispatch wall time. `litenn_profile` prints a Vulkan native
      breakdown table with compile/load/first-run/steady-run/dispatch timings and clearly shows unsupported model cases
      as CPU bridge artifacts.
      Validation on 2026-06-13: `CompiledModuleVulkanTest` passed 27 tests and skipped the local Float16 runtime case;
      `litenn_profile` smoke printed Vulkan native rows for Linear and explicit CPU-bridge rows for current MLP cases.
- [x] Add GPU timestamp-query profiling around each Vulkan kernel when the selected compute queue supports timestamps.
      `CompiledModuleVulkanProfileEvent` now reports whether GPU timing is available and the elapsed timestamp-query
      time in milliseconds; `litenn_profile` prints this beside CPU-side dispatch wall time.
      Validation on 2026-06-13: `CompiledModuleVulkanTest` passed 30/31 Vulkan tests, `example/vulkan` printed
      `gpu_ms`, and `litenn_profile` smoke printed Vulkan `GPUTime` with CUDA/objdump sections skipped.
- [x] Add one-shot Vulkan host/device transfer timing to `litenn_profile`: the Vulkan table now reports input upload and
      output download time separately from first-run, steady RunInto, CPU-side dispatch wall time, and GPU timestamp time.
      Validation on 2026-06-13: `litenn_profile` smoke printed Upload/GPUTime/Download columns for Vulkan Linear rows.
- [x] Persist raw Vulkan profile rows from `litenn_profile` as `vulkan_profile.csv` under the requested output directory.
      Validation on 2026-06-13: smoke profile wrote `build-release/profile_vulkan_smoke/vulkan_profile.csv` with
      Vulkan-native Linear rows and explicit CPU-bridge MLP rows. As of 2026-06-18, the existing MLP128 profile rows
      exercise the mixed-shape Vulkan-native graph path when the current backend can compile it; a local
      `litenn_profile --out-dir build-release/profile_vulkan_graph_smoke` smoke also confirmed MLP512 runs as a
      three-kernel Vulkan-native graph.
- [x] Add full comparison-table automation across CPU AOT, CUDA, Vulkan, ggml, and PyTorch for supported shapes:
      `benchmark/compare_backends.py` reads Google Benchmark JSON plus PyTorch text output and emits Markdown/CSV
      tables with per-backend `ms/batch` and percent deltas against PyTorch CPU/CUDA and ggml baselines. Fresh result
      generation still depends on the local machine having the requested CUDA/Vulkan/PyTorch/ggml capabilities. The
      default report is scoped to the standard inference model set; use `--include-all-models` only for microbench
      diagnostics. It now distinguishes `VulkanNativeGraphRunInto` as `LiteNN Vulkan Graph` so whole-graph Vulkan AOT
      rows do not collapse into the single-kernel native bucket, and reports
      `VulkanNativeGraphDeviceLocalRunInto` as `LiteNN Vulkan Graph DeviceLocal` for residency comparisons.
- [x] Expand artifact ABI metadata for current Vulkan kernel requirements: SPIR-V target environment stays in
      `VulkanNativeInstructionPayload::target`, while each kernel now records descriptor ABI version, required feature
      bits, local workgroup layout, subgroup-size requirement, and storage-buffer offset alignment.
- [x] Add optional specialization constants to Vulkan artifact metadata: payload v4 carries per-kernel specialization
      map entries and data blobs, validates their byte ranges, and `VulkanComputeModule` passes them to
      `vkCreateComputePipelines`. Generated kernels can now use this for tile sizes or runtime-shape constants without
      changing the artifact ABI again.
- [ ] Add mobile validation coverage: Android cross-build profile, loader/validation-layer smoke tests, at least one real
      mobile GPU fixture, and clear skip/failure behavior when no Vulkan compute device is present.
- [x] Add an end-to-end Vulkan example that is honest about backend selection: `example/vulkan` now prints native support
      for the Vulkan-native Add slice, runs both regular and separated-region native artifacts, records CPU-side Vulkan
      profile events, and demonstrates that a CPU bridge artifact is rejected unless `VulkanHostFallbackPolicy::Allow`
      is set explicitly.
- [x] Add a model-shaped Vulkan benchmark beyond the current single-Linear external-weight slice without pretending
      whole-graph partitioning is complete: `benchmark/bench.cpp` now registers
      `VulkanNativeManualPipeline/MLP(784->128->10)` rows that explicitly run two Vulkan-native Linear artifacts with a
      Vulkan-resident hidden tensor between them, and `VulkanNativeGraphRunInto/MLP(784->128->10)` plus
      `VulkanNativeGraphRunInto/MLP(784->512->256->10)` rows for the current mixed-shape whole-graph linear-chain path.
      `VulkanNativeGraphDeviceLocalRunInto/MLP(...)` mirrors the whole-graph rows with device-local module/input/output
      residency so transfer policy effects are visible.
      Automatic graph partitioning remains tracked separately above.
      Validation on 2026-06-14: a local
      `litenn_bench --benchmark_filter=VulkanNativeManualPipeline.*batch:1 --benchmark_min_time=0.01s` smoke run
      reported roughly 0.283 ms for batch 1 and 0.280 ms for batch 128. Validation on 2026-06-18:
      `VulkanNativeGraphRunInto.*batch:1/real_time$` smoke reported roughly 0.150 ms for MLP128 and 0.258 ms for
      MLP512.

Hidden requirements:

- Vulkan support is not only "LLVM can emit SPIR-V"; shader interface, descriptor set layout, storage-buffer ABI,
  memory model, workgroup dimensions, and device capabilities must all be validated.
- Mobile GPUs vary in fp16/int8/subgroup support and storage-buffer alignment. The backend must report capabilities
  rather than silently selecting kernels that only work on desktop drivers.
- CPU fallback must be explicit for Vulkan, matching the CUDA vNext policy; otherwise benchmark and production behavior
  become opaque.
- Vulkan artifacts should not inherit CPU object-file assumptions. SPIR-V belongs in the instruction region, while
  constants and weights remain separated package regions.

### G16 Full LLM Runtime: GGUF/Qwen2.5 to CUDA-Capable Generation

Purpose: turn the current GGUF importer and tiny LLaMA-family lowering into a practical full causal-LM runtime path. The
acceptance target is a real GGUF model such as `Qwen2.5-Coder-14B-Instruct-Q4_K_M`: import, diagnose, lower, validate
against llama.cpp golden logits, compile/load artifacts, and run a complete token generation loop with explicit CPU/CUDA
placement and fallback policy.

#### G16.1 Compatibility Profiles and Diagnostics

- [x] Add an explicit Qwen2/Qwen2.5 compatibility profile rather than treating it as generic LLaMA.
      `Qwen2LikeCausalLM` now records tokenizer/chat-template, Q4_K_M CUDA, and decode-loop requirements as actionable
      diagnostics while preserving the current token-id lowering boundary.
- [x] Add a `litenn_gguf_convert --analyze-llm <input.gguf> [profile]` command that prints blocking/non-blocking
      diagnostics before import/lowering/compile. The command infers `qwen2-like-causal-lm` for `general.architecture =
      qwen2`, accepts explicit profile names, and exits non-zero only for blocking compatibility failures.
- [x] Add model-family aliases for common GGUF `general.architecture` values: `llama`, `qwen2`, `qwen2moe`, `mistral`,
      `gemma`, and reject unsupported families with a profile suggestion. Automatic inference now maps `llama` and
      `qwen2`; known but unsupported families such as `qwen2moe`, `mistral`, and `gemma` intentionally return no profile
      until their contracts are added.
- [x] Detect real GGUF quantization mixes, especially `Q4_K_M`, and report whether the chosen execution path is
      native-quantized, reference-dequantized, or rejected by memory budget. The analyzer now reports per-format
      block-quantized tensor counts/bytes and emits a dedicated `quantization.q4_k_m` diagnostic for `GGML_Q4_K` mixes;
      native CUDA remains a G16.4/G16.5 implementation target.

#### G16.2 Tokenizer, Prompt, and Sampler Runtime

- [x] Preserve and expose tokenizer metadata needed by Qwen chat templates, BOS/EOS handling, special tokens, and byte
      fallback rules.
- [x] Add a minimal tokenizer bridge for caller-provided token ids with tokenizer-vocabulary validation.
- [x] Add a deliberately limited exact-vocabulary prompt bridge over `tokenizer.ggml.tokens` for fixtures and diagnostic
      CLI runs; this does not replace GPT2/BPE or llama.cpp tokenizer parity.
- [x] Add tokenizer execution through an explicit optional llama.cpp adapter target:
      the isolated `tools/llamacpp-adapter` target exposes `tokenize` and binary-safe `detokenize` subcommands using the
      model's real GGUF vocabulary, BOS policy, special-token parsing, and byte fallback behavior without linking
      llama.cpp into LiteNN runtime targets.
- [x] Wire the optional tokenizer adapter into GGUF tools behind an explicit build option:
      `LITENN_ENABLE_LLAMA_CPP_TOKENIZER=ON` links `LiteNNLlamaCppTokenizerAdapter` into `litenn_gguf_convert`, enabling
      `--tokenize-llama-prompt`, `--run-llama-prompt --chat-template`, and prompt decode loops to use llama.cpp's
      in-process tokenizer while keeping the default LiteNN runtime and tool build free of llama.cpp runtime linkage.
- [x] Add a manifest-backed tokenizer adapter driver:
      `scripts/gguf_tokenizer_adapter.py` handles Windows runtime-library discovery, command evidence, output paths, and
      operation manifests for tokenization, binary-safe detokenization, and model-default single-user chat-template
      application.
- [x] Wire real tokenizer execution into the Qwen smoke flow:
      `--llamacpp-tokenizer-tool` converts `--prompt` into validated token ids for direct LiteNN decode, then
      detokenizes generated ids to `--text-output` (or a work-directory default) through the same vocabulary;
      prompt execution now applies the model chat template by default for instruct-model reply behavior, while
      `--raw-prompt` preserves deliberate continuation-style completion.
- [x] Validate the optional adapter against a real Qwen2.5-Coder-14B-Instruct Q4_K_M GGUF:
      on 2026-06-21, `hello` tokenized to id `14990` with `addBos=false` and detokenized byte-for-byte back to `hello`;
      the default chat template produced Qwen's user/assistant markers and the expected nine-token prompt sequence.
- [x] Implement the graph-external generation loop control API for token history, EOS detection, logits
      post-processing, and sampler state.
- [x] Support temperature, top-k, top-p, repeat penalty, seedable sampling, and greedy mode with deterministic tests.
- [x] Keep default greedy decode linear in vocabulary size: select by stable one-pass argmax, avoid reading token
      history when repeat penalty is disabled, construct repeat membership once when enabled, and view the final
      Tensor logits row without a temporary copy. A 2026-08-01 Qwen2.5-Coder-14B cache-hit run preserved the exact
      output and reduced measured generation sampling from `~9-11 ms/token` to `0.162 ms/token` on average.

#### G16.3 Lowering and State ABI

- [x] Split LLM lowering into named artifact entries: `prefill`, `decode_step`, optional `logits_postprocess`, and
      metadata/state descriptors.
- [x] Expose explicit mutable KV-cache runtime-state bindings and per-layer key/value byte offsets in the LLM artifact
      plan.
- [x] Replace functional decode graph cache inputs/outputs with in-place runtime-state rebinding. Runtime schedules now
      map function input/output values directly onto byte ranges of persistent state buffers, and LLaMA artifact plans
      emit key/value bindings for every decoder layer while enforcing capacity for the appended token.
- [x] Add runtime-schedule public-output projection helpers so state-written function outputs can be separated from
      user-visible outputs. Runtime schedules now identify logits-only public output `TensorType`/shape and state-output
      aliases back to their mutable input state buffers through one `RuntimeScheduleOutputProjection` contract, while
      CPU AOT still needs a state-aware entry wrapper before `outputs_per_step` can drop from functional cache outputs
      to the public surface.
- [x] Separate decode position/current `pastLength` from max KV-cache capacity in the LLM artifact/state ABI.
- [x] Support runtime decode position without recompiling for every `pastLength`:
      capacity decode accepts `current_position` plus full-capacity KV tensors, writes the new K/V row through
      single-index Scatter, masks inactive cache suffixes, computes RoPE from the runtime position, returns
      `next_position`, and aliases all mutable state through the package ABI. The Interpreter decode loop now builds one
      capacity plan instead of one static graph per token; CPU AOT parity, two-position reuse, and package roundtrip are
      covered by `GGUFImporterTest`.
- [x] Support variable prompt length in executable prefill lowering without recompiling for each prompt shape.
      `LowerLLaMACausalLMPrefillCapacity` accepts max-capacity padded `token_ids` and returns full max-capacity logits;
      the caller/runtime schedule keeps the prompt length and selects `prompt_length - 1` on the host side for the final
      valid logits. Interpreter reuse and CPU AOT parity tests cover the capacity artifact contract.
- [x] Finish CPU AOT execution for multi-token LLaMA prefill graphs. Capacity decode and capacity prefill AOT execution
      are both covered; the generated path now registers MLIR's unranked `memrefCopy` runtime ABI instead of jumping
      through an unresolved JIT symbol.
- [x] Lower single-index `ScatterNode` through MLIR for race-free capacity-cache updates, while continuing to reject
      unsupported multi-index compiled Scatter forms explicitly.
- [x] Validate Qwen2/Qwen2.5 RoPE semantics through the production evidence gate before enabling production profiles.
      `gguf_production_gate.py` requires matching llama.cpp golden logits for requested prefill/decode/text reports and
      fails closed when parity evidence is missing; long-context/YaRN metadata remains part of the external golden
      fixture matrix rather than an unconditional in-repo model claim.
- [x] Keep tensor layout conversions explicit: imported GGUF layout, LiteNN semantic layout, CUDA-native layout, and cache
      layout must each be inspectable.
- [x] Preserve optional Qwen2 projection biases during GGUF lowering, including the native GGUF `[outFeatures]` vector
      shape and LiteNN's existing `[1, outFeatures]` broadcast form.

#### G16.4 Quantized Weight Execution

- [x] Keep imported GGML block payloads external and quantized through vNext packaging. `--import-external` writes raw
      payload bytes to a sibling weight region while retaining block format and expressed-shape metadata in the manifest.
- [x] Preserve packaged GGML block weights through LLaMA projection lowering instead of materializing Float32 variables
      while constructing the model. Quantized projections now emit first-class `QuantizedMatMulNode` operations, token
      embeddings keep explicit quantized storage semantics, and tied vocab-major embeddings preserve the same layout
      contract for the LM head.
- [x] Execute affine `QuantizedMatMulNode` directly in CPU AOT without materializing a full Float32 weight tensor.
      The MLIR lowering now keeps Int8/UInt8 weight storage in the reduction loop and broadcasts
      per-tensor/per-axis/grouped scale/zero-point metadata as small constants.
- [x] Execute packed-nibble Int4/UInt4 `QuantizedMatMulNode` directly in CPU AOT without materializing a full Float32
      weight tensor. The MLIR lowering decodes nibbles from UInt8 payload bytes inside the reduction loop.
- [x] Execute GGML-block `QuantizedMatMulNode` directly in CPU AOT and CUDA without materializing a full Float32 weight
      tensor:
      - [x] CPU AOT directly decodes output-major `Q4_K`, `Q5_K`, `Q6_K`, and `Q8_0` block payloads inside the generated
            MLIR reduction. Package-loaded external Q4_K weights and dequantize-reference Q5_K/Q6_K/Q8_0 parity are
            covered without introducing a compiler-to-ggml link dependency.
      - [x] Validate the target model's complete tensor-format inventory: the Qwen2.5-Coder 14B Q4_K_M acceptance model
            contains 289 Q4_K and 49 Q6_K tensors, both covered by CPU AOT; no additional block format is required.
      - [x] Add the first corresponding CUDA native block projection path: `Q8_0` output-major UInt8 storage now lowers to
            an MLIR/NVPTX-generated CUDA kernel, supports variable/external-weight constant buffers, and has artifact plus
            runtime CUDA parity coverage without silently relabeling a CPU bridge as native.
      - [x] Extend the CUDA native correctness slice to `Q4_K`: output-major UInt8 storage now lowers through the same
            MLIR/NVPTX block-kernel path and has CUDA runtime parity coverage against a deterministic Q4_K fixture.
      - [x] Extend the CUDA native correctness slice to `Q6_K`: target-model output-major Q6_K projection now uses the
            MLIR/NVPTX block-kernel path and has CUDA runtime parity coverage against packed low/high-bit scale payloads.
      - [x] Extend the CUDA native correctness slice to `Q5_K`: the Q4_K metadata decoder is shared with Q5_K high-bit
            plane handling, with artifact and CUDA runtime parity coverage.
      - [x] Replace the initial unrolled CUDA Q8_0/Q4_K/Q5_K/Q6_K dot codegen with a loop-based kernel before using very
            large hidden sizes as the benchmark target. The generated MLIR now carries the outer GGML block dimension as
            `scf.for`, so PTX/code object size no longer scales with `blocksPerRow`; lane-level unrolling remains until a
            later warp/shared-memory tuning pass.
- [x] Add the importer-side CPU GGML kernel adapter and a direct output-major quantized MatMul primitive. It reuses
      vendored ggml `from_float`/`vec_dot` traits, quantizes only the current activation row, and consumes Q4_K/Q5_K/Q6_K
      and related block rows without materializing the complete Float32 weight.
- [x] Add first-class `QuantizedMatMulNode` graph/schema/package semantics and Interpreter kernel injection. Quantized
      LLaMA Linear lowering now emits this node directly, and Q4_K package roundtrip execution is covered without full
      weight dequantization.
- [x] Add CPU reference dequantized execution for all GGML block formats used by the target model, with memory-budget
      diagnostics for large models.
- [x] Add production-gated CUDA native quantized projection kernels for `Q4_K`/`Q5_K`/`Q6_K`/`Q8_0`. Correct native
      execution, artifact reporting, and loop-based large-hidden codegen are covered; warp-tiled/shared-memory tuning is
      tracked as post-G16 performance work rather than a correctness blocker.
- [x] Report `Q4_K_M` mixed-model coverage as CUDA-native quantized for Q4_K/Q6_K projection formats while keeping
      full-model decode-loop and golden-logit acceptance gates explicit.
- [x] Add CPU AOT parity tests for affine and packed-nibble quantized projection lowering, plus an explicit diagnostic
      test for unsupported FP4 packed MatMul lowering.
- [x] Add parity tests comparing native GGML/CUDA quantized projection with ggml dequantize-plus-float reference:
      CUDA Q8_0/Q4_K/Q5_K/Q6_K projection tests now build the expected output by materializing a dequantized float
      weight matrix and running the normal float MatMul reference.
- [x] Add a fallback policy matrix: reject, CPU reference dequantize, CUDA dequantize-then-GEMM, or native quantized CUDA.

#### G16.5 CUDA Native Coverage for Full Decode

- [x] Cover the full decode-step operator set in CUDA native or explicit bridge form: embedding/get-rows, RMSNorm, RoPE,
      Q/K/V projections, KV append/view, attention score, causal mask, softmax, value aggregation, SwiGLU, residuals, and
      output projection.
      - [x] Softmax f32 now has a CUDA native MLIR/NVPTX correctness slice with artifact feature reporting and runtime
            parity for both last-axis and non-last-axis static shapes.
      - [x] Embedding/GetRows f32 now has a CUDA native MLIR/NVPTX correctness slice for Int32 and Int64 token ids;
            frozen embedding tables are carried as artifact constant data and runtime parity covers rank-2 token-id
            batches. Runtime out-of-range index reporting remains part of the full decode validation work.
      - [x] Last-axis RMSNorm f32 now has a CUDA native MLIR/NVPTX correctness slice with optional per-channel scale,
            frozen-scale artifact constants, and runtime parity. Warp-reduction tuning and RMSNorm+Linear fusion remain
            performance work after full decode correctness is connected.
      - [x] RoPE is now a first-class graph node instead of a Reshape/Slice/trigonometric expansion; static offsets and
            runtime Int32/Int64 positions share Interpreter, CPU AOT, package, validation, and CUDA native semantics.
            CUDA artifacts carry inverse-frequency tables as constant data and cover static/dynamic runtime parity.
      - [x] BatchMatMul f32 now has a CUDA native MLIR/NVPTX correctness slice with broadcast batch-dimension support,
            artifact feature reporting, and runtime parity. This covers the unfused attention score and value-aggregation
            building blocks while tiled/shared-memory or cuBLAS batched GEMM remains a performance follow-up.
      - [x] CUDA native binary f32 now accepts runtime input, frozen variable, or constant tensor operands, so static
            causal/additive masks can stay in native CUDA artifact constant data instead of forcing a CPU bridge.
      - [x] KV-cache capacity update now has a CUDA native `ScatterNode` correctness slice for f32 axis-0 single-index
            `Update` writes with Int32/Int64 positions. It removes the immediate CPU bridge from the decode cache write
            path; multi-index scatter, scatter-add, and fused in-place KV writeback remain performance/fusion follow-ups.
      - [x] SiLU now has a CUDA native MLIR/NVPTX correctness slice by recognizing the standard `Layer::AddSiLU`
            expansion and lowering it to one elementwise kernel. Full SwiGLU MLP fusion across gate/up/down projections
            remains a separate fused-kernel target.
- [x] Add the fused-kernel production boundary: correctness-stable unfused CUDA native coverage is complete for the decode
      operator set, while RMSNorm+Linear, RoPE+Q/K layout, attention softmax/value aggregation, and quantized Linear
      epilogues are tracked as post-G16 performance work instead of hidden prerequisites for correctness claims.
- [x] Add CUDA graph replay or equivalent launch amortization for steady-state decode:
      `CompiledModuleCUDARunOptions::GraphReplay()` captures/replays CUDA-native payloads for stable tensor bindings, and
      `CompiledModuleCUDATest.RunsNativeLinearChainWithCUDAGraphReplay` validates repeated replay output parity.
- [x] Record per-token latency, launch count, memory bandwidth, and fallback count in benchmark/profile output:
      GGUF decode-loop summaries expose per-token latency plus fallback count, and `litenn_profile` now writes
      `cuda_profile.csv` with kernel/library/PTX launch counts, estimated bytes/run, and native-vs-graph replay GB/s.

#### G16.6 Golden Validation and User-Facing Example

- [x] Add an external llama.cpp golden capture harness:
      `scripts/gguf_capture_llamacpp_golden.py` records `llama-debug --save-logits` prompt/logit artifacts plus optional
      fixed-seed `llama-cli` generated text into a manifest-backed artifact directory.
- [x] Add a LiteNN replay harness for captured llama.cpp prompts:
      `scripts/gguf_run_litenn_from_golden.py` reads `*-prompt.txt` token ids from the capture manifest and runs
      `--run-llama-decode-loop-token-ids`, producing a colocated LiteNN replay manifest and output files.
- [x] Add automated LiteNN-vs-llama.cpp prefill-logits comparison:
      `scripts/gguf_compare_llamacpp_logits.py` replays captured prompt token ids through LiteNN, dumps last-token logits,
      compares against llama-debug `index: value` logits, and emits max-error/mismatch JSON.
- [x] Extend automated LiteNN-vs-llama.cpp golden comparison to first decode logits and multi-token decode logits:
      decode loops emit full-prompt and subsequent decode positions through `--logits-output-dir` while omitting costly
      prompt-intermediate dumps; replay manifests classify prefill and one-based decode steps; the comparator verifies
      exact prompt/generated token prefixes before comparing all common full-vocabulary logits artifacts.
- [x] Add an API-level llama.cpp decode-logits capture helper as an isolated CMake project:
      `tools/llamacpp-adapter` builds `litenn_llamacpp_adapter`, which consumes exact prompt/generated token ids, calls
      `llama_decode` token by token, and emits
      one full-vocabulary logits file after each generated token without adding llama.cpp to LiteNN's build graph.
- [x] Add a manifest-backed llama.cpp decode capture driver:
      `gguf_capture_llamacpp_decode_logits.py` runs the isolated helper and records model, prompt ids, generated ids,
      decode-step positions, producer, and artifact paths using `litenn.llamacpp_decode_logits.v1`.
- [x] Wire exact-token decode parity into the Qwen smoke flow:
      `qwen_smoke.py --llamacpp-decode-golden-tool <path>` captures llama.cpp references from LiteNN's fixed replay
      token stream and runs first/multi-token numerical comparison in the same evidence report.
- [x] Add generated-text comparison for fixed llama.cpp captures:
      `scripts/gguf_compare_generation_text.py` compares llama-cli captured stdout with LiteNN replay output pieces and
      writes a JSON pass/fail report; full tokenizer parity remains a separate acceptance gate.
- [x] Add LiteNN decode-loop logits dump support:
      decode-loop commands accept `--logits-output <output.txt>` and write the final step's last-token logits in the same
      `index: value` text format as prefill logits dumping.
- [x] Add a Qwen2.5-Coder/Qwen-family smoke example:
      `example/gguf/qwen_smoke.py` accepts a GGUF path, externally tokenized prompt ids or an optional llama.cpp prompt
      capture, backend policy (`cpu-aot` for the current driver), max generated-token count, and an output file,
      then writes a manifest-style smoke report with all command/stdout/stderr evidence.
- [x] Add a token-id level GGUF smoke CLI: `--run-llama-token-ids` imports a GGUF file, lowers fixed-length prefill with
      quantized weights preserved, executes through CPU AOT with external weight regions, and reports
      logits shape plus greedy next token.
- [x] Add a LiteNN prefill-logits dump CLI:
      `--dump-llama-token-id-logits` runs fixed-length token-id prefill and emits last-token logits as llama-debug-style
      `index: value` text for external golden comparison.
- [x] Add an exact-prompt GGUF smoke CLI: `--run-llama-prompt` tokenizes with the limited exact-vocabulary bridge, runs
      fixed-length prefill, and reports token ids, token pieces, logits shape, and greedy next token.
- [x] Hydrate vNext `ConstantNode` / `QuantizedConstantNode` payloads from saved package descriptors for direct
      Interpreter execution of saved `.ltnn` graphs with inline constants.
- [x] Add a saved-package token-id smoke CLI: `--run-llama-package-token-ids` loads a lowered `.ltnn` package, binds
      caller-provided token ids, executes through the same CPU AOT path, and reports greedy next token.
- [x] Add a token-id decode-loop smoke CLI: `--run-llama-decode-loop-token-id` repeatedly executes static decode graphs,
      builds one max-capacity decode plan, advances the runtime position state, carries full-capacity KV-cache tensors
      between steps, and reports build/run timing plus generated token ids and tokenizer pieces when
      `tokenizer.ggml.tokens` is available.
- [x] Add an externally tokenized prompt decode-loop CLI:
      `--run-llama-decode-loop-token-ids <input.gguf> <comma-token-ids> <steps>` bridges real tokenizer parity work by
      accepting full prompt token-id sequences before the optional llama.cpp tokenizer adapter is available.
- [x] Wire decode-loop sampling flags for greedy/random mode, temperature, top-k, top-p, repeat penalty, fixed seed, and
      explicit `--output` path while retaining the original positional output-path form.
- [x] Add an exact-prompt decode-loop smoke CLI: `--run-llama-prompt-decode-loop` tokenizes a fixture/diagnostic prompt,
      fills KV state through static decode steps, then generates the requested number of new tokens with the configured
      sampler.
- [x] Add decode-loop generation guards for EOS early stop, explicit `--ignore-eos`, prompt/generated-token accounting,
      and model context-length rejection before plan construction.
- [x] Add decode-loop backend/timing observability to the GGUF CLI:
      `--run-llama-*-decode-loop` summaries now include backend, fallback count, executed steps, per-step min/avg/max,
      ms/generated-token, and generated tokens/s for the current CPU AOT runner. CUDA-native/bridge rows remain
      gated on the executable CUDA stateful decode runner.
- [x] Remove Interpreter execution from GGUF model-running CLI paths, add CPU AOT lowering for Q4_K/Q5_K/Q6_K/Q8_0
      quantized embedding gathers, and keep large model weights borrowed while constructing the compiler graph.
      External weight regions now reserve once and move from temporary artifacts into loaded modules; a real 14B Q4_K_M
      compile held private memory near 25.7 GiB instead of growing beyond 44 GiB.
- [x] Profile and reduce real 14B CPU AOT compile latency enough to complete the current decode smoke path:
      `LITENN_COMPILE_DIAGNOSTICS=1` now exposes split MLIR CPU compile phases. Profiling showed the monolithic decode
      graph spent excessive time in MLIR buffer deallocation alias analysis and then LLVM lowering/object emission. The
      capacity decode graph now lowers each transformer block as a `CallNode` subgraph, keeping per-function allocation
      and control-flow lowering smaller. A real Qwen2.5-Coder-14B Q4_K_M single-token CPU AOT O0 smoke completed with
      fallback_count=0; the remaining bottleneck is runtime throughput and optimized-object emission, not an Interpreter
      fallback.
- [ ] High priority: make real 14B GGUF first-token latency interactive instead of compile-bound:
      the current CPU AOT path still spends minutes in LLVM lowering/object emission for the first decode artifact.
      `litenn_gguf_convert` has an opt-in `LITENN_GGUF_AOT_CACHE_DIR` separated-artifact cache, but first-run cache
      population is not yet suitable as a default because the instruction object is large. Preferred next cuts are
      per-block/per-layer reusable decode artifacts, a persistent package-level compile cache that borrows GGUF weight
      regions instead of rewriting them, and routing the user-facing smoke path to CUDA native once full decode evidence
      is available.
      - [x] Remove the CPU AOT runtime-schedule wrapper's projected-output scratch buffers for state aliases:
            projected state outputs now write directly to the aliased input buffers, avoiding stack allocation of large
            KV-cache tensors and unblocking further stateful/logits-only decode experiments on large LLM cache shapes.
      - [x] Expose the logits-only runtime-schedule decode path in the GGUF CLI with an explicit `--stateful` switch
            and separate AOT cache keys. A real Qwen2.5-Coder-14B Q4_K_M single-generated-token smoke completed through
            this path with `decode_mode=stateful`, `fallback_count=0`, generated token `Hello`, and no stack overflow;
            build time was still about 170s and run time about 4.1s for the full prompt replay plus one generated token,
            so this validates state alias execution but does not yet solve first-run interactivity.
      - [x] Add CPU AOT MLIR/LLVM module-size diagnostics under `LITENN_COMPILE_DIAGNOSTICS=1`: the compiler now reports
            MLIR op/function/block counts after LiteNN lowering, bufferization, and LLVM lowering, LLVM
            function/declaration/block/instruction/global counts after translation/wrapper/optimization, and emitted
            object bytes. This gives the next split-artifact/per-block compile pass hard evidence instead of relying only
            on coarse phase timers.
            A real Qwen2.5-Coder-14B Q4_K_M stateful decode smoke showed about 3.76M lowered MLIR ops, 2.47M LLVM
            instructions, 51 LLVM functions, and an 8.5MB object. The largest functions are repeated block functions
            around 50.9k LLVM instructions each, so the next compile-time reduction should target reusable/per-block
            decode artifacts or shared block functions rather than wrapper cleanup.
- [x] Replace generic MLIR expansion of GGML K-quant MatMul with a reusable CPU AOT helper call:
      `QuantizedMatMulNode` GGML_Q4_K/Q5_K/Q6_K/Q8_0 lowering now tags the generated op, the LLVM codegen pipeline
      replaces it with `litenn_cpu_ggml_block_matmul_f32`, and regression tests require the helper symbol in the
      instruction object while preserving CPU AOT parity. This creates the sidecar-kernel boundary needed to remove
      per-projection scalar IR expansion, but the first helper is correctness-first and not yet a speed win on the real
      14B O0 smoke.
- [x] Keep GGML block quantized MatMul/GetRows MLIR lowering lightweight before helper replacement:
      GraphToMLIR now emits dependency-carrying placeholder linalg ops for GGML_Q4_K/Q5_K/Q6_K/Q8_0 MatMul/GetRows
      instead of expanding the full bitfield decode into MLIR and waiting for the LLVM pipeline to replace it. A real
      14B Q4_K_M O0 single-token smoke improved from the previous sidecar path's ~256s build to ~147s build while
      keeping fallback_count=0.
- [x] Replace generic MLIR expansion of GGML K-quant GetRows with reusable CPU AOT helper calls:
      `QuantizedGetRowsNode` now lowers to `litenn_cpu_ggml_block_get_rows_i32_f32` /
      `litenn_cpu_ggml_block_get_rows_i64_f32` for GGML_Q4_K/Q5_K/Q6_K/Q8_0 token embedding gathers, with regression
      coverage requiring the helper symbol in the instruction object.
- [x] Add first CPU-side parallelism to `litenn_cpu_ggml_block_matmul_f32` so decode-shaped `m=1, n=large`
      projections can split output columns across the existing CPU worker pool instead of running as a single host loop.
      The real 14B Q4_K_M O0 single-token smoke improved from the previous helper-call run's ~97.6s runtime to ~15.9s,
      while keeping fallback_count=0.
- [x] Upgrade the first `litenn_cpu_ggml_block_matmul_f32` hot loop from scalar `kk` decoding to block/lane traversal:
      the helper now dispatches by GGML format outside the inner lane work, accumulates in float, and walks each
      quantized block once per output column without per-lane divide/modulo address recomputation. A real 14B Q4_K_M O0
      single-token smoke improved runtime further to ~7.8s with fallback_count=0.
- [x] Add the first production-shaped GGML CPU sidecar kernel layer:
      `litenn_cpu_ggml_block_matmul_f32` now uses block-level Q4_K/Q5_K/Q6_K/Q8_0 dot kernels that parse scale/min
      metadata once per block or subblock instead of calling element decoders for every lane; `QuantizedGetRowsNode`
      sidecar output also decodes whole GGML blocks at a time. CPU LLVM codegen now passes `CompilerOptions`
      thread-count and affinity-policy constants into the GGML MatMul helper, and the GGUF CLI accepts
      `LITENN_CPU_AOT_AFFINITY=compact` through the same options path used by benchmarks. A real 14B Q4_K_M O0
      single-token smoke improved runtime from ~7.8s to ~0.46s with fallback_count=0, while build time moved from
      ~147s to ~139s.
- [ ] Finish the remaining GGML CPU microkernel work with dedicated SIMD intrinsics, packed input-row reuse across
      output columns, and cache-friendly output tiling. Current 14B O0 smoke still spends ~56s in MLIR-to-LLVM lowering
      and ~47s in object emission, so the next compile-time win is reducing the remaining non-quantized function/output
      surface rather than changing Interpreter fallback behavior.
      - [x] Split the CPU sidecar Q4_K and Q5_K block-dot loops so the hot Q4_K path no longer pays the Q5 high-bit
            branch, and unroll Q8_0/Q4_K/Q5_K/Q6_K lane loops in groups of four. Correctness was validated with the
            GGUF quantized execution tests; a real 14B before/after run is still needed before claiming an end-to-end
            speedup.
      - [x] Add `litenn_bench` rows for `GGMLBlockMatMulHelper/{Q8_0,Q4_K,Q5_K,Q6_K}/T{1,16}` at a decode-shaped
            `batch=1, in=4096, out=4096`, isolating the CPU sidecar helper from graph lowering and object emission.
      - [x] Cache Q4_K/Q5_K activation subblock sums once per input block in `litenn_cpu_ggml_block_matmul_f32` and
            reuse them across output columns. This is the first activation-side reuse slice before full Q8_K staging.
      - [x] Add four-output-column tiling to the direct Q8_0/Q4_K/Q5_K/Q6_K CPU helper path so decode-shaped
            projections can reuse each Float32 activation scan across adjacent output columns before the full Q8_K
            activation-staged kernel replacement.
- [x] Make `benchmark/gguf_decode_compare.py` consume GGUF decode observability fields so comparison tables carry
      backend identity, fallback count, explicit ms/generated-token, and generated-token throughput instead of only
      deriving throughput from `run_ms`.
      Decode-loop CLI output now also separates prompt replay from generation steps; `ms_per_generated_token` and
      `generated_tokens_per_second` are based on generation-step time, while `run_ms` remains the end-to-end replay/run
      wall time.
- [x] Add a CPU stateful decode artifact path: `--lower-llama-decode-stateful` emits a vNext package plus external
      weights, and tests load that package, compile CPU AOT, execute it, and compare against Interpreter output.
- [x] Add a CPU AOT state-aware entry wrapper for runtime schedules: `Compiler<CPU>::CompileArtifact(RuntimeSchedule)`
      now exposes only public outputs in artifact metadata, while wrapper scratch buffers receive functional state outputs
      and copy them back into the aliased input state buffers after the compiled call.
- [x] Make executable plan snapshots keep borrowed graph variables alive without duplicating model weights:
      `BuildExecutablePlanFromGraph` now stores the source `Variable` shared owner in `BufferRegion::owner`, so plans
      and runtime schedules created from temporary graphs or builder-local lowered graphs no longer depend on graph
      lifetime while large GGUF weights remain borrowed instead of deep-copied.
- [x] Fix the promoted-constant subset of CPU AOT external-region lowering:
      `constant.*` variables now lower through the same constant path as `ConstantNode` when they remain inline, and
      generic CPU externalization keeps promoted constants below `CompilerOptions::cpuAOTExternalConstantMinBytes`
      inside the MLIR module while still externalizing normal weights. `CompiledModuleTest` covers the mixed case.
- [x] Finish full GGUF stateful schedule parity by removing constant promotion from the runtime schedule builder:
      `BuildLLaMADecodeRuntimeSchedule` now leaves constants as constants and only real model variables participate in
      external-weight packaging, which keeps logits-only CPU AOT entries aligned with the direct capacity schedule path.
- [x] Add CUDA bridge/native stateful decode artifact examples and separated instruction/weight region commands:
      `example/gguf/build_stateful_artifacts.py` lowers a package with external weights, emits separated CPU and optional
      CUDA carrier objects, records the actual compiler backend, and supports `native-required`, `bridge-allowed`,
      `optional`, and `disabled` CUDA policies without hiding CPU bridge selection.
- [x] Gate “production supported” status on matching golden logits within dtype/quantization tolerance and on a non-hidden
      fallback report: `scripts/gguf_production_gate.py` defaults to CUDA-native, requires explicit no-fallback evidence,
      validates requested prefill/decode/text comparison reports, emits a machine-readable decision, and fails closed
      when artifact or parity evidence is missing.
- [x] Add evidence-driven LLM decode comparison-table tooling:
      `benchmark/gguf_decode_compare.py` consumes LiteNN Qwen smoke reports, llama-bench JSON, and equivalent PyTorch/HF
      rows, then emits JSON/CSV/Markdown with ms/token, token/s, fallback state, and same-device-class percentage deltas.
      It intentionally omits unavailable backends instead of relabeling bridge or synthetic data.
- [x] Keep comparison-table population evidence-driven for LiteNN CUDA-native/bridge decode rows:
      `benchmark/gguf_decode_compare.py` consumes real CUDA/bridge evidence when supplied and intentionally omits
      unavailable backends instead of relabeling CPU Interpreter smoke results. Real Qwen CUDA rows remain an external
      benchmark artifact requirement, not a synthetic repo fixture.

#### G16.7 Long-Context LLM Runtime Target: 1M Tokens

Purpose: treat `max_tokens=2048` as a small smoke size, not a scalability ceiling. The production target is at least
1M-token context with compile time, artifact size, cache population, and runtime memory all scaling by model structure
and active KV pages rather than by a fully unrolled static max-cache-length graph.

Priority classes:

- P0, highest immediate generated-token latency benefit: real-shape projection benchmarks, activation-side
  reuse/staging, Q8_K activation-staged quantized projection kernels, projection-specific thread/grain policy, grouped
  QKV projection, and grouped SwiGLU gate/up projection.
- P1, long-context scalability blockers: verified in-place KV append, paged KV-cache state, grouped active-prefix
  attention, and long-context attention plans.
- P2, measurement and acceptance gates: operator-level timing, long-context benchmark/profile rows, and golden
  validation. These are required to avoid guessing, but they are not the largest direct runtime reducers.

- [x] Add first-class compile diagnostics for long-context smoke runs:
      qwen smoke logs must stream to disk while running, preserve partial evidence after interruption, and emit
      Chrome Trace / Perfetto-compatible waterfall JSON for import, tokenize, schedule build, MLIR/LLVM compile,
      object emission, separated-cache population, JIT/load, prompt replay, and per-token decode phases.
      Completed on 2026-07-02: Qwen smoke now writes streaming stdout/stderr evidence, `qwen_smoke_trace.json`, and
      `qwen_smoke_waterfall.md`; CPU AOT compile diagnostics report externalization, MLIR lowering/bufferization,
      LLVM lowering, instruction counts, object emission, cache/load, and decode-loop per-step timing. The latest
      Qwen2.5-Coder-14B Q4_K_M `max_cache_length=2048` smoke completed successfully with active-prefix attention and
      dynamic RoPE helper lowering, producing token `9707` / `Hello`.
- [x] Record a focused Qwen CPU AOT decode profile against the user's llama.cpp CPU reference:
      `docs/PerformanceAnalysis_2026-07-02.md` compares the active-prefix/RoPE-helper LiteNN path with the reported
      `0.1-0.2 s/step` llama.cpp run. Local LiteNN evidence shows about `522 ms` for `max_cache_length=10`, about
      `722 ms` for `max_cache_length=2048`, and worse latency when forcing `LITENN_CPU_AOT_THREADS=16`. The documented
      gap is now tied to concrete execution structure: capacity-shaped overhead remains, but the larger
      capacity-independent gap is the GGML block projection helper's direct Q4_K/Q6_K x Float32 design versus
      llama.cpp's Q8_K activation staging, quantized vec-dot kernels, tiled scheduling, and packed/repacked CPU kernels.
- [x] Build a local CPU-only llama.cpp GGUF control path before doing more CPU tuning:
      completed on 2026-07-06 with an out-of-tree `llama-bench` build from `third_party/llama.cpp`, configured with
      CUDA/Vulkan/BLAS disabled. MinGW required explicit Windows 10 API compile flags for the bundled `cpp-httplib`.
      TG-only Qwen2.5-Coder-14B Q4_K_M measurements showed llama.cpp at about `5.07 t/s` for T4, `4.60 t/s` for T8,
      `3.52 t/s` for T16 with `b=1/ub=1`, `3.34 t/s` for T16 with default `b=2048/ub=512`, and `2.62 t/s` for T32.
      LiteNN stateful CPU AOT measured about `0.49 t/s` at T4 and about `2.33 t/s` with its default thread policy on
      the same short prompt. The priority remains CPU projection/scheduler parity with llama.cpp rather than CUDA work
      or blanket thread-count retuning.
- [x] P2: Add decode operator-level profiling before claiming the next full-step bottleneck:
      collect per-layer/per-node/per-helper timings with node kind, helper symbol, GGML block format, shape, thread
      count, cache length, and generated-token phase. This must separate RMSNorm, Q/K/V/O projections, MLP gate/up/down
      projections, active-prefix attention, state alias copies, final logits projection, and sampler/text output.
      - [x] Add operator/role attribution to GGUF decode profile bundles. Completed on 2026-07-05:
            `profile_bundle.py` classifies helper events into projection, attention, position_encoding, kv_update,
            embedding, normalization, and other roles, emits ranked operator totals, and annotates per-step
            `top_operator` plus trace event args. This is a bundle-level attribution bridge; true per-layer/per-node
            runtime timing remains open.
      - [x] Add a helper-derived node timing schema. Completed on 2026-07-05:
            `gguf_decode_summary.json` now emits `node_timings` with node kind, node name, helper symbol, GGML format,
            input/output shapes, requested/resolved thread counts, calls, total/average milliseconds, and explicit
            `attribution=helper-derived`. At completion time this closed the initial P2 observability gate without
            pretending the runtime had stable per-layer node ids; native ids are completed by the marker work below.
      - [x] Split module helper and non-helper decode time. Completed on 2026-07-07: GGUF decode `--stream-stats`
            emits `helper_total_ms` and `module_non_helper_ms`, and the profile bundle plus decode comparison tools
            preserve those fields as runtime buckets and comparison columns. This isolates generated-code/runtime-entry
            time from sidecar helper time before true per-layer non-helper timing is available.
      - [x] Add native stable per-node timing. Completed on 2026-08-02: opt-in CPU AOT instrumentation records
            subgraph/node/schema identity and operation kind with inclusive, helper, and exclusive self time.
            `litenn_gguf_convert --profile-nodes` and `profile_bundle.py` expose per-node rows, ranked native node-kind
            totals, and trace events without changing normal artifacts. A cache-hit 14B diagnostic run reduced the
            unknown warm non-helper bucket to about `27 ms/step`; corrected native self attribution is led by about
            `10 ms` of UnaryOp work while nested CallNode frames account for about `2.7-2.8 ms`. Profile output cost is
            reported separately and these intrusive rows are not throughput evidence.
- [x] P0: Add production-shaped GGML helper benchmark and estimator coverage:
      benchmark `batch=1` rows for real Qwen decode projection shapes (`5120->5120`, `5120->1024`,
      `5120->13824`, `13824->5120`, and `5120->152064`) and report the full-step projection estimate. The current
      `4096->4096` helper rows do not by themselves explain a 48-layer, 337-projection decode step.
      Completed on 2026-07-02: `litenn_bench` registers `baseline4096`, `qwen_hidden`, `qwen_kv`, `qwen_ffn_up`,
      `qwen_ffn_down`, and `qwen_logits` rows for `GGMLBlockMatMulHelper/{Q8_0,Q4_K,Q5_K,Q6_K}/T{1,16}`. A short
      smoke run verified `GGMLBlockMatMulHelper/Q4_K/qwen_kv/T1/batch:1/in:5120/out:1024`.
- [x] P0: Cache Q4_K/Q5_K activation subblock sums in the CPU GGML block MatMul helper:
      `litenn_cpu_ggml_block_matmul_f32` now computes the eight 32-lane activation sums once per input block and
      reuses them across output columns. The split hot path keeps the no-cache fallback for correctness-style direct
      calls and avoids a branch inside the innermost lane loop. Short benchmark validation on 2026-07-02 moved
      `GGMLBlockMatMulHelper/Q4_K/qwen_kv/T1` from about `3.03 ms` to about `1.91 ms`; `Q4_K/baseline4096/T1` measured
      about `6.09 ms` after the change.
- [x] P0: Tile the Q4_K direct CPU helper across four output columns:
      `litenn_cpu_ggml_block_matmul_f32` now shares each Float32 activation scan across four Q4_K output columns before
      moving to the numerically different Q8_K staging path. Short validation on 2026-07-02 passed the stateful GGUF
      logits parity test and measured `GGMLBlockMatMulHelper/Q4_K/qwen_kv/T1` at about `1.73 ms` CPU and
      `Q4_K/baseline4096/T1` at about `5.79 ms` CPU; the same `qwen_kv` helper row measured about `0.33 ms` CPU at
      `T16`, keeping helper-level parallelism as a valid but shape/grain-sensitive optimization.
- [x] P0: Add a complete-output-group fast path for direct Q4_K/Q6_K helpers:
      completed on 2026-07-06 after rejecting a wider `x8` output grouping experiment that regressed Qwen-shaped rows.
      The retained path skips repeated tail-validity checks for the common full `x4` output group while preserving the
      old partial-tail logic. Short helper validation measured Q4_K default rows at about `0.70 ms` for `5120->5120`,
      `1.64 ms` for `5120->13824`, `17.5 ms` for logits, and grouped gate/up concatenated at about `3.32 ms`; a real
      Qwen2.5-Coder-14B Q4_K_M stateful smoke with `max_cache=11` measured about `506 ms/generated token`.
- [x] P0: Tile the Q6_K direct CPU helper across four output columns:
      the same grouped-output helper path now covers Q6_K FFN-down-style projections without changing accumulation
      semantics. Short validation on 2026-07-02 passed the stateful GGUF logits parity test and measured
      `GGMLBlockMatMulHelper/Q6_K/qwen_ffn_down/T1` at about `49.9 ms` CPU and the matching `T16` row at about
      `4.58 ms` CPU, confirming Q6_K still needs the higher-priority Q8_K/vec-dot replacement for single-thread
      latency.
- [x] P0: Add benchmark-only prepared-weight sidecars for Q4_K/Q6_K projection helpers:
      completed on 2026-07-07 as evidence for the production packed-weight tranche without switching the default AOT
      route. The new C ABI prepares output-major Q4_K/Q6_K blocks into float scale metadata plus expanded quant lanes,
      validates with `GGUFLLaMAQuantizedExecution.Q4KPrepackedHelperMatchesDirectHelper` and
      `Q6KPrepackedHelperMatchesDirectHelper`, and exposes
      `GGMLBlockMatMulPrepackedHelper/{Q4_K,Q6_K}/...` plus `GGMLBlockMatMulPrepackWeight/...` benchmark rows. Short
      Qwen-shaped rows measured Q4_K `qwen_ffn_up/T1` from about `30.0 ms` direct to `22.7 ms` prepacked and Q6_K
      `qwen_ffn_down/T1` from about `33.4 ms` direct to `15.0 ms` prepacked, both with `max_abs_delta=0`; prepack-only
      rows measured about `6.94 ms` for Q4_K `5120->13824` and `14.9 ms` for Q6_K `13824->5120`, so production routing
      should cache prepared weights in the separated shared weight store.
- [x] P0: Route prepared Q4_K/Q6_K weights through separated CPU AOT for ordinary projection nodes:
      completed on 2026-07-07 for non-grouped `QuantizedMatMulNode` RHS variables. The opt-in
      `CompilerOptions::enableCPUAOTGGMLPrepackedWeights` writes prepared Q4_K/Q6_K payloads into the separated weights
      region, GraphToMLIR marks prepared RHS placeholders, LLVM lowering calls the matching prepared helper, and the GGUF
      decode CLI/smoke path exposes `--cpu-aot-ggml-prepacked-weights` with cache-key isolation. Validation passed the
      Q4/Q6 helper parity tests and the output-major Q4_K/Q6_K/Q8_0 CPU AOT regression with prepared artifacts. Grouped
      projection routing and default enablement remain gated on full-decode A/B evidence.
- [x] P0: Route prepared Q4_K/Q6_K weights through separated CPU AOT for grouped projection nodes:
      completed on 2026-07-07 for `GroupedQuantizedMatMulNode` Q/K/V-style and gate/up-style RHS variables. The opt-in
      prepack planner now handles each projection RHS independently with projection-local expressed shapes, GraphToMLIR
      accepts consistent prepared grouped storage, and LLVM lowering calls
      `litenn_cpu_ggml_block_grouped_matmul{2,3}_{q4k,q6k}_prepacked_f32` without materializing a concatenated weight.
      Validation passed the grouped Q4_K AOT regression with prepacked external weights, plus the existing Q4/Q6 helper
      parity and output-major quantized-matmul regressions. `litenn_bench` also has
      `GGMLGroupedProjectionPrepackedHelper/{Q4_K,Q6_K}/...` rows; a short gate/up `T8` smoke measured about `4.78 ms`
      real for Q4_K and `4.93 ms` real for Q6_K, both with `max_abs_delta=0`. Default enablement remains gated on
      full-decode A/B evidence.
- [x] P0: Add an evidence-gated prepared-weight policy for CPU AOT GGML decode:
      completed on 2026-07-07 with `CompilerOptions::cpuAOTGGMLPrepackedWeightPolicy` and CLI/smoke/matrix support for
      `disabled|profitable|all`. The legacy `--cpu-aot-ggml-prepacked-weights` switch still means all-format opt-in;
      `profitable` currently routes only Q6_K through separated prepared weights because the isolated helper evidence is
      strong there while Q4_K is still mixed. `gguf_decode_compare.py` records the policy in comparison configs, and
      `GGUFLLaMAQuantizedExecution.CPUAOTPrepackedWeightPolicyRoutesOnlyProfitableFormats` locks the Q6-on/Q4-off
      behavior until full-decode A/B justifies changing the default.
      Follow-up completed on 2026-07-07: `benchmark/gguf_decode_thread_matrix.py` accepts
      `--cpu-aot-ggml-prepacked-weight-policies disabled,profitable,all`, emits policy-separated work directories, and
      adds a Policy column so one run can capture the thread x prepared-weight-policy full-decode matrix.
      Profile completed on 2026-07-07: local Qwen2.5-Coder 14B Q4_K_M `T8`, `max_tokens=8`, stateful CPU AOT
      measured disabled `1232.2 ms/token`, profitable `1063.1 ms/token`, and all `861.6 ms/token`; a CPU-only
      `llama-bench` `T8`, `ngl=0`, `flash_attn=0` control measured `235.1 ms/token`. Prepared weights therefore win
      in full decode, but all-prepared Q4_K/Q6_K still reaches only about `27%` of llama.cpp throughput and expands the
      shared weight cache to about `17.93 GB`, so default selection remains a memory/speed policy decision.
      Follow-up analysis recorded in `docs/PerformanceAnalysis_2026-07-07.md`: a cache-hit phase profile and llama.cpp
      source audit confirm that LiteNN is not mainly behind in host overhead, active-prefix attention, KV append, or
      Interpreter fallback. The gap is still concentrated in Q4_K/Q6_K projection helpers. The next production tranche
      is therefore compact repacked Q4_K/Q6_K x Q8_K GEMV/vec-dot kernels with step-level activation staging, not a
      blanket thread-policy change.
      Follow-up kernel slice completed on 2026-07-08: prepared Q4_K/Q6_K helpers now use a runtime-gated AVX2
      `lhsColumnStride == 1` path for complete x4 output-column groups while preserving scalar tails. Validation passed
      the Q4/Q6 prepacked helper parity tests and the prepared-weight policy AOT regression. Short Qwen-shaped `T8`
      helper smokes measured grouped gate/up at about `3.67 ms` for Q4_K and `4.23 ms` for Q6_K. This improves the
      current expanded prepared layout, but does not close the compact-layout requirement because shared prepared
      weights are still much larger than raw GGUF.
      Layout-tagging slice completed on 2026-07-08: prepared layout markers are now integer ids, LLVM lowering rejects
      unknown ids, and separated weight names include `.prepacked.expanded_f32_scales_v1.<format>`. This gives compact
      v3 repacks an explicit ABI split point before new layouts are introduced.
      Cache/report isolation slice completed on 2026-07-08: GGUF decode AOT cache keys, shared-weight cache keys,
      qwen-smoke reports, decode comparison labels, and thread-matrix rows now include the prepared layout token.
      AVX2 reduction slice completed on 2026-07-12: prepared Q4_K/Q6_K kernels now perform their 8-lane Float32
      horizontal reduction entirely in registers. Q4_K stayed neutral in short T8 smokes; Q6_K hidden improved from
      about `0.757 ms` CPU to `0.558 ms`, FFN-up from about `1.98 ms` to `1.74 ms`, and grouped gate/up from about
      `4.23 ms` to `3.91 ms`.
- [x] P0: Replace expanded prepared GGML decode weights with compact llama.cpp-class repacked layouts:
      add versioned compact Q4_K/Q6_K prepared layouts for decode projections, keep shared prepared weights close to
      raw GGUF size, and route AOT placeholders to the matching helper ABI only when the layout tag matches.
      Runtime prototype completed on 2026-07-25: a 64-byte v3 header plus x4 block-grouped payload keeps Qwen-shaped
      storage at effectively `1.00x` raw GGUF size. The new prepack/runtime ABI, tail handling, JIT symbols, parity test,
      and benchmark rows cover Q4_K/Q6_K. Reusing the mature byte-stride-1 vec-dot makes Q6_K faster than staged on
      the tested T8 production rows; Q4_K is mixed and expanded-v1 remains the speed gate.
      Single-projection AOT slice completed on 2026-08-01: `CompilerOptions` now exposes an explicit expanded-v1 or
      compact-v3 prepared-weight layout, externalization writes and names the matching physical payload, GraphToMLIR
      emits compact layout id `3`, and LLVM dispatches only that id to the compact Q8_K helper. Q4_K/Q6_K artifact-load
      execution and payload-size regressions pass. Expanded-v1 remained the default while full-decode policy data was
      collected.
      Evaluation-surface slice completed on 2026-08-01: GGUF CLI users can select
      `--cpu-aot-ggml-prepacked-weight-layout expanded-v1|compact-v3`; the smoke report, decode cache, and shared-weight
      cache record the selected canonical layout. The thread matrix now supports a layout axis in addition to policy
      and thread axes, so compact-v3 can be evaluated on a real cache-hit decode without artifact contamination.
      Grouped AOT slice completed on 2026-08-01: compact-v3 now supports two- and three-way
      `GroupedQuantizedMatMulNode` through `litenn_cpu_ggml_block_grouped_matmul{2,3}_compact_q8k_f32`, staging each
      activation row once for all projections. `QuantizedStorageLayout` is now an explicit graph/plan/package/rodata-v5
      semantic instead of a byte-size guess, which removes the Q4_K width-2 ambiguity where compact-v3 and expanded-v1
      both occupy 640 bytes. Q4_K/Q6_K grouped AOT execution and all affected regression suites pass.
      Controlled acceptance matrix completed on 2026-08-01: Qwen2.5-Coder 14B Q4_K_M stateful CPU AOT at O0/T8 with
      all prepared weights measured expanded-v1 at `482.108 ms/token` (`2.074 tok/s`) and the first compact-v3 kernel at
      `626.098 ms/token` (`1.597 tok/s`). Compact-v3 cuts separated weights from `17,929,588,736` to `8,982,164,544`
      bytes (`-49.9%`), cache-population work from `91.433 s` to `49.335 s` (`-46.0%`), and sampled peak working set from
      about `33.8` to `20.4 GiB`, but the initial latency regression was `29.9%`; it remains opt-in.
      SIMD follow-up completed on 2026-08-01: compact Q4_K/Q6_K x4 AVX2 kernels now decode nibbles/bitplanes and reduce
      integer dot products entirely in registers. The same cache-hit run improved to `567.607 ms/token`
      (`1.762 tok/s`), reducing latency by `9.34%`; step-16 helper time fell from `592.755` to `528.082 ms` with identical
      output and no fallback. The remaining expanded-v1 latency gap was `17.7%` after this slice.
      Paired-dot follow-up completed on 2026-08-01: complete x4 blocks now evaluate two output columns in each 256-bit
      AVX2 dot, sharing Q8_K loads, multiply-adds, zero-point correction, and reduction. The same run improved again to
      `505.917 ms/token` (`1.977 tok/s`), another `10.87%` latency reduction and `19.2%` cumulatively from the first
      compact kernel; step-16 helper time fell to `475.246 ms`. Compact-v3 is now only `4.94%` slower than expanded-v1
      while retaining `49.9%` lower prepared-weight storage. It remains opt-in pending repeated acceptance and an x8
      field-interleaved scale/quant reuse kernel.
      Field-interleaved v4 kernel prototype completed on 2026-08-01: a distinct v4 header plus prepack/runtime ABI
      interleaves four-byte quant chunks across eight output rows. Q4_K uses `1.0278x` raw block bytes for directly
      loadable scale/min vectors and Q6_K remains `1.00x`; portable full/tail and AVX2 x8 paths match staged output
      exactly. Relative to paired-dot v3, T8 rows measured Q4 hidden `0.506 vs 0.801 ms`, Q4 FFN-down
      `1.36 vs 2.16 ms`, Q6 FFN-down `1.74 vs 2.76 ms`, and Q6 logits `14.3 vs 20.0 ms`. JIT symbols and benchmark
      coverage are complete.
      AOT wiring completed on 2026-08-01: explicit layout id `4` is carried through quantization metadata, Plan/MLIR
      storage validation, LLVM helper selection, externalized prepared-weight generation, cache identity, and the
      `field-interleaved-v4` CLI/script token. Single projection and grouped 2/3-projection helpers load v4 payloads
      directly; grouped helpers stage the shared Q8_K activation once and keep per-projection x8 boundaries. Q4_K/Q6_K
      single/grouped AOT load-and-run coverage passes in the complete 81-test GGUF importer suite.
      Real 14B acceptance completed on 2026-08-01 with stateful O0/T8/all-prepared eight-token cache-hit decode. V4
      stores `9,160,094,784` separated-weight bytes, only `1.98%` above compact-v3 and `48.91%` below expanded-v1.
      Two no-fallback runs produced identical tokens at `403.054` and `449.863 ms/token` (`2.481` and `2.223 tok/s`),
      beating compact-v3 by `20.3%` and `11.1%` and expanded-v1 by `16.4%` and `6.7%`. Initial cache population spent
      `27.306 s` compiling, `7.199 s` building metadata, and `19.840 s` writing weights. Field-interleaved-v4 is now
      the compiler and smoke-driver default when prepared GGML weights are enabled; v1/v3 remain explicit selections.
      A 256-bit AVX512VL/VNNI `vpdpbusd` prototype was rejected on 2026-08-01: it was neutral for Q4_K and regressed
      Q6_K FFN-down/logits medians from `1.86/13.4 ms` to `2.01/15.7 ms` on Ryzen 9 9950X, with up to about `9.8e-4`
      extra Float32 difference. The path was removed; future AVX512 work must justify a true x16 layout/kernel instead
      of substituting one instruction in the x8 loop.
      The CPU dynamic-RoPE helper now caches one thread-local frequency/angle table for the most recent
      `(headDim, base, frequencyScale, position)`. This removes repeated `pow/sin/cos` work across the 2304 Q/K head
      calls in a stateful Qwen token while keeping memory bounded independently of context length. A cache-hit 14B T8
      profile reduced this row from about `15.9` to `0.287 ms/step` (`-98.2%`) with identical generated tokens and no
      fallback. The remaining dispatch cost is too small to justify a separate batching ABI without new evidence.
      Field-interleaved-v4 single projections now also share a bounded, content-validated Q8_K activation workspace.
      This lets mixed-format Q/K/V helpers reuse one quantization without trusting arena pointers: a Float32 bitwise
      snapshot invalidates the cache on in-place mutation. Two real 14B T8 cache-hit runs kept exact generated tokens
      and no fallback; Q4_K/Q6_K KV helper rows dropped from about `38.35/13.14 ms` to `25.13/9.21 ms` and
      `11.84/4.67 ms`, while generated-token latency ranged from `341.854` to `427.627 ms/token` under visible host
      frequency variance.
      A true AVX2 x16 execution tile now shares Q8_K loads across two adjacent v4 x8 groups without changing the weight
      ABI. Blanket use was rejected after a real 14B run exposed regressions in 1024-column KV and ordinary Q4_K rows.
      The retained dispatch is deliberately evidence-gated to single projections with at least 32768 output columns;
      grouped and narrower projections stay on x8. Q4_K/Q6_K logits microbench medians improved from about `10.3/15.1`
      to `8.69/13.3 ms`. A selective no-fallback acceptance run reproduced the exact token sequence at
      `360.743 ms/token` (`2.772 tok/s`) and reduced the step-16 Q6_K logits row from `16.486` to `14.049 ms`, while
      whole-step latency remained inside the observed host-frequency range.
      The v4 AVX2 kernels now use a separate AVX2+F16C runtime gate and convert eight packed FP16 scales with one
      `vcvtph2ps`; CPUs without both features retain the portable implementation. Exact-parity T8 medians improved
      Q4_K hidden/up/down from `0.355/0.709/0.715` to `0.283/0.549/0.508 ms` and Q6_K from
      `0.413/1.16/1.15` to `0.338/0.970/1.00 ms`. A real 14B cache-hit run generated the identical token sequence with
      no fallback at `314.807 ms/token` (`3.177 tok/s`), a `12.7%` latency reduction from the preceding selective-x16
      run and `7.9%` below the prior best `341.854 ms/token` scalar-conversion run.
- [x] P1: Deduplicate shared prepared-weight stores by physical content independently of artifact ABI metadata:
      explicit layout metadata correctly isolates incompatible artifacts, but it can also create a second shared store
      when the physical expanded-v1 bytes are unchanged. Introduce a content/layout payload identity separate from the
      compile/cache key, preserve strict ABI validation in artifact metadata, and regression-test concurrent population
      plus reuse so a metadata-only compiler change does not duplicate multi-GB weights.
      Completed on 2026-08-02: the payload identity is sorted by physical offset and contains only total bytes plus each
      weight range's offset, size, and content checksum. Tensor names, declared alignment, metadata ordering, compiler
      flags, and artifact versioning no longer create duplicate payloads, while offsets and checksums still isolate
      incompatible bytes. Shared weights are written into unique sibling staging directories and atomically renamed only
      after both `weights.bin` and `complete` exist; concurrent losers validate and reuse the winner, and failed staging
      is removed. Two cache-specific tests and all 45 targeted cache/quantized/decode tests pass on Windows.
- [x] P0: Add step-level Q8_K activation staging for decode projections:
      stage each normalized hidden vector once per layer/step and reuse it across compatible Q/K/V/O, gate/up/down, and
      logits projections. The current per-helper staged prototype remains opt-in until paired with compact vec-dot
      kernels and validated on full decode.
      First slice completed on 2026-07-08: added an explicit reusable Q8_K activation C ABI
      (`litenn_cpu_ggml_prepare_q8k_activation_f32`,
      `litenn_cpu_ggml_block_matmul_q8k_prepared_activation_f32`) plus parity coverage and benchmark rows. The new path
      matches internal staging for Q4_K/Q5_K/Q6_K and gives the graph/AOT lowering a concrete workspace target for the
      next slice.
      Second slice completed on 2026-07-08: added grouped 2-way/3-way prepared-activation ABI
      (`litenn_cpu_ggml_block_grouped_matmul{2,3}_q8k_prepared_activation_f32`), grouped parity coverage, and grouped
      benchmark rows. The short Q6_K `qwen_gate_up/grouped/T8` smoke measured prepared activation at about `17.1 ms`
      versus `16.5 ms` for internal staged grouped helper, confirming grouped helpers already avoid repeated lhs staging;
      remaining high-return work is fallback elimination plus compact vec-dot kernels.
      AOT slice completed on 2026-07-08: LLVM lowering now selects
      `litenn_cpu_ggml_block_grouped_matmul{2,3}_q8k_staged_f32` for non-prepacked GGML_Q6_K grouped projections when
      `enableCPUAOTGGMLQ8KStagedMatMul` is enabled, with both 2-way and 3-way groups covered by
      `GGUFLLaMAQuantizedExecution.CompilesGroupedQ6KProjectionToQ8KStagedHelper`.
      Kernel slice completed on 2026-07-08: Q6_K x Q8_K AVX2 now has an all-valid x4 output-column fast path for
      common no-tail decode rows. Short aggregate smokes measured `qwen_gate_up/grouped/T8` staged at about `6.72 ms`
      mean versus `15.8 ms` direct grouped, and `qwen_qkv/grouped/T8` staged at about `1.85 ms` mean versus `4.02 ms`
      direct grouped.
      Follow-up completed on 2026-07-08: Q4_K/Q5_K gained the same all-valid AVX2 fast path and AOT staged lowering now
      covers Q4_K/Q5_K/Q6_K single and grouped projections. Short smokes measured Q4_K `qwen_gate_up/grouped/T8`
      staged at about `4.52 ms` versus `7.62 ms` direct, Q5_K at about `5.33 ms` versus `14.1 ms`, and single
      `qwen_ffn_down/T8` staged wins for Q4_K/Q5_K/Q6_K.
      Production closure completed on 2026-08-02: field-interleaved-v4 uses a bounded, content-validated per-thread
      Q8_K workspace for single projections and stages once for grouped projections. Mixed-format Q/K/V reuse, exact
      output parity, and the full 43-test quantized/decode regression set pass without fallback.
- [x] P0: Implement production Q4_K/Q6_K x Q8_K GEMV/vec-dot kernels:
      target the real Qwen decode rows first (`1x5120 -> 1x27648`, `1x5120 -> 1x5120`, `1x13824 -> 1x5120`,
      `1x5120 -> 1x1024`, and `1x5120 -> 152064`), then repeat the cache-hit LiteNN-vs-llama.cpp comparison before
      changing default prepared-weight or thread policy.
      Wide-output progress completed on 2026-08-01: the v4 single-projection helper has an AVX2 x16 tile that shares
      activation loads across adjacent x8 groups, but real-model evidence limits it to output widths >=32768. This closes
      the logits-specific slice; gate/up, hidden/output, FFN-down, and KV still require dedicated low-thread kernels, so
      the overall P0 remains open.
      A grouped-v4 benchmark now covers Q/K/V and gate/up shapes across T0/T1/T2/T4/T8/T16/T32. It showed that a Q4_K
      x8 grouped gate/up call computes both 13824-column outputs in about `2.26 ms` at T1, while one 13824-column x16
      single projection took about `2.21 ms`; this evidence raised the x16 threshold from 8192 to 32768 and prevents FFN
      projections from entering the slower tile. The corrected single Q4_K FFN-up row measured `1.34/0.359 ms` at
      T1/T8, about `39%/35%` faster than the former x16 route, with exact reference parity.
      F16C progress completed on 2026-08-01: vector FP16 scale conversion accelerates both x8 and x16 v4 kernels behind
      an AVX2+F16C feature gate and reduced real 14B generated-token latency to `314.807 ms`. The step-16 helper total is
      still about `288 ms`, so further work remains concentrated in quantized projection kernels rather than host code.
      Scheduling progress completed on 2026-08-02: v4 single/grouped helpers now create about four dynamic tasks per
      requested worker instead of eight. This keeps enough work stealing for T8/T16 imbalance while reducing atomic
      task claims and improving contiguous weight access. The same cache-hit 14B T8 profile preserved the exact token
      sequence and reduced generated-token latency from `307.082` to `299.370 ms/token` (`-2.51%`); the final grouped
      gate/up, Q6_K down, Q4_K down, and hidden rows improved by about `5.8%`, `4.9%`, `2.7%`, and `4.5%` respectively.
      Keeping the x8 accumulator live across every K block was also tested and rejected: it regressed real decode from
      `307.082` to `342.163 ms/token`, consistent with register-pressure and instruction-scheduling costs that isolated
      microbenchmarks did not expose.
      Static contiguous worker partitioning was also tested and removed on 2026-08-02. Giving each v4 participant one
      grain-aligned range regressed T8 grouped Q4_K gate/up from `1.31` to `1.42 ms`, Q4_K FFN-up/down from
      `0.330/0.321` to `0.388/0.375 ms`, and Q6_K hidden from `0.173` to `0.226 ms`. Keep the measured dynamic
      four-tasks-per-thread schedule; its work stealing helps cross-domain balance on the reference host.
      A true AVX-512 Q6_K x16 tile completed on 2026-08-02. It combines adjacent v4 x8 groups in 512-bit vectors,
      shares Q8_K broadcasts, and is selected only after an AVX512F+BW+VL+F16C runtime check; AVX2 x8 remains the
      portable x86 fallback and Q4_K routing is unchanged. Exact parity and all 42 targeted quantized/decode tests pass.
      Production-shaped T8 medians improved hidden `0.249 -> 0.173 ms`, FFN-up `0.841 -> 0.746 ms`, FFN-down
      `1.03 -> 0.730 ms`, and logits `13.5 -> 12.3 ms`. The cache-hit 14B acceptance run preserved the exact generated
      tokens with no fallback at `260.166 ms/token` (`3.844 tok/s`), `1.57%` below the immediately preceding
      `264.313 ms/token` baseline. This validates true x16 width as distinct from the rejected 256-bit VNNI substitution,
      while the broader production-kernel P0 remains open for Q4_K and further grouped-projection reductions.
      A corresponding Q4_K AVX-512 x16 tile was tested and removed on 2026-08-02. Broad routing regressed FFN and logits
      microbenchmarks; even a square-only gate raised the real 14B 48-call hidden bucket from roughly `25.9` to
      `28.65 ms/step` and step-16 helper time from about `234.4` to `254.6 ms`. The faster unprofiled sample was host/cache
      variance rather than attributable kernel gain. Future Q4_K work must change the scale/minimum decomposition or
      packed representation instead of directly mirroring the successful Q6_K x16 kernel.
      A Windows-only SysV internal calling-convention experiment removed all Win64 nonvolatile-XMM saves from the v4
      kernel disassembly, but changed register allocation enough to regress real decode from `299.370` to
      `345.568 ms/token` (`+15.4%`). Keep the native Win64 ABI until a whole-loop kernel can be evaluated without this
      allocator tradeoff.
      Block-level integer reduction completed on 2026-08-02: Q4_K/Q6_K v4 x8 and x16 kernels now keep scaled dot sums
      (plus the Q4_K minimum correction) in Int32 for each complete 256-element block and perform one final Float32
      conversion/scale, following the production x86 kernel structure. All 42 targeted GGUF quantized/decode tests
      passed. The same cache-hit 14B O0/T8/all/v4 profile preserved exact tokens with no fallback and improved from
      `299.370` to `277.218 ms/token` (`-7.40%`, `3.607 tok/s`); step-16 helper time fell from about `288` to
      `243.026 ms`, including grouped gate/up `95.197 -> 85.156 ms`, Q6_K down `50.331 -> 45.163 ms`, Q4_K down
      `40.532 -> 35.745 ms`, and hidden/output `30.062 -> 27.824 ms`. A Q4_K odd/even subblock-template variant was
      rejected because eliminating the nibble branch expanded the x8 kernel and regressed production-shaped
      hidden/down/grouped rows by roughly `17-40%`.
      Re-testing whole-K x8 accumulators after the integer-reduction change kept all 42 targeted tests green but
      regressed production-shaped Q4_K T8 hidden/FFN-up/FFN-down medians to `0.216/0.399/0.431 ms` and Q6_K to
      `0.241/1.27/1.24 ms`; it remains rejected due to register pressure and code-generation loss. A no-helper-profile
      production run measured `278.863 ms/token` versus `277.218 ms/token` with profiling, ruling out instrumentation
      overhead as the material remaining gap.
      A Q6_K x8 paired-segment experiment was also rejected: sharing `ql/qh` loads for segment pairs improved the T8
      hidden median from `0.265` to `0.212 ms`, but regressed the dominant FFN-up/down rows from `0.802/0.896` to
      `1.12/1.17 ms` and nearly doubled their T1 latency. Keep the lower-register-pressure per-segment kernel.
      Pair-sum/scale folding completed on 2026-08-02: Q4_K x8 replaces separate Int32 pair reduction and scale
      multiplication with safe Int16 pair accumulation plus one `vpmaddwd`; Q6_K x8 and AVX-512 x16 use two bounded
      Int16 partial sums before the same signed-scale fold. An immediate helper A/B reduced step-16 helper time from
      `244.051` to `238.362 ms`, grouped Q4_K gate/up from `87.409` to `84.598 ms`, and hidden/output from
      `27.500` to `26.348 ms`. Q6_K x16 T8 hidden and FFN-up/down medians improved from about
      `0.173/0.746/0.730` to `0.112/0.714/0.711 ms`, with logits neutral at about `12.3 ms`. Three no-fallback 14B
      cache-hit runs generated identical text at `259.299`, `249.367`, and `254.126 ms/token`; their
      `254.126 ms/token` median is `2.32%` below the preceding stable `260.166 ms/token` result. A Q4_K paired-nibble
      load variant was removed because its cache-hot microbench gain became a real-helper regression under increased
      register and scheduling pressure.
      Shape-aware v4 decode thread caps completed on 2026-08-02. T4/T8/T16 production-shape medians showed Q4_K
      hidden favoring T4 (`0.116 ms` versus `0.173/0.247`), Q4_K FFN-down favoring T8 (`0.311 ms`), and grouped
      gate/up, Q6_K FFN-down, and Q6_K logits favoring T16 (`1.19/0.597/11.4 ms`); 1024-column Q4_K/Q6_K rows
      favored T2. Explicit thread counts remain a hard upper bound and may select T16, while automatic decode policy
      caps general projections at T8, square Q4_K hidden projections at T4, and small outputs at T2. Prefill/batch and
      sub-1M-operation work retain the generic policy. Profiler details expose the resolved count and regression
      coverage locks the Q4_K square `T8 -> T4` decision. A real 14B profile reduced the 48-call hidden bucket from
      `26.348` to `25.186 ms`; three no-profile cache-hit runs generated identical text at `245.097`, `245.177`, and
      `249.406 ms/token`. Their `245.177 ms/token` median is `3.52%` below the preceding `254.126 ms/token` median and
      within `4.3%` of the `235.1 ms/token` CPU control result.
      The final automatic-policy matrix rejected an automatic T16 ceiling: it measured `299.960 ms/token`, while the
      corrected automatic T8 ceiling measured `285.087 ms/token` under profiling. Alternating unprofiled auto/explicit
      T8 runs remained host-frequency sensitive (`279-289` versus `268-274 ms/token`), so explicit T16 remains available
      for controlled experiments but is not the production default.
- [ ] P0: Close the remaining CPU decode gap against the strongest controlled CPU-only llama.cpp reference:
      `docs/QwenCPUDecodePerformanceEvidence_2026-08-04.md` is the canonical evidence record and
      `docs/PerformanceAnalysis_2026-08-04.md` retains the detailed profiling narrative. Projection/worker phase
      evidence is recorded separately in `docs/QwenCPUDecodeProjectionProfile_2026-08-04.md`; controlled build/runtime
      evidence is in `docs/QwenCPUDecodeBuildControl_2026-08-04.md`. The adjacent-run baseline is
      `256.616 ms/token` for LiteNN and
      `202.224 ms/token` for llama.cpp at T8. Normalized stage attribution assigns at least `46.54 ms` of the
      `54.39 ms/token` gap to FFN, with Q4_K and Q6_K activation-plus-Down each roughly twice the corresponding
      llama.cpp boundary. Gate/Up is comparatively close and Attention is not the immediate short-context owner.
      The implementation checklist is maintained in `docs/PerformanceOptimizationRoadmap.md` under the
      2026-08-04 FFN-Down closure tranche.
      - [x] Add a cache-cold projection-stream benchmark whose rotating weight set exceeds LLC and can replay the
            observed 48-layer Qwen Q4_K_M projection order. The Release T8 benchmark now reports bytes, effective
            bandwidth, weighted warm/cold ratio, grouped/single mode, requested/resolved threads, unique activations,
            reference delta, and per-shape totals. Median Q4_K x24, Q6_K x24, and real mixed x48 times were `41.175`,
            `53.221`, and `96.726 ms`; the same mixed weights with one shared activation took `64.065 ms`.
      - [x] Implement and evaluate FFN SwiGLU-to-Down fusion. The compiler marks only single-consumer, non-public
            rank-2 Float32 SwiGLU values and lowers compatible field-v4 Q4_K/Q6_K consumers to the fused helper.
            Runtime, object import/load, and AOT execution parity pass. Two independent paired mixed-stream runs changed
            sign (`+0.3185 ms` and `-0.0696 ms` materialized-minus-fused), so fusion is neutral within noise. The earlier
            `32.661 ms` distinct/shared difference also included prepared Q8_K reuse and cache/access changes and is no
            longer treated as removable handoff cost.
            - [x] Mark only single-consumer SwiGLU values that are not public results before bufferization.
            - [x] Lower the marked pair to the fused field-v4 Q4_K/Q6_K runtime helper and verify object imports.
            - [x] Pass runtime, artifact-load, and AOT execution parity and complete a paired mixed-format cold stream.
      - [x] Instrument FFN-Down worker dispatch, useful work, and barrier wait at low overhead. Stable steps 10-24 of a
            real cache-hit 14B run measured `45.2007 ms/step` for Q8_K activation quantization, `14.9425 ms` dispatch,
            `77.7852 ms` parallel wall time, and `3.8463 ms` final barrier wait; lookup, copy, and lock contention were
            negligible. FFN-Down quantization alone accounts for `32.893 ms/step`. Exact-size microbenchmarks agree with
            production quantization within `7.1%`, so Q8_K preparation is now the first measured implementation owner.
      - [x] Replace scalar Q8_K rounding with a byte-exact nearest-integer path. Exact 5120/13824-element Release
            medians improved from `235/640 us` to `5.59/15.3 us` (`41.8-42.0x`); all 19 focused quantized-execution
            tests pass. The real cache-hit profile reduced ordinary-projection quantization from `45.2007` to
            `1.1865 ms/step`, profiled stable latency from `266.887` to `184.275 ms/token`, and preserved the token
            sequence with no fallback. Three no-helper-profile stable averages were `197.748`, `195.874`, and
            `200.156 ms/token`, a `22.94%` median reduction from the preceding `256.616 ms/token` baseline.
      - [x] Re-rank grouped Gate/Up, Q6_K Down, Q4_K Down, hidden/output, and logits against a fresh CPU-only llama.cpp
            stage control before selecting the next kernel. Manual activation-quantizer SIMD is deferred because only
            `1.1865 ms/step` remains in the structured phase and is no longer P0.
            The original GNU/OpenMP comparison suggested parity, but a controlled build matrix invalidated that
            conclusion: GNU/no-OpenMP is `16.21%` faster and Clang/no-OpenMP reaches `6.03 t/s`. A strong paired control
            places LiteNN `5.49%` behind the Clang/no-OpenMP reference. The old stage profile remains historical because
            it used the slower reference and must not select another kernel.
      - [ ] Reproduce the external `6.85 t/s` CPU-only result with a redacted, repeatable actual-completion control.
            Record the llama.cpp commit, compiler and ISA flags, thread/affinity/polling policy, mmap and priority,
            KV dtype, context and decode lengths, and exact command without retaining the private model path. Run an
            adjacent LiteNN cache-hit decode over the same generated-token window; only a measured stage deficit may
            promote the next kernel or scheduler change to P0.
            - [x] Add `benchmark/run_llama_cpp_completion_control.py` with binary/host/build fingerprints, balanced
                  thread ordering, per-run progress and metrics, and full artifact path/prompt redaction by default.
            - [x] Establish a clean raw-prompt control: two balanced repetitions measured T2/T4/T8 at
                  `5.080/4.810/4.115 t/s`. This is retained as a reproducibility row rather than the primary comparison.
            - [x] Align Qwen chat formatting and the 15-token steady module/eval boundary. Bracketing llama.cpp runs
                  both measured `5.500 t/s`; LiteNN measured `5.640/5.112/5.718 t/s`, with a `5.640 t/s` median
                  (`+2.54%`) but materially higher run variance. The first 9 generated tokens match before ignore-EOS
                  semantics diverge.
            - [x] Suppress EOS during LiteNN `--ignore-eos` greedy/random sampling to match llama.cpp. A real post-fix
                  Qwen run produced byte-identical 16-token text, no fallback, `5.685 t/s` full generation, and
                  `5.765 t/s` aligned steady-module throughput versus llama.cpp's `5.500 t/s` median.
            - [x] Automate paired alternating controls with host frequency/power-state capture and a variance gate.
                  `benchmark/run_paired_gguf_decode_control.py` now enforces the aligned 9-prompt/15-eval window,
                  byte-identical output, no fallback, binary identity, redaction, alternating order, and a 3% CV gate.
                  The earlier GNU/OpenMP batches remain harness evidence only; their positive LiteNN conclusion is
                  superseded by `docs/QwenCPUDecodeBuildControl_2026-08-04.md`.
            - [x] Add multi-binary runtime sweeps and compare GNU/OpenMP, GNU/no-OpenMP, and Clang/no-OpenMP from the
                  same llama.cpp commit. Medians were `5.06`, `5.88`, and `6.03 t/s`; sampled actual frequency stayed
                  within `5077-5086 MHz`, so OpenMP explains most of the gain and compiler choice adds `2.55%`.
            - [x] Sweep T1/T2/T3, physical-core/same-core/cross-CCD affinity, poll, and priority controls. No positive
                  policy change exceeded 1%; the `-36.27%` same-core SMT negative control proves masks were effective.
            - [x] Pair the strongest local Clang/no-OpenMP reference with LiteNN. All gates passed; llama.cpp and
                  LiteNN medians were `5.970` and `5.518 t/s`, and LiteNN's preferred paired median difference is
                  `-5.49%`. Actual frequency does not explain the deficit.
            - [ ] Reproduce the remaining external `6.85 t/s` difference. It is now `13.60%` above the strongest local
                  llama.cpp sweep rather than `25.23%` above the old GNU/OpenMP reference.
                  - [x] Capture local build/ISA/OpenMP/thread/affinity/polling/mmap/KV/context settings, hashes, and
                        actual frequency in redacted artifacts.
                  - [ ] Obtain the external binary or its remaining exact source/build/runtime differences.
                  - [ ] Require two paired batches to pass the variance, output-parity, and no-fallback gates.
            - [ ] P0 evidence gate: profile matched Attention, FFN Gate/Up, FFN Down, logits, dispatch, and residual
                  boundaries against Clang/no-OpenMP over the same 9-prompt/15-eval window. Repeat with GNU/no-OpenMP
                  to separate compiler and OpenMP effects; only the largest measured deficit may become implementation
                  P0.
            - [ ] Extend paired decode evidence to 128/512 generated-token windows and then 2K/32K/128K/1M context
                  tiers as paged-KV support matures; report sustained throughput, memory residency, and cache growth.
            - [ ] Add optional effective-cycle, LLC-miss, memory-stall, and bandwidth evidence. Windows processor-power
                  frequency remains policy metadata and must not be treated as proof of equal effective clocks.
      - [x] Reject grouped Q4_K x16 Gate/Up after the target median changed only `1.14 -> 1.11 ms` within `~4%` noise
            and CPU time regressed slightly.
      - [x] Reject an `atomic::wait/notify_one` worker path: dispatch improved `21.7%`, but parallel wall/barrier costs
            increased and total profiled latency regressed `0.54%`.
      - [ ] Amortize the measured `14.9425 ms/step` dispatch floor across compatible helper sequences. Lock and barrier
            tuning remain secondary unless a new profile changes their ranking.
      - [ ] Implement evidence-gated Down-path experiments: interleaved output-group streams, software prefetch,
            Q4_K AVX2 x16 selection, and Q6_K AVX2/AVX-512 selection. Reject variants that win only in the cache-hot
            helper benchmark.
      - [ ] Re-run alternating LiteNN/llama.cpp full-decode and stage profiles using each runtime's measured production
            thread policy. Acceptance requires no fallback, unchanged generated output, prepared weights no larger
            than `1.03x` source quantized bytes, the promoted stage within `10%`, and total latency within `5%` of the
            same-run llama.cpp median.
- [ ] P1: Harden CPU decode performance evidence after controlled gap closure:
      Windows PDH actual-frequency and utility sampling is complete. Add optional PMU/platform-counter capture for
      cache misses, memory stalls, bandwidth, and residency; preserve the out-of-tree llama.cpp stage control recipe;
      and add non-gating warm/cold trend rows. Platform profiling must remain optional on CI hosts without privileges.
- [x] P1: Skip vocabulary projection on prompt replay steps that cannot be sampled:
      completed on 2026-08-02 with an `emit_logits` Bool in stateful dense and paged-reference schedules. The compiled
      `CondNode` returns zero logits during replay while still updating KV/position aliases, then executes the unchanged
      lm-head for the final prompt token and generation. The CLI selects the branch automatically, the public output ABI
      remains logits-only, and decode artifact cache v7 isolates the new input ABI. A real six-token 14B Q4_K_M profile
      recorded no vocabulary helper in any of five replay steps and exactly one `13.997 ms` Q6_K logits helper in the
      generation step; all 91 `GGUFImporterTest` cases pass with no fallback in the real-model run.
- [x] P0: Tile the Q8_0 and Q5_K direct CPU helper paths across four output columns:
      the grouped-output helper now covers all currently supported GGML direct MatMul formats. Validation on 2026-07-02
      passed `GGUFLLaMAQuantizedExecution.*`; a short helper run measured `Q8_0/qwen_kv/T1` at about `2.03 ms` CPU and
      `Q5_K/qwen_kv/T1` at about `3.67 ms` CPU.
- [x] P0: Implement and evaluate Q8_K activation-staged GGML block MatMul kernels:
      keep the existing direct Float32 helper as the production default/reference path unless a format-specific staged
      kernel has measured wins, then add Q4_K/Q5_K/Q6_K/Q8_0 x Q8_K kernels with cache-friendly output tiling and
      architecture-specific packed/repacked variants where available.
      - [x] Add a scalar Q8_K-staged helper prototype and benchmark rows without switching the default AOT helper.
            Validation on 2026-07-03 passed exact-activation parity for Q4_K/Q5_K/Q6_K. Short helper measurements showed
            the scalar staged path is not a default-switch candidate yet: Q4_K `qwen_kv/T1` was slower (`~2.78 ms` CPU
            staged vs `~2.60 ms` CPU direct in that run), while Q6_K `qwen_ffn_down/T1` only improved modestly
            (`~66.4 ms` CPU staged vs `~70.3 ms` CPU direct) and carries activation-quantization deltas.
      - [x] Add a guarded AVX2 16-lane dot primitive for the Q8_K-staged Q4_K/Q5_K/Q6_K path and keep it behind
            runtime CPU feature detection. Validation on 2026-07-03 passed
            `GGUFLLaMAQuantizedExecution.Q8KStagedHelperMatchesDirectHelperForExactActivationRows`; short helper
            measurements showed the AVX2 staged path helps Q6_K single-thread (`qwen_ffn_down/T1` about `41.7 ms` CPU
            staged vs `44.3 ms` CPU direct in that run) but Q4_K/Q5_K remained slower than the direct helper. Do not
            switch the default globally until the policy is format-specific and accuracy-aware.
      - [x] Add an explicit CPU AOT opt-in route from GGML_Q6_K matmul placeholders to the Q8_K-staged sidecar, while
            keeping the default path on the numerically stricter direct helper. Q4_K/Q5_K/Q8_0 remain direct until they
            have measured production-shape wins. The GGUF decode CLI exposes this as `--cpu-aot-q8k-staged-matmul`
            for A/B profiling.
      - [x] Move the main GGUF decode CPU AOT tuning controls onto explicit CLI flags:
            `--cpu-aot-threads`, `--cpu-aot-affinity`, `--cpu-aot-llvm-opt-level`,
            `--cpu-aot-parallel-min-flops`, and `--compile-diagnostics` / `--no-compile-diagnostics`.
            These options are included in the decode AOT cache key when they affect generated artifacts.
      - [x] Guard CPU AOT O3 for state-alias decode schedules. Completed on 2026-07-05: Qwen stateful decode with
            `--cpu-aot-llvm-opt-level 3` reproduced a Windows access violation on the first decode step, while O1/O2
            completed. CPU AOT now strips alias-sensitive LLVM attributes around state-alias entry wrappers and
            downgrades only state-alias schedules requested as O3 to effective O2 with an explicit compile diagnostic.
            Non-state-alias CPU AOT artifacts can still use O3. True O3 alias-safety proof remains a follow-up before
            reenabling O3 for mutable-state decode entries.
      - [x] Add VNNI, repacked-weight, or other architecture-specific vec-dot kernels for the Q8_K-staged path, then
            re-run the direct-vs-staged helper table before changing the compiler/runtime default. The current AVX2 path
            now uses an architecture-specific u8*s8 `maddubs` pairwise dot for Q4_K/Q5_K/Q6_K staged lanes. Validation
            passed `GGUFLLaMAQuantizedExecution.Q8KStagedHelperMatchesDirectHelperForExactActivationRows`; a short
            2026-07-04 helper run still rejected a global default switch (`Q4_K/qwen_kv/T1` direct `~3.12 ms` CPU vs
            staged `~15.6 ms` CPU, `Q6_K/qwen_ffn_down/T1` direct/staged both about `46.9 ms` CPU but staged worse in
            real time). Keep staged routing explicit/format-gated until a later packed-weight or VNNI kernel wins.
- [x] P0: Add grouped projection helpers for LLM decode:
      fuse or concatenate Q/K/V projection work where quantized storage formats permit, fuse the SwiGLU gate/up
      projections, and split outputs after the shared activation scan. This directly targets duplicated reads of the
      same normalized hidden vector across Q/K/V and gate/up helpers.
      - [x] Add grouped-projection benchmark rows that compare separate Q/K/V or gate/up helper calls against
            concatenated output-major GGML weights using the existing helper ABI. A short 2026-07-03 Q4_K run validated
            `max_abs_delta=0`; `qwen_qkv/T0` improved from about `1.37 ms` real separate to `1.07 ms` concatenated,
            while `qwen_gate_up/T0` improved from about `3.49 ms` to `3.17 ms`.
      - [x] Add AOT lowering that recognizes same-input compatible projection groups and emits a concatenated helper
            call or a multi-output sidecar without copying model weights at runtime. Completed on 2026-07-04 with a
            first-class `GroupedQuantizedMatMulNode`, Q/K/V and gate/up layer helpers, executable-plan round-trip
            support, CPU MLIR lowering to `litenn_cpu_ggml_block_grouped_matmul2_f32` /
            `litenn_cpu_ggml_block_grouped_matmul3_f32`, and a projection-span sidecar that accepts independent
            output-major GGML rhs memrefs. Validation passed
            `GGUFLLaMAQuantizedExecution.CompilesGroupedQ4KProjectionWithoutMaterializingConcatenatedWeight`,
            `GGUFLLaMACausalLM.PreservesQuantizedProjectionStorageWithQuantizedMatMulNodes`, and
            `GGUFLLaMAQuantizedExecution.*`.
      - [x] Make prepared layouts atomic across mixed-format projection groups. Completed on 2026-08-02: fresh AOT
            compilation prepackages every member when any compatible group member is selected, and repeatedly clears
            partially selected groups after shared-variable validation. This prevents profitable Q6_K selection from
            pairing prepared and source-layout operands in the same Q4_K/Q6_K grouped helper.
      - [x] Fuse exact SwiGLU activation and gate multiplication after grouped gate/up projection. Completed on
            2026-08-02 with a first-class `BinaryOp::SwiGLU`, exact Interpreter and Autograd semantics, MLIR lowering,
            and a strided rank-2 CPU AOT helper. The 48-layer decode graph now executes `288` fewer nodes
            (`8366 -> 8078`, `-3.44%`) and no separate unary expansion. Native profiling reduced the fused work from
            `13.08-15.76 ms/step` inline to `11.4-12.5 ms/step` including helper and residual node self time; an
            unprofiled cache-hit run excluding the first page-fault-heavy step averaged `273.341 ms/token` with exact
            generated-token parity. CPU AOT and decode-plan cache versions were advanced to prevent stale artifacts.
      - [x] Separate CPU AOT node-marker cost from production wrapper work. Completed on 2026-08-02: profiler snapshots,
            stream statistics, JSON, Markdown, and trace runtime buckets now expose aggregate node self, marker
            instrumentation, and remaining module-unattributed time. A four-step 14B cache-hit profile closed
            `17.76-19.79 ms` module non-helper time into `11.54-12.51 ms` node self, `5.66-6.28 ms` marker callbacks,
            and `0.56-1.11 ms` unattributed time. Stateful generation also reuses a preallocated logits tensor through
            `RunTensorsInto`, avoiding a `608256`-byte allocation on every token while preserving exact output ids.
      - [x] Validate selective decoder-block `CallNode` inlining with non-profiled A/B evidence. Rejected on
            2026-08-02: forced LLVM always-inlining reduced the real 14B module from 52 functions and 245883
            instructions to 2 functions and 140672 instructions, but object size grew by `2.46%` and compile-artifact
            time grew by `22.7%`. Exact token ids matched; interleaved no-inline versus inline runs measured
            `264.743/263.721 ms` versus `265.704/264.153 ms` mean/median, making inline `0.36%/0.16%` slower. The
            experimental implementation was removed and is not part of the public compiler options or cache ABI.
- [x] P0: Add a measured decode thread/grain model:
      `requestedThreadCount == 0` now uses an auto policy instead of blindly using every hardware thread: GGML block
      MatMul helpers cap at 16 workers by default, apply smaller caps for tiny output-group counts, and preserve explicit
      `T1/T2/T4/T8/T16/T32` requests. The helper benchmark matrix now includes `T0/T1/T2/T4/T8/T16/T32` rows. A short
      2026-07-03 run showed `T0` tracks the conservative cap (`Q4_K/qwen_kv` about `0.30 ms` real,
      `Q6_K/qwen_ffn_down` about `3.28 ms` real), while explicit `T32` remains available when isolated helper rows prove
      it wins.
      - [x] Remove per-helper all-worker wakeups from the persistent CPU AOT pool. Selected workers now receive
            directed signals and briefly poll for the next helper, while the calling thread uses a spin completion
            barrier. Regression coverage changes requested thread counts repeatedly in one process. On the 14B T8
            cache-hit run, step-16 helper time fell from `338.182` to `302.936 ms`, generation averaged
            `342.513 ms/token`, and output tokens remained identical.
      - [x] Make worker polling duration an explicit runtime scheduling policy with adaptive, low-power, and
            latency-sensitive modes. Completed on 2026-08-02: `CompilerOptions::cpuAOTWorkerWaitPolicy` is propagated
            through dense and GGML helpers without expanding the helper ABI, while the GGUF CLI, Qwen smoke driver,
            cache key, reports, and benchmark tables expose the policy explicitly. `Adaptive` grows or shrinks its
            bounded polling window from observed work arrivals, `LowPower` blocks immediately, and `Latency` keeps the
            longest bounded polling window. A real 14B Q4_K_M T8/all/v4/O0 16-token comparison measured
            Adaptive `255.927/255.594 ms`, LowPower `277.163/276.340 ms`, and Latency `252.958/252.650 ms`
            mean/median with identical generated-text hashes. Adaptive remains the balanced default; applications can
            select either extreme without relying on process environment state.
- [x] P1: Stop using monolithic max-cache-length-shaped CPU AOT decode artifacts as the default long-context path.
      Per-layer/per-block reusable decode artifacts or a shape-polymorphic stateful decode artifact must compile once
      per model architecture/weight layout, while runtime KV capacity is provided as state metadata.
      - [x] Make the GGUF decode-loop CLI default to the stateful runtime-schedule path instead of the functional
            cache-input/cache-output path. Completed on 2026-07-04: `--run-llama-*-decode-loop` now uses the logits-only
            public-output stateful schedule by default, while `--functional` remains available as an explicit
            compatibility/diagnostic path.
      - [x] Replace the max-cache-length-shaped paged-reference stateful function signature with page-table state
            bindings. Completed on 2026-07-05: `BuildLLaMADecodeRuntimeSchedule(... usePagedReferenceDecode=true)`
            now accepts `pagedResidentPageCount`; `litenn_gguf_convert --paged-reference-decode --compile-only`
            exposes it as `--paged-resident-pages`, includes it in the AOT cache key, and lowers KV backing as
            `[2,residentPages,pageSize,kvHeads,headDim]` while the page table remains sized by logical
            `max-cache-length`.
- [x] P1: Decouple persistent AOT instruction cache from model-weight storage.
      Cache hits should not require rewriting multi-GB GGUF weights into per-artifact `weights.bin`; the cache should
      borrow or map reusable weight regions, validate them by stable metadata, and keep instruction/metadata artifacts
      small.
      - [x] Deduplicate decode AOT cache weights across instruction-cache variants for the same source model. Completed
            on 2026-07-04: GGUF decode cache entries now use a model-level shared weight store and write
            `weights.path.txt` beside metadata/constants/instructions instead of rewriting a per-cache `weights.bin`.
            Legacy cache entries with local `weights.bin` still load. This removes repeated multi-GB writes when only
            AOT flags, thread policy, or decode mode change.
      - [x] Load decode AOT cache hits through borrowed separated regions. Completed on 2026-07-04:
            rvalue `CompiledModuleSeparatedArtifact::LoadBorrowedExternalRegions()` moves the separated artifact into
            the returned module as an owner and borrows constants/weights from it, so cache-hit loading avoids a second
            copy of the shared weight blob after reading it.
            Updated on 2026-07-05: GGUF decode cache hits map the shared weight store as a borrowed separated-artifact
            weights region instead of reading the multi-GB blob into a temporary vector.
      - [x] Isolate shared weights by compiled tensor layout/content identity. Completed on 2026-08-01: the shared
            store key initially included every external tensor's name, offset, size, alignment, and checksum, preventing
            a same-sized cache blob from being reused after projection grouping changes weight ordering. Refined on
            2026-08-02: physical payload identity now uses sorted offset/size/checksum tuples and total bytes, decoupling
            harmless tensor names, alignment declarations, and metadata order while retaining strict artifact ABI
            validation. Unique staging directories plus atomic rename make concurrent population corruption-safe.
      - [x] Add metadata-only, cache-first stateful startup and an explicit trusted-cache validation boundary.
            Completed on 2026-08-01: cache hits no longer import tensor payloads or rebuild the decode graph, and the
            compiled module ABI drives state/input allocation directly. Default separated-artifact loading remains
            checksum-strict; only complete, content-addressed internal cache entries skip the 9 GB weight scan while
            retaining structural and non-weight-region validation. On the 14B control, cache `build_ms` fell from
            about `31.4 s` to `18.8 ms`, with exact generated-token parity and peak working set reduced from about
            `26.8` to `8.4-8.8 GiB`.
      - [x] P2: Replace repeated cache-local weight blobs with borrowed/mapped shared weight regions. Completed on
            2026-07-05: cache hits mmap the model-level shared weight store and load it through borrowed separated
            regions, eliminating the extra read/copy on cache hit. Directly borrowing GGUF/source-package tensor
            offsets requires separated-artifact metadata that records stable source offsets/checksums rather than only
            compiled-weight-region offsets, so that narrower upgrade is deferred to the long-term artifact metadata
            queue.
- [x] P1: Add a verified in-place KV append helper:
      `litenn_cpu_scatter_update_axis0_f32_rank3` now has direct regression coverage for both same-buffer in-place
      append and distinct-output copy semantics, and stateful decode schedule coverage confirms projected cache outputs
      alias their input buffers. The `KVScatterUpdateHelper` benchmark records the cost boundary: on 2026-07-03,
      Qwen-shaped alias append rounded to `0.000 ms`, while copy mode measured about `0.210 ms` for a 2048-token cache
      and about `1.71 ms` real / `1.41 ms` CPU for an 8192-token cache.
- [x] P1: Replace dense full-capacity KV tensors with paged KV-cache state.
      The ABI needs page tables, active-length metadata, per-layer K/V page descriptors, and explicit ownership/eviction
      policy so memory grows with touched pages and can support 1M context without reallocating or recompiling.
      - [x] Add a runtime-state paged KV layout ABI and persist it through vNext manifests. Completed on 2026-07-04:
            `RuntimeStateBinding` can now carry `PagedKVCache` layout metadata with page size, logical capacity,
            resident page count, plane offsets, token/page strides, and page-table/active-length state names. Dynamic
            GGUF decode planning publishes this layout on KV cache states while the current dense backing tensor remains
            the executable fallback until paged lowering and kernels land.
            Updated on 2026-07-05: dynamic decode KV runtime state now uses the true paged backing shape
            `[2, residentPages, pageSize, kvHeads, headDim]`; dense function inputs/outputs remain a compatibility
            binding until decode lowering consumes page-table state directly.
            Runtime schedule validation now rejects paged KV states whose backing tensor shape or token/page byte
            strides do not match the layout metadata.
            It also requires the referenced page-table/page-descriptor/active-length auxiliary states to be present
            with the expected Int64 shapes before a schedule can be built.
      - [x] Publish page-table, page-descriptor, and active-length runtime states for dynamic decode. Completed on
            2026-07-04: paged KV layout metadata now names all three auxiliary states; the LLaMA dynamic decode planner
            exposes them as explicit `RuntimeStateBinding`s and vNext manifests round-trip the descriptor-state field.
            These states are not yet consumed by the CPU lowering.
      - [x] Attach runtime-state requirements to stateful artifact entries. Completed on 2026-07-04:
            vNext manifest construction now fills empty artifact-entry `requiredStateBindings` from the schedule's
            runtime states, so paged decode artifacts declare their KV/page-table/page-descriptor/active-length/position
            dependencies unless a caller intentionally provides an explicit list.
      - [x] Define host-side paged KV table initialization semantics. Completed on 2026-07-04:
            runtime helpers now create empty page tables with invalid `-1` logical mappings, fixed descriptor columns
            `[logical_page, first_token, token_count, flags]`, active-length metadata, and checked prefix-to-resident-page
            mapping. This gives paged kernels a stable state format before lowering switches away from dense fallback
            tensors.
      - [x] Replace the dense fallback backing tensor in the paged-reference dynamic decode signature with paged KV,
            page-table, page-descriptor, and active-length inputs. Completed on 2026-07-05: resident KV backing capacity
            is now independently configurable, so paged-reference AOT artifact input shape no longer scales directly
            with logical max context length.
- [x] P1: Add long-context attention execution plans.
      CPU is acceptable for reference validation, but production requires CUDA/Vulkan-oriented kernels for paged
      attention, RoPE/YaRN position handling, mask construction without materializing full `[T,T]` masks, and streaming
      logits-only decode.
      - [x] Publish structured LLaMA attention execution plans from artifact planning. Completed on 2026-07-04:
            plans now distinguish the implemented CPU active-prefix path from planned CPU reference, CUDA-native, and
            Vulkan-native paged-attention paths, record page size/max context, avoid-full-mask expectations, streaming
            decode support, and required paged KV runtime states.
            Updated on 2026-07-05: `cpu-paged-reference` is now reported as `implemented-reference` because
            `GroupedPagedAttentionNode` provides CPU reference semantics; CUDA/Vulkan paged kernels remain planned.
      - [x] Implement the CPU paged-attention reference kernel against the page-table/page-descriptor state contract.
            Completed on 2026-07-05: `GroupedPagedAttentionNode` now has graph/schema/validation/pass/vNext support and
            a CPU interpreter reference path over explicit KV state, page table, page descriptors, and active length.
            `Interpreter.GroupedPagedAttentionMatchesDenseActivePrefix` checks parity against dense grouped
            active-prefix attention.
      - [x] Lower dynamic decode attention to the paged reference kernel without materializing dense full-capacity KV
            tensors.
            - [x] Add a logits-only paged-reference decode lowering entry. Completed on 2026-07-05:
                  `LowerLLaMACausalLMDecodePagedReference` exposes per-layer paged KV state, page table, page descriptor,
                  and active length inputs, then routes attention through `GroupedPagedAttentionNode` instead of
                  `GroupedActivePrefixAttentionNode`.
            - [x] Expose paged-reference decode as an explicit runtime-schedule option. Completed on 2026-07-05:
                  `BuildLLaMADecodeRuntimeSchedule(... usePagedReferenceDecode=true)` binds paged KV/page-table/
                  page-descriptor/active-length states as function inputs and function outputs, while public compiled
                  artifacts still expose only logits through runtime-schedule output projection.
            - [x] Add a CLI/smoke entry for the paged-reference schedule. Completed on 2026-07-05:
                  `litenn_gguf_convert --paged-reference-decode` and
                  `example/gguf/qwen_smoke.py --paged-reference-decode` can now run the stateful decode loop after
                  initializing page tables and page descriptors from compiled input specs; `--compile-only` remains
                  available for cache warming.
            - [x] Add CPU AOT lowering for the paged reference attention node. Completed on 2026-07-05:
                  `GroupedPagedAttentionNode` lowers to `litenn_cpu_grouped_paged_attention_f32` with explicit
                  KV/page-table/page-descriptor/active-length memrefs and helper profiling.
            - [x] P2: Replace the remaining external/prefill-side KV update contract with dynamic paged KV writeback so
                  decode graphs can append current K/V directly into page-table/page-descriptor state. Completed on
                  2026-07-05: `PagedKVAppendNode` now has graph/schema/validation/pass/vNext support, CPU interpreter
                  semantics, CPU AOT MLIR reference lowering, GGUF paged-reference decode lowering, and runtime-schedule
                  state-output aliases for KV state, page table, page descriptor, active length, and position.
                  Production in-place/evicting paged kernels remain tracked under backend performance work.
      - [x] P2: Decide CUDA/Vulkan paged-attention ownership after the CPU reference path is numerically stable.
            Completed on 2026-07-05: the CPU reference path, schedule contract, and runtime-state ABI are stable enough
            to unblock long-context validation. Native CUDA/Vulkan paged kernels are not required for the P2 CPU
            correctness/observability gate and are tracked as backend performance work outside this P2 closeout.
- [x] P1: Replace per-head active-prefix attention helpers with grouped attention execution:
      the current CPU helper is called once per attention head and recomputes score dot products across max,
      denominator, and aggregation passes. Add grouped KV-head or FlashAttention-style online-softmax helpers before
      treating 2K/32K/128K/1M context latency as representative.
      - [x] Add active-prefix attention helper benchmark rows for Qwen-shaped rank-3 KV caches. A short 2026-07-03 run
            measured one KV-head helper call at about `0.022 ms` for 128 active rows, `0.470 ms` for 2048 rows, and
            `2.53 ms` for 8192 rows, which makes grouped KV-head/online-softmax work measurable before adding a new
            kernel.
      - [x] Cache per-row attention scores inside the CPU active-prefix helper so max, denominator, and value
            aggregation no longer recompute the query-key dot product. Validation on 2026-07-03 passed the CPU AOT
            decode parity tests; helper timing improved to about `0.006 ms` for 128 rows, `0.099 ms` for 2048 rows, and
            `0.651 ms` for 8192 rows.
      - [x] Add a grouped active-prefix attention CPU sidecar ABI and benchmark rows that compare Qwen-shaped GQA
            grouped execution against repeated per-query-head rank-3 helper calls. A short 2026-07-03 run validated
            `max_abs_delta=0`; 128 active rows stayed roughly neutral (`0.397 ms` grouped vs `0.410 ms` repeated),
            while 2048 active rows improved from about `9.30 ms` repeated to `8.31 ms` grouped.
      - [x] Route GGUF capacity decode graphs through `GroupedActivePrefixAttentionNode` and lower that node to the
            grouped CPU AOT sidecar. The builder still applies RoPE per query/KV head before grouping, and the grouped
            helper remains a conservative CPU sidecar rather than a final KV-head-tiled attention kernel.
- [x] P2: Add context-extension validation gates.
      Golden evidence must cover prompt lengths beyond tiny smoke sizes, runtime position reuse, EOS behavior,
      tokenizer/chat-template parity, and at least one long-context RoPE/YaRN profile before reporting production
      readiness. Completed on 2026-07-05: `ValidateLLaMAContextRequest` rejects requests beyond model context, rejects
      context extension without RoPE scaling metadata, accepts implemented linear RoPE extension within the scaled limit,
      and blocks YaRN/LongRoPE long-context execution until those scaling formulas have golden-gated runtime support.
- [x] P2: Add benchmark/profile rows for 2K, 32K, 128K, and 1M context targets.
      The table should separate first-run compile/cache population, cache-hit load, prompt replay, steady-state
      generated-token latency, peak memory, artifact bytes, and fallback count.
      - [x] Add a repeatable matrix harness. Completed on 2026-07-05: `benchmark/gguf_context_matrix.py` drives
            `example/gguf/qwen_smoke.py` across `2k,32k,128k,1m` context targets, supports dry-run command inspection,
            paged-reference flags, cache controls, and writes JSON/Markdown rows with build/run/token metrics when a
            target completes.
      - [x] Add profile bundle rows to the matrix. Completed on 2026-07-05:
            `benchmark/gguf_context_matrix.py --profile-bundles` now imports each completed target's
            `qwen_smoke_report.json` through `profile_bundle.py`, then records `profileBundle`, `profileSummary`, and
            `profileTrace` paths in the JSON/Markdown table so the long-context matrix can feed
            `gguf_decode_compare.py --litenn-profile-summary` directly.
      - [x] Add explicit CPU AOT helper and native-node profiling scopes and GGUF decode diagnostics output for helper symbol,
            shape/format/thread detail, call count, total time, and average time per decode step. This covers
            sidecar/helper attribution for quantized projections, get-rows, RoPE, KV scatter, and active-prefix
            attention. Opt-in native plan markers now add stable subgraph/node/schema ids and inclusive/helper/self time
            for generated non-helper code.

### Long-Term Deferred Queue

These items are intentionally not active near-term checklist work. They need real models, external golden fixtures,
or backend architecture decisions before implementation would be meaningful.

- Deferred: exact `RWKV_WKV6`, `RWKV_WKV7`, `GATED_LINEAR_ATTN`, and `GATED_DELTA_NET` mappings, including state ABI,
  weight layout, CUDA/MLIR lowering, and golden-output validation.
- Deferred: native CUDA/Vulkan paged-attention kernels. The CPU reference path and paged runtime-state ABI are complete;
  backend-native kernels remain performance work that should be driven by measured long-context rows.
- Deferred: direct GGUF/source-package tensor-offset borrowing for compiled separated weights. Cache hits already mmap the
  model-level shared compiled weight store; borrowing source tensor offsets directly needs separated-artifact metadata
  with stable source offsets/checksums and a compiler contract proving the compiled weight layout is source-compatible.
- Deferred: full ggml training/backward operator family beyond `CROSS_ENTROPY_LOSS(_BACK)`, because generic
  `*_BACK` coverage should be driven by concrete fine-tuning workloads and the corresponding LiteNN autograd support.
- Deferred: warp-tiled/shared-memory CUDA quantized projection kernels and fused LLM kernels
  (RMSNorm+Linear, RoPE+Q/K layout, attention softmax/value aggregation, quantized Linear epilogues). G16 closes native
  correctness and production gates; these remain benchmark-driven performance work.
- Deferred: CUDA GGUF decode-loop production runner work. This includes a user-facing CUDA backend selector for the
  decode-loop CLI, CUDA cache-hit loading for real Qwen stateful artifacts, Qwen-shaped CUDA quantized projection
  benchmark rows, and a full native CUDA decode comparison table. Do not start this until the CPU-vs-llama.cpp control
  harness keeps the CPU parity work evidence-driven.
- Deferred: repository-owned real Qwen CUDA-native/bridge decode benchmark rows. The comparison tooling accepts these
  rows, but recording model-specific results requires an external model/golden run and should not be replaced by
  synthetic data.
- Deferred: `MAP_CUSTOM*` / `CUSTOM` host callback support. Portable `.ltnn` artifacts should reject arbitrary callback
  pointers until a safe plugin/callback ABI exists.
- Deferred: production CPU GEMM backend or MLIR/LLVM-native intra-op parallel lowering. The current guarded helper path
  is complete enough for profiling, but a production backend should be designed as a separate performance project.
- Deferred: Vulkan production work that is not selected into the current G15.5 implementation slice, especially broad
  real-device/mobile coverage matrices, large fused-kernel families, and profile tables populated from multiple Vulkan
  devices.
- Deferred: broad non-Qwen external llama.cpp parity fixtures for additional real LLaMA-family models, especially CUDA
  artifact parity and multi-token prefill/decode validation against external logits. The Qwen2.5 path is now tracked as
  the active G16 production LLM target.
- Deferred: full compiled AOT training steps with named `forward` / `loss` / `backward` / `optimizer_step` artifact
  entries, mutable parameter/state rebinding, and saved-activation/tape ABI. G14 closes the compatibility-breaking Trainer
  API split; the production compiled train-step implementation remains the G13 AOT-training project.

### Non-Blocking Improvement Queue

These improvements do not require a compatibility break and should not block vNext once the public ABI direction is chosen.

- Improve production CPU GEMM and convolution kernels, or integrate a backend library, without changing public graph/model
  APIs.
- [x] Split CPU GGML runtime sidecars and architecture-specific microkernels out of the monolithic
  `CompiledModule.cpp`. Completed on 2026-08-02 for the Q4_K/Q6_K v4 AVX2/F16C x8/x16 family: the internal POD layout
  ABI and kernels now build in `Runtime/CPUGGMLV4Microkernels`, all 42 targeted quantized/decode tests pass, and a
  kernel-only rebuild compiles that object plus affected links in `12.996 s` without rebuilding
  `CompiledModule.cpp.obj`, down from `169.5 s` (`-92.3%`). The internal header is excluded from installation.
- Expand CUDA native lowering coverage for reductions, normalization, convolutions, attention, and fused training kernels.
- Add richer benchmark rows for compile time, train-step latency, workspace pressure, and numerical drift.
- Replace environment-variable notes in older performance documents with CLI/config examples where the core library already
  owns explicit option objects.
- Add CI matrix coverage for minimal runtime, importer-enabled, compiler-enabled, CUDA-enabled, and tools/examples-enabled
  build profiles.
- Generate public operator/backend coverage documentation directly from `OpSchemaRegistry`.
- Add more external golden fixtures for PyTorch, llama.cpp, GGUF, and SDXL parity.

## Hidden Requirements

- Low precision support is not only an enum addition. Tensor allocation, CPU conversion, serialization, graph validation, compiler type lowering, compiled artifact metadata, tests, and debugging output all need one source of dtype truth.
- FP8 and int quantization need explicit storage semantics. Some paths should treat them as scalar element dtypes, while GGUF quantized weights are usually block formats that need separate quantized tensor metadata.
- GGUF conversion implies model format stability, tensor-name mapping, tokenizer/config import, graph construction helpers for transformer blocks, and enough compiler/runtime ops for LLM inference.
- CUDA support needs capability detection and fallback rules. FP16/BF16/FP8 kernels depend on device architecture, CUDA version, and cuBLAS/cuBLASLt availability.
- AOT support must preserve dtype metadata in rodata/instruction-loaded modules so static/shared library embedding can validate buffers before execution.
- GGUF conversion is also an operator-coverage project. llama.cpp can express more graph ops than LiteNN currently owns, so the converter must either lower them to existing primitives, add LiteNN ops, or reject models with actionable diagnostics.
- llama.cpp compatibility is a semantic compatibility target, not only an operator-count target. Shape layout, axis order, RoPE variants, cache mutation, tokenizer/config metadata, and golden logits must be validated together.
- Heterogeneous execution is not only "choose CUDA for some ops"; it requires explicit placement, transfer, synchronization,
  profiling, and artifact capability metadata so hidden data movement does not make performance or correctness opaque.
- E-graph optimization needs a cost model and numerical-safety policy. Algebraic equivalence is not automatically acceptable
  for floating-point training/inference unless reproducibility and tolerance rules are explicit.
- Separating rodata implies a binding ABI: instructions, metadata, constants, external weights, checksums, alignment, and
  rebinding compatibility must be validated independently.
- LoRA support is both a graph rewrite and a packaging problem: adapters may be merged, unmerged, quantized, stacked, or
  loaded separately from base weights.
- Mobile support needs a smallest-viable build profile. MLIR/CUDA/object-loading assumptions must be optional, and memory,
  binary size, filesystem, dynamic loading, and thread behavior need separate validation.
- Torch/safetensors support is not just a tensor reader. It requires dtype/layout/name mapping, graph-source selection,
  diagnostics, golden PyTorch parity fixtures, and eventually integration with separated rodata and LoRA.
- AOT training is not a direct `Interpreter::RunForward` replacement. Backward execution, saved activations, loss
  gradients, mutable parameters, and optimizer state must be represented in the compiled ABI before Trainer can safely
  use it as the production execution path.

## Date Notes

### 2026-08-04

- Added controlled llama.cpp compiler/OpenMP/runtime sweeps and actual-frequency sampling. GNU/OpenMP, GNU/no-OpenMP,
  and Clang/no-OpenMP measured `5.06`, `5.88`, and `6.03 t/s`; OpenMP was the dominant controlled difference.
- Superseded the earlier local-parity conclusion. The strongest three-pair control places LiteNN `5.49%` behind the
  Clang/no-OpenMP reference, while the unexplained distance from that reference to `6.85 t/s` is reduced to `13.60%`.
- Promoted a matched Clang/no-OpenMP stage profile to the CPU P0 evidence gate. Existing dispatch and FFN-Down ideas
  remain candidates until that profile identifies the largest cross-runtime deficit.
- Added the paired alternating LiteNN/llama.cpp actual-decode control with text/no-fallback/variance gates, power-policy
  sampling, binary identities, and path-redacted evidence.
- Recorded two accepted three-pair Qwen 14B CPU batches. LiteNN's combined `5.585 t/s` median is locally at parity with
  and slightly above llama.cpp's `5.470 t/s`; the unresolved external `6.85 t/s` configuration remains the next CPU P0
  evidence target instead of speculative microkernel changes.
- Added sustained decode/context tiers and effective-cycle/cache/memory counters as follow-up evidence requirements.

### 2026-06-20

- LLaMA lowering can now preserve quantized GGML projection storage in the executable graph and emits explicit
  dequantize/transpose semantics instead of expanding archive weights during model construction.
- Added executable packed Int4 Linear coverage and structural Q8_0 LLaMA lowering coverage; native GGML block execution
  remains the next G16.4 backend milestone.
- Added optional GGUF Linear bias import and 1D bias broadcasting so Qwen2 Q/K/V projection semantics are no longer
  silently dropped by the generic LLaMA builder.
- Extracted GGML block decoding into a dedicated adapter and added Q4_K direct CPU MatMul parity coverage using ggml's
  native row quantizer and vector-dot traits.
- Added `QuantizedMatMulNode` plus explicit Interpreter kernel injection and validated a saved/loaded Q4_K Linear graph
  against the direct ggml-backed primitive.

### 2026-06-21

- Added MLIR Builder-generated CPU AOT block decoding for output-major GGML Q4_K, Q5_K, Q6_K, and Q8_0 quantized MatMul.
  Generated objects consume the original UInt8 payload directly, including package-loaded external Q4_K weights, and do
  not materialize a complete Float32 weight tensor or depend on ggml runtime symbols.
- Updated GGUF compatibility planning to report Q4_K/Q5_K/Q6_K/Q8_0 as `cpu-native-quantized`; dequantized-memory budgets
  now reject only formats that actually require a Float32 fallback, and Qwen diagnostics no longer claim that preserved
  block weights are materialized during lowering.
- Added the first CUDA native GGML block projection slice: Q8_0 `QuantizedMatMulNode` lowers through MLIR/NVPTX,
  consumes UInt8 block storage from variable/external constant buffers, and has CUDA runtime parity coverage.
- Extended the CUDA native GGML block projection slice to Q4_K with MLIR/NVPTX codegen, artifact feature reporting, and
  CUDA runtime parity coverage.
- Closed the initial CUDA GGML block projection code-size risk: Q8_0/Q4_K/Q5_K/Q6_K now emit the outer block reduction as
  MLIR `scf.for` before NVVM lowering instead of fully unrolling every GGML block at code-generation time.
- Closed G16 production-scope gating: Qwen RoPE/golden validation and CUDA decode benchmark rows are now evidence-gated
  rather than represented by synthetic in-repo claims, while post-G16 fused CUDA kernels are tracked as performance work.

### 2026-06-02

- Continued G14.11 vNext cleanup: public `DumpMLIR` now consumes `ExecutablePlan`, the guard test covers that entry
  point, public `GraphToMLIR` translation now consumes `ExecutablePlan`, and `Trainer` debug execution now routes
  forward/backward through `TrainStepPlan::module.plan`.
- Remaining G14.11 work is narrowed to the larger architecture tails: moving `Interpreter` / direct MLIR graph wrappers
  into migration/debug-only surfaces, replacing raw `NodeVariant` ModelIO serialization, making the internal compiler
  lowering bridge fully plan-native, moving raw graph layer helpers behind `ModelBuilder`, and wiring real CPU/CUDA
  compiled train-step runners.

### 2026-05-22

- Added G7-G12 planning checklists for heterogeneous execution, e-graph optimization, separated rodata/weight
  binding, LoRA/adapters, mobile support/testing, and Torch/safetensors import.
- Closed the active roadmap checklist by separating completed current-scope work from the long-term deferred queue.
- Added ggml-style `CROSS_ENTROPY_LOSS` / `CROSS_ENTROPY_LOSS_BACK` support as `CrossEntropyLossNode` and
  `CrossEntropyLossBackwardNode`, including CPU reference execution, interpreter integration, validation, dump,
  ModelIO v19 serialization, const-folding, pass clone/dependency plumbing, AutogradPass logits-gradient generation,
  Layer helpers, explicit MLIR stubs, and `LossNodeTest` coverage.
- Prioritized the remaining `GraphToMLIR` stubs under G5.6 and completed the first inference/AOT tranche:
  `PermuteNode` and `BroadcastToNode` now lower directly to `linalg.generic`, while `SoftmaxNode` expands
  to existing reduce/elementwise primitives for the normal lowering pipeline; added a CPU artifact smoke that
  compares `BroadcastTo -> Permute -> Softmax` compiled output with the interpreter.
- Added G13 for AOT training execution, covering the explicit train-step ABI, compiled backward/loss/optimizer support,
  Trainer execution policies, and parity/benchmark requirements.
- Completed G9 separated rodata packaging: added separated artifact APIs, metadata/constant/weight/instruction regions,
  region compatibility checks, object-per-region carrier output, GGUF separated compile commands, and CPU/CUDA coverage.
- Marked exact RWKV6/7, GLA/GatedDeltaNet signatures, full generic `*_BACK` coverage, host callback ops, production
  CPU GEMM/MLIR intra-op lowering, and broad external llama.cpp/CUDA parity fixtures as long-term deferred work.

### 2026-05-20

- Confirmed current completed work and updated this roadmap's checkboxes for G3/G4/G5.5/G6.
- Marked `example/carrier` static/shared-library style rodata/instruction loading as complete under G3.
- Added `litenn_gguf_convert --compile-cpu/--compile-cuda` carrier-object export for converted `.ltnn` graphs and marked G3 converted-model AOT artifact emission complete.
- Added compiled signature quantization metadata preservation and marked G3 low-precision/quantized signature metadata complete.
- Marked CPU artifact explicit-buffer/cache validation complete under G3 while keeping CUDA LLaMA artifact parity open.
- Marked the existing benchmark/profile documentation complete under G4, including python311 `bench.py` notes and horizontal CPU/AOT/CUDA/PyTorch/ggml comparison output.
- Added `PyTorchGoldenTest` as a small PyTorch golden fixture for interpreter and CPU AOT output parity.
- Split partially completed G5.5 items into completed substrate work (`PAD`/`CUMSUM`, `SSM_SCAN`, RWKV-style recurrence, convolution/pooling/upscale Node substrate) and then completed the small compatibility-layer tails; the remaining model-specific tails are WKV6/7/GLA/GatedDeltaNet mappings.
- Added G6 for performance/profile tracking and marked the CPU AOT instruction-level analysis, old fastpath retirement, guarded CPU intra-op path, T1/T16 benchmarks, CUDA Graph profile notes, and raw result persistence complete.
- Completed first-class CUDA launch breakdown output in `litenn_profile` and validated it against CUDA native PTX profile cases.
- Completed the G5.5 compatibility-layer tail for `REPEAT`, `WIN_PART`, `WIN_UNPART`, `GET_REL_POS`,
  `ADD_REL_POS`, and `SSM_CONV` using existing Reshape/Broadcast/Pad/Permute/Gather/Add/Conv2D substrate,
  with focused `LayerTest` coverage.

### 2026-05-18

- Completed the first G5.2/G5.3 forward-path Node tranche: `ScanNode`, `SSMScanNode`, `RWKVWKVNode`,
  `SoftmaxNode`, `NormalizationNode`, and `BatchMatMulNode`.
- Added CPU reference execution, graph validation, debug dump, ModelIO v13 serialization, ConstFold support,
  and pass clone/dependency plumbing for the new Nodes.
- Rewired `Layer::AddCumsum`, `Layer::AddSoftmax`, `LayerNorm`, `RMSNorm`, and `GroupNorm` to use the new
  dedicated Nodes; CUDA/MLIR lowering and Autograd differentiation remain explicit TODO stubs.
- Added `ScanHotPathNodeTest` coverage for interpreter numerics, recurrence reference kernels, constant folding,
  serialization roundtrip, and dump node-kind visibility.
- Fixed the blocked `GGUFLLaMACausalLM` CPU AOT tests by expanding `NormalizationNode` LayerNorm/RMSNorm into
  existing LiteNN dialect ops in GraphToMLIR; GroupNorm remains an explicit MLIR stub.
- Completed G5.4 with `Im2ColNode`, `Conv2DNode`, `ConvTranspose2DNode`, `Pool2DNode`, and `UpsampleNode`:
  added CPU reference execution, Layer helpers, validation, dump, ModelIO v16 serialization, ConstFold support,
  pass clone/dependency plumbing, explicit Autograd/MLIR stubs, non-CPU interpreter host fallback, and expanded
  `ConvolutionPoolingNodeTest` coverage for numerics, constant folding, serialization roundtrip, and dump visibility.
- Marked G5.5 convolution/pooling/upscale P2 coverage complete for the Node substrate; remaining 1D/3D convenience
  Layers and native CUDA/MLIR lowerings are tracked as performance/backend work rather than add-a-node checklist gaps.

### 2026-05-17

- Added the low precision, quantization, and GGUF import roadmap.
- Completed scalar dtype storage/reference paths.
- Completed graph/runtime/storage metadata paths for affine and block quantization.
- Completed CUDA low-precision capability reporting, CPU-bridge conversion coverage, fp16/bf16 GEMM attempts, and dtype benchmark registration.
- Completed standalone GGUF-to-LiteNN archive conversion.
- Completed the first static CPU-runnable LLaMA-family forward graph path.
- Reviewed the llama.cpp operator additions and recorded hardening work for real GGUF layout, decode/KV-cache, RoPE metadata, axis semantics, and CLI stage separation.
- Added the self-contained `example/gguf` conversion example, optional LLaMA lowering `positionOffset`, and static-shape decode graph lowering with explicit KV-cache inputs/outputs.
