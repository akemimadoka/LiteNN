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

Purpose: keep performance claims tied to repeatable profile/benchmark evidence across CPU AOT, CUDA native, CUDA Graph, PyTorch, and ggml.

Status: completed for current profile/benchmark and guarded intra-op tranche; deeper production CPU kernel backend work is deferred to the long-term queue.

- [x] Profile CPU AOT at instruction level and document whether generated code is scalar or vectorized.
- [x] Remove the misleading old CPU scalar "fast path" benchmark/compiler branch.
- [x] Add CPU AOT intra-op thread-policy benchmarks for default/T1/T16.
- [x] Implement a guarded CPU AOT intra-op path for large static f32 fused Linear/MLP chains.
- [x] Add a persistent CPU worker pool for the current AOT helper path.
- [x] Add CUDA native and CUDA Graph profile/benchmark notes, including comparison with PyTorch CUDA.
- [x] Persist raw CPU/CUDA profile and benchmark outputs under `benchmark/results/`.
- [x] Move CPU intra-op parallelism into the optimized MLIR/LLVM lowering path or a production CPU GEMM backend:
      deferred as long-term backend work after the current guarded helper path and profiling tools.
- [x] Extend `litenn_profile` with first-class CPU AOT instruction stats instead of relying on manual objdump report synthesis.
- [x] Extend `litenn_profile` with CUDA launch breakdowns.

Completed notes:

- `docs/PerformanceAnalysis_2026-05-19.md` records CPU instruction-level findings, CPU intra-op results, CUDA native/CUDA Graph profile results, and the old fastpath retirement rationale.
- `docs/PerformanceOptimizationRoadmap.md` tracks the performance-specific P0-P5 checklist and current validation numbers.
- CPU AOT now keeps `LITENN_CPU_AOT_THREADS=1` on the MLIR packed/zmm fallback path, while larger static f32 fused chains can call `litenn_cpu_matmul_bias_relu_parallel_f32`.
- CUDA Graph replay is currently the best CUDA inference path for pointer-stable static-shape runs; local batch-512 MLP512 graph replay reaches the same reported time as PyTorch CUDA in the 2026-05-19 run.
- Completed on 2026-05-20: `litenn_profile` writes raw `.o` and `.s` files, counts packed/scalar FMA, vector loads, broadcasts, gathers/scatters, stack vector ops, and falls back from `subgraph_0` to the first function for fused helper artifacts.
- Completed on 2026-05-20: `litenn_profile` prints CUDA launch breakdowns with backend kind, binary kind, kernel/library/PTX counts, workspace bytes, compile/load time, first native run, steady native run, first CUDA Graph run, and steady CUDA Graph replay time.
- Completed on 2026-05-20: rare numerical helper substrate landed for `OUT_PROD`, `TIMESTEP_EMBEDDING`, and the currently ggml-supported `SOLVE_TRI` variant, with CPU interpreter/reference tests and serialization coverage.
- Completed on 2026-05-20: optimizer-only graph ops landed as `SGDStepNode` and `AdamWStepNode`, keeping optimizer state explicit and serializable.

### G7: Heterogeneous Execution

Purpose: allow one graph to execute across multiple devices/backends while keeping graph semantics deterministic,
buffer ownership explicit, and compiled artifacts loadable without interpreter-only hidden state.

#### G7.1 Device Placement Contract

- [ ] Add graph-level device-placement metadata for params, variables, intermediate values, and results.
- [ ] Define automatic placement defaults: keep current single-device behavior when no placement metadata is present.
- [ ] Add explicit copy/transfer edges or runtime transfer plans instead of hiding cross-device moves inside arbitrary nodes.
- [ ] Validate placement consistency, unsupported device ops, and illegal host/device aliasing with actionable diagnostics.

#### G7.2 Runtime Scheduling

- [ ] Split execution into per-device segments with explicit input/output buffer boundaries.
- [ ] Add a CPU/CUDA mixed-execution smoke test where only a subgraph segment runs on CUDA and the rest remains on CPU.
- [ ] Track synchronization points and stream/event ownership for CUDA segments.
- [ ] Add profiling output that reports per-device time, transfer time, and synchronization overhead.

#### G7.3 AOT and Artifact Support

- [ ] Extend compiled artifact metadata with per-segment backend kind, required device capabilities, and transfer ABI.
- [ ] Support loading heterogeneous artifacts from separated rodata/instruction regions.
- [ ] Reject artifacts when a required backend/device capability is unavailable, with fallback policy documented.

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
- [ ] Add backend-aware cost models for CPU AOT, CUDA native, and interpreter fallback.
      （当前阻塞：需要 backend-calibrated latency/throughput data、layout/transfer/workspace memory model、
      numerical-safety policy, and an extractor that can choose between fused/decomposed/materialized alternatives
      rather than only applying local deterministic simplifications。）
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

- [ ] Add a LoRA metadata model: target module/name, rank, alpha, dropout policy, dtype, and merge mode.
- [ ] Add Layer helpers that apply LoRA as `base(x) + scale * (x @ A @ B)` for linear layers.
- [ ] Support both unmerged runtime adapters and merged-weight export.
- [ ] Define compatibility rules for quantized base weights and low-precision adapter weights.

#### G10.2 Import and Serialization

- [ ] Add LiteNN serialization for adapter tensors and adapter metadata.
- [ ] Add safetensors LoRA import for common naming schemes used by Hugging Face/PEFT-style adapters.
- [ ] Add diagnostics for unmatched target names, shape/rank mismatches, and unsupported adapter variants.
- [ ] Add roundtrip tests for saving/loading base model plus one or more adapters.

#### G10.3 Runtime and AOT

- [ ] Add interpreter execution tests for unmerged LoRA.
- [ ] Add CPU AOT tests for merged LoRA export.
- [ ] Add optional runtime adapter binding for AOT when adapter weights are kept separate from base rodata.
- [ ] Add benchmark coverage for merged versus unmerged adapters.

### G11: Mobile Support and Test Matrix

Purpose: make LiteNN usable on mobile targets with constrained memory, predictable binary size, and repeatable device tests.

#### G11.1 Build and Portability

- [ ] Define supported first targets: Android arm64-v8a and iOS arm64 simulator/device if toolchains are available.
- [ ] Make compiler/MLIR/CUDA features optional so a minimal interpreter/runtime build is possible.
- [ ] Audit C++ standard library, filesystem, reflection, dynamic loading, and thread usage for mobile constraints.
- [ ] Add CMake presets or toolchain documentation for mobile builds.

#### G11.2 Runtime Constraints

- [ ] Add allocator hooks or arena-style allocation for predictable memory usage.
- [ ] Add binary-size and model-size reporting for mobile builds.
- [ ] Add CPU feature detection for ARM NEON and future mobile GPU/NNAPI/CoreML delegation points.
- [ ] Define unsupported features explicitly, such as CUDA-only paths and desktop object loading where unavailable.

#### G11.3 Testing

- [ ] Add host-side cross-compile smoke tests in CI once toolchains are available.
- [ ] Add on-device or emulator smoke tests for tensor ops, model loading, and a small inference graph.
- [ ] Add mobile package examples using separated rodata/weights from G9.
- [ ] Track performance and memory baselines for at least one small MLP/CNN and one tiny transformer block.

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

- [ ] Define an explicit train-step graph ABI: model inputs, targets/loss inputs, parameters, optimizer state, and updated
  parameters/state must be visible as graph inputs/outputs or bindable state.
- [ ] Remove hidden interpreter-only activation/tape dependencies from compiled training by representing saved activations
  as explicit values, explicit workspace buffers, or a documented recomputation strategy.
- [ ] Decide whether the first compiled trainer emits one fused train-step artifact or separate forward/loss/backward/
  optimizer artifacts with a runtime scheduler.
- [ ] Add validation diagnostics that reject AOT training when backward nodes still require interpreter-local state.

#### G13.2 Compiler and Runtime Support

- [ ] Extend compiled module metadata so artifacts can expose multiple named entry points such as `forward`, `loss`,
  `backward`, and `optimizer_step`.
- [ ] Teach the CPU AOT path to compile backward/loss subgraphs with stable tensor specs instead of wrapping only
  `graph.Forward()`.
- [ ] Add a CUDA AOT training path after CPU semantics are stable, including stream/workspace ownership and explicit
  synchronization points.
- [ ] Preserve rodata/instruction separation for training artifacts, including mutable parameter/state binding rules.

#### G13.3 Trainer API

- [ ] Add a trainer execution policy such as `Interpreter`, `AOT`, and `Auto`, with clear fallback/error behavior.
- [ ] Keep `Trainer<Device, Optimizer>` as the high-level API, but route production-capable paths through compiled
  train-step artifacts when available.
- [ ] Keep a reference interpreter trainer for correctness checks, constant evaluation, and unsupported graph debugging.
- [ ] Add examples that train the same small model through interpreter and AOT paths, then compare loss and updated weights.

#### G13.4 Validation and Benchmarking

- [ ] Add golden tests comparing interpreter training and AOT training for Linear, MLP, softmax cross entropy, and AdamW/SGD.
- [ ] Add gradient parity tests that cover saved activations, broadcasting, reductions, and parameter sharing.
- [ ] Add benchmark rows for interpreter trainer, CPU AOT trainer, CUDA AOT trainer, PyTorch, and ggml where applicable.
- [ ] Track compile time, train-step latency, memory/workspace use, and numerical drift separately.

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
- [ ] Add asynchronous execution and synchronization primitives: reusable command buffers, fences/timeline semaphores,
      queue ownership rules, and `RunManyTensorsInto` semantics that do not serialize every dispatch through a full
      wait.
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
      Vulkan-native Linear rows and explicit CPU-bridge MLP rows.
- [ ] Add full comparison tables across CPU AOT, CUDA, Vulkan, ggml, and PyTorch for supported shapes.
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
      Vulkan-resident hidden tensor between them. Automatic graph partitioning and fused multi-layer schedules remain
      tracked separately above.
      Validation on 2026-06-14: a local
      `litenn_bench --benchmark_filter=VulkanNativeManualPipeline.*batch:1 --benchmark_min_time=0.01s` smoke run
      reported roughly 0.283 ms for batch 1 and 0.280 ms for batch 128.

Hidden requirements:

- Vulkan support is not only "LLVM can emit SPIR-V"; shader interface, descriptor set layout, storage-buffer ABI,
  memory model, workgroup dimensions, and device capabilities must all be validated.
- Mobile GPUs vary in fp16/int8/subgroup support and storage-buffer alignment. The backend must report capabilities
  rather than silently selecting kernels that only work on desktop drivers.
- CPU fallback must be explicit for Vulkan, matching the CUDA vNext policy; otherwise benchmark and production behavior
  become opaque.
- Vulkan artifacts should not inherit CPU object-file assumptions. SPIR-V belongs in the instruction region, while
  constants and weights remain separated package regions.

### Long-Term Deferred Queue

These items are intentionally not active near-term checklist work. They need real models, external golden fixtures,
or backend architecture decisions before implementation would be meaningful.

- Deferred: exact `RWKV_WKV6`, `RWKV_WKV7`, `GATED_LINEAR_ATTN`, and `GATED_DELTA_NET` mappings, including state ABI,
  weight layout, CUDA/MLIR lowering, and golden-output validation.
- Deferred: full ggml training/backward operator family beyond `CROSS_ENTROPY_LOSS(_BACK)`, because generic
  `*_BACK` coverage should be driven by concrete fine-tuning workloads and the corresponding LiteNN autograd support.
- Deferred: `MAP_CUSTOM*` / `CUSTOM` host callback support. Portable `.ltnn` artifacts should reject arbitrary callback
  pointers until a safe plugin/callback ABI exists.
- Deferred: production CPU GEMM backend or MLIR/LLVM-native intra-op parallel lowering. The current guarded helper path
  is complete enough for profiling, but a production backend should be designed as a separate performance project.
- Deferred: Vulkan production work that is not selected into the current G15.5 implementation slice, especially broad
  real-device/mobile coverage matrices, large fused-kernel families, and profile tables populated from multiple Vulkan
  devices.
- Deferred: broad external llama.cpp parity fixtures for real LLaMA-family models, especially CUDA artifact parity and
  multi-token prefill/decode validation against external logits.
- Deferred: full compiled AOT training steps with named `forward` / `loss` / `backward` / `optimizer_step` artifact
  entries, mutable parameter/state rebinding, and saved-activation/tape ABI. G14 closes the compatibility-breaking Trainer
  API split; the production compiled train-step implementation remains the G13 AOT-training project.

### Non-Blocking Improvement Queue

These improvements do not require a compatibility break and should not block vNext once the public ABI direction is chosen.

- Improve production CPU GEMM and convolution kernels, or integrate a backend library, without changing public graph/model
  APIs.
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
