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

Status: in progress. On 2026-05-17, LiteNN's CPU AOT path now covers tiny token-id LLaMA-family artifacts end-to-end for static decode graphs and for a minimal single-token full-graph prefill run; two-token full-graph prefill is additionally covered through artifact compile/load smoke. CUDA artifact parity and broader multi-token prefill runtime coverage remain tracked here.

- [ ] Compile converted models to CPU/CUDA AOT artifacts with rodata/instruction separation.
- [ ] Preserve quantized and low-precision metadata in compiled signatures.
- [x] Add runtime loader examples for static/shared library embedding.
- [ ] Support CUDA backend selection for lowered LLaMA graphs once G2 decode semantics are stable.
- [x] Validate that AOT artifacts can consume externally provided weights/cache buffers without interpreter-only assumptions.

Known progress from review:

- Completed on 2026-05-17: `GraphToMLIR` now lowers `GetRowsNode`, unblocking token-id embedding lookup for CPU AOT compilation of lowered LLaMA graphs.
- Completed on 2026-05-17: tiny LLaMA full-graph CPU artifacts now have compile/load smoke coverage for 2-token prefill, and 1-token prefill is validated end-to-end against the CPU interpreter after `CompileArtifact().Load()`.
- Completed on 2026-05-17: tiny static decode graphs now execute through CPU `CompileArtifact().Load()` with explicit `past_key_N`/`past_value_N` inputs and updated-cache outputs, matching the interpreter without hidden interpreter-only cache state.
- Completed on 2026-05-20: `example/carrier` demonstrates object carrier generation plus static/shared-library style loading through rodata/instruction addresses.
- Completed on 2026-05-20: CPU artifact tests cover external input/output buffers and explicit KV-cache inputs/outputs; CUDA artifact parity for lowered LLaMA graphs remains open.

### G4: Validation and Benchmarks

Purpose: make correctness and performance claims traceable across CPU
single-thread, CPU multithread, CUDA, AOT, PyTorch, and llama.cpp baselines.

- [ ] Add golden tests against llama.cpp or PyTorch for small fixtures.
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
- Remaining open item: direct external llama.cpp/PyTorch golden-output comparison for small fixtures, beyond deterministic in-tree fixtures.

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
      MLIR lowering 暂为 stub，复用解释器路径。）
- [x] `BroadcastToNode`: 显式将某些维度从 1 扩到指定大小（含插入前导单位维），
      不复制数据但暴露明确的 shape 推导；替代 BinaryOp 隐式广播作为前置步骤。
      解锁：`REPEAT`、`UPSCALE-Nearest` 的 Layer 实现。
      （已完成：Graph/Validator/Dump/ModelIO v12/Interpreter CPU reference + non-CPU host fallback
      through interpreter、ConstFold、ForwardOnly/Fusion/Inline/Autograd clone/dependency 接入、
      `Layer/BroadcastTo.h`、`DataMovementNodeTest` 覆盖前导维插入与 singleton dim 扩展；
      MLIR lowering 与 Autograd differentiation 暂为显式 stub。）
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
      CUDA/MLIR fused lowering 与 Autograd differentiation 暂为显式 stub。）
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
- [ ] G2.2 P2 `REPEAT`：由 `BroadcastTo + Reshape` 或专用 Layer 收口。
- [ ] G2.2 P2 `WIN_PART`、`WIN_UNPART`、`GET_REL_POS`、`ADD_REL_POS`：以 ggml 兼容 Layer 落地，
      明确标注为 compatibility-only，不引入新 Node。
- [x] G2.2 P2 `CONV_1D/2D/3D`、`CONV_TRANSPOSE_*`、`IM2COL`、`POOL_*`、`UPSCALE`：由 G5.4 驱动。
      （已完成：G5.4 Node substrate 已闭合，包含 generic `Im2ColNode`、NCHW `Conv2DNode`、
      `ConvTranspose2DNode`、`Pool2DNode` 和 `UpsampleNode` 的 CPU/reference 路径、Layer 包装、
      序列化、常量折叠、pass clone/dependency、dump/validator、解释器 host fallback、MLIR 显式 stub
      与测试覆盖。后续 `CONV_1D/3D` 专用 Layer、CUDA/cuDNN native kernel 与 MLIR lowering
      属于性能/后端扩展，不再是 G5.0 checklist 缺口。）
- [x] G2.2 P2 `SSM_SCAN`：由 G5.2 `SSMScanNode` 驱动。
      （已完成：`SSM_SCAN` 的最小 CPU reference Node，含验证、序列化、解释器、常量折叠与 pass 接入。）
- [ ] G2.2 P2 `SSM_CONV`：等待真实 Mamba 目标模型和参数布局。
- [x] G2.2 P2 RWKV-style recurrence substrate：由 G5.2 `RWKVWKVNode` 驱动。
      （已完成：`RWKVWKVNode` 最小 CPU reference，含验证、序列化、解释器、常量折叠与 pass 接入。）
- [ ] G2.2 P2 `RWKV_WKV6/7`、`GATED_LINEAR_ATTN`、`GATED_DELTA_NET` 真实变体映射。
- [ ] G2.2 P2 训练/反向相关算子：与 G5 各 Node 的 autograd 实现一并推进，仅当真实训练用例出现时启用。

### G6: Performance, Profiling, and Backend Optimization

Purpose: keep performance claims tied to repeatable profile/benchmark evidence across CPU AOT, CUDA native, CUDA Graph, PyTorch, and ggml.

Status: initial CPU/CUDA profile and intra-op tranche completed on 2026-05-20; deeper CPU kernel backend work remains open.

- [x] Profile CPU AOT at instruction level and document whether generated code is scalar or vectorized.
- [x] Remove the misleading old CPU scalar "fast path" benchmark/compiler branch.
- [x] Add CPU AOT intra-op thread-policy benchmarks for default/T1/T16.
- [x] Implement a guarded CPU AOT intra-op path for large static f32 fused Linear/MLP chains.
- [x] Add a persistent CPU worker pool for the current AOT helper path.
- [x] Add CUDA native and CUDA Graph profile/benchmark notes, including comparison with PyTorch CUDA.
- [x] Persist raw CPU/CUDA profile and benchmark outputs under `benchmark/results/`.
- [ ] Move CPU intra-op parallelism into the optimized MLIR/LLVM lowering path or a production CPU GEMM backend.
- [ ] Extend `litenn_profile` with first-class instruction stats and CUDA launch breakdowns instead of relying on manual report synthesis.

Completed notes:

- `docs/PerformanceAnalysis_2026-05-19.md` records CPU instruction-level findings, CPU intra-op results, CUDA native/CUDA Graph profile results, and the old fastpath retirement rationale.
- `docs/PerformanceOptimizationRoadmap.md` tracks the performance-specific P0-P5 checklist and current validation numbers.
- CPU AOT now keeps `LITENN_CPU_AOT_THREADS=1` on the MLIR packed/zmm fallback path, while larger static f32 fused chains can call `litenn_cpu_matmul_bias_relu_parallel_f32`.
- CUDA Graph replay is currently the best CUDA inference path for pointer-stable static-shape runs; local batch-512 MLP512 graph replay reaches the same reported time as PyTorch CUDA in the 2026-05-19 run.

## Hidden Requirements

- Low precision support is not only an enum addition. Tensor allocation, CPU conversion, serialization, graph validation, compiler type lowering, compiled artifact metadata, tests, and debugging output all need one source of dtype truth.
- FP8 and int quantization need explicit storage semantics. Some paths should treat them as scalar element dtypes, while GGUF quantized weights are usually block formats that need separate quantized tensor metadata.
- GGUF conversion implies model format stability, tensor-name mapping, tokenizer/config import, graph construction helpers for transformer blocks, and enough compiler/runtime ops for LLM inference.
- CUDA support needs capability detection and fallback rules. FP16/BF16/FP8 kernels depend on device architecture, CUDA version, and cuBLAS/cuBLASLt availability.
- AOT support must preserve dtype metadata in rodata/instruction-loaded modules so static/shared library embedding can validate buffers before execution.
- GGUF conversion is also an operator-coverage project. llama.cpp can express more graph ops than LiteNN currently owns, so the converter must either lower them to existing primitives, add LiteNN ops, or reject models with actionable diagnostics.
- llama.cpp compatibility is a semantic compatibility target, not only an operator-count target. Shape layout, axis order, RoPE variants, cache mutation, tokenizer/config metadata, and golden logits must be validated together.

## Date Notes

### 2026-05-20

- Confirmed current completed work and updated this roadmap's checkboxes for G3/G4/G5.5/G6.
- Marked `example/carrier` static/shared-library style rodata/instruction loading as complete under G3.
- Marked CPU artifact explicit-buffer/cache validation complete under G3 while keeping CUDA LLaMA artifact parity open.
- Marked the existing benchmark/profile documentation complete under G4, including python311 `bench.py` notes and horizontal CPU/AOT/CUDA/PyTorch/ggml comparison output.
- Split partially completed G5.5 items into completed substrate work (`PAD`/`CUMSUM`, `SSM_SCAN`, RWKV-style recurrence, convolution/pooling/upscale Node substrate) and still-open model-specific tails (`REPEAT`, window/relative-position compatibility layers, `SSM_CONV`, WKV6/7/GLA/GatedDeltaNet mappings).
- Added G6 for performance/profile tracking and marked the CPU AOT instruction-level analysis, old fastpath retirement, guarded CPU intra-op path, T1/T16 benchmarks, CUDA Graph profile notes, and raw result persistence complete.

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
