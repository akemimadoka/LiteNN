# LiteNN Roadmap

This document is the append-only planning entry for LiteNN. New requirements should be added as dated sections, keeping older plans intact unless a later section explicitly supersedes them.

## 2026-05-17: Low Precision, Quantization, and GGUF Import

### Hidden Requirements

- Low precision support is not only an enum addition. Tensor allocation, CPU conversion, serialization, graph validation, compiler type lowering, compiled artifact metadata, tests, and debugging output all need one source of dtype truth.
- FP8 and int quantization need explicit storage semantics. Some paths should treat them as scalar element dtypes, while GGUF quantized weights are usually block formats that need separate quantized tensor metadata.
- GGUF conversion implies model format stability, tensor-name mapping, tokenizer/config import, graph construction helpers for transformer blocks, and enough compiler/runtime ops for LLM inference.
- CUDA support needs capability detection and fallback rules. FP16/BF16/FP8 kernels depend on device architecture, CUDA version, and cuBLAS/cuBLASLt availability.
- AOT support must preserve dtype metadata in rodata/instruction-loaded modules so static/shared library embedding can validate buffers before execution.
- GGUF conversion is also an operator-coverage project. llama.cpp can express more graph ops than LiteNN currently owns, so the converter must either lower them to existing primitives, add LiteNN ops, or reject models with actionable diagnostics.

### P0: Scalar DType Foundation

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

### P1: Quantized Tensor Storage

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

### P2: CUDA Low Precision Kernels

Status: completed for capability reporting, CPU-bridge conversion coverage, fp16/bf16 GEMM attempts, and benchmark registration on 2026-05-17.

- [x] Add CUDA capability detection for fp16, bf16, fp8, and int8 storage / matmul policy.
- [x] Use cuBLAS/cuBLASLt for supported GEMM cases and explicit fallback paths for unsupported devices.
- [x] Add conversion coverage for f32 <-> fp16/bf16/fp8/int8 through the existing synchronous CPU bridge.
- [x] Add benchmark coverage per dtype for CUDA device MatMul, exposing native and fallback behavior.

Completed notes:

- `CUDALowPrecisionCapabilities` reports compute capability, cuBLASLt build support, storage support, and native MatMul support for low-precision dtypes.
- `DeviceTraits<CUDA>::DoBinaryOp(MatMul)` still keeps the existing fp32/fp64 cuBLAS path and now attempts fp16/bf16 `cublasGemmEx` when the build and device report support.
- FP8 and int8 native GEMM are intentionally not marked as implemented yet. They need explicit accumulation/output semantics and cuBLASLt kernel policy before they are safe for production inference.
- CUDA dtype conversion remains synchronous through CPU reference conversion. Dedicated CUDA conversion kernels are a later performance task, not a correctness blocker.

### P3: GGUF Reader and Converter

Status: completed for standalone GGUF-to-LiteNN archive conversion on 2026-05-17.

- [x] Read GGUF metadata, tensor directory, tensor payloads, and ggml quantized block formats from `third_party/llama.cpp`.
- [x] Map GGUF tensors to LiteNN variables with stable names and shape validation.
- [x] Import tokenizer/config metadata needed by LLaMA-like models.
- [x] Emit LiteNN model files that can be loaded without linking llama.cpp at runtime.

Completed notes:

- `tools/gguf/litenn_gguf_convert` now converts GGUF files into LiteNN `.ltnn` archives through a small vendored ggml/gguf support library instead of linking the full llama.cpp runtime.
- GGUF K/V metadata is preserved in graph metadata with scalar/array widening into LiteNN's metadata value model, so tokenizer/config keys survive conversion.
- GGUF tensor names are preserved as graph variable names, and supported ggml block-quantized payloads are archived as LiteNN variables with block quantization metadata.
- Model format version 6 now persists graph variable names plus metadata, which is the storage substrate the GGUF converter relies on.
- The first converter intentionally emits a weight-archive graph with an empty forward subgraph; executable LLaMA-family graph lowering remains a P4 task.

### GGUF/Llama.cpp Operator Gap Checklist

Audit source: `third_party/llama.cpp/ggml/include/ggml.h` `ggml_op`, `ggml_unary_op`, and `ggml_glu_op` enums, plus llama.cpp model graph builders. This list tracks operators that matter beyond the GGUF container format itself.

P0: required for common LLaMA-family decode/inference:

- [ ] Embedding / row lookup: `GET_ROWS`, `GET_ROWS_BACK` lowering or a dedicated embedding node.
- [ ] RMSNorm: `RMS_NORM` plus epsilon metadata; backward can remain deferred for inference-only import.
- [ ] RoPE: `ROPE` with mode, base, scale, and position handling compatible with llama.cpp metadata.
- [ ] Attention mask and softmax: `SOFT_MAX`, `DIAG_MASK_INF`, `TRI`, scale, and causal masking behavior.
- [ ] Quantized weight MatMul: `MUL_MAT` over GGML block formats with dequantization or fused low-precision kernels.
- [ ] KV cache updates/views: `VIEW`, `CPY`, `SET`, `CONT`, `RESHAPE`, `PERMUTE`, `TRANSPOSE`, and slicing semantics, or a higher-level KV cache op.
- [ ] MLP activation path: `SILU`, `GLU` / `SWIGLU`, `MUL`, `ADD`, `SCALE`, and broadcast helpers such as `REPEAT` / `ADD1`.

P1: needed by popular variants, MoE models, or efficient attention:

- [ ] MoE routing: `MUL_MAT_ID`, `ADD_ID`, `TOP_K`, `ARGSORT`, and gather/scatter style row selection.
- [ ] Additional normalizations: `NORM`, `GROUP_NORM`, `L2_NORM`.
- [ ] Fused attention option: `FLASH_ATTN_EXT` as either a fused op or a converter rewrite to LiteNN attention primitives.
- [ ] Activation coverage: `GELU`, `GELU_ERF`, `GELU_QUICK`, `SIGMOID`, `TANH`, `RELU`, `LEAKY_RELU`, `CLAMP`, `HARDSWISH`, `HARDSIGMOID`.
- [ ] Shape/data movement: `CONCAT`, `PAD`, `ROLL`, `ARANGE`, `CUMSUM`, `SUM_ROWS`, and robust broadcast semantics.

P2: architecture-specific model families and multimodal support:

- [ ] SSM/Mamba style ops: `SSM_CONV`, `SSM_SCAN`.
- [ ] RWKV and gated attention: `RWKV_WKV6`, `RWKV_WKV7`, `GATED_LINEAR_ATTN`, `GATED_DELTA_NET`.
- [ ] Vision/multimodal ops: `CONV_1D/2D/3D` equivalents, `CONV_TRANSPOSE_*`, `IM2COL`, `POOL_*`, `UPSCALE`, `WIN_PART`, `WIN_UNPART`, `GET_REL_POS`, `ADD_REL_POS`.
- [ ] Loss/training/backward ops only if converted models need training or fine-tuning: `*_BACK`, `CROSS_ENTROPY_LOSS`, optimizer ops.

P3: unsupported in the first converter unless a real model requires them:

- [ ] Custom callback ops: `MAP_CUSTOM*`, `CUSTOM`.
- [ ] Optimizer-only graph ops: `OPT_STEP_ADAMW`, `OPT_STEP_SGD`.
- [ ] Rare numerical helpers with no first target model dependency: `SOLVE_TRI`, `OUT_PROD`, `TIMESTEP_EMBEDDING`.

### P4: Transformer Graph Coverage

- [ ] Add graph/layer helpers for RMSNorm, RoPE, attention KV cache, SwiGLU/MLP, and causal masking.
- [ ] Lower common LLaMA-family models from GGUF metadata into LiteNN Graph.
- [ ] Keep a CPU reference execution path for correctness before relying on CUDA/AOT.

### P5: AOT LLM Artifacts

- [ ] Compile converted models to CPU/CUDA AOT artifacts with rodata/instruction separation.
- [ ] Preserve quantized and low-precision metadata in compiled signatures.
- [ ] Add runtime loader examples for static/shared library embedding.

### P6: Validation and Benchmarks

- [ ] Add golden tests against llama.cpp or PyTorch for small fixtures.
- [ ] Track CPU single-thread, CPU multithread, CUDA, and AOT performance in one horizontal benchmark table.
- [ ] Add numerical tolerance policy per dtype and quantization format.
