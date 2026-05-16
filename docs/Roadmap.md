# LiteNN Roadmap

This document is the append-only planning entry for LiteNN. New requirements should be added as dated sections, keeping older plans intact unless a later section explicitly supersedes them.

## 2026-05-17: Low Precision, Quantization, and GGUF Import

### Hidden Requirements

- Low precision support is not only an enum addition. Tensor allocation, CPU conversion, serialization, graph validation, compiler type lowering, compiled artifact metadata, tests, and debugging output all need one source of dtype truth.
- FP8 and int quantization need explicit storage semantics. Some paths should treat them as scalar element dtypes, while GGUF quantized weights are usually block formats that need separate quantized tensor metadata.
- GGUF conversion implies model format stability, tensor-name mapping, tokenizer/config import, graph construction helpers for transformer blocks, and enough compiler/runtime ops for LLM inference.
- CUDA support needs capability detection and fallback rules. FP16/BF16/FP8 kernels depend on device architecture, CUDA version, and cuBLAS/cuBLASLt availability.
- AOT support must preserve dtype metadata in rodata/instruction-loaded modules so static/shared library embedding can validate buffers before execution.

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

- [ ] Add CUDA capability detection for fp16, bf16, and fp8.
- [ ] Use cuBLAS/cuBLASLt for supported GEMM cases and explicit fallback paths for unsupported devices.
- [ ] Add conversion kernels for f32 <-> fp16/bf16/fp8/int8.
- [ ] Add benchmark coverage per dtype and backend.

### P3: GGUF Reader and Converter

- [ ] Read GGUF metadata, tensor directory, tensor payloads, and ggml quantized block formats from `third_party/llama.cpp`.
- [ ] Map GGUF tensors to LiteNN variables with stable names and shape validation.
- [ ] Import tokenizer/config metadata needed by LLaMA-like models.
- [ ] Emit LiteNN model files that can be loaded without linking llama.cpp at runtime.

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
