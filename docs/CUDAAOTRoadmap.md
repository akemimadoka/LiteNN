# LiteNN CUDA AOT Roadmap

## 目标

CUDA AOT 的最终目标是让 `Compiler<CUDA>` 生成设备特定可执行产物，部署侧可以通过 rodata/instruction 地址或 carrier object 加载，并直接在 CUDA Tensor 上执行。当前已经完成 CUDA Tensor 边界 + CPU AOT bridge，下一步是把 image ABI、runtime loader 和 MLIR GPU codegen 分阶段推进。

## 目标分解

### P0：image/backend 基础设施

- [x] 在 rodata ABI 中记录 `CompiledModuleBackend`，区分 CPU native object 与后续 CUDA native binary。
- [x] 暴露 `CompiledModuleArtifact::Backend()`、`CompiledModule<CPU>::Backend()`、`CompiledModule<CUDA>::Backend()`。
- [x] 将 rodata format 提升到 v3，并在 loader 中校验非法 backend metadata。
- [ ] 为未来 CUDA native image 增加 feature flags / launch table schema。

### P1：CUDA native runtime shell

- [ ] 设计 CUDA native instruction payload：PTX/cubin/fatbin bytes、kernel symbol table、launch metadata、workspace metadata。
- [ ] 新增 CUDA Driver API RAII 封装：module load/unload、function lookup、kernel launch、driver error diagnostics。
- [ ] 明确 stream 策略：默认 stream、外部 stream 传入、同步/异步错误边界。
- [ ] 保持 carrier object 符号契约不变：仍通过 `<prefix>_rodata{,_size}` 和 `<prefix>_instructions{,_size}` 承载。

### P2：CUDA codegen MVP

- [ ] 从 forward-only、静态 shape、单 subgraph 开始。
- [ ] 先支持 elementwise/broadcast 和简单 unary/binary op，避免一开始就处理复杂 fusion/control flow。
- [ ] 对 MatMul 保留 cuBLAS 调用路径，后续再评估自定义 kernel 或 fusion epilogue。
- [ ] 建立 CPU Interpreter / CPU AOT / CUDA AOT 三方数值回归。

### P3：调度、内存与融合

- [ ] 规划临时 buffer/workspace，减少 kernel 间分配。
- [ ] 支持 kernel fusion launch table 和 shape-specialized kernels。
- [ ] 引入 stream-aware copy、async execution、event/synchronization policy。
- [ ] 扩展到 fused MatMulBiasAdd/ReLU、reduce、concat/slice。

### P4：生产化验证

- [ ] CUDA AOT carrier 静态库/动态库加载示例。
- [ ] CUDA unavailable / driver mismatch / arch mismatch 的诊断测试。
- [ ] benchmark 覆盖 CPU AOT、CUDA bridge、CUDA native。
- [ ] 文档化 backend/image ABI 迁移策略，并进入 release checklist。

## 当前选择

当前 `Compiler<CUDA>` 仍生成 CPU native object，并由 `CompiledModule<CUDA>` 在 CUDA Tensor 边界做 CPU↔CUDA copy，因此 `Backend()` 返回 `CompiledModuleBackend::CPUNative`。真正 CUDA native codegen 落地后，artifact 才应写入 `CompiledModuleBackend::CUDANative`，并由 CUDA native loader 处理 `instructions` 中的 GPU binary bundle。
