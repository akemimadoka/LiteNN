# LiteNN CUDA AOT Roadmap

## 目标

CUDA AOT 的最终目标是让 `Compiler<CUDA>` 生成设备特定可执行产物，部署侧可以通过 rodata/instruction 地址或 carrier object 加载，并直接在 CUDA Tensor 上执行。当前已经完成 CUDA Tensor 边界 + CPU AOT bridge，下一步是把 image ABI、runtime loader 和 MLIR GPU codegen 分阶段推进。

## 目标分解

### P0：image/backend 基础设施

- [x] 在 rodata ABI 中记录 `CompiledModuleBackend`，区分 CPU native object 与后续 CUDA native binary。
- [x] 暴露 `CompiledModuleArtifact::Backend()`、`CompiledModule<CPU>::Backend()`、`CompiledModule<CUDA>::Backend()`。
- [x] 将 rodata format 提升到 v3，并在 loader 中校验非法 backend metadata。
- [x] 为未来 CUDA native image 增加 launch table schema。
- [x] 为 CUDA native image 增加 feature flags 与 scalar data 区，当前标记静态 shape、单 subgraph、`f32` elementwise Add MVP。

### P1：CUDA native runtime shell

- [x] 设计 CUDA native instruction payload：PTX/cubin/fatbin bytes、kernel symbol table、launch metadata、workspace metadata。
- [x] 新增 CUDA Driver API RAII 封装：module load/unload、function lookup、kernel launch、driver error diagnostics。
- [x] 明确 stream 策略：默认 stream、外部 stream 传入、同步/异步错误边界。
- [x] 保持 carrier object 符号契约不变：仍通过 `<prefix>_rodata{,_size}` 和 `<prefix>_instructions{,_size}` 承载。
- [x] 将 `CompiledModule<CUDA>` native loader 接到 `CUDANativeInstructionPayload` 与 `CUDADriverModule`；`CUDANative` image 会反序列化 payload、加载 Driver module，并按 launch table 传入 CUDA Tensor 指针。

### P2：CUDA codegen MVP

- [x] 从 forward-only、静态 shape、单 subgraph 开始，已支持 `Float32` 双输入单输出 `BinaryOp::Add` native PTX。
- [ ] 扩展 elementwise/broadcast 和简单 unary/binary op，避免一开始就处理复杂 fusion/control flow。
- [ ] 对 MatMul 保留 cuBLAS 调用路径，后续再评估自定义 kernel 或 fusion epilogue。
- [x] 建立最小 CPU AOT bridge / CUDA native 回归：`CompiledModuleCUDATest` 覆盖 unsupported graph fallback 与 native Add 执行。
- [ ] 扩展 CPU Interpreter / CPU AOT / CUDA AOT 三方数值回归矩阵。

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

当前 `Compiler<CUDA>` 采用“native 优先、bridge fallback”的策略：静态 shape、单 subgraph、`Float32` elementwise Add 会生成 `CompiledModuleBackend::CUDANative` artifact，`instructions` 中承载 `CUDANativeInstructionPayload` 和 PTX；其他图仍生成 `CompiledModuleBackend::CPUNative` artifact，并由 `CompiledModule<CUDA>` 在 CUDA Tensor 边界做 CPU↔CUDA copy。

P1/P2 当前已经提供：

- `CUDANativeInstructionPayload`：序列化 PTX/cubin/fatbin bytes、target、feature flags、scalar data、workspace、kernel launch table 和参数表，直接作为 CUDA native artifact 的 `instructions` payload。
- `CUDADriverModule`：基于 CUDA Driver API 的 RAII module shell，支持 module load/unload、默认或外部 stream launch、可选同步和错误诊断。当前 `CUDADeviceTest.DriverModuleLaunchesPTXKernel` 用 PTX smoke 覆盖最小 launch 路径。
- `CompiledModule<CUDA>` native loader：根据 rodata backend 在 CPU bridge 与 CUDA native loader 间分派；native loader 复制 image bytes 后可独立于 carrier library 生命周期执行。
- `Compiler<CUDA>` 最小 native codegen：识别 `[lhs, rhs] -> Add(lhs, rhs)` 的 `f32` 静态图，生成 `litenn_add_f32` PTX kernel 和 launch table。
