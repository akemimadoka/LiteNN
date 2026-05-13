# LiteNN CUDA AOT Roadmap

## 目标

CUDA AOT 的最终目标是让 `Compiler<CUDA>` 生成设备特定可执行产物，部署侧可以通过 rodata/instruction 地址或 carrier object 加载，并直接在 CUDA Tensor 上执行。当前已经完成 CUDA Tensor 边界 + CPU AOT bridge、CUDA native image/runtime loader、静态 `f32` elementwise unary/binary + 同 rank broadcast PTX codegen MVP，以及 `f32` MatMul/cuBLAS library-call 路径；下一步继续扩展 reduce、fusion epilogue，以及 MLIR GPU/NVVM 路径。

## 目标分解

### P0：image/backend 基础设施

- [x] 在 rodata ABI 中记录 `CompiledModuleBackend`，区分 CPU native object 与后续 CUDA native binary。
- [x] 暴露 `CompiledModuleArtifact::Backend()`、`CompiledModule<CPU>::Backend()`、`CompiledModule<CUDA>::Backend()`。
- [x] 将 rodata format 提升到 v3，并在 loader 中校验非法 backend metadata。
- [x] 为未来 CUDA native image 增加 launch table schema。
- [x] 为 CUDA native image 增加 feature flags 与 scalar data 区，当前标记静态 shape、单 subgraph、`f32` elementwise Negate/Abs/Sqrt、Add/Subtract/Multiply/Divide、同 rank broadcast 和 MatMul/cuBLAS MVP。

### P1：CUDA native runtime shell

- [x] 设计 CUDA native instruction payload：PTX/cubin/fatbin bytes、kernel symbol table、launch metadata、workspace metadata。
- [x] 新增 CUDA Driver API RAII 封装：module load/unload、function lookup、kernel launch、driver error diagnostics。
- [x] 明确 stream 策略：默认 stream、外部 stream 传入、同步/异步错误边界。
- [x] 保持 carrier object 符号契约不变：仍通过 `<prefix>_rodata{,_size}` 和 `<prefix>_instructions{,_size}` 承载。
- [x] 将 `CompiledModule<CUDA>` native loader 接到 `CUDANativeInstructionPayload` 与 `CUDADriverModule`；`CUDANative` image 会反序列化 payload、加载 Driver module，并按 launch table 传入 CUDA Tensor 指针。

### P2：CUDA codegen MVP

- [x] 从 forward-only、静态 shape、单 subgraph 开始，已支持 `Float32` 单输入单输出 `UnaryOp::Negate/Abs/Sqrt` 和双输入单输出 `BinaryOp::Add/Subtract/Multiply/Divide` native PTX。
- [x] 扩展静态同 rank elementwise broadcast，覆盖 `[N, C] + [1, C]`、`[N, C] - [N, 1]` 等 bias/scale 类形态。
- [ ] 扩展更多 unary/binary op，避免一开始就处理复杂 fusion/control flow。
- [x] 对 `Float32` MatMul 保留 cuBLAS 调用路径：`CUDANative` payload 可表示 library call，运行时直接在 CUDA Tensor 指针上调用 cuBLAS；后续再评估自定义 kernel 或 fusion epilogue。
- [x] 建立最小 CPU AOT bridge / CUDA native 回归：`CompiledModuleCUDATest` 覆盖 unsupported graph fallback、native MatMul/cuBLAS、native Negate/Abs/Sqrt、native Add/Subtract/Multiply/Divide 和同 rank broadcast 执行。
- [ ] 扩展 CPU Interpreter / CPU AOT / CUDA AOT 三方数值回归矩阵。

### P3：调度、内存与融合

- [ ] 规划临时 buffer/workspace，减少 kernel 间分配。
- [ ] 支持 kernel fusion launch table 和 shape-specialized kernels。
- [ ] 引入 stream-aware copy、async execution、event/synchronization policy。
- [ ] 扩展到 fused MatMulBiasAdd/ReLU、reduce、concat/slice。

### P2.5：PTX 模板收敛与 MLIR NVPTX 路线

当前 elementwise/broadcast CUDA native codegen 仍使用手写 PTX 字符串模板。它的价值是验证 `CUDANativeInstructionPayload`、Driver module、launch table 和 CUDA Tensor 指针 ABI；长期目标是将 PTX 来源切到 MLIR GPU/NVVM 或更结构化的 kernel generator。

- [x] **R0：模板隔离**：将手写 PTX 和 kernel symbol 生成从 `CompiledModule.cpp` 拆到独立 `CUDANativeCodegen` 模块，`CompiledModule.cpp` 只保留 graph matcher、payload 组装和 runtime 分派。
- [ ] **R1：行为基线（部分完成）**：保留现有 native unary/binary/broadcast/cuBLAS 回归；已新增 codegen/payload inspection 覆盖 kernel symbol、PTX NUL 结尾、feature flags 与 broadcast PTX 形态，后续继续补 artifact 级 ABI inspection。
- [ ] **R2：MLIR NVVM 试点（部分完成）**：新增可选实验路径，当前用 MLIR LLVM/NVVM dialect 生成最小 `UnaryOp::Negate` `f32` elementwise kernel，完成 MLIR -> LLVM IR -> NVPTX PTX emission，并接入 `Compiler<CUDA>` unary payload；后续补 `gpu.func`/`gpu.module` lowering 版本和更多 unary op。
- [ ] **R3：逐步替换模板**：按 unary -> same-shape binary -> broadcast binary 的顺序替换手写 PTX；每一步保留 fallback 到模板路径直到测试稳定。
- [ ] **R4：生产化清理**：稳定后删除对应手写 PTX 模板，补充 target arch/compute capability 策略、diagnostics、benchmark 和 release checklist。

阶段交付标准：

- R0 完成后，`CompiledModule.cpp` 不再直接拼接 PTX 文本，只负责 graph matcher、feature flags、payload 组装和 artifact 序列化；`CUDANativeCodegen` 负责 kernel symbol、PTX 文本和文本到 payload bytes 的转换。
- R1 完成后，现有 `CompiledModuleCUDATest` 与 `CompiledModuleTest` 可覆盖 native artifact 的加载/运行，新增测试需要直接检查 PTX payload 的 NUL 结尾、kernel symbol、feature flags 和 broadcast shape metadata 不发生 ABI 漂移。
- R2 完成后，MLIR/NVVM 试点必须是可选路径，失败时能回退到模板 codegen；当前第一步只覆盖 `UnaryOp::Negate`，验证 MLIR 生成的 NVPTX PTX 可放入现有 payload 并通过 CUDA Driver 执行，避免同时改动 payload、runtime loader 和 broadcast 索引逻辑。
- R3/R4 完成后，模板路径只作为临时 fallback 或被删除；目标 arch、compute capability、diagnostics 和 benchmark 进入 release checklist。

设计边界：

- MLIR/NVPTX 只替换 kernel codegen 来源，不改变 `Graph -> Compiler -> CompiledModuleArtifact -> CompiledModule<CUDA>` 的 artifact/load/run 分层。
- `CUDANativeInstructionPayload` 继续承载 target、binary bytes/library-call 标记、feature flags、scalar data、kernel launch table 和参数表。
- cuBLAS MatMul 继续走 `LibraryCall` payload；MLIR GPU/NVVM 优先解决自定义 elementwise/reduce/fusion kernel。

### P4：生产化验证

- [ ] CUDA AOT carrier 静态库/动态库加载示例。
- [ ] CUDA unavailable / driver mismatch / arch mismatch 的诊断测试。
- [ ] benchmark 覆盖 CPU AOT、CUDA bridge、CUDA native。
- [ ] 文档化 backend/image ABI 迁移策略，并进入 release checklist。

## 当前选择

当前 `Compiler<CUDA>` 采用“native 优先、bridge fallback”的策略：静态 shape、单 subgraph、`Float32` elementwise Negate/Abs/Sqrt/Add/Subtract/Multiply/Divide、同 rank elementwise broadcast，以及 `Float32` 二维 MatMul 会生成 `CompiledModuleBackend::CUDANative` artifact。`instructions` 中承载 `CUDANativeInstructionPayload`，可表示 PTX/cubin/fatbin driver kernel，也可表示 cuBLAS library call；其他图仍生成 `CompiledModuleBackend::CPUNative` artifact，并由 `CompiledModule<CUDA>` 在 CUDA Tensor 边界做 CPU↔CUDA copy。

P1/P2 当前已经提供：

- `CUDANativeInstructionPayload`：序列化 PTX/cubin/fatbin bytes 或 library-call 标记、target、feature flags、scalar data、workspace、kernel launch table 和参数表，直接作为 CUDA native artifact 的 `instructions` payload。
- `CUDADriverModule`：基于 CUDA Driver API 的 RAII module shell，支持 module load/unload、默认或外部 stream launch、可选同步和错误诊断。当前 `CUDADeviceTest.DriverModuleLaunchesPTXKernel` 用 PTX smoke 覆盖最小 launch 路径。
- `CompiledModule<CUDA>` native loader：根据 rodata backend 在 CPU bridge 与 CUDA native loader 间分派；native loader 复制 image bytes 后可独立于 carrier library 生命周期执行。
- `Compiler<CUDA>` 最小 native codegen：识别 `[input] -> UnaryOp(input)`、`[lhs, rhs] -> BinaryOp(lhs, rhs)` 的 `f32` 静态 elementwise 图，以及二维 `Float32` MatMul。elementwise/broadcast 生成对应 PTX kernel 和 launch table，MatMul 生成 cuBLAS library-call payload。
