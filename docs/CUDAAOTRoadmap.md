# LiteNN CUDA AOT Roadmap

## 目标

CUDA AOT 的最终目标是让 `Compiler<CUDA>` 生成设备特定可执行产物，部署侧可以通过 rodata/instruction 地址或 carrier object 加载，并直接在 CUDA Tensor 上执行。当前已经完成 CUDA Tensor 边界 + CPU AOT bridge、CUDA native image/runtime loader、静态 `f32` elementwise unary/binary + 同 rank broadcast MLIR/NVPTX codegen、`f32` MatMul/cuBLAS library-call 路径、reduce、concat/slice、MatMulBiasAdd/ReLU epilogue 的 CUDA native 路径，以及 stream/sync 运行选项、benchmark 覆盖和 P4 release 验证清单。

## 目标分解

### P0：image/backend 基础设施

- [x] 在 rodata ABI 中记录 `CompiledModuleBackend`，区分 CPU native object 与后续 CUDA native binary。
- [x] 暴露 `CompiledModuleArtifact::Backend()`、`CompiledModule<CPU>::Backend()`、`CompiledModule<CUDA>::Backend()`。
- [x] 将 rodata format 提升到 v3，并在 loader 中校验非法 backend metadata。
- [x] 为未来 CUDA native image 增加 launch table schema。
- [x] 为 CUDA native image 增加 feature flags 与 scalar data 区，当前标记静态 shape、单 subgraph、`f32` elementwise Negate/Abs/Sqrt/Exp/Log/Sin/Cos、Add/Subtract/Multiply/Divide/Max/Min、同 rank broadcast、MatMul/cuBLAS、reduce、concat/slice、MatMulBiasAdd/ReLU epilogue、multi-kernel launch 与 workspace。

### P1：CUDA native runtime shell

- [x] 设计 CUDA native instruction payload：PTX/cubin/fatbin bytes、kernel symbol table、launch metadata、workspace metadata。
- [x] 新增 CUDA Driver API RAII 封装：module load/unload、function lookup、kernel launch、driver error diagnostics。
- [x] 明确 stream 策略：默认 stream、外部 stream 传入、同步/异步错误边界。
- [x] 保持 carrier object 符号契约不变：仍通过 `<prefix>_rodata{,_size}` 和 `<prefix>_instructions{,_size}` 承载。
- [x] 将 `CompiledModule<CUDA>` native loader 接到 `CUDANativeInstructionPayload` 与 `CUDADriverModule`；`CUDANative` image 会反序列化 payload、加载 Driver module，并按 launch table 传入 CUDA Tensor 指针。

### P2：CUDA codegen MVP

- [x] 从 forward-only、静态 shape、单 subgraph 开始，已支持 `Float32` 单输入单输出 `UnaryOp::Negate/Abs/Sqrt/Exp/Log/Sin/Cos` 和双输入单输出 `BinaryOp::Add/Subtract/Multiply/Divide/Max/Min` native PTX。
- [x] 扩展静态同 rank elementwise broadcast，覆盖 `[N, C] + [1, C]`、`[N, C] - [N, 1]` 等 bias/scale 类形态。
- [x] 扩展更多 unary/binary op：已完成 `Exp/Log/Sin/Cos` 与 `Max/Min`；`Pow/Tan/Arc*` 因 libdevice/外部符号与精度策略暂后置。
- [x] 对 `Float32` MatMul 保留 cuBLAS 调用路径：`CUDANative` payload 可表示 library call，运行时直接在 CUDA Tensor 指针上调用 cuBLAS；后续再评估自定义 kernel 或 fusion epilogue。
- [x] 建立最小 CPU AOT bridge / CUDA native 回归：`CompiledModuleCUDATest` 覆盖 unsupported graph fallback、native MatMul/cuBLAS、native Negate/Abs/Sqrt/Exp/Log/Sin/Cos、native Add/Subtract/Multiply/Divide/Max/Min 和同 rank broadcast 执行。
- [x] 扩展 CPU Interpreter / CPU AOT / CUDA AOT 三方数值回归矩阵，覆盖 native unary/binary/broadcast/cuBLAS MatMul 与 CPU AOT bridge fallback。

### P3：调度、内存与融合

- [x] 接通 CUDA native workspace allocation 与 `Workspace` launch argument plumbing，payload/kernel workspace byte range 会统一分配并做边界校验；后续再用实际算子消耗 workspace 以减少 kernel 间分配。
- [x] 支持 multi-kernel launch table 与 shape-specialized kernels：concat 按输入生成多 kernel，MatMulBiasAdd/ReLU 生成 cuBLAS MatMul + PTX epilogue 的混合 launch。
- [x] 引入 stream-aware copy、async execution、event/synchronization policy：`CompiledModuleCUDARunOptions` 可传入外部 CUDA stream 并选择是否同步；native PTX launch、cuBLAS MatMul 和同 dtype host/device copy 走 stream-aware 路径，CPU bridge 明确拒绝异步执行。
- [x] 扩展到 fused MatMulBiasAdd/ReLU、reduce、concat/slice，并新增 CUDA runtime 数值回归。

### P2.5：PTX 模板收敛与 MLIR NVPTX 路线

当前 unary、same-shape binary、same-rank broadcast binary、reduce、concat/slice 与 MatMulBiasAdd/ReLU epilogue CUDA native codegen 均走 MLIR GPU/NVVM 路径，并已从手写 PTX 字符串模板收敛为 `OpBuilder` 结构化生成。旧手写 PTX fallback 已删除；MLIR/NVPTX 生成失败时对应 graph 视为 CUDA native unsupported，并回退 CPU AOT bridge。

- [x] **R0：模板隔离**：将手写 PTX 和 kernel symbol 生成从 `CompiledModule.cpp` 拆到独立 `CUDANativeCodegen` 模块，`CompiledModule.cpp` 只保留 graph matcher、payload 组装和 runtime 分派。
- [x] **R1：行为基线**：保留现有 native unary/binary/broadcast/cuBLAS 回归；已新增 codegen/payload inspection 覆盖 kernel symbol、PTX NUL 结尾、feature flags 与 broadcast PTX 形态，并补充 `Compiler<CUDA>::CompileArtifact` 产物级 ABI inspection，覆盖 rodata backend、payload flags、scalar metadata、kernel launch/argument metadata 与 cuBLAS library-call payload。
- [x] **R2：MLIR GPU/NVVM 试点**：新增可选实验路径，当前通过 `OpBuilder` 生成 `gpu.module`/`gpu.func` 表达 `f32` elementwise unary/binary kernel，经 GPUToNVVM、LLVM IR、NVPTX 后端生成 PTX，并接入 `Compiler<CUDA>` payload；已覆盖 `UnaryOp::Negate/Abs/Sqrt/Exp/Log/Sin/Cos` 与 `BinaryOp::Add/Subtract/Multiply/Divide/Max/Min`，其中 `Abs/Sqrt/Exp/Log/Sin/Cos/Max/Min` 使用 NVVM intrinsic 或等价 LLVM op，避免 standalone PTX 依赖 libdevice 外部符号。
- [x] **R3：逐步替换模板**：unary、same-shape binary、same-rank broadcast binary、reduce、concat/slice 与 MatMulBiasAdd/ReLU epilogue 已接入 MLIR GPU/NVVM；MLIR 失败时返回 unsupported 并回退 CPU AOT bridge，避免继续扩大裸 PTX 面积。
- [x] **R4：生产化清理**：删除旧手写 PTX 模板和 fallback 调用；新增 `LITENN_CUDA_AOT_TARGET=sm_<digits>` 目标架构策略，默认 `sm_30`；补充 artifact/target diagnostics、CUDA native carrier-style 验证、benchmark 覆盖和 release checklist。

阶段交付标准：

- R0 完成后，`CompiledModule.cpp` 不再直接拼接 PTX 文本，只负责 graph matcher、feature flags、payload 组装和 artifact 序列化；`CUDANativeCodegen` 负责 kernel symbol、PTX 文本和文本到 payload bytes 的转换。
- R1 完成后，现有 `CompiledModuleCUDATest` 与 `CompiledModuleTest` 可覆盖 native artifact 的加载/运行，新增测试需要直接检查 PTX payload 的 NUL 结尾、kernel symbol、feature flags 和 broadcast shape metadata 不发生 ABI 漂移。
- R2 完成后，MLIR/NVVM 试点必须是可选路径，旧 op 失败时能回退到模板 codegen；当前已验证 `UnaryOp::Negate/Abs/Sqrt/Exp/Log/Sin/Cos` 与 `BinaryOp::Add/Subtract/Multiply/Divide/Max/Min` 的 MLIR 生成 NVPTX PTX 可放入现有 payload 并通过 CUDA Driver 执行，避免同时改动 payload、runtime loader 和 broadcast 索引逻辑。
- R3/R4 完成后，模板路径被删除；目标 arch、compute capability 与 diagnostics 进入验证矩阵，benchmark 和 release checklist 进入 P4 收尾。

设计边界：

- MLIR/NVPTX 只替换 kernel codegen 来源，不改变 `Graph -> Compiler -> CompiledModuleArtifact -> CompiledModule<CUDA>` 的 artifact/load/run 分层。
- `CUDANativeInstructionPayload` 继续承载 target、binary bytes/library-call 标记、feature flags、scalar data、kernel launch table 和参数表。
- cuBLAS MatMul 可继续走 `LibraryCall` payload；混合 payload 支持 per-kernel 分派，因此 MatMulBiasAdd/ReLU 可在同一 payload 中先调用 cuBLAS，再执行 MLIR/NVPTX epilogue kernel。

### P4：生产化验证

- [x] CUDA AOT carrier-style exported symbol 验证：新增 `CompiledModuleArtifact::FromExportedSymbols` 加载 `CUDANative` artifact 并运行 cuBLAS MatMul 的回归。
- [x] CUDA native target diagnostics：新增非法 `LITENN_CUDA_AOT_TARGET` 格式测试，确保 target arch 错误在 codegen/compile 阶段显式失败。
- [x] CUDA unavailable / driver mismatch / arch mismatch 的更完整诊断测试：`CUDADeviceTest` 覆盖无效 device index、非法 image JIT log、unsupported target JIT diagnostics；CUDA 不可用或设备数不足时按测试前置条件显式 skip。
- [x] benchmark 覆盖 CPU AOT、CUDA bridge、CUDA native：`litenn_bench` 注册 Interpreter、CPU AOT Run/RunInto、CUDA CPU-bridge RunInto，以及 CUDANative MatMul/cuBLAS RunInto 条目。
- [x] 文档化 backend/image ABI 迁移策略，并进入 release checklist。

P4 release checklist：

- artifact ABI：保持 rodata v3 backend tag、`CUDANativeInstructionPayload` feature mask、scalar/workspace/launch table schema 向后显式校验；新增 feature 必须扩展 known-mask 并补 payload inspection。
- loader/runtime：carrier-style exported symbols、CPU bridge fallback、CUDANative driver module load、mixed library/PTX launch、workspace 边界和 stream/sync 选项都必须有回归。
- diagnostics：`LITENN_CUDA_AOT_TARGET` 格式错误在 compile/codegen 阶段失败；driver load failure 必须携带 CUDA error name、JIT error log 或 info log，便于定位 PTX/driver/arch mismatch。
- validation：CUDA P4 收尾至少运行 focused build、`CompiledModuleTest|CompiledModuleCUDATest` CTest、`CUDADeviceTest.exe`，benchmark-enabled build 的 `litenn_bench` compile 与 `--benchmark_list_tests=true` registry smoke。

## 当前选择

当前 `Compiler<CUDA>` 采用“native 优先、bridge fallback”的策略：静态 shape、单 subgraph、`Float32` elementwise Negate/Abs/Sqrt/Exp/Log/Sin/Cos/Add/Subtract/Multiply/Divide/Max/Min、同 rank elementwise broadcast、reduce Sum/Mean/Max、concat/slice、二维 MatMul，以及 FusionPass 产生的 MatMulBiasAdd/ReLU 会生成 `CompiledModuleBackend::CUDANative` artifact。`instructions` 中承载 `CUDANativeInstructionPayload`，可表示 PTX/cubin/fatbin driver kernel、cuBLAS library call 或混合 multi-kernel launch；其他图仍生成 `CompiledModuleBackend::CPUNative` artifact，并由 `CompiledModule<CUDA>` 在 CUDA Tensor 边界做 CPU↔CUDA copy。

P1/P2 当前已经提供：

- `CUDANativeInstructionPayload`：序列化 PTX/cubin/fatbin bytes 或 library-call 标记、target、feature flags、scalar data、workspace、kernel launch table 和参数表，直接作为 CUDA native artifact 的 `instructions` payload；runtime 支持 per-kernel library/PTX 分派与 workspace 参数。
- `CUDADriverModule`：基于 CUDA Driver API 的 RAII module shell，支持 module load/unload、默认或外部 stream launch、可选同步和错误诊断；module load 失败时会携带 CUDA JIT error/info log，便于定位 PTX 版本、外部符号或 driver JIT 问题。当前 `CUDADeviceTest.DriverModuleLaunchesPTXKernel` 用 PTX smoke 覆盖最小 launch 路径。
- `CompiledModule<CUDA>` native loader：根据 rodata backend 在 CPU bridge 与 CUDA native loader 间分派；native loader 复制 image bytes 后可独立于 carrier library 生命周期执行。
- `Compiler<CUDA>` native codegen：识别 `[input] -> UnaryOp(input)`、`[lhs, rhs] -> BinaryOp(lhs, rhs)`、broadcast binary、reduce、concat/slice、二维 `Float32` MatMul，以及 MatMulBiasAdd/ReLU fusion。custom kernels 由 MLIR GPU/NVVM 生成 PTX，MatMul 生成 cuBLAS library-call payload，fusion epilogue 使用混合 launch table。
