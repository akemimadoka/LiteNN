# LiteNN AOT 部署指南

## 三层对象

- `Graph`：训练或推理前端 IR。
- `CompiledModuleArtifact`：拥有 rodata/native object bytes 的编译期产物。
- `CompiledModule<CPU>`：运行期已加载 CPU native module，可执行 `Run` / `RunInto` / `RunManyInto`。
- `CompiledModule<CUDA>`：CUDA 运行期 module。`CPUNative` artifact 通过 CPU AOT bridge 执行；`CUDANative` artifact 反序列化 CUDA payload、加载 Driver module，并直接使用 CUDA Tensor 指针 launch kernel。

## 推荐部署路径

1. 训练后使用 `ExtractForwardOnlyGraph` 提取推理图。
2. 编译得到 `CompiledModuleArtifact`。
3. 选择以下一种承载方式：
   - 直接持久化 rodata + instructions bytes
   - `WriteObjectFile()` 生成 carrier object，再打包进静态库或共享库
4. 在部署侧使用 `CompiledModule<CPU>::Load`、`CompiledModule<CUDA>::Load(image, CUDA{})` 或 `artifact.Load()` 恢复运行时模块。

## 兼容与生命周期

- `Load` 返回后，原始 rodata/instruction 地址可释放。
- `Run`/`RunInto`/`RunManyInto` 可并发，但要与 `Load` / 析构分离。
- image 兼容性依赖 rodata header 校验；不同 format version、pointer size、endianness、target triple、backend 必须拒绝加载或进入明确的 bridge 路径。
- CUDA AOT 支持两类 image：`CPUNative` 的 instructions 是 CPU native object bytes，`Load(image, CUDA{})` 会在 CUDA Tensor 边界做 CPU↔CUDA copy；`CUDANative` 的 instructions 是 `CUDANativeInstructionPayload`，在原有 `instructions` 字节区承载 PTX/cubin/fatbin、feature flags、scalar data、launch table 和 workspace metadata，并通过 CUDA Driver module 生命周期管理执行。
- 当前 CUDA native codegen 是 P2 MVP：覆盖静态 shape、单 subgraph、`Float32` 单输入单输出 elementwise Negate/Abs/Sqrt、双输入单输出 elementwise Add/Subtract/Multiply/Divide、同 rank binary broadcast，以及二维 `Float32` MatMul/cuBLAS library call。不匹配的图会回退到 CPU AOT bridge。

## 包分发建议

- 头文件/模块消费：通过安装后的 `find_package(LiteNN)`。
- 编译器组件：需要显式请求 `COMPONENTS Compiler`，并保证 LLVM/MLIR 包前缀可见。
- Conan：复用仓库 CMake install/export 逻辑，版本与兼容规则以 `docs/Versioning.md` 为准。

## 升级前必须重验的内容

- carrier object 导出符号前缀
- `CompiledModuleArtifact` metadata schema
- `CompiledModule::Load` 的兼容检查逻辑
- 外部 `find_package + import LiteNN` smoke
