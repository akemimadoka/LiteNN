# LiteNN AOT 部署指南

## 三层对象

- `Graph`：训练或推理前端 IR。
- `CompiledModuleArtifact`：拥有 rodata/native object bytes 的编译期产物。
- `CompiledModule<CPU>`：运行期已加载 CPU native module，可执行 `Run` / `RunInto` / `RunManyInto`。
- `CompiledModule<CUDA>`：第一版 CUDA AOT bridge，使用 CUDA Tensor 作为输入/输出边界，内部复用 CPU AOT image/JIT。

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
- 当前 CUDA AOT bridge 的 instructions 仍是 CPU native object bytes；`Load(image, CUDA{})` 不要求 carrier library 在执行期间继续存在，但执行时会发生 CPU↔CUDA copy。原生 CUDA AOT 需要新增 backend metadata、GPU binary 打包和 CUDA Driver module 生命周期管理。

## 包分发建议

- 头文件/模块消费：通过安装后的 `find_package(LiteNN)`。
- 编译器组件：需要显式请求 `COMPONENTS Compiler`，并保证 LLVM/MLIR 包前缀可见。
- Conan：复用仓库 CMake install/export 逻辑，版本与兼容规则以 `docs/Versioning.md` 为准。

## 升级前必须重验的内容

- carrier object 导出符号前缀
- `CompiledModuleArtifact` metadata schema
- `CompiledModule::Load` 的兼容检查逻辑
- 外部 `find_package + import LiteNN` smoke
