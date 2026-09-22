LiteNN
====

[![ci](https://github.com/akemimadoka/LiteNN/actions/workflows/ci.yml/badge.svg)](https://github.com/akemimadoka/LiteNN/actions/workflows/ci.yml)

学习用的 C++26 编译器风格神经网络库。

~~人类含量很低，只有最初的 Tensor/Device/Graph 部分是全人工古法编程的，其他部分合作完成~~

当前仓库已经包含：

- 静态计算图前端、Autograd、Interpreter 运行时
- 基于 MLIR/LLVM 的 CPU AOT 编译与 CompiledModule 加载执行
- 模型保存/加载、训练 API、MNIST 与 carrier 示例
- 可安装的 CMake 包导出，支持 `find_package(LiteNN)`

## 文档

详细文档索引见 [docs/README.md](docs/README.md)。常用入口：

- [docs/Architecture.md](docs/Architecture.md)：架构与路线图
- [docs/Versioning.md](docs/Versioning.md)：版本、兼容与弃用策略
- [docs/APIGuide.md](docs/APIGuide.md)：public API 入口与升级边界
- [docs/AOTDeployment.md](docs/AOTDeployment.md)：AOT artifact / carrier / 部署路径
- [docs/Troubleshooting.md](docs/Troubleshooting.md)：常见构建与消费问题排查
- [CHANGELOG.md](CHANGELOG.md)：用户可见改动记录

## 构建

仅构建基础运行时：

```powershell
cmake -S . -B build
cmake --build build
```

## 开发格式化

仓库使用根目录的 `.clang-format` 统一格式化所有已跟踪的 C/C++ 源码文件。手动格式化：

```powershell
python scripts/format_sources.py
```

clone 后 Git 不会自动安装仓库里的 hook 模板。需要在本地执行一次：

Linux/macOS:

```sh
sh scripts/install_git_hooks.sh
```

Windows:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File scripts/install_git_hooks.ps1
```

安装后，pre-commit hook 会在提交前运行全量 `clang-format`，并重新暂存本次提交中已经暂存的文件。

启用 MLIR/AOT 编译器：

```powershell
cmake -S . -B build -DLITENN_ENABLE_MLIR=ON -DCMAKE_PREFIX_PATH="<llvm-cmake-dir>;<mlir-cmake-dir>"
cmake --build build
```

## 临时产物管理

性能测试和 GGUF AOT 实验应复用 `build/.litenn-cache/gguf-shared-weights` 中的共享权重，避免每个实验目录复制一份多 GiB 的 `weights.bin`。可以先盘点 `build` 中最大的直接子项：

```powershell
python311 scripts/manage_build_artifacts.py --root build
```

按容量、目录数和递归文件数生成清理计划时默认只预览，并自动保护 `.litenn-cache`、`.litenn-shared-weights` 以及包含共享缓存、CMake 构建树或链接的目录。确认列表后才添加 `--apply`：

```powershell
python311 scripts/manage_build_artifacts.py --root build --max-total-gib 32 --max-entries 80 --max-files 10000 --keep qwen_speed_cache
python311 scripts/manage_build_artifacts.py --root build --max-total-gib 32 --max-entries 80 --max-files 10000 --keep qwen_speed_cache --apply
```

只检查是否超限（不删除文件，超限退出码为 `1`），可以在实验前后执行：

```powershell
python311 scripts/manage_build_artifacts.py --root build --max-total-gib 32 --max-entries 80 --max-files 10000 --keep qwen_speed_cache --check
```

脚本只把 `build` 的直接子项作为清理单元，不跟随符号链接或 Windows junction；默认拒绝仓库根目录和 CMake 构建树。受保护文件也计入容量和数量，无法达到上限时会提示；`--apply` 后仍然超限也返回 `1`，不会以删除共享权重的方式强行达标。统计值为逻辑大小，不是磁盘实际分配空间。真正的 CMake 输出建议继续放在 `build-release` 等独立目录，不与实验产物混放。

实验尽量固定复用同一个 `--workdir` 和 `--aot-cache-dir`，只在需要保留对照证据时另建目录，不为每次运行复制模型或共享权重。清理前停止使用这些目录的进程，并用 `--keep <直接子项名>` 保护正在使用的缓存、尚未归档的结果以及自定义共享目录。按年龄清理依据的是修改时间，不代表最后访问时间；脚本不会自动识别运行中的实验，也不会在模型启动时自动删除旧结果。

## 安装

当前包版本为 `0.1.0`，安装后会导出 `LiteNNConfig.cmake` 和 `LiteNNConfigVersion.cmake`：

- `0.x` 阶段 CMake 包按同 minor 版本兼容匹配，避免在 pre-1.0 阶段误接受潜在 breaking minor 升级

```powershell
cmake --install build --prefix <install-prefix>
```

安装树包含：

- `include/LiteNN...` 公开头文件
- `lib/libLiteNN.*` 基础运行时库
- `lib/libLiteNNCompiler.*` 可选 AOT 编译器库（仅 `LITENN_ENABLE_MLIR=ON` 时）
- `lib/cmake/LiteNN/` 包配置与导出 targets

## 作为 CMake 包使用

基础运行时：

```cmake
find_package(LiteNN CONFIG REQUIRED)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE LiteNN::LiteNN)
```

可选 AOT Compiler 组件：

```cmake
find_package(LiteNN CONFIG REQUIRED COMPONENTS Compiler)

add_executable(app main.cpp)
target_link_libraries(app PRIVATE LiteNN::LiteNNCompiler)
```

请求 `Compiler` 组件时，调用方还需要把 LLVM/MLIR 的 CMake 包前缀加入 `CMAKE_PREFIX_PATH`。

仓库里的 `cmake/PackageSmokeTest` 提供了最小外部消费工程，可用于验证安装后的 `find_package(LiteNN)` 链路。

## Conan 打包

仓库根目录现在提供 `conanfile.py`，会复用现有的 CMake install/export 逻辑来生成 Conan 包。

导出并创建基础 runtime 包：

```powershell
conan create . -s compiler=gcc -s compiler.version=<gcc-version> -s compiler.cppstd=gnu26 --build=missing
```

如需把可选 AOT 编译器一起打进包中，可启用 `with_mlir` 选项；这要求构建环境里已经能被 CMake 发现到 LLVM/MLIR 的包配置：

```powershell
conan create . -s compiler=gcc -s compiler.version=<gcc-version> -s compiler.cppstd=gnu26 -o litenn/*:with_mlir=True --build=missing
```

当前默认 `conan profile detect` 在 Windows 上通常会生成 MSVC profile，但 LiteNN 代码目前依赖实验性 C++26 反射工具链，因此需要改用与现有构建一致的 GCC/Clang profile。

消费侧继续使用现有的 CMake 包入口：

```cmake
find_package(LiteNN CONFIG REQUIRED)
target_link_libraries(app PRIVATE LiteNN::LiteNN)
```

## 调试 Dump API

基础运行时现在提供 Graph 文本 dump：

```cpp
#include <LiteNN.h>

std::string graphText = LiteNN::Debug::DumpGraph(graph);
```

启用 `LITENN_ENABLE_MLIR=ON` 并链接 `LiteNN::LiteNNCompiler` 后，还可以输出不同阶段的 MLIR 和 CompiledModule metadata：

```cpp
#include <LiteNN/Compiler/Dump.h>

std::string inputMlir = LiteNN::Debug::DumpMLIR(graph, LiteNN::Debug::MLIRDumpStage::InputDialect);
std::string loweredMlir = LiteNN::Debug::DumpMLIR(graph, LiteNN::Debug::MLIRDumpStage::AfterLowering);
std::string metadata = LiteNN::Debug::DumpCompiledModuleMetadata(artifact);
```
