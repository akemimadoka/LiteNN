LiteNN
====

学习用的 C++26 编译器风格神经网络库。

~~人类含量很低，只有最初的 Tensor/Device/Graph 部分是全人工古法编程的，其他部分合作完成~~

当前仓库已经包含：

- 静态计算图前端、Autograd、Interpreter 运行时
- 基于 MLIR/LLVM 的 CPU AOT 编译与 CompiledModule 加载执行
- 模型保存/加载、训练 API、MNIST 与 carrier 示例
- 可安装的 CMake 包导出，支持 `find_package(LiteNN)`

## 构建

仅构建基础运行时：

```powershell
cmake -S . -B build
cmake --build build
```

启用 MLIR/AOT 编译器：

```powershell
cmake -S . -B build -DLITENN_ENABLE_MLIR=ON -DCMAKE_PREFIX_PATH="<llvm-cmake-dir>;<mlir-cmake-dir>"
cmake --build build
```

## 安装

当前包版本为 `0.1.0`，安装后会导出 `LiteNNConfig.cmake` 和 `LiteNNConfigVersion.cmake`：

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
