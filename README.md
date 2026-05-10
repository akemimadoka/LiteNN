LiteNN
====

学习用的 C++26 编译器风格神经网络库。

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
