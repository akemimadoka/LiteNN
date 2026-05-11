# LiteNN 常见问题排查

## 1. `find_package(LiteNN)` 找不到包

- 检查安装前缀是否加入 `CMAKE_PREFIX_PATH`
- 检查安装树下是否存在 `lib/cmake/LiteNN/LiteNNConfig.cmake`
- 如果请求 `COMPONENTS Compiler`，还要确认 LLVM/MLIR 的 CMake 包前缀可见

## 2. `import LiteNN;` 可以编译但链接失败

- 确认使用的是安装后的导出 targets，而不是只手工包含 `LiteNN.ixx`
- 确认安装树包含 `lib/cmake/LiteNN/modules/` 下的 CMake module metadata
- 确认链接的是 `LiteNN::LiteNN`，而不是只把 `.ixx` 手工加到项目里

## 3. MSYS2/MinGW 下 sanitizer 无法启用

- 这是已知限制；仓库会在配置阶段直接拒绝
- 建议改用 WSL/Linux 或支持 AddressSanitizer 的 MSVC 路径

## 4. Conan 在 Windows 上默认 profile 不可用

- LiteNN 当前要求 reflection-capable GCC/Clang 工具链
- 默认 MSVC profile 通常不满足这一前提，需要切换到与仓库构建一致的 GCC/Clang profile

## 5. `CompiledModule::Load` 后运行崩溃或拒绝加载

- 检查 image 的 target triple、pointer size、endianness 是否匹配
- 检查 rodata/instructions 是否成对来源于同一 artifact/carrier
- 检查是否在承载 LLVM 的库卸载之后仍继续使用 loaded module

## 6. 编辑器静态诊断与真实构建不一致

- 当前实验性 C++26 reflection/modules 工具链可能不被 IDE 完整理解
- 优先信任仓库的 CMake/Ninja 构建结果，再判断是否为编辑器假错