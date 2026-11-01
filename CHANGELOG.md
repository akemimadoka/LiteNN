# 更新日志

LiteNN 的所有重要变更都会记录在此文件中。

格式参考 Keep a Changelog，并补充本仓库特有的序列化、image 兼容性和包发布相关说明。

## 未发布

### 新增

- 支持 `import LiteNN;` 的 named module 导出路径，包括安装/导出元数据和外部包 smoke 验证。
- 新增 `CompiledModuleArtifact`，将编译期产物所有权与运行时加载语义分层。
- 新增 `MemorySafetyTest` 与可配置的 sanitizer 入口。
- 新增 Graph/MLIR/CompiledModule dump 工具，用于运行时与编译器诊断。
- 新增训练 API、模型保存/加载、forward-only 提取、batch 辅助接口和 carrier 示例。
- 新增 vNext 模型包（`SaveVNextModelPackage` / `LoadVNextModelPackage`）与 separated artifact，作为持久化和部署格式。
- 新增可选的 CUDA 与 Vulkan native 后端，按能力检查启用，fallback 在 schedule 和 profile 中可见。
- 新增 GGUF、Torch manifest、safetensors 与 LoRA 导入，以及面向 Qwen2.5-Coder 14B 的 stateful CPU AOT 解码路径。
- 新增 `LiteNN/ProductionSupport.h`，供工具和 CLI 查询各路径的支持级别与 ABI。

### 变更

- pre-1.0 阶段的 CMake 包版本匹配规则从 same-major compatibility 收紧为 same-minor compatibility。
- 发布/兼容策略以及面向维护者的工程文档现统一收录在 `docs/` 下。
- CI 的 LLVM/MLIR 依赖固定到发行版，不再跟随 apt.llvm.org 的 trunk 快照。

### 移除

- 移除 raw Graph archive（`SaveModel` / `LoadModel`）和 public runtime/compiler 的 raw `Graph&` 入口，改用 vNext 模型包。此次移除未经过 `docs/Versioning.md` 规定的弃用周期。

### 修复

- 修复 Qwen2 的 NeoX RoPE 配对，此前生成结果在早期 token 就与 llama.cpp 分歧。
- 修复 CPU AOT 线程池在默认线程数下的间歇性死锁。
- CPU AOT 工作线程遵守宿主进程的 CPU 亲和域。
- 修复 bounded 激活标量尾部的溢出边界。

### 兼容性说明

- 头文件入口和 C++ named module 入口都受支持，但两者当前仍共享同一套运行时 ABI。
- vNext 模型包与 `CompiledModule` image 兼容规则见 `docs/Versioning.md`。

## 0.1.0 - 2026-05-11

### 新增

- 静态图前端、Autograd、Validation、Interpreter 运行时，以及基于 MLIR/LLVM 的 CPU AOT 路径。
- 可安装的 CMake 包导出和 Conan recipe，支持基础 LiteNN runtime，以及可选的 compiler 组件导出。
- 模型保存/加载、签名 API、线程安全约束，以及 carrier/object-file 加载流程。

### 说明

- 这是首个可打包发布的基线版本；`1.0` 前的版本治理规则见 `docs/Versioning.md`。