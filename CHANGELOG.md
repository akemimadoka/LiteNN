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

### 变更

- pre-1.0 阶段的 CMake 包版本匹配规则从 same-major compatibility 收紧为 same-minor compatibility。
- 发布/兼容策略以及面向维护者的工程文档现统一收录在 `docs/` 下。

### 兼容性说明

- 头文件入口和 C++ named module 入口都受支持，但两者当前仍共享同一套运行时 ABI。
- `Graph` 模型序列化与 `CompiledModule` image 兼容规则见 `docs/Versioning.md`。

## 0.1.0 - 2026-05-11

### 新增

- 静态图前端、Autograd、Validation、Interpreter 运行时，以及基于 MLIR/LLVM 的 CPU AOT 路径。
- 可安装的 CMake 包导出和 Conan recipe，支持基础 LiteNN runtime，以及可选的 compiler 组件导出。
- 模型保存/加载、签名 API、线程安全约束，以及 carrier/object-file 加载流程。

### 说明

- 这是首个可打包发布的基线版本；`1.0` 前的版本治理规则见 `docs/Versioning.md`。