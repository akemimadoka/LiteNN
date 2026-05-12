# LiteNN 版本与兼容策略

## 版本语义

LiteNN 使用语义化版本号 `MAJOR.MINOR.PATCH`，但在 `1.0.0` 之前采用更保守的兼容承诺：

- `0.y.z -> 0.y.(z+1)`：允许 bugfix、文档更新、诊断增强、实现替换；不应引入公开 API 或序列化格式破坏。
- `0.y.z -> 0.(y+1).0`：允许 source/binary/package compatibility 破坏；调用方需要重新验证升级。
- `1.y.z` 之后：遵循通常的 semver，breaking change 进入新的 major。

为与这条规则一致，CMake 包版本文件在 `0.x` 阶段使用 same-minor 匹配；进入 `1.x` 后再恢复 same-major 匹配。

## 公开兼容面

当前需要显式治理的兼容面包括：

- 头文件 public API：`LiteNN.h` 与 `LiteNN/...` 下的公开声明。
- C++ named module：`import LiteNN;` 对应的公开 surface 与 runtime ABI。
- CMake package surface：`LiteNN::LiteNN`、可选 `LiteNN::LiteNNCompiler`、组件名、导出路径。
- Conan recipe surface：选项名、默认值、package layout、cmake package integration。
- Graph model serialization：`SaveModel/LoadModel` 的 on-disk binary format。
- CompiledModule image ABI：rodata header、metadata schema、instruction image compatibility checks。

## 弃用策略

LiteNN 当前没有正式的 API deprecation pipeline；从现在开始采用以下规则：

- 如果一个 public API 要移除，先在一个 minor release 中标记 deprecated，并在 README/changelog 中给出替代路径。
- 进入 `1.0.0` 前，deprecated API 最少保留一个 `0.y` minor 周期。
- 进入 `1.0.0` 后，deprecated API 最少保留一个 major 周期内的两个 minor release。

如果某项接口由于安全性、数据损坏风险或严重 ABI 缺陷必须直接撤回，必须在 changelog 的 Compatibility Notes 中单独说明。

## Graph 模型序列化规则

`SaveModel/LoadModel` 的格式版本必须按以下原则演进：

- 新增可选字段或向后兼容 metadata：提升 format minor version 或在现有 version 下引入显式 feature flag。
- 改变已有字段语义、删除字段、改变 Node payload 编码：提升 format major version。
- 新 reader 应尽量兼容旧 writer；旧 reader 不要求兼容新 writer。
- 每次格式变更都必须更新架构文档、`docs/Versioning.md` 和 changelog。

如果未来需要跨版本迁移工具，优先提供“旧格式读入 -> Graph -> 新格式写出”的显式升级路径，而不是在 Loader 中静默重写。

## CompiledModule image 规则

`CompiledModuleArtifact` / `CompiledModule::Load` 的 image ABI 必须保持以下约定：

- rodata header 中的 magic、format version、pointer size、endianness、target triple、backend 是强校验字段。
- 任何导致 native object bytes 解释方式变化的修改，都必须提升 image format version。
- image format 兼容性不等于 source compatibility；即使 C++ API 保持稳定，不兼容 image 也必须被显式拒绝加载。
- carrier object 导出的 symbol 前缀规则一旦变化，必须记为 breaking packaging/deployment change。

## 发布流程最小要求

每次正式版本发布前至少完成：

- 更新 `project(LiteNN VERSION ...)`。
- 更新 `CHANGELOG.md`。
- 重新验证安装导出、Conan 包、外部 `find_package` smoke。
- 如果触及 serialization/image/package surface，更新本文件和部署文档。
