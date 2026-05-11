# LiteNN API Guide

## 推荐入口

- 纯 runtime / interpreter：`#include <LiteNN.h>` 或 `import LiteNN;`
- 可选 AOT compiler：`#include <LiteNN/Compiler/CompiledModule.h>` 与相关 compiler headers
- 调试 dump：`#include <LiteNN/Debug/Dump.h>` 或 compiler dump headers

## 典型使用路径

### 构图与解释执行

1. 构建 `Graph` 与 `Subgraph`
2. 调用 `Validation::ValidateGraph(graph)`
3. 用 `Runtime::Interpreter<D>` 执行 `RunForward` / `RunBackward`

### 训练

1. 构建含 `Variable` 的训练图
2. 使用 `Training::CPUTrainer<Optimizer>`
3. 通过 `Step` / `StepSoftmaxCrossEntropy` / `StepSoftmaxCrossEntropyBatch` 执行训练循环

### AOT 编译与部署

1. 可选先 `ExtractForwardOnlyGraph(graph)`
2. `Compiler<CPU>::CompileArtifact(graph)`
3. `artifact.Load()` 或 `CompiledModule<CPU>::Load(...)`
4. `Run` / `RunInto` / `RunManyInto`

## 稳定性边界

- `LiteNN.h` / `import LiteNN;`：当前视为 runtime 主入口。
- `LiteNN::LiteNNCompiler`：可选组件，需要 `LITENN_ENABLE_MLIR=ON`。
- `CompiledModuleArtifact` / carrier object workflow：视为部署入口的一部分，变更需要更新 changelog 与兼容策略。

## 不推荐的直接依赖

- 直接依赖 `NodeVariant` 的内部布局或 `inline namespace Node` 的反射实现细节。
- 依赖 `Graph` / `Subgraph` 的内部容器类型或地址稳定性以外的实现细节。
- 将测试辅助、dump 输出文本、异常字符串当作长期稳定 API。

## 升级建议

- 优先通过 `LiteNN.h` 或 `import LiteNN;` 使用 public surface。
- 升级前阅读 `CHANGELOG.md` 与 `docs/Versioning.md`。
- 如果使用了 model serialization 或 carrier object，升级时必须额外验证 binary compatibility。