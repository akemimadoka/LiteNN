# LiteNN Pass 编写指南

## Pass 基本要求

- 输入输出语义必须可由测试说明清楚。
- 进入后端执行或编译前，Graph 应能重新通过 `ValidateGraph`。
- 如果 Pass 引入新的 Graph invariant，必须同步更新架构文档和测试。

## 推荐流程

1. 写出变换前后的最小 Graph 例子。
2. 明确是否要求 fixpoint 执行。
3. 明确是否需要跨子图递归。
4. 为失败诊断补充足够上下文。
5. 为典型正例、反例和回归场景补测试。

## 测试要求

- 单元测试：最小 IR 结构和边界条件。
- 集成测试：与 Autograd/Inline/ConstFold/Fusion 或编译链组合后的行为。
- 如果 Pass 影响 deployment path，需要额外验证 forward-only/AOT/export 路径。

## 常见错误

- 持有 `Subgraph::Nodes()` 或 `GetNodeEntry()` 返回引用跨 `AddNode` / `AddParam` 使用。
- 忽略多输出 `NodeOutput{node, port}` 的 port 传播。
- 只改 forward path，忘记同步 backward/call/cond/while 相关语义。
- 引入新 Node/metadata 后未更新 serialization、dump 或 validator。