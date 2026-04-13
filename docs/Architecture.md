# LiteNN 架构文档

## 项目概述

LiteNN 是一个使用 **C++26** 实现的神经网络库，采用**编译器前端**风格进行设计。核心思想是将神经网络计算表示为一个静态计算图（Graph），通过 Pass 系统对其进行变换和优化，最终由后端执行。目前实现了基于解释执行的运行时（Interpreter），AOT 编译器后端待实现。

---

## 整体架构

```
用户代码
  ↓ 构建
Graph (前端表示，设备无关)
  ↓ Pass 变换
  AutogradPass → 生成 backward 子图
  InlinePass   → 展开 CallNode（规划中）
  FusionPass   → 算子融合（规划中）
  ↓
Graph (优化后)
  ↓
Runtime::Interpreter<D>    ← 当前实现（逐节点解释执行）
Compiler<D>                ← 规划中（AOT 编译）
  ↓
CompiledModule<D>::Run()
```

---

## 核心概念

### Graph

Graph 是计算的**封闭世界**容器，对外只暴露 forward/backward 两个入口点。

- 封闭性使得跨子图的全局优化成为可能（如内联、公共子表达式消除）
- 持有所有 Subgraph（通过 `deque` 保证引用稳定）、Variable、ActivationSlot
- `SubgraphId` / `NodeId` 采用 **arena + 索引**风格，避免裸指针，利于克隆、序列化

```
Graph
├── subgraphs_: deque<Subgraph>        // 不会导致已有引用失效
├── variables_: vector<shared_ptr<Variable>>
├── activationSlots_: vector<ActivationSlot>
├── forward_: SubgraphId
└── backward_: optional<SubgraphId>   // AutogradPass 后填充
```

### Subgraph

一个有明确参数和返回值的计算块，内部是 **按拓扑序存储**的节点 arena。

- 节点通过 `NodeId`（vector 下标）引用
- 节点之间的数据流通过 `NodeOutput{nodeId, port}` 表示（支持多输出节点）
- 每个节点附带 `outputInfos`（类型和形状元信息），无需执行即可做静态分析

```
Subgraph
├── params_: vector<SubgraphParam>     // 输入参数声明 (dtype, shape)
├── results_: vector<NodeOutput>       // 输出引用
└── nodes_: vector<NodeEntry>          // arena，拓扑序
    └── NodeEntry { NodeVariant node, vector<OutputInfo> outputInfos }
```

> **注意**：`AddNode`/`AddParam` 可能导致 `GetNodeEntry`/`Nodes()` 等返回的引用或 span 失效（vector 扩容）。构建期间应只使用返回的 NodeId，不持有节点引用。

### 节点类型（NodeVariant）

核心原则：**最小原语集**，高级操作通过 CallNode 组合。

| 节点 | 说明 |
|------|------|
| `ParamRefNode` | 引用当前 Subgraph 的第 i 个参数 |
| `ConstantNode` | 持有常量张量（`Tensor<PolymorphicDevice>`） |
| `VariableRefNode` | 引用 Graph 级别的可训练参数 |
| `UnaryOpNode` | 原语一元操作（见 UnaryOp 枚举） |
| `BinaryOpNode` | 原语二元操作（见 BinaryOp 枚举） |
| `CastNode` | 显式类型转换（`input` → `targetType`） |
| `CallNode` | 调用另一个 Subgraph（复合操作入口） |
| `CondNode` | 条件分支（`thenBranch` / `elseBranch`，各为 SubgraphId） |
| `SaveActivationNode` | 保存激活值到 ActivationStore（透传，不阻断数据流） |
| `LoadActivationNode` | 从 ActivationStore 读取激活值 |

NodeVariant 通过 C++26 反射从 `inline namespace Node` 的成员类型自动生成 `std::variant`，无需手动枚举。

### 原语操作

**UnaryOp**：`Negate`, `Abs`, `Sqrt`, `Exp`, `Log`, `Sin`, `Cos`, `Tan`, `Arcsin`, `Arccos`, `Arctan`, `Transpose`, `LogicalNegation`

**BinaryOp**：`Add`, `Subtract`, `Multiply`（Hadamard 积）, `Divide`, `MatMul`, `Less`, `Greater`, `Equal`

高级操作（如 ReLU、Sigmoid）通过 Subgraph 组合表达，不扩充枚举。

### Variable

可训练参数，Graph 级别共享，所有 Subgraph 通过 `VariableRefNode` 访问。

- 持有 `data_` 和 `grad_`（均为 `Tensor<PolymorphicDevice>`）
- Graph 是设备无关的前端表示，Variable 使用 PolymorphicDevice 存储
- 构造为私有，通过 `Variable::Create<D>(Tensor<D>)` 工厂方法创建

### ActivationStore

前向/反向子图共享中间激活值的机制，避免重计算。

- `ActivationSlot` 在 AutogradPass 的分析阶段静态分配（当前无 WhileNode，所有槽位静态可确定）
- 前向中 `SaveActivationNode` 写入，反向中 `LoadActivationNode` 读取
- Interpreter 持有 `vector<optional<Tensor<D>>> activationStore_`，`RunForward` 前自动初始化，`RunBackward` 直接复用

---

## Device 抽象

基于 C++26 Concept，通过 `DeviceTraits<D>` 特化提供设备实现。

```cpp
template <typename T>
concept Device = requires { /* Allocate, Deallocate, ZeroFill, CopyToCPU,
                               CopyFromCPU, ConvertTo, DoUnaryOp, DoBinaryOp */ };
```

- **CPU**：当前唯一完整实现，支持所有 Op 和广播
- **PolymorphicDevice**：类型擦除包装，用于 Graph 级别的设备无关存储（Variable、ConstantNode）
  - 使用 `IsSameDeviceType()` 检查底层设备类型（注意：只比较类型，不比较实例，多设备场景需注意）
- 未来可添加 CUDA/Metal 等设备特化

---

## Pass 系统

Pass 是 `Graph → Graph` 的变换，接口为：

```cpp
struct Pass {
    virtual void Run(Graph& graph) = 0;
};
```

### AutogradPass

自动微分 Pass，核心方法 `ProcessSubgraph(graph, fwdId)` 可递归处理任意子图（支持 CallNode 中的嵌套子图）。

**流程**：
1. **分析**：统计消费者数量，确定需要保存的中间激活值（`AnalyzeSavedValues`）
2. **Augmented Forward**：复制原 forward 子图，在需要保存的节点后插入 `SaveActivationNode`；更新 CallNode 指向 callee 的 augmented forward
3. **Backward 构建**：
   - 参数：`[forward_inputs..., grad_outputs...]`
   - 逆拓扑序遍历前向节点，为每种 Op 生成对应梯度节点
   - 多消费者的值通过 `BinaryOp::Add` 累积梯度
   - 结果：`[grad_inputs..., grad_variables...]`

**已支持操作的梯度规则**：

| Op | 梯度 |
|----|------|
| Add | `da=dy, db=dy` |
| Subtract | `da=dy, db=-dy` |
| Multiply | `da=dy*b, db=dy*a` |
| Divide | `da=dy/b, db=-dy*a/b²` |
| MatMul | `da=dy@bᵀ, db=aᵀ@dy` |
| Negate | `dx=-dy` |
| Transpose | `dx=transpose(dy)` |
| Abs | `dx=dy*sign(x)` |
| Sqrt | `dx=dy*0.5/output`（保存前向输出）|
| Cast | 可微源类型 cast 回；Bool/Int 源不传播 |
| CallNode | 递归对 callee 做 autograd，通过 callee backward 传播 |
| Less/Greater/Equal/LogicalNegation | 不可微，不传播 |

**待实现**：`Power`, `Maximum`, `Minimum`, `CondNode` 的梯度

---

## Runtime

### Interpreter\<D\>

逐节点解释执行 Graph，用于**调试和功能验证**。不是最终的执行后端。

- 维护 slot 表 `vector<vector<Tensor<D>>>` 按 NodeId 存储各节点输出
- `RunForward` 自动初始化 ActivationStore
- `RunBackward` 复用已填充的 ActivationStore
- `RunSubgraph` 是通用入口，支持执行任意子图

---

## Layer 系统

高级操作通过工厂函数构建 Subgraph，而非独立的 Layer 类。

```cpp
// Layer/ReLU.h
SubgraphId BuildReLU(Graph& graph, DataType dtype, ShapeView shape);
// ReLU(x) = x * Cast(x > 0, dtype)
// 组合：Greater + Cast + Multiply，共 4 个节点（含常量 0）
```

---

## 设计偏好

1. **编译器前端风格**：Graph 是静态前端表示，不持有执行逻辑；Pass 做变换；后端做执行
2. **最小原语集**：BinaryOp/UnaryOp 保持精简，复杂操作通过 Subgraph 组合
3. **先讨论后实现**：涉及架构的改动，先充分讨论再动手
4. **AOT 优先**：目标是 AOT 编译，Interpreter 仅用于调试
5. **C++26 特性**：大量使用 Reflection (`<meta>`)、Concepts、`deducing this`、`constexpr`/`consteval`、`template for`
6. **静态保存激活值**：Autograd 采用保存中间值而非重计算，当前设计下槽位静态可确定
7. **控制流**：支持 `CondNode`（结构化条件分支），`WhileNode`（循环）暂缓——动态迭代次数需要 tape 机制，与 AOT 目标有张力

---

## 文件结构

```
src/
├── LiteNN.h                      // 聚合 include 入口
├── LiteNN.ixx                    // C++26 module 导出（TODO: 完善）
└── LiteNN/
    ├── Misc.h                    // 工具：EnumDispatch, ShapeView, EnumToString
    ├── Operators.h               // UnaryOp, BinaryOp, DataType, OpTraits
    ├── Device.h                  // Device concept, CPU, PolymorphicDevice
    ├── Device.cpp                // PolymorphicDevice 实现
    ├── Tensor.h                  // Tensor<D> 模板，所有张量操作
    ├── Graph.h                   // Graph, Subgraph, NodeVariant 等核心类型
    ├── Pass.h                    // Pass 基类
    ├── Layer/
    │   └── ReLU.h               // BuildReLU 子图工厂
    ├── Pass/
    │   └── AutogradPass.h       // 自动微分 Pass
    └── Runtime/
        └── Interpreter.h        // 解释执行器

tests/
├── TensorTest.cpp               // Tensor 基础操作测试
└── InterpreterTest.cpp          // Graph/Interpreter/AutogradPass 测试
```

---

## TODO List

### 近期

- [x] **BinaryOp 补全**：添加 `Power`, `Maximum`, `Minimum` 及其 Device 实现和梯度规则
- [x] **ReduceNode**：新节点类型，支持 `ReduceSum`/`ReduceMean`/`ReduceMax`（带 axis 参数）——实现标量 loss 的必要条件
- [x] **ReshapeNode**：新节点类型，改变 shape 不改变数据——CNN→FC 衔接必需
- [x] **CondNode 梯度**：AutogradPass 支持 `CondNode` 的反向传播
- [x] **Stride 支持**：Tensor 添加了元素级 stride 基础设施（`strides_` 成员、`ComputeContiguousStrides`、`Strides()`、`IsContiguous()`），当前所有 tensor 仍为连续布局，为未来 lazy transpose/slice view 打基础
- [x] **多 GPU 实例**：`PolymorphicDevice` 新增 `IsSameDevice` 接口，基于 `operator==` 进行实例级别比较；`CopyToDevice` 已改用实例级检查

### 中期

- [ ] **InlinePass**：展开 CallNode，为跨子图优化铺路
- [ ] **ConstFoldPass**：常量折叠
- [ ] **FusionPass**：算子融合（通用） + 设备特定 FusionPass
- [ ] **Concat/SliceNode**：多分支网络（skip connection、attention）必需
- [ ] **WhileNode + TapeStack**：支持循环，需引入动态长度的 ActivationStore 机制
- [x] **Variable gradient accumulation across CallNode**：`EmitCallGrad` 和 `EmitCondGrad` 现在正确提取 callee backward 的 variable 梯度输出并传播到父级；`ProcessSubgraph` 合并直接 VariableRefNode 梯度与 callee 传播的梯度；CondNode 分支 variable 集合不同时通过 wrapper 子图补零对齐

### 长期

- [ ] **AOT Compiler\<D\>**：将 Graph 编译为设备特定可执行代码（`CompiledModule<D>`）
- [ ] **JIT / Template Interpreter**：`Runtime/` 目录预留了扩展空间
- [ ] **更多 Device**：CUDA、Metal 等 GPU 后端
- [ ] **LiteNN.ixx module 导出**：完善 C++26 module 接口
- [ ] **Interprocedural Specialization（封闭世界跨子图优化）**：Graph 本身是封闭世界，除前向/后向入口外的所有子图只需处理内部调用。对不适合 inline 的子图，可对所有 call site 的参数属性求交集，利用调用中共同的特征进行特化优化（部分求值、分支消除、更激进的常量折叠等）。类似 Dart type flow graph 的思路。预期最大收益点：AutogradPass 生成图的结构性冗余消除（零梯度路径剪枝、dead branch elimination）、WhileNode 引入后循环体的跨子图优化。建议在 InlinePass + ConstFoldPass 基础设施就绪、WhileNode 实现后评估 ROI
