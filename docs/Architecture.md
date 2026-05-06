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
  InlinePass   → 展开 CallNode（已实现）
  ConstFoldPass→ 常量折叠 + 恒等消除 + 死节点消除（已实现）
  FusionPass   → 算子融合（已实现基础模式）
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
- 持有所有 Subgraph（通过 `deque` 保证引用稳定）、Variable、ActivationSlot、TapeSlot
- `SubgraphId` / `NodeId` 采用 **arena + 索引**风格，避免裸指针，利于克隆、序列化

```
Graph
├── subgraphs_: deque<Subgraph>        // 不会导致已有引用失效
├── variables_: vector<shared_ptr<Variable>>
├── activationSlots_: vector<ActivationSlot>
├── tapeSlots_: vector<TapeSlot>       // 循环体栈式激活值槽位
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
| `WhileNode` | 函数式循环：`while condBranch(carry): carry = bodyBranch(carry)`，输出最终 carry 值 |
| `SaveActivationNode` | 保存激活值到 ActivationStore（透传，不阻断数据流） |
| `LoadActivationNode` | 从 ActivationStore 读取激活值 |
| `TapeSaveActivationNode` | 栈式保存激活值到 TapeStore（循环体内使用，每次迭代 push，透传） |
| `TapeLoadActivationNode` | 从 TapeStore 弹出最近的激活值（LIFO，循环反向时使用） |
| `ReduceOpNode` | 归约操作（ReduceSum/Mean/Max，带 axis 参数） |
| `ReshapeNode` | 改变形状不改变数据 |
| `ConcatNode` | 沿指定轴拼接多个张量（所有输入除 axis 维度外 shape 相同） |
| `SliceNode` | 沿指定轴提取连续切片（axis, start, length） |
| `FusedOpNode` | 融合操作：持有 FusionPattern 标签 + body 子图 + args，语义等价于执行 body |

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

### ActivationStore 与 TapeStack

前向/反向子图共享中间激活值的机制，避免重计算。

**ActivationStore**（静态槽位）：
- `ActivationSlot` 在 AutogradPass 的分析阶段静态分配
- 前向中 `SaveActivationNode` 写入，反向中 `LoadActivationNode` 读取
- Interpreter 持有 `vector<optional<Tensor<D>>> activationStore_`，`RunForward` 前自动初始化，`RunBackward` 直接复用
- 用于一次性保存/加载（CallNode、CondNode 的激活值，WhileNode 的迭代计数器）

**TapeStack**（动态栈式槽位）：
- `TapeSlot` 在 AutogradPass 分析循环体时静态分配，但实际存储大小由迭代次数决定
- 前向每次迭代 `TapeSaveActivationNode` push 一个值，反向每次迭代 `TapeLoadActivationNode` pop（LIFO）
- Interpreter 持有 `vector<vector<Tensor<D>>> tapeStore_`，天然形成栈
- 用于 WhileNode 循环体内的 Backpropagation Through Time（BPTT）
- 支持嵌套循环：每层循环有独立的 TapeSlot 集合

---

## Device 抽象

基于 C++26 Concept，通过 `DeviceTraits<D>` 特化提供设备实现。

```cpp
template <typename T>
concept Device = requires { /* Allocate, Deallocate, ZeroFill, CopyToCPU,
                               CopyFromCPU, ConvertTo, DoUnaryOp, DoBinaryOp,
                               DoReduceOp, DoConcatOp, DoSliceOp */ };
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
   - 在循环体内（`insideLoop=true`），使用 `TapeSlot`（`AddTapeSlot`）代替 `ActivationSlot`，生成 `TapeSaveActivationNode`/`TapeLoadActivationNode`
   - WhileNode 额外注册：一个 ActivationSlot 保存迭代计数，K 个 TapeSlot 保存每次迭代的 carry 值
2. **Augmented Forward**：复制原 forward 子图，在需要保存的节点后插入 `SaveActivationNode` / `TapeSaveActivationNode`；更新 CallNode 指向 callee 的 augmented forward
   - WhileNode 生成计数版包装子图（cond/body 各增一个 counter carry），前向结束后 SaveActivation 保存最终计数值
3. **Backward 构建**：
   - 参数：`[forward_inputs..., grad_outputs...]`
   - 逆拓扑序遍历前向节点，为每种 Op 生成对应梯度节点
   - 多消费者的值通过 `BinaryOp::Add` 累积梯度
   - WhileNode 生成反向 WhileNode（BPTT）：读取迭代计数 N，构建反向 condBranch（`counter > 0`）和 bodyBranch（TapeLoad carry → 调用 body backward → 更新 variable 梯度累积器 → counter-1），循环结束后提取 grad_inputs 和 grad_variables
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
| Concat | `dx_i = Slice(dy, axis, offset_i, axisDim_i)` — 每个输入的梯度是 dy 沿 axis 的切片 |
| Slice | `dx = Concat([zeroBefore, dy, zeroAfter], axis)` — 零填充恢复原始 shape |
| CallNode | 递归对 callee 做 autograd，通过 callee backward 传播 |
| CondNode | 分两路：各自对 thenBranch / elseBranch 做 autograd；选中分支的 backward 执行，另一分支梯度为零 |
| WhileNode | BPTT：构建反向 WhileNode，每次迭代从 TapeStack pop carry 值并调用 body backward；variable 梯度跨迭代累积 |
| Less/Greater/Equal/LogicalNegation | 不可微，不传播 |

**待实现**：`Power`, `Maximum`, `Minimum` 的梯度

### InlinePass

内联 Pass，将 CallNode 展开为其 callee 子图的内容。消除函数调用开销并暴露跨子图优化机会。

**运行顺序**: `AutogradPass → InlinePass → ConstFoldPass → FusionPass`

**算法**：迭代 fixpoint，反复扫描所有原有子图，直到没有 CallNode 被内联。

**处理策略**：
- `CallNode`：内联其 callee，将 ParamRefNode 替换为调用者传入的实参
- `CondNode` 的分支子图：不内联（运行时才确定分支）
- `FusedOpNode` 的 body 子图：不内联（body 保持独立以利于 AOT 代码生成）
- 以上子图内部的 CallNode 在 fixpoint 的其他迭代中被独立处理

**ParamRefNode 替换**：通过 `paramOutputMap` 保存完整的 `NodeOutput`（含 port 信息），正确处理多输出节点的非 0 port 情况。

### ConstFoldPass

常量折叠 Pass，在编译期评估所有输入均为常量的节点，并消除恒等操作，减少运行时计算量。

**三阶段算法**：

1. **常量传播 + 求值**：标记 `isConst` 属性，对全常量输入的 `UnaryOp/BinaryOp/Cast/ReduceOp/ReshapeNode/ConcatNode/SliceNode` 用 `DeviceTraits<CPU>` 在编译期求值，替换为 `ConstantNode`
2. **恒等消除**：识别 `x+0→x`, `x*1→x`, `x*0→0`, `x-0→x`, `Negate(Negate(x))→x` 等模式。仅当非常量操作数的 shape 与输出 shape 相同时才消除（避免广播语义改变）
3. **死节点消除 + 子图重建**：从 Results 反向标记可达节点，仅复制活节点到新子图

**不折叠的节点**：`CallNode`（内联后再折叠）、`CondNode`、`FusedOpNode`、`SaveActivationNode`（有副作用）、`LoadActivationNode`（依赖运行时状态）、`VariableRefNode`（训练期间变化）

### FusionPass

算子融合 Pass，在 AutogradPass 之后运行，独立处理所有子图。将匹配到的融合模式替换为 `FusedOpNode`。

**运行顺序**: `AutogradPass → InlinePass → ConstFoldPass → FusionPass`

**当前支持的融合模式**：

| FusionPattern | 描述 | 匹配条件 |
|---|---|---|
| `MatMulBiasAdd` | `y = MatMul(a, b) + c` | MatMul 输出仅一个消费者（Add），Add 另一操作数为 bias |
| `ElementWiseChain` | 2+ 逐元素操作链 | 每个中间结果仅一个消费者，且均为逐元素操作 |

**流程**：
1. **Consumer 分析**：统计每个 `NodeOutput` 的消费者数量和正向消费者索引
2. **模式检测**：按优先级（MatMulBiasAdd > ElementWiseChain）匹配，已融合节点不重叠
3. **Body 子图构建**：将融合区域提取为独立子图（参数 = 外部输入，结果 = 最终输出）
4. **子图重写**：用 `FusedOpNode{pattern, bodyId, args}` 替换融合区域

**Interpreter 支持**：`FusedOpNode` 的 Execute 直接代理到 `RunSubgraph(body)`，语义完全等价。AOT 编译器可据 `FusionPattern` 生成优化内核。

---

## Runtime

### Interpreter\<D\>

逐节点解释执行 Graph，用于**调试和功能验证**。不是最终的执行后端。

- 维护 slot 表 `vector<vector<Tensor<D>>>` 按 NodeId 存储各节点输出
- `RunForward` 自动初始化 ActivationStore 和 TapeStore
- `RunBackward` 复用已填充的 ActivationStore / TapeStore
- `RunSubgraph` 是通用入口，支持执行任意子图（包括 WhileNode 的 condBranch/bodyBranch）

---

## Layer 系统

高级操作通过工厂函数构建 Subgraph，而非独立的 Layer 类。

```cpp
// Layer/ReLU.h
SubgraphId BuildReLU(Graph& graph, DataType dtype, ShapeView shape);
// ReLU(x) = max(x, 0)
// 组合：Max + Constant，共 2 个节点（含常量 0）
```

---

## 设计偏好

1. **编译器前端风格**：Graph 是静态前端表示，不持有执行逻辑；Pass 做变换；后端做执行
2. **最小原语集**：BinaryOp/UnaryOp 保持精简，复杂操作通过 Subgraph 组合
3. **先讨论后实现**：涉及架构的改动，先充分讨论再动手
4. **AOT 优先**：目标是 AOT 编译，Interpreter 仅用于调试
5. **C++26 特性**：大量使用 Reflection (`<meta>`)、Concepts、`deducing this`、`constexpr`/`consteval`、`template for`
6. **静态保存激活值**：Autograd 采用保存中间值而非重计算；循环体使用 TapeStack（动态栈存储），循环体外使用 ActivationSlot（静态槽位）
7. **控制流**：支持 `CondNode`（结构化条件分支）和 `WhileNode`（函数式循环，BPTT 反向传播）

---

## 文件结构

```
src/
├── LiteNN.h                      // 聚合 include 入口
├── LiteNN.ixx                    // C++26 module 导出（TODO: 完善）
└── LiteNN/
    ├── Misc.h                    // 工具：EnumDispatch, ShapeView, EnumToString
    ├── Operators.h               // UnaryOp, BinaryOp, DataType, FusionPattern, OpTraits
    ├── Device.h                  // Device concept, CPU, PolymorphicDevice
    ├── Device.cpp                // PolymorphicDevice 实现
    ├── Tensor.h                  // Tensor<D> 模板，所有张量操作
    ├── Graph.h                   // Graph, Subgraph, NodeVariant 等核心类型
    ├── Pass.h                    // Pass 基类
    ├── Layer/
    │   └── ReLU.h               // BuildReLU 子图工厂
    ├── Pass/
    │   ├── AutogradPass.h       // 自动微分 Pass
    │   ├── InlinePass.h         // 内联 Pass（展开 CallNode）
    │   ├── ConstFoldPass.h      // 常量折叠 Pass
    │   └── FusionPass.h         // 算子融合 Pass
    └── Runtime/
        └── Interpreter.h        // 解释执行器

src/LiteNN/Compiler/             // AOT 编译器后端（LITENN_ENABLE_MLIR=ON 启用）
├── CMakeLists.txt               // TableGen + LiteNNCompiler 库
├── CompiledModule.h/cpp         // Compiler<CPU> + CompiledModule<CPU>，object image + rodata/instruction loader
├── Dialect/
│   ├── LiteNNDialect.td         // Dialect + Enum Attrs
│   ├── LiteNNOps.td             // 全部 Op 定义
│   ├── LiteNNDialect.h/cpp      // Dialect 注册
│   └── LiteNNOps.h/cpp          // Op 自定义实现
├── Pass/
│   ├── LowerLiteNNPass.h/cpp    // litenn dialect → linalg/arith/scf/memref
│   ├── BufferizationPipeline.h/cpp  // tensor → memref（one-shot-bufferize）
│   └── LLVMCodegenPipeline.h/cpp    // memref/linalg → LLVM dialect → llvm::Module
└── Translation/
    ├── GraphToMLIR.h            // 公开 API
    └── GraphToMLIR.cpp          // Graph → MLIR 翻译器

tests/
├── TensorTest.cpp               // Tensor 基础操作测试
├── InterpreterTest.cpp          // Graph/Interpreter/AutogradPass 测试
├── InlinePassTest.cpp           // InlinePass 内联测试
├── ConstFoldPassTest.cpp        // ConstFoldPass 常量折叠测试
├── FusionPassTest.cpp           // FusionPass 融合模式测试
├── ConcatSliceTest.cpp          // ConcatNode/SliceNode 前向/梯度/Pass 集成测试
├── WhileNodeTest.cpp            // WhileNode 前向/自动微分测试
├── CompilerTest.cpp             // Graph→MLIR 翻译 + verify 测试（需 LITENN_ENABLE_MLIR=ON）
├── LoweringPassTest.cpp         // LowerLiteNNPass 测试（需 LITENN_ENABLE_MLIR=ON）
├── BufferizationPassTest.cpp    // Bufferization 流水线测试（需 LITENN_ENABLE_MLIR=ON）
├── LLVMCodegenPassTest.cpp      // LLVM IR 生成测试（需 LITENN_ENABLE_MLIR=ON）
└── CompiledModuleTest.cpp       // Step5 object image / rodata+instruction 加载测试（需 LITENN_ENABLE_MLIR=ON）
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

- [x] **InlinePass**：展开 CallNode，为跨子图优化铺路；迭代 fixpoint 算法；通过 `paramOutputMap` 正确处理多输出参数替换；不内联 CondNode 分支和 FusedOpNode body
- [x] **ConstFoldPass**：常量折叠；三阶段算法（常量传播 + 恒等消除 + 死节点消除）；CPU 设备编译期求值；支持 `x+0`, `x*1`, `x*0`, `x-0`, `Negate(Negate(x))` 恒等消除；广播 shape 不兼容时保留原操作
- [x] **FusionPass**：算子融合 Pass，支持 `MatMulBiasAdd` 和 `ElementWiseChain` 融合模式；新增 `FusedOpNode` 持有 `FusionPattern` 标签 + body 子图；Interpreter 通过执行 body 保持语义等价；在 AutogradPass 之后运行
- [x] **Concat/SliceNode**：沿指定轴拼接/切片张量；Device concept 新增 `DoConcatOp`/`DoSliceOp`（outer/axisDim/inner 分解）；AutogradPass 支持互反梯度（Concat backward = Slice，Slice backward = zero-pad + Concat）；ConstFoldPass/InlinePass/FusionPass 均已支持
- [x] **WhileNode + TapeStack**：函数式循环节点（`while condBranch(carry): carry = bodyBranch(carry)`）；TapeStack（`TapeSaveActivationNode` / `TapeLoadActivationNode`）用于循环体 BPTT；augmented forward 嵌入迭代计数 counter carry；AutogradPass 生成反向 WhileNode，每次迭代 TapeLoad carry + 调用 body backward + 累积 variable 梯度；InlinePass/ConstFoldPass/FusionPass 均已支持
- [x] **Variable gradient accumulation across CallNode**：`EmitCallGrad` 和 `EmitCondGrad` 现在正确提取 callee backward 的 variable 梯度输出并传播到父级；`ProcessSubgraph` 合并直接 VariableRefNode 梯度与 callee 传播的梯度；CondNode 分支 variable 集合不同时通过 wrapper 子图补零对齐

### 长期

- **AOT Compiler\<D\>**：将 Graph 编译为设备特定可执行代码（`CompiledModule<D>`）
  - [x] **Step 1 — MLIR Dialect + Graph→MLIR 翻译器**：定义 `litenn` MLIR Dialect（ODS TableGen，全部 Op），实现 `translateGraphToMLIR()`；SSA 化激活值（Save/Load passthrough）；支持 CondNode/WhileNode region 内联；CompilerTest 4 项测试通过
  - [x] **Step 2 — Lowering Passes**：`litenn` Dialect → 标准 MLIR Dialect（`linalg`/`arith`/`scf`）；tensor 操作映射到 `linalg.generic`/`arith.*`；CondOp/WhileOp → `scf.if`/`scf.while`
  - [x] **Step 3 — Bufferization**：tensor → memref（`one-shot-bufferize`）；内存分配策略
  - [x] **Step 4 — LLVM IR 生成**：convert-linalg-to-loops + convert-scf-to-cf + arith/math/index/memref/func/cf-to-LLVM + reconcile-casts → `translateModuleToLLVMIR()`；LLVMCodegenPassTest 3 项测试通过
  - [x] **Step 5 — CompiledModule\<CPU\> 接口**：`Compiler<CPU>::Compile(graph) → CompiledModule<CPU>`；生成 native object instruction bytes + rodata 元数据；`CompiledModule<CPU>::Load({rodata, instructions})` 可从静态库/动态库暴露的地址恢复执行；`CompiledModule<CPU>::Run(inputs) → outputs` 与 Interpreter 前向接口对齐；`WriteObjectFile()` 生成 carrier object（导出 `<prefix>_rodata`, `<prefix>_rodata_size`, `<prefix>_instructions`, `<prefix>_instructions_size`）
- [ ] **JIT / Template Interpreter**：`Runtime/` 目录预留了扩展空间
- [ ] **更多 Device**：CUDA、Metal 等 GPU 后端
- [ ] **LiteNN.ixx module 导出**：完善 C++26 module 接口
- [ ] **Interprocedural Specialization（封闭世界跨子图优化）**：Graph 本身是封闭世界，除前向/后向入口外的所有子图只需处理内部调用。对不适合 inline 的子图，可对所有 call site 的参数属性求交集，利用调用中共同的特征进行特化优化（部分求值、分支消除、更激进的常量折叠等）。类似 Dart type flow graph 的思路。预期最大收益点：AutogradPass 生成图的结构性冗余消除（零梯度路径剪枝、dead branch elimination）、WhileNode 引入后循环体的跨子图优化。建议在 InlinePass + ConstFoldPass 基础设施就绪、WhileNode 实现后评估 ROI
