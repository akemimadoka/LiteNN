# LiteNN 架构文档

## 项目概述

LiteNN 是一个使用 **C++26** 实现的神经网络库，采用**编译器前端**风格进行设计。核心思想是将神经网络计算表示为一个静态计算图（Graph），通过 Pass 系统对其进行变换和优化，最终由后端执行。目前实现了基于解释执行的运行时（Interpreter）和 CPU AOT 编译后端（MLIR/LLVM）。

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
Compiler<CPU>              ← 当前实现（MLIR/LLVM AOT 编译）
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
- 可选保存 forward input/output 名称；shape/dtype 签名始终从 forward subgraph 推导，避免 metadata 漂移

```
Graph
├── subgraphs_: deque<Subgraph>        // 不会导致已有引用失效
├── variables_: vector<shared_ptr<Variable>>
├── activationSlots_: vector<ActivationSlot>
├── tapeSlots_: vector<TapeSlot>       // 循环体栈式激活值槽位
├── inputNames_: vector<string>        // 可选公开输入名
├── outputNames_: vector<string>       // 可选公开输出名
├── forward_: SubgraphId
└── backward_: optional<SubgraphId>   // AutogradPass 后填充
```

公开签名 API：

- `SetInputNames` / `SetOutputNames` / `SetInputName` / `SetOutputName`
- `InputSignature()` / `OutputSignature()` 返回 `NamedTensorSpec{name, dtype, shape}`
- 未设置名称时使用稳定默认名：`input0`, `input1`, `output0`, ...
- `FindInput(name)` / `FindOutput(name)` 用于把命名绑定映射回位置索引

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

**BinaryOp**：`Add`, `Subtract`, `Multiply`（Hadamard 积）, `Divide`, `MatMul`, `Pow`, `Max`, `Min`, `Less`, `Greater`, `Equal`

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
| Pow | `da=dy*b*a^b/a, db=dy*a^b*log(a)` |
| Max | `da=dy*(a>=b), db=dy*(a<b)` |
| Min | `da=dy*(a<=b), db=dy*(a>b)` |
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

### ForwardOnlyPass

`ExtractForwardOnlyGraph(graph)` 从训练图提取生产推理用 forward-only Graph。

- 只克隆 forward 可达的 Subgraph，丢弃 backward 入口和不可达训练辅助子图
- `SaveActivationNode` / `TapeSaveActivationNode` 被视为透传节点并剥离
- `LoadActivationNode` / `TapeLoadActivationNode` 出现在 forward 路径时会报错，避免把反向专用状态带入推理图
- 复用原 Graph 的 Variable `shared_ptr`，因此可直接使用训练后的权重；需要独立权重副本时仍可通过 ModelIO 或手动拷贝实现
- 保留公开 input/output 名称

---

## Runtime

### Interpreter\<D\>

逐节点解释执行 Graph，用于**调试和功能验证**。不是最终的执行后端。

- 维护 slot 表 `vector<vector<Tensor<D>>>` 按 NodeId 存储各节点输出
- `RunForward` 自动初始化 ActivationStore 和 TapeStore
- `RunBackward` 复用已填充的 ActivationStore / TapeStore
- `RunSubgraph` 是通用入口，支持执行任意子图（包括 WhileNode 的 condBranch/bodyBranch）

**线程安全边界**：
- `Graph` 在构建和 Pass 阶段是可变对象，不支持并发修改；完成 Pass 后应视为只读 IR。
- 只读 `Graph` 可被多个线程共享执行，但每个线程必须使用独立的 `Interpreter<D>` 实例，因为 `Interpreter` 持有 `ActivationStore` / `TapeStore` 运行状态。
- 同一个 `Interpreter<D>` 实例不支持并发 `RunForward` / `RunBackward`。训练路径中 `RunBackward` 依赖同实例最近一次 `RunForward` 填充的 activation/tape。
- `CompiledModule<CPU>::Run` 不修改 module 状态，输入/输出 buffer 由调用方/本次调用拥有，因此可在同一个已加载 module 上并发调用；`Compile` / `Load` / 析构仍应作为生命周期操作与并发 `Run` 分离。

### CompiledModule\<CPU\>

`CompiledModule<CPU>` 是当前 AOT 运行时封装。

- `Compiler<CPU>::Compile(graph)` 生成 native object instruction bytes 和 rodata
- `CompiledModule<CPU>::Load({rodata, instructions})` 会复制 image 数据并创建 JIT loader；调用方传入的 rodata/instruction 地址可在 `Load` 返回后释放
- rodata 当前包含 magic、format version、pointer size、endianness、target triple、命名 input/output specs
- `InputSpecs()` / `OutputSpecs()` 返回 `CompiledTensorSpec{dtype, shape, name}`
- `FindInput(name)` / `FindOutput(name)` 支持命名签名查询
- `WriteObjectFile()` 生成 carrier object，导出 `<prefix>_rodata`, `<prefix>_rodata_size`, `<prefix>_instructions`, `<prefix>_instructions_size`
- `RunInto(inputs, outputs)` 可复用调用方提供的输出 Tensor，适合生产推理循环中减少外层输出分配；`Run(inputs)` 仍保留为便捷接口

---

## CPU AOT 性能优化路径

当前 CPU AOT 的性能目标分两层看待：一层是与 PyTorch 默认配置对比的端到端吞吐，另一层是与 PyTorch 单线程对比的 kernel/codegen 质量。PyTorch 默认会使用多线程 BLAS/oneDNN；LiteNN 当前 AOT 生成的是单线程 native loop，因此性能分析应先固定线程数，再判断是否进入多线程后端工作。

已落地的优化：

- **Release 构建基线**：benchmark 必须使用 `CMAKE_BUILD_TYPE=Release` 的 MLIR/AOT 构建；Debug 构建会把差距夸大到不可参考。
- **目标相关 LLVM codegen**：object 生成阶段使用 host CPU/features、`CodeGenOptLevel::Aggressive`，并在发射对象文件前对 LLVM module 跑 O3 pipeline。
- **MLIR fast-math 标注**：在 lowered arith op 上标注 `reassoc|contract`，允许 LLVM 对浮点累加做更积极的向量化和 FMA 合并。
- **MatMul loop order**：`BinaryOp::MatMul` lowering 不再直接依赖默认 `linalg.matmul` loop order，而是生成 `M,K,N` 的 `linalg.generic`，让 innermost `N` 连续访问 RHS 和输出行。
- **MatMulBiasAdd AOT 融合**：`FusionPattern::MatMulBiasAdd` 在 LowerLiteNNPass 中直接 lowering 为 bias 初始化 + `M,K,N` matmul 累加，避免先 matmul 再单独执行一次 Add 输出遍历。
- **函数边界布局收紧**：One-shot bufferize 的 function boundary/unknown type conversion 使用 identity layout map，减少动态 layout/stride 对 LLVM 优化的干扰。
- **Interpreter reference kernel 改善**：CPU reference MatMul 改为 `i,k,j` 累加并预零输出，解释器路径也具备更合理的 cache/SIMD 访问形态，但它仍不是性能主路径。
- **输出复用接口**：`CompiledModule::RunInto` 支持调用方复用输出 Tensor；当前主要用于生产接口完整性，端到端 benchmark 仍应同时观察 wrapper 校验、输出 copy 和 generated kernel 的占比。

当前 profile 结论：

- 大 batch MLP 的主要差距来自 dense MatMul kernel。与 PyTorch 默认 16 线程相比，LiteNN 单线程 AOT 会显著落后；与 PyTorch 单线程相比，差距主要来自 BLAS 级别的 blocking/packing、寄存器 tiling、prefetch 和更成熟的 micro-kernel。
- 小 batch 下 fixed overhead（JIT wrapper、输入输出校验、临时 allocation/copy）更明显；大 batch 下 GEMM 计算占主导。
- `MatMul + bias` 的独立 Add pass 在宽输出上是可见但次要的开销，已通过 AOT 融合减少一次输出遍历。

后续优化顺序：

1. **单线程 GEMM 质量**：为常见 `f32` MatMul 引入 tiled/packed kernel（手写 micro-kernel、MLIR tiling/packing，或调用 BLAS/oneDNN 的单线程路径），目标先把 PyTorch 单线程差距压到 1.2-1.5x 内。
2. **Destination-passing / buffer reuse**：让 AOT 子图直接写入 caller output 和规划好的临时 buffer，减少内部 tensor allocation、最终 output copy 和 wrapper 开销。
3. **多线程 CPU backend**：在单线程 kernel 质量稳定后，再接入 OpenMP/threadpool/oneDNN 多线程，否则多线程只会掩盖单核效率问题。
4. **profile 工具化**：补齐 Graph/MLIR/LLVM dump 和 per-op/per-shape benchmark，持续记录 Linear/MLP/CNN/Reduce 的性能回归基线。

---

## Training API

`LiteNN::Training::CPUTrainer<OptimizerT>` 是当前 CPU 训练路径的轻量封装，负责把 Graph、Interpreter、AutogradPass、loss gradient 和 Optimizer 串起来。

- 构造时可自动为缺失 backward 的 Graph 运行 `AutogradPass`，随后调用 `ValidateGraph`
- `Forward(inputs)` 只执行前向
- `Step(inputs, outputGradients)` 执行 forward → backward → store variable gradients → optimizer step
- `StepSoftmaxCrossEntropy(inputs, targetClass)` 为单输出 logits 图提供常见分类训练入口
- `StepSoftmaxCrossEntropyBatch(inputs, targetClasses)` 支持 `[batch, classes]` logits，loss 和 logits gradient 均按 batch 平均
- `TrainerOptions` 可控制是否自动构建 backward、是否写回 `Variable::Grad()`、是否在 backward 前清零梯度

优化器公共工具位于 `LiteNN::Optimizer`：

- `ZeroGradients(Graph&)` 清空所有 Variable 的 `grad_`
- `StoreVariableGradients(Graph&, backwardResults, inputGradientCount)` 将 Autograd 结果中的 variable gradients 写回 `Variable::Grad()`
- `InferInputGradientCount(Graph&)` 根据 backward 参数数量和 forward 输出数量推导输入梯度数量

当前 Trainer 仍是 CPU + eager training API；参数组、学习率调度、checkpoint 训练循环和吞吐优化留在后续 P1/P2。

---

## Model Serialization

`LiteNN::Serialization::SaveModel/LoadModel` 提供 Graph + Variable 权重的二进制保存/加载能力。

- 文件包含 magic、format version、forward/backward 入口、公开 input/output 名称、Variable data、ActivationSlot/TapeSlot、所有 Subgraph 和节点 payload
- Variable 只保存 `Data()`，加载时由 `Variable::Create` 重新初始化 `Grad()` 为同 shape/dtype/device 的零张量
- 当前格式为内部 binary format，适合 checkpoint 与本库内推理模型导出；跨版本迁移策略仍需后续完善
- 加载后会调用 `ValidateGraph`，保证损坏或不兼容模型尽早失败

---

## Layer 系统

高级操作通过工厂函数构建 Subgraph，而非独立的 Layer 类。

```cpp
// Layer/Activation.h
SubgraphId BuildReLU(Graph& graph, DataType dtype, ShapeView shape);
// ReLU(x) = max(x, 0)

NodeOutput AddGELU(Subgraph& sg, NodeOutput input);
SubgraphId BuildGELU(Graph& graph, DataType dtype, ShapeView shape);
// GELU(x) = x * 0.5 * (1 + tanh(0.7978… * (x + 0.044715 * x³)))  [tanh 近似]

NodeOutput AddELU(Subgraph& sg, NodeOutput input, double alpha = 1.0);
SubgraphId BuildELU(Graph& graph, DataType dtype, ShapeView shape, double alpha = 1.0);
// ELU(x) = max(x,0) + min(alpha*(exp(x)-1), 0)

// Layer/Softmax.h
NodeOutput AddSoftmax(Subgraph& sg, NodeOutput input, std::size_t axis = 1);
SubgraphId BuildSoftmax(Graph& graph, DataType dtype, ShapeView shape, std::size_t axis = 1);
// softmax 沿 axis 归一化，采用 max-shift 保证数值稳定

// Layer/LayerNorm.h
struct LayerNormLayer { SubgraphId subgraph; VariableIdx gammaVariable, betaVariable;
                        std::size_t featureSize; DataType dtype; double eps; };
LayerNormLayer CreateLayerNorm(Graph& graph, std::size_t featureSize, DataType dtype, double eps = 1e-5);
NodeOutput AddLayerNorm(Subgraph& sg, NodeOutput input, const LayerNormLayer& layer);
// LayerNorm 归一化最后一维（featureSize），含 learnable gamma/beta，输入形状 [batch, featureSize]
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
    ├── Initializer/
    │   └── Initializer.h         // Xavier/He/Normal/Uniform/Zero 初始化工具
    ├── Layer/
    │   ├── Layer.h              // 常用层聚合入口
    │   ├── Linear.h             // LinearLayer + BuildLinear
    │   ├── Activation.h         // BuildReLU/BuildSigmoid/BuildTanh/AddGELU/BuildGELU/AddELU/BuildELU
    │   ├── Softmax.h            // AddSoftmax/BuildSoftmax（max-shift 数值稳定）
    │   ├── LayerNorm.h          // LayerNormLayer/CreateLayerNorm/AddLayerNorm
    │   └── Flatten.h            // BuildFlatten
    ├── Optimizer/
    │   ├── Optimizer.h          // 优化器接口
    │   ├── OptimizerUtils.h     // 梯度校验、清零、写回等训练公共工具
    │   ├── SGD.h                // SGD + momentum/weight_decay
    │   ├── Adam.h               // Adam
    │   └── Loss.h               // SoftmaxCrossEntropyWithLogits
    ├── Pass/
    │   ├── AutogradPass.h       // 自动微分 Pass
    │   ├── InlinePass.h         // 内联 Pass（展开 CallNode）
    │   ├── ConstFoldPass.h      // 常量折叠 Pass
    │   ├── FusionPass.h         // 算子融合 Pass
    │   └── ForwardOnlyPass.h    // 训练图 → forward-only 推理图提取
    ├── Validation/
    │   └── GraphValidator.h     // Graph 静态校验与诊断
    ├── Serialization/
    │   └── ModelIO.h            // Graph + Variable 权重保存/加载
    ├── Training/
    │   └── Trainer.h            // CPUTrainer 训练 API
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
├── GraphValidatorTest.cpp       // Graph 静态验证器与运行时输入诊断测试
├── AutogradRegressionTest.cpp   // Autograd shape/order 契约回归测试
├── ThreadSafetyTest.cpp         // 只读 Graph + 独立 Interpreter 并发 smoke 测试
├── TrainingTest.cpp             // CPUTrainer、loss step、Variable::Grad 写回测试
├── ModelIOTest.cpp              // SaveModel/LoadModel forward/backward/variable 回归测试
├── SignatureTest.cpp            // Graph 命名 input/output 签名测试
├── ForwardOnlyPassTest.cpp      // 从训练图提取 forward-only 推理图测试
├── InlinePassTest.cpp           // InlinePass 内联测试
├── ConstFoldPassTest.cpp        // ConstFoldPass 常量折叠测试
├── FusionPassTest.cpp           // FusionPass 融合模式测试
├── ConcatSliceTest.cpp          // ConcatNode/SliceNode 前向/梯度/Pass 集成测试
├── WhileNodeTest.cpp            // WhileNode 前向/自动微分测试
├── CompilerTest.cpp             // Graph→MLIR 翻译 + verify 测试（需 LITENN_ENABLE_MLIR=ON）
├── LoweringPassTest.cpp         // LowerLiteNNPass 测试（需 LITENN_ENABLE_MLIR=ON）
├── BufferizationPassTest.cpp    // Bufferization 流水线测试（需 LITENN_ENABLE_MLIR=ON）
├── LLVMCodegenPassTest.cpp      // LLVM IR 生成测试（需 LITENN_ENABLE_MLIR=ON）
└── CompiledModuleTest.cpp       // Step5 object image / rodata+instruction 加载测试（需 LITENN_ENABLE_MLIR=ON）└── LayerTest.cpp                // Softmax/GELU/ELU/LayerNorm 层正确性测试```

---

## 生产化差距与路线图

LiteNN 当前已经具备静态 Graph、Pass 系统、Autograd、Interpreter、基础融合、控制流、MLIR/LLVM AOT 和 CompiledModule 加载执行能力。整体定位应视为“编译器式 NN runtime 原型进入可验证阶段”，但距离可稳定接入生产项目，还需要补齐可靠性、模型生态、AOT ABI、性能和工程发布能力。

**已关闭的关键风险**：
- [x] **AOT/MLIR 测试退出 crash**：此前 `BufferizationPassTest.WhileNode`、`CompiledModuleTest.RunsAfterLoadingFromRodataAndInstructionAddresses`、`CompiledModuleTest.WritesCarrierObjectFile` 在进程退出阶段触发 MLIR TLS 析构 crash；该问题已修复，不再作为生产化阻塞项跟踪。

### P0：正确性与稳定性

- [x] **Graph 静态验证器**：新增 `LiteNN::Validation::ValidateGraph`，在 Pass 入口、Interpreter 入口和 Graph→MLIR 翻译入口检查 shape/dtype、参数数量、结果数量、NodeOutput 合法性、子图调用签名、VariableRef 下标、控制流分支输入输出一致性。目标是在执行或编译前给出明确错误，而不是在后端或运行期崩溃。
- [x] **Autograd 完整性回归**：系统覆盖广播梯度、变量梯度顺序、多变量、多输出、共享 callee、CondNode/WhileNode、Concat/Slice、Reduce、FusedOpNode 相关组合。`AutogradRegressionTest` 覆盖广播参数梯度 reduce 回原 shape、多输出 grad_output 累加到共享输入，以及 `backward results = [grad_inputs..., grad_variables...]` 中 variable gradient 按 variableIndex 排序的约定；既有 `InterpreterTest`、`WhileNodeTest`、`ConcatSliceTest`、`FusionPassTest` 覆盖共享 callee、嵌套 call、Cond/While、Concat/Slice、Reduce 和 Autograd→Fusion 流水线。
- [x] **错误信息与诊断上下文**：公共执行/编译入口在失败时携带可定位上下文。GraphValidator 错误包含 subgraphId/nodeId/node kind 与 expected/got dtype+shape；Interpreter 节点执行异常补充 `subgraphId/nodeId/node kind`；Interpreter 和 CompiledModule 输入校验补充 expected/got dtype+shape。Tensor/Device 内部错误仍保持底层异常，由公共入口包裹或由 GraphValidator 提前拦截。
- [x] **线程安全边界**：明确 Graph、Interpreter、CompiledModule、MLIR context/JIT 初始化和销毁的线程模型。只读 Graph 可跨线程共享；每个线程必须使用独立 `Interpreter<D>`；同一个 `Interpreter<D>` 不支持并发 `RunForward` / `RunBackward`；`CompiledModule<CPU>::Run` 可在输入/输出 buffer 独立时并发调用。`ThreadSafetyTest` 覆盖只读 Graph + 独立 Interpreter，`CompiledModuleTest.ConcurrentRunUsesIndependentInputAndOutputBuffers` 覆盖 AOT Run 并发 smoke。
- [ ] **CI 矩阵**：覆盖 Windows/Linux、Debug/Release、`LITENN_ENABLE_MLIR=ON/OFF`、MinGW/Clang/MSVC（如支持）。所有测试需纳入可复现的持续集成。（暂时推迟）
- [ ] **内存安全验证**：引入 ASAN/UBSAN（或平台等价工具）、泄漏检查和长时间循环测试，覆盖 Tensor 视图、PolymorphicDevice、CompiledModule image 生命周期。（暂时推迟）

### P1：用户可用性

- [x] **模型保存/加载**：新增 `LiteNN::Serialization::SaveModel/LoadModel`，序列化 magic/version、forward/backward 入口、公开 input/output 名称、Variable data、ActivationSlot/TapeSlot、Subgraph、NodeVariant payload，并在加载后运行 `ValidateGraph`。当前已支持本库内部 checkpoint/推理模型 roundtrip；跨版本迁移策略仍需继续完善。
- [x] **输入输出命名与签名 API**：`Graph` 支持 `SetInputNames`/`SetOutputNames`、`InputSignature`/`OutputSignature`、`FindInput`/`FindOutput`；`CompiledModule<CPU>` rodata 保存命名 specs，并通过 `InputSpecs`/`OutputSpecs`/`FindInput`/`FindOutput` 查询。当前绑定执行仍按位置传参，命名 binding helper 可后续补。
- [x] **训练 API**：新增 `LiteNN::Training::CPUTrainer<OptimizerT>`，封装 forward、backward、loss gradient、`Variable::Grad()` 写回、梯度清零和 optimizer step；新增 `Optimizer::ZeroGradients`、`StoreVariableGradients`、`InferInputGradientCount` 等公共工具。参数组、学习率调度、epoch/batch loop 仍在后续 P1/P2 跟踪。
- [x] **Batch 训练与推理**：新增 `SoftmaxCrossEntropyWithLogitsBatch` 和 `CPUTrainer::StepSoftmaxCrossEntropyBatch`，支持 `[batch, classes]` logits 的平均 loss/gradient；Graph/Interpreter/CompiledModule 可通过 batch-shaped tensor 签名进行 batch 推理。吞吐优化和 MNIST mini-batch 示例仍留到 P2/示例扩展。
- [x] **常用算子覆盖（部分）**：已新增 `Softmax`（带 max-shift 数值稳定）、`GELU`（tanh 近似）、`ELU`（带 alpha 参数）、`LayerNorm`（learnable gamma/beta，2D 输入）层构建工具，通过 `Layer::AddSoftmax`/`BuildSoftmax`、`AddGELU`/`BuildGELU`、`AddELU`/`BuildELU`、`CreateLayerNorm`/`AddLayerNorm` 提供访问。Conv2D、Pooling、BatchNorm、Embedding、Gather/Scatter、Pad 仍待补充。
- [x] **示例体系（部分）**：MNIST interpreter 示例扩展了 MLP 路径（`--hidden-size N`），支持 `--save` / `--load` 在训练后保存权重或直接加载推理模型；end-to-end 演示完整 train → save → load → evaluate 工作流。AOT 静态库/动态库加载示例和训练 checkpoint 循环仍待补充。

### P1：AOT 产品化

- [x] **CompiledModule image ABI 版本化**：rodata 包含 magic、version、pointer size、endianness、target triple 和命名 input/output specs；`Load` 会进行兼容性检查并拒绝 ABI 不匹配 image。instruction 仍是 native object bytes，依赖当前 object loader。
- [ ] **静态库/动态库加载示例**：除了内存中的 `compiled.Image()`，需要展示 carrier object 链接进静态库/动态库后，通过导出符号地址构造 `CompiledModuleImage` 并执行。
- [x] **Forward-only inference graph 提取**：新增 `ExtractForwardOnlyGraph` / `ForwardOnlyPass`，只保留 forward 可达子图，剥离 Save/TapeSave activation 节点，丢弃 backward 入口和训练辅助 slot。MNIST interpreter/AOT 示例改为训练后提取 forward-only 推理图。
- [ ] **对象文件与 JIT 路径分层**：JIT/MCJIT 更适合作为验证路径；生产部署应优先稳定 object/link/load 路径。需要清晰区分“编译期生成 object”和“运行时加载 image”的职责。
- [ ] **CompiledModule 生命周期**：明确 rodata/instruction 内存是否复制、何时可释放、Run 是否可并发、析构顺序是否依赖 MLIR/LLVM 全局状态。

### P2：性能工程

- [ ] **CPU kernel 后端优化**：当前 CPU 更接近 reference backend。MatMul/Conv/Reduce 等核心算子需要接入 BLAS、oneDNN、OpenBLAS 或手写 SIMD kernel。
- [ ] **内存规划器**：减少逐节点 Tensor 分配，支持 activation reuse、arena allocator、in-place 分析和临时 buffer 生命周期规划。
- [ ] **Pass benchmark 驱动**：FusionPass、ConstFoldPass、InlinePass 的收益需要通过 benchmark 验证，并建立性能回归基线。
- [ ] **AOT 优化策略**：围绕 linalg/LLVM pipeline 增加目标相关优化、vectorization、loop tiling、constant parameter folding、shape specialization。
- [ ] **性能基准套件**：和 NumPy/PyTorch/ONNX Runtime 在小模型、MLP、CNN、控制流图上的 correctness 和吞吐对比。

### P2：工程发布与维护

- [ ] **CMake install/export/package**：支持 `find_package(LiteNN)`、安装头文件/库、导出 targets、可选 MLIR 组件。
- [ ] **C++26 module 导出完善**：`LiteNN.ixx` 需要成为稳定入口或明确标记 experimental，避免 include/module 两套入口语义漂移。
- [ ] **Graph/MLIR dump 工具**：提供 Graph textual dump、Pass 前后 dump、MLIR dump、CompiledModule metadata dump，便于调试和 issue 复现。
- [ ] **版本与兼容策略**：建立 release 版本号、changelog、弃用策略、Graph/CompiledModule 序列化版本迁移规则。
- [ ] **文档质量**：补齐 API guide、设计约束、Pass 编写指南、Device 后端接入指南、AOT 部署指南和常见错误排查。

---

## TODO List

### 近期

- [x] **BinaryOp 补全**：添加 `Pow`, `Max`, `Min` 及其 Device 实现和梯度规则
- [x] **ReduceNode**：新节点类型，支持 `ReduceSum`/`ReduceMean`/`ReduceMax`（带 axis 参数）——实现标量 loss 的必要条件
- [x] **ReshapeNode**：新节点类型，改变 shape 不改变数据——CNN→FC 衔接必需
- [x] **CondNode 梯度**：AutogradPass 支持 `CondNode` 的反向传播
- [x] **Stride 支持**：Tensor 添加了元素级 stride 基础设施（`strides_` 成员、`ComputeContiguousStrides`、`Strides()`、`IsContiguous()`），当前所有 tensor 仍为连续布局，为未来 lazy transpose/slice view 打基础
- [x] **多 GPU 实例**：`PolymorphicDevice` 新增 `IsSameDevice` 接口，基于 `operator==` 进行实例级别比较；`CopyToDevice` 已改用实例级检查
- [x] **新增激活函数与归一化层**：`AddGELU`/`BuildGELU`（tanh 近似）、`AddELU`/`BuildELU`（带 alpha 参数）新增至 `Activation.h`；`AddSoftmax`/`BuildSoftmax`（max-shift 数值稳定，支持任意 axis）新增至 `Softmax.h`；`LayerNormLayer`/`CreateLayerNorm`/`AddLayerNorm`（learnable gamma/beta，2D 输入）新增至 `LayerNorm.h`；均已通过 `LayerTest.cpp` 13 项测试覆盖
- [x] **MNIST MLP 与保存/加载示例**：`mnist_common.h` 新增 `BuildTrainableMlpGraph`（Linear→ReLU→Linear MLP）、`SaveMnistModel`、`LoadMnistInferenceModel`；`interpreter.cpp` 支持 `--hidden-size`、`--save`、`--load` 三个 CLI 选项，演示完整 train → save → load → evaluate 工作流

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
