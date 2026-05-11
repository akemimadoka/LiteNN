# LiteNN Device 后端接入指南

## 目标

新增设备类型时，需要提供 `DeviceTraits<D>` 特化并满足 `Device` concept 的全部操作集。

## 最低实现面

- `Allocate` / `Deallocate`
- `ZeroFill`
- `CopyToCPU` / `CopyFromCPU`
- `ConvertTo`
- `DoUnaryOp` / `DoBinaryOp`
- `DoReduceOp` / `DoConcatOp` / `DoSliceOp`

## 语义要求

- shape/dtype 错误应尽量在公共入口或 validator 暴露，不要只依赖底层 UB。
- `CopyToDevice` / `CopyToCPU` 必须正确处理 `PolymorphicDevice` 与具体设备间的复制。
- 如果设备实例具有身份语义，需要正确定义 `operator==`，以支撑 `IsSameDevice`。

## 接入步骤

1. 先让基础 Tensor 分配/复制测试通过。
2. 再补 unary/binary/reduce/concat/slice。
3. 最后验证 interpreter、training、serialization、forward-only 与 AOT 边界。

## 最低验证清单

- `TensorTest`
- `InterpreterTest`
- `TrainingTest`（如果后端支持训练）
- 相关内存/线程安全回归

目前 CPU 是完整参考实现；新后端应先对齐 CPU 语义，再讨论性能优化。