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

## CUDA 后端当前状态

CUDA 通过 `-DLITENN_ENABLE_CUDA=ON` 启用，并依赖 CMake `CUDAToolkit` 包。第一版不要求 NVCC 编译 CUDA kernel，链接 `CUDA::cudart` 与 `CUDA::cublas`，因此实现范围先聚焦在 Device 语义和 AOT 边界：

- `CUDA{ .deviceIndex = 0 }` 表示一个具体 CUDA device，`operator==` 按 `deviceIndex` 比较，支撑 `PolymorphicDevice::IsSameDevice()`。
- `Allocate` / `Deallocate` / `ZeroFill` 使用 CUDA Runtime 的 `cudaMalloc` / `cudaFree` / `cudaMemset`。
- `CopyToCPU` / `CopyFromCPU` / `ConvertTo` 支持同 dtype 直接拷贝，不同 dtype 通过 CPU 临时缓冲转换。
- `DoBinaryOp(MatMul)` 对 `Float32/Float64` 二维矩阵使用 cuBLAS；其他 unary/binary/reduce/concat/slice 当前走 CPU fallback：先拷回 host，用 CPU reference op 求值，再拷回 CUDA。这样可以先打通 Tensor、Interpreter 与 PolymorphicDevice 语义，性能优化留给后续 kernel 化。
- `Compiler<CUDA>` / `CompiledModule<CUDA>` 当前是 AOT bridge：公开 API 接收 CUDA Tensor，内部复用 CPU AOT artifact/JIT，再将输出拷回 CUDA device。它验证了对象文件、rodata/instruction 地址加载和 CUDA Tensor 边界，但还不是原生 GPU codegen。`Backend()` 当前返回 `CPUNative`，表示 instructions 仍是 CPU native object。
- `CUDADeviceCount()` 和 `IsCUDADeviceAvailable()` 用于运行时探测；测试会在无 CUDA device 时跳过。

后续 CUDA 原生化建议按风险从低到高推进：elementwise/broadcast kernel、reduce kernel、stream/async copy、device allocator/cache、AOT CUDA codegen。

CUDA 原生 AOT 需要额外补齐：

- MLIR GPU/NVVM lowering 或独立 CUDA kernel 生成路径。
- PTX/cubin/fatbin 打包进 image，并在 rodata metadata 标识 backend/target。详细阶段拆分见 `CUDAAOTRoadmap.md`。
- CUDA Driver API module loading、kernel symbol lookup 与 launch ABI。
- stream、workspace allocator、临时 buffer 生命周期与异步错误处理。

## 最低验证清单

- `TensorTest`
- `InterpreterTest`
- `CUDADeviceTest`（`LITENN_ENABLE_CUDA=ON` 且有 CUDA device 时）
- `CompiledModuleCUDATest`（`LITENN_ENABLE_MLIR=ON` + `LITENN_ENABLE_CUDA=ON` 且有 CUDA device 时）
- `TrainingTest`（如果后端支持训练）
- 相关内存/线程安全回归

目前 CPU 是完整参考实现；新后端应先对齐 CPU 语义，再讨论性能优化。
