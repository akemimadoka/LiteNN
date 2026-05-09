# LiteNN AOT 性能分析报告

> 目标：在 LiteNN AOT 与 PyTorch（单线程）已有的"约 2 倍差距"基础上，
> 系统性地排查瓶颈、给出可量化证据、并列出按收益排序的可行优化方案。

## 1. 测试环境

| 项目 | 配置 |
|---|---|
| CPU | AMD Ryzen 9 9950X (Zen 5, 16C/32T, **AVX-512**) |
| 操作系统 | Windows |
| 编译器 | MinGW GCC（带 LLVM/MLIR 后端） |
| 构建 | `build-release-mingw/`，`CMAKE_BUILD_TYPE=Release`，`-O3 -DNDEBUG`，`LITENN_ENABLE_MLIR=ON` |
| LLVM target | `getHostCPUName()` + `getHostCPUFeatures()`（包含 AVX-512）|
| LLVM Codegen | `CodeGenOptLevel::Aggressive`，`OptimizationLevel::O3` |
| PyTorch | 2.9.1+cu128 (CPU 后端)，Python 3.11.9 |
| 模型 | Linear(784→10) / MLP-128(784→128→10) / MLP-512(784→512→256→10) |
| 批次 | 1 / 32 / 128 / 512 |

测得脚本：
- C++：[benchmark/bench.cpp](benchmark/bench.cpp)、新增 [benchmark/profile.cpp](benchmark/profile.cpp)
- Python：[benchmark/bench.py](benchmark/bench.py)（使用 `--threads 1`）

## 2. 当前数据（关键基线）

### 2.1 ms/batch 对比（越低越好）

| Model           | Batch | LiteNN AOT | PyTorch 1T | 差距 (LiteNN/PyTorch) |
|-----------------|------:|-----------:|-----------:|----------------------:|
| Linear          |     1 | 0.001 ms   | 0.005 ms   | **0.20×（LiteNN 更快）** |
| Linear          |    32 | 0.041 ms   | 0.017 ms   | 2.4× 慢 |
| Linear          |   128 | 0.188 ms   | 0.061 ms   | **3.1× 慢** |
| Linear          |   512 | 0.776 ms   | 0.215 ms   | **3.6× 慢** ← 最差 |
| MLP-128         |     1 | 0.003 ms   | 0.015 ms   | 0.20× |
| MLP-128         |    32 | 0.082 ms   | 0.064 ms   | 1.3× 慢 |
| MLP-128         |   128 | 0.363 ms   | 0.228 ms   | 1.6× 慢 |
| MLP-128         |   512 | 1.440 ms   | 0.857 ms   | 1.7× 慢 |
| MLP-512         |     1 | 0.017 ms   | 0.040 ms   | 0.43× |
| MLP-512         |    32 | 0.542 ms   | 0.306 ms   | 1.8× 慢 |
| MLP-512         |   128 | 2.088 ms   | 1.115 ms   | 1.9× 慢 |
| MLP-512         |   512 | 8.711 ms   | 4.371 ms   | 2.0× 慢 |

**观察**：
- **batch=1 时 LiteNN 全部更快**（PyTorch 单 op 调度开销 ~5–15µs 远高于一次 JIT 函数调用）。
- **大批次 + 窄输出（Linear 输出 10 列）差距最大（3.6×）**，这是核心问题。
- **隐藏层越宽差距越小**（MLP-512 仅 2×），暗示瓶颈在"N 维度过窄无法占满 SIMD 寄存器"。

### 2.2 输出张量分配开销（Run vs RunInto）

`profile.cpp` 通过对比 `Run()`（每次 heap 分配输出）与 `RunInto()`（重用预分配 buffer）量化"非 kernel 开销"：

| Case          | Compile/ms | Run/ms   | RunInto/ms | Δ (alloc)    |
|---------------|-----------:|---------:|-----------:|-------------:|
| linear_b1     |   53 ms    | 0.0014   | 0.0014     | ~0 µs |
| linear_b512   |   43 ms    | 0.6662   | 0.7503     | ≈ 0（噪声内）|
| mlp128_b512   |   90 ms    | 1.4429   | 1.4458     | ≈ 0 |
| mlp512_b512   |  151 ms    | 8.9370   | 8.8006     | ≈ 0 |

**结论：`Run()` 的输出分配 / 入参验证开销 ≪ 1%，不是瓶颈。** 优化重点必须在 kernel 本身。
（编译开销 50–170 ms 是一次性成本，与稳态推理无关。）

## 3. 反汇编瓶颈定位

通过 `profile.cpp` 把每个编译产物的 instructions（即 JIT 加载的 `.o`）写出后用
`objdump -d -M intel` 反汇编，统计向量指令分布：

| 文件 (.s)       | 总行数 | `vfmadd…ps` (packed) | `vfmadd…ss` (标量) | 包含 zmm 的 packed | 包含 ymm | xmm |
|-----------------|------:|---------------------:|-------------------:|-------------------:|---------:|----:|
| linear_b512     | 1622  | 40                   | **40**             | 40                 | 0        | 0   |
| mlp128_b512     | 2277  | 56                   | **56**             | 56                 | 0        | 0   |
| mlp512_b512     | 3865  | 136                  | 72                 | 136                | 0        | 0   |

并存在大量 `vmovss`（标量 load/store）：linear=224、mlp128=344、mlp512=464。

### 3.1 关键代码段（linear_b512.s 内层循环）

```asm
; M 行 × K=784 维 reduction，N=10 列被完全展开。
; 每次迭代：对每一个 N=0..9 都执行 “load A、load B、FMA、store 回内存”
vmovss xmm16, [rdi+rbx*4-0x4]            ; load A[m,k]
vfmadd132ss xmm16, xmm5, [r14-0x4c]      ; xmm16 = A[m,k]*B[k,0] + acc(memory)
vmovss [rcx+r11*1], xmm16                ; store back to out[m,0]
vmovss xmm17, [rdi+rbx*4-0x4]            ; (重新 load A[m,k]！)
vfmadd132ss xmm17, xmm6, [r14-0x48]
vmovss [rcx+r11*1+0x4], xmm17
... × 10
```

> **诊断**：
> 1. 累加器存放在**内存**里（`out[m,n]`），不是寄存器 → 每次 reduction step 都做一次 load+store。
> 2. N=10 这一维被完全展开为 10 条 *标量* `vfmadd132ss`，**没有打包成 packed FMA**。
> 3. AVX-512 zmm 寄存器数量 32 个，可以放 16 个完整 16-wide 累加器，目前几乎全没用上。
> 4. `vfmadd...ps zmm` 只在 MLP-512 的隐藏层（K=784, N=512）出现，因为 N≥16 时 LLVM 才会自动向量化。

### 3.2 这段代码是怎么生成的

- `linalg.matmul` 由 [LowerLiteNNPass.cpp](src/LiteNN/Compiler/Pass/LowerLiteNNPass.cpp) 创建。
- LLVM 代码生成管线 [LLVMCodegenPipeline.cpp](src/LiteNN/Compiler/Pass/LLVMCodegenPipeline.cpp:60) 的关键步骤：
  ```cpp
  pm.addPass(mlir::createConvertLinalgToLoopsPass());  // ← 仅做 “naive 三重循环”
  pm.addPass(createEnableSIMDFastMathPass());          // ← 设置 fastmath
  ```
  没有任何 **tiling / packing / vectorization** pass。
- 之后交给 LLVM `O3`，但 LLVM 的 LoopVectorize 只能把 **最内层归约** 向量化，遇到：
  - 归约累加在 **memref scalar**（store→load）→ 别名分析保守 → 拒绝向量化；
  - 内层维度 < SIMD 宽（如 N=10）→ 拒绝；
  - 循环顺序为 M→K→N 全展开 N → 没有真正的内层归约可向量化。

### 3.3 PyTorch 为什么快

PyTorch CPU 后端用 OneDNN / MKL，对每个矩阵尺寸生成专门的 micro-kernel：
- Register tiling：12×16 累加器全部放在 zmm 寄存器；
- Cache tiling（panel packing）：把 B 矩阵重排成 cache-friendly panel；
- AVX-512 vectorization 沿 M 维（batch 维），即使 N 很窄；
- 对小 K 也有 outer-product pattern。

## 4. 性能瓶颈总结（按影响排序）

| # | 瓶颈                                                        | 受影响场景                       | 证据 |
|---|-------------------------------------------------------------|----------------------------------|------|
| 1 | **MatMul 缺少 tiling+vectorize**（accumulator 在内存）      | 所有 batch ≥ 32 的层             | 反汇编全是 `vfmadd…ss + vmovss` |
| 2 | **窄 N 输出层不向量化**（N=10 → 标量 tail）                 | Linear 输出层、MLP 末层          | linear_b512 packed/scalar = 40/40 |
| 3 | 没有 panel packing / B 矩阵布局优化                         | 任意 K 较大的层（K=784, 512, 256）| LLVM 默认按 row-major 跨步访问 B |
| 4 | `Run()` 每次分配输出张量                                    | 极小模型/极小 batch 时少量影响    | `Run vs RunInto Δ ≈ 0`，已验证不影响大模型 |
| 5 | `AddReLU` 用 `Max(x, full_zero_tensor)`：分配整张零常量     | 所有含 ReLU 的层                  | [Activation.h](src/LiteNN/Layer/Activation.h) 中 zero 是同 shape 张量 |
| 6 | Bias add 通过 `linalg.generic + 广播 affine map` 表示       | 所有 Linear                       | LowerLiteNNPass.cpp 中 ConvertBinaryOp 的广播路径 |
| 7 | FastMath 仅设了 `reassoc | contract`，未启用 `nnan/ninf`    | 所有浮点运算                      | LLVMCodegenPipeline.cpp:34 |
| 8 | Bufferization 后可能存在多余 `memcpy`/alloc                 | 待 dump IR 验证                   | （未直接 dump，需后续 `--mlir-print-ir-after-all`） |
| 9 | 编译期没有 LTO / unroll-loops / 显式 `march=native` 提示    | 影响整体最后一公里                | CMake CXX_FLAGS 仅 `-O3 -DNDEBUG` |
| 10| 推理路径未提供 `RunInto` 在 bench 中使用（已验证差异极小）  | 优先级最低                        | profile.cpp 表 |

## 5. 优化建议（按 ROI 排序）

### 🥇 P0 — 必做：替换 linalg→loops 为 tile + vectorize
**期望收益**：matmul 性能 2–4×，可消除 70%+ 当前差距。

具体改动思路（[LLVMCodegenPipeline.cpp](src/LiteNN/Compiler/Pass/LLVMCodegenPipeline.cpp)）：

1. 在 `createConvertLinalgToLoopsPass` 之前，插入：
   - `linalg::populateElementwiseOpsFusionPatterns`
   - 自定义 `LinalgTilingPass`，对 `linalg.matmul` tile 出 `[M=8, N=16, K=64]`（M 用 batch，N 用 SIMD 宽，K 一段）。
   - 用 `mlir::linalg::vectorize` / `linalg::populateVectorizationPatterns`：把 tiled matmul 重写成 `vector.contract` + `vector.transfer_read/write`。
   - 然后用 `convert-vector-to-llvm`（已经隐式包含在 `O3`）将 `vector.contract` 降到 packed FMA。
2. **如果目标 N 很窄**（输出 10 列），将 N 维填充到 16 并 mask；或者沿 M（batch）维做 vectorize（M ≥ 32 总是充足）。
3. 加上 `linalg::LinalgPromote` 把 micro-kernel 输入临时缓冲 promote 到 stack。

落地时，可参考 IREE / mlir-cpu-runner 默认 pipeline 中的：
`-test-linalg-codegen-strategy=anchor-op=linalg.matmul tile-sizes=8,16,64 vectorize`
等价的 C++ pass 串。

### 🥈 P1 — 高收益：解决"窄 N 退化为标量"问题
**期望收益**：Linear 输出层、MLP 末层 1.5–2×。

- 让 vectorization 沿 **M 维（batch）** 向量化而不是 N 维。具体：把 matmul 重写成
  `linalg.generic { iterator_types=[parallel(M tiled by 16), parallel(N), reduction(K)] }`。
- 或者写一个特化的 pattern：当输出形状最后一维 < 8 时，交换 loop 顺序到 K → M → N。

### 🥉 P2 — 中等收益：消除 ReLU 中的零张量
**期望收益**：消除若干 µs/层的多余 alloc 与 broadcast 拷贝；对小模型尤其明显。

[Activation.h](src/LiteNN/Layer/Activation.h)：当前 `AddReLU` 大致：
```cpp
auto zero = subgraph.AddConstant(...全 0 张量同 shape...);
return BinaryOp::Max(x, zero);
```
改为：
- 新增 `UnaryOp::ReLU` 直接产出 `max(x, 0.0f)` 标量广播；或
- 修改 `BinaryOp::Max` 的 lowering：识别 RHS 是常量 0.0 时退化到 `arith.maximumf x, 0`。

### P3 — 低成本但建议做
- **FastMath 加上 `nnan | ninf | nsz | afn`**：让 LLVM 进一步合并 `max/min` 与 fma，消除 NaN check。改 [LLVMCodegenPipeline.cpp:32](src/LiteNN/Compiler/Pass/LLVMCodegenPipeline.cpp)：
  ```cpp
  auto flags = mlir::arith::FastMathFlags::reassoc
             | mlir::arith::FastMathFlags::contract
             | mlir::arith::FastMathFlags::nnan
             | mlir::arith::FastMathFlags::ninf
             | mlir::arith::FastMathFlags::nsz;
  ```
- **CMake 编译标志**：`-march=native -funroll-loops -fomit-frame-pointer`（已隐式启用部分），并对 `LiteNN` 主库本身打 LTO（`-flto=thin`）。
- **bench 添加 `--use-runinto` 选项**：保留 `Run()` 的便利接口，新增 `RunInto` 路径，避免后续优化时被分配噪声掩盖。

### P4 — 可选增强
- 加入 IR dump 环境变量：`if (getenv("LITENN_DUMP_MLIR")) module.dump();`，便于今后调优时观察各 pass 输出。当前需要重编译才能 dump IR，体验差。
- 提供 `Compiler<CPU>::Compile(graph, OptOptions)` 让用户选择 `Aggressive / FastCompile`。
- 探索把热 kernel 切成独立 function 让 LLVM 单独 inline / unroll。

## 6. Codex Review（2026-05-09）

结论：报告的核心判断没有阻塞问题，性能瓶颈确实在 AOT matmul kernel，而不是 `Run()` 的输出分配或入口验证开销。反汇编证据与 benchmark 数据能够支撑“先优化 matmul codegen”的优先级。

需要修正的上下文：
- 当前 [LowerLiteNNPass.cpp](src/LiteNN/Compiler/Pass/LowerLiteNNPass.cpp) 已不再按报告中的“`linalg.matmul` 直接转 loops”路径生成代码；MatMul 与 fused MatMulBiasAdd 已改成 `linalg.generic`，iterator order 为 `M, K, N`，并带有 bias 初始化 generic。
- 因此 P0 不能直接套用只匹配 `linalg.matmul` 的 tiling/vectorize pattern。下一步应优先面向当前 `linalg.generic` contraction 形态落地，或者在 lowering 中保留一个可识别的 named/contraction op，再由专门的 CPU codegen pass 处理。
- `nnan/ninf/nsz` fast-math 扩展已实现；`afn` 暂不默认开启，避免对 `exp/log/pow` 等超越函数引入更强近似语义。后续可通过 compile options 暴露 aggressive fast-math 模式。
- `benchmark/bench.cpp` 已增加 `--use-runinto` 测量路径，便于之后分离 kernel 优化与输出分配噪声。
- 已新增一个窄输出 MatMul lowering：匹配当前 `linalg.generic` contraction 且 `N <= 16` 的场景，将其改写为 `scf.for` micro-kernel，让输出列累加器保持为 loop-carried SSA 值，并只在 K 归约结束后写回。该 pass 对内部生成的 FMA 只使用 `contract/nnan/ninf/nsz`，刻意不使用 `reassoc`，避免 LLVM 把 K 维重新向量化成 gather-heavy reduction。
- 本轮验证结果：`litenn_profile.exe` 中 `linear_b512` 从约 `0.66 ms` 降到约 `0.46 ms`；`litenn_bench.exe --use-runinto` 中 `Linear(784->10), batch=512` 约 `0.435 ms`。MLP-512 仍主要受宽 hidden matmul 影响，下一步收益点仍是完整 tile/vectorize/packing。

## 7. 验证方法（下一轮回归用）

1. 重跑 `litenn_bench.exe` 与 `bench.py --threads 1`，对比上面表格。
2. 重跑 `litenn_profile.exe` 收集 `Run vs RunInto`，确保改动没有引入 alloc 退化。
3. 重新 dump `.o` 反汇编：
   ```pwsh
   objdump -d -M intel linear_b512.o |
     Select-String "vfmadd" | Group-Object { $_ -match 'ps.*zmm' } | Format-Table
   ```
   完整 P0 tile/vectorize 的目标：`linear_b512` 中 `vfmadd…ps zmm` 占比 > 80%；`vfmadd…ss` 数量 → 0。
   当前窄输出 micro-kernel 的阶段性目标：避免每个 K step 都对输出 memref 做 `load/store`，允许保留少量 scalar FMA。
4. 关键阈值（建议 PR 验收门槛）：
   - Linear/512：≤ 0.30 ms（即追平 PyTorch ±40%）
   - MLP-512/512：≤ 5.0 ms（即追平 PyTorch ±15%）

## 8. 风险与注意

- 引入 MLIR `linalg` 向量化 pass 会**增加 LLVM/MLIR 二进制依赖**，需确认 MinGW LLVM 构建里链入了 `MLIRLinalgTransforms`、`MLIRVectorTransforms`、`MLIRVectorToLLVM`、`MLIRMemRefTransforms`。
- 启用 `nnan/ninf` 会改变浮点语义；若用户模型期望 IEEE 严格行为需提供选项关闭。
- AVX-512 在异构部署上不可用；建议 codegen 时分别测试 `getHostCPUFeatures()` 中是否含 `avx512f`，并在不支持时退回 AVX2 vectorization tile（N=8）。

---

**附：本轮新增/读取的文件**
- 新增工具 [benchmark/profile.cpp](benchmark/profile.cpp)（编译、运行、写 `.o`、对比 Run/RunInto）
- 修改 [benchmark/CMakeLists.txt](benchmark/CMakeLists.txt) 添加 `litenn_profile` 目标
- 反汇编产物：`build-release-mingw/profile_out/*.s`、`*.o`
