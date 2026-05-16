# LiteNN vs PyTorch 性能分析（2026-05-16）

> 本报告以 [benchmark/results/backend_pytorch_comparison_2026-05-16.md](../benchmark/results/backend_pytorch_comparison_2026-05-16.md) 的最新一轮 LiteNN（CPU AOT / CUDA Bridge / CUDA Native）与 PyTorch（CPU 1T/16T、CUDA）对比数据为依据，沿用 [PerformanceAnalysis.md](PerformanceAnalysis.md) 的格式，对 **CPU AOT、CUDA Bridge、CUDA Native** 三条执行路径分别定位瓶颈，并给出可落地、按 ROI 排序的改造方案。本报告 **不覆盖** 原 PerformanceAnalysis.md，专门用于供 GPT/Codex review。

## 1. 测试环境

| 项目 | 配置 |
|---|---|
| CPU | AMD Ryzen 9 9950X (Zen 5, 16C/32T, AVX-512) |
| GPU | NVIDIA GeForce RTX 4090 (Ada, 128 SM, 16384 CUDA core, ~83 TF FP32) |
| OS | Windows |
| 编译器 | MinGW GCC，`build-cuda-mlir-mingw-gcc-bench/`，`LITENN_BUILD_BENCHMARKS=ON`，`LITENN_ENABLE_MLIR=ON`，`LITENN_ENABLE_CUDA=ON` |
| LLVM target | host CPU 特性（包含 AVX-512），`O3 + Aggressive` |
| CUDA AOT 目标 | 默认 `sm_30`（环境变量 `LITENN_CUDA_AOT_TARGET` 可改） |
| PyTorch | 2.9.1+cu128 / Python 3.11.9 |
| 基准脚本 | [benchmark/bench.cpp](../benchmark/bench.cpp)、[benchmark/bench.py](../benchmark/bench.py)，`--benchmark_min_time=0.1s` |

## 2. 关键数据回顾（ms/batch，越低越好）

### 2.1 模型推理

| Model | Batch | LiteNN AOT RunInto | CUDA Bridge | CUDA Native(部分) | PyTorch CPU 1T | PyTorch CPU 16T | PyTorch CUDA |
|---|---:|---:|---:|---:|---:|---:|---:|
| Linear(784→10) | 1 | **0.0006** | 0.063 | — | 0.005 | 0.005 | 0.015 |
| Linear(784→10) | 32 | **0.0034** | 0.067 | — | 0.016 | 0.012 | 0.016 |
| Linear(784→10) | 128 | **0.013** | 0.113 | — | 0.047 | 0.029 | 0.016 |
| Linear(784→10) | 512 | **0.053** | 0.408 | — | 0.177 | 0.085 | 0.017 |
| MLP-128 | 1 | **0.0025** | 0.061 | — | 0.014 | 0.014 | 0.035 |
| MLP-128 | 32 | **0.021** | 0.098 | — | 0.064 | 0.033 | 0.037 |
| MLP-128 | 128 | **0.086** | 0.311 | — | 0.211 | 0.068 | 0.036 |
| MLP-128 | 512 | 0.347 | 1.54 | — | 0.815 | **0.187** | 0.036 |
| MLP-512 | 1 | **0.016** | 0.150 | — | 0.039 | 0.042 | 0.052 |
| MLP-512 | 32 | 0.115 | 0.575 | — | 0.296 | **0.107** | 0.057 |
| MLP-512 | 128 | 0.455 | 1.27 | — | 1.03 | **0.253** | 0.055 |
| MLP-512 | 512 | 2.04 | 3.38 | — | 4.03 | **0.748** | **0.059** |

### 2.2 单算子 MatMul（128×128）

| Batch | LiteNN CUDA Native | PyTorch CPU 1T | PyTorch CPU 16T | PyTorch CUDA |
|---:|---:|---:|---:|---:|
| 1 | **1.20** | 0.002 | 0.002 | 0.007 |
| 32 | 0.909 | 0.010 | 0.009 | 0.011 |
| 128 | 0.716 | 0.032 | 0.013 | 0.007 |
| 512 | 0.740 | 0.122 | 0.032 | **0.007** |

### 2.3 直观结论

- ✅ **CPU AOT 单线程已经全面快于 PyTorch CPU 1T**；MLP-512/512 上是 **2.0×**，Linear/512 上是 **3.3×**，达到既定目标。
- ❌ **CPU AOT 与 PyTorch CPU 16T 仍有 1.5–2.7× 差距**（MLP-512/512 2.04 vs 0.75），核心缺口是 **单算子 intra-op 并行**。
- ❌ **CUDA Bridge 比同模型 CPU AOT 还慢一个量级**，这是 **结构性 bug**——它根本没有在 GPU 上执行算子。
- ❌ **CUDA Native MatMul 全部都慢于 PyTorch CUDA**（batch=512 下 0.740 vs 0.007，**100× 差距**），batch=1 的 1.20 ms 完全是 per-call 启动开销主导。

下面分三条路径展开。

---

## 3. 路径 A：CPU AOT（已优于 PyTorch 1T，差距在多线程）

### 3.1 当前态

继 [PerformanceAnalysis.md §6.4](PerformanceAnalysis.md) 落地的 row-tile micro-kernel + selective RHS packing 之后，单线程 CPU AOT 已经在所有规格上稳定快于 PyTorch CPU 1T（参见 §2.1 中标 **加粗**）。`Run vs RunInto` 差距 ≤ 噪声，证明输出分配与入参校验不是瓶颈，问题完全集中在 kernel 与并行度。

### 3.2 与 PyTorch CPU 16T 的差距来源

| 规格 | LiteNN 1T | PyT 16T | 差距 | 主导原因 |
|---|---:|---:|---:|---|
| Linear/512 | 0.053 | 0.085 | 0.62× (LiteNN 更快) | 单层算子小，PyT 调度开销 > 并行收益 |
| MLP-128/512 | 0.347 | 0.187 | 1.85× | hidden=128 单 kernel ≈ 0.1ms，PyT 用 BLAS 多线程拆 batch |
| MLP-512/512 | 2.04 | 0.748 | 2.73× | hidden=512 是热点，PyT 在 batch 维并行 16 线程 |

LiteNN 的所有 CPU 数字均为 **单 benchmark 线程 + 单线程 kernel**。而 PyTorch CPU 16T 内部对每个 GEMM 通过 OneDNN/MKL 沿 batch（M 维）拆 16 路。这意味着差距随 **算术强度 × batch** 同步放大，与 P0 早期定位的 micro-kernel 问题已经无关。

### 3.3 当前 CPU AOT 路径剩余的 micro 问题

参考已落地的 `analyze_asm.ps1` 数据：`subgraph_0` 中 hot kernel 已达 `PackedFMA=96, ScalarFMA=0, Scatter=0, StackVectorOp=0`，**指令层基本无脏数据**。这说明：

1. 单线程不可能再有数量级提升。
2. 真正剩下的低风险条目：
   - **K 维 panel packing**：[PerformanceAnalysis.md](PerformanceAnalysis.md) 中已经为 `N ≥ 256` 做了 RHS packing，但还没有为 K 维做 cache blocking。当 K=784 时，B panel 是 `784 × Nstep × 4B`（约 100 KB），已经超过 L1。沿 K 维切 Kblock（例如 128/256）+ M 维 register tile 的"经典 6.5.x panel + micro-kernel"模式应该能再压一档大模型 hidden layer 的时延。
   - **`N=128` 不进行 packing 的临界点**：当前数据看，MLP-128 在 batch=512 时 0.347 ms。若能不退化 MLP-128 baseline 而把 packing 阈值降到 `N ≥ 128`，理论可省一次 K-stride read pass。建议加一个 cost model：`Kbytes × Ntile > L2/2 ⇒ pack`。
   - **MLP 末层 + softmax/loss 融合**：当前只融合了 `MatMul+BiasAdd+ReLU`，没有融合最后一层的 `MatMul+BiasAdd+(Softmax)`。在推理路径上影响小，在训练路径（参考第 5.x 训练性能）上影响大。

### 3.4 真正的"P0"：CPU 单算子并行

PyTorch 16T 的优势 100% 来自 batch（M 维）拆分。LiteNN AOT 想追上必须在 **同一个生成 object 内** 引入可控的 intra-op 并行，且不能引入平台线程库的强依赖（CUDA AOT 也面临一致问题）。建议方案：

1. **编译期生成"tile plan"**：把宽 hidden 层的 `scf.for M` 外层切成 `[M / Mtile]` 个 work item，把整个 micro-kernel 抽成一个 `func.func`，主入口生成一个 dispatch loop（参数为 `worker_id, worker_count`）。
2. **运行时提供 thread pool API**：LiteNN runtime 增加一个 `ParallelFor(int begin, int end, void(*)(int, void*))` 的 ABI（类似 OneDNN omp wrapper），生成 object 里只插 `extern "C"` 调用，不依赖任何平台线程库符号。
3. **触发阈值**：用编译期估算 FLOPs，超过阈值（例如 1e6 FLOPs/call）才生成并行入口；小 batch / 小层继续走 serial path，避免线程调度开销（参考 [PerformanceAnalysis.md §6.1 中 `--parallel-requests` 的反噬现象](PerformanceAnalysis.md)）。
4. **数据竞争边界**：M 维拆分天然不写同一行，bias/relu 都是 elementwise，无需 reduce；K 维不拆，避免做 partial-sum reduce。

> 这一步是 LiteNN CPU 后端走向"真·和 OneDNN/MKL 同一档"的必经路。预计 MLP-512/512 可以从 2.04 ms 降到 ~0.30–0.45 ms（16T 接近 OneDNN 0.748 也不奇怪，因为 9950X 单核 AVX-512 比典型 Intel 服务器更强）。

---

## 4. 路径 B：CUDA Bridge（结构性瓶颈）

### 4.1 当前行为（关键发现）

`Compiler<CUDA>::Compile(graph, CUDA{})` 对非单算子 MatMul 图会落入 **CPU bridge** 后端（`CompiledModuleBackend::CPUNative`）。其 `RunInto` 实现见 [src/LiteNN/Compiler/CompiledModule.cpp:3044](../src/LiteNN/Compiler/CompiledModule.cpp)：

```cpp
// 对每个 input：在 CPU 上 new Tensor<CPU>，cudaMemcpyAsync D2H + sync
// impl_->cpuModule.RunInto(cpuInputs, cpuOutputs);   // 跑 CPU AOT
// 对每个 output：cudaMemcpyAsync H2D + sync
```

也就是说 **CUDA Bridge ≠ 在 GPU 上跑模型**，而是 "D2H → CPU AOT → H2D"。它在 benchmark 表里被列在 "CUDA" 一栏会让读者严重误解，并且：

- **每次 Run 都做 2 次往返 cudaMemcpy + 同步**（输入 + 输出各一轮）。
- **每个 tensor 在 CPU 端 `Tensor<CPU>(Uninitialized, ...)` 用默认 heap allocator**，**没有用 `cudaMallocHost` 的 pinned 内存**，因此 PCIe 传输只能达到 pageable 带宽（~10–15 GB/s on x16 4.0）。
- **每次 H2D / D2H 都强制 `cudaStreamSynchronize`**（[CUDA.cpp:670](../src/LiteNN/Device/CUDA.cpp)），无法掩盖延迟。

### 4.2 量化拆解

以 MLP-512/512 为例：
- CPU AOT RunInto：2.04 ms
- CUDA Bridge：3.38 ms → **额外开销 1.34 ms**

每次调用的额外往返字节数 ≈ `(input 512×784 + output 512×10) × 4B ≈ 1.6 MB`。在 pageable 模式下大约就是 0.1–0.2 ms 的纯传输 + 4 次 cudaMalloc/Free（每 `Tensor<CPU>` 一次 `new[]`，每个 GPU 输出已经预分配）+ 4 次同步。其余 ~1 ms 主要来自 **小尺寸 cudaMemcpy 的固定开销**（每次几十 µs 的 launch+sync）。

Linear/1 的 CUDA Bridge 是 0.063 ms vs CPU AOT 0.0006 ms，差 100×；几乎全部都是 cudaMemcpy + sync 的固定成本。

### 4.3 改进建议（按 ROI 排序）

#### P0：明确"Bridge ≠ CUDA execution"，并在 benchmark/文档中改名

- 把 `BMCUDABridgeRunInto` 改名为 `BMCUDABridgeFallback`，列名改为 "CUDA→CPU Fallback"。
- README、docs 中表格也要标注：在没有原生 CUDA backend 的算子组合上，bridge 表现 **不能** 用来评估 GPU 后端性能。

#### P1：让 bridge 至少不要白白慢

- **Pinned host buffer 复用**：在 `Impl` 里给每个 input/output spec 预分配 `cudaMallocHost` 的 host 缓冲，每次 RunInto 重用。预期 H2D/D2H 带宽从 ~12 GB/s 提到 ~25 GB/s，等价为 0.6–0.8 ms 的复合收益。
- **取消"按 tensor 单独同步"**：把所有 H2D 提交完后只同步一次；CPU 跑完后所有 D2H 也只同步一次。
- **复用 CPU 中转 Tensor**：当前每次 RunInto 都 `new Tensor<CPU>(Uninitialized,…)`，可以挂在 `impl_` 上池化。

#### P2：把 CUDA Native backend 的覆盖面扩到 MLP

当前只有 `MatMul` 单算子能进 CUDANative 路径（参见 §5）。一旦补齐 `BiasAdd / ReLU / MatMulBiasAddReLU` 的 CUDA 原生 lowering，bridge 路径就只用于真正没实现的算子，而不是 MLP/Linear 这种 first-class 模型。

---

## 5. 路径 C：CUDA Native（最严重的瓶颈）

### 5.1 关键数据

| Batch | LiteNN MatMul(128²) | PyTorch CUDA | 倍率 |
|---:|---:|---:|---:|
| 1 | 1.20 ms | 0.007 ms | **171×** |
| 32 | 0.909 ms | 0.011 ms | **83×** |
| 128 | 0.716 ms | 0.007 ms | **102×** |
| 512 | 0.740 ms | 0.007 ms | **106×** |

batch=1..512 几乎是常数 ~0.7–1.2 ms，与 batch 完全无关。**强烈提示瓶颈是 per-call 固定开销，而不是 kernel 本体**。

### 5.2 反汇编/源码级瓶颈定位

按代码路径逐个列出 per-call 固定开销（每个 `RunInto` 都会跑一遍）：

1. **每次都 `cublasCreate` + `cublasDestroy`** —— `RunCUDANativeLibraryCall` → `DeviceTraits<CUDA>::DoBinaryOp` → `TryCUBLASMatMul`，里面 `CUBLASHandle handle(options);` 是 stack 局部对象，调用 `cublasCreate`。**`cublasCreate` 在 RTX 4090 上典型 200–800 µs**（首次更慢，包含 driver init + workspace prepare）。这是 1.2 ms baseline 的主要来源。代码位置：[CUDA.cpp:378](../src/LiteNN/Device/CUDA.cpp)。
2. **每次都 `cuModuleGetFunction(name)`**（即使本次走的是 cuBLAS 而不是 PTX，多 kernel payload 仍会走 PTX 分支） —— [CUDA.cpp:584](../src/LiteNN/Device/CUDA.cpp)。每次都做 `std::string(functionName)` 分配 + driver lookup（~5–20 µs）。
3. **每次都 `CUDANativeWorkspaceBuffer workspace(device, workspaceBytes)`**：`cudaMalloc` + `cudaFree` 一对——[CompiledModule.cpp:2479](../src/LiteNN/Compiler/CompiledModule.cpp)。即便 `workspaceBytes==0` 也走构造函数，但确实跳过 alloc；当包含 PTX kernel 时一定会 alloc。`cudaMalloc/Free` 在 4090 上 ~50–150 µs。
4. **`CUDADeviceGuard guard(device.deviceIndex)`**：`cudaSetDevice` 一次，~5–10 µs。
5. **强制 `cudaStreamSynchronize`**（`options.synchronize=true` 默认）。

仅 #1 一项就把 batch=1 的 1.2 ms 解释了 60–70%。这与"batch 升到 512 后时延几乎不变"的现象完全一致——kernel 本体只占几十 µs。

### 5.3 PyTorch CUDA 为什么 0.007 ms

PyTorch CUDA 后端：
- cuBLAS handle 在 device init 时 **进程内单例**（per thread / per stream cached）。
- `cublasLt` heuristic 选 kernel 后缓存，后续 call 只是 `cublasLtMatmul` 的纯 launch（~2–5 µs）。
- 不在每次 call 做 `cudaMalloc`，workspace 是单例长生命周期。
- 没有 host-side `string` 查 kernel name。
- 没有强制 sync（除非显式 `torch.cuda.synchronize()`）。这导致 ms/batch 看到的只是 launch overhead。

### 5.4 LiteNN CUDA Native 真实算力利用率估算

MatMul(M=512, K=N=128) 的 FLOPs ≈ 2·512·128·128 ≈ 16.8 MFLOPs。RTX 4090 峰值 FP32 ≈ 83 TFLOPs。
- 理论时延：16.8e6 / 83e12 ≈ **0.20 µs**。
- 实测 740 µs，**SM 利用率 < 0.03%**。

CPU AOT 在同尺寸下大约 50 µs（推算自 MLP-512/512 占比），所以 **当前 CUDA Native 实现比 CPU AOT 还慢 15×**。这本质是工程问题，不是硬件/算法问题。

### 5.5 改进建议（按 ROI 排序）

#### 🥇 P0：彻底消除 per-call 固定开销

| # | 改动 | 文件 | 预期节省 |
|---|---|---|---:|
| 1 | `CUBLASHandle` 改成 `CompiledModule<CUDA>::Impl` 成员，启动一次构造、销毁一次 | [CUDA.cpp](../src/LiteNN/Device/CUDA.cpp) [CompiledModule.cpp:2950](../src/LiteNN/Compiler/CompiledModule.cpp) | **~500 µs/call** |
| 2 | 在 `CompiledModule<CUDA>::Impl` 缓存 `CUfunction`（按 kernel name 在 `LoadCompiledModule` 时一次性 `cuModuleGetFunction`） | [CompiledModule.cpp:2998-3010](../src/LiteNN/Compiler/CompiledModule.cpp) [CUDA.cpp:580](../src/LiteNN/Device/CUDA.cpp) | ~10–30 µs/call |
| 3 | 把 `CUDANativeWorkspaceBuffer` 改成 `Impl` 上的池化对象（大小按 `max(payload.workspaceBytes, each kernel.workspaceBytes)`，启动时一次性 alloc） | [CompiledModule.cpp:2340](../src/LiteNN/Compiler/CompiledModule.cpp) | ~100 µs/call |
| 4 | 默认 `synchronize=false`；只在 `Run`（返回 owning tensor）的最后一次同步 | [CompiledModule.cpp:3071](../src/LiteNN/Compiler/CompiledModule.cpp) | 0–200 µs/call（释放 PyTorch-CUDA 那种 ms 级假象） |
| 5 | 删除 hot path 上的 `std::string(functionName)` 临时分配（改 `string_view` + `frozen_map`/`char*`） | [CUDA.cpp:584](../src/LiteNN/Device/CUDA.cpp) | 1–3 µs/call |

#### 🥈 P1：扩大 CUDA Native 覆盖面

当前只有"单 MatMul"图能 CUDANative。模型场景下从未触发，所以模型推理全走 bridge（§4）。需要：

1. 在 `Compiler<CUDA>` 的 dispatch 里，把 `MatMul + BiasAdd + Activation` 整条链路放到 `CUDANative`：cuBLAS 之后追加一条简单的 PTX/MLIR 生成的 epilog kernel（fused bias+ReLU）。
2. 提供 fallback：未覆盖算子时单独把该算子拆出来走 bridge，而不是整图退化。
3. CUDA Graph：对静态 shape 模型，把多 kernel launch 包成 `cudaGraph_t`，下次执行只 `cudaGraphLaunch`（µs 级 overhead）。RTX 4090 上对 MLP-512/512 batch=1 推理这能从"几 ms"压到 ~50 µs。

#### 🥉 P2：扔掉 `sm_30` 默认，启用 Ada/Hopper 特性

`LITENN_CUDA_AOT_TARGET=sm_30` 默认值意味着 NVPTX 输出的 PTX 是 Kepler-class，driver 在 4090 上会做一次 **JIT compile to sm_89**，并且生成的 SASS 不会使用 Ampere/Ada 的 Tensor Core、async copy 等。建议：
- 默认 `sm_75`（Turing-baseline，覆盖 99% 现役 GPU），并允许 `LITENN_CUDA_AOT_TARGET=native` 自动检测当前设备 SM 版本。
- 对 cuBLAS library call 没影响（cuBLAS 内部用 SASS），但对未来手写的 PTX kernel 影响巨大。

#### P3：用 cuBLASLt 取代 cuBLAS legacy + 显式 algo selection

`cublasSgemm` 走的是 legacy heuristic；对 K=N=128 这种小矩阵，cuBLASLt + `CUBLASLT_MATMUL_PREF_*` 可以选到 Tensor Core 路径（即使 FP32，4090 上有 TF32 fallback）。预期对 batch≥128 的 MatMul 时延能从 cuBLAS legacy 的 ~5 µs 进一步降到 ~2 µs，且 CTA 大小能更贴合 4090。

#### P4：长期方向（如果 LiteNN 想真正自研 CUDA kernel）

- 把 MLIR → NVPTX 链路打通到能生成 `wmma` Tensor Core 调用；
- 用 `nvcc --gpu-architecture=compute_89 --ptxas-options=-v` 看 register/spill；
- 把目前 cuBLAS library-call payload 与 PTX payload 的混合 ABI 抽象成稳定 IR，避免重复维护两套 dispatch。

---

## 6. 训练路径（前瞻）

bench 当前只有推理，没有训练。但既然 LiteNN 已经有 [src/LiteNN/Training/](../src/LiteNN/Training) 和 Autograd 测试，下面三点是预判可能的下一个瓶颈，建议建立 baseline：

1. **反向图同样会走 CUDA Bridge**：如果推理路径的 bridge 问题不解决，训练会一次 forward + 一次 backward 各 2 次 PCIe 往返。
2. **优化器更新当前若用 CPU 实现**：每个 step 必然 D2H/H2D 整个权重 tensor，对 hidden=512 的层每步 ~1 MB 往返，1k step 训练就是 ~GB 级冗余传输。
3. **Loss/Softmax 没有融合**：CPU AOT 路径上 softmax + NLL 是独立算子，每次走完整 reduce + exp + log。可与上一层 MatMul 融合。

建议在 v0.4 阶段补一组 `bench_train.cpp`，至少覆盖：MNIST MLP-128 / MLP-512 单 step + 100 step 的 forward+backward+SGD。

---

## 7. 总体优先级路线图

| 顺序 | 任务 | 影响面 | 难度 | 影响数据 |
|---|---|---|---|---|
| 1 | CUDA Native：cuBLAS handle / workspace / CUfunction 池化 | CUDA Native MatMul/all-shape | 低 | 1.20→~0.05 ms（batch=1），其他 ~0.05 ms |
| 2 | benchmark 文档 + 列名把 "CUDA Bridge" 改为 "CUDA→CPU Fallback" | 沟通 / 项目可信度 | 极低 | 0（避免外部读者误解） |
| 3 | CUDA Bridge：pinned host buffer + 单次 sync | CUDA Bridge 全部 | 中 | 0.4–1.5 ms 收益 |
| 4 | CUDA Native：扩展 BiasAdd/ReLU/Fused epilog + Graph capture | 把模型从 bridge 拿回 CUDANative | 高 | MLP-512/512 GPU 路径 3.38 → 目标 <0.2 ms |
| 5 | CPU AOT：runtime threadpool + 编译期 dispatch loop | CPU AOT 大 hidden | 高 | MLP-512/512 2.04 → 目标 0.3–0.5 ms |
| 6 | CUDA AOT target 默认改为 `sm_75` 或 `native` | 未来 PTX kernel | 低 | 长期收益 |
| 7 | K 维 panel packing + cost model 阈值下调 | CPU AOT 中等 hidden | 中 | 5–15% |
| 8 | 训练路径 baseline + Softmax+NLL 融合 | 训练 | 中 | 待测 |

## 8. 给 review 的开放问题

- **CUDA Native 是否还要继续维护"cuBLAS legacy + PTX 混合 payload"两条 ABI**，还是应该收敛到 cuBLASLt + 自生成 epilog？长期看自生成 epilog 是 LiteNN 区别于"cuBLAS 薄包装"的关键卖点。
- **CPU 单算子并行**的运行时 ABI 是放在 LiteNN runtime 里（自带 thread pool），还是允许用户注入（类似 OneDNN 的 `dnnl_threadpool_iface_t`）？后者更灵活，但会让生成 object 多一个 extern 依赖。
- **是否接受将默认 `synchronize=false`**？这会让 `RunInto` 在 CUDA 路径上的"完成"语义改变；推理服务端通常会显式同步，但本地小脚本会"看起来变快但结果还没出来"。
- **CUDA Graph 的 capture/replay** 是否要做成默认行为？capture 一次的代价较高，重 launch 极便宜，需要权衡 first-call 延时。
- **是否引入"CompiledModule fingerprint cache"**：同一 graph + device + AOT target 不重复编译。CUDA AOT 当前编译耗时较高（PTX 生成 + driver JIT），这会显著改善冷启动。
