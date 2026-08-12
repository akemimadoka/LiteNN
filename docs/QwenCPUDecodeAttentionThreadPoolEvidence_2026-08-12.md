# Qwen CPU Decode Attention and Thread-Pool Evidence - 2026-08-12

## 结论摘要

本报告记录 LiteNN 在固定 Qwen2.5-Coder 14B Q4_K_M 解码轨迹上的位置分段剖析、分组注意力并行化和
CPU 线程池唤醒优化。原始模型路径和所有本机绝对路径均未写入仓库。

结论分为三种证据强度：

1. **已证实的归因**：优化前的六个 profile 进程中，`attention.core` 是唯一在位置 `1-16` 到
   `113-128` 之间稳定增加的阶段，绝对增量为 `15.8-18.7 ms/token`。QKV、RoPE、KV append 等阶段没有
   同类增长，因此 128-token 窗口内的 LiteNN 位置斜率主要归属于 active-prefix attention。
2. **已证实的局部优化**：按 KV head 并行后，Qwen GQA 的 8 个 KV head 成为独立任务；线程池再通过
   polling-to-sleep 握手只唤醒真正休眠的 worker。`context=128` 微基准由串行 `0.365 ms` 降至
   `0.048 ms`，`context=2048` 由 `6.72 ms` 降至 `0.956 ms`，并保持逐元素完全一致。
3. **方向性整模收益，尚未正式验收**：最终三对 clean/profile 运行中，profile module 中位数由相邻的
   优化前 campaign 的 `217.827 ms/token` 降至 `202.546 ms/token`；`attention.core` 末段由
   `19.210` 降至 `6.679 ms/token`。但 clean 侧有一个 `221.933 ms/token` 离群进程，且位置分箱
   clean/profile overhead 门禁未通过，因此不能把 `7.02%` 的整模下降作为已验收回归数字。

优化后，`attention.core` 只占 profile module 中位数约 `3.91%`。短上下文的下一项实现工作不应继续
盲目改写 attention；应先完成稳定的跨 runtime 配对验收、已确认的向量 SwiGLU 缺口，以及对占约
`80.71%` 的投影阶段做跨 runtime 归因。Attention 仍是 2K 到 1M 长上下文的扩展重点。

## 测量边界

- 模型：本地 Qwen2.5-Coder-14B-Instruct Q4_K_M GGUF，模型路径已脱敏。
- 执行：CPU stateful AOT、cache hit、LLVM O0、8 个 helper threads、adaptive 调度、不指定 affinity。
- 权重：全部预打包为 field-interleaved-v4；运行过程中不允许 Interpreter 或 source-weight fallback。
- 轨迹：9 个固定 prompt token 加 128 个固定 decode token；自然 argmax 仍单独记录用于漂移诊断。
- 位置分箱：`1-16`、`17-48`、`49-80`、`81-112`、`113-128`。
- 配对：每个 campaign 含三对交替 clean/profile 进程；门禁包含 token、fallback、cache、功耗方案、
  whole/bin variance、profile overhead 和 stage coverage。
- 固定轨迹摘要：prompt SHA-256
  `c283080bdc6c1a7c05bb55f4d175d15bf4876f4e2cb63ecca6d50361b7e46917`，decode SHA-256
  `d6774768347feb151d99068777ea924274f12eb50036fc3d4c77adb2ccb55f87`。

## 优化前位置归因

首轮使用两个独立的三对 campaign。两轮都因严格方差或 overhead 门禁被拒绝，不能用于绝对吞吐验收；
但六个 profile 进程都在同一阶段呈现同方向、同量级的位置增长，因此足以选择下一项局部实现工作。

| Campaign | Clean / profile module 中位数 | Clean / profile CV | 验收 |
| --- | ---: | ---: | --- |
| A | `213.936 / 204.521 ms` | `3.948% / 7.803%` | 拒绝 |
| B | `225.324 / 229.130 ms` | `5.877% / 1.260%` | 拒绝 |

`attention.core` 的逐进程首末分箱变化如下：

| Campaign / run | `1-16` | `113-128` | 增量 |
| --- | ---: | ---: | ---: |
| A1 | `1.864 ms` | `18.726 ms` | `+16.862 ms` |
| A2 | `2.087 ms` | `17.986 ms` | `+15.898 ms` |
| A3 | `2.504 ms` | `20.522 ms` | `+18.017 ms` |
| B1 | `2.359 ms` | `18.313 ms` | `+15.953 ms` |
| B2 | `2.642 ms` | `18.448 ms` | `+15.806 ms` |
| B3 | `2.671 ms` | `21.341 ms` | `+18.669 ms` |

这个增量没有出现在 QKV projection、RoPE、KV append、attention output、Gate/Up、Down 或 logits 中。
因此可以排除“整个模型随时间共同降频”作为该斜率的充分解释，并把 grouped active-prefix attention
提升为实现目标。

## 局部实现和微基准

Grouped active-prefix attention 现在把每个 KV head 作为一个任务。Qwen2.5 GQA 的一个 KV head 负责 5 个
query heads，因此 8 个任务互不争用输出和 score scratch。低于约一百万标量操作的短工作保持串行，
避免固定调度开销。

| Active context | Grouped T1 | 初版 grouped T8 | 握手优化后 T8 | T1 到最终 T8 |
| ---: | ---: | ---: | ---: | ---: |
| 128 | `0.365 ms` | `0.124 ms` | `0.048 ms` | `7.60x` |
| 2048 | `6.72 ms` | `1.01 ms` | `0.956 ms` | `7.03x` |

`context=16/64` 仍由阈值选择串行，避免小任务回退。所有重复调用和 T1/T8 结果都满足
`max_abs_delta=0`。

初版并行在微基准有效，却没有改善整模：相邻三对 campaign 的 profile module 为
`217.827 ms/token`，`attention.core` 仍为 `10.141 ms/token`，末段仍为 `19.210 ms/token`。瀑布图解释了
这个反差：48 次 attention 调用平均每次 parallel wall 为 `0.247160 ms`，其中 dispatch 就占
`0.191633 ms`，约为 parallel wall 的 `77.5%`。

旧线程池在任务发布时无条件向所有辅助 worker 释放 semaphore，即使 worker 正在自适应轮询并已看见新
generation。新握手由 worker 在真正进入睡眠前设置 `sleeping`，发布者只唤醒这些 worker；轮询命中的
worker 直接执行。销毁路径也只唤醒实际休眠者。旧日志曾把 signal 重复累计为 14，修正后的计数表示
实际内核/semaphore 唤醒数；旧实现实际每次最多唤醒 7 个辅助 worker。

| 每次 attention 调用 | 初版并行 | 握手优化后 | 变化 |
| --- | ---: | ---: | ---: |
| Dispatch | `0.191633 ms` | `0.061233 ms` | `-68.05%` |
| Parallel wall | `0.247160 ms` | `0.150719 ms` | `-39.02%` |
| Final barrier | `0.044646 ms` | `0.023096 ms` | `-48.28%` |
| 实际 signal worker | `7` | 平均 `4` | 减少内核唤醒 |

同一类单次瀑布运行的完整 forced-generation 平均由 `210.961 ms/token` 降到 `199.894 ms/token`，即
`4.740` 到 `5.003 token/s`。这是相邻单次 A/B，只用于确认瀑布阶段变化与端到端方向一致，不代替
多进程门禁。

## 优化后整模数据

最终三对 campaign 全程保持 balanced 功耗方案，token、fallback、cache hit、stage coverage 和轨迹门禁
均通过。Profile overhead 中位数为 `-0.741%`，profile module CV 为 `0.786%`。验收仍被 clean module
CV `5.716%` 和位置分箱 clean/profile overhead 拒绝；第三个 clean 进程的 `221.933 ms/token` 是主要
离群点，报告保留该进程，不做事后删除。

| 指标 | 初版 attention 并行 | 线程池握手后 | 方向性变化 |
| --- | ---: | ---: | ---: |
| Clean module 中位数 | `219.022 ms` | `202.611 ms` | `-7.49%` |
| Profile module 中位数 | `217.827 ms` | `202.546 ms` | `-7.02%` |
| `attention.core` 全窗 | `10.141 ms` | `7.913 ms` | `-21.97%` |
| `attention.core` 末段 | `19.210 ms` | `6.679 ms` | `-65.23%` |
| `attention.core` 首末增量 | `17.134 ms` | `4.073 ms` | `-76.24%` |

最终 profile 的阶段中位数如下。它们可用于 LiteNN 内部预算排序，但不能单独证明相对其他 runtime 的
实现缺口。

| 阶段 | `ms/token` | Module 占比 |
| --- | ---: | ---: |
| QKV projection | `20.381` | `10.06%` |
| RoPE | `0.238` | `0.12%` |
| KV append | `0.047` | `0.02%` |
| Attention core | `7.913` | `3.91%` |
| Attention output projection | `14.447` | `7.13%` |
| Gate/Up projection | `70.771` | `34.94%` |
| SwiGLU activation | `13.425` | `6.63%` |
| Down projection | `45.383` | `22.41%` |
| Norm | `0.394` | `0.19%` |
| Logits projection | `12.491` | `6.17%` |
| Generated-code residual | `17.157` | `8.47%` |

五组量化 projection 合计约 `163.473 ms/token`，占 module 的 `80.71%`。这说明短上下文 attention 的
优先级已经下降，但“占比最大”不等于“相对 llama.cpp 的缺口最大”：新的 projection 工作必须先用匹配
阶段和 cache-cold 证据确认，而不能从本表直接选择一个内核重写。

## 首次缓存发布的资源证据

资源问题已经独立归档到 `docs/QwenFirstCacheMemoryEvidence_2026-08-12.md`。连续采样和所有权审计确认，
GGUF importer 曾为约 9 GB 推理权重额外分配同尺寸 gradient；改为 frozen variable 后，首次构建峰值从
先前约 `27.37/27.49 GB` 的 working set/private 单点观测降到 `18.566/18.679 GB` 的连续峰值。模块构造后
源权重 owner 会立即释放，九步 decode 的活动 RSS/private 约为 `9.4-9.9/9.5-10.0 GB`。

剩余首次构建峰值来自 source payload 与 `9.160 GB` prepared region 在 externalization/object emission 期间
重叠，后续应作为流式读取和原子发布问题处理，而不是继续归因到训练梯度或运行态泄漏。

## 决策和后续门禁

1. 已完成的位置归因、KV-head 并行和线程池握手保留；它们均有正确性、微基准和瀑布证据。
2. 短上下文下不再继续无证据地扩展 attention kernel。先做至少五对、功耗稳定的 LiteNN/reference 固定
   轨迹验收；保留所有离群进程，不通过事后剔除使门禁变绿。
3. 已有跨 runtime 证据确认 strict SwiGLU 仍有约 `10.833 ms/token` 缺口，向量数学 provider 仍是短
   上下文最高优先级实现项。
4. 对 projection 聚合预算做匹配阶段剖析，区分 Q4_K/Q6_K compute、cache-cold 权重带宽、dispatch、
   barrier 和 residual；只有被接受的跨 runtime 差值才能选择下一个 kernel。
5. 首次 cache 发布的梯度副本和运行态 source-owner 泄漏已经关闭；下一阶段以连续峰值低于必要 payload
   的 `1.5x` 为门禁，评估 source mmap 和 prepared-region 流式原子发布。
6. 长上下文按 2K、8K、32K、128K、1M 分层验证。Attention 在 128 token 只占约 `3.91%`，但其复杂度
   随 active context 增长；在线 softmax、score-buffer 消除和 paged attention 应由这些层级的数据触发。
