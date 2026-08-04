# Qwen CPU Decode Build and Runtime Control - 2026-08-04

This report owns the build-, scheduler-, and frequency-controlled Qwen2.5-Coder-14B Q4_K_M CPU decode evidence gathered
after the first paired comparison. It supersedes the performance conclusion in
`QwenCPUDecodePairedControl_2026-08-04.md`; that earlier report remains useful as evidence for the paired harness and
variance problem, but its GNU/OpenMP llama.cpp binary was not the strongest local reference.

## Scope and Method

- Host: AMD Ryzen 9 9950X, 16 physical cores and 32 logical CPUs, Windows, balanced power policy.
- Workload: Qwen chat template with 9 prompt tokens and 16 requested generated tokens. The steady comparison contains
  15 eval/module calls after the first generated token.
- Runtime: CPU-only, flash attention off, F16 KV, mmap/repack/warmup enabled, greedy sampling, EOS ignored.
- llama.cpp: commit `b81c2cdd7`, Release, `GGML_NATIVE=ON`, CUDA and BLAS off.
- Correctness gates: byte-identical generated text, identical prompt/eval token windows, no LiteNN fallback, and a
  per-runtime coefficient of variation below 3%.
- Scheduling: configuration and build profiles were alternated in forward/reverse order for three repetitions each.
- Frequency: Windows PDH `Processor Information` counters sampled actual frequency and processor utility throughout
  each process. The report uses utility-weighted actual frequency rather than the fixed power-policy frequency.

The repository-owned tools are `benchmark/run_llama_cpp_configuration_sweep.py` and
`benchmark/run_paired_gguf_decode_control.py`. Their emitted commands and metadata redact the model, executable, prompt,
and other absolute paths by default.

## Runtime Strategy Sweep

The first 11-profile sweep used the GNU/OpenMP binary and varied only threads, affinity, polling, and priority. All
profiles passed output and variance gates.

| Profile | Median t/s | Difference from T2 baseline | Interpretation |
| --- | ---: | ---: | --- |
| T2 baseline | 4.66 | 0.00% | Best general thread count in this batch |
| Priority 2 | 4.68 | +0.43% | Noise-sized improvement |
| Poll 0 | 4.66 | 0.00% | Polling policy is not the missing throughput |
| Poll 100 | 4.63 | -0.64% | No benefit |
| T3 | 4.55 | -2.36% | More threads regress token decode |
| Cross-CCD strict T2 | 4.56 | -2.15% | Cross-CCD placement is mildly worse |
| T1 | 3.89 | -16.52% | Insufficient parallelism |
| Same-core SMT strict T2 | 2.97 | -36.27% | Negative control confirms affinity masks took effect |

The same unpinned T2 command later moved from `4.66` to `5.07-5.50 t/s` across batches. This host-state drift is much
larger than the priority or polling effects, so non-alternating absolute numbers cannot support a cross-runtime claim.

## Build Matrix

Three llama.cpp binaries were compiled from the same source commit with `-march=native`. The profiles differed only in
compiler and OpenMP availability. Each row is an alternating three-run median from the same accepted sweep.

| Compiler | OpenMP | SHA-256 | Median ms/token | Median t/s | CV | Weighted actual MHz |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| GNU 16.1 | On | `e0e0ea64c69656d08b11c1a9bfbf9ea5ef1843fc2e82e923978d5dc0aeb35271` | 197.64 | 5.06 | 0.41% | 5084 |
| GNU 16.1 | Off | `cce57380dcba64ed4e73d4bebe53065b4b515e05e80e8c67e89c7580998bcdbf` | 170.02 | 5.88 | 0.85% | 5077 |
| Clang 22.1.8 | Off | `b062b8bb356be73f2ecf1f72285119ca4551f54004940d12c489083375ed3c23` | 165.93 | 6.03 | 0.38% | 5086 |

Measured effects:

1. Disabling OpenMP under GNU improves throughput by `16.21%` (`5.06 -> 5.88 t/s`). This is the dominant controlled
   difference.
2. Replacing GNU with Clang while keeping OpenMP off adds `2.55%` (`5.88 -> 6.03 t/s`).
3. Weighted actual frequency is within `0.18%` across all three rows. Neither gain is a clock-frequency artifact.
4. Clang/no-OpenMP is `19.17%` faster than the original GNU/OpenMP reference. The old reference materially
   underestimated llama.cpp CPU capability on this host.

## Strong Paired LiteNN Control

The Clang/no-OpenMP binary was then paired directly with the unchanged LiteNN CPU AOT cache-hit path. All three pairs
passed text, token-window, no-fallback, and variance gates.

| Pair | Order | llama.cpp t/s | LiteNN t/s | LiteNN paired difference | llama.cpp MHz | LiteNN MHz |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | llama.cpp -> LiteNN | 5.800 | 5.518 | -4.86% | 5127 | 5106 |
| 2 | LiteNN -> llama.cpp | 6.000 | 5.431 | -9.49% | 5020 | 5115 |
| 3 | llama.cpp -> LiteNN | 5.970 | 5.642 | -5.49% | 5062 | 5115 |
| Median | Alternating | 5.970 | 5.518 | -5.49% paired median | 5062 | 5115 |

llama.cpp CV was `1.82%`; LiteNN CV was `1.92%`. The ratio of independent medians would place LiteNN `7.57%` behind,
but the `-5.49%` median of per-pair differences is the preferred result because each pair controls adjacent host state.
LiteNN had equal or higher measured frequency in two pairs, including the slowest relative run, so frequency does not
explain the deficit.

## Conclusions

1. The previous conclusion that LiteNN was `1.6-3.4%` ahead of local llama.cpp is invalid. It compared against a
   GNU/OpenMP build whose OpenMP configuration alone costs about `16%` on this decode workload.
2. The strongest reproducible local reference is currently Clang/no-OpenMP at about `5.97-6.03 t/s`. Against it,
   LiteNN is about `5.5%` behind under the preferred paired statistic.
3. The independently observed `6.85 t/s` result is now only `13.60%` above the best local llama.cpp sweep median and
   `14.74%` above the strong paired llama.cpp median. Most of the former `25%` discrepancy has been explained by the
   build configuration.
4. Thread count still matters, but explicit affinity, polling, and priority did not reveal a material missing gain.
   Same-core SMT and cross-CCD controls instead confirm that the sweep can detect harmful placement.
5. The earlier Attention/FFN/logits stage comparison used the slower GNU/OpenMP reference. It must not select the next
   kernel. The next CPU P0 evidence task is a low-overhead, same-window stage profile against Clang/no-OpenMP, followed
   by a matched GNU/no-OpenMP profile to separate OpenMP overhead from compiler code generation.
6. Any next implementation target must be the largest measured stage deficit under the stronger reference. Existing
   dispatch and FFN-Down ideas remain candidates, not confirmed owners, until that profile is complete.

## Reproduction and Acceptance

The checked-in sweep supports multiple binaries through `--binary NAME=PATH` and selects them per profile with
`binary=NAME`. A cross-runtime result is accepted only when all of the following hold:

- three or more alternating repetitions;
- CV at or below 3% for each runtime;
- byte-identical generated text and identical 9-prompt/15-eval boundaries;
- no LiteNN fallback and an AOT cache hit;
- binary hashes and build metadata captured;
- actual frequency sampled during each process;
- model, prompt, executable, and private absolute paths absent from committed artifacts.
