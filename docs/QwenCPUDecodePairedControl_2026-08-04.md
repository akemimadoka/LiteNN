# Qwen CPU Decode Paired Control - 2026-08-04

This document records the first repository-owned, paired and alternating actual-completion comparison between LiteNN
CPU AOT and a local CPU-only llama.cpp build. It is the evidence owner for the variance-gated result; implementation
priorities derived from it live in `PerformanceOptimizationRoadmap.md` and the canonical G16.7 checklist lives in
`Roadmap.md`.

## Question

The earlier prompt-aligned controls put LiteNN slightly ahead of the local llama.cpp median, but LiteNN had one slow
run and the runtimes were measured in separate brackets. This experiment asks whether the result survives:

- alternating runtime order;
- an identical Qwen chat template and generated token sequence;
- the same 15-call steady decode boundary;
- explicit no-fallback and text-parity gates;
- host power-policy and processor-frequency sampling; and
- a per-runtime coefficient-of-variation limit of 3%.

It does not attempt to explain or supersede the independently observed `6.85 t/s` llama.cpp result, whose exact binary
and runtime configuration are not yet available.

## Environment

- Host: AMD Ryzen 9 9950X, 32 logical CPUs, Windows 10 build 26200.
- Power plan: Windows Balanced for every sampled process boundary.
- Model class: Qwen2.5-Coder 14B Instruct Q4_K_M GGUF. The private model path is not retained.
- Prompt: SHA-256 `2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824`, formatted by
  the Qwen chat template into 9 prompt tokens.
- Decode: 16 requested generated tokens. llama.cpp reports 15 eval calls after charging the first generated token to
  prompt evaluation; LiteNN uses the matching 15 post-first-generation `module_run_ms` samples.
- LiteNN: Release CPU AOT cache hit, LLVM optimization level 0, T8 adaptive workers, default affinity,
  field-interleaved-v4 prepared Q4_K/Q6_K weights, no fallback.
- LiteNN executable SHA-256: `515fe267133d8f32623d7afeebe9b762e7c05e8c979d009ec33e718e22f97254`.
- llama.cpp: commit `b81c2cdd7`, Release, GNU 16.1, T2, CPU-only, flash attention off, F16 KV, mmap, repack, warmup,
  polling 50, normal priority, greedy sampling, and EOS suppression.
- llama.cpp executable SHA-256: `e0e0ea64c69656d08b11c1a9bfbf9ea5ef1843fc2e82e923978d5dc0aeb35271`.
- llama.cpp build metadata reports `GGML_NATIVE=ON`, `GGML_LLAMAFILE=ON`, `GGML_OPENMP=ON`, CUDA and BLAS off. Its
  runtime ISA report includes AVX2, F16C, FMA, AVX-512, AVX-512 VNNI, and AVX-512 BF16.

The command shape was:

```text
python311 benchmark/run_paired_gguf_decode_control.py \
  --model <model.gguf> \
  --litenn <litenn_gguf_convert> \
  --llamacpp-tokenizer-tool <litenn_llamacpp_adapter> \
  --llama-completion <llama-completion> \
  --aot-cache-dir <cache> \
  --output-dir <output> \
  --prompt hello --predict 16 --repetitions 3 \
  --litenn-threads 8 --llama-threads 2 \
  --variance-threshold-percent 3
```

Odd pairs run llama.cpp then LiteNN; even pairs reverse that order. Two independent three-pair batches were executed.
Commands, model and cache paths, and raw logs are redacted in the repository-level artifacts.

## Results

### Batch A

| Pair | Order | llama.cpp | LiteNN | Paired LiteNN difference |
| ---: | --- | ---: | ---: | ---: |
| 1 | llama.cpp -> LiteNN | `5.510 t/s` | `5.696 t/s` | `+3.37%` |
| 2 | LiteNN -> llama.cpp | `5.420 t/s` | `5.532 t/s` | `+2.07%` |
| 3 | llama.cpp -> LiteNN | `5.490 t/s` | `5.624 t/s` | `+2.45%` |
| Median | alternating | `5.490 t/s` | `5.624 t/s` | `+2.45%` paired median |

llama.cpp CV was `0.86%`; LiteNN CV was `1.46%`. Both passed the 3% variance gate.

### Batch B

| Pair | Order | llama.cpp | LiteNN | Paired LiteNN difference |
| ---: | --- | ---: | ---: | ---: |
| 1 | llama.cpp -> LiteNN | `5.450 t/s` | `5.546 t/s` | `+1.76%` |
| 2 | LiteNN -> llama.cpp | `5.510 t/s` | `5.649 t/s` | `+2.51%` |
| 3 | llama.cpp -> LiteNN | `5.450 t/s` | `5.540 t/s` | `+1.64%` |
| Median | alternating | `5.450 t/s` | `5.546 t/s` | `+1.76%` paired median |

llama.cpp CV was `0.63%`; LiteNN CV was `1.10%`. Both passed the 3% variance gate.

### Combined View

| Runtime | Runs | Median latency | Median throughput | Throughput range | CV |
| --- | ---: | ---: | ---: | ---: | ---: |
| llama.cpp CPU-only | 6 | `182.895 ms/token` | `5.470 t/s` | `5.420-5.510 t/s` | `0.68%` |
| LiteNN CPU AOT | 6 | `179.052 ms/token` | `5.585 t/s` | `5.532-5.696 t/s` | `1.22%` |

The six paired differences range from `+1.64%` to `+3.37%`; their median is `+2.26%`. All six pairs generated the
same 68 UTF-8 bytes, all used a 9-prompt/15-eval window, and no LiteNN fallback occurred.

The Windows processor-power API reported 4300 MHz for all logical CPUs in every sample. This confirms that the API's
reported policy frequency did not change between runtimes. It is not an effective-cycle or residency measurement on
this Ryzen host and therefore cannot rule out short boost, cache, or memory-stall differences. PMU or platform-profiler
evidence remains necessary for that claim.

## Conclusions

1. The earlier local parity result is reproducible. Under this aligned short-decode boundary, LiteNN CPU AOT is not
   behind the bundled llama.cpp build and has a small `1.6-3.4%` paired lead.
2. The lead is not large enough to justify a new kernel solely from end-to-end noise. Both batches pass the runtime CV
   gate, but their paired medians differ (`+2.45%` versus `+1.76%`) and the test covers only 15 steady calls.
3. The current local llama.cpp binary is not the unresolved target. The external `6.85 t/s` observation is `25.23%`
   above the combined local llama.cpp median and `22.64%` above the combined LiteNN median. Its exact compiler, commit,
   ISA, thread affinity, polling, context, and command must be reproduced before selecting another CPU P0 kernel.
4. Process wall time is not a decode comparison. The LiteNN command also performs metadata import, tokenization,
   detokenization, report generation, and subprocess orchestration; the headline deliberately compares only matched
   steady runtime boundaries.
5. Short-context parity does not establish long-context readiness. Longer generation windows and 2K/32K/128K/1M
   context tiers must measure paged-KV growth, attention scaling, memory residency, and sustained frequency separately.

## Planning Decisions

- P0: reproduce the exact `6.85 t/s` external configuration through the paired runner. If the gap persists, capture
  matched Attention/FFN/logits boundaries and promote only the largest measured deficit.
- P1: extend paired controls to sustained 128- and 512-token generation windows, then add context tiers as paged-KV
  capacity becomes production-ready. Keep text parity, no fallback, cache hit, and variance gates mandatory.
- P1: add optional effective-cycle, LLC-miss, memory-stall, and bandwidth sampling. Treat the Windows 4300 MHz policy
  sample as metadata rather than proof of equal effective clocks.
- P2: retain cold-stream and helper microbenchmarks as diagnostics, but do not promote a microkernel from a cache-hot
  win unless paired full-decode and cold-stream evidence improve together.
