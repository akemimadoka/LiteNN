# llama.cpp CPU Stage Profile

This tool measures coarse CPU decode stages without using `cb_eval`. The callback path synchronizes the backend at
every selected tensor and is retained only for historical/cumulative-cut diagnostics. The aggregate path applies a
benchmark-only patch to a detached llama.cpp worktree and records stage transitions on thread 0 after ggml's existing
node barriers. It does not add a scheduler synchronization.

## Prepare An Instrumented Source Tree

```powershell
python311 benchmark\llama_cpp_stage_profile\prepare_instrumented_source.py `
  --source third_party\llama.cpp `
  --worktree build\llama-stage-instrumented-source
```

The preparation command is idempotent. It refuses to reuse an unrelated directory or a worktree at a different
revision. The main `third_party/llama.cpp` checkout remains clean.

Build a clean and an instrumented llama.cpp with identical compiler, target, sysroot, ISA, and OpenMP settings. On the
Windows Clang/MinGW control used by the Qwen benchmark, both `CMAKE_*_COMPILER_TARGET` and the MinGW sysroot are part of
the performance identity:

```powershell
cmake -S build\llama-stage-instrumented-source -B build\llamacpp-clang-stage-counters -G Ninja `
  -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF -DGGML_OPENMP=OFF -DGGML_NATIVE=ON `
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF -DLLAMA_BUILD_SERVER=OFF `
  -DCMAKE_C_COMPILER=clang.exe -DCMAKE_CXX_COMPILER=clang++.exe `
  -DCMAKE_C_COMPILER_TARGET=x86_64-w64-windows-gnu `
  -DCMAKE_CXX_COMPILER_TARGET=x86_64-w64-windows-gnu `
  "-DCMAKE_C_FLAGS=--sysroot=C:/msys64/mingw64 -D_WIN32_WINNT=0x0A00" `
  "-DCMAKE_CXX_FLAGS=--sysroot=C:/msys64/mingw64 -D_WIN32_WINNT=0x0A00"
cmake --build build\llamacpp-clang-stage-counters --target llama --parallel
```

Configure this profiler once against the clean libraries and once against the instrumented libraries. Its CMake file
detects the aggregate-counter ABI from the selected source tree; clean builds keep the baseline and callback modes.

## Run The Strict Gate

Use a clean profiler as `--baseline-binary` and an instrumented profiler as `--binary`. Exact replay performs one
prompt prefill and then measures the supplied decode tokens. Token values are hashed and redacted in saved process
metadata.

```powershell
python311 benchmark\run_llama_cpp_stage_control.py `
  --model <model.gguf> `
  --baseline-binary clang-noopenmp=<clean-profiler.exe> `
  --binary clang-noopenmp=<instrumented-profiler.exe> `
  --mode aggregate --threads 8 --warmup 0 --steps 15 --repetitions 3 `
  --prefill-token-ids <comma-separated-prompt-token-ids> `
  --decode-token-ids <comma-separated-decode-token-ids> `
  --overhead-threshold-percent 3 --stage-variance-threshold-percent 15 `
  --output-json build\stage-control\control.json `
  --output-md build\stage-control\control.md
```

Acceptance requires complete aggregate stage shape, `95-102%` stage coverage, at most `3%` whole-token overhead,
at most `3%` whole-run CV, and at most `15%` CV for every promoted stage. Raw artifacts belong under ignored build or
benchmark-output directories; model paths must not be committed.
