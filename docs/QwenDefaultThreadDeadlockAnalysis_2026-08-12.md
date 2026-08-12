# Qwen Default-Thread Deadlock Analysis (2026-08-12)

## Scope

This report closes an intermittent Windows deadlock in the default-thread CPU AOT Qwen diagnostic loop. The failure
was first observed after three normal steps and was later reproduced at step 27. Explicit T8 runs completed, while the
default policy resolved to the 32 logical processors available on the test system and exercised more worker states.

No machine-specific model or artifact path is retained here.

## Reproduction

The controlled workload used a Qwen2.5-Coder 14B Q4_K_M model, stateful paged-reference decode, a fixed 24-token
trajectory after a nine-token prompt, LLVM O0, strict activation math, adaptive worker wait, eight selected checkpoint
positions, and eight selected blocks. This diagnostic ABI exposed 153 outputs per step.

Two direct invocations completed all 32 forwards before the failure was reproduced. They measured `446.668` and
`397.957 ms/generated token`, produced no fallback, and wrote the same block-47 index-23 checksum as explicit T8. This
demonstrates why a single successful run was not a sufficient concurrency gate.

Running the same workload through `example/gguf/qwen_smoke.py`, including subprocess pipes and 100 ms memory sampling,
reproduced the stall at step 27. The last 26 steps had completed normally at roughly `376-449 ms` each. The process
continued consuming CPU but made no checkpoint or log progress.

## Live Stack Evidence

GDB was attached to the live stalled process without terminating it first:

- the main thread was spinning in `LiteNNCPUThreadPool::ParallelFor`;
- its requested worker count was 31;
- `workersDone_` remained at 30;
- all 31 worker threads were blocked in `std::binary_semaphore::acquire`;
- the caller was a grouped mixed GGML projection helper.

The failure is therefore a lost worker wakeup at the shared thread-pool barrier. It is not a slow projection, model
state error, Python pipe backpressure, or memory sampler pause.

## Initial Root Cause And First Fix

The worker generation and sleeping state were atomic, but blocking used a separate binary semaphore. The protocol
attempted to transfer ownership of a semaphore permit through `sleeping.exchange(false)`. Under an unlucky transition
between active polling and blocking, one selected worker could observe neither usable work nor a consumable permit.
The caller then waited forever for exact equality between 31 requested and 30 completed workers.

The first fix removed the binary semaphore and waited directly on the worker's generation atomic:

1. the dispatcher increments `generation` with release ordering;
2. it calls `generation.notify_one()` only when the worker reports sleeping;
3. the worker calls `generation.wait(observedGeneration)` while the value remains unchanged;
4. a notification that races ahead of the actual wait is safe because `wait(oldValue)` immediately observes the new
   generation instead of requiring a separately retained permit.

The adaptive polling and sleeping-worker notification policy remained intact. The initial validation below passed,
but a longer independent campaign later disproved this repair as a complete production fix.

## Validation

| Gate | Result |
| --- | --- |
| Rapid participant-count regression | 4096 calls per process, alternating auto/T16/T4/T8/T2/auto |
| Wait-policy coverage | adaptive, low-power, and latency |
| Repeated stress | 5 processes, 20,480 total calls, all passed |
| CPU parallel regression suite | 8/8 passed |
| Quantized and grouped-attention tests | 22/22 passed |
| Real wrapper rerun | 32/32 forwards, no fallback, `431.876 ms/generated token` |
| Numerical identity | block-47 index-23 checksum `e63f6bb0a81ad20c`, unchanged from successful controls |
| Peak resident memory | 18,811,240,448 bytes, consistent with the established diagnostic build envelope |

The post-fix grouped-attention microbenchmark retained exact output and measured T8 medians of `0.040 ms` at context
128 and `0.690 ms` at context 2048. These do not regress the previous `0.048/0.956 ms` control observations.

## Follow-up Reproduction And Superseding Fix

A later three-prompt natural-generation campaign reproduced the same barrier failure in its third process at forward
step 45. Live GDB evidence again found `desiredWorkers=31`, `workersDone=30`, the caller spinning in `ParallelFor`, and
all worker threads blocked through libwinpthread. This invalidates the claim that generation-only atomic wait/notify
was sufficient on the deployed Windows/MinGW runtime.

The superseding protocol retains adaptive user-space polling but uses a per-worker mutex and condition variable for
the blocking transition. A worker publishes `sleeping`, acquires its wait mutex, and sleeps on the predicate
`generation != observedGeneration || stopping`. A dispatcher that observes a sleeping worker acquires the same mutex
before notification. Therefore every race has one of two outcomes: the worker observes the changed generation before
sleeping, or the notifier is ordered after the worker's atomic unlock-and-wait transition.

The superseding repair passed five concurrent 4096-call mixed-participant stress processes, all 8 CPU-parallel tests,
and all 22 focused quantized/attention tests. The previously stalled real case completed 103 forwards, and a complete
second 192-token natural-generation campaign finished under the default 32-thread policy. Grouped-attention T8
medians remained `0.045 ms` at context 128 and `0.664 ms` at context 2048 with exact output.

## Conclusion

The first repair is explicitly superseded. The default-thread diagnostic P0 is closed against the mutex/condition-
variable protocol, which now has two captured real failure sites, a targeted synchronization repair, concurrent stress,
focused suites, a 103-forward retry, a complete multi-process campaign, and a performance gate. Explicit T8 remains a
useful reproducibility control, but is no longer required to avoid the observed lost-participant barrier.
