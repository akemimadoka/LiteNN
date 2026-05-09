"""PyTorch CPU Inference Benchmark

与 bench.cpp（LiteNN）对比：相同的模型结构、批次大小、预热次数和计时次数。

模型：
  Linear      784 → 10
  MLP-128     784 → 128 → ReLU → 10
  MLP-512     784 → 512 → ReLU → 256 → ReLU → 10

批次大小：1 / 32 / 128 / 512；自适应迭代次数（目标计时约 2 秒）。

运行方式（需要 Python 3.11 + PyTorch CPU）：
  python311 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  python311 benchmark/bench.py
  python311 benchmark/bench.py --threads 1
"""
from __future__ import annotations

import argparse
import sys
import time

try:
    import torch
    import torch.nn as nn
except ImportError:
    print("PyTorch not found. Install with:")
    print("  pip install torch --index-url https://download.pytorch.org/whl/cpu")
    sys.exit(1)

WARMUP      = 5
BATCH_SIZES = [1, 32, 128, 512]
TARGET_S    = 2.0   # target measurement time per configuration


def make_linear() -> nn.Module:
    return nn.Sequential(nn.Linear(784, 10))


def make_mlp128() -> nn.Module:
    return nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 10),
    )


def make_mlp512() -> nn.Module:
    return nn.Sequential(
        nn.Linear(784, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 10),
    )


def bench(model: nn.Module, batch: int, warmup: int = WARMUP) -> tuple[float, float]:
    """Returns (mean_ms_per_batch, throughput_samples_per_sec)."""
    model.eval()
    x = torch.randn(batch, 784)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        # Probe: 1 iteration to estimate per-call cost
        t_probe = time.perf_counter()
        model(x)
        probe_ms = (time.perf_counter() - t_probe) * 1000.0
        iters = int(max(10, min(1000, TARGET_S * 1000.0 / max(probe_ms, 0.001))))
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        t1 = time.perf_counter()
    total_ms  = (t1 - t0) * 1000.0
    mean_ms   = total_ms / iters
    throughput = batch * iters / (total_ms * 1e-3)
    return mean_ms, throughput


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch CPU inference benchmark")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Set both torch intra-op and inter-op thread counts before benchmarking.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError("--threads must be greater than 0")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    print("PyTorch CPU Inference Benchmark")
    print(
        f"PyTorch {torch.__version__}  |  device: CPU"
        f"  |  threads: {torch.get_num_threads()}/{torch.get_num_interop_threads()}"
    )
    print("=" * 72)
    print(f"{'Model':<24} {'Batch':>6} {'ms/batch':>14} {'samples/sec':>14}")
    print("-" * 72)

    models = [
        ("Linear(784->10)",        make_linear()),
        ("MLP(784->128->10)",      make_mlp128()),
        ("MLP(784->512->256->10)", make_mlp512()),
    ]

    for batch in BATCH_SIZES:
        for name, model in models:
            mean_ms, tput = bench(model, batch)
            print(f"{name:<24} {batch:>6} {mean_ms:>12.3f}ms {tput:>12.0f}/s")
        if batch != BATCH_SIZES[-1]:
            print()

    print("=" * 72)


if __name__ == "__main__":
    main()
