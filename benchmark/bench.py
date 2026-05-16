"""PyTorch CPU/CUDA Inference Benchmark

与 bench.cpp（LiteNN）对比：相同的模型结构、批次大小、预热次数和计时次数。

模型：
  Linear      784 → 10
  MLP-128     784 → 128 → ReLU → 10
  MLP-512     784 → 512 → ReLU → 256 → ReLU → 10

批次大小：1 / 32 / 128 / 512；自适应迭代次数（目标计时约 2 秒）。

运行方式（CPU）：
  python311 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
  python311 benchmark/bench.py
  python311 benchmark/bench.py --threads 1

运行方式（CUDA，可选）：
    python311 -m pip install torch --index-url https://download.pytorch.org/whl/cu128
    python311 benchmark/bench.py --device cuda

如果请求 CUDA 但当前 PyTorch/驱动/设备不支持，脚本会打印 skip 信息并正常退出。
"""
from __future__ import annotations

import argparse
import sys
import time

try:
    import torch
    import torch.nn as nn
except ImportError as exc:
    print("PyTorch import failed:")
    print(f"  {exc}")
    print("Install or repair PyTorch with:")
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


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def bench(model: nn.Module, batch: int, device: torch.device, warmup: int = WARMUP) -> tuple[float, float]:
    """Returns (mean_ms_per_batch, throughput_samples_per_sec)."""
    model = model.to(device)
    model.eval()
    x = torch.randn(batch, 784, device=device)
    with torch.inference_mode():
        for _ in range(warmup):
            model(x)
        synchronize(device)
        # Probe: 1 iteration to estimate per-call cost
        synchronize(device)
        t_probe = time.perf_counter()
        model(x)
        synchronize(device)
        probe_ms = (time.perf_counter() - t_probe) * 1000.0
        iters = int(max(10, min(1000, TARGET_S * 1000.0 / max(probe_ms, 0.001))))
        synchronize(device)
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        synchronize(device)
        t1 = time.perf_counter()
    total_ms  = (t1 - t0) * 1000.0
    mean_ms   = total_ms / iters
    throughput = batch * iters / (total_ms * 1e-3)
    return mean_ms, throughput


def format_device_label(device: torch.device) -> str:
    if device.type == "cuda":
        return f"CUDA:{device.index} ({torch.cuda.get_device_name(device)})"
    return "CPU"


def get_cuda_skip_reason(cuda_device: int) -> str | None:
    if not torch.cuda.is_available():
        return "CUDA is not available in this PyTorch build or no CUDA device is visible."
    if cuda_device < 0:
        return "--cuda-device must be greater than or equal to 0"
    if cuda_device >= torch.cuda.device_count():
        return f"requested CUDA device {cuda_device} is out of range (count={torch.cuda.device_count()})"
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PyTorch CPU/CUDA inference benchmark")
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="Set both torch intra-op and inter-op thread counts before benchmarking.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda", "all"),
        default="all",
        help="Select which device benchmarks to run. Default: all.",
    )
    parser.add_argument(
        "--cuda-device",
        type=int,
        default=0,
        help="CUDA device index to benchmark when --device includes cuda.",
    )
    return parser.parse_args()


def run_device_benchmarks(device: torch.device) -> None:
    print(f"[{format_device_label(device)}]")
    print("=" * 72)
    print(f"{'Model':<24} {'Batch':>6} {'ms/batch':>14} {'samples/sec':>14}")
    print("-" * 72)

    models = [
        ("Linear(784->10)",        make_linear),
        ("MLP(784->128->10)",      make_mlp128),
        ("MLP(784->512->256->10)", make_mlp512),
    ]

    for batch in BATCH_SIZES:
        for name, build_model in models:
            mean_ms, tput = bench(build_model(), batch, device)
            print(f"{name:<24} {batch:>6} {mean_ms:>12.3f}ms {tput:>12.0f}/s")
        if batch != BATCH_SIZES[-1]:
            print()

    print("=" * 72)


def main() -> None:
    args = parse_args()
    if args.threads is not None:
        if args.threads <= 0:
            raise ValueError("--threads must be greater than 0")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(args.threads)

    print("PyTorch Inference Benchmark")
    print(
        f"PyTorch {torch.__version__}"
        f"  |  threads: {torch.get_num_threads()}/{torch.get_num_interop_threads()}"
    )

    if args.device in ("cpu", "all"):
        run_device_benchmarks(torch.device("cpu"))

    if args.device in ("cuda", "all"):
        skip_reason = get_cuda_skip_reason(args.cuda_device)
        if skip_reason is not None:
            print(f"[skip] CUDA benchmark skipped: {skip_reason}")
        else:
            if args.device == "all":
                print()
            run_device_benchmarks(torch.device("cuda", args.cuda_device))


if __name__ == "__main__":
    main()
