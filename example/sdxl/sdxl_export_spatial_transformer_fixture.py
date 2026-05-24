#!/usr/bin/env python3
"""Export a real SDXL SpatialTransformer parity fixture for LiteNN.

The fixture uses Stability-AI/generative-models to run one real
SpatialTransformer module with random deterministic inputs. It writes one
LiteNN input-binding safetensors file and one reference-output safetensors file.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from pathlib import Path
from typing import Any


def missing_modules(names: list[str]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def tensor_to_f32_payload(tensor: Any) -> tuple[list[int], bytes]:
    import numpy as np

    array = tensor.detach().float().cpu().contiguous().numpy().astype(np.dtype("<f4"), copy=False)
    return [int(dim) for dim in array.shape], array.tobytes(order="C")


def write_safetensors(path: Path, tensors: dict[str, Any], metadata: dict[str, str]) -> None:
    header: dict[str, Any] = {}
    if metadata:
        header["__metadata__"] = metadata
    payloads: list[bytes] = []
    offset = 0
    for name, tensor in tensors.items():
        shape, payload = tensor_to_f32_payload(tensor)
        end = offset + len(payload)
        header[name] = {"dtype": "F32", "shape": shape, "data_offsets": [offset, end]}
        payloads.append(payload)
        offset = end

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header_bytes)))
        stream.write(header_bytes)
        for payload in payloads:
            stream.write(payload)


def resolve_module(root: Any, dotted_path: str) -> Any:
    current = root
    for part in dotted_path.split("."):
        if part.isdecimal():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generative-models", required=True, type=Path, help="Path to Stability-AI/generative-models")
    parser.add_argument("--config", required=True, type=Path, help="SDXL config YAML")
    parser.add_argument("--checkpoint", required=True, type=Path, help="SDXL safetensors checkpoint")
    parser.add_argument("--inputs-output", required=True, type=Path, help="LiteNN input-binding safetensors output")
    parser.add_argument("--reference-output", required=True, type=Path, help="PyTorch reference safetensors output")
    parser.add_argument(
        "--module",
        default="model.diffusion_model.middle_block.1",
        help="Module path below the loaded generative-models engine",
    )
    parser.add_argument("--height", type=int, default=64, help="Image height used to derive middle-block feature shape")
    parser.add_argument("--width", type=int, default=64, help="Image width used to derive middle-block feature shape")
    parser.add_argument("--feature-height", type=int)
    parser.add_argument("--feature-width", type=int)
    parser.add_argument("--context-tokens", type=int, default=77)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--device", default=None, help="Default: cuda when available, otherwise cpu")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sys.path.insert(0, str(args.generative_models))
    missing = missing_modules(["einops", "numpy", "omegaconf", "safetensors", "sgm", "torch"])
    if missing:
        print("Cannot export SDXL SpatialTransformer fixture; missing Python modules:")
        for name in missing:
            print(f"  - {name}")
        print("Install the generative-models runtime dependencies for python311, then rerun this helper.")
        return 2

    import torch
    from omegaconf import OmegaConf
    from sgm.util import load_model_from_config

    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("--height and --width must be divisible by 8")
    feature_h = args.feature_height if args.feature_height is not None else max(args.height // 32, 1)
    feature_w = args.feature_width if args.feature_width is not None else max(args.width // 32, 1)
    if feature_h <= 0 or feature_w <= 0:
        raise ValueError("feature dimensions must be positive")
    if args.context_tokens <= 0:
        raise ValueError("--context-tokens must be positive")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, str(args.checkpoint))
    if model is None:
        raise RuntimeError("generative-models returned no model")
    model.to(device)
    model.float()
    model.eval()

    module = resolve_module(model, args.module)
    module.to(device)
    module.float()
    module.eval()
    channels = int(getattr(module, "in_channels"))
    context_width = int(module.transformer_blocks[0].attn2.to_k.in_features)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    features = torch.randn((1, channels, feature_h, feature_w), generator=generator, device=device, dtype=torch.float32)
    context = torch.randn((1, args.context_tokens, context_width), generator=generator, device=device, dtype=torch.float32)
    with torch.inference_mode():
        reference = module(features, context=context)

    metadata = {
        "litenn.kind": "sdxl_spatial_transformer_fixture",
        "litenn.module": args.module,
        "litenn.feature_height": str(feature_h),
        "litenn.feature_width": str(feature_w),
        "litenn.context_tokens": str(args.context_tokens),
        "litenn.seed": str(args.seed),
    }
    write_safetensors(args.inputs_output, {"features": features, "context": context[0]}, metadata)
    write_safetensors(args.reference_output, {"features_out": reference}, metadata)
    print(f"Wrote LiteNN inputs: {args.inputs_output}")
    print(f"  features: {[int(dim) for dim in features.shape]}")
    print(f"  context: {[int(dim) for dim in context[0].shape]}")
    print(f"Wrote PyTorch reference: {args.reference_output}")
    print(f"  features_out: {[int(dim) for dim in reference.shape]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
