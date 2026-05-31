#!/usr/bin/env python3
"""Run one Stability-AI/generative-models SDXL UNet step from LiteNN-style inputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any


def missing_modules(names: list[str]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


def normalize_output_dtype(value: str) -> str:
    normalized = value.upper()
    aliases = {
        "FLOAT32": "F32",
        "TORCH.FLOAT32": "F32",
        "FP32": "F32",
        "FLOAT16": "F16",
        "TORCH.FLOAT16": "F16",
        "FP16": "F16",
        "HALF": "F16",
        "BFLOAT16": "BF16",
        "TORCH.BFLOAT16": "BF16",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"F32", "F16", "BF16"}:
        raise ValueError(f"unsupported output dtype {value!r}; expected F32, F16, or BF16")
    return normalized


def load_safetensors(path: Path) -> tuple[dict[str, Any], memoryview]:
    data = path.read_bytes()
    if len(data) < 8:
        raise ValueError(f"{path} is too small to be a safetensors file")
    header_size = struct.unpack("<Q", data[:8])[0]
    header_end = 8 + header_size
    if header_end > len(data):
        raise ValueError(f"{path} is truncated before payload")
    header = json.loads(data[8:header_end])
    if not isinstance(header, dict):
        raise ValueError(f"{path} safetensors header root must be an object")
    return header, memoryview(data)[header_end:]


def read_tensor(header: dict[str, Any], payload: memoryview, name: str, device: str) -> Any:
    import numpy as np
    import torch

    raw = header.get(name)
    if not isinstance(raw, dict):
        raise KeyError(f"input tensor {name!r} not found")
    dtype = raw.get("dtype")
    shape = raw.get("shape")
    offsets = raw.get("data_offsets")
    if not isinstance(dtype, str):
        raise ValueError(f"input tensor {name!r} has invalid dtype")
    if not isinstance(shape, list) or not all(isinstance(dim, int) and dim >= 0 for dim in shape):
        raise ValueError(f"input tensor {name!r} has invalid shape")
    if not isinstance(offsets, list) or len(offsets) != 2:
        raise ValueError(f"input tensor {name!r} has invalid data_offsets")
    begin, end = offsets
    if not isinstance(begin, int) or not isinstance(end, int) or begin < 0 or end < begin or end > len(payload):
        raise ValueError(f"input tensor {name!r} has out-of-range data_offsets")
    raw_bytes = payload[begin:end]
    if dtype == "F32":
        array = np.frombuffer(raw_bytes, dtype="<f4").reshape(shape)
        return torch.from_numpy(array.copy()).to(device=device)
    if dtype == "F16":
        array = np.frombuffer(raw_bytes, dtype="<f2").reshape(shape)
        return torch.from_numpy(array.copy()).to(device=device)
    if dtype == "BF16":
        values = np.frombuffer(raw_bytes, dtype="<u2").astype(np.uint32)
        array = (values << 16).view(np.float32).reshape(shape)
        return torch.from_numpy(array.copy()).to(device=device, dtype=torch.bfloat16)
    raise ValueError(f"input tensor {name!r} uses unsupported dtype {dtype!r}; expected F32, F16, or BF16")


def tensor_to_payload(tensor: Any, dtype: str) -> tuple[list[int], bytes]:
    import numpy as np
    import torch

    dtype = normalize_output_dtype(dtype)
    if dtype == "F32":
        array = tensor.detach().float().cpu().contiguous().numpy().astype(np.dtype("<f4"), copy=False)
    elif dtype == "F16":
        array = tensor.detach().to(torch.float16).cpu().contiguous().numpy().astype(np.dtype("<f2"), copy=False)
    else:
        array = (
            tensor.detach()
            .to(torch.bfloat16)
            .cpu()
            .contiguous()
            .view(torch.uint16)
            .numpy()
            .astype(np.dtype("<u2"), copy=False)
        )
    return [int(dim) for dim in array.shape], array.tobytes(order="C")


def write_safetensors(path: Path, name: str, tensor: Any, dtype: str, metadata: dict[str, str]) -> None:
    dtype = normalize_output_dtype(dtype)
    shape, payload = tensor_to_payload(tensor, dtype)
    header: dict[str, Any] = {
        "__metadata__": metadata,
        name: {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [0, len(payload)],
        },
    }
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header_bytes)))
        stream.write(header_bytes)
        stream.write(payload)


def ensure_batched_context(context: Any) -> Any:
    if context.ndim == 2:
        return context.unsqueeze(0)
    return context


def ensure_batched_vector(vector: Any) -> Any:
    if vector.ndim == 1:
        return vector.unsqueeze(0)
    return vector


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generative-models", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--inputs", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--latent-tensor", default="latent")
    parser.add_argument("--timestep-tensor", default="timestep")
    parser.add_argument("--context-tensor", default="context")
    parser.add_argument("--vector-tensor", default="vector_cond")
    parser.add_argument("--output-tensor", default="noise_pred")
    parser.add_argument("--output-dtype", default="F32", help="F32, F16, or BF16")
    parser.add_argument("--device", default=None, help="Default: cuda when available, otherwise cpu")
    parser.add_argument("--fp32", action="store_true", help="Disable autocast and keep UNet in float32")
    parser.add_argument("--allow-pretrained-download", action="store_true")
    args = parser.parse_args()

    args.output_dtype = normalize_output_dtype(args.output_dtype)
    sys.path.insert(0, str(args.generative_models))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    missing = missing_modules(
        [
            "einops",
            "imwatermark",
            "numpy",
            "omegaconf",
            "open_clip",
            "pytorch_lightning",
            "safetensors",
            "sgm",
            "timm",
            "torch",
            "torchvision",
            "transformers",
        ]
    )
    if missing:
        print("Cannot run reference SDXL UNet step; missing Python modules:")
        for name in missing:
            print(f"  - {name}")
        print("Install the generative-models runtime dependencies for python311, then rerun this helper.")
        return 2

    import torch
    from omegaconf import OmegaConf
    from sgm.util import load_model_from_config

    if not args.allow_pretrained_download:
        from sdxl_export_conditioning import install_no_pretrained_conditioner_patches

        install_no_pretrained_conditioner_patches()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.split(":", 1)[0]
    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, str(args.checkpoint))
    if model is None:
        raise RuntimeError("generative-models returned no model")
    model.to(device)
    model.eval()
    if not args.fp32 and device_type != "cpu":
        model.model.to(dtype=torch.float16)

    header, payload = load_safetensors(args.inputs)
    latent = read_tensor(header, payload, args.latent_tensor, device)
    timestep = read_tensor(header, payload, args.timestep_tensor, device).reshape(latent.shape[0])
    context = ensure_batched_context(read_tensor(header, payload, args.context_tensor, device))
    vector = ensure_batched_vector(read_tensor(header, payload, args.vector_tensor, device))

    if not args.fp32 and device_type != "cpu":
        latent = latent.to(torch.float16)
        context = context.to(torch.float16)
        vector = vector.to(torch.float16)

    autocast_scope = torch.autocast(device_type) if device_type == "cuda" and not args.fp32 else nullcontext()
    with torch.inference_mode():
        with autocast_scope:
            output = model.model.diffusion_model(latent, timesteps=timestep, context=context, y=vector)

    write_safetensors(
        args.output,
        args.output_tensor,
        output,
        args.output_dtype,
        {
            "litenn.kind": "sdxl_reference_unet_step",
            "litenn.input": str(args.inputs),
            "litenn.output_dtype": args.output_dtype,
        },
    )
    print(
        f"Wrote reference UNet output {args.output}:{args.output_tensor} "
        f"shape={[int(dim) for dim in output.shape]} dtype={args.output_dtype}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
