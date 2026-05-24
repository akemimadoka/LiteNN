#!/usr/bin/env python3
"""Export SDXL prompt conditioning tensors for LiteNN AOT input binding.

This bridge intentionally reuses Stability-AI/generative-models for tokenizer
and text-encoder execution. LiteNN can then bind the exported safetensors file
into compiled smoke graphs or future fixed-shape UNet artifacts.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import struct
import sys
from contextlib import nullcontext
from dataclasses import asdict
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
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [offset, end],
        }
        payloads.append(payload)
        offset = end

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.write(struct.pack("<Q", len(header_bytes)))
        stream.write(header_bytes)
        for payload in payloads:
            stream.write(payload)


def maybe_squeeze_batch(tensor: Any, squeeze: bool) -> Any:
    if squeeze and getattr(tensor, "ndim", 0) == 3 and tensor.shape[0] == 1:
        return tensor[0]
    return tensor


def build_value_dict(args: argparse.Namespace) -> dict[str, Any]:
    from sgm.inference.api import Discretization, Guider, Sampler, SamplingParams

    params = SamplingParams(
        width=args.width,
        height=args.height,
        steps=1,
        sampler=Sampler.EULER_EDM,
        discretization=Discretization.EDM,
        guider=Guider.VANILLA,
        scale=args.cfg_scale,
        aesthetic_score=args.aesthetic_score,
        negative_aesthetic_score=args.negative_aesthetic_score,
        orig_width=args.orig_width or args.width,
        orig_height=args.orig_height or args.height,
        crop_coords_top=args.crop_top,
        crop_coords_left=args.crop_left,
    )
    value_dict = asdict(params)
    value_dict["prompt"] = args.prompt
    value_dict["negative_prompt"] = args.negative_prompt
    value_dict["target_width"] = args.target_width or args.width
    value_dict["target_height"] = args.target_height or args.height
    return value_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generative-models", required=True, type=Path, help="Path to Stability-AI/generative-models")
    parser.add_argument("--config", required=True, type=Path, help="SDXL config YAML")
    parser.add_argument("--checkpoint", required=True, type=Path, help="SDXL safetensors checkpoint")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--height", default=1024, type=int)
    parser.add_argument("--width", default=1024, type=int)
    parser.add_argument("--batch", default=1, type=int)
    parser.add_argument("--device", default=None, help="Default: cuda when available, otherwise cpu")
    parser.add_argument("--fp32", action="store_true", help="Disable conditioner fp16 execution on CUDA")
    parser.add_argument("--cfg-scale", default=6.0, type=float)
    parser.add_argument("--aesthetic-score", default=5.0, type=float)
    parser.add_argument("--negative-aesthetic-score", default=5.0, type=float)
    parser.add_argument("--orig-height", type=int)
    parser.add_argument("--orig-width", type=int)
    parser.add_argument("--target-height", type=int)
    parser.add_argument("--target-width", type=int)
    parser.add_argument("--crop-top", default=0, type=int)
    parser.add_argument("--crop-left", default=0, type=int)
    parser.add_argument("--keep-context-batch", action="store_true", help="Keep cross-attention as [B,T,C]")
    parser.add_argument(
        "--force-uc-zero-embedding",
        action="append",
        default=["txt"],
        help="Embedder input key to zero in unconditional conditioning; repeatable",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("--height and --width must be divisible by 8")
    if args.batch <= 0:
        raise ValueError("--batch must be positive")

    sys.path.insert(0, str(args.generative_models))
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
        print("Cannot export SDXL conditioning; missing Python modules:")
        for name in missing:
            print(f"  - {name}")
        print("Install the generative-models runtime dependencies for python311, then rerun this helper.")
        return 2

    import torch
    from omegaconf import OmegaConf
    from sgm.inference.helpers import get_batch, get_unique_embedder_keys_from_conditioner
    from sgm.util import load_model_from_config

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.split(":", 1)[0]

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, str(args.checkpoint))
    if model is None:
        raise RuntimeError("generative-models returned no model")
    model.to(device)
    model.eval()
    if not args.fp32 and device_type != "cpu":
        model.conditioner.half()

    value_dict = build_value_dict(args)
    keys = get_unique_embedder_keys_from_conditioner(model.conditioner)
    batch, batch_uc = get_batch(keys, value_dict, [args.batch], device=device)

    autocast_scope = torch.autocast(device_type) if device_type == "cuda" and not args.fp32 else nullcontext()
    with torch.inference_mode():
        with autocast_scope:
            with model.ema_scope():
                cond, uncond = model.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=args.force_uc_zero_embedding,
                )

    output_tensors: dict[str, Any] = {}
    for key in sorted(cond):
        if torch.is_tensor(cond[key]):
            output_tensors[f"cond.{key}"] = cond[key]
        if key in uncond and torch.is_tensor(uncond[key]):
            output_tensors[f"uncond.{key}"] = uncond[key]

    if "crossattn" in cond:
        squeeze_context = not args.keep_context_batch
        output_tensors["context"] = maybe_squeeze_batch(cond["crossattn"], squeeze_context)
        output_tensors["negative_context"] = maybe_squeeze_batch(uncond["crossattn"], squeeze_context)
        output_tensors["context_cfg"] = torch.cat((uncond["crossattn"], cond["crossattn"]), dim=0)
    if "vector" in cond:
        output_tensors["vector_cond"] = cond["vector"]
        output_tensors["negative_vector_cond"] = uncond["vector"]
        output_tensors["vector_cond_cfg"] = torch.cat((uncond["vector"], cond["vector"]), dim=0)
    if "concat" in cond:
        output_tensors["concat_cond"] = cond["concat"]
        output_tensors["negative_concat_cond"] = uncond["concat"]
        output_tensors["concat_cond_cfg"] = torch.cat((uncond["concat"], cond["concat"]), dim=0)

    metadata = {
        "litenn.kind": "sdxl_conditioning",
        "litenn.prompt": args.prompt,
        "litenn.negative_prompt": args.negative_prompt,
        "litenn.height": str(args.height),
        "litenn.width": str(args.width),
        "litenn.batch": str(args.batch),
        "litenn.context_batch_squeezed": str(not args.keep_context_batch),
    }
    write_safetensors(args.output, output_tensors, metadata)
    print(f"Wrote {args.output} with {len(output_tensors)} F32 tensor(s)")
    for name, tensor in output_tensors.items():
        print(f"  {name}: {[int(dim) for dim in tensor.shape]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
