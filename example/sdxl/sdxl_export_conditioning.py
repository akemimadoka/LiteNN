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


def install_no_pretrained_conditioner_patches() -> None:
    """Instantiate text-encoder structures without fetching pretrained weights.

    The checkpoint supplies the conditioner state. Loading canonical HF/OpenCLIP
    weights first is only a slow bootstrap step and can block on cache/network.
    """
    import inspect

    import open_clip
    from transformers import CLIPTextConfig, CLIPTextModel

    original_create_model_and_transforms = open_clip.create_model_and_transforms
    open_clip_parameters = set(inspect.signature(original_create_model_and_transforms).parameters)

    def create_model_and_transforms_without_weights(model_name: str, *args: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = None
        if "pretrained_text" in open_clip_parameters:
            kwargs["pretrained_text"] = False
        if "load_weights" in open_clip_parameters:
            kwargs["load_weights"] = False
        return original_create_model_and_transforms(model_name, *args, **kwargs)

    def clip_text_model_without_weights(cls: type[Any], version: str, *args: Any, **kwargs: Any) -> Any:
        if version != "openai/clip-vit-large-patch14":
            print(f"Warning: using CLIP ViT-L/14 text config for unsupported version {version!r}")
        config = CLIPTextConfig(
            vocab_size=49408,
            hidden_size=768,
            intermediate_size=3072,
            projection_dim=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            max_position_embeddings=77,
        )
        return cls(config)

    open_clip.create_model_and_transforms = create_model_and_transforms_without_weights
    CLIPTextModel.from_pretrained = classmethod(clip_text_model_without_weights)


def set_conditioner_device(conditioner: Any, device: str) -> None:
    for embedder in getattr(conditioner, "embedders", []):
        if hasattr(embedder, "device"):
            embedder.device = device


def load_conditioner_state(conditioner: Any, checkpoint: Path) -> tuple[list[str], list[str], int]:
    import torch

    if checkpoint.suffix == ".safetensors":
        from safetensors import safe_open

        state: dict[str, Any] = {}
        with safe_open(str(checkpoint), framework="pt", device="cpu") as archive:
            for key in archive.keys():
                if key.startswith("conditioner."):
                    state[key.removeprefix("conditioner.")] = archive.get_tensor(key)
    elif checkpoint.suffix == ".ckpt":
        raw = torch.load(checkpoint, map_location="cpu")
        raw_state = raw.get("state_dict", raw)
        state = {
            key.removeprefix("conditioner."): value
            for key, value in raw_state.items()
            if key.startswith("conditioner.")
        }
    else:
        raise ValueError("checkpoint must be .safetensors or .ckpt")

    missing, unexpected = conditioner.load_state_dict(state, strict=False)
    return list(missing), list(unexpected), len(state)


def load_conditioner(args: argparse.Namespace, config: Any, device: str) -> Any:
    if args.full_model:
        from sgm.util import load_model_from_config

        model = load_model_from_config(config, str(args.checkpoint))
        if model is None:
            raise RuntimeError("generative-models returned no model")
        model.to(device)
        model.eval()
        conditioner = model.conditioner
    else:
        from sgm.util import instantiate_from_config

        if not args.allow_pretrained_download:
            install_no_pretrained_conditioner_patches()
        conditioner_config = config.model.params.conditioner_config
        conditioner = instantiate_from_config(conditioner_config)
        missing, unexpected, loaded = load_conditioner_state(conditioner, args.checkpoint)
        print(
            f"Loaded {loaded} conditioner tensor(s) from checkpoint "
            f"({len(missing)} missing, {len(unexpected)} unexpected)"
        )
        if args.verbose_load_state and missing:
            print("missing conditioner keys:")
            for key in missing:
                print(f"  {key}")
        if args.verbose_load_state and unexpected:
            print("unexpected conditioner keys:")
            for key in unexpected:
                print(f"  {key}")
        conditioner.to(device)
        conditioner.eval()

    set_conditioner_device(conditioner, device)
    for param in conditioner.parameters():
        param.requires_grad = False
    return conditioner


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
        "--full-model",
        action="store_true",
        help="Load the full generative-models pipeline instead of only the conditioner",
    )
    parser.add_argument(
        "--allow-pretrained-download",
        action="store_true",
        help="Allow conditioner-only construction to fetch canonical HF/OpenCLIP pretrained weights before checkpoint load",
    )
    parser.add_argument(
        "--verbose-load-state",
        action="store_true",
        help="Print missing/unexpected keys when loading conditioner-only weights",
    )
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

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device_type = device.split(":", 1)[0]

    config = OmegaConf.load(args.config)
    conditioner = load_conditioner(args, config, device)
    if not args.fp32 and device_type != "cpu":
        conditioner.half()

    value_dict = build_value_dict(args)
    keys = get_unique_embedder_keys_from_conditioner(conditioner)
    batch, batch_uc = get_batch(keys, value_dict, [args.batch], device=device)

    autocast_scope = torch.autocast(device_type) if device_type == "cuda" and not args.fp32 else nullcontext()
    with torch.inference_mode():
        with autocast_scope:
            cond, uncond = conditioner.get_unconditional_conditioning(
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
