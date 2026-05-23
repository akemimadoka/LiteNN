#!/usr/bin/env python3
"""Generate an SDXL reference image through Stability-AI/generative-models.

This is a validation helper for the LiteNN port. It exercises the original
runtime with the same checkpoint and prompt so LiteNN-imported subgraphs can be
compared against a known-good pipeline while the full graph frontend is still
being built.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path


def missing_modules(names: list[str]) -> list[str]:
    return [name for name in names if importlib.util.find_spec(name) is None]


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
    parser.add_argument("--steps", default=30, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--device", default=None, help="Default: cuda when available, otherwise cpu")
    parser.add_argument("--fp32", action="store_true", help="Disable fp16 conditioner/UNet weights")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    sys.path.insert(0, str(args.generative_models))

    missing = missing_modules(
        [
            "einops",
            "kornia",
            "numpy",
            "omegaconf",
            "open_clip",
            "PIL",
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
        print("Cannot run Stability-AI SDXL reference generation; missing Python modules:")
        for name in missing:
            print(f"  - {name}")
        print("Install the generative-models runtime dependencies for python311, then rerun this helper.")
        return 2

    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from PIL import Image
    from sgm.inference.api import Discretization, Guider, Sampler, SamplingParams, get_sampler_config
    from sgm.inference.helpers import do_sample
    from sgm.util import load_model_from_config

    if args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("--height and --width must be divisible by 8")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    config = OmegaConf.load(args.config)
    model = load_model_from_config(config, str(args.checkpoint))
    if model is None:
        raise RuntimeError("generative-models returned no model")
    model.to(device)
    model.eval()
    if not args.fp32:
        model.conditioner.half()
        model.model.half()

    params = SamplingParams(
        width=args.width,
        height=args.height,
        steps=args.steps,
        sampler=Sampler.EULER_EDM,
        discretization=Discretization.EDM,
        guider=Guider.VANILLA,
    )
    value_dict = asdict(params)
    value_dict["prompt"] = args.prompt
    value_dict["negative_prompt"] = args.negative_prompt
    value_dict["target_width"] = args.width
    value_dict["target_height"] = args.height

    with torch.inference_mode():
        samples = do_sample(
            model,
            get_sampler_config(params),
            value_dict,
            1,
            args.height,
            args.width,
            4,
            8,
            force_uc_zero_embeddings=["txt"],
            device=device,
        )

    image = samples[0].detach().float().cpu().permute(1, 2, 0).numpy()
    image = np.clip(image * 255.0 + 0.5, 0, 255).astype(np.uint8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
