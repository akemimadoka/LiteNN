#!/usr/bin/env python3
"""Run the current LiteNN SDXL prompt-to-image harness.

The default path is intentionally a small 64x64 smoke pipeline. It exports
prompt conditioning with Stability-AI/generative-models, compiles LiteNN UNet
and VAE carrier DLL/SO artifacts, runs Euler denoising, decodes the latent, and
writes a PNG. Use --unet-probe unet-full-fixed with larger dimensions for the
complete fixed-shape graph once compile time and memory are acceptable.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path


def run_step(name: str, command: list[str], *, dry_run: bool = False) -> None:
    print(f"\n== {name} ==", flush=True)
    print(" ".join(command), flush=True)
    if dry_run:
        return
    start = time.perf_counter()
    completed = subprocess.run(command, text=True)
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(f"{name} failed with exit code {completed.returncode}")
    print(f"{name}: ok {elapsed * 1000.0:.3f} ms", flush=True)


def link_command(cxx: str, obj: Path, library: Path, def_file: Path | None) -> list[str]:
    if os.name == "nt":
        if def_file is None:
            raise ValueError("Windows linking requires a .def file")
        return [cxx, "-shared", str(obj), str(def_file), "-o", str(library)]
    return [cxx, "-shared", str(obj), "-o", str(library)]


def library_path(workdir: Path, name: str) -> Path:
    if os.name == "nt":
        return workdir / f"{name}.dll"
    return workdir / f"lib{name}.so"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exe", required=True, type=Path, help="Path to litenn_sdxl_example")
    parser.add_argument("--generative-models", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--workdir", required=True, type=Path)
    parser.add_argument("--output-png", type=Path)
    parser.add_argument("--height", default=64, type=int)
    parser.add_argument("--width", default=64, type=int)
    parser.add_argument("--steps", default=2, type=int)
    parser.add_argument("--seed", default=1234, type=int)
    parser.add_argument("--cfg-scale", default=6.0, type=float)
    parser.add_argument("--sigma-max", default=14.6146, type=float)
    parser.add_argument("--sigma-min", default=0.0292, type=float)
    parser.add_argument("--rho", default=3.0, type=float)
    parser.add_argument("--scheduler", choices=["linear", "edm"], default="edm")
    parser.add_argument("--denoiser-contract", choices=["epsilon", "denoised", "sgm-edm", "sgm-eps", "sgm-v"], default="sgm-edm")
    parser.add_argument("--cfg-mode", choices=["auto", "none", "dual"], default="dual")
    parser.add_argument("--unet-probe", choices=["unet-conditioning-smoke", "unet-full-fixed"], default="unet-conditioning-smoke")
    parser.add_argument("--context-tokens", default=77, type=int)
    parser.add_argument("--vae-mid-attention-policy", choices=["auto", "force", "skip"], default="skip")
    parser.add_argument("--vae-attention-max-mib", default=512, type=int)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--unet-symbol-prefix", default="litenn_sdxl_unet")
    parser.add_argument("--vae-symbol-prefix", default="litenn_sdxl_vae")
    parser.add_argument("--allow-pretrained-download", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.height <= 0 or args.width <= 0 or args.height % 8 != 0 or args.width % 8 != 0:
        raise ValueError("--height and --width must be positive and divisible by 8")
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.vae_attention_max_mib <= 0:
        raise ValueError("--vae-attention-max-mib must be positive")

    args.workdir.mkdir(parents=True, exist_ok=True)
    output_png = args.output_png or (args.workdir / "image.png")
    conditioning = args.workdir / "conditioning.safetensors"
    final_latent = args.workdir / "final_latent.safetensors"
    decoded = args.workdir / "decoded_image.safetensors"
    unet_manifest = args.workdir / "unet_manifest.json"
    vae_manifest = args.workdir / "vae_manifest.json"
    unet_graph = args.workdir / "unet.ltnn"
    vae_graph = args.workdir / "vae.ltnn"
    unet_obj = args.workdir / "unet.obj"
    vae_obj = args.workdir / "vae.obj"
    unet_library = library_path(args.workdir, "unet")
    vae_library = library_path(args.workdir, "vae")
    unet_def = args.workdir / f"{args.unet_symbol_prefix}_exports.def" if os.name == "nt" else None
    vae_def = args.workdir / f"{args.vae_symbol_prefix}_exports.def" if os.name == "nt" else None

    scripts_dir = Path(__file__).resolve().parent
    conditioning_command = [
        args.python,
        str(scripts_dir / "sdxl_export_conditioning.py"),
        "--generative-models",
        str(args.generative_models),
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--prompt",
        args.prompt,
        "--negative-prompt",
        args.negative_prompt,
        "--height",
        str(args.height),
        "--width",
        str(args.width),
        "--cfg-scale",
        str(args.cfg_scale),
        "--output",
        str(conditioning),
    ]
    if args.allow_pretrained_download:
        conditioning_command.append("--allow-pretrained-download")

    common_probe = [
        args.python,
        str(scripts_dir / "sdxl_manifest_probe.py"),
        "--config",
        str(args.config),
        "--safetensors",
        str(args.checkpoint),
        "--height",
        str(args.height),
        "--width",
        str(args.width),
    ]

    steps = [
        ("export-conditioning", conditioning_command),
        (
            "emit-unet-manifest",
            common_probe
            + [
                "--probe",
                args.unet_probe,
                "--context-tokens",
                str(args.context_tokens),
                "--emit-probe-manifest",
                str(unet_manifest),
            ],
        ),
        ("import-unet", [str(args.exe), "--import", str(unet_manifest), str(args.checkpoint), str(unet_graph), "--allow-extra-tensors"]),
        ("compile-unet", [str(args.exe), "--compile-object", str(unet_graph), str(unet_obj), args.unet_symbol_prefix]),
        ("link-unet", link_command(args.cxx, unet_obj, unet_library, unet_def)),
        (
            "denoise-latent",
            [
                str(args.exe),
                "--denoise-latent",
                str(unet_library),
                str(conditioning),
                str(final_latent),
                args.unet_symbol_prefix,
                "--steps",
                str(args.steps),
                "--seed",
                str(args.seed),
                "--scheduler",
                args.scheduler,
                "--sigma-max",
                str(args.sigma_max),
                "--sigma-min",
                str(args.sigma_min),
                "--rho",
                str(args.rho),
                "--denoiser-contract",
                args.denoiser_contract,
                "--cfg-mode",
                args.cfg_mode,
                "--cfg-scale",
                str(args.cfg_scale),
            ],
        ),
        (
            "emit-vae-manifest",
            common_probe
            + [
                "--probe",
                "vae-decode-full",
                "--vae-mid-attention-policy",
                args.vae_mid_attention_policy,
                "--vae-attention-max-mib",
                str(args.vae_attention_max_mib),
                "--emit-probe-manifest",
                str(vae_manifest),
            ],
        ),
        ("import-vae", [str(args.exe), "--import", str(vae_manifest), str(args.checkpoint), str(vae_graph), "--allow-extra-tensors"]),
        ("compile-vae", [str(args.exe), "--compile-object", str(vae_graph), str(vae_obj), args.vae_symbol_prefix]),
        ("link-vae", link_command(args.cxx, vae_obj, vae_library, vae_def)),
        (
            "decode-latent",
            [
                str(args.exe),
                "--load-dll-with-inputs",
                str(vae_library),
                str(final_latent),
                args.vae_symbol_prefix,
                "--output",
                str(decoded),
            ],
        ),
        (
            "write-png",
            [
                args.python,
                str(scripts_dir / "sdxl_tensor_to_png.py"),
                "--input",
                str(decoded),
                "--output",
                str(output_png),
            ],
        ),
    ]

    for name, command in steps:
        run_step(name, command, dry_run=args.dry_run)

    print(f"\nWrote {output_png}", flush=True)
    if args.unet_probe != "unet-full-fixed":
        print("Note: this used a smoke denoiser probe, so the image validates the LiteNN pipeline rather than full SDXL quality.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
