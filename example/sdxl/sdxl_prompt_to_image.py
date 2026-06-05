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
import json
import os
import subprocess
import sys
import time
from pathlib import Path


DTYPE_BYTES = {
    "F64": 8,
    "torch.float64": 8,
    "float64": 8,
    "F32": 4,
    "torch.float32": 4,
    "float32": 4,
    "F16": 2,
    "torch.float16": 2,
    "float16": 2,
    "BF16": 2,
    "torch.bfloat16": 2,
    "bfloat16": 2,
    "I64": 8,
    "torch.int64": 8,
    "torch.long": 8,
    "I32": 4,
    "torch.int32": 4,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


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


def shape_elements(shape: object, label: str) -> int:
    if not isinstance(shape, list) or not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError(f"{label} has invalid tensor shape")
    count = 1
    for dim in shape:
        count *= dim
    return count


def dtype_nbytes(dtype: object, label: str) -> int:
    if not isinstance(dtype, str):
        raise ValueError(f"{label} has invalid dtype")
    try:
        return DTYPE_BYTES[dtype]
    except KeyError as exc:
        raise ValueError(f"{label} uses unsupported dtype {dtype!r} for size estimation") from exc


def estimate_manifest_tensor_bytes(path: Path) -> int:
    root = json.loads(path.read_text(encoding="utf-8"))
    tensors = root.get("tensors")
    if not isinstance(tensors, list):
        return 0
    total = 0
    for index, tensor in enumerate(tensors):
        if not isinstance(tensor, dict):
            raise ValueError(f"{path}: tensor entry {index} must be an object")
        dtype = tensor.get("target_dtype", tensor.get("dtype"))
        total += shape_elements(tensor.get("shape"), f"{path}: tensor entry {index}") * dtype_nbytes(
            dtype, f"{path}: tensor entry {index}"
        )
    return total


def manifest_stats(path: Path) -> dict[str, int]:
    root = json.loads(path.read_text(encoding="utf-8"))
    tensors = root.get("tensors")
    nodes = root.get("nodes")
    inputs = root.get("inputs")
    outputs = root.get("outputs")
    tensor_count = len(tensors) if isinstance(tensors, list) else 0
    node_count = len(nodes) if isinstance(nodes, list) else 0
    input_count = len(inputs) if isinstance(inputs, list) else 0
    output_count = len(outputs) if isinstance(outputs, list) else 0
    return {
        "tensor_count": tensor_count,
        "node_count": node_count,
        "input_count": input_count,
        "output_count": output_count,
        "tensor_bytes": estimate_manifest_tensor_bytes(path),
    }


def print_manifest_stats(path: Path, label: str) -> None:
    stats = manifest_stats(path)
    tensor_mib = stats["tensor_bytes"] / (1024.0 * 1024.0)
    print(
        f"{label} manifest: tensors={stats['tensor_count']} nodes={stats['node_count']} "
        f"inputs={stats['input_count']} outputs={stats['output_count']} "
        f"target_payload={tensor_mib:.1f} MiB",
        flush=True,
    )


def enforce_manifest_budget(path: Path, label: str, budget_mib: int) -> None:
    if budget_mib <= 0:
        return
    estimated_bytes = estimate_manifest_tensor_bytes(path)
    estimated_mib = estimated_bytes / (1024.0 * 1024.0)
    print(f"{label} manifest tensor payload estimate: {estimated_mib:.1f} MiB", flush=True)
    if estimated_bytes > budget_mib * 1024 * 1024:
        raise RuntimeError(
            f"{label} manifest tensor payload estimate {estimated_mib:.1f} MiB exceeds "
            f"--max-unet-weight-mib={budget_mib}. This path would import and compile very large weights and "
            "can still drive CPU AOT memory into tens of GiB. Use the smoke probe, lower precision/quantization, "
            "or pass --max-unet-weight-mib 0 only when intentionally running the full compile on a large-memory host."
        )


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
    parser.add_argument("--unet-compute-dtype", default="F32", choices=["F32", "F16", "BF16"])
    parser.add_argument("--vae-compute-dtype", default="F32", choices=["F32", "F16", "BF16"])
    parser.add_argument(
        "--conditioning-output-dtype",
        choices=["F32", "F16", "BF16"],
        help="Safetensors dtype for exported conditioning; defaults to --unet-compute-dtype",
    )
    parser.add_argument("--vae-mid-attention-policy", choices=["auto", "force", "skip"], default="skip")
    parser.add_argument("--vae-attention-max-mib", default=512, type=int)
    parser.add_argument(
        "--max-unet-weight-mib",
        default=2048,
        type=int,
        help="Preflight budget for imported UNet tensor payloads before AOT compile; use 0 to disable the guard",
    )
    parser.add_argument(
        "--inline-model-weights",
        action="store_true",
        help="Embed imported tensor payloads in the vNext package instead of writing sibling .weights.bin files",
    )
    parser.add_argument(
        "--external-weight-min-bytes",
        default=0,
        type=int,
        help="Minimum variable payload size to place in external .weights.bin files",
    )
    parser.add_argument("--aot-load-mode", choices=["dll", "image-regions"], default="dll")
    parser.add_argument(
        "--cpu-aot-llvm-opt-level",
        default=0,
        type=int,
        choices=[0, 1, 2, 3],
        help="LLVM optimization level forwarded to litenn_sdxl_example image-region compilation",
    )
    parser.add_argument(
        "--png-range",
        choices=["auto", "zero-one", "minus-one-one"],
        default="auto",
        help="Value range passed to sdxl_tensor_to_png.py",
    )
    parser.add_argument("--skip-image-validation", action="store_true", help="Write the PNG without postprocess sanity checks")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cxx", default=os.environ.get("CXX", "g++"))
    parser.add_argument("--unet-symbol-prefix", default="litenn_sdxl_unet")
    parser.add_argument("--vae-symbol-prefix", default="litenn_sdxl_vae")
    parser.add_argument("--allow-pretrained-download", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Emit UNet/VAE manifests, print payload budgets, then stop before import or compile",
    )
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
    if args.max_unet_weight_mib < 0:
        raise ValueError("--max-unet-weight-mib must be non-negative")
    if args.external_weight_min_bytes < 0:
        raise ValueError("--external-weight-min-bytes must be non-negative")
    conditioning_output_dtype = args.conditioning_output_dtype or args.unet_compute_dtype

    args.workdir.mkdir(parents=True, exist_ok=True)
    output_png = args.output_png or (args.workdir / "image.png")
    conditioning = args.workdir / "conditioning.safetensors"
    final_latent = args.workdir / "final_latent.safetensors"
    decoded = args.workdir / "decoded_image.safetensors"
    unet_manifest = args.workdir / "unet_manifest.json"
    vae_manifest = args.workdir / "vae_manifest.json"
    unet_graph = args.workdir / "unet.ltnn.json"
    vae_graph = args.workdir / "vae.ltnn.json"
    unet_weights = args.workdir / "unet.weights.bin"
    vae_weights = args.workdir / "vae.weights.bin"
    unet_obj = args.workdir / "unet.obj"
    vae_obj = args.workdir / "vae.obj"
    unet_library = library_path(args.workdir, "unet")
    vae_library = library_path(args.workdir, "vae")
    unet_def = args.workdir / f"{args.unet_symbol_prefix}_exports.def" if os.name == "nt" else None
    vae_def = args.workdir / f"{args.vae_symbol_prefix}_exports.def" if os.name == "nt" else None
    unet_region_dir = args.workdir / "unet_regions"
    vae_region_dir = args.workdir / "vae_regions"
    unet_rodata = unet_region_dir / f"{args.unet_symbol_prefix}.rodata.bin"
    unet_instructions = unet_region_dir / f"{args.unet_symbol_prefix}.instructions.obj"
    vae_rodata = vae_region_dir / f"{args.vae_symbol_prefix}.rodata.bin"
    vae_instructions = vae_region_dir / f"{args.vae_symbol_prefix}.instructions.obj"

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
        "--output-dtype",
        conditioning_output_dtype,
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

    unet_import_command = [str(args.exe), "--import-package", str(unet_manifest), str(args.checkpoint), str(unet_graph), "--allow-extra-tensors"]
    vae_import_command = [str(args.exe), "--import-package", str(vae_manifest), str(args.checkpoint), str(vae_graph), "--allow-extra-tensors"]
    if not args.inline_model_weights:
        unet_import_command += [
            "--external-weights",
            str(unet_weights),
            "--external-weight-min-bytes",
            str(args.external_weight_min_bytes),
        ]
        vae_import_command += [
            "--external-weights",
            str(vae_weights),
            "--external-weight-min-bytes",
            str(args.external_weight_min_bytes),
        ]

    emit_unet_manifest_step = (
        "emit-unet-manifest",
        common_probe
        + [
            "--probe",
            args.unet_probe,
            "--compute-dtype",
            args.unet_compute_dtype,
            "--context-tokens",
            str(args.context_tokens),
            "--emit-probe-manifest",
            str(unet_manifest),
        ],
    )
    emit_vae_manifest_step = (
        "emit-vae-manifest",
        common_probe
        + [
            "--probe",
            "vae-decode-full",
            "--compute-dtype",
            args.vae_compute_dtype,
            "--vae-mid-attention-policy",
            args.vae_mid_attention_policy,
            "--vae-attention-max-mib",
            str(args.vae_attention_max_mib),
            "--emit-probe-manifest",
            str(vae_manifest),
        ],
    )

    if args.preflight_only:
        steps = [emit_unet_manifest_step, emit_vae_manifest_step]
    else:
        steps = [
            ("export-conditioning", conditioning_command),
            emit_unet_manifest_step,
            ("import-unet", unet_import_command),
        ]

    if args.preflight_only:
        for name, command in steps:
            run_step(name, command, dry_run=args.dry_run)
            if not args.dry_run and name == "emit-unet-manifest":
                print_manifest_stats(unet_manifest, "UNet")
                enforce_manifest_budget(unet_manifest, "UNet", args.max_unet_weight_mib)
            if not args.dry_run and name == "emit-vae-manifest":
                print_manifest_stats(vae_manifest, "VAE")
        print("\nPreflight complete; stopped before import, compile, denoise, decode, and PNG writing.", flush=True)
        return 0

    denoise_options = [
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
    ]
    if args.aot_load_mode == "image-regions":
        steps.extend(
            [
                (
                    "compile-unet-regions",
                    [
                        str(args.exe),
                        "--compile-image-regions",
                        str(unet_graph),
                        str(unet_region_dir),
                        args.unet_symbol_prefix,
                        "--cpu-aot-llvm-opt-level",
                        str(args.cpu_aot_llvm_opt_level),
                    ],
                ),
                (
                    "denoise-latent",
                    [
                        str(args.exe),
                        "--denoise-latent-image",
                        str(unet_rodata),
                        str(unet_instructions),
                        str(conditioning),
                        str(final_latent),
                    ]
                    + denoise_options,
                ),
            ]
        )
    else:
        steps.extend(
            [
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
                    ]
                    + denoise_options,
                ),
            ]
        )

    if not args.preflight_only:
        steps.extend(
            [
                emit_vae_manifest_step,
                ("import-vae", vae_import_command),
            ]
        )

    if args.aot_load_mode == "image-regions":
        steps.extend(
            [
                (
                    "compile-vae-regions",
                    [
                        str(args.exe),
                        "--compile-image-regions",
                        str(vae_graph),
                        str(vae_region_dir),
                        args.vae_symbol_prefix,
                        "--cpu-aot-llvm-opt-level",
                        str(args.cpu_aot_llvm_opt_level),
                    ],
                ),
                (
                    "decode-latent",
                    [
                        str(args.exe),
                        "--run-image-with-inputs",
                        str(vae_rodata),
                        str(vae_instructions),
                        str(final_latent),
                        "--output",
                        str(decoded),
                    ],
                ),
            ]
        )
    else:
        steps.extend(
            [
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
            ]
        )

    steps.append(
        (
            "write-png",
            [
                args.python,
                str(scripts_dir / "sdxl_tensor_to_png.py"),
                "--input",
                str(decoded),
                "--output",
                str(output_png),
                "--range",
                args.png_range,
            ]
            + ([] if args.skip_image_validation else ["--validate"]),
        )
    )

    for name, command in steps:
        run_step(name, command, dry_run=args.dry_run)
        if not args.dry_run and name == "emit-unet-manifest":
            print_manifest_stats(unet_manifest, "UNet")
            enforce_manifest_budget(unet_manifest, "UNet", args.max_unet_weight_mib)
        if not args.dry_run and name == "emit-vae-manifest":
            print_manifest_stats(vae_manifest, "VAE")

    print(f"\nWrote {output_png}", flush=True)
    if args.unet_probe != "unet-full-fixed":
        print("Note: this used a smoke denoiser probe, so the image validates the LiteNN pipeline rather than full SDXL quality.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
