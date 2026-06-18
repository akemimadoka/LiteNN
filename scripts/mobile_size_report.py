#!/usr/bin/env python3
"""Report LiteNN mobile binary and model/package sizes.

Examples:
  python311 scripts/mobile_size_report.py --build-dir build-android-arm64-vulkan
  python311 scripts/mobile_size_report.py --build-dir build-android-arm64-vulkan --assets-dir build/mobile-assets --json out.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


BINARY_SUFFIXES = {
    ".a",
    ".dll",
    ".dylib",
    ".exe",
    ".lib",
    ".so",
}

ASSET_SUFFIXES = {
    ".ltnn",
    ".gguf",
    ".safetensors",
}

SEPARATED_REGION_SUFFIXES = (
    ".metadata.bin",
    ".constants.bin",
    ".weights.bin",
    ".instructions.bin",
    ".rodata.bin",
)


@dataclass(frozen=True)
class SizeEntry:
    category: str
    path: Path
    size: int


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return
    for path in root.rglob("*"):
        if path.is_file():
            yield path


def is_binary_artifact(path: Path) -> bool:
    return path.suffix.lower() in BINARY_SUFFIXES


def is_model_asset(path: Path) -> bool:
    name = path.name.lower()
    return path.suffix.lower() in ASSET_SUFFIXES or any(name.endswith(suffix) for suffix in SEPARATED_REGION_SUFFIXES)


def collect_entries(build_dirs: list[Path], asset_dirs: list[Path]) -> list[SizeEntry]:
    entries: list[SizeEntry] = []
    for root in build_dirs:
        for path in iter_files(root):
            if is_binary_artifact(path):
                entries.append(SizeEntry("binary", path, path.stat().st_size))
    for root in asset_dirs:
        for path in iter_files(root):
            if is_model_asset(path):
                entries.append(SizeEntry("asset", path, path.stat().st_size))
    entries.sort(key=lambda entry: (entry.category, -entry.size, str(entry.path).lower()))
    return entries


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{size} B"


def path_for_report(path: Path, roots: list[Path]) -> str:
    resolved = path.resolve()
    for root in roots:
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return path.as_posix()


def totals(entries: list[SizeEntry]) -> dict[str, int]:
    result: dict[str, int] = {}
    for entry in entries:
        result[entry.category] = result.get(entry.category, 0) + entry.size
    result["all"] = sum(entry.size for entry in entries)
    return result


def write_markdown(entries: list[SizeEntry], roots: list[Path]) -> str:
    total = totals(entries)
    lines = [
        "# LiteNN Mobile Size Report",
        "",
        "| Category | Total |",
        "| --- | ---: |",
    ]
    for category in ("binary", "asset", "all"):
        lines.append(f"| {category} | {human_size(total.get(category, 0))} |")
    lines.extend(
        [
            "",
            "| Category | Size | Path |",
            "| --- | ---: | --- |",
        ]
    )
    for entry in entries:
        lines.append(f"| {entry.category} | {human_size(entry.size)} | `{path_for_report(entry.path, roots)}` |")
    lines.append("")
    return "\n".join(lines)


def write_json(entries: list[SizeEntry], roots: list[Path]) -> dict[str, object]:
    return {
        "totals": totals(entries),
        "entries": [
            {
                "category": entry.category,
                "size": entry.size,
                "path": path_for_report(entry.path, roots),
            }
            for entry in entries
        ],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", action="append", default=[], type=Path, help="Build directory to scan")
    parser.add_argument("--assets-dir", action="append", default=[], type=Path, help="Model/package asset directory to scan")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report output")
    parser.add_argument("--json", type=Path, help="Optional JSON report output")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    build_dirs = [path.resolve() for path in args.build_dir]
    asset_dirs = [path.resolve() for path in args.assets_dir]
    if not build_dirs and not asset_dirs:
        raise SystemExit("provide at least one --build-dir or --assets-dir")

    roots = [*build_dirs, *asset_dirs]
    entries = collect_entries(build_dirs, asset_dirs)
    markdown = write_markdown(entries, roots)
    print(markdown)

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown, encoding="utf-8")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(write_json(entries, roots), indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
