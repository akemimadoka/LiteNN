import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

INCLUDE_GUARD_PATTERN = re.compile(
    r"^\s*#ifndef\s+(LITENN_\w+)\s*$\n^\s*#define\s+\1\s*$", re.MULTILINE)
DEFINE_GUARDS_PLACEHOLDER = "%DEFINE_GUARDS%"
INCLUDE_PLACEHOLDER = "%INCLUDE%"
UNDEF_GUARDS_PLACEHOLDER = "%UNDEF_GUARDS%"

# 获取所有 .h 的路径及 guard 宏名称，并构造 LiteNN.ixx
# 细节：通过将依赖头文件提到 guard 之外，我们可以通过提前定义或 undef 来控制是否包含实际定义内容，而其他依赖应当已有 guard 而并未因为 undef 而重复包含
# 从而我们可以做到在 export 块之外包含依赖，而在 export 之内仅包含实际定义内容


@dataclass
class HeaderInfo:
    include_path: str
    guard: str


def ToAbsolutePath(path: str) -> Path:
    return Path(path).resolve()


def ToIncludePath(path: Path, include_root: Path) -> str:
    try:
        return path.relative_to(include_root).as_posix()
    except ValueError as exc:
        raise SystemExit(
            f"Header '{path}' is outside include root '{include_root}', cannot generate stable module include path") from exc


collected_headers: list[HeaderInfo] = []
missing_guard_headers: list[str] = []

args = argparse.ArgumentParser(description="Generate LiteNN.ixx from template")
args.add_argument("--input-files", nargs="+", required=True,
                  help="List of input header files to include in the module")
args.add_argument("--input-template", type=str, required=True,
                  help="Path to the module file template")
args.add_argument("--output", type=str, required=True,
                  help="Output path for generated LiteNN.ixx")
parsed_args = args.parse_args()

input_files = [ToAbsolutePath(path) for path in parsed_args.input_files]
template_path = ToAbsolutePath(parsed_args.input_template)
include_root = template_path.parent
output_path = ToAbsolutePath(parsed_args.output)

for path in input_files:
    if path.suffix != ".h":
        continue

    content = path.read_text(encoding="utf-8")
    match = INCLUDE_GUARD_PATTERN.search(content)
    if match:
        collected_headers.append(HeaderInfo(
            include_path=ToIncludePath(path, include_root),
            guard=match.group(1),
        ))
    else:
        missing_guard_headers.append(str(path))

if missing_guard_headers:
    print("Missing include guards in module input headers:", file=sys.stderr)
    for path in missing_guard_headers:
        print(f"  - {path}", file=sys.stderr)
    raise SystemExit(1)

if not collected_headers:
    raise SystemExit("No header inputs were provided to module generator")

template = template_path.read_text(encoding="utf-8")

define_guards = "\n".join(
    f"#define {header.guard}" for header in collected_headers)
include_headers = "\n".join(
    f'#include "{header.include_path}"' for header in collected_headers)
undef_guards = "\n".join(
    f"#undef {header.guard}" for header in collected_headers)
generated_content = template.replace(DEFINE_GUARDS_PLACEHOLDER, define_guards)
generated_content = generated_content.replace(
    INCLUDE_PLACEHOLDER, include_headers)
generated_content = generated_content.replace(
    UNDEF_GUARDS_PLACEHOLDER, undef_guards)

output_path.parent.mkdir(parents=True, exist_ok=True)
output_path.write_text(generated_content, encoding="utf-8")

print(
    f"Generated {output_path} with {len(collected_headers)} headers included")
