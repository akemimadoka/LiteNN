param(
	[string]$ClangFormat = "clang-format",
	[switch]$Check
)

$ErrorActionPreference = "Stop"

$root = (& git rev-parse --show-toplevel 2>$null)
if (-not $root)
{
	$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}
else
{
	$root = $root.Trim()
}

$extensions = @(
	"*.c",
	"*.cc",
	"*.cpp",
	"*.cxx",
	"*.h",
	"*.hh",
	"*.hpp",
	"*.hxx",
	"*.cu",
	"*.cuh"
)

$files = & git -C $root ls-files -- $extensions |
	Where-Object {
		$_ -and
		$_ -notmatch '^(third_party|build|build-|\.cache|\.clangd)/' -and
		$_ -notmatch '(^|/)(__pycache__|CMakeFiles)(/|$)'
	}

if (-not $files)
{
	Write-Host "No source files found for clang-format."
	exit 0
}

if ($Check)
{
	& $ClangFormat --dry-run --Werror --style=file $files
}
else
{
	& $ClangFormat -i --style=file $files
}
