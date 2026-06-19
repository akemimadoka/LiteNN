param(
	[string]$ClangFormat = "clang-format",
	[switch]$Check
)

$ErrorActionPreference = "Stop"

$python = Get-Command python3 -ErrorAction SilentlyContinue
if (-not $python)
{
	$python = Get-Command python -ErrorAction SilentlyContinue
}
if (-not $python)
{
	$python = Get-Command py -ErrorAction SilentlyContinue
}

if (-not $python)
{
	throw "Python is required to run scripts/format_sources.py."
}

$args = @((Join-Path $PSScriptRoot "format_sources.py"), "--clang-format", $ClangFormat)
if ($Check)
{
	$args += "--check"
}

& $python.Source @args
