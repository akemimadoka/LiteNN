$ErrorActionPreference = "Stop"

$root = (& git rev-parse --show-toplevel 2>$null)
if (-not $root)
{
	throw "install_git_hooks.ps1 must be run inside a Git worktree."
}
$root = $root.Trim()

$source = Join-Path $root "scripts/hooks/pre-commit"
$targetDir = Join-Path $root ".git/hooks"
$target = Join-Path $targetDir "pre-commit"

if (-not (Test-Path -LiteralPath $source))
{
	throw "Missing hook template: $source"
}

New-Item -ItemType Directory -Force -Path $targetDir | Out-Null
Copy-Item -LiteralPath $source -Destination $target -Force

if (-not $IsWindows)
{
	& chmod +x $target
}

Write-Host "Installed LiteNN pre-commit hook: $target"
