Param(
    [string]$RepoRoot = $PSScriptRoot
)

$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    # OneDrive-synced PCs have no Automation\.venv; use the shared workspace venv one level up
    $venvPython = Join-Path (Split-Path $RepoRoot -Parent) ".venv\Scripts\python.exe"
}
if (-not (Test-Path $venvPython)) {
    Write-Error "Missing venv python at $venvPython"
    exit 1
}

Set-Location $RepoRoot
& $venvPython -m spad93.spad93_gui
