# PL + TRPL dual-gate viewer.  Arguments are passed through, e.g.
#   .\run_pl_trpl_dual_gate_viewer.ps1 --folder <scan> --calib <gtd1_calibration.json> --ndmap <cache.npz | folder>
$RepoRoot = $PSScriptRoot
$venvPython = Join-Path $RepoRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) {
    Write-Error "Missing venv python at $venvPython"
    exit 1
}

Set-Location $RepoRoot
& $venvPython (Join-Path $RepoRoot "data processing\pl_trpl_dual_gate_viewer.py") @args
