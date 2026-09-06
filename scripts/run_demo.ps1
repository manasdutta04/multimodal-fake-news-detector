# Run Streamlit demo (Windows PowerShell) from repo root.
# Usage: .\scripts\run_demo.ps1
# Optional: .\scripts\run_demo.ps1 -CheckpointsDir "D:\...\dataset\checkpoints"

param(
    [string]$CheckpointsDir = $env:CHECKPOINTS_DIR
)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path $PSScriptRoot -Parent)

if (-not $CheckpointsDir) {
    if (Test-Path ".\dataset\checkpoints") {
        $CheckpointsDir = (Resolve-Path ".\dataset\checkpoints").Path
    } elseif (Test-Path ".\checkpoints") {
        $CheckpointsDir = (Resolve-Path ".\checkpoints").Path
    } else {
        Write-Host "Put weights at dataset\checkpoints\ (Drive layout), or pass -CheckpointsDir"
        exit 1
    }
}

$env:CHECKPOINTS_DIR = $CheckpointsDir
Write-Host "CHECKPOINTS_DIR=$env:CHECKPOINTS_DIR"
python scripts/check_checkpoints.py --dir $env:CHECKPOINTS_DIR
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

streamlit run app/streamlit_app.py
