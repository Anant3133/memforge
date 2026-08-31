# MemForge Pipeline Runner PowerShell Shortcut
$ErrorActionPreference = "Stop"

Write-Host "Activating MemForge Virtual Environment..." -ForegroundColor Cyan
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    & ".\.venv\Scripts\Activate.ps1"
}

Write-Host "Launching MemForge Master Pipeline..." -ForegroundColor Green
& ".\.venv\Scripts\python.exe" scripts/run_pipeline.py
