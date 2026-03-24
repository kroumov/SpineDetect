# Run full pipeline: download -> downsample -> chunk -> degrade (1 chunk) -> noise -> interpolate
# Usage: .\run_pipeline.ps1   (from pipelines/local in PowerShell)

$ErrorActionPreference = "Stop"

$env:CLOUD_LOGS_DIR = ""

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent (Split-Path -Parent $ScriptDir)
Set-Location $ProjectRoot

$VenvPath = Join-Path $ProjectRoot "sample_random_volume\microns_env\Scripts\Activate.ps1"
if (Test-Path $VenvPath) {
    & $VenvPath
}

Write-Host "=== [1/6] Download ==="
python (Join-Path $ScriptDir "download.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [2/6] Downsample ==="
python (Join-Path $ScriptDir "downsample.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [3/6] Chunk ==="
python (Join-Path $ScriptDir "chunk.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [4/6] Degrade (1 chunk) ==="
python (Join-Path $ScriptDir "degrade.py") -n 1
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [5/6] Noise ==="
python (Join-Path $ScriptDir "noise.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== [6/6] Interpolate ==="
python (Join-Path $ScriptDir "interpolate.py")
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

Write-Host "=== Pipeline complete ==="
