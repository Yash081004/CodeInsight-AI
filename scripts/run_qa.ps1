# CodeInsight AI — run full QA from repo root
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $MyInvocation.MyCommand.Path) | Out-Null
Set-Location ..
if (Test-Path ".venv\Scripts\python.exe") {
    & .venv\Scripts\python.exe scripts/run_qa.py @args
} else {
    py -3.13 scripts/run_qa.py @args
}
exit $LASTEXITCODE
