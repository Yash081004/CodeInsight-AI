# CodeInsight AI — live API demonstration (requires backend on http://127.0.0.1:8000)
# Usage: .\scripts\DEMO.ps1

$Base = "http://127.0.0.1:8000/api"

function Invoke-Api($Method, $Path, $Body) {
    $json = $Body | ConvertTo-Json -Depth 6
    return Invoke-RestMethod -Uri "$Base$Path" -Method $Method -ContentType "application/json" -Body $json
}

Write-Host "`n=== 1. Health ===" -ForegroundColor Cyan
try {
    $health = Invoke-RestMethod -Uri "$Base/health"
    $health | ConvertTo-Json
} catch {
    Write-Host "ERROR: Start backend first: py -3.13 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== 2. Similarity (two snippets) ===" -ForegroundColor Cyan
Write-Host "Note: Similarity scores are NOT proof of logical equivalence." -ForegroundColor Yellow
$sim = Invoke-Api POST "/similarity/analyze" @{
    code_a = "def add(a,b):`n    return a+b"
    code_b = "def sum_two(x,y):`n    return x+y"
    threshold = 0.75
}
$sim | Select-Object codebert_score, tfidf_score, is_similar, decision_model, explanation | Format-List

Write-Host "`n=== 3. Classification (factorial-style correct code) ===" -ForegroundColor Cyan
$cls = Invoke-Api POST "/classification/analyze" @{
    code = @"
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result = result * i
    return result
"@
}
$cls | Select-Object model_type, model_info, predicted_label, friendly_label, confidence | Format-List

Write-Host "`n=== 4. Invalid Python (syntax) ===" -ForegroundColor Cyan
$bad = Invoke-Api POST "/classification/analyze" @{ code = "def broken(`n    pass" }
$bad | Select-Object code_valid, code_error, predicted_label, feedback | Format-List

Write-Host "`n=== 5. Baseline comparison (from saved metrics files) ===" -ForegroundColor Cyan
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$baseline = Get-Content (Join-Path $root "artifacts/metrics/baseline_classification_metrics.json") | ConvertFrom-Json
$experiment = Get-Content (Join-Path $root "artifacts/metrics/experiments/codebert_classifier_20261019_4ep/codebert_classification_metrics.json") | ConvertFrom-Json
Write-Host "Baseline test macro F1: $($baseline.macro_f1) (N=$($baseline.sample_count))"
Write-Host "Experiment test macro F1: $($experiment.macro_f1) (N=$($experiment.sample_count), best epoch $($experiment.best_epoch))"
Write-Host "Production API loads: models/codebert_classifier/ (not the experiment folder)." -ForegroundColor Yellow

Write-Host "`nDone. Open http://localhost:5173 for the UI after npm run dev." -ForegroundColor Green
