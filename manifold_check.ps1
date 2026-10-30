$exe = if (Test-Path ".\target\release\mamba3_sov.exe") { ".\target\release\mamba3_sov.exe" } else { ".\target\release\leibniz_node.exe" }

$labels = @{
    1 = "KJV/Ancient"
    2 = "Metaphysics (2x)"
    5 = "Physics (2x)"
    7 = "Computation (2x)"
    8 = "Web Docs (3x)"
}

foreach ($id in $labels.Keys) {
    Write-Host "--- Checking Label $id : $($labels[$id]) ---" -ForegroundColor Cyan
    & $exe chat "the nature of" --label-id $id --length 30 --temp 0.1
    Write-Host "`n"
}

Write-Host "--- Checking Global Parallel Scan (Label 0) ---" -ForegroundColor Yellow
Measure-Command { & $exe chat "is the" --label-id 0 --length 10 }