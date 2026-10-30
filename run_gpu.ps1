# Automatically configure CUDA and MSVC (cl.exe) for GPU acceleration inside this session
$ErrorActionPreference = "Stop"

# 1. Locate CUDA Toolkit
$cudaCandidates = @(
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4",
    "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1",
    "E:\cuda"
)

$cudaRoot = $null
foreach ($path in $cudaCandidates) {
    if (Test-Path "$path\bin\nvcc.exe") {
        $cudaRoot = $path
        break
    }
}

if ($cudaRoot) {
    $env:CUDA_PATH = $cudaRoot
    $env:CUDA_ROOT = $cudaRoot
    $env:PATH = "$cudaRoot\bin;$cudaRoot\bin\x64;$cudaRoot\lib\x64;$env:PATH"
    Write-Host "[GPU Setup] Using CUDA from: $cudaRoot" -ForegroundColor Green
} else {
    Write-Host "[GPU Setup] Warning: nvcc.exe not found in standard paths." -ForegroundColor Yellow
}

# 2. Locate MSVC cl.exe (Visual Studio 2022)
$vsPath = "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC"
if (Test-Path $vsPath) {
    $msvcVersion = Get-ChildItem $vsPath | Sort-Object Name -Descending | Select-Object -First 1
    if ($msvcVersion) {
        $clDir = Join-Path $msvcVersion.FullName "bin\Hostx64\x64"
        if (Test-Path "$clDir\cl.exe") {
            $env:PATH = "$clDir;$env:PATH"
            Write-Host "[GPU Setup] Injected MSVC (cl.exe) from: $clDir" -ForegroundColor Green
        }
    }
}

# 3. Execute Cargo with the CUDA feature flag
if ($args.Count -eq 0) {
    Write-Host "Running default chat on GPU..." -ForegroundColor Cyan
    cargo run --release --features cuda -- chat
} else {
    cargo run --release --features cuda -- @args
}

