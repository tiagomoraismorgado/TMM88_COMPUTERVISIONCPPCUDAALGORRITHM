param(
    [string]$Configuration = 'Release'
)

$ErrorActionPreference = 'Stop'

if (-Not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Error "cmake is not installed or not in PATH. Install CMake and retry."
    exit 1
}

if (-Not (Test-Path build)) {
    New-Item -ItemType Directory -Path build | Out-Null
}

Push-Location build
cmake .. -DCMAKE_BUILD_TYPE=$Configuration
cmake --build . --config $Configuration
Pop-Location

if (Test-Path "build\cuda_mad_exploits.exe") {
    Write-Host "Build successful"
} else {
    throw "Build failed"
}
