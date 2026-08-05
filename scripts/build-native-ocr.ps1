[CmdletBinding()]
param(
    [string]$Python = "",
    [string]$OpenCvDirectory = $env:OpenCV_DIR,
    [ValidateSet("Debug", "Release", "RelWithDebInfo", "MinSizeRel")]
    [string]$Configuration = "Release"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$sourceDirectory = Join-Path $repositoryRoot "native"
$buildDirectory = Join-Path $repositoryRoot ".build\native-ocr"

if ([string]::IsNullOrWhiteSpace($Python)) {
    foreach ($candidate in @(
        (Join-Path $repositoryRoot ".venv-nvidia-cu126\Scripts\python.exe"),
        (Join-Path $repositoryRoot ".venv-cpu\Scripts\python.exe"),
        (Join-Path $repositoryRoot ".venv-build\Scripts\python.exe")
    )) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            $Python = $candidate
            break
        }
    }
}
if ([string]::IsNullOrWhiteSpace($Python)) {
    $Python = (Get-Command python.exe -ErrorAction Stop).Source
}
$Python = [System.IO.Path]::GetFullPath($Python)
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable not found: $Python"
}

$cmake = (Get-Command cmake.exe -ErrorAction Stop).Source
$configureArguments = @(
    "-S", $sourceDirectory,
    "-B", $buildDirectory,
    "-DPython3_EXECUTABLE=$Python"
)
if (-not [string]::IsNullOrWhiteSpace($OpenCvDirectory)) {
    $configureArguments += "-DOpenCV_DIR=$OpenCvDirectory"
}

& $cmake @configureArguments
if ($LASTEXITCODE -ne 0) {
    throw "Native OCR configure failed. Install MSVC Build Tools and an OpenCV C++ SDK, then set OpenCV_DIR."
}
& $cmake --build $buildDirectory --config $Configuration
if ($LASTEXITCODE -ne 0) {
    throw "Native OCR build failed with exit code $LASTEXITCODE."
}

& $Python -c "from backend.native_ocr import implementation_name; assert implementation_name() == 'kaor_native+opencv'; print(implementation_name())"
if ($LASTEXITCODE -ne 0) {
    throw "The extension was built but could not be imported by the selected Python runtime."
}
Write-Host "Native OCR frame-change extension is ready in backend\."
