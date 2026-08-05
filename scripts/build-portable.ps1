[CmdletBinding()]
param(
    [Alias("Profile")]
    [ValidateSet("cpu", "amd", "nvidia-cu126")]
    [string]$RuntimeProfile = "cpu",

    [string]$Python = "py",

    [string]$Version = "0.2.0",

    [string]$OutputDirectory = "artifacts\releases",

    [string]$LocalInferenceRuntimeDirectory = "",

    [switch]$SkipInstall,

    [switch]$SkipFrontend,

    [switch]$SkipTests,

    [switch]$SkipArchive,

    [switch]$RequireAccelerator,

    [switch]$BundleDiarizationModels,

    [switch]$AllowOversizedArchive
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$torchVersion = "2.11.0"
$torchAudioVersion = "2.11.0"
$torchVisionVersion = "0.26.0"
$githubReleaseAssetLimit = 2GB
$uvrModelName = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
$uvrConfigName = "model_bs_roformer_ep_317_sdr_12.9755.yaml"
$diarizationModelNames = @("vad_multilingual_marblenet.nemo", "titanet_large.nemo")

function Invoke-External {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $false)]
        [string[]]$Arguments = @()
    )

    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Resolve-RepositoryPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [string]$RepositoryRoot
    )

    $candidate = if ([System.IO.Path]::IsPathRooted($Path)) {
        [System.IO.Path]::GetFullPath($Path)
    }
    else {
        [System.IO.Path]::GetFullPath((Join-Path $RepositoryRoot $Path))
    }

    $rootPrefix = [System.IO.Path]::GetFullPath($RepositoryRoot).TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    if (-not $candidate.StartsWith($rootPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Build paths must stay inside the repository: $candidate"
    }
    return $candidate
}

function Get-RelativePackagePath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FullName,

        [Parameter(Mandatory = $true)]
        [string]$PackageRoot
    )

    $prefix = [System.IO.Path]::GetFullPath($PackageRoot).TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    $resolved = [System.IO.Path]::GetFullPath($FullName)
    if (-not $resolved.StartsWith($prefix, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Package manifest path escaped the package root: $resolved"
    }
    return $resolved.Substring($prefix.Length).Replace('\', '/')
}

function Assert-File {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file not found: $Path"
    }
}

if ($Version -notmatch '^\d+\.\d+\.\d+([-.][0-9A-Za-z.-]+)?$') {
    throw "Version must be a SemVer-compatible value without a leading v: $Version"
}
if (-not [Environment]::Is64BitOperatingSystem) {
    throw "Portable releases require 64-bit Windows."
}

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$requirementsPath = Join-Path $repositoryRoot "requirements-$RuntimeProfile.txt"
$specPath = Join-Path $repositoryRoot "Kaor.spec"
$webDirectory = Join-Path $repositoryRoot "apps\web"
$modelsDirectory = Join-Path $repositoryRoot "models"
$ocrModelsDirectory = Join-Path $modelsDirectory "paddlex"
$ocrManifestPath = Join-Path $repositoryRoot "licenses\PP-OCRV6-MODEL-MANIFEST.json"
$diarizationModelsDirectory = Join-Path $modelsDirectory "diarization"
$diarizationModelPaths = @(
    $diarizationModelNames | ForEach-Object { Join-Path $diarizationModelsDirectory $_ }
)
$existingDiarizationModels = @(
    $diarizationModelPaths | Where-Object { Test-Path -LiteralPath $_ -PathType Leaf }
)
$validDiarizationModels = @(
    $existingDiarizationModels | Where-Object { (Get-Item -LiteralPath $_).Length -ge 10000 }
)
$diarizationModelsBundled = (
    [bool]$BundleDiarizationModels -and
    $validDiarizationModels.Count -eq $diarizationModelPaths.Count
)
if ($BundleDiarizationModels -and -not $diarizationModelsBundled) {
    throw "Incomplete models\diarization input. Provide both complete NeMo model archives or remove the partial files."
}
$binDirectory = Join-Path $repositoryRoot "bin"
$fontsDirectory = Join-Path $repositoryRoot "fonts"
$docsDirectory = Join-Path $repositoryRoot "docs"
$outputRoot = Resolve-RepositoryPath -Path $OutputDirectory -RepositoryRoot $repositoryRoot
$buildRoot = Resolve-RepositoryPath -Path ".build\portable\$RuntimeProfile" -RepositoryRoot $repositoryRoot
$buildHome = Resolve-RepositoryPath -Path ".build\runtime-home" -RepositoryRoot $repositoryRoot
$buildCache = Resolve-RepositoryPath -Path ".build\runtime-cache" -RepositoryRoot $repositoryRoot
$venvDirectory = Resolve-RepositoryPath -Path ".venv-$RuntimeProfile" -RepositoryRoot $repositoryRoot
$venvPython = Join-Path $venvDirectory "Scripts\python.exe"

# Keep build-time model and framework caches inside the repository. This makes
# release builds work for accounts whose normal profile cache is read-only and
# prevents Paddle/PaddleX from silently downloading into a maintainer's home.
New-Item -ItemType Directory -Path $buildHome -Force | Out-Null
New-Item -ItemType Directory -Path $buildCache -Force | Out-Null
$env:HOME = $buildHome
$env:USERPROFILE = $buildHome
$env:XDG_CACHE_HOME = $buildCache
$env:PADDLE_HOME = Join-Path $buildCache "paddle"
$env:PADDLE_PDX_CACHE_HOME = $ocrModelsDirectory
$env:PYINSTALLER_CONFIG_DIR = Join-Path $buildCache "pyinstaller"
New-Item -ItemType Directory -Path $env:PYINSTALLER_CONFIG_DIR -Force | Out-Null

Assert-File -Path $requirementsPath
Assert-File -Path $specPath
Assert-File -Path $ocrManifestPath
if (-not (Test-Path -LiteralPath $ocrModelsDirectory -PathType Container)) {
    throw "Missing models\paddlex. Portable releases must contain offline OCR models."
}
$ocrManifest = try {
    Get-Content -LiteralPath $ocrManifestPath -Raw -Encoding utf8 | ConvertFrom-Json
}
catch {
    throw "Invalid PP-OCRv6 model manifest: $($_.Exception.Message)"
}
if ([int]$ocrManifest.schema_version -ne 1 -or @($ocrManifest.models).Count -ne 2) {
    throw "Unsupported or incomplete PP-OCRv6 model manifest."
}
foreach ($model in @($ocrManifest.models)) {
    $modelDirectoryName = [string]$model.directory
    if ([IO.Path]::GetFileName($modelDirectoryName) -ne $modelDirectoryName) {
        throw "Unsafe PP-OCRv6 model directory in manifest: $modelDirectoryName"
    }
    $modelDirectory = Join-Path $ocrModelsDirectory "official_models\$modelDirectoryName"
    foreach ($file in @($model.files)) {
        $fileName = [string]$file.name
        $expectedHash = ([string]$file.sha256).ToLowerInvariant()
        if ([IO.Path]::GetFileName($fileName) -ne $fileName -or
            $expectedHash -notmatch '^[0-9a-f]{64}$' -or
            [long]$file.size -le 0) {
            throw "Invalid PP-OCRv6 file record in manifest: $modelDirectoryName/$fileName"
        }
        $modelFile = Join-Path $modelDirectory $fileName
        Assert-File -Path $modelFile
        $modelFileInfo = Get-Item -LiteralPath $modelFile
        if ($modelFileInfo.Length -ne [long]$file.size) {
            throw "PP-OCRv6 file size mismatch: $modelDirectoryName/$fileName"
        }
        $actualHash = (Get-FileHash -LiteralPath $modelFile -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $expectedHash) {
            throw "PP-OCRv6 SHA-256 mismatch: $modelDirectoryName/$fileName"
        }
    }
}
$ocrParameterFiles = @(Get-ChildItem -LiteralPath $ocrModelsDirectory -Recurse -Filter "*.pdiparams" -File)
if ($ocrParameterFiles.Count -lt 2) {
    throw "Incomplete offline OCR models: expected detector and recognizer parameter files."
}
Assert-File -Path (Join-Path $fontsDirectory "NotoSansSC\NotoSansSC-Regular.ttf")
foreach ($documentationPath in @(
    "ARCHITECTURE.zh-CN.md",
    "USER_GUIDE.zh-CN.md",
    "TROUBLESHOOTING.zh-CN.md",
    "images\kaor-workbench.png"
)) {
    Assert-File -Path (Join-Path $docsDirectory $documentationPath)
}

if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
    $pythonLeaf = Split-Path -Leaf $Python
    $launcherArguments = if ($pythonLeaf -in @("py", "py.exe")) { @("-3.12") } else { @() }
    Invoke-External -FilePath $Python -Arguments ($launcherArguments + @("-m", "venv", $venvDirectory))
}

$pythonInfo = & $venvPython -c "import platform, struct, sys; print(sys.version_info.major, sys.version_info.minor, struct.calcsize(chr(80)) * 8, platform.system(), sep=chr(124))"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to inspect the build Python runtime."
}
if ($pythonInfo.Trim() -ne "3|12|64|Windows") {
    throw "Expected Python 3.12 x64 on Windows, found: $($pythonInfo.Trim())"
}

$torchIndex = if ($RuntimeProfile -eq "nvidia-cu126") {
    "https://download.pytorch.org/whl/cu126"
}
else {
    "https://pypi.tuna.tsinghua.edu.cn/simple"
}
$torchSuffix = if ($RuntimeProfile -eq "nvidia-cu126") { "+cu126" } else { "" }

if (-not $SkipInstall) {
    # Install the matching Torch family first from the selected index. Installing the common audio
    # requirements first can let a configured third-party mirror select a CPU
    # wheel for the NVIDIA profile.
    Invoke-External -FilePath $venvPython -Arguments @(
        "-m", "pip", "install", "--disable-pip-version-check",
        "--index-url", $torchIndex,
        "torch==$torchVersion$torchSuffix",
        "torchaudio==$torchAudioVersion$torchSuffix",
        "torchvision==$torchVisionVersion$torchSuffix"
    )
    Invoke-External -FilePath $venvPython -Arguments @(
        "-m", "pip", "install", "--disable-pip-version-check", "-r", $requirementsPath
    )
    Invoke-External -FilePath $venvPython -Arguments @(
        "-m", "pip", "install", "--disable-pip-version-check",
        "pyinstaller==6.14.2", "pytest==8.4.1", "imageio-ffmpeg==0.6.0"
    )
}

if (-not (Test-Path -LiteralPath (Join-Path $binDirectory "ffmpeg.exe") -PathType Leaf)) {
    New-Item -ItemType Directory -Path $binDirectory -Force | Out-Null
    $prepareFfmpeg = "import imageio_ffmpeg, pathlib, shutil; source=pathlib.Path(imageio_ffmpeg.get_ffmpeg_exe()); target=pathlib.Path(r'$binDirectory')/'ffmpeg.exe'; shutil.copy2(source, target); print(target)"
    Invoke-External -FilePath $venvPython -Arguments @("-c", $prepareFfmpeg)
}
Assert-File -Path (Join-Path $binDirectory "ffmpeg.exe")

$audioRuntimeCheck = @"
import importlib.util, torch, torchaudio, torchvision
required = ('audio_separator', 'funasr', 'huggingface_hub', 'nemo', 'onnxruntime')
missing = [name for name in required if importlib.util.find_spec(name) is None]
assert not missing, f'missing audio runtime packages: {missing}'
assert torch.__version__.startswith('$torchVersion'), torch.__version__
assert torchaudio.__version__.startswith('$torchAudioVersion'), torchaudio.__version__
assert torchvision.__version__.startswith('$torchVisionVersion'), torchvision.__version__
assert torchvision.extension._has_ops(), 'torchvision native operators are unavailable'
expected_cuda = $($RuntimeProfile -eq 'nvidia-cu126')
assert bool(torch.version.cuda) == expected_cuda, (torch.__version__, torch.version.cuda)
audio_is_cu126 = '+cu126' in torchaudio.__version__.lower()
vision_is_cu126 = '+cu126' in torchvision.__version__.lower()
assert audio_is_cu126 == expected_cuda, (torchaudio.__version__, expected_cuda)
assert vision_is_cu126 == expected_cuda, (torchvision.__version__, expected_cuda)
from audio_separator.separator import Separator
from audio_separator.separator.architectures.mdxc_separator import MDXCSeparator
from audio_separator.separator.roformer.roformer_loader import RoformerLoader
from funasr import AutoModel
import nemo.collections.asr
from nemo.collections.asr.models import ClusteringDiarizer
from nemo.collections.asr.models.configs.diarizer_config import NeuralDiarizerInferenceConfig
print('Audio runtime verified:', torch.__version__, torch.version.cuda)
"@
Invoke-External -FilePath $venvPython -Arguments @("-c", $audioRuntimeCheck)

$paddleCheck = if ($RuntimeProfile -eq "nvidia-cu126") {
    "from backend.ocr_engines import _import_paddle_ocr_class, detect_ocr_capabilities; caps=detect_ocr_capabilities(); assert caps.paddle_available, caps.error; assert caps.paddleocr_available; assert caps.cuda_compiled; assert _import_paddle_ocr_class(); print('NVIDIA Paddle CUDA runtime verified')"
}
else {
    "from backend.ocr_engines import _import_paddle_ocr_class, detect_ocr_capabilities; caps=detect_ocr_capabilities(); assert caps.paddle_available, caps.error; assert caps.paddleocr_available; assert not caps.cuda_compiled; assert _import_paddle_ocr_class(); print('$($RuntimeProfile.ToUpperInvariant()) Paddle CPU runtime verified')"
}
Invoke-External -FilePath $venvPython -Arguments @("-c", $paddleCheck)

if ($RequireAccelerator) {
    if ($RuntimeProfile -eq "nvidia-cu126") {
        Invoke-External -FilePath $venvPython -Arguments @(
            "-c", "import paddle; assert paddle.device.cuda.device_count() > 0; print(paddle.device.cuda.get_device_name(0))"
        )
        Invoke-External -FilePath $venvPython -Arguments @(
            "-c", "import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))"
        )
    }
    elseif ($RuntimeProfile -eq "amd") {
        Write-Warning "AMD acceleration belongs to the managed Vulkan local-model runtime and is validated after one-click deployment. OCR/UVR/ASR use CPU kernels in this profile."
    }
}

if (-not $SkipFrontend) {
    $npmCommand = Get-Command "npm.cmd" -ErrorAction Stop
    Invoke-External -FilePath $npmCommand.Source -Arguments @("ci", "--prefix", $webDirectory)
    Invoke-External -FilePath $npmCommand.Source -Arguments @("run", "build", "--prefix", $webDirectory)
}
Assert-File -Path (Join-Path $webDirectory "dist\index.html")

if (-not $SkipTests) {
    $env:KAOR_NO_BROWSER = "1"
    Invoke-External -FilePath $venvPython -Arguments @("-m", "pytest", (Join-Path $repositoryRoot "tests\backend"), "-q")
    Invoke-External -FilePath $venvPython -Arguments @(
        "-m", "compileall", "-q",
        (Join-Path $repositoryRoot "backend"),
        (Join-Path $repositoryRoot "kaor.py"),
        (Join-Path $repositoryRoot "scripts\kaor_frozen_entry.py")
    )
}

if (Test-Path -LiteralPath $buildRoot) {
    Remove-Item -LiteralPath $buildRoot -Recurse -Force
}
New-Item -ItemType Directory -Path $buildRoot -Force | Out-Null
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$pyinstallerDist = Join-Path $buildRoot "dist"
$pyinstallerWork = Join-Path $buildRoot "work"
$env:KAOR_BUILD_PROFILE = $RuntimeProfile
$env:KAOR_BUILD_VERSION = $Version
$env:KAOR_REPOSITORY_ROOT = $repositoryRoot
Invoke-External -FilePath $venvPython -Arguments @(
    "-m", "PyInstaller",
    "--noconfirm",
    "--clean",
    "--distpath", $pyinstallerDist,
    "--workpath", $pyinstallerWork,
    $specPath
)

$builtApplication = Join-Path $pyinstallerDist "Kaor"
Assert-File -Path (Join-Path $builtApplication "Kaor.exe")
Assert-File -Path (Join-Path $builtApplication "KaorAudioWorker.exe")

$flavor = switch ($RuntimeProfile) {
    "cpu" { "CPU" }
    "amd" { "AMD" }
    "nvidia-cu126" { "NVIDIA" }
}
$packageName = "Kaor-Windows-x64-$flavor"
$packageDirectory = Join-Path $outputRoot $packageName
if (Test-Path -LiteralPath $packageDirectory) {
    Remove-Item -LiteralPath $packageDirectory -Recurse -Force
}
Copy-Item -LiteralPath $builtApplication -Destination $packageDirectory -Recurse

$packageModelsDirectory = Join-Path $packageDirectory "models"
New-Item -ItemType Directory -Path $packageModelsDirectory -Force | Out-Null
$packageOcrModelsDirectory = Join-Path $packageModelsDirectory "paddlex\official_models"
foreach ($model in @($ocrManifest.models)) {
    $modelDirectoryName = [string]$model.directory
    $sourceModelDirectory = Join-Path $ocrModelsDirectory "official_models\$modelDirectoryName"
    $destinationModelDirectory = Join-Path $packageOcrModelsDirectory $modelDirectoryName
    New-Item -ItemType Directory -Path $destinationModelDirectory -Force | Out-Null
    foreach ($file in @($model.files)) {
        $fileName = [string]$file.name
        Copy-Item `
            -LiteralPath (Join-Path $sourceModelDirectory $fileName) `
            -Destination (Join-Path $destinationModelDirectory $fileName) `
            -Force
    }
}
if ($diarizationModelsBundled) {
    Copy-Item -LiteralPath $diarizationModelsDirectory -Destination $packageModelsDirectory -Recurse -Force
}
if (-not (Test-Path -LiteralPath (Join-Path $packageDirectory "bin") -PathType Container)) {
    Copy-Item -LiteralPath $binDirectory -Destination (Join-Path $packageDirectory "bin") -Recurse
}
else {
    Get-ChildItem -LiteralPath $binDirectory -Force | Copy-Item -Destination (Join-Path $packageDirectory "bin") -Recurse -Force
}
if ($LocalInferenceRuntimeDirectory) {
    $runtimeSource = [System.IO.Path]::GetFullPath($LocalInferenceRuntimeDirectory)
    if (-not (Test-Path -LiteralPath $runtimeSource -PathType Container)) {
        throw "Local inference runtime directory not found: $runtimeSource"
    }
    $runtimeDestination = Join-Path $packageDirectory "bin\local-inference"
    if (Test-Path -LiteralPath $runtimeDestination) {
        Remove-Item -LiteralPath $runtimeDestination -Recurse -Force
    }
    Copy-Item -LiteralPath $runtimeSource -Destination $runtimeDestination -Recurse
}

Copy-Item -LiteralPath $fontsDirectory -Destination (Join-Path $packageDirectory "fonts") -Recurse -Force
$packageDocsDirectory = Join-Path $packageDirectory "docs"
if (Test-Path -LiteralPath $packageDocsDirectory) {
    Remove-Item -LiteralPath $packageDocsDirectory -Recurse -Force
}
Copy-Item -LiteralPath $docsDirectory -Destination $packageDocsDirectory -Recurse -Force
New-Item -ItemType Directory -Path (Join-Path $packageDirectory "data") -Force | Out-Null

foreach ($document in @(
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md",
    "SECURITY.md"
)) {
    Copy-Item -LiteralPath (Join-Path $repositoryRoot $document) -Destination (Join-Path $packageDirectory $document) -Force
}
Copy-Item -LiteralPath (Join-Path $repositoryRoot "licenses") -Destination (Join-Path $packageDirectory "licenses") -Recurse -Force

$pythonDependencies = & $venvPython -m pip freeze --all
if ($LASTEXITCODE -ne 0) {
    throw "Failed to generate the Python dependency inventory."
}
$pythonDependencies | Set-Content -LiteralPath (Join-Path $packageDirectory "DEPENDENCIES-PYTHON.txt") -Encoding utf8

$acceleration = switch ($RuntimeProfile) {
    "cpu" { [ordered]@{ ocr = "cpu"; audio = "cpu"; local_translation = "cpu" } }
    "amd" { [ordered]@{ ocr = "cpu"; audio = "cpu"; local_translation = "vulkan-amd" } }
    "nvidia-cu126" { [ordered]@{ ocr = "cuda-12.6"; audio = "cuda-12.6"; local_translation = "cuda-or-vulkan" } }
}
$bundledModels = @("paddlex-ocr")
$onDemandModels = @("uvr-bs-roformer", "language-specific-asr", "local-translation-gguf")
if ($diarizationModelsBundled) {
    $bundledModels += "nemo-diarization-models"
}
else {
    $onDemandModels += "nemo-diarization-models"
}
$releaseMetadata = [ordered]@{
    schema_version = 1
    product = "Kaor"
    version = $Version
    runtime_profile = $RuntimeProfile
    platform = "windows"
    architecture = "x86_64"
    acceleration = $acceleration
    python = "3.12"
    torch = [ordered]@{
        version = $torchVersion
        index = $torchIndex
    }
    torchvision = $torchVisionVersion
    models = [ordered]@{
        bundled = $bundledModels
        on_demand = $onDemandModels
    }
    managed_local_runtime_bundled = [bool]$LocalInferenceRuntimeDirectory
    build_time_utc = [DateTime]::UtcNow.ToString("o")
}
$releaseMetadata | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath (Join-Path $packageDirectory "RELEASE.json") -Encoding utf8

@(
    "Kaor version: $Version"
    "Runtime profile: $RuntimeProfile"
    "Python build runtime: $($pythonInfo.Trim())"
    "Torch index: $torchIndex"
    "Model policy: OCR bundled; BS-Roformer assets, ASR, optional diarization, and local translation models use managed downloads"
    "Built at UTC: $([DateTime]::UtcNow.ToString('o'))"
) | Set-Content -LiteralPath (Join-Path $packageDirectory "BUILD-INFO.txt") -Encoding utf8

$requiredReleaseFiles = @(
    "Kaor.exe",
    "KaorAudioWorker.exe",
    "_internal\Cython\Utility\CppSupport.cpp",
    "bin\ffmpeg.exe",
    "README.md",
    "CHANGELOG.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "RELEASE.json",
    "DEPENDENCIES-PYTHON.txt",
    "models\paddlex\official_models\PP-OCRv6_medium_det\inference.json",
    "models\paddlex\official_models\PP-OCRv6_medium_det\inference.pdiparams",
    "models\paddlex\official_models\PP-OCRv6_medium_det\inference.yml",
    "models\paddlex\official_models\PP-OCRv6_medium_det\README.md",
    "models\paddlex\official_models\PP-OCRv6_medium_rec\inference.json",
    "models\paddlex\official_models\PP-OCRv6_medium_rec\inference.pdiparams",
    "models\paddlex\official_models\PP-OCRv6_medium_rec\inference.yml",
    "models\paddlex\official_models\PP-OCRv6_medium_rec\README.md",
    "docs\USER_GUIDE.zh-CN.md",
    "docs\TROUBLESHOOTING.zh-CN.md",
    "docs\ARCHITECTURE.zh-CN.md",
    "docs\images\kaor-workbench.png",
    "licenses\FFMPEG-SOURCE.txt",
    "licenses\GPL-3.0.txt",
    "licenses\APACHE-2.0.txt",
    "licenses\PP-OCRV6-MODEL-MANIFEST.json",
    "licenses\PP-OCRV6-MODEL-NOTICE.txt",
    "licenses\GPT-SoVITS-MIT.txt",
    "licenses\BS-ROFORMER-MODEL-NOTICE.txt",
    "licenses\LLAMA_CPP-MIT.txt",
    "licenses\QWEN3-NOTICE.txt",
    "licenses\ZFTURBO-MSS-MIT.txt",
    "fonts\NotoSansSC\NotoSansSC-Regular.ttf",
    "fonts\NotoSansSC\LICENSE"
)
if ($diarizationModelsBundled) {
    $requiredReleaseFiles += @(
        $diarizationModelNames | ForEach-Object { "models\diarization\$_" }
    )
}
foreach ($relativePath in $requiredReleaseFiles) {
    Assert-File -Path (Join-Path $packageDirectory $relativePath)
}
$packagedOcrModels = @(Get-ChildItem -LiteralPath (Join-Path $packageModelsDirectory "paddlex") -Recurse -Filter "*.pdiparams" -File)
if ($packagedOcrModels.Count -lt 2) {
    throw "Release validation failed; detector and recognizer OCR models were not packaged."
}
foreach ($assetName in @($uvrModelName, $uvrConfigName)) {
    if (Test-Path -LiteralPath (Join-Path $packageModelsDirectory "uvr\$assetName") -PathType Leaf) {
        throw "Release validation failed; BS-Roformer assets must be downloaded on first use, not redistributed in the portable archive: $assetName"
    }
}
if (Test-Path -LiteralPath (Join-Path $packageModelsDirectory "asr")) {
    throw "Release validation failed; downloaded ASR checkpoints must not be embedded in every runtime profile."
}

# Run the console-enabled companion executable. This catches PyInstaller hidden
# import regressions and proves the portable audio worker has the same runtime
# as the main application without requiring Python on the target machine.
$previousErrorActionPreference = $ErrorActionPreference
try {
    # Several ML libraries emit informational startup messages on stderr. Capture
    # them for diagnostics and judge the worker by its exit code plus JSON result.
    $ErrorActionPreference = "Continue"
    $probeLines = @(& (Join-Path $packageDirectory "KaorAudioWorker.exe") probe 2>&1)
    $probeExitCode = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $previousErrorActionPreference
}
if ($probeExitCode -ne 0) {
    throw "Packaged audio worker probe failed: $($probeLines -join [Environment]::NewLine)"
}
$probeJson = $probeLines | Where-Object { $_.ToString().TrimStart().StartsWith("{") } | Select-Object -Last 1
if (-not $probeJson) {
    throw "Packaged audio worker did not emit a JSON probe result."
}
$probe = $probeJson.ToString() | ConvertFrom-Json
foreach ($requiredField in @("torch_version", "torch_cuda_version", "error")) {
    if ($probe.PSObject.Properties.Name -notcontains $requiredField) {
        throw "Packaged audio worker probe did not report $requiredField."
    }
}
if (-not [string]::IsNullOrWhiteSpace([string]$probe.error)) {
    throw "Packaged audio worker runtime import probe failed: $($probe.error)"
}
if ([string]::IsNullOrWhiteSpace([string]$probe.torch_version) -or
    -not ([string]$probe.torch_version).StartsWith($torchVersion, [System.StringComparison]::Ordinal)) {
    throw "Packaged audio worker reported an invalid Torch version: $($probe.torch_version)"
}
foreach ($property in @(
    "torch_available",
    "torchvision_available",
    "torchvision_ops_available",
    "audio_separator_available",
    "nemo_available",
    "funasr_available"
)) {
    if (-not $probe.$property) {
        throw "Packaged audio worker is missing $property."
    }
}
$probeHasCudaRuntime = -not [string]::IsNullOrWhiteSpace([string]$probe.torch_cuda_version)
if ($RuntimeProfile -eq "nvidia-cu126" -and -not $probeHasCudaRuntime) {
    throw "Packaged NVIDIA audio worker contains a CPU-only Torch runtime."
}
if ($RuntimeProfile -ne "nvidia-cu126" -and $probeHasCudaRuntime) {
    throw "Packaged $RuntimeProfile audio worker unexpectedly contains a CUDA Torch runtime: $($probe.torch_cuda_version)"
}
if ($RuntimeProfile -eq "nvidia-cu126" -and $RequireAccelerator -and -not $probe.cuda_available) {
    throw "Packaged NVIDIA audio worker did not detect CUDA on this build host."
}
$probe | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $packageDirectory "AUDIO-RUNTIME-PROBE.json") -Encoding utf8

$manifestPath = Join-Path $packageDirectory "SHA256SUMS.txt"
$manifestLines = foreach ($file in @(Get-ChildItem -LiteralPath $packageDirectory -Recurse -File | Sort-Object FullName)) {
    if ($file.FullName -eq $manifestPath) {
        continue
    }
    $relative = Get-RelativePackagePath -FullName $file.FullName -PackageRoot $packageDirectory
    $digest = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$digest *$relative"
}
$manifestLines | Set-Content -LiteralPath $manifestPath -Encoding ascii

if ($SkipArchive) {
    Write-Host "Portable directory ready: $packageDirectory"
    exit 0
}

$publishedArchivePath = Join-Path $outputRoot "$packageName.zip"
$archivePath = if ($RuntimeProfile -eq "nvidia-cu126") {
    Join-Path $buildRoot "$packageName.zip"
}
else {
    $publishedArchivePath
}
$checksumPath = "$publishedArchivePath.sha256"
$summaryPath = "$publishedArchivePath.release.json"
foreach ($oldPath in @($archivePath, $publishedArchivePath, $checksumPath, $summaryPath)) {
    if (Test-Path -LiteralPath $oldPath) {
        Remove-Item -LiteralPath $oldPath -Force
    }
}
if ($RuntimeProfile -eq "nvidia-cu126") {
    $escapedPackageName = [Regex]::Escape($packageName)
    Get-ChildItem -LiteralPath $outputRoot -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "^${escapedPackageName}\.zip\.\d{3}$" } |
        Remove-Item -Force
    foreach ($oldSplitAsset in @(
        "$packageName.parts.json",
        "$packageName.parts.sha256",
        "$packageName-Setup.exe",
        "$packageName-Setup.cs"
    )) {
        $oldSplitPath = Join-Path $outputRoot $oldSplitAsset
        if (Test-Path -LiteralPath $oldSplitPath -PathType Leaf) {
            Remove-Item -LiteralPath $oldSplitPath -Force
        }
    }
}

$tarCommand = Get-Command "tar.exe" -ErrorAction SilentlyContinue
if ($null -ne $tarCommand) {
    Push-Location $outputRoot
    try {
        Invoke-External -FilePath $tarCommand.Source -Arguments @("-a", "-c", "-f", $archivePath, $packageName)
    }
    finally {
        Pop-Location
    }
}
else {
    Compress-Archive -LiteralPath $packageDirectory -DestinationPath $archivePath -CompressionLevel Optimal
}
Assert-File -Path $archivePath
$archive = Get-Item -LiteralPath $archivePath
if ($RuntimeProfile -eq "nvidia-cu126") {
    $unpackedSizeBytes = [long](
        Get-ChildItem -LiteralPath $packageDirectory -Recurse -File |
            Measure-Object -Property Length -Sum
    ).Sum
    $splitScript = Join-Path $PSScriptRoot "split-portable-archive.ps1"
    Assert-File -Path $splitScript
    Invoke-External -FilePath "powershell.exe" -Arguments @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $splitScript,
        "-ArchivePath", $archivePath,
        "-OutputDirectory", $outputRoot,
        "-PackageDirectoryName", $packageName,
        "-UnpackedSizeBytes", $unpackedSizeBytes.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "-Version", $Version,
        "-PartSizeBytes", (1900MB).ToString([System.Globalization.CultureInfo]::InvariantCulture)
    )

    $partsManifestPath = Join-Path $outputRoot "$packageName.parts.json"
    $partsChecksumPath = Join-Path $outputRoot "$packageName.parts.sha256"
    $setupPath = Join-Path $outputRoot "$packageName-Setup.exe"
    $setupSourcePath = Join-Path $outputRoot "$packageName-Setup.cs"
    foreach ($requiredSplitAsset in @($partsManifestPath, $partsChecksumPath, $setupPath, $setupSourcePath)) {
        Assert-File -Path $requiredSplitAsset
    }
    $partFiles = @(Get-ChildItem -LiteralPath $outputRoot -Filter "$packageName.zip.???" -File)
    if ($partFiles.Count -lt 2) {
        throw "NVIDIA split release validation failed; expected at least two archive parts."
    }
    foreach ($partFile in $partFiles) {
        if ($partFile.Length -ge $githubReleaseAssetLimit) {
            throw "NVIDIA split release part exceeds GitHub's 2 GiB limit: $($partFile.Name)"
        }
    }

    Write-Host "NVIDIA portable split release ready: $outputRoot"
    Write-Host "Complete archive size: $($archive.Length) bytes"
    Write-Host "Release parts: $($partFiles.Count)"
    Write-Host "Double-click installer: $setupPath"
    exit 0
}

if (-not $AllowOversizedArchive -and $archive.Length -gt $githubReleaseAssetLimit) {
    throw "Archive is $($archive.Length) bytes, exceeding GitHub's 2 GiB per-asset limit. Remove unintended runtime/model files before publishing."
}
$archivePath = $publishedArchivePath
if (-not (Test-Path -LiteralPath $archivePath -PathType Leaf)) {
    Copy-Item -LiteralPath $archive.FullName -Destination $archivePath -Force
    $archive = Get-Item -LiteralPath $archivePath
}
$hash = (Get-FileHash -LiteralPath $archivePath -Algorithm SHA256).Hash.ToLowerInvariant()
"$hash *$packageName.zip" | Set-Content -LiteralPath $checksumPath -Encoding ascii
[ordered]@{
    schema_version = 1
    product = "Kaor"
    version = $Version
    runtime_profile = $RuntimeProfile
    asset = "$packageName.zip"
    size_bytes = $archive.Length
    sha256 = $hash
} | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $summaryPath -Encoding utf8

Write-Host "Portable archive ready: $archivePath"
Write-Host "Archive size: $($archive.Length) bytes"
Write-Host "SHA-256: $hash"
