[CmdletBinding()]
param(
    [string]$WorkingDirectory = ".build\split-roundtrip"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $false)][string[]]$Arguments = @()
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Start-SetupProcess {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    $quotedArguments = @($Arguments | ForEach-Object { '"' + ($_ -replace '"', '\"') + '"' })
    return Start-Process `
        -FilePath $FilePath `
        -ArgumentList $quotedArguments `
        -WorkingDirectory (Split-Path -Parent $FilePath) `
        -Wait `
        -PassThru
}

function Invoke-SetupChecked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    $process = Start-SetupProcess -FilePath $FilePath -Arguments $Arguments
    if ($process.ExitCode -ne 0) {
        $errorLog = Join-Path `
            (Split-Path -Parent $FilePath) `
            (([System.IO.Path]::GetFileNameWithoutExtension($FilePath)) + ".error.log")
        $details = if (Test-Path -LiteralPath $errorLog) {
            Get-Content -LiteralPath $errorLog -Raw -Encoding utf8
        }
        else {
            "No Setup error log was written."
        }
        throw "Setup failed with exit code $($process.ExitCode): $details"
    }
}

function Get-FileInventory {
    param([Parameter(Mandatory = $true)][string]$Root)
    $prefix = [System.IO.Path]::GetFullPath($Root).TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
    $inventory = [ordered]@{}
    foreach ($file in @(Get-ChildItem -LiteralPath $Root -Recurse -File | Sort-Object FullName)) {
        $relative = $file.FullName.Substring($prefix.Length).Replace('\', '/')
        $inventory[$relative] = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    }
    return $inventory
}

function Add-TestZipEntry {
    param(
        [Parameter(Mandatory = $true)][System.IO.Compression.ZipArchive]$Archive,
        [Parameter(Mandatory = $true)][string]$Name,
        [byte[]]$Content = @(),
        [int]$ExternalAttributes = 0
    )
    $entry = $Archive.CreateEntry($Name, [System.IO.Compression.CompressionLevel]::NoCompression)
    if ($ExternalAttributes -ne 0) {
        $entry.ExternalAttributes = $ExternalAttributes
    }
    if ($Content.Length -gt 0) {
        $stream = $entry.Open()
        try {
            $stream.Write($Content, 0, $Content.Length)
        }
        finally {
            $stream.Dispose()
        }
    }
}

function Test-RejectedArchive {
    param(
        [Parameter(Mandatory = $true)][ValidateSet("traversal", "duplicate", "symlink", "ads")][string]$Kind,
        [Parameter(Mandatory = $true)][string]$ExpectedError,
        [Parameter(Mandatory = $true)][string]$Root,
        [Parameter(Mandatory = $true)][string]$SplitScript
    )

    $caseRoot = Join-Path $Root $Kind
    $releaseRoot = Join-Path $caseRoot "release"
    $installRoot = Join-Path $caseRoot "installed"
    New-Item -ItemType Directory -Path $releaseRoot -Force | Out-Null

    $archivePath = Join-Path $caseRoot "Kaor-Windows-x64-NVIDIA.zip"
    $archiveStream = [System.IO.File]::Open(
        $archivePath,
        [System.IO.FileMode]::CreateNew,
        [System.IO.FileAccess]::ReadWrite,
        [System.IO.FileShare]::None
    )
    try {
        $archive = [System.IO.Compression.ZipArchive]::new(
            $archiveStream,
            [System.IO.Compression.ZipArchiveMode]::Create,
            $false
        )
        try {
            Add-TestZipEntry -Archive $archive -Name "Kaor-Windows-x64-NVIDIA/Kaor.exe" -Content ([Text.Encoding]::ASCII.GetBytes("fixture"))
            $payload = New-Object byte[] 4KB
            [System.Random]::new(20260805 + $Kind.Length).NextBytes($payload)
            Add-TestZipEntry -Archive $archive -Name "Kaor-Windows-x64-NVIDIA/runtime/payload.bin" -Content $payload

            if ($Kind -eq "traversal") {
                Add-TestZipEntry -Archive $archive -Name "Kaor-Windows-x64-NVIDIA/../../outside.txt" -Content ([Text.Encoding]::ASCII.GetBytes("outside"))
            }
            elseif ($Kind -eq "duplicate") {
                Add-TestZipEntry -Archive $archive -Name "Kaor-Windows-x64-NVIDIA/runtime/"
                Add-TestZipEntry -Archive $archive -Name "Kaor-Windows-x64-NVIDIA/runtime/"
            }
            elseif ($Kind -eq "symlink") {
                # 0xA1FF0000 is a Unix symbolic-link mode (0120777) in ZIP external attributes.
                Add-TestZipEntry `
                    -Archive $archive `
                    -Name "Kaor-Windows-x64-NVIDIA/runtime/link" `
                    -Content ([Text.Encoding]::UTF8.GetBytes("../Kaor.exe")) `
                    -ExternalAttributes -1577123840
            }
            else {
                Add-TestZipEntry `
                    -Archive $archive `
                    -Name "Kaor-Windows-x64-NVIDIA/Kaor.exe:payload" `
                    -Content ([Text.Encoding]::ASCII.GetBytes("alternate stream"))
            }
        }
        finally {
            $archive.Dispose()
        }
    }
    finally {
        $archiveStream.Dispose()
    }

    $readStream = [System.IO.File]::OpenRead($archivePath)
    try {
        $readArchive = [System.IO.Compression.ZipArchive]::new(
            $readStream,
            [System.IO.Compression.ZipArchiveMode]::Read,
            $false
        )
        try {
            [long]$unpackedSize = 0
            foreach ($entry in $readArchive.Entries) {
                $unpackedSize += $entry.Length
            }
        }
        finally {
            $readArchive.Dispose()
        }
    }
    finally {
        $readStream.Dispose()
    }

    & $SplitScript `
        -ArchivePath $archivePath `
        -OutputDirectory $releaseRoot `
        -PackageDirectoryName "Kaor-Windows-x64-NVIDIA" `
        -UnpackedSizeBytes $unpackedSize `
        -Version "0.2.0-test" `
        -PartSizeBytes 1KB | Out-Null

    $manifestPath = Join-Path $releaseRoot "Kaor-Windows-x64-NVIDIA.parts.json"
    $setupPath = Join-Path $releaseRoot "Kaor-Windows-x64-NVIDIA-Setup.exe"
    $process = Start-SetupProcess -FilePath $setupPath -Arguments @("--headless", $manifestPath, $installRoot)
    if ($process.ExitCode -eq 0) {
        throw "The installer accepted the $Kind attack archive."
    }

    $errorLog = Join-Path $releaseRoot "Kaor-Windows-x64-NVIDIA-Setup.error.log"
    $errorText = Get-Content -LiteralPath $errorLog -Raw -Encoding utf8
    if ($errorText -notmatch [Regex]::Escape($ExpectedError)) {
        throw "The $Kind archive failed for an unexpected reason: $errorText"
    }
    if ((Test-Path -LiteralPath (Join-Path $caseRoot "outside.txt")) -or
        (Test-Path -LiteralPath (Join-Path $installRoot "outside.txt"))) {
        throw "The traversal fixture wrote outside the staging directory."
    }
    if (Test-Path -LiteralPath (Join-Path $installRoot "Kaor-Windows-x64-NVIDIA")) {
        throw "The rejected $Kind archive left an installed package behind."
    }
    $temporaryItems = @(
        Get-ChildItem -LiteralPath $installRoot -Force -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -like ".kaor-extract-*" -or $_.Name -like ".Kaor-Windows-x64-NVIDIA.zip.assembling-*" }
    )
    if ($temporaryItems.Count -ne 0) {
        throw "The rejected $Kind archive left temporary extraction files behind."
    }
}

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$resolvedWork = if ([System.IO.Path]::IsPathRooted($WorkingDirectory)) {
    [System.IO.Path]::GetFullPath($WorkingDirectory)
}
else {
    [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot $WorkingDirectory))
}
$repositoryPrefix = $repositoryRoot.TrimEnd('\', '/') + [System.IO.Path]::DirectorySeparatorChar
if (-not $resolvedWork.StartsWith($repositoryPrefix, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Roundtrip test directory must stay inside the repository: $resolvedWork"
}
if (Test-Path -LiteralPath $resolvedWork) {
    Remove-Item -LiteralPath $resolvedWork -Recurse -Force
}

$fixtureRoot = Join-Path $resolvedWork "source\Kaor-Windows-x64-NVIDIA"
$nestedRoot = Join-Path $fixtureRoot "runtime\nested"
$releaseRoot = Join-Path $resolvedWork "release"
$installRoot = Join-Path $resolvedWork "installed"
New-Item -ItemType Directory -Path $nestedRoot -Force | Out-Null
New-Item -ItemType Directory -Path $releaseRoot -Force | Out-Null

"portable executable fixture" | Set-Content -LiteralPath (Join-Path $fixtureRoot "Kaor.exe") -Encoding ascii
"runtime file" | Set-Content -LiteralPath (Join-Path $nestedRoot "runtime.txt") -Encoding utf8
$payload = New-Object byte[] 384KB
$random = [System.Random]::new(20260805)
$random.NextBytes($payload)
[System.IO.File]::WriteAllBytes((Join-Path $fixtureRoot "runtime\payload.bin"), $payload)

$archivePath = Join-Path $resolvedWork "Kaor-Windows-x64-NVIDIA.zip"
Compress-Archive -LiteralPath $fixtureRoot -DestinationPath $archivePath -CompressionLevel Optimal
$unpackedSize = (Get-ChildItem -LiteralPath $fixtureRoot -Recurse -File | Measure-Object -Property Length -Sum).Sum
$partSizeArgument = ([long](64KB)).ToString([System.Globalization.CultureInfo]::InvariantCulture)

$splitScript = Join-Path $PSScriptRoot "split-portable-archive.ps1"
$manifestOutput = @(
    & powershell.exe `
        -NoProfile `
        -ExecutionPolicy Bypass `
        -File $splitScript `
        -ArchivePath $archivePath `
        -OutputDirectory $releaseRoot `
        -PackageDirectoryName "Kaor-Windows-x64-NVIDIA" `
        -UnpackedSizeBytes ([long]$unpackedSize) `
        -Version "0.2.0-test" `
        -PartSizeBytes $partSizeArgument
)
if ($LASTEXITCODE -ne 0) {
    throw "Split script failed with exit code $LASTEXITCODE."
}

$manifestPath = $manifestOutput | Where-Object { $_ -is [string] -and $_.EndsWith(".parts.json") } | Select-Object -Last 1
if (-not $manifestPath) {
    $manifestPath = Join-Path $releaseRoot "Kaor-Windows-x64-NVIDIA.parts.json"
}
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
$parts = @(Get-ChildItem -LiteralPath $releaseRoot -Filter "*.zip.???" -File | Sort-Object Name)
if ($parts.Count -lt 2 -or $parts.Count -ne @($manifest.parts).Count) {
    throw "Expected multiple complete release parts."
}
foreach ($part in $parts) {
    if ($part.Length -gt 64KB) {
        throw "Roundtrip test generated an oversized part: $($part.Name)"
    }
}
foreach ($required in @(
    "Kaor-Windows-x64-NVIDIA.parts.json",
    "Kaor-Windows-x64-NVIDIA.parts.sha256",
    "Kaor-Windows-x64-NVIDIA-Setup.exe",
    "Kaor-Windows-x64-NVIDIA-Setup.cs"
)) {
    if (-not (Test-Path -LiteralPath (Join-Path $releaseRoot $required) -PathType Leaf)) {
        throw "Missing roundtrip release asset: $required"
    }
}

$setupPath = Join-Path $releaseRoot "Kaor-Windows-x64-NVIDIA-Setup.exe"
Invoke-SetupChecked -FilePath $setupPath -Arguments @("--verify-only", $manifestPath)
Invoke-SetupChecked -FilePath $setupPath -Arguments @("--headless", $manifestPath, $installRoot)

$sourceInventory = Get-FileInventory -Root $fixtureRoot
$installedFixture = Join-Path $installRoot "Kaor-Windows-x64-NVIDIA"
$installedInventory = Get-FileInventory -Root $installedFixture
if ($sourceInventory.Count -ne $installedInventory.Count) {
    throw "Extracted file count differs from the source fixture."
}
foreach ($relative in $sourceInventory.Keys) {
    if (-not $installedInventory.Contains($relative) -or $installedInventory[$relative] -ne $sourceInventory[$relative]) {
        throw "Extracted fixture mismatch: $relative"
    }
}

$tamperedPart = $parts[0].FullName
$backupPart = "$tamperedPart.original"
Copy-Item -LiteralPath $tamperedPart -Destination $backupPart
try {
    $stream = [System.IO.File]::Open($tamperedPart, [System.IO.FileMode]::Open, [System.IO.FileAccess]::ReadWrite, [System.IO.FileShare]::None)
    try {
        $value = $stream.ReadByte()
        if ($value -lt 0) {
            throw "The first release part was unexpectedly empty."
        }
        $stream.Position = 0
        $stream.WriteByte($value -bxor 0x01)
        $stream.Flush($true)
    }
    finally {
        $stream.Dispose()
    }

    $tamperProcess = Start-SetupProcess -FilePath $setupPath -Arguments @("--verify-only", $manifestPath)
    if ($tamperProcess.ExitCode -eq 0) {
        throw "The assembler accepted a tampered release part."
    }
}
finally {
    Move-Item -LiteralPath $backupPart -Destination $tamperedPart -Force
}

Invoke-SetupChecked -FilePath $setupPath -Arguments @("--verify-only", $manifestPath)

$adversarialRoot = Join-Path $resolvedWork "adversarial"
Test-RejectedArchive `
    -Kind "traversal" `
    -ExpectedError "Unsafe Windows archive name" `
    -Root $adversarialRoot `
    -SplitScript $splitScript
Test-RejectedArchive `
    -Kind "duplicate" `
    -ExpectedError "Duplicate archive path" `
    -Root $adversarialRoot `
    -SplitScript $splitScript
Test-RejectedArchive `
    -Kind "symlink" `
    -ExpectedError "Symbolic links are not accepted" `
    -Root $adversarialRoot `
    -SplitScript $splitScript
Test-RejectedArchive `
    -Kind "ads" `
    -ExpectedError "Unsafe Windows archive name" `
    -Root $adversarialRoot `
    -SplitScript $splitScript

$failureReleaseRoot = Join-Path $resolvedWork "split-failure-cleanup"
New-Item -ItemType Directory -Path $failureReleaseRoot -Force | Out-Null
$lockedArchive = [System.IO.File]::Open(
    $archivePath,
    [System.IO.FileMode]::Open,
    [System.IO.FileAccess]::Read,
    [System.IO.FileShare]::None
)
$splitFailed = $false
try {
    try {
        & $splitScript `
            -ArchivePath $archivePath `
            -OutputDirectory $failureReleaseRoot `
            -PackageDirectoryName "Kaor-Windows-x64-NVIDIA" `
            -UnpackedSizeBytes ([long]$unpackedSize) `
            -Version "0.2.0-test" `
            -PartSizeBytes 64KB | Out-Null
    }
    catch {
        $splitFailed = $true
    }
}
finally {
    $lockedArchive.Dispose()
}
if (-not $splitFailed) {
    throw "The locked source archive did not trigger the split failure fixture."
}
$incompleteAssets = @(Get-ChildItem -LiteralPath $failureReleaseRoot -File -Force)
if ($incompleteAssets.Count -ne 0) {
    throw "A failed split left incomplete release assets: $($incompleteAssets.Name -join ', ')"
}

Write-Host "Portable split roundtrip passed: $resolvedWork"
