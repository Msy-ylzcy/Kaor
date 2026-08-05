[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string]$ArchivePath,

    [Parameter(Mandatory = $true)]
    [string]$OutputDirectory,

    [Parameter(Mandatory = $true)]
    [string]$PackageDirectoryName,

    [Parameter(Mandatory = $true)]
    [long]$UnpackedSizeBytes,

    [string]$Version = "0.2.0",

    [ValidateRange(1KB, 2047MB)]
    [long]$PartSizeBytes = 1900MB,

    [string]$Compiler = "",

    [string]$AssemblerSource = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Windows PowerShell 5.1 does not initialize $PSScriptRoot while parameter
# default expressions are evaluated when this script is launched with -File.
if ([string]::IsNullOrWhiteSpace($AssemblerSource)) {
    $AssemblerSource = Join-Path $PSScriptRoot "portable-split-assembler\Program.cs"
}

$bufferSize = 8MB
$githubAssetLimit = 2GB

function Assert-File {
    param([Parameter(Mandatory = $true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file not found: $Path"
    }
}

function Resolve-CSharpCompiler {
    param([string]$RequestedCompiler)

    if ($RequestedCompiler) {
        $resolved = [System.IO.Path]::GetFullPath($RequestedCompiler)
        Assert-File -Path $resolved
        return $resolved
    }

    $candidates = @(
        (Join-Path $env:WINDIR "Microsoft.NET\Framework64\v4.0.30319\csc.exe"),
        (Join-Path $env:WINDIR "Microsoft.NET\Framework\v4.0.30319\csc.exe")
    )
    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate -PathType Leaf) {
            return $candidate
        }
    }

    $command = Get-Command "csc.exe" -ErrorAction SilentlyContinue
    if ($null -ne $command) {
        return $command.Source
    }
    throw "C# compiler not found. The release build requires the Windows .NET Framework csc.exe so the offline Setup tool is reproducible and auditable."
}

function Invoke-Checked {
    param(
        [Parameter(Mandatory = $true)][string]$FilePath,
        [Parameter(Mandatory = $true)][string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $FilePath $($Arguments -join ' ')"
    }
}

function Get-LowerSha256 {
    param([Parameter(Mandatory = $true)][string]$Path)
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

if (-not [Environment]::Is64BitOperatingSystem) {
    throw "NVIDIA portable release splitting requires 64-bit Windows."
}
if ($Version -notmatch '^\d+\.\d+\.\d+([-.][0-9A-Za-z.-]+)?$') {
    throw "Version must be a SemVer-compatible value without a leading v: $Version"
}
if ($PartSizeBytes -ge $githubAssetLimit) {
    throw "PartSizeBytes must remain below GitHub's 2 GiB per-asset limit."
}
if ($UnpackedSizeBytes -le 0) {
    throw "UnpackedSizeBytes must be positive."
}
if ([string]::IsNullOrWhiteSpace($PackageDirectoryName)) {
    throw "PackageDirectoryName must not be empty."
}

$resolvedArchive = [System.IO.Path]::GetFullPath($ArchivePath)
$resolvedOutput = [System.IO.Path]::GetFullPath($OutputDirectory)
$resolvedSource = [System.IO.Path]::GetFullPath($AssemblerSource)
Assert-File -Path $resolvedArchive
Assert-File -Path $resolvedSource
if ([System.IO.Path]::GetExtension($resolvedArchive) -ine ".zip") {
    throw "Only a complete ZIP archive can be split: $resolvedArchive"
}
if ([System.IO.Path]::GetFileName($PackageDirectoryName) -ne $PackageDirectoryName) {
    throw "PackageDirectoryName must be a single directory name."
}

$archive = Get-Item -LiteralPath $resolvedArchive
if ($archive.Length -le $PartSizeBytes) {
    throw "The archive fits in one part ($($archive.Length) bytes); use a normal single-ZIP release."
}

New-Item -ItemType Directory -Path $resolvedOutput -Force | Out-Null
$archiveName = $archive.Name
$assetPrefix = [System.IO.Path]::GetFileNameWithoutExtension($archiveName)
if ($archiveName -cne "Kaor-Windows-x64-NVIDIA.zip" -or $assetPrefix -cne "Kaor-Windows-x64-NVIDIA") {
    throw "Unexpected NVIDIA archive name: $archiveName"
}
if ($PackageDirectoryName -cne $assetPrefix) {
    throw "PackageDirectoryName must exactly match the NVIDIA archive root: $assetPrefix"
}

$manifestPath = Join-Path $resolvedOutput "$assetPrefix.parts.json"
$checksumPath = Join-Path $resolvedOutput "$assetPrefix.parts.sha256"
$setupPath = Join-Path $resolvedOutput "$assetPrefix-Setup.exe"
$publishedSourcePath = Join-Path $resolvedOutput "$assetPrefix-Setup.cs"
if ($resolvedSource.Equals(
        [System.IO.Path]::GetFullPath($publishedSourcePath),
        [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "AssemblerSource must not be the generated Setup source path: $publishedSourcePath"
}

$escapedPrefix = [Regex]::Escape($assetPrefix)
function Remove-SplitAssets {
    # This exact prefix is fixed above, so unrelated release assets are retained.
    Get-ChildItem -LiteralPath $resolvedOutput -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match "^${escapedPrefix}\.zip\.\d{3}$" } |
        Remove-Item -Force
    foreach ($generatedPath in @($manifestPath, $checksumPath, $setupPath, $publishedSourcePath)) {
        if (Test-Path -LiteralPath $generatedPath -PathType Leaf) {
            Remove-Item -LiteralPath $generatedPath -Force
        }
    }
}

Remove-SplitAssets
try {
    $compilerPath = Resolve-CSharpCompiler -RequestedCompiler $Compiler
    $compilerArguments = @(
        "/nologo",
        "/optimize+",
        "/target:winexe",
        "/platform:anycpu",
        "/out:$setupPath",
        "/reference:System.dll",
        "/reference:System.Core.dll",
        "/reference:System.Drawing.dll",
        "/reference:System.Windows.Forms.dll",
        "/reference:System.Runtime.Serialization.dll",
        "/reference:System.IO.Compression.dll",
        "/reference:System.IO.Compression.FileSystem.dll",
        $resolvedSource
    )
    Invoke-Checked -FilePath $compilerPath -Arguments $compilerArguments
    Assert-File -Path $setupPath
    Copy-Item -LiteralPath $resolvedSource -Destination $publishedSourcePath -Force

    $partRecords = [System.Collections.Generic.List[object]]::new()
    $input = [System.IO.File]::Open($resolvedArchive, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Read, [System.IO.FileShare]::Read)
    try {
        $buffer = New-Object byte[] $bufferSize
        $partNumber = 1
        while ($input.Position -lt $input.Length) {
            if ($partNumber -gt 999) {
                throw "The archive requires more than 999 parts; increase PartSizeBytes."
            }
            $partName = "{0}.zip.{1:D3}" -f $assetPrefix, $partNumber
            $partPath = Join-Path $resolvedOutput $partName
            $part = [System.IO.File]::Open($partPath, [System.IO.FileMode]::CreateNew, [System.IO.FileAccess]::Write, [System.IO.FileShare]::None)
            try {
                [long]$written = 0
                while ($written -lt $PartSizeBytes -and $input.Position -lt $input.Length) {
                    $remaining = [Math]::Min([long]$buffer.Length, $PartSizeBytes - $written)
                    $read = $input.Read($buffer, 0, [int]$remaining)
                    if ($read -le 0) {
                        break
                    }
                    $part.Write($buffer, 0, $read)
                    $written += $read
                }
                $part.Flush($true)
            }
            finally {
                $part.Dispose()
            }

            $partInfo = Get-Item -LiteralPath $partPath
            if ($partInfo.Length -le 0 -or $partInfo.Length -gt $PartSizeBytes -or $partInfo.Length -ge $githubAssetLimit) {
                throw "Generated release part has an invalid size: $partName ($($partInfo.Length) bytes)"
            }
            $partRecords.Add([pscustomobject][ordered]@{
                name = $partName
                size_bytes = $partInfo.Length
                sha256 = Get-LowerSha256 -Path $partPath
            })
            $partNumber += 1
        }
    }
    finally {
        $input.Dispose()
    }

    if ($partRecords.Count -lt 2) {
        throw "A split NVIDIA release must contain at least two parts."
    }
    $partTotal = ($partRecords | Measure-Object -Property size_bytes -Sum).Sum
    if ([long]$partTotal -ne $archive.Length) {
        throw "Split size mismatch: expected $($archive.Length), generated $partTotal bytes."
    }

    $setupInfo = Get-Item -LiteralPath $setupPath
    $sourceInfo = Get-Item -LiteralPath $publishedSourcePath
    $manifest = [ordered]@{
        schema_version = 1
        product = "Kaor"
        version = $Version
        runtime_profile = "nvidia-cu126"
        archive_name = $archiveName
        archive_size_bytes = $archive.Length
        archive_sha256 = Get-LowerSha256 -Path $resolvedArchive
        unpacked_size_bytes = $UnpackedSizeBytes
        package_directory = $PackageDirectoryName
        part_size_bytes = $PartSizeBytes
        parts = @($partRecords)
        assembler = [ordered]@{
            name = $setupInfo.Name
            size_bytes = $setupInfo.Length
            sha256 = Get-LowerSha256 -Path $setupPath
        }
        assembler_source = [ordered]@{
            name = $sourceInfo.Name
            size_bytes = $sourceInfo.Length
            sha256 = Get-LowerSha256 -Path $publishedSourcePath
        }
    }
    $manifestJson = $manifest | ConvertTo-Json -Depth 8
    [System.IO.File]::WriteAllText(
        $manifestPath,
        $manifestJson,
        [System.Text.UTF8Encoding]::new($false)
    )

    $lines = [System.Collections.Generic.List[string]]::new()
    $lines.Add("$($manifest.archive_sha256) *$archiveName")
    foreach ($record in $partRecords) {
        $lines.Add("$($record.sha256) *$($record.name)")
    }
    $lines.Add("$($manifest.assembler.sha256) *$($manifest.assembler.name)")
    $lines.Add("$($manifest.assembler_source.sha256) *$($manifest.assembler_source.name)")
    $manifestHash = Get-LowerSha256 -Path $manifestPath
    $lines.Add("$manifestHash *$([System.IO.Path]::GetFileName($manifestPath))")
    $lines | Set-Content -LiteralPath $checksumPath -Encoding ascii

    Write-Host "NVIDIA split release ready: $resolvedOutput"
    Write-Host "Complete archive: $archiveName ($($archive.Length) bytes, SHA-256 $($manifest.archive_sha256))"
    Write-Host "Parts: $($partRecords.Count), maximum $PartSizeBytes bytes each"
    Write-Host "Double-click assembler: $setupPath"
    Write-Output $manifestPath
}
catch {
    $splitError = $_
    try {
        Remove-SplitAssets
    }
    catch {
        Write-Warning "Failed to remove one or more incomplete split assets: $($_.Exception.Message)"
    }
    throw $splitError
}
