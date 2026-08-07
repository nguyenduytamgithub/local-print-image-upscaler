[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$CpuOnly,
    [switch]$SkipTesseract
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$appDir = [IO.Path]::GetFullPath((Join-Path $here "..\.."))
$sharedV3Python = Join-Path $appDir "engines\V3\.venv\Scripts\python.exe"
$localV5Python = Join-Path $here ".venv\Scripts\python.exe"
$requirements = Join-Path $here "requirements.lock"
$modelSetup = Join-Path $here "model_setup.py"
$lamaDir = Join-Path $env:USERPROFILE ".cache\torch\hub\checkpoints"
$lamaModel = Join-Path $lamaDir "big-lama.pt"
$lamaUri = "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt"
$lamaSha256 = "7BA7AA7AC37A4D41FDBBEBA3A2AF7EAD18058552997E3A3CD1A3B2210C9E6B4C"
$tessdataRevision = "e2aad9b983032bb1beff9133104a67cdbb87ca4d"
$tessdataDir = Join-Path $here "models\tessdata"
$vietnameseModel = Join-Path $tessdataDir "vie.traineddata"
$vietnameseUri = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/$tessdataRevision/vie.traineddata"
$vietnameseSha256 = "B6B49293D95D0B6DBD8780174627E82C75BE957B6F4ED9862155540D6B00BB45"
$tesseractPackageId = "UB-Mannheim.TesseractOCR"
$tesseractPackageVersion = "5.4.0.20240606"

$runtimeCheck = @'
from __future__ import annotations

import importlib.metadata as metadata
import sys
from pathlib import Path


def normalized(name: str) -> str:
    return name.lower().replace("_", "-")


expected: dict[str, tuple[str, str]] = {}
for raw in Path(sys.argv[1]).read_text(encoding="utf-8").splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    if "==" not in line:
        raise SystemExit(f"Unpinned requirement in runtime lock: {line}")
    name, version = line.split("==", 1)
    expected[normalized(name)] = (name, version)

problems: list[str] = []
for _, (name, version) in sorted(expected.items()):
    try:
        actual = metadata.version(name)
    except metadata.PackageNotFoundError:
        problems.append(f"missing {name}=={version}")
        continue
    if actual != version:
        problems.append(f"{name}: expected {version}, got {actual}")

try:
    import cv2
    import numpy
    import PIL
    import psd_tools
    import torch
    import torchvision
    import transformers
except Exception as exc:
    problems.append(f"runtime import failed: {exc}")
else:
    if torch.__version__.split("+", 1)[0] != "2.12.1":
        problems.append(f"torch: expected 2.12.1, got {torch.__version__}")
    if torchvision.__version__.split("+", 1)[0] != "0.27.1":
        problems.append(f"torchvision: expected 0.27.1, got {torchvision.__version__}")

if problems:
    raise SystemExit("Runtime check failed:\n  - " + "\n  - ".join(problems))

device = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU fallback"
flavor = f"CUDA {torch.version.cuda}" if torch.version.cuda else "CPU wheel"
print(f"Python packages: OK ({flavor}; active device: {device})")
'@

$torchFlavorCheck = @'
import sys

try:
    import torch
    import torchvision
except Exception:
    raise SystemExit(1)

wanted = sys.argv[1]
base_ok = (
    torch.__version__.split("+", 1)[0] == "2.12.1"
    and torchvision.__version__.split("+", 1)[0] == "0.27.1"
)
flavor_ok = (torch.version.cuda is None) if wanted == "cpu" else (torch.version.cuda == "12.6")
raise SystemExit(0 if base_ok and flavor_ok else 1)
'@

# Passing source containing quotes directly through Windows' native command-line
# parser is lossy. Base64 keeps the embedded validation scripts byte-for-byte.
$runtimeCheckBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($runtimeCheck))
$runtimeCheckRunner = "import base64;exec(base64.b64decode('$runtimeCheckBase64'))"
$torchFlavorCheckBase64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($torchFlavorCheck))
$torchFlavorCheckRunner = "import base64;exec(base64.b64decode('$torchFlavorCheckBase64'))"

function Test-SupportedPython {
    param([Parameter(Mandatory)] [string]$Candidate)
    if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { return $false }
    & $Candidate -c "import struct,sys; raise SystemExit(0 if (3,11) <= sys.version_info[:2] < (3,14) and struct.calcsize('P') == 8 else 1)" *> $null
    return $LASTEXITCODE -eq 0
}

function Assert-SupportedPython {
    param([Parameter(Mandatory)] [string]$Candidate)
    if (-not (Test-SupportedPython -Candidate $Candidate)) {
        throw "V5 requires 64-bit CPython 3.11, 3.12 or 3.13: $Candidate"
    }
}

function Find-BasePython {
    $candidates = [System.Collections.Generic.List[string]]::new()
    $portableRoot = Join-Path $appDir "engines\V3\.python"
    Get-ChildItem -LiteralPath $portableRoot -Recurse -Filter python.exe -File -ErrorAction SilentlyContinue |
        Sort-Object { $_.FullName.Length } |
        ForEach-Object { $candidates.Add($_.FullName) }
    $systemPython = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($systemPython) { $candidates.Add($systemPython.Source) }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-SupportedPython -Candidate $candidate) { return $candidate }
    }
    throw "64-bit CPython 3.11-3.13 was not found. Install a supported Python, then rerun setup_v5.ps1."
}

function Test-NvidiaRuntime {
    $nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
    if (-not $nvidiaSmi) {
        $systemNvidiaSmi = Join-Path $env:WINDIR "System32\nvidia-smi.exe"
        if (Test-Path -LiteralPath $systemNvidiaSmi -PathType Leaf) {
            $nvidiaSmi = $systemNvidiaSmi
        }
    }
    if (-not $nvidiaSmi) { return $false }
    $nvidiaSmiPath = if ($nvidiaSmi -is [string]) { $nvidiaSmi } else { $nvidiaSmi.Source }
    & $nvidiaSmiPath --query-gpu=name --format=csv,noheader *> $null
    return $LASTEXITCODE -eq 0
}

function Find-Tesseract {
    $discovered = Get-Command tesseract.exe -ErrorAction SilentlyContinue
    if ($discovered) { return $discovered.Source }
    $candidates = @(
        (Join-Path $env:ProgramFiles "Tesseract-OCR\tesseract.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Tesseract-OCR\tesseract.exe")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return [IO.Path]::GetFullPath($candidate)
        }
    }
    return $null
}

function Install-Tesseract {
    $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
    if (-not $winget) {
        throw (
            "Tesseract OCR is missing and WinGet is unavailable. Install Tesseract 5 for Windows " +
            "from https://tesseract-ocr.github.io/tessdoc/Installation.html, or rerun with -SkipTesseract."
        )
    }
    Write-Host "Installing the revision-pinned Tesseract OCR engine with WinGet..."
    & $winget.Source install --id $tesseractPackageId --exact --version $tesseractPackageVersion `
        --source winget --silent --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        throw "WinGet could not install $tesseractPackageId $tesseractPackageVersion."
    }
    $installed = Find-Tesseract
    if (-not $installed) {
        throw "Tesseract installation finished, but tesseract.exe was not found. Restart Windows and rerun -CheckOnly."
    }
    return $installed
}

function Test-TesseractRuntime {
    param([Parameter(Mandatory)] [string]$Executable)
    $versionOutput = & $Executable --version 2>&1
    if ($LASTEXITCODE -ne 0 -or -not $versionOutput) {
        throw "Tesseract could not start: $Executable"
    }
    $languages = & $Executable --tessdata-dir $tessdataDir --list-langs 2>&1
    if ($LASTEXITCODE -ne 0 -or ($languages -join "`n") -notmatch "(?m)^vie\r?$") {
        throw "Tesseract could not load the verified Vietnamese model: $vietnameseModel"
    }
    Write-Host "Tesseract OCR: OK ($($versionOutput[0]))"
}

function Get-VerifiedDownload {
    param(
        [Parameter(Mandatory)] [string]$Uri,
        [Parameter(Mandatory)] [string]$Destination,
        [Parameter(Mandatory)] [string]$Sha256
    )
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    $temporary = "$Destination.part"
    try {
        $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
        if ($curl) {
            & $curl.Source -L --fail --retry 3 --silent --show-error -o $temporary $Uri
            if ($LASTEXITCODE -ne 0) { throw "Download failed: $Uri" }
        }
        else {
            Invoke-WebRequest -Uri $Uri -OutFile $temporary -UseBasicParsing
        }
        $actual = (Get-FileHash -LiteralPath $temporary -Algorithm SHA256).Hash
        if ($actual -ne $Sha256) {
            throw "SHA-256 mismatch for $Uri. Expected $Sha256, got $actual."
        }
        Move-Item -LiteralPath $temporary -Destination $Destination -Force
    }
    finally {
        if (Test-Path -LiteralPath $temporary -PathType Leaf) {
            Remove-Item -LiteralPath $temporary -Force
        }
    }
}

$hasNvidia = Test-NvidiaRuntime
$useCpu = [bool]$CpuOnly -or -not $hasNvidia
if (-not $CpuOnly -and -not $hasNvidia) {
    Write-Host "No working NVIDIA driver was detected; selecting the official CPU runtime." -ForegroundColor Yellow
}

if ($CheckOnly) {
    $python = if (Test-Path -LiteralPath $localV5Python -PathType Leaf) {
        $localV5Python
    }
    else {
        $sharedV3Python
    }
    if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
        Write-Host "V5 runtime is missing. Run setup_v5.ps1 without -CheckOnly." -ForegroundColor Red
        exit 1
    }
    Assert-SupportedPython -Candidate $python
    Write-Host "Checking the same V5 runtime selected by upscale.cmd: $python"
    & $python -c $runtimeCheckRunner $requirements
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python -m pip check
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    & $python $modelSetup --check-only --lama $lamaModel
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    if (-not (Test-Path -LiteralPath $vietnameseModel -PathType Leaf)) {
        Write-Host "Vietnamese OCR model is missing. Run setup_v5.ps1 without -CheckOnly." -ForegroundColor Red
        exit 1
    }
    $actualVietnameseHash = (Get-FileHash -LiteralPath $vietnameseModel -Algorithm SHA256).Hash
    if ($actualVietnameseHash -ne $vietnameseSha256) {
        Write-Host "Vietnamese OCR model SHA-256 mismatch." -ForegroundColor Red
        exit 1
    }
    Write-Host "Vietnamese OCR model: OK"
    $tesseract = Find-Tesseract
    if ($tesseract) {
        Test-TesseractRuntime -Executable $tesseract
    }
    elseif ($SkipTesseract) {
        Write-Warning "Tesseract was skipped; OCR hints will be unavailable."
    }
    else {
        Write-Host "Tesseract OCR is missing. Run setup_v5.ps1, or use -SkipTesseract deliberately." -ForegroundColor Red
        exit 1
    }
    Write-Host "V5 installation check: PASS" -ForegroundColor Green
    exit 0
}

# A CPU-only V5 runtime is kept isolated so setup never replaces CUDA PyTorch in
# the shared V3 environment. The launcher also prefers this local V5 runtime.
if (Test-Path -LiteralPath $localV5Python -PathType Leaf) {
    $python = $localV5Python
}
elseif (-not $useCpu -and (Test-SupportedPython -Candidate $sharedV3Python)) {
    $python = $sharedV3Python
    Write-Host "Reusing the compatible V3 CUDA environment: $python"
}
else {
    if (Test-Path -LiteralPath $localV5Python -PathType Leaf) {
        Assert-SupportedPython -Candidate $localV5Python
    }
    else {
        $basePython = Find-BasePython
        Write-Host "Creating an isolated V5 environment with $basePython..."
        & $basePython -m venv (Join-Path $here ".venv")
        if ($LASTEXITCODE -ne 0) { throw "Could not create the V5 Python environment." }
    }
    $python = $localV5Python
}
Assert-SupportedPython -Candidate $python

Write-Host "Installing the exact common Python lock..."
& $python -m pip install --disable-pip-version-check --only-binary=:all: --no-deps `
    "pip==26.2" "setuptools==78.1.0" "wheel==0.47.0"
if ($LASTEXITCODE -ne 0) { throw "Could not prepare the pinned Python installer." }
& $python -m pip install --disable-pip-version-check --only-binary=:all: --no-deps -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Could not install the pinned V5 packages." }

$wantedTorchFlavor = if ($useCpu) { "cpu" } else { "cu126" }
& $python -c $torchFlavorCheckRunner $wantedTorchFlavor *> $null
if ($LASTEXITCODE -ne 0) {
    $torchIndex = if ($useCpu) {
        "https://download.pytorch.org/whl/cpu"
    }
    else {
        "https://download.pytorch.org/whl/cu126"
    }
    Write-Host "Installing official PyTorch 2.12.1/$wantedTorchFlavor..."
    & $python -m pip install --disable-pip-version-check --only-binary=:all: --no-deps `
        --force-reinstall --index-url $torchIndex "torch==2.12.1" "torchvision==0.27.1"
    if ($LASTEXITCODE -ne 0) { throw "Could not install PyTorch from the official index." }
}
& $python -c $torchFlavorCheckRunner $wantedTorchFlavor *> $null
if ($LASTEXITCODE -ne 0) {
    throw "PyTorch installed, but its version/flavor does not match 2.12.1/$wantedTorchFlavor."
}

& $python -c $runtimeCheckRunner $requirements
if ($LASTEXITCODE -ne 0) { throw "V5 Python runtime validation failed." }
& $python -m pip check
if ($LASTEXITCODE -ne 0) { throw "V5 Python dependency graph is inconsistent." }
if (-not $useCpu) {
    & $python -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" *> $null
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "The CUDA wheel is installed, but the NVIDIA driver is not usable by PyTorch. V5 will run on CPU until the driver is updated."
    }
}

if (-not (Test-Path -LiteralPath $lamaModel -PathType Leaf) -or
    (Get-FileHash -LiteralPath $lamaModel -Algorithm SHA256).Hash -ne $lamaSha256) {
    Write-Host "Downloading and verifying the LaMa checkpoint..."
    Get-VerifiedDownload -Uri $lamaUri -Destination $lamaModel -Sha256 $lamaSha256
}

if (-not (Test-Path -LiteralPath $vietnameseModel -PathType Leaf) -or
    (Get-FileHash -LiteralPath $vietnameseModel -Algorithm SHA256).Hash -ne $vietnameseSha256) {
    Write-Host "Downloading and verifying the official Vietnamese Tesseract model..."
    Get-VerifiedDownload -Uri $vietnameseUri -Destination $vietnameseModel -Sha256 $vietnameseSha256
}

$tesseract = Find-Tesseract
if (-not $tesseract -and -not $SkipTesseract) {
    $tesseract = Install-Tesseract
}
if ($tesseract) {
    Test-TesseractRuntime -Executable $tesseract
}
else {
    Write-Warning "Tesseract was skipped; V5 works, but OCR naming/grouping hints are unavailable."
}

Write-Host "Downloading exact, revision-pinned SAM 2.1, Grounding DINO and ViTMatte snapshots..."
& $python $modelSetup --lama $lamaModel
if ($LASTEXITCODE -ne 0) { throw "Could not prepare the V5 models." }

Write-Host ""
Write-Host "V5 setup completed successfully." -ForegroundColor Green
Write-Host "Runtime: $python"
Write-Host "Run from the RESIZE root: .\upscale layers <image-or-folder> 1"
Write-Host "Use n=4 only when the V3 model set is also installed."
