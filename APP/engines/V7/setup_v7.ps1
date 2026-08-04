[CmdletBinding()]
param(
    [switch]$CheckOnly,
    [switch]$Offline,
    [switch]$SkipLanguageModel,
    [switch]$SkipTesseract
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$appDir = [IO.Path]::GetFullPath((Join-Path $here "..\.."))
$python = Join-Path $here ".venv\Scripts\python.exe"
$requirements = Join-Path $here "requirements.lock"
$modelSetup = Join-Path $here "model_setup.py"
$modelRoot = Join-Path $here "models"
$paddleRoot = Join-Path $modelRoot "paddlex\official_models"
$tessdataDir = Join-Path $modelRoot "tessdata"
$vietnameseModel = Join-Path $tessdataDir "vie.traineddata"
$vietnameseRevision = "e2aad9b983032bb1beff9133104a67cdbb87ca4d"
$vietnameseUri = "https://raw.githubusercontent.com/tesseract-ocr/tessdata_best/$vietnameseRevision/vie.traineddata"
$vietnameseSha256 = "B6B49293D95D0B6DBD8780174627E82C75BE957B6F4ED9862155540D6B00BB45"
$tesseractPackageId = "UB-Mannheim.TesseractOCR"
$tesseractPackageVersion = "5.4.0.20240606"

$paddleModels = @(
    @{
        Name = "PP-OCRv6_medium_det"
        Uri = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_medium_det_infer.tar"
        Files = @{
            "inference.json" = "0F1A7EC35DA36173529C7A60238B7F7919E3831929C3F700AD90AD4896ADECD5"
            "inference.pdiparams" = "85218D2E3D98F5A21C58B4220627BE923A97AEE5DB3CC71F39536AB31AC53960"
            "inference.yml" = "7298D5EAD546584AF2504D03355F881AC7A7BC0EB1E282D3E159277C1D0AF871"
        }
    },
    @{
        Name = "PP-OCRv6_medium_rec"
        Uri = "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-OCRv6_medium_rec_infer.tar"
        Files = @{
            "inference.json" = "0B2E25E990BD072F1BF77D59D67D508BCE6C4BD44AF6624E0FB27D6DA2CD00E8"
            "inference.pdiparams" = "1B01C79A914587933F615569E75DE54F2E638EBB5D3F3B3C1B38C24EDE8C7319"
            "inference.yml" = "991B700FACF5B50A7DE193468207D5F4255B538DDE0D312AE3B7C7A9B6873129"
        }
    }
)

function Test-SupportedPython {
    param([Parameter(Mandatory)] [string]$Candidate)
    if (-not (Test-Path -LiteralPath $Candidate -PathType Leaf)) { return $false }
    & $Candidate -c "import struct,sys;raise SystemExit(0 if sys.version_info[:2]==(3,11) and struct.calcsize('P')==8 else 1)" *> $null
    return $LASTEXITCODE -eq 0
}

function Find-BasePython {
    $portableRoot = Join-Path $appDir "engines\V3\.python"
    $candidates = [System.Collections.Generic.List[string]]::new()
    Get-ChildItem -LiteralPath $portableRoot -Recurse -Filter python.exe -File -ErrorAction SilentlyContinue |
        Sort-Object { $_.FullName.Length } |
        ForEach-Object { $candidates.Add($_.FullName) }
    $system = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($system) { $candidates.Add($system.Source) }
    foreach ($candidate in ($candidates | Select-Object -Unique)) {
        if (Test-SupportedPython -Candidate $candidate) { return $candidate }
    }
    throw "V7 requires 64-bit CPython 3.11. Install V3 first or install CPython 3.11."
}

function Get-VerifiedDownload {
    param(
        [Parameter(Mandatory)] [string]$Uri,
        [Parameter(Mandatory)] [string]$Destination,
        [string]$Sha256 = ""
    )
    if ($Offline) {
        throw "Offline mode forbids download: $Uri"
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Destination) | Out-Null
    $part = "$Destination.part"
    try {
        $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
        if ($curl) {
            & $curl.Source -L --fail --retry 3 --silent --show-error -o $part $Uri
            if ($LASTEXITCODE -ne 0) { throw "Download failed: $Uri" }
        }
        else {
            Invoke-WebRequest -Uri $Uri -OutFile $part -UseBasicParsing
        }
        if ($Sha256) {
            $actual = (Get-FileHash -LiteralPath $part -Algorithm SHA256).Hash
            if ($actual -ne $Sha256) {
                throw "SHA-256 mismatch for $Uri. Expected $Sha256, got $actual."
            }
        }
        Move-Item -LiteralPath $part -Destination $Destination -Force
    }
    finally {
        if (Test-Path -LiteralPath $part -PathType Leaf) {
            Remove-Item -LiteralPath $part -Force
        }
    }
}

function Test-PaddleModel {
    param(
        [Parameter(Mandatory)] [string]$Directory,
        [Parameter(Mandatory)] [hashtable]$Files
    )
    foreach ($entry in $Files.GetEnumerator()) {
        $path = Join-Path $Directory $entry.Key
        if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $false }
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $entry.Value) { return $false }
    }
    return $true
}

function Install-PaddleModel {
    param([Parameter(Mandatory)] [hashtable]$Spec)
    $target = Join-Path $paddleRoot $Spec.Name
    if (Test-PaddleModel -Directory $target -Files $Spec.Files) {
        Write-Host "$($Spec.Name): verified"
        return
    }
    if ($Offline) {
        throw "Offline mode requires an existing verified model: $target"
    }
    if (Test-Path -LiteralPath $target) {
        throw "Existing model failed SHA-256 verification: $target. Move it aside and rerun setup."
    }
    $downloads = Join-Path $modelRoot "downloads"
    $archive = Join-Path $downloads "$($Spec.Name).tar"
    Write-Host "Downloading $($Spec.Name) from the official Paddle archive..."
    Get-VerifiedDownload -Uri $Spec.Uri -Destination $archive
    $tempRoot = Join-Path $modelRoot (".extract-" + $Spec.Name + "-" + [Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $tempRoot | Out-Null
    try {
        $tar = Get-Command tar.exe -ErrorAction SilentlyContinue
        if (-not $tar) { throw "Windows tar.exe is required to unpack the official Paddle model." }
        & $tar.Source -xf $archive -C $tempRoot
        if ($LASTEXITCODE -ne 0) { throw "Could not unpack $archive" }
        $candidate = Get-ChildItem -LiteralPath $tempRoot -Directory -Recurse |
            Where-Object { Test-PaddleModel -Directory $_.FullName -Files $Spec.Files } |
            Select-Object -First 1
        if (-not $candidate) { throw "Extracted $($Spec.Name) failed the pinned file-hash check." }
        New-Item -ItemType Directory -Force -Path $paddleRoot | Out-Null
        Move-Item -LiteralPath $candidate.FullName -Destination $target
        if (-not (Test-PaddleModel -Directory $target -Files $Spec.Files)) {
            throw "Published $($Spec.Name) failed verification."
        }
    }
    finally {
        if (Test-Path -LiteralPath $tempRoot -PathType Container) {
            $resolvedTemp = [IO.Path]::GetFullPath($tempRoot)
            $resolvedRoot = [IO.Path]::GetFullPath($modelRoot).TrimEnd('\') + '\'
            if (-not $resolvedTemp.StartsWith($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)) {
                throw "Refusing unsafe cleanup path: $resolvedTemp"
            }
            Remove-Item -LiteralPath $resolvedTemp -Recurse -Force
        }
        if (Test-Path -LiteralPath $archive -PathType Leaf) {
            Remove-Item -LiteralPath $archive -Force
        }
    }
}

function Find-Tesseract {
    $found = Get-Command tesseract.exe -ErrorAction SilentlyContinue
    if ($found) { return $found.Source }
    foreach ($candidate in @(
        (Join-Path $env:ProgramFiles "Tesseract-OCR\tesseract.exe"),
        (Join-Path ${env:ProgramFiles(x86)} "Tesseract-OCR\tesseract.exe")
    )) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) { return $candidate }
    }
    return $null
}

function Test-Tesseract {
    if ($SkipTesseract) {
        Write-Warning "Tesseract was deliberately skipped. V7 will never auto-green OCR regions."
        return
    }
    $executable = Find-Tesseract
    if (-not $executable) {
        throw "Tesseract is missing. Run setup without -CheckOnly/-Offline, or use -SkipTesseract deliberately."
    }
    if (-not (Test-Path -LiteralPath $vietnameseModel -PathType Leaf)) {
        throw "Verified Vietnamese tessdata is missing: $vietnameseModel"
    }
    $actual = (Get-FileHash -LiteralPath $vietnameseModel -Algorithm SHA256).Hash
    if ($actual -ne $vietnameseSha256) {
        throw "Vietnamese tessdata SHA-256 mismatch. Expected $vietnameseSha256, got $actual."
    }
    $version = @(& $executable --version 2>&1)
    if ($LASTEXITCODE -ne 0 -or -not $version -or $version[0] -notmatch '^tesseract v5\.') {
        throw "V7 requires a working Tesseract 5.x executable."
    }
    $languages = & $executable --tessdata-dir $tessdataDir --list-langs 2>&1
    if ($LASTEXITCODE -ne 0 -or ($languages -join "`n") -notmatch "(?m)^vie\r?$") {
        throw "Tesseract cannot load the verified Vietnamese model."
    }
    Write-Host "Tesseract Vietnamese vote: verified ($($version[0]))"
}

function Ensure-Tesseract {
    if ($SkipTesseract) {
        Write-Warning "Tesseract was deliberately skipped. V7 will never auto-green OCR regions."
        return
    }
    $executable = Find-Tesseract
    if (-not $executable) {
        if ($Offline) {
            throw "Offline mode cannot install missing Tesseract."
        }
        $winget = Get-Command winget.exe -ErrorAction SilentlyContinue
        if (-not $winget) { throw "Tesseract is missing and WinGet is unavailable. Install Tesseract 5 or rerun with -SkipTesseract." }
        & $winget.Source install --id $tesseractPackageId --exact --version $tesseractPackageVersion `
            --source winget --silent --accept-package-agreements --accept-source-agreements
        if ($LASTEXITCODE -ne 0) { throw "WinGet could not install Tesseract." }
        $executable = Find-Tesseract
    }
    if (-not $executable) {
        throw "Tesseract installation completed but tesseract.exe was not found."
    }
    if (-not (Test-Path -LiteralPath $vietnameseModel -PathType Leaf) -or
        (Get-FileHash -LiteralPath $vietnameseModel -Algorithm SHA256).Hash -ne $vietnameseSha256) {
        if ($Offline) {
            throw "Offline mode cannot download missing or invalid Vietnamese tessdata."
        }
        Write-Host "Downloading and verifying official Vietnamese tessdata_best..."
        Get-VerifiedDownload -Uri $vietnameseUri -Destination $vietnameseModel -Sha256 $vietnameseSha256
    }
    Test-Tesseract
}

function Test-Runtime {
    if (-not (Test-SupportedPython -Candidate $python)) { throw "V7 Python runtime is missing or unsupported." }
    & $python -B $modelSetup --check-runtime-only
    if ($LASTEXITCODE -ne 0) { throw "V7 exact package-lock check failed." }
    & $python -B -m pip --disable-pip-version-check check
    if ($LASTEXITCODE -ne 0) { throw "V7 dependency graph is inconsistent." }
}

function Test-RuntimeImports {
    & $python -c "import paddle,paddleocr,paddlex,cv2,numpy,PIL,fontTools,freetype,uharfbuzz; assert paddle.__version__=='3.3.1'; assert paddleocr.__version__=='3.7.0'; assert paddlex.__version__=='3.7.2'; print('V7 packages: verified')"
    if ($LASTEXITCODE -ne 0) { throw "V7 runtime import/version check failed." }
}

if ($CheckOnly -or $Offline) {
    # Verification mode must be side-effect free: no package import that may
    # initialize a user cache, no WinGet, no model hub request, and no download.
    $env:HF_HUB_OFFLINE = "1"
    $env:TRANSFORMERS_OFFLINE = "1"
    $env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "True"
    $env:PIP_DISABLE_PIP_VERSION_CHECK = "1"
    $env:PYTHONDONTWRITEBYTECODE = "1"
    Test-Runtime
    $checkArguments = @($modelSetup, "--check-only")
    if ($SkipLanguageModel) { $checkArguments += "--skip-language-model" }
    & $python -B @checkArguments
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Test-Tesseract
    $checkLabel = if ($CheckOnly) { "check-only" } else { "offline verification" }
    Write-Host "V7 $checkLabel`: PASS" -ForegroundColor Green
    exit 0
}

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    $basePython = Find-BasePython
    Write-Host "Creating isolated V7 OCR environment with $basePython..."
    & $basePython -m venv (Join-Path $here ".venv")
    if ($LASTEXITCODE -ne 0) { throw "Could not create the V7 environment." }
}

& $python -m pip install --disable-pip-version-check --only-binary=:all: --no-deps `
    "pip==26.2" "setuptools==78.1.0" "wheel==0.47.0"
if ($LASTEXITCODE -ne 0) { throw "Could not prepare the pinned installer." }
& $python -m pip install --disable-pip-version-check --only-binary=:all: --no-deps -r $requirements
if ($LASTEXITCODE -ne 0) { throw "Could not install the exact V7 package lock." }
Test-Runtime
Test-RuntimeImports

foreach ($spec in $paddleModels) { Install-PaddleModel -Spec $spec }
Ensure-Tesseract

$modelArguments = @($modelSetup)
if ($SkipLanguageModel) { $modelArguments += "--skip-language-model" }
& $python @modelArguments
if ($LASTEXITCODE -ne 0) { throw "V7 model setup failed." }

Write-Host ""
Write-Host "V7 setup completed successfully." -ForegroundColor Green
Write-Host "Run from the RESIZE root: .\upscale repair <image-or-folder> 1"
