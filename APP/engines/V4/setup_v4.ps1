[CmdletBinding()]
param(
    [switch]$CheckOnly
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$appDir = [IO.Path]::GetFullPath((Join-Path $here "..\.."))
$shared = Join-Path $appDir "shared\tools"
$venvPython = Join-Path $here ".venv\Scripts\python.exe"
$requirements = Join-Path $here "requirements.lock"
$deepScript = Join-Path $here "deep_raster_v4.py"
$v3Python = Join-Path $appDir "engines\V3\.venv\Scripts\python.exe"
$hatModel = Join-Path $appDir "engines\V3\models\Real_HAT_GAN_sharper.pth"
$hatModelSha256 = "5800B67136006EB8CAB3B4ED7C8D73B6A195BB18E6CC709B674F9AA069C00271"

$gmicDir = Join-Path $shared "gmic\gmic-4.0.2-cli-win64"
$gmicExe = Join-Path $gmicDir "gmic.exe"
$resvgDir = Join-Path $shared "resvg\0.47.0"
$resvgExe = Join-Path $resvgDir "resvg.exe"
$scribusDir = Join-Path $shared "scribus\1.6.6"
$scribusExe = Join-Path $scribusDir "Scribus.exe"

$downloads = @(
    @{
        Name = "G'MIC 4.0.2"
        Uri = "https://gmic.eu/files/windows/gmic_4.0.2_cli_win64.zip"
        Sha256 = "6D0F553C383A93CED4B006B067A49277F45AD69A030287ADDA7ABD4D4DE15163"
    },
    @{
        Name = "resvg 0.47.0"
        Uri = "https://github.com/linebender/resvg/releases/download/v0.47.0/resvg-win64.zip"
        Sha256 = "5684E59CEAA53CE720B49EFB441B0918AE99D04E8CE3F6F753664524592D67F1"
    },
    @{
        Name = "Scribus 1.6.6"
        Uri = "https://downloads.sourceforge.net/project/scribus/scribus/1.6.6/scribus-1.6.6-windows-x64.exe"
        Sha256 = "4C7313DA22B8DAA025DAB0A2D57E82E8B1827C8A46A02DB1D4E78B6037B40264"
    }
)

function Get-MissingComponents {
    $missing = [System.Collections.Generic.List[string]]::new()
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) { $missing.Add("Python V4 environment") }
    if (-not (Test-Path -LiteralPath $gmicExe -PathType Leaf)) { $missing.Add("G'MIC 4.0.2") }
    if (-not (Test-Path -LiteralPath $resvgExe -PathType Leaf)) { $missing.Add("resvg 0.47.0") }
    if (-not (Test-Path -LiteralPath $scribusExe -PathType Leaf)) { $missing.Add("Scribus 1.6.6") }
    if (-not (Test-Path -LiteralPath $deepScript -PathType Leaf)) { $missing.Add("V4 Deep engine") }
    return $missing
}

function Get-MissingPrintComponents {
    $missing = [System.Collections.Generic.List[string]]::new()
    if (-not (Test-Path -LiteralPath $v3Python -PathType Leaf)) { $missing.Add("V3 CUDA Python runtime") }
    if (-not (Test-Path -LiteralPath $hatModel -PathType Leaf)) { $missing.Add("Real_HAT_GAN_sharper model") }
    return $missing
}

function Get-VerifiedDownload {
    param(
        [Parameter(Mandatory)] [hashtable]$Spec,
        [Parameter(Mandatory)] [string]$Destination
    )
    Write-Host "Downloading $($Spec.Name)..."
    $curl = Get-Command curl.exe -ErrorAction SilentlyContinue
    if ($curl) {
        & $curl.Source -L --fail --retry 3 --silent --show-error -o $Destination $Spec.Uri
        if ($LASTEXITCODE -ne 0) { throw "Download failed: $($Spec.Name)" }
    }
    else {
        Invoke-WebRequest -Uri $Spec.Uri -OutFile $Destination -UseBasicParsing
    }
    $actual = (Get-FileHash -LiteralPath $Destination -Algorithm SHA256).Hash
    if ($actual -ne $Spec.Sha256) {
        throw "SHA-256 mismatch for $($Spec.Name). Expected $($Spec.Sha256), got $actual."
    }
}

function Find-BasePython {
    $pythonRoot = Join-Path $appDir "engines\V3\.python"
    $portable = Get-ChildItem -LiteralPath $pythonRoot -Recurse -Filter python.exe -File -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($portable) { return $portable.FullName }

    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) { return $python.Source }

    throw "Python 3.11+ was not found. Install Python first, then rerun this setup."
}

$missing = Get-MissingComponents
if ($CheckOnly) {
    $printMissing = Get-MissingPrintComponents
    if ($missing.Count -gt 0 -or $printMissing.Count -gt 0) {
        Write-Host "V4 Deep Print is not ready:" -ForegroundColor Yellow
        @($missing) + @($printMissing) | ForEach-Object { Write-Host "  - $_" }
        exit 1
    }
    & $venvPython -c "import cv2, numpy, pikepdf, PIL, vtracer; print('Python packages: OK')"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    $actualHatSha256 = (Get-FileHash -LiteralPath $hatModel -Algorithm SHA256).Hash
    if ($actualHatSha256 -ne $hatModelSha256) {
        Write-Host "HAT model SHA-256 mismatch." -ForegroundColor Red
        exit 1
    }
    & $v3Python -c "import torch; assert torch.cuda.is_available(); print('CUDA:', torch.cuda.get_device_name(0))"
    if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    Write-Host "V4 Print toolchain: OK"
    Write-Host "  $gmicExe"
    Write-Host "  $resvgExe"
    Write-Host "  $scribusExe"
    Write-Host "  $hatModel"
    exit 0
}

$tempRoot = Join-Path ([IO.Path]::GetTempPath()) ("resize-v4-setup-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tempRoot | Out-Null
try {
    if (-not (Test-Path -LiteralPath $venvPython -PathType Leaf)) {
        $basePython = Find-BasePython
        Write-Host "Creating V4 Python environment with $basePython..."
        & $basePython -m venv (Join-Path $here ".venv")
        if ($LASTEXITCODE -ne 0) { throw "Could not create the V4 Python environment." }
    }
    & $venvPython -m pip install --disable-pip-version-check -r $requirements
    if ($LASTEXITCODE -ne 0) { throw "Could not install the pinned V4 Python packages." }

    if (-not (Test-Path -LiteralPath $gmicExe -PathType Leaf)) {
        $archive = Join-Path $tempRoot "gmic.zip"
        Get-VerifiedDownload -Spec $downloads[0] -Destination $archive
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $gmicDir) | Out-Null
        Expand-Archive -LiteralPath $archive -DestinationPath (Split-Path -Parent $gmicDir) -Force
    }

    if (-not (Test-Path -LiteralPath $resvgExe -PathType Leaf)) {
        $archive = Join-Path $tempRoot "resvg.zip"
        Get-VerifiedDownload -Spec $downloads[1] -Destination $archive
        New-Item -ItemType Directory -Force -Path $resvgDir | Out-Null
        Expand-Archive -LiteralPath $archive -DestinationPath $resvgDir -Force
    }

    if (-not (Test-Path -LiteralPath $scribusExe -PathType Leaf)) {
        $installer = Join-Path $tempRoot "scribus-installer.exe"
        Get-VerifiedDownload -Spec $downloads[2] -Destination $installer
        New-Item -ItemType Directory -Force -Path (Split-Path -Parent $scribusDir) | Out-Null
        $scribusDirArgument = '/DIR="{0}"' -f $scribusDir
        $startParameters = @{
            FilePath = $installer
            ArgumentList = @("/VERYSILENT", "/SUPPRESSMSGBOXES", "/NORESTART", "/SP-", $scribusDirArgument)
            Wait = $true
            PassThru = $true
            WindowStyle = "Hidden"
        }
        $process = Start-Process @startParameters
        if ($process.ExitCode -ne 0) { throw "Scribus installer exited with code $($process.ExitCode)." }
    }

    $stillMissing = Get-MissingComponents
    if ($stillMissing.Count -gt 0) {
        throw "Setup finished with missing components: $($stillMissing -join ', ')"
    }
    & $venvPython -c "import cv2, numpy, pikepdf, PIL, vtracer; print('Python packages: OK')"
    if ($LASTEXITCODE -ne 0) { throw "V4 Python import check failed." }
    Write-Host ""
    Write-Host "V4 setup completed successfully." -ForegroundColor Green
    Write-Host "Run from the RESIZE root: .\upscale print <image> 10 --width-mm <mm>"
    $printMissing = Get-MissingPrintComponents
    if ($printMissing.Count -gt 0) {
        Write-Warning "V4 Deep Print still needs: $($printMissing -join ', '). V4 VECTOR can run without V3/CUDA."
    }
}
finally {
    $resolvedTemp = [IO.Path]::GetFullPath($tempRoot)
    $systemTemp = [IO.Path]::GetFullPath([IO.Path]::GetTempPath())
    if ((Test-Path -LiteralPath $resolvedTemp -PathType Container) -and $resolvedTemp.StartsWith($systemTemp, [StringComparison]::OrdinalIgnoreCase)) {
        Remove-Item -LiteralPath $resolvedTemp -Recurse -Force
    }
}
