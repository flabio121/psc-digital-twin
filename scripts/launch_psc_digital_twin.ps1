param(
    [int]$Port = 8501,
    [switch]$InstallDesktopShortcut
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$AppPath = Join-Path $RepoRoot "app.py"
$PythonPath = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$ShortcutName = "PSC Degradation App.lnk"

function Install-DesktopShortcut {
    $desktopPath = [Environment]::GetFolderPath("DesktopDirectory")
    $powerShellPath = Join-Path $env:SystemRoot "System32\WindowsPowerShell\v1.0\powershell.exe"

    if ([string]::IsNullOrWhiteSpace($desktopPath)) {
        throw "Windows did not return a Desktop folder path."
    }
    if (-not (Test-Path -LiteralPath $powerShellPath)) {
        throw "Could not find Windows PowerShell at: $powerShellPath"
    }

    $shortcutPath = Join-Path $desktopPath $ShortcutName
    $shell = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($shortcutPath)
    $shortcut.TargetPath = $powerShellPath
    $shortcut.Arguments = "-NoProfile -ExecutionPolicy Bypass -File `"$PSCommandPath`""
    $shortcut.WorkingDirectory = $RepoRoot
    $shortcut.Description = "Launch the PSC Degradation Digital Twin"
    $shortcut.IconLocation = "$powerShellPath,0"
    $shortcut.Save()

    Write-Host "Desktop shortcut installed: $shortcutPath" -ForegroundColor Green
}

if (-not (Test-Path -LiteralPath $AppPath)) {
    throw "Could not find the PSC Digital Twin app at: $AppPath"
}

if ($InstallDesktopShortcut) {
    Install-DesktopShortcut
    exit 0
}

if (-not (Test-Path -LiteralPath $PythonPath)) {
    Write-Host "The app's Python environment is missing." -ForegroundColor Red
    Write-Host "From the app folder, run:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv" -ForegroundColor Yellow
    Write-Host "  .\.venv\Scripts\python.exe -m pip install -r requirements.txt" -ForegroundColor Yellow
    Read-Host "Press Enter to close"
    exit 1
}

& $PythonPath -c "import streamlit" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Streamlit is not installed in the app's Python environment." -ForegroundColor Red
    Write-Host "Run: .\.venv\Scripts\python.exe -m pip install -r requirements.txt" -ForegroundColor Yellow
    Read-Host "Press Enter to close"
    exit 1
}

$existingListener = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($existingListener) {
    Write-Host "Port $Port is already in use; opening the existing local page." -ForegroundColor Yellow
    Start-Process "http://localhost:$Port/"
    exit 0
}

Set-Location $RepoRoot
Write-Host "Starting PSC Degradation Digital Twin on http://localhost:$Port/"
Write-Host "Leave this window open while using the app. Press Ctrl+C to stop it."
Start-Process "http://localhost:$Port/"
& $PythonPath -m streamlit run $AppPath --server.port $Port --server.headless true
