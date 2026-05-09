$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$pidFile = Join-Path $projectRoot ".run\processes.json"

if (-not (Test-Path $pidFile)) {
    Write-Host "No managed process file found at $pidFile"
    Write-Host "If the app is still running, stop those terminals manually."
    exit 0
}

$state = Get-Content $pidFile | ConvertFrom-Json
$stopped = @()
$missing = @()

foreach ($name in "frontend", "backend", "ollama") {
    $procInfo = $state.processes.$name

    if (-not $procInfo.managed -or -not $procInfo.pid) {
        continue
    }

    $proc = Get-Process -Id $procInfo.pid -ErrorAction SilentlyContinue
    if ($proc) {
        Stop-Process -Id $procInfo.pid -Force
        $stopped += "${name} ($($procInfo.pid))"
    } else {
        $missing += "${name} ($($procInfo.pid))"
    }
}

Remove-Item $pidFile -Force

Write-Host ""
Write-Host "Stop summary"
Write-Host "------------"

if ($stopped.Count -gt 0) {
    Write-Host "Stopped: $($stopped -join ', ')"
} else {
    Write-Host "Stopped: none"
}

if ($missing.Count -gt 0) {
    Write-Host "Already gone: $($missing -join ', ')"
}
