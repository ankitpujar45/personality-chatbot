$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$runDir = Join-Path $projectRoot ".run"
$pidFile = Join-Path $runDir "processes.json"

function Test-UrlReady {
    param([string]$Url)

    try {
        Invoke-WebRequest -Uri $Url -UseBasicParsing | Out-Null
        return $true
    } catch {
        return $false
    }
}

function Test-PythonModule {
    param(
        [string]$PythonCommand,
        [string]$ModuleName
    )

    $proc = Start-Process -FilePath $PythonCommand `
        -ArgumentList "-c", "import $ModuleName" `
        -WindowStyle Hidden `
        -PassThru `
        -Wait

    return $proc.ExitCode -eq 0
}

function Wait-ForUrl {
    param(
        [string]$Url,
        [int]$TimeoutSeconds = 45
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            Invoke-WebRequest -Uri $Url -UseBasicParsing | Out-Null
            return $true
        } catch {
            Start-Sleep -Seconds 2
        }
    }

    return $false
}

function Start-ServiceWindow {
    param(
        [string]$Name,
        [string]$Command,
        [string]$WorkingDirectory
    )

    $windowCommand = @"
Set-Location '$WorkingDirectory'
`$host.UI.RawUI.WindowTitle = 'Personality Chatbot - $Name'
$Command
"@

    return Start-Process -FilePath "powershell.exe" `
        -ArgumentList "-NoProfile", "-ExecutionPolicy", "Bypass", "-NoExit", "-Command", $windowCommand `
        -WorkingDirectory $WorkingDirectory `
        -WindowStyle Minimized `
        -PassThru
}

if (-not (Test-Path $runDir)) {
    New-Item -ItemType Directory -Path $runDir | Out-Null
}

$pythonExe = Join-Path $projectRoot ".venv\Scripts\python.exe"
$frontendDir = Join-Path $projectRoot "frontend\persona-ui"
$frontendNodeModules = Join-Path $frontendDir "node_modules"
$ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
$systemPythonCommand = Get-Command python -ErrorAction SilentlyContinue
$backendPython = $null
$backendPythonLabel = $null

if ((Test-Path $pythonExe) -and (Test-PythonModule -PythonCommand $pythonExe -ModuleName "uvicorn")) {
    $backendPython = $pythonExe
    $backendPythonLabel = ".venv"
} elseif ($systemPythonCommand -and (Test-PythonModule -PythonCommand $systemPythonCommand.Source -ModuleName "uvicorn")) {
    $backendPython = $systemPythonCommand.Source
    $backendPythonLabel = "system"
} else {
    throw "Could not find a Python runtime with 'uvicorn' installed. Reinstall the project dependencies first."
}

if (-not (Test-Path $frontendNodeModules)) {
    throw "Missing frontend dependencies at '$frontendNodeModules'. Run 'npm install' in frontend\persona-ui first."
}

if (-not $ollamaCommand) {
    throw "Ollama is not installed or not on PATH. Install it from https://ollama.com/download."
}

$state = [ordered]@{
    project_root = $projectRoot
    started_at = (Get-Date).ToString("o")
    processes = [ordered]@{
        ollama = [ordered]@{
            managed = $false
            pid = $null
            status = "already running"
        }
        backend = [ordered]@{
            managed = $false
            pid = $null
            status = "already running"
        }
        frontend = [ordered]@{
            managed = $false
            pid = $null
            status = "already running"
        }
    }
}

if (-not (Test-UrlReady -Url "http://127.0.0.1:11434/api/tags")) {
    $ollamaProcess = Start-ServiceWindow `
        -Name "Ollama" `
        -WorkingDirectory $projectRoot `
        -Command "& '$($ollamaCommand.Source)' serve"

    $state.processes.ollama.managed = $true
    $state.processes.ollama.pid = $ollamaProcess.Id
    $state.processes.ollama.status = "started"
}

if (-not (Test-UrlReady -Url "http://127.0.0.1:8000/")) {
    $backendProcess = Start-ServiceWindow `
        -Name "Backend" `
        -WorkingDirectory $projectRoot `
        -Command "& '$backendPython' -m uvicorn main1:app --host 127.0.0.1 --port 8000"

    $state.processes.backend.managed = $true
    $state.processes.backend.pid = $backendProcess.Id
    $state.processes.backend.status = "started"
}

if (-not (Test-UrlReady -Url "http://localhost:3000")) {
    $frontendProcess = Start-ServiceWindow `
        -Name "Frontend" `
        -WorkingDirectory $frontendDir `
        -Command "`$env:BROWSER='none'; npm.cmd start"

    $state.processes.frontend.managed = $true
    $state.processes.frontend.pid = $frontendProcess.Id
    $state.processes.frontend.status = "started"
}

$state | ConvertTo-Json -Depth 5 | Set-Content -Path $pidFile

$backendReady = Wait-ForUrl -Url "http://127.0.0.1:8000/"
$frontendReady = Wait-ForUrl -Url "http://localhost:3000"

Write-Host ""
Write-Host "Personality Chatbot status"
Write-Host "--------------------------"
Write-Host "Backend Python: $backendPythonLabel"
Write-Host "Ollama:   $($state.processes.ollama.status)"
Write-Host "Backend:  $($state.processes.backend.status)"
Write-Host "Frontend: $($state.processes.frontend.status)"
Write-Host ""
Write-Host "Backend URL:  http://127.0.0.1:8000  (ready: $backendReady)"
Write-Host "Frontend URL: http://localhost:3000  (ready: $frontendReady)"
Write-Host ""
Write-Host "To stop managed services later, run:"
Write-Host ".\stop-project.ps1"
