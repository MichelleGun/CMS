param([switch]$UsePip)

$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskEnvironmentNames = @('UV_CACHE_DIR', 'UV_LINK_MODE', 'PIP_CACHE_DIR', 'TEMP', 'TMP')
$taskOriginalEnvironment = @{}
foreach ($taskName in $taskEnvironmentNames) {
    $taskOriginalEnvironment[$taskName] = [Environment]::GetEnvironmentVariable($taskName, 'Process')
}

function Invoke-Checked {
    param([string]$Executable, [string[]]$Arguments, [string]$FailureMessage)
    & $Executable @Arguments
    if ($LASTEXITCODE -ne 0) { throw $FailureMessage }
}

Push-Location -LiteralPath $taskProjectRoot
try {
    $taskPython = $null
    $taskVersionProbe = 'import sys; print(sys.executable) if sys.version_info[:2] == (3,11) else sys.exit(1)'
    if (Get-Command py -ErrorAction SilentlyContinue) {
        $taskCandidate = & py -3.11 -c $taskVersionProbe
        if ($LASTEXITCODE -eq 0) { $taskPython = ($taskCandidate | Select-Object -Last 1).Trim() }
    }
    if (-not $taskPython) {
        $taskPythonCommand = Get-Command python -ErrorAction SilentlyContinue
        if ($taskPythonCommand -and $taskPythonCommand.Source -notlike '*WindowsApps*') {
            $taskCandidate = & $taskPythonCommand.Source -c $taskVersionProbe
            if ($LASTEXITCODE -eq 0) { $taskPython = ($taskCandidate | Select-Object -Last 1).Trim() }
        }
    }
    if (-not $taskPython) {
        throw 'Install 64-bit Python 3.11 with the Python launcher, then rerun setup.cmd. See README.md.'
    }

    $taskBuildTemp = Join-Path $taskProjectRoot '.cache\tmp'
    New-Item -ItemType Directory -Path $taskBuildTemp -Force | Out-Null
    $env:TEMP = $taskBuildTemp
    $env:TMP = $taskBuildTemp
    $env:UV_LINK_MODE = 'copy'
    if (-not $env:UV_CACHE_DIR) { $env:UV_CACHE_DIR = Join-Path $taskProjectRoot '.cache\uv' }
    $env:PIP_CACHE_DIR = Join-Path $taskProjectRoot '.cache\pip'
    $taskVenvPython = Join-Path $taskProjectRoot '.venv\Scripts\python.exe'

    if ((Get-Command uv -ErrorAction SilentlyContinue) -and -not $UsePip) {
        Invoke-Checked -Executable 'uv' -Arguments @('sync', '--locked', '--python', $taskPython) -FailureMessage 'Dependency installation failed.'
    }
    else {
        if (-not (Test-Path -LiteralPath $taskVenvPython)) {
            Invoke-Checked -Executable $taskPython -Arguments @('-m', 'venv', '.venv') -FailureMessage 'Virtual environment creation failed.'
        }
        Invoke-Checked -Executable $taskVenvPython -Arguments @('-c', $taskVersionProbe) -FailureMessage 'Existing .venv uses the wrong Python version. Start in a fresh extracted folder.'
        Invoke-Checked -Executable $taskVenvPython -Arguments @('-m', 'ensurepip', '--upgrade') -FailureMessage 'Could not bootstrap pip in the virtual environment.'
        Invoke-Checked -Executable $taskVenvPython -Arguments @('-m', 'pip', 'install', '-r', 'requirements-dev.txt') -FailureMessage 'Package installation failed. Check internet access.'
        Invoke-Checked -Executable $taskVenvPython -Arguments @('-m', 'pip', 'install', '--no-deps', '-e', '.') -FailureMessage 'Local project installation failed.'
    }

    Invoke-Checked -Executable $taskVenvPython -Arguments @('-m', 'tello_lab', 'doctor') -FailureMessage 'Environment check failed.'
    Invoke-Checked -Executable $taskVenvPython -Arguments @('-m', 'pytest', '-q') -FailureMessage 'Offline tests failed.'
    Invoke-Checked -Executable $taskVenvPython -Arguments @('-m', 'ruff', 'check', 'src', 'tests') -FailureMessage 'Lint check failed.'
    Write-Output 'Setup complete. Run .\run.cmd check after connecting to the drone Wi-Fi.'
    Write-Output 'Setup sent no drone commands. Read README.md before any flight.'
}
finally {
    Pop-Location
    foreach ($taskName in $taskEnvironmentNames) {
        [Environment]::SetEnvironmentVariable($taskName, $taskOriginalEnvironment[$taskName], 'Process')
    }
}
