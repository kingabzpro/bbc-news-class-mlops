@echo off
setlocal enabledelayedexpansion

echo ===== MLOps Project Environment Setup (Windows) =====

REM Check if PowerShell is available
where powershell >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Error: PowerShell is required but not found.
    exit /b 1
)

REM Check Python version
powershell -Command "& {
    $pythonInstalled = $false
    try {
        $pythonVersion = python --version 2>&1
        if ($pythonVersion -match '(\d+)\.(\d+)\.(\d+)') {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]
            if ($major -ge 3 -and $minor -ge 9) {
                Write-Host 'Python 3.9+ is installed:' $pythonVersion
                $pythonInstalled = $true
            } else {
                Write-Host 'Error: Python 3.9+ is required but found:' $pythonVersion
                exit 1
            }
        }
    } catch {
        Write-Host 'Error: Python not found'
        exit 1
    }

    if (-not $pythonInstalled) {
        Write-Host 'Error: Python 3.9+ is required'
        exit 1
    }
}"

if %ERRORLEVEL% neq 0 (
    echo Please install Python 3.9 or higher and try again.
    exit /b 1
)

REM Check Git
powershell -Command "& {
    try {
        $gitVersion = git --version
        Write-Host 'Git is installed:' $gitVersion
    } catch {
        Write-Host 'Error: Git is required but not found'
        exit 1
    }
}"

if %ERRORLEVEL% neq 0 (
    echo Please install Git and try again.
    exit /b 1
)

REM Check Git LFS
powershell -Command "& {
    try {
        $gitLfsVersion = git lfs --version
        Write-Host 'Git LFS is installed:' $gitLfsVersion
    } catch {
        Write-Host 'Error: Git LFS is required but not found'
        exit 1
    }
}"

if %ERRORLEVEL% neq 0 (
    echo Please install Git LFS and try again.
    exit /b 1
)

REM Check for UV
powershell -Command "& {
    $uvInstalled = $false
    try {
        $uvVersion = uv --version
        Write-Host 'UV is installed:' $uvVersion
        $uvInstalled = $true
    } catch {
        Write-Host 'UV not found, will install...'
    }

    if (-not $uvInstalled) {
        Write-Host 'Installing UV...'
        try {
            # Use pip to install UV
            python -m pip install uv
            $uvVersion = uv --version
            Write-Host 'UV installed successfully:' $uvVersion
        } catch {
            Write-Host 'Error installing UV'
            exit 1
        }
    }
}"

if %ERRORLEVEL% neq 0 (
    echo Failed to install UV. Please install it manually.
    exit /b 1
)

REM Initialize Git LFS
echo Initializing Git LFS...
git lfs install

REM Create virtual environment
echo Creating virtual environment...
powershell -Command "& {
    try {
        if (Test-Path '.venv') {
            Write-Host 'Virtual environment already exists'
        } else {
            uv venv .venv
            Write-Host 'Virtual environment created successfully'
        }
    } catch {
        Write-Host 'Error creating virtual environment'
        exit 1
    }
}"

if %ERRORLEVEL% neq 0 (
    echo Failed to create virtual environment.
    exit /b 1
)

REM Install dependencies
echo Installing dependencies...
powershell -Command "& {
    try {
        # Activate virtual environment and install dependencies
        .\.venv\Scripts\Activate.ps1
        uv pip install -r requirements-dev.txt
        Write-Host 'Dependencies installed successfully'
    } catch {
        Write-Host 'Error installing dependencies:' $_
        exit 1
    }
}"

if %ERRORLEVEL% neq 0 (
    echo Failed to install dependencies.
    exit /b 1
)

echo ===== Environment Setup Complete =====
echo To activate the environment, run: .\.venv\Scripts\Activate.ps1
echo To deactivate, run: deactivate

endlocal

