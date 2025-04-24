@echo off
setlocal EnableDelayedExpansion

REM Repository URL (replace with your actual repo URL)
set REPO_URL=<repository-url>
for %%f in (%REPO_URL%) do set REPO_NAME=%%~nxf
set REPO_NAME=%REPO_NAME:.git=%

REM Step 1: Clone the repository
echo Cloning the repository...
if exist "%REPO_NAME%" (
    echo Directory %REPO_NAME% already exists. Pulling latest changes...
    cd "%REPO_NAME%"
    git pull
    cd ..
) else (
    git clone %REPO_URL%
)
cd "%REPO_NAME%"

REM Step 2: Install uv if not already installed
where uv >nul 2>&1
if %ERRORLEVEL% NEQ 0 (
    echo Installing uv...
    curl -LsSf https://astral.sh/uv/install.sh | sh
)

REM Step 3: Create and activate virtual environment
echo Creating virtual environment...
uv venv

echo Activating virtual environment...
call .venv\Scripts\activate

REM Step 4: Install dependencies
echo Installing dependencies with uv...
uv sync

echo Installation complete! Virtual environment is active.
echo To deactivate the virtual environment, run: deactivate
pause