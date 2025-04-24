@echo off
setlocal EnableDelayedExpansion

REM Step 0: Check if Python 3.13 is installed
python --version 2>nul | findstr /C:"3.13" >nul
if %ERRORLEVEL% NEQ 0 (
    echo Python 3.13 not found. Downloading and installing Python 3.13...
    REM Download Python 3.13 installer
    curl -o python-installer.exe https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe

    REM Install Python 3.13 silently and add to PATH
    echo Installing Python 3.13...
    python-installer.exe /quiet InstallAllUsers=1 PrependPath=1

    REM Clean up
    del python-installer.exe
) else (
    echo Python 3.13 is already installed.
)


REM Repository URL
set REPO_URL=https://github.com/Jafagervik/aigintel.git
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
