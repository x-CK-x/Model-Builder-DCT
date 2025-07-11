@echo off
rem ═══════════════════════════════════════════════════════════════════
rem  JointTagger – Conda + UV launcher  (works from plain cmd.exe)
rem  •   Creates env “jointagger” on first run
rem  •   run_local_conda.bat --update   → re-solve environment.yml
rem  •   Activates env in THIS window, installs uv + wheels, runs app.py
rem ═══════════════════════════════════════════════════════════════════

:: ---------- CONFIG --------------------------------------------------
set "CONDA_ROOT=%USERPROFILE%\Miniconda3"
set "ENV_NAME=jointagger"
set "PY_VER=3.11.9"
:: --------------------------------------------------------------------

setlocal enabledelayedexpansion

:: 0) Install Miniconda automatically if missing
if not exist "%CONDA_ROOT%\Scripts\conda.exe" (
    echo > Installing Miniconda to %CONDA_ROOT% ...
    set "MINICONDA_URL=https://repo.anaconda.com/miniconda/Miniconda3-latest-Windows-x86_64.exe"
    set "MINICONDA_EXE=%TEMP%\miniconda_installer.exe"
    powershell -Command "Invoke-WebRequest -Uri '%MINICONDA_URL%' -OutFile '%MINICONDA_EXE%'" || goto :fail
    if exist "%MINICONDA_EXE%" (
        start /wait "" "%MINICONDA_EXE%" /InstallationType=JustMe /AddToPath=0 /RegisterPython=0 /S /D=%CONDA_ROOT%
        del "%MINICONDA_EXE%"
    ) else (
        echo XXX Miniconda download failed.
        goto :fail
    )
)

echo make ENV?

rem  Make conda functions available in this shell
call "%CONDA_ROOT%\Scripts\activate.bat" base

echo conda base active!

:: check for --update flag
set "DO_UPDATE=0"
if "%~1"=="--update" set "DO_UPDATE=1"

:: -----------------------------------------------------------------
:: 1) Create env ONLY if its folder does not yet exist
:: -----------------------------------------------------------------
if not exist "%CONDA_ROOT%\envs\%ENV_NAME%" (
    echo > Creating Conda env "%ENV_NAME%" …
    if exist environment.yml (
        call conda env create -n %ENV_NAME% -f environment.yml || goto :fail
    ) else (
        call conda create -y -n %ENV_NAME% python=%PY_VER% conda-forge::pytorch-gpu || goto :fail
    )
) else (
    echo CONDA ENV by name FOUND!!!
)

echo env update?

:: -----------------------------------------------------------------
:: 2) Optional --update  → conda env update
:: -----------------------------------------------------------------
if %DO_UPDATE%==1 (
    echo > Updating Conda env "%ENV_NAME%" …
    if exist environment.yml (
        call conda env update -n %ENV_NAME% -f environment.yml --prune || goto :fail
    ) else (
        echo XXX No environment.yml found; skipping update.
    )
)

echo activating env

:: activate env in THIS shell
call conda activate %ENV_NAME%

echo running app

:: launch the app ------------------------------------------------------
echo > Launching JointTagger …
call python app.py
goto :eof

:fail
echo XXX Setup failed – see messages above.
exit /b 1
