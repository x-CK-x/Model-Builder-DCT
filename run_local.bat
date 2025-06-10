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
if not exist "%CONDA_ROOT%\Scripts\conda.exe" (
    echo XXX Cannot find conda at %CONDA_ROOT%
    pause
    exit /b 1
)

rem  Make conda functions available in this shell
call "%CONDA_ROOT%\Scripts\activate.bat" base

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
)

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

:: activate env in THIS shell
call conda activate %ENV_NAME%

:: launch the app ------------------------------------------------------
echo > Launching JointTagger …
call python app.py
goto :eof

:fail
echo XXX Setup failed – see messages above.
exit /b 1
