@echo off
rem One-click launcher for DeepTutor (backend + frontend together).
rem
rem Usage (double-click, or from a terminal):
rem   start.cmd                  # production mode: builds web\, then starts
rem   start.cmd --dev            # Next.js dev server with hot reload
rem   start.cmd --init           # run the config wizard first, then start
rem   start.cmd --check          # verify the environment and exit (no start)
rem
rem Extra flags are passed through to `deeptutor start`
rem (--detach, --no-browser, --home ...). Stop with Ctrl+C in this window
rem (or `deeptutor stop` after --detach). Default ports: frontend 3782,
rem backend 8001 - your configured ports may differ; the startup banner
rem prints the actual URLs.
rem
rem Interpreter resolution: %DEEPTUTOR_PYTHON% if set, else the first of
rem python / .venv / the conda env named "deeptutor" that can import
rem deeptutor. If none qualifies, a fresh .venv is bootstrapped with
rem `pip install -e .` (first run only, needs Python 3.11+).

setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
cd /d "%~dp0"

set "PY="
set "ARGS="
set "RUN_INIT=0"

for %%a in (%*) do (
  if "%%a"=="--init" (set "RUN_INIT=1") else (set "ARGS=!ARGS! %%a")
)

call :log "locating Python with DeepTutor installed..."

if defined DEEPTUTOR_PYTHON (
  "%DEEPTUTOR_PYTHON%" -I -c "import deeptutor" >nul 2>nul && set "PY=%DEEPTUTOR_PYTHON%"
)
if not defined PY (
  python -I -c "import deeptutor" >nul 2>nul && set "PY=python"
)
if not defined PY (
  if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" -I -c "import deeptutor" >nul 2>nul && set "PY=.venv\Scripts\python.exe"
  )
)
if not defined PY (
  where conda >nul 2>nul && call :conda_deeptutor
)

if defined PY goto :python_found
call :log "no Python with DeepTutor found - bootstrapping .venv (first run only)"
where python >nul 2>nul
if errorlevel 1 (
  set "FAILMSG=Python not found. Install Python 3.11+ from https://python.org and re-run."
  goto :fail
)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
  set "FAILMSG=Python 3.11+ required."
  goto :fail
)
python -m venv .venv
if errorlevel 1 (
  set "FAILMSG=failed to create .venv"
  goto :fail
)
set "PY=.venv\Scripts\python.exe"
call :log "installing DeepTutor into .venv - pip install -e . - this can take a few minutes"
"!PY!" -m pip install -e .
if errorlevel 1 (
  set "FAILMSG=installation failed - see the pip output above"
  goto :fail
)

:python_found
"%PY%" -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)"
if errorlevel 1 (
  set "FAILMSG=Python 3.11+ required."
  goto :fail
)
for /f "usebackq tokens=*" %%v in (`"%PY%" --version 2^>^&1`) do set "PYVER=%%v"
call :log "Python: %PY% (%PYVER%)"

if exist web (
  if not exist deeptutor_web\server.js (
    where node >nul 2>nul
    if errorlevel 1 (
      call :warn "Node.js 20+ not found - the frontend needs it (https://nodejs.org)."
    ) else (
      for /f "tokens=*" %%v in ('node --version') do call :log "Node.js: %%v"
      if not exist web\node_modules call :log "first start will run npm ci + build the web app (one time)"
    )
  )
)

if not exist data\user\settings\agents.yaml (
  call :warn "no LLM provider configured yet."
  call :warn "run 'start.cmd --init' for the setup wizard, or configure later in Web Settings."
)

echo %* | findstr /c:"--check" >nul
if not errorlevel 1 (
  call :log "environment check passed"
  pause
  exit /b 0
)

if "%RUN_INIT%"=="1" (
  call :log "running the setup wizard (deeptutor init)"
  "%PY%" -m deeptutor_cli init
  if errorlevel 1 (
    set "FAILMSG=init wizard failed"
    goto :fail
  )
)

call :log "starting DeepTutor (close this window or press Ctrl+C to stop both servers)..."
"%PY%" -m deeptutor_cli start%ARGS%
set "EXITCODE=%ERRORLEVEL%"
call :log "DeepTutor exited (code %EXITCODE%)."
pause
exit /b %EXITCODE%

:conda_deeptutor
rem Locate the conda env named "deeptutor" and use its python if it has the package.
set "CENV="
for /f "tokens=1,*" %%i in ('conda env list 2^>nul ^| findstr /r /c:"^deeptutor "') do set "CENV=%%j"
if not defined CENV exit /b 0
if "!CENV:~0,1!"=="*" (
  for /f "tokens=*" %%p in ("!CENV:~1!") do set "CENV=%%p"
)
if not defined CENV exit /b 0
if not exist "!CENV!\python.exe" exit /b 0
"!CENV!\python.exe" -I -c "import deeptutor" >nul 2>nul && set "PY=!CENV!\python.exe"
exit /b 0

:log
echo [start] %~1
exit /b 0

:warn
echo [start] WARNING: %~1
exit /b 0

:fail
echo [start] ERROR: %FAILMSG%
pause
exit /b 1
