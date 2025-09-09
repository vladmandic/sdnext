@echo off

if not defined PYTHON (set PYTHON=python)
if not defined VENV_DIR (set "VENV_DIR=%~dp0%venv")
set ERROR_REPORTING=FALSE
mkdir tmp 2>NUL

:: Conda-Specific Activation
call conda activate "%VENV_DIR%"
if %ERRORLEVEL% neq 0 (
    echo Failed to activate Conda environment: "%VENV_DIR%"
    goto :show_stdout_stderr
)

echo Using Conda env: %VENV_DIR%

:: Check Python
%PYTHON% -c "" >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :check_pip
echo Cannot launch python
goto :show_stdout_stderr

:check_pip
%PYTHON% -mpip --help >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :launch
echo Pip missing in Conda environment, attempting install...
%PYTHON% -m ensurepip >tmp/stdout.txt 2>tmp/stderr.txt
if %ERRORLEVEL% == 0 goto :launch
echo Cannot install pip
goto :show_stdout_stderr

:launch
echo Starting application...
%PYTHON% launch.py --allow-code --enable_insecure_extension_access --use_cuda --xformers %*
pause
exit /b

:show_stdout_stderr
echo.
echo exit code: %errorlevel%

for /f %%i in ("tmp\stdout.txt") do set size=%%~zi
if %size% equ 0 goto :show_stderr
echo.
echo stdout:
type tmp\stdout.txt

:show_stderr
for /f %%i in ("tmp\stderr.txt") do set size=%%~zi
if %size% equ 0 goto :endofscript
echo.
echo stderr:
type tmp\stderr.txt

:endofscript
echo Launch Failed
pause
