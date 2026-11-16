@echo off
setlocal
"__LSS_GATE_PYTHON__" -I -B "%~dp0lss_gh.py" %*
exit /b %errorlevel%
