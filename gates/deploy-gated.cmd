@echo off
setlocal
"__LSS_GATE_PYTHON__" -I -B "%~dp0deploy_gated.py" %*
exit /b %errorlevel%
