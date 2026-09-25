@echo off
cd /d "%~dp0"
py -3 -m core_rag %*
if errorlevel 9009 python -m core_rag %*
