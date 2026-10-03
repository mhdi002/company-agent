@echo off
rem ProposalAgent - install everything and start the app (Windows). Options: see run.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run.ps1" %*
if errorlevel 1 pause
