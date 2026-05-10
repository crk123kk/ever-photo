@echo off
echo Stopping Ever Photo...

taskkill /FI "WINDOWTITLE eq Backend*" /T /F >nul 2>&1
if %errorlevel% equ 0 (echo Backend stopped.) else (echo Backend not running.)

taskkill /FI "WINDOWTITLE eq Frontend*" /T /F >nul 2>&1
if %errorlevel% equ 0 (echo Frontend stopped.) else (echo Frontend not running.)

echo Done.
pause
