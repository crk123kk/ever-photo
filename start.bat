@echo off
echo Starting Ever Photo Backend...
start "Backend" cmd /k "cd /d %~dp0backend && set FORCE_CPU=1 && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --timeout-keep-alive 120"

echo Starting Ever Photo Frontend...
start "Frontend" cmd /k "cd /d %~dp0frontend && npm run dev"

echo.
echo Backend:  http://localhost:8000
echo Frontend: http://localhost:3000
echo.
echo Note: FORCE_CPU=1 is set by default.
echo       Remove it after installing PyTorch with Blackwell (sm_120) support.
echo.
echo Press any key to stop all services...
pause >nul
