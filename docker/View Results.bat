@echo off
REM ---------------------------------------------------------------------------
REM LA Fat Segmentation — View Results (Dashboard)
REM ---------------------------------------------------------------------------
REM Double-click this to launch the interactive QA dashboard.
REM Opens your default browser to http://localhost:5006 automatically.
REM Close this terminal window or press Ctrl+C to stop the dashboard.
REM ---------------------------------------------------------------------------

set DATA_DIR=%USERPROFILE%\Desktop\la-fat-data

echo ============================================================
echo   LA FAT SEGMENTATION — DASHBOARD
echo ============================================================
echo.
echo   Starting dashboard server...
echo   Opening browser to http://localhost:5006
echo   Close this window or press Ctrl+C to stop.
echo.

REM Start dashboard in background, then open browser
start "" docker run --rm -v "%DATA_DIR%:/workspace" -p 5006:5006 la-fat dashboard

REM Wait a moment for the server to start, then open the browser
timeout /t 3 /nobreak >nul
start http://localhost:5006

echo.
echo Dashboard is running. Press any key to stop the server.
pause >nul
