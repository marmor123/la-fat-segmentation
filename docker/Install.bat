@echo off
REM ---------------------------------------------------------------------------
REM LA Fat Segmentation — Windows Installer
REM ---------------------------------------------------------------------------
REM Double-click this file to install.  Requires Docker Desktop.
REM Re-running is safe — your data folder is never deleted.
REM ---------------------------------------------------------------------------

echo ============================================================
echo   LA FAT SEGMENTATION — INSTALLER
echo ============================================================
echo.

REM --- Check for Docker ----------------------------------------------------
where docker >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Docker is not installed or not on PATH.
    echo Please install Docker Desktop from https://www.docker.com/products/docker-desktop/
    echo.
    pause
    exit /b 1
)
echo [OK] Docker found.

REM --- Load image -----------------------------------------------------------
set IMAGE_TAR=%~dp0la-fat-image.tar
if not exist "%IMAGE_TAR%" (
    echo [ERROR] Image file not found: %IMAGE_TAR%
    echo Please ensure la-fat-image.tar is in the same folder as this script.
    pause
    exit /b 1
)

echo Loading Docker image...
docker load -i "%IMAGE_TAR%"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Failed to load Docker image.
    pause
    exit /b 1
)
echo [OK] Image loaded.

REM --- Create data folder on Desktop ---------------------------------------
set DATA_DIR=%USERPROFILE%\Desktop\la-fat-data
if not exist "%DATA_DIR%" (
    mkdir "%DATA_DIR%"
    mkdir "%DATA_DIR%\data\raw"
    mkdir "%DATA_DIR%\data\intermediate"
    mkdir "%DATA_DIR%\outputs"
    echo [OK] Created data folder: %DATA_DIR%
) else (
    echo [OK] Data folder already exists: %DATA_DIR%
)

REM --- Copy desktop shortcuts -----------------------------------------------
set SCRIPT_DIR=%~dp0
copy /Y "%SCRIPT_DIR%Process Scans.bat" "%USERPROFILE%\Desktop\Process Scans.bat" >nul
copy /Y "%SCRIPT_DIR%View Results.bat"   "%USERPROFILE%\Desktop\View Results.bat" >nul
echo [OK] Desktop shortcuts created.

echo.
echo ============================================================
echo   INSTALL COMPLETE
echo ============================================================
echo.
echo   Data folder:  %DATA_DIR%
echo   Desktop shortcuts:
echo     - "Process Scans"  (double-click to process CT scans)
echo     - "View Results"   (double-click to view dashboard)
echo.
echo   Usage:
echo     1. Drop .nii.gz CT scans into: %DATA_DIR%\data\raw\
echo     2. Double-click "Process Scans" on your Desktop
echo     3. Double-click "View Results" to view QA dashboards
echo.
pause
