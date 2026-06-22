@echo off
REM ---------------------------------------------------------------------------
REM LA Fat Segmentation — Process Scans
REM ---------------------------------------------------------------------------
REM Double-click this to process all CT scans in your data folder.
REM Already-completed patients are skipped automatically.
REM This window stays open so you can read the summary when done.
REM ---------------------------------------------------------------------------

set DATA_DIR=%USERPROFILE%\Desktop\la-fat-data

echo ============================================================
echo   LA FAT SEGMENTATION — PROCESSING SCANS
echo ============================================================
echo.
echo   Data:   %DATA_DIR%\data\raw\
echo   Output: %DATA_DIR%\outputs\
echo.

docker run --rm -v "%DATA_DIR%:/workspace" la-fat pipeline

echo.
echo Processing complete. You may close this window.
echo.
pause
