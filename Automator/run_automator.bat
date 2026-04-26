@echo off
setlocal EnableDelayedExpansion

set "REPO=C:\Users\virat.arya\ETG\SoftsDatabase - Documents\Database\Hardmine\ICEBREAKER\COT_Prediction"
set "LOG=%REPO%\Automator\automator.log"
set "INGEST=%REPO%\Ingest\Ingest.py"
set "PARQUET=Database\cot_prediction.parquet"

for /f %%a in ('powershell -NoProfile -Command "Get-Date -Format yyyy-MM-dd"') do set TODAY=%%a
for /f %%a in ('powershell -NoProfile -Command "Get-Date -Format HH:mm:ss"') do set NOW=%%a

echo. >> "%LOG%"
echo ======================================================== >> "%LOG%"
echo [%TODAY% %NOW%] Automator started >> "%LOG%"

:: Step 1: Run Ingest
echo [%TODAY% %NOW%] Running Ingest.py... >> "%LOG%"
python "%INGEST%" >> "%LOG%" 2>&1

if %ERRORLEVEL% neq 0 (
    echo [%TODAY% %NOW%] ERROR: Ingest.py failed. Aborting. >> "%LOG%"
    exit /b 1
)
echo [%TODAY% %NOW%] Ingest.py completed. >> "%LOG%"

:: Step 2: Git commit and push
cd /d "%REPO%"

for /f "delims=" %%i in ('git status --porcelain "%PARQUET%" 2^>^&1') do set CHANGED=%%i

if "!CHANGED!"=="" (
    echo [%TODAY% %NOW%] Parquet unchanged - skipping commit. >> "%LOG%"
    goto :done
)

git add "%PARQUET%" >> "%LOG%" 2>&1
git commit -m "update: cot_prediction %TODAY%" >> "%LOG%" 2>&1

if %ERRORLEVEL% neq 0 (
    echo [%TODAY% %NOW%] ERROR: git commit failed. >> "%LOG%"
    exit /b 1
)

git push >> "%LOG%" 2>&1

if %ERRORLEVEL% neq 0 (
    echo [%TODAY% %NOW%] ERROR: git push failed. >> "%LOG%"
    exit /b 1
)

echo [%TODAY% %NOW%] Pushed to GitHub. Streamlit will auto-refresh. >> "%LOG%"

:done
echo [%TODAY% %NOW%] Automator finished. >> "%LOG%"
endlocal
