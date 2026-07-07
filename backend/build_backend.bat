@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM =========================================================
REM Guardian Vision - Build Backend (Nuitka standalone)
REM
REM Fixes/Improvements:
REM  - No deprecated torch plugin
REM  - Includes ffmpeg.exe/ffprobe.exe explicitly (no "No data files in bin" warning)
REM  - Includes package data for mmcv/mmengine/mmpose/mmdet/ultralytics
REM  - Auto-detects ultralytics + mmpose config folders (no hard-coded paths)
REM  - Cleans old build folders
REM  - Generates run_backend.bat in dist with GV_SEGMENT_SEC=60 etc.
REM =========================================================

cd /d "%~dp0"
set "BACKEND_DIR=%CD%"
set "OUTDIR=%BACKEND_DIR%\build"
set "ASSETS_DIR=%BACKEND_DIR%\assets"
set "BIN_DIR=%BACKEND_DIR%\bin"

echo.
echo [INFO] Working dir: "%BACKEND_DIR%"

REM -----------------------------
REM 0) Stop running backend (prevents file locks / confusion)
REM -----------------------------
echo.
echo [STOP] Closing any running server_ws.exe (if any)...
taskkill /F /IM server_ws.exe >nul 2>&1

REM -----------------------------
REM 1) Activate conda env
REM -----------------------------
set "ENV_NAME=mmpose-gpu"
echo.
echo [ENV] Activating conda env: %ENV_NAME%

call conda activate %ENV_NAME% >nul 2>&1
if errorlevel 1 (
  if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\miniconda3\Scripts\activate.bat" %ENV_NAME%
  ) else if exist "%USERPROFILE%\Miniconda3\Scripts\activate.bat" (
    call "%USERPROFILE%\Miniconda3\Scripts\activate.bat" %ENV_NAME%
  ) else (
    echo [ERROR] Could not activate conda. Use Anaconda Prompt then run build_backend.bat again.
    pause
    exit /b 1
  )
)

echo.
echo [CHECK] Python:
python -c "import sys; print(sys.executable); print(sys.version)"

echo.
echo [CHECK] Nuitka:
python -m nuitka --version

REM -----------------------------
REM 2) Verify required files/folders
REM -----------------------------
echo.
echo [CHECK] Verifying required folders/files...

if not exist "%BACKEND_DIR%\server_ws.py" (
  echo [ERROR] server_ws.py not found in "%BACKEND_DIR%"
  pause
  exit /b 1
)

if not exist "%ASSETS_DIR%" (
  echo [ERROR] assets folder not found: "%ASSETS_DIR%"
  pause
  exit /b 1
)

if not exist "%BIN_DIR%\ffmpeg.exe" (
  echo [ERROR] ffmpeg.exe not found: "%BIN_DIR%\ffmpeg.exe"
  pause
  exit /b 1
)

if not exist "%BIN_DIR%\ffprobe.exe" (
  echo [ERROR] ffprobe.exe not found: "%BIN_DIR%\ffprobe.exe"
  pause
  exit /b 1
)

REM Optional files: create if missing so include-data-file won't fail
if not exist "%BACKEND_DIR%\gv_users.db" (
  echo [WARN] gv_users.db not found; creating empty file...
  type nul > "%BACKEND_DIR%\gv_users.db"
)

if not exist "%BACKEND_DIR%\gv_config.json" (
  echo [WARN] gv_config.json not found; creating default config...
  echo { } > "%BACKEND_DIR%\gv_config.json"
)

REM -----------------------------
REM 3) Auto-detect package dirs (no hard-coded paths)
REM -----------------------------
echo.
echo [DISCOVER] Locating ultralytics/mmpose paths...

for /f "usebackq delims=" %%i in (`python -c "import os, ultralytics; print(os.path.dirname(ultralytics.__file__))"`) do set "ULTRA_DIR=%%i"
for /f "usebackq delims=" %%i in (`python -c "import os, mmpose; print(os.path.dirname(mmpose.__file__))"`) do set "MMPOSE_DIR=%%i"

echo [INFO] ultralytics: "%ULTRA_DIR%"
echo [INFO] mmpose:      "%MMPOSE_DIR%"

set "EXTRA_DATA="

if exist "%ULTRA_DIR%\cfg" (
  set "EXTRA_DATA=!EXTRA_DATA! --include-data-dir=""%ULTRA_DIR%\cfg=ultralytics\cfg"""
)
if exist "%ULTRA_DIR%\assets" (
  set "EXTRA_DATA=!EXTRA_DATA! --include-data-dir=""%ULTRA_DIR%\assets=ultralytics\assets"""
)
if exist "%MMPOSE_DIR%\.mim\configs" (
  set "EXTRA_DATA=!EXTRA_DATA! --include-data-dir=""%MMPOSE_DIR%\.mim\configs=mmpose\configs"""
)

echo.
echo [INFO] Extra data:
echo !EXTRA_DATA!

REM -----------------------------
REM 4) Clean old build outputs
REM -----------------------------
echo.
echo [CLEAN] Removing old build folders...
if exist "%OUTDIR%\server_ws.dist" rmdir /s /q "%OUTDIR%\server_ws.dist"
if exist "%OUTDIR%\server_ws.build" rmdir /s /q "%OUTDIR%\server_ws.build"

REM -----------------------------
REM 5) Build with Nuitka
REM -----------------------------
echo.
echo [BUILD] Running Nuitka...

python -m nuitka "%BACKEND_DIR%\server_ws.py" ^
  --standalone ^
  --follow-imports ^
  --assume-yes-for-downloads ^
  --output-dir="%OUTDIR%" ^
  --output-filename="server_ws.exe" ^
  --include-data-dir="%ASSETS_DIR%=assets" ^
  --include-data-file="%BACKEND_DIR%\gv_users.db=gv_users.db" ^
  --include-data-file="%BACKEND_DIR%\gv_config.json=gv_config.json" ^
  --include-data-file="%BIN_DIR%\ffmpeg.exe=bin\ffmpeg.exe" ^
  --include-data-file="%BIN_DIR%\ffprobe.exe=bin\ffprobe.exe" ^
  --include-package=mmcv ^
  --include-package=mmengine ^
  --include-package=mmpose ^
  --include-package=mmdet ^
  --include-package=ultralytics ^
  --include-package-data=mmcv ^
  --include-package-data=mmengine ^
  --include-package-data=mmpose ^
  --include-package-data=mmdet ^
  --include-package-data=ultralytics ^
  --include-module=mmcv._ext ^
  !EXTRA_DATA!

if errorlevel 1 (
  echo.
  echo [ERROR] Nuitka build failed.
  pause
  exit /b 1
)

REM -----------------------------
REM 6) Verify dist contents
REM -----------------------------
echo.
echo [VERIFY] Dist bin contents:
if not exist "%OUTDIR%\server_ws.dist\bin" mkdir "%OUTDIR%\server_ws.dist\bin"
dir "%OUTDIR%\server_ws.dist\bin"

if not exist "%OUTDIR%\server_ws.dist\bin\ffmpeg.exe" (
  echo [ERROR] ffmpeg.exe missing in dist\bin (recording will fail).
  pause
  exit /b 1
)

REM -----------------------------
REM 7) Create run_backend.bat (consistent runtime env)
REM -----------------------------
echo.
echo [POST] Creating run_backend.bat in dist...

(
  echo @echo off
  echo setlocal EnableExtensions
  echo cd /d "%%~dp0"
  echo REM ---- Runtime defaults (edit if needed) ----
  echo set "GV_RECORD_DIR="
  echo set "GV_SEGMENT_SEC=60"
  echo set "GV_KEEP_HOURS=720"
  echo REM Use bundled ffmpeg
  echo set "GV_FFMPEG=%%CD%%\bin\ffmpeg.exe"
  echo echo [RUN] GV_RECORD_DIR=%%GV_RECORD_DIR%%
  echo echo [RUN] GV_SEGMENT_SEC=%%GV_SEGMENT_SEC%%
  echo echo [RUN] GV_FFMPEG=%%GV_FFMPEG%%
  echo echo Running server_ws.exe...
  echo "%%CD%%\server_ws.exe"
) > "%OUTDIR%\server_ws.dist\run_backend.bat"

echo.
echo [DONE] Build finished.
echo Dist folder:
echo   "%OUTDIR%\server_ws.dist"
echo.
echo To run backend (recommended):
echo   cd /d "%OUTDIR%\server_ws.dist"
echo   run_backend.bat
echo.
pause
