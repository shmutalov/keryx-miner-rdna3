@echo off
setlocal
title Keryx RDNA3 Miner (RX 7900 XT, Vulkan)

REM ============================================================================
REM  Keryx miner - RDNA3 / Vulkan fork  (AMD Radeon RX 7900 XT)
REM  Edit the values below, then double-click this file to start mining.
REM ============================================================================

REM --- Your Keryx payout address ---
set "MINING_ADDRESS=keryx:qz4zajm2kecfy5c6qctepdnt4666yz0qhff5alnhyy6jjzfm94svxylck57rh"

REM --- Pool and worker name (these are also the built-in defaults) ---
REM set "POOL=stratum+tcp://krx.suprnova.cc:4404"
set "POOL=stratum+tcp://krx.suprnova.cc:4401"
REM set "POOL=stratum+tcp://krx.baikalmine.com:9020"
REM set "POOL=stratum+tcp://eu.miningcrib.com:7212"
REM set "POOL=stratum+tcp://pool.ddsolutions.ai:5555"
REM set "POOL=stratum+tcp://sg.keryx.dongqn.com:5555"
REM set "POOL=stratum+tcp://multipooldd.com:5555"
REM set "POOL=grpc://127.0.0.1:22110"

set "WORKER=rx7900xt"

REM --- PoM batch size (nonces per GPU dispatch). The built-in default (32768,
REM     ~25 ms on the 7900 XT) is fine; uncomment to override. ---
REM set "KERYX_POM_V4_BATCH=32768"

REM --- Testing only: 1 skips the OPoI model download and inference. Every block
REM     on both networks is PoM, which needs the model, so with 1 nothing is
REM     mined. Keep 0 to mine. ---
set "KERYX_POW_ONLY=0"

REM set RUST_LOG=debug
set RUST_LOG=info,llama.cpp=error
REM set RUST_BACKTRACE=1
REM Append every AI prompt and answer, private requests included, to
REM target\release\ai-requests.log (same as --log-ai). Put a file path instead of 1
REM to log elsewhere.
set "KERYX_LOG_AI=1"
REM --high (Qwen3.8-27B) is gated at 24 GB; the 20 GB 7900 XT needs the gate off to run it.
set KERYX_SKIP_VRAM_GATE=1
REM Cap each inference request at 32K tokens. The model allows 64K, which fits the
REM 7900 XT with only ~0.3 GiB of VRAM to spare.
set KERYX_LLAMA_CTX=32768

REM ----------------------------------------------------------------------------
cd /d "%~dp0"
set "RELEASE_DIR=%~dp0target\release\"
set "MINER=%RELEASE_DIR%keryx-miner-rdna3.exe"

REM The miner runs from target\release, next to its models and ipfs.exe. If cargo builds
REM elsewhere (a target dir set in .cargo\config.toml or CARGO_TARGET_DIR, e.g. a short path for
REM MSVC), ask cargo where, and copy a newer build in on every (re)start. Skipped when cargo is
REM not installed or builds into target\ itself.
set "BUILT="
for /f "usebackq delims=" %%d in (`powershell -NoProfile -Command "try { $t = (cargo metadata --format-version 1 --no-deps 2>$null | ConvertFrom-Json).target_directory; if ($t) { [IO.Path]::GetFullPath($t) } } catch {}"`) do set "BUILT=%%d\release\keryx-miner-rdna3.exe"
if /i "%BUILT%"=="%MINER%" set "BUILT="

call :sync_build
if not exist "%MINER%" (
    echo ERROR: miner binary not found at:
    echo   %MINER%
    echo Build it first:  cargo build --release
    echo.
    pause
    exit /b 1
)

echo Starting Keryx miner on the RX 7900 XT ...
echo   Address : %MINING_ADDRESS%
echo   Pool    : %POOL%
echo   Worker  : %WORKER%
echo.

:run
call :sync_build
"%MINER%" --mining-address "%MINING_ADDRESS%" --keryxd-address "%POOL%" --worker "%WORKER%" --very-light

echo.
echo Miner exited (code %errorlevel%). Restarting in 5s -- close this window to stop.
timeout /t 5 >nul
goto run

REM Copy a newer cargo build (if any) next to the models; /D copies only when it is newer.
:sync_build
if defined BUILT if exist "%BUILT%" xcopy /D /Y /Q "%BUILT%" "%RELEASE_DIR%" >nul
exit /b
