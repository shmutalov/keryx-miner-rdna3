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
set "POOL=stratum+tcp://krx.suprnova.cc:4404"
REM set "POOL=stratum+tcp://krx.suprnova.cc:4401"
REM set "POOL=stratum+tcp://krx.baikalmine.com:9020"
REM set "POOL=stratum+tcp://eu.miningcrib.com:7212"
REM set "POOL=stratum+tcp://pool.ddsolutions.ai:5555"
REM set "POOL=stratum+tcp://sg.keryx.dongqn.com:5555"

set "WORKER=rx7900xt"

REM --- GPU PoW batch size (nonces per dispatch). 4194304 (4M) is tuned for the
REM     7900 XT (~710 MH/s). Lower it if the rig feels unresponsive. ---
set "KERYX_VULKAN_WORKLOAD=4194304"

REM --- PoW-only: mine kHeavyHash shares on the pool WITHOUT OPoI models or
REM     llama-server. Comment out the next line (REM in front) to run the full
REM     miner instead (downloads the OPoI model GGUFs and launches llama-server;
REM     requires target\release\llama\llama-server.exe to be present). ---
set "KERYX_POW_ONLY=0"
set "KERYX_SKIP_LEGACY_MODELS=1"

set RUST_LOG=debug
set RUST_BACKTRACE=1

REM ----------------------------------------------------------------------------
cd /d "%~dp0"
set "MINER=%~dp0target\release\keryx-miner-rdna3.exe"

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
"%MINER%" --mining-address "%MINING_ADDRESS%" --keryxd-address "%POOL%" --worker "%WORKER%"

echo.
echo Miner exited (code %errorlevel%). Restarting in 5s -- close this window to stop.
timeout /t 5 >nul
goto run
