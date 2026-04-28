@echo off
chcp 65001 >nul
set "DIFFUSION_RUNTIME_PYTHON=E:\Anaconda3\envs\sar_diff\python.exe"
set "DIFFUSION_DEFAULT_STEPS=8"
set "DIFFUSION_DEFAULT_MAX_KEYPOINTS=2048"
set "DIFFUSION_DEFAULT_EXTRACTOR_POLICY=cascade"
set "DIFFUSION_DEFAULT_EXTRACTORS=superpoint aliked"
set "DIFFUSION_DEFAULT_MATCH_PREPROCESS=rgb"
set "DIFFUSION_TIMEOUT_SECONDS=900"

echo ======================================
echo SAR-Optical Web App with Diffusion + LightGlue
echo ======================================
echo Diffusion runtime: %DIFFUSION_RUNTIME_PYTHON%
echo Generator: E:\checkPoint\checkPoint\checkpoint-25001-lcm-adv\model.safetensors
echo URL: http://localhost:5000
echo ======================================

start http://localhost:5000
python app.py
pause
