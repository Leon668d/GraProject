@echo off
chcp 65001 >nul
set "WEB_APP_DIR=C:\Users\86158\Desktop\codxRoot\workspace\code\web_app"
set "OUT_DIR=C:\Users\86158\Desktop\codxRoot\workspace\diffusion_lightglue_matcher_compare_100"
set "PYTHON=E:\Anaconda3\envs\sar_diff\python.exe"

if not exist "%OUT_DIR%" mkdir "%OUT_DIR%"
cd /d "%WEB_APP_DIR%"

"%PYTHON%" ".\scripts\sweep_diffusion_lightglue_params.py" ^
  --pairs-csv "C:\Users\86158\Desktop\codxRoot\workspace\diffusion_lightglue_param_sweep\filtered_pairs.csv" ^
  --checkpoint "E:\checkPoint\checkPoint\checkpoint-25001-lcm-adv\model.safetensors" ^
  --output-dir "%OUT_DIR%" ^
  --limit 100 ^
  --verify-limit 0 ^
  --steps 8 ^
  --keypoints 2048 ^
  --extractors superpoint disk aliked ^
  --top-k 3 ^
  --device cuda ^
  --skip-filter ^
  --timeout-seconds 900 ^
  --contact-sheet-limit 90 ^
  > "%OUT_DIR%\sweep_stdout.log" 2> "%OUT_DIR%\sweep_stderr.log"
