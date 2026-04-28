@echo off
chcp 65001 >nul
echo ======================================
echo 🔧 SAR-光学影像配准 Web 系统
echo ======================================
echo.

echo [1/4] 降级 NumPy...
pip uninstall numpy -y >nul 2>&1
pip install "numpy>=1.24.0,<2.0" -q
echo ✅ NumPy 已降级

echo.
echo [2/4] 重新安装依赖...
pip install torch torchvision -q
pip install opencv-python rasterio matplotlib pillow -q
echo ✅ 依赖已安装

echo.
echo [3/4] 验证安装...
python -c "import torch, numpy, cv2, flask; print('✅ 所有依赖正常！')"
if errorlevel 1 (
    echo ❌ 依赖安装失败！
    pause
    exit /b 1
)

echo.
echo [4/4] 启动 Web 服务器...
echo ======================================
echo 🌐 访问地址：http://localhost:5000
echo ======================================
echo.
echo 正在启动，请稍候...
echo.

start http://localhost:5000
python app.py

pause
