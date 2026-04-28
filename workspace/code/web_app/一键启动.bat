@echo off
chcp 65001 >nul
echo ======================================
echo 🔧 修复并启动 SAR-光学配准 Web 系统
echo ======================================
echo.

echo [1/3] 降级 NumPy...
pip uninstall numpy -y >nul 2>&1
pip install "numpy>=1.24.0,<2.0" -q
if errorlevel 1 (
    echo ❌ NumPy 安装失败！
    pause
    exit /b 1
)
echo ✅ NumPy 已降级到 1.x

echo.
echo [2/3] 验证依赖...
python -c "import flask, numpy, cv2; print('✅ 依赖正常！')" >nul 2>&1
if errorlevel 1 (
    echo ⚠️  部分依赖缺失，正在安装...
    pip install flask opencv-python pillow -q
)

echo.
echo [3/3] 启动 Web 服务器...
echo ======================================
echo 🌐 访问地址：http://localhost:5000
echo ======================================
echo.

start http://localhost:5000
python app.py

pause
