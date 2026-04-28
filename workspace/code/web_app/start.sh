#!/bin/bash
# SAR-光学影像配准 Web 系统启动脚本

echo "======================================"
echo "🛰️  SAR-光学影像配准 Web 系统"
echo "======================================"
echo ""

# 检查 Python
if ! command -v python3 &> /dev/null; then
    echo "❌ 错误：未找到 Python3"
    exit 1
fi

echo "✅ Python 版本：$(python3 --version)"
echo ""

# 安装依赖
echo "📦 正在安装依赖..."
pip3 install -r requirements.txt -q
echo "✅ 依赖安装完成"
echo ""

# 启动应用
echo "🚀 启动 Web 服务器..."
echo "📍 访问地址：http://localhost:5000"
echo "======================================"
echo ""

python3 app.py
