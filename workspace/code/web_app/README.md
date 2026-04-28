# SAR-光学影像配准 Web 系统

## 🌐 系统特点

- ✅ **图形化界面**：无需命令行，浏览器即可使用
- ✅ **拖拽上传**：支持 SAR 和光学影像拖拽上传
- ✅ **实时可视化**：棋盘格叠加图和彩色合成图
- ✅ **结果下载**：一键下载配准结果（GeoTIFF 格式）
- ✅ **GPU 加速**：自动检测并使用 GPU（如果可用）

## 🚀 快速开始

### 1. 安装依赖

```bash
cd web_app
pip3 install -r requirements.txt
```

### 2. 启动系统

```bash
# Linux/Mac
bash start.sh

# Windows
python app.py
```

### 3. 访问系统

打开浏览器访问：**http://localhost:5000**

## 📖 使用说明

### 步骤 1：上传影像

1. **上传 SAR 影像**
   - 点击或拖拽 SAR 影像文件
   - 支持格式：`.tif`, `.tiff`, `.npy`
   - 单通道（高动态范围）

2. **上传光学影像**
   - 点击或拖拽光学影像文件
   - 支持格式：`.tif`, `.tiff`
   - RGB 三通道

### 步骤 2：执行配准

- 点击"🚀 开始配准"按钮
- 等待配准完成（通常几秒到几分钟）

### 步骤 3：查看结果

系统会显示两种可视化结果：

1. **棋盘格叠加图**
   - SAR 和光学影像交替显示
   - 用于专业验证配准精度
   - 边界对齐情况一目了然

2. **彩色合成图**
   - SAR = 红色通道
   - 光学 = 绿色 + 蓝色通道
   - 重合区域显示黄色
   - 直观展示配准效果

### 步骤 4：下载结果

- 点击"💾 下载配准结果"
- 下载 GeoTIFF 格式的配准后 SAR 影像
- 保留原始地理坐标信息

## 🏗️ 系统架构

```
web_app/
├── app.py              # Flask 后端
├── templates/
│   └── index.html      # 前端界面
├── requirements.txt    # Python 依赖
├── start.sh           # 启动脚本
└── README.md          # 本文档
```

## 🔧 API 接口

### 上传文件

```
POST /api/upload
Content-Type: multipart/form-data

参数:
- sar: SAR 影像文件
- optical: 光学影像文件

返回:
{
    "success": true,
    "session_id": "uuid",
    "sar_path": "...",
    "optical_path": "..."
}
```

### 执行配准

```
POST /api/register
Content-Type: application/json

{
    "session_id": "uuid"
}

返回:
{
    "success": true,
    "checkerboard_url": "...",
    "overlay_url": "...",
    "registered_url": "..."
}
```

### 获取结果

```
GET /api/results/<session_id>/<filename>
```

### 检查模型状态

```
GET /api/model/status

返回:
{
    "model_loaded": true,
    "device": "cuda"
}
```

## 📊 演示模式

如果没有预训练模型，系统会自动进入演示模式：

- ✅ 可以上传文件
- ✅ 可以查看示例结果
- ⚠️ 配准结果为示例数据

## 🎨 界面预览

### 上传界面
- 渐变色背景
- 卡片式布局
- 拖拽上传区域
- 实时文件信息显示

### 结果界面
- 并排显示两种可视化
- 高分辨率图像展示
- 一键下载按钮

## ⚙️ 配置选项

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `FLASK_ENV` | development | 运行环境 |
| `MAX_CONTENT_LENGTH` | 100MB | 最大上传文件大小 |
| `INPUT_SIZE` | 512 | 输入图像尺寸 |

### 模型配置

编辑 `app.py` 中的 `initialize_model()` 函数：

```python
register = SAROpticalRegister(
    model_path='checkpoints/model.pth',  # 模型路径
    device='cuda',                        # cuda 或 cpu
    input_size=512                        # 256 或 512
)
```

## 🐛 故障排除

### 问题 1：端口被占用

```
Error: Address already in use
```

**解决**：修改端口号

```python
app.run(host='0.0.0.0', port=5001, debug=True)
```

### 问题 2：依赖安装失败

```
ERROR: Could not find a version that satisfies the requirement...
```

**解决**：升级 pip

```bash
pip3 install --upgrade pip
pip3 install -r requirements.txt
```

### 问题 3：GPU 未检测到

```
device: cpu
```

**解决**：

1. 确保已安装 CUDA
2. 安装 GPU 版 PyTorch

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## 📄 许可证

MIT License

## 👨‍💻 作者

林浩铭 - 哈尔滨工业大学 - 计算机科学与技术

## 📧 联系

如有问题，请提交 Issue 或联系作者。
