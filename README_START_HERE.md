# SAR-Optical Registration Project Code Package

这个文件夹只保留当前项目的有效代码、前端静态资源、内置演示样例和轻量实验依据文件，用于项目启动、代码讲解和答辩演示。

## 1. 答辩前先读哪几份文档

建议按这个顺序阅读：

```text
1. 毕业设计知识补课文档.md
2. 导师可能提问与回答.md
3. 代码讲解路线.md
4. ARCHITECTURE_AND_CODE_SIZE.md
5. README_START_HERE.md
```

每份文档的用途：

```text
毕业设计知识补课文档.md：补 SAR、扩散、LightGlue、Homography、RMSE 等基础概念。
导师可能提问与回答.md：准备导师追问，尤其是方法选择、实验评价、局限和创新点。
代码讲解路线.md：按模块讲代码，不逐行背代码。
ARCHITECTURE_AND_CODE_SIZE.md：说明系统架构、模块职责和核心代码量。
README_START_HERE.md：说明如何启动项目和交付包里有哪些内容。
```

## 2. 启动 Web 系统

进入 Web 项目目录：

```powershell
cd C:\Users\86158\Desktop\codxRoot\workspace\deliverables\project_effective_code_20260428\workspace\code\web_app
```

使用当前本机 Python 启动：

```powershell
& E:\Anaconda3\python.exe .\app.py
```

启动后访问：

```text
http://127.0.0.1:5000/dashboard
```

如果前端样式没有刷新，浏览器执行 `Ctrl + F5`。

## 3. 运行依赖

基础 Web 依赖见：

```text
workspace/code/web_app/requirements.txt
```

扩散 + LightGlue 推理默认依赖外部环境：

```text
E:\Anaconda3\envs\sar_diff\python.exe
```

预训练生成器权重没有复制进代码包，需要继续放在原路径：

```text
E:\checkPoint\checkPoint\checkpoint-25001-lcm-adv\model.safetensors
```

## 4. 目录结构

```text
workspace/code/web_app/
  app.py                         Flask 主程序与 API
  scripts/
    diffusion_lightglue_worker.py        扩散生成 + LightGlue 配准 worker
    filter_registration_pairs.py         样本质量过滤脚本
    sweep_diffusion_lightglue_params.py  参数搜索脚本
    make_failure_contact_sheet.py        边界案例 contact sheet 生成脚本
  static/
    css/                         前端样式
    js/                          Dashboard 前端逻辑
    demo_samples/                内置答辩样例和展示图片
  templates/                     Jinja 页面模板和 dashboard partials
  models/
    baseline.py
    dense_field.py
    multiscale.py                旧 CNN 路线模型代码，仅保留代码，不复制权重
```

轻量实验依据文件：

```text
workspace/diffusion_lightglue_cascade_study_100/
workspace/diffusion_lightglue_param_sweep/
workspace/acd_pretrained_registration_baseline/
```

项目过程文档：

```text
workspace/docs/project/
workspace/WORKSPACE_MAP.md
```

## 5. 代码讲解顺序

1. `app.py`：讲 Flask 路由、登录、上传、模型状态检查、扩散配准 API、内置样例 API。
2. `scripts/diffusion_lightglue_worker.py`：讲 SAR 到 Fake Optical，再用 LightGlue 估计几何关系。
3. `templates/partials/`：讲 dashboard 页面如何拆成配置区、结果区、实验依据区和记录区。
4. `static/js/dashboard/`：讲前端如何加载内置样例、渲染指标、切换图像和处理图片加载兜底。
5. `static/css/dashboard_console.css`：讲答辩展示控制台的页面结构和响应式布局。
6. `scripts/filter_registration_pairs.py` 与 `scripts/sweep_diffusion_lightglue_params.py`：讲样本过滤和参数搜索如何支撑最终默认配置。

## 6. 导师问系统架构和代码量时怎么讲

详细说明见根目录：

```text
ARCHITECTURE_AND_CODE_SIZE.md
```

简短口径：

```text
系统分为表现层、服务层、推理层、实验层和数据证据层。
表现层负责 Web 展示，服务层负责 API 和任务管理，推理层负责扩散生成与 LightGlue 配准，
实验层负责样本过滤和参数搜索，数据证据层负责内置样例和实验统计。
当前交付包核心有效源代码约 9.4k 行，不把图片、CSV、模型权重、数据库、日志和缓存计入代码量。
```

答辩时建议强调：

```text
我的工作不是单独调用一个模型，而是把跨模态配准整理成一条可运行、可展示、可评估的工程链路：
SAR -> Fake Optical -> LightGlue 匹配 -> 几何质量门控 -> SAR 配准结果 -> Web 可视化与实验依据。
```

## 7. 未包含内容

这个代码包没有复制以下内容：

```text
原始数据集 E:\Data
预训练权重 E:\checkPoint\checkPoint\checkpoint-25001-lcm-adv\model.safetensors
历史上传文件 uploads/
历史推理结果 results/
运行日志 logs/
SQLite 运行数据库 app_data.db
Python 缓存 __pycache__/
旧 CNN 权重 .pth
```

这些内容不属于有效代码，且会显著增大交付文件夹体积。
