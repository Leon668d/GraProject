# MultiRes + SEN12MS 完整执行路线图

## 1. 总体策略

当前最稳的路线不是直接拿大规模 `SEN12MS` 作为最终训练集，而是分两阶段推进：

1. 先把 `MultiResSAR-datasets` 这条线做成可答辩、可展示、可写论文的正式成果。
2. 再引入 `SEN12MS` 做跨模态预训练增强，最后回到 `MultiRes` 做微调。

原因如下：

- `MultiResSAR-datasets` 更贴近当前项目场景、Web 系统和毕业设计目标。
- 现有训练脚本、模型推理链路和可视化结果都已围绕 `MultiRes` 打通。
- `SEN12MS` 规模大、整理成本高、算力成本高，更适合作为第二阶段的增强数据，而不是第一阶段主线。

一句话概括：

`先用 MultiRes 做成品，再用 SEN12MS 做增强。`

---

## 2. 当前已有成果

截至目前，项目已经具备以下基础：

- Web 系统主链路可运行，后端主文件：
  - `gra/web_system/web_app/app.py`
- 训练脚本已支持：
  - `translation`
  - `homography`
- 数据集构造脚本已支持：
  - 合成平移扰动
  - 合成单应扰动
- 当前多尺度单应模型已跑过正式训练一轮：
  - 输出目录：`gra/web_system/web_app/training_runs/homography_highres_20260414`
  - 当前最好验证集 `RMSE` 约为 `2.3824`
- `homography` 训练结束后已支持自动推广最佳模型到正式目录：
  - `gra/web_system/web_app/models/multiscale_homography_best.pth`
- 实验模型也已保留：
  - `gra/web_system/web_app/models/multiscale_homography_experimental_best.pth`

这意味着当前已经不是“从零开始”，而是进入“正式训练、验收、写论文、扩展预训练”的阶段。

---

## 3. 第一阶段：先把 MultiRes 做成正式成品

### 3.1 目标

第一阶段目标不是追求最复杂模型，而是做出一版稳定、可展示、可解释、可答辩的正式模型。

要求至少满足：

- 模型训练过程稳定
- 验证指标可复现
- Web 系统能直接调用
- 结果图可用于论文和答辩展示

### 3.2 建议固定训练配置

建议先固定一版正式训练参数，不要每次大改。

推荐配置：

- 数据集：`MultiResSAR-datasets/High-Resolution`
- 任务：`homography`
- 输入尺寸：`256`
- batch size：`4`
- epochs：`20~30`
- 最大角点扰动：`16~20`

推荐命令模板：

```powershell
.\.cnn_runtime\Scripts\python train_multiscale.py `
  --dataset-root C:\Users\86158\Desktop\codxRoot\MultiResSAR-datasets `
  --resolution-subset High-Resolution `
  --task homography `
  --epochs 24 `
  --batch-size 4 `
  --input-size 256 `
  --max-corner-shift 18 `
  --output-dir training_runs\homography_highres_seed42 `
  --seed 42 `
  --init-checkpoint models\multiscale_homography_best.pth
```

### 3.3 正式训练建议

不要只跑一次。

建议至少跑 3 轮，仅修改随机种子：

- `seed = 42`
- `seed = 52`
- `seed = 62`

目的：

- 看模型是否稳定
- 看 best checkpoint 是否偶然
- 为论文实验提供更可靠结果

### 3.4 每轮训练必须保留的产物

每次训练都应保留：

- `history.json`
- `last.pth`
- `multiscale_homography_best.pth`
- 一份训练记录文档

训练记录文档至少写明：

- 训练时间
- 数据集
- 输入尺寸
- batch size
- epochs
- seed
- best val RMSE
- 最终是否入选正式模型

---

## 4. 第二阶段：做正式验收，不只看 loss

### 4.1 验收目标

你的毕业设计不是只交一个 `.pth` 文件，而是要证明：

- 模型训练有效
- 系统端结果更好
- 结果是可解释的

所以必须做“训练指标 + Web 结果”的双验收。

### 4.2 建议构建固定演示样例集

建议固定一组样例用于比较：

- 5 对简单样例
- 5 对中等样例
- 5 对困难样例

总计 `10~20` 对即可。

### 4.3 每个样例要保存的结果

每个样例建议保存：

- 原始输入图
- 旧模型结果
- 新模型结果
- `checkerboard.png`
- `false_color_overlay.png`
- `difference_map.png`
- `contour_overlay.png`

如果是形变模型，还应保留：

- `deformation_heatmap.png`

### 4.4 建议制作验收表

每个样例记录以下字段：

- 样例编号
- 使用模型
- 推理耗时
- difference mean
- 是否优于旧模型
- 主观观察结论

这张表后面可以直接变成论文实验表和答辩讲稿。

---

## 5. 第三阶段：整理可答辩成果

### 5.1 模型版本管理

最终只保留少量关键模型，不要目录里堆满无说明权重。

建议保留：

- `baseline_best.pth`
- `multiscale_best.pth`
- `multiscale_homography_best.pth`
- `multiscale_homography_experimental_best.pth`

其余模型可归档到训练目录，但不要作为正式演示入口。

### 5.2 Web 系统需要保留的展示页面

建议固定以下截图用于论文和答辩：

- 登录页
- 控制台首页
- 模型选择区
- 结果展示区
- 历史任务页
- 详情页
- 日志展示区

### 5.3 论文实验部分建议结构

实验章节建议写成：

1. 数据集与实验设置
2. 模型结构说明
3. 训练策略与参数
4. 指标结果
5. 可视化结果
6. 对比分析
7. 失败案例分析

这样老师会更容易理解你的工作是完整的，而不是只做了一个演示网页。

---

## 6. 第四阶段：再引入 SEN12MS 做预训练增强

### 6.1 SEN12MS 的正确定位

`SEN12MS` 更适合做：

- 跨模态预训练
- 提升 SAR/Optical 共享特征表达能力

不适合直接作为最终“真实配准标注数据集”。

因此建议定位为：

`大规模跨模态预训练集`

而不是：

`最终配准模型唯一训练集`

### 6.2 引入顺序

推荐顺序：

1. 整理 SEN12MS 为配对格式
2. 先跑小子集
3. 做 `translation` 预训练
4. 做 `homography` 预训练
5. 回到 `MultiRes` 做微调

### 6.3 不建议一开始全量训练

第一次只建议使用：

- `2万 ~ 5万` 对样本

原因：

- 节省 AutoDL 成本
- 降低数据整理风险
- 便于快速验证方案是否有效

### 6.4 SEN12MS 训练后如何使用

正确流程：

```text
SEN12MS 预训练
-> 得到更好的跨模态初始化权重
-> 在 MultiRes 上微调
-> 用微调后的模型作为最终模型
```

不要直接把预训练权重拿去当最终答辩模型。

---

## 7. 第五阶段：什么时候再去 AutoDL

### 7.1 当前不建议立刻租卡

在以下条件满足之前，不建议马上去 AutoDL：

- `MultiRes` 正式训练方案还没完全稳定
- 演示样例和验收表还没做好
- 最终正式模型还没锁定

### 7.2 推荐的上云时机

满足以下条件后，再上 AutoDL 最划算：

- 本地 `MultiRes` 主线已经跑通
- 你明确要做 `SEN12MS` 子集预训练
- 你已经知道要跑什么命令、保存什么模型、看什么指标

### 7.3 推荐 GPU

建议优先：

- `1x RTX 4090 24GB`

备选：

- `1x A5000 24GB`

除非后面模型显著变大，否则不建议一开始上更贵卡型。

---

## 8. 推荐执行顺序

建议你严格按下面顺序推进：

### Step 1

继续跑 `MultiRes` 的 `homography` 正式训练，至少 3 个 seed。

### Step 2

建立固定演示样例集。

### Step 3

完成“旧模型 vs 新模型”的系统结果对比。

### Step 4

选出最终正式模型，锁定：

- `models/multiscale_homography_best.pth`

### Step 5

把实验表、结果图、系统截图整理出来，写进论文。

### Step 6

再开始 `SEN12MS` 的小规模整理与预训练。

### Step 7

回到 `MultiRes` 做微调。

### Step 8

完成最终答辩模型与最终 PPT。

---

## 9. 一周内建议任务安排

### 第 1 天

- 固定正式训练参数
- 跑第 1 轮 `seed=42`

### 第 2 天

- 跑第 2 轮 `seed=52`
- 跑第 3 轮 `seed=62`

### 第 3 天

- 比较三轮训练结果
- 选出候选正式模型

### 第 4 天

- 建立演示样例集
- 批量跑 Web 验收

### 第 5 天

- 完成实验对比表
- 整理结果图和失败案例

### 第 6 天

- 写论文实验部分
- 截系统图

### 第 7 天

- 如果主线稳定，再准备 `SEN12MS` 子集整理方案

---

## 10. 结论

你的主线应该是：

```text
MultiRes 做成品
-> Web 验收
-> 论文与答辩材料完善
-> SEN12MS 做增强预训练
-> 回到 MultiRes 微调
-> 最终答辩
```

这条路线的优点是：

- 风险低
- 成本可控
- 更符合毕业设计目标
- 能把“算法 + 系统 + 实验 + 展示”四部分统一起来

如果后续继续推进，优先级永远是：

`先把 MultiRes 做稳，再谈 SEN12MS 放大训练。`
