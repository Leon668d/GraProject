# AutoDL 协作与操作指南

## 1. 目标

这份文档解决两件事：

1. 如何把当前项目迁移到 AutoDL 上训练。
2. 你如何与我协作，让我持续辅助你完成部署、训练、验收和选模。

当前结论很明确：

- 本机 Windows 环境不适合继续承担大规模 Sentinel-1/2 训练。
- AutoDL 更适合承担大数据预训练任务。
- 最稳的路线不是“直接在 AutoDL 上训最终模型”，而是：

```text
AutoDL 上做 Sentinel-1/2 预训练
-> 回到 MultiRes 做微调
-> 用现有 Web 系统做固定样例验收
```

---

## 2. 当前推荐的整体方案

### 2.1 训练主线

建议按三阶段执行：

1. `Stage A: Sentinel-1/2 translation 预训练`
   目标：学习跨模态粗配准能力。

2. `Stage B: Sentinel-1/2 small residual 几何微调`
   目标：先学小旋转、小缩放、小平移残差。
   不建议一上来直接训 full homography。

3. `Stage C: 回到 MultiRes 微调`
   目标：让最终模型真正适配你的 Web 系统和论文任务。

### 2.2 为什么这么做

- Sentinel-1/2 数据量大，适合学跨模态特征。
- MultiRes 更贴近你的真实任务和答辩展示。
- 你当前 Web 验收链路已经围绕 MultiRes 建好了。

所以：

`Sentinel-1/2 负责预训练，MultiRes 负责最终定版。`

---

## 3. 推荐的 AutoDL 配置

### 3.1 实例建议

优先选择：

- `1x RTX 4090 24GB`

备选：

- `1x RTX 3090 24GB`

资源建议：

- GPU：`24GB`
- CPU：`8-16` 核
- 内存：`32-64GB`
- 系统盘：`50GB`
- 数据盘：`200GB`

### 3.2 原因

- 你当前代码是单卡训练逻辑，先上单卡最稳。
- 24GB 显存足够支撑更大的 batch 和更稳定的预训练。
- Sentinel-1/2 数据下载、解压、整理后会明显占空间。

---

## 4. AutoDL 上建议的目录结构

建议统一成下面这样：

```text
/root/autodl-tmp/
  gra/
    web_system/web_app/
  datasets/
    sentinel12_raw/
    sentinel12_flat/
      sar/
      optical/
      meta/
  training_runs/
```

说明：

- 项目代码放在 `/root/autodl-tmp/gra`
- 训练数据放在 `/root/autodl-tmp/datasets`
- 输出单独放 `/root/autodl-tmp/training_runs`

---

## 5. 你如何和我协作

最重要的一点：

**我不能直接“远程接管”你 AutoDL 网页里的实例。**

我能高效辅助你的方式是：

1. 你在 AutoDL 上执行我给你的命令。
2. 你把终端输出贴给我。
3. 我根据输出给你下一条命令。

也就是：

`你负责执行，我负责决策。`

这是当前最现实、最稳定的协作方式。

---

## 6. 怎么把“会话”连接到那台机器

这里要分两种情况。

### 6.1 情况 A：当前这个对话仍然运行在你本机

这是你现在最可能的情况。

此时：

- 我能访问的是你当前本机工作目录。
- 我不能直接看到 AutoDL 远端文件系统。
- 我也不能自动切换到那台机器，除非你把这个会话本身开在那台机器里。

换句话说：

**你不能靠一句话把我“传送”到 AutoDL。**

如果当前会话在本机，那么连接远端机器的办法只有两种：

1. 你继续在 AutoDL 终端里执行命令，再把输出发给我。
2. 你在 AutoDL 机器上重新启动一个新的 Codex/CLI 会话，让那个会话直接运行在远端。

### 6.2 情况 B：你想让我直接工作在 AutoDL 机器上

要做到这一点，你需要：

1. 先通过 SSH 或 JupyterLab 进入 AutoDL 机器。
2. 在那台机器上启动新的 Codex/CLI 会话。
3. 让新的会话工作目录指向项目目录，例如：

```bash
cd /root/autodl-tmp/gra
```

然后在那个远端会话里继续和我对话。

这时我看到的环境就会变成：

- Linux 文件系统
- AutoDL 的 GPU
- AutoDL 的 Python 环境
- 远端项目目录

这才叫真正把会话切到那台机器。

### 6.3 你应该怎么理解“连接会话”

你可以把它理解成：

- 不是“让我远程连 SSH”
- 而是“你在远端机器上重新打开一个新的工作会话”

只有这样，我才能直接在那台机器上跑命令、改文件、训练模型。

### 6.4 最推荐的方法

推荐你这样做：

1. 在 AutoDL 上开实例。
2. 用 `VSCode Remote-SSH` 或 AutoDL `JupyterLab Terminal` 连进去。
3. 在远端项目目录启动新的 Codex 会话。
4. 以后训练相关操作都在远端会话里进行。

如果你暂时做不到这一步，也没关系。

你仍然可以用“半手动协作模式”：

- 你在 AutoDL 终端执行命令
- 我根据输出继续指导你

---

## 7. 你第一次进入 AutoDL 后应该做什么

无论你是用 JupyterLab 还是 SSH，第一次进入后先执行这几条：

```bash
nvidia-smi
python --version
conda env list
df -h
pwd
ls /root
ls /root/autodl-tmp
```

然后把输出发给我。

我会根据你的真实环境继续给你：

- 建环境命令
- 代码上传命令
- 数据下载命令
- 数据整理命令
- 训练命令

---

## 8. 推荐的 AutoDL 操作方式

### 8.1 连接方式

优先级建议：

1. `VSCode Remote-SSH`
2. `JupyterLab Terminal`
3. 纯 SSH

原因：

- VSCode 最适合看代码、改文件、跑命令
- JupyterLab 最适合图形化操作和快速试跑
- 纯 SSH 最轻，但对新手不够友好

### 8.2 长训练不要裸跑

长时间训练建议使用：

- `tmux`
- `screen`
- 或日志重定向

示例：

```bash
python train_multiscale.py ... > train.log 2>&1
tail -f train.log
```

### 8.3 环境配好后保存镜像

这个非常重要。

当你把：

- Python 环境
- CUDA 依赖
- 项目代码
- 常用工具

都配好之后，应当立即保存镜像。

这样下次重建实例时不用重新配一遍。

---

## 9. 迁移到 AutoDL 后的第一阶段任务

建议按顺序执行：

1. 创建 AutoDL 实例
2. 进入实例终端
3. 检查 GPU 和磁盘
4. 创建训练环境
5. 上传或拉取项目代码
6. 下载 Sentinel-1/2 数据
7. 整理成当前训练脚本能读的 `sar/optical` 结构
8. 跑一次 smoke 训练
9. 跑正式预训练
10. 导出 best checkpoint

---

## 10. 迁移后如何继续用我

以后你每次给我发消息，最好按这个格式：

```text
我现在在 AutoDL。
当前路径：
执行的命令：
输出如下：
我希望下一步做什么：
```

例如：

```text
我现在在 AutoDL。
当前路径：/root/autodl-tmp/gra/web_system/web_app
执行的命令：nvidia-smi
输出如下：
[粘贴输出]
我希望下一步开始配置环境。
```

这样我就能直接给你下一条最合适的命令。

---

## 11. 你现在最该做的事

当前建议的最优先动作不是继续在本机折腾，而是：

1. 去 AutoDL 开 `4090` 或 `3090` 单卡实例
2. 进入终端
3. 执行下面这 7 条命令
4. 把输出发给我

```bash
nvidia-smi
python --version
conda env list
df -h
pwd
ls /root
ls /root/autodl-tmp
```

只要你把这一步做完，我就可以开始带你从 0 部署。

---

## 12. 一句话结论

最稳的协作方式是：

`你在 AutoDL 上执行，我根据输出继续指挥。`

如果你想让我真正直接操作那台机器，那么你需要：

`在 AutoDL 机器上重新启动一个新的远端会话。`

只有那样，我看到的环境才会真正切换到远端。
