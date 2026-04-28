# 远端训练最小上传清单

这个项目的 Web 端可以继续只在本地运行。
远端 Linux 服务器只需要训练链路，不需要 Flask、模板和静态资源。

## 必传文件

- `train_multiscale.py`
- `training_dataset.py`
- `requirements.train.txt`
- `scripts/bootstrap_linux.sh`
- `scripts/train_remote.sh`
- `scripts/train_homography_highres.sh`
- `models/multiscale.py`
- `models/multiscale_homography_stage3_full_best.pth`

## 当前默认不上传

- `app.py`
- `templates/`
- `static/`
- `test_images/`
- `start.sh`
- `requirements.txt`
- 其他与 Web 端相关的脚本和资源
- 低表现模型
- 训练日志和历史输出

## 上传后的远端目录建议

```text
/root/project/web_app_train_only
/root/autodl-tmp/MultiResSAR-datasets
```

## 上传后执行

```bash
cd /root/project/web_app_train_only
bash scripts/bootstrap_linux.sh
DATASET_ROOT=/root/autodl-tmp/MultiResSAR-datasets bash scripts/train_homography_highres.sh
```

## 如果你后续需要这些功能，再额外补传

- 固定样本可视化评估：
  需要补传 `run_fixed_sample_compare.py`、`app.py`，以及相关 Web 推理依赖
- 其他任务或模型：
  需要补传对应模型定义和 checkpoint
