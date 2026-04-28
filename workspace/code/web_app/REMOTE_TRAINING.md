# Remote Training Flow

This project is developed locally and then uploaded to a remote Linux server for training.
The web app can stay local. The remote Linux server only needs the training path.
The scripts in `scripts/` are meant for that workflow.

## 1. What to upload

The simplest approach is still to upload the whole `web_app/` directory to the Linux server.
Only the training path is used remotely:

- `train_multiscale.py`
- `training_dataset.py`
- `models/`
- `run_fixed_sample_compare.py` when fixed-sample evaluation is enabled
- `scripts/`
- `requirements.train.txt`

The Flask app is not part of the remote execution flow.

Recommended remote layout:

```text
/root/project/web_app
/root/autodl-tmp/MultiResSAR-datasets
```

If you keep a different dataset path, that is fine. The training script reads it from `DATASET_ROOT`.

## 2. One-time setup on the Linux server

```bash
cd /root/project/web_app
bash scripts/bootstrap_linux.sh
```

Notes:

- `bootstrap_linux.sh` does not force a fresh PyTorch install if your image already has `torch` and `torchvision`.
- It installs only training dependencies from `requirements.train.txt`.
- If you need a clean virtual environment, use:

```bash
cd /root/project/web_app
USE_VENV=1 TORCH_INDEX_URL=https://download.pytorch.org/whl/cu124 bash scripts/bootstrap_linux.sh
```

## 3. One-click training for your current main run

This is the current default for your high-resolution homography training:

```bash
cd /root/project/web_app
DATASET_ROOT=/root/autodl-tmp/MultiResSAR-datasets bash scripts/train_homography_highres.sh
```

Default values in that script:

- task: `homography`
- subset: `High-Resolution`
- epochs: `10`
- batch size: `2`
- grad accumulation: `2`
- learning rate: `3e-4`
- input size: `512`
- workers: `4`
- init checkpoint: `models/multiscale_homography_stage3_full_best.pth`

The script will create:

- logs in `logs/`
- checkpoints and manifests in `training_runs/<run_name>/`

## 4. Override parameters without editing the script

Example:

```bash
cd /root/project/web_app
RUN_NAME=homography_stage4_gpu0 \
DATASET_ROOT=/root/autodl-tmp/MultiResSAR-datasets \
EPOCHS=16 \
BATCH_SIZE=4 \
NUM_WORKERS=8 \
INIT_CHECKPOINT=models/multiscale_homography_best.pth \
bash scripts/train_homography_highres.sh
```

## 5. Generic remote entry script

If you want to run translation or low-resolution experiments, use the generic wrapper:

```bash
cd /root/project/web_app
TASK=translation \
RESOLUTION_SUBSET=Low-Resolution \
DATASET_ROOT=/root/autodl-tmp/MultiResSAR-datasets \
bash scripts/train_remote.sh
```

Common environment variables:

- `DATASET_ROOT`: required, remote dataset root
- `RUN_NAME`: optional custom run name
- `OUTPUT_DIR`: optional absolute or project-relative output path
- `INIT_CHECKPOINT`: optional absolute or project-relative checkpoint path
- `EPOCHS`, `BATCH_SIZE`, `GRAD_ACCUM_STEPS`, `LEARNING_RATE`
- `NUM_WORKERS`, `PREFETCH_FACTOR`
- `CACHE_IMAGES=0|1`
- `DISABLE_AMP=0|1`
- `SKIP_PROMOTION=0|1`
- `FIXED_EVAL_SAMPLES=1,10,12`

## 6. Long-running execution

Use `tmux` on the remote server so training keeps running after you disconnect.

```bash
tmux new -s homography
cd /root/project/web_app
DATASET_ROOT=/root/autodl-tmp/MultiResSAR-datasets bash scripts/train_homography_highres.sh
```

Detach with `Ctrl-b` then `d`.

Later:

```bash
tmux attach -t homography
```

To watch the latest log:

```bash
tail -f logs/*.log
```

## 7. Recommended local-to-remote workflow

1. Modify training code locally in `workspace/code/web_app`.
2. Upload the updated `web_app/` directory to the remote Linux server.
3. Keep local web deployment separate from the training server.
4. SSH into the server and enter the project directory.
5. Run `bash scripts/bootstrap_linux.sh` if training dependencies changed.
6. Start training with `bash scripts/train_homography_highres.sh`.
7. Check `logs/` and `training_runs/` on the remote machine.
