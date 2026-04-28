from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.multiscale import MultiScaleHomographyCNN, MultiScaleRegistrationCNN
from training_dataset import (
    SyntheticHomographyRegistrationDataset,
    SyntheticShiftRegistrationDataset,
    discover_pairs,
    split_pairs,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"
FIXED_SAMPLE_EVAL_SCRIPT = BASE_DIR / "run_fixed_sample_compare.py"
DEFAULT_FIXED_EVAL_BASELINE = MODEL_DIR / "multiscale_highres_ft_best.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multi-scale SAR/optical registration model.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing SAR/optical folders or MultiResSAR subsets.")
    parser.add_argument("--sar-dir", default="", help="Optional override for the SAR image directory.")
    parser.add_argument("--optical-dir", default="", help="Optional override for the optical image directory.")
    parser.add_argument(
        "--resolution-subset",
        default="",
        choices=["", "High-Resolution", "Low-Resolution"],
        help="Use a MultiResSAR subset such as High-Resolution or Low-Resolution.",
    )
    parser.add_argument("--output-dir", default="training_runs/multiscale", help="Where checkpoints and logs are saved.")
    parser.add_argument("--task", default="translation", choices=["translation", "homography"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--input-size", type=int, default=512)
    parser.add_argument("--max-shift", type=int, default=32)
    parser.add_argument("--max-corner-shift", type=float, default=32.0)
    parser.add_argument("--max-rotation-deg", type=float, default=10.0)
    parser.add_argument("--max-scale-jitter", type=float, default=0.08)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-seed-offset", type=int, default=100000, help="Offset added to the global seed for deterministic validation transforms.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--prefetch-factor", type=int, default=2, help="Prefetch batches per worker when num_workers > 0.")
    parser.add_argument("--init-checkpoint", default="", help="Optional checkpoint for fine-tuning.")
    parser.add_argument("--max-pairs", type=int, default=0, help="Optional cap for quick smoke tests.")
    parser.add_argument("--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps for small-GPU training.")
    parser.add_argument("--disable-amp", action="store_true", help="Disable automatic mixed precision on CUDA.")
    parser.add_argument("--skip-promotion", action="store_true", help="Do not auto-promote the best checkpoint into models/.")
    parser.add_argument("--fixed-eval-samples", default="", help="Comma-separated sample ids for fixed Web-style comparison, e.g. 1,10,12,77,100.")
    parser.add_argument("--fixed-eval-python", default="python", help="Python interpreter used to run fixed sample evaluation.")
    parser.add_argument("--fixed-eval-baseline", default=str(DEFAULT_FIXED_EVAL_BASELINE), help="Baseline model path used during fixed sample comparison.")
    cache_group = parser.add_mutually_exclusive_group()
    cache_group.add_argument("--cache-images", dest="cache_images", action="store_true", help="Keep preprocessed image pairs in worker memory.")
    cache_group.add_argument("--no-cache-images", dest="cache_images", action="store_false", help="Disable in-memory image caching.")
    parser.set_defaults(cache_images=None)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_data_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(args.dataset_root).resolve()
    if args.resolution_subset:
        subset_root = root / args.resolution_subset
        sar_dir = subset_root / "R"
        optical_dir = subset_root / "L"
        return sar_dir, optical_dir

    sar_dir = Path(args.sar_dir).resolve() if args.sar_dir else root / "sar"
    optical_dir = Path(args.optical_dir).resolve() if args.optical_dir else root / "optical"
    return sar_dir, optical_dir


def load_checkpoint_if_needed(model: nn.Module, checkpoint_path: str) -> None:
    if not checkpoint_path:
        return

    path = Path(checkpoint_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model_state = model.state_dict()
    filtered_state_dict = {}
    dropped_keys = []
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if model_state[key].shape != value.shape:
            dropped_keys.append(key)
            continue
        filtered_state_dict[key] = value
    missing, unexpected = model.load_state_dict(filtered_state_dict, strict=False)
    allowed_prefixes = ("match_head.",)
    allowed_missing = all(key.startswith(allowed_prefixes) for key in missing)
    allowed_unexpected = all(key.startswith(allowed_prefixes) for key in unexpected)
    allowed_dropped = all(key.startswith(allowed_prefixes) for key in dropped_keys)
    if (missing or unexpected or dropped_keys) and not (allowed_missing and allowed_unexpected and allowed_dropped):
        raise RuntimeError(
            f"Checkpoint mismatch for {path.name}: missing={missing}, unexpected={unexpected}, dropped={dropped_keys}"
        )


def should_cache_images(args: argparse.Namespace, pair_count: int) -> bool:
    if args.cache_images is not None:
        return bool(args.cache_images)
    return pair_count <= 256 and args.input_size <= 512


def parse_sample_ids(sample_text: str) -> list[int]:
    if not sample_text.strip():
        return []

    sample_ids: list[int] = []
    for token in sample_text.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        sample_ids.append(int(stripped))
    return sample_ids


def build_dataloaders(args: argparse.Namespace):
    sar_dir, optical_dir = resolve_data_dirs(args)
    pairs = discover_pairs(sar_dir, optical_dir)

    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    if len(pairs) < 2:
        raise RuntimeError(
            "At least 2 paired samples are required to start training. "
            f"Current pair count: {len(pairs)}. SAR dir: {sar_dir}. Optical dir: {optical_dir}."
        )

    train_pairs, val_pairs = split_pairs(pairs, train_ratio=args.train_ratio, seed=args.seed)
    if not val_pairs:
        raise RuntimeError("Validation split is empty. Please add more paired samples.")

    cache_images = should_cache_images(args, len(pairs))
    val_seed = int(args.seed + args.val_seed_offset)

    if args.task == "homography":
        train_dataset = SyntheticHomographyRegistrationDataset(
            train_pairs,
            input_size=args.input_size,
            max_corner_shift=args.max_corner_shift,
            max_translation=args.max_shift,
            max_rotation_deg=args.max_rotation_deg,
            max_scale_jitter=args.max_scale_jitter,
            cache_images=cache_images,
            deterministic=False,
            base_seed=args.seed,
        )
        val_dataset = SyntheticHomographyRegistrationDataset(
            val_pairs,
            input_size=args.input_size,
            max_corner_shift=args.max_corner_shift,
            max_translation=args.max_shift,
            max_rotation_deg=args.max_rotation_deg,
            max_scale_jitter=args.max_scale_jitter,
            cache_images=cache_images,
            deterministic=True,
            base_seed=val_seed,
        )
    else:
        train_dataset = SyntheticShiftRegistrationDataset(
            train_pairs,
            input_size=args.input_size,
            max_shift=args.max_shift,
            cache_images=cache_images,
            deterministic=False,
            base_seed=args.seed,
        )
        val_dataset = SyntheticShiftRegistrationDataset(
            val_pairs,
            input_size=args.input_size,
            max_shift=args.max_shift,
            cache_images=cache_images,
            deterministic=True,
            base_seed=val_seed,
        )

    loader_kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if args.num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = max(2, args.prefetch_factor)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, train_pairs, val_pairs, cache_images, val_seed, val_dataset


def pair_records_to_json(pairs) -> list[dict[str, str]]:
    return [
        {
            "index": index,
            "sar_path": str(pair.sar_path),
            "optical_path": str(pair.optical_path),
        }
        for index, pair in enumerate(pairs)
    ]


def write_validation_manifest(
    output_dir: Path,
    args: argparse.Namespace,
    train_pairs,
    val_pairs,
    val_dataset,
    val_seed: int,
) -> tuple[Path, Path]:
    split_manifest_path = output_dir / "data_split.json"
    validation_manifest_path = output_dir / "validation_manifest.json"

    split_manifest = {
        "task": args.task,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "resolution_subset": args.resolution_subset,
        "train_ratio": args.train_ratio,
        "seed": args.seed,
        "val_seed": val_seed,
        "train_pairs": pair_records_to_json(train_pairs),
        "val_pairs": pair_records_to_json(val_pairs),
    }
    validation_manifest = {
        "task": args.task,
        "deterministic": True,
        "base_seed": val_seed,
        "sample_count": len(val_pairs),
        "samples": [val_dataset.describe_sample(index) for index in range(len(val_pairs))],
    }

    with split_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(split_manifest, file, ensure_ascii=False, indent=2)
    with validation_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(validation_manifest, file, ensure_ascii=False, indent=2)

    return split_manifest_path, validation_manifest_path


def run_fixed_sample_compare(
    args: argparse.Namespace,
    checkpoint_path: Path,
    output_dir: Path,
    epoch: int,
) -> dict | None:
    sample_ids = parse_sample_ids(args.fixed_eval_samples)
    if not sample_ids:
        return None

    if not FIXED_SAMPLE_EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"Fixed sample evaluation script not found: {FIXED_SAMPLE_EVAL_SCRIPT}")

    compare_dir = output_dir / "fixed_eval" / f"epoch_{epoch:03d}"
    compare_dir.mkdir(parents=True, exist_ok=True)

    command = [
        args.fixed_eval_python,
        str(FIXED_SAMPLE_EVAL_SCRIPT),
        "--model-path",
        str(checkpoint_path),
        "--output-dir",
        str(compare_dir),
        "--dataset-root",
        str(Path(args.dataset_root).resolve()),
        "--sample-ids",
        ",".join(str(sample_id) for sample_id in sample_ids),
    ]
    if args.resolution_subset:
        command.extend(["--resolution-subset", args.resolution_subset])
    if args.sar_dir:
        command.extend(["--sar-dir", str(Path(args.sar_dir).resolve())])
    if args.optical_dir:
        command.extend(["--optical-dir", str(Path(args.optical_dir).resolve())])
    if args.fixed_eval_baseline:
        command.extend(["--baseline-model", str(Path(args.fixed_eval_baseline).resolve())])

    process = subprocess.run(
        command,
        cwd=str(BASE_DIR),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )

    stdout_path = compare_dir / "stdout.log"
    stderr_path = compare_dir / "stderr.log"
    stdout_path.write_text(process.stdout or "", encoding="utf-8")
    stderr_path.write_text(process.stderr or "", encoding="utf-8")

    if process.returncode != 0:
        return {
            "success": False,
            "epoch": epoch,
            "path": str(compare_dir),
            "error": process.stderr.strip() or process.stdout.strip() or "Fixed sample evaluation failed.",
        }

    stdout_lines = [line.strip() for line in (process.stdout or "").splitlines() if line.strip()]
    if not stdout_lines:
        return {
            "success": False,
            "epoch": epoch,
            "path": str(compare_dir),
            "error": "Fixed sample evaluation completed without JSON output.",
        }

    aggregate = json.loads(stdout_lines[-1])
    return {
        "success": True,
        "epoch": epoch,
        "path": str(compare_dir),
        "aggregate": aggregate,
    }


def configure_runtime(device: torch.device) -> None:
    if device.type != "cuda":
        return

    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")


def compute_epoch_metrics(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.cuda.amp.GradScaler | None = None,
    grad_accum_steps: int = 1,
    use_amp: bool = False,
    use_channels_last: bool = False,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_mae = 0.0
    total_rmse = 0.0
    total_samples = 0

    if training:
        optimizer.zero_grad(set_to_none=True)

    amp_context = torch.autocast if use_amp and device.type == "cuda" else None

    for step_index, batch in enumerate(loader, start=1):
        sar = batch["sar"].to(device, non_blocking=device.type == "cuda")
        optical = batch["optical"].to(device, non_blocking=device.type == "cuda")
        target = batch["target"].to(device, non_blocking=device.type == "cuda")
        if use_channels_last and sar.dim() == 4 and optical.dim() == 4:
            sar = sar.contiguous(memory_format=torch.channels_last)
            optical = optical.contiguous(memory_format=torch.channels_last)

        autocast_ctx = amp_context(device_type="cuda", dtype=torch.float16) if amp_context else nullcontext()
        with torch.set_grad_enabled(training), autocast_ctx:
            prediction = model(sar, optical)
            loss = criterion(prediction, target)
            if training:
                scaled_loss = loss / max(1, grad_accum_steps)
                if scaler is not None and use_amp and device.type == "cuda":
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                should_step = (step_index % max(1, grad_accum_steps) == 0) or (step_index == len(loader))
                if should_step:
                    if scaler is not None and use_amp and device.type == "cuda":
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

        batch_size = target.size(0)
        error = prediction.detach() - target
        mae = error.abs().mean(dim=1).sum().item()
        rmse = torch.sqrt((error ** 2).mean(dim=1)).sum().item()

        total_loss += loss.item() * batch_size
        total_mae += mae
        total_rmse += rmse
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "mae": total_mae / total_samples,
        "rmse": total_rmse / total_samples,
        "samples": float(total_samples),
    }


def save_checkpoint(
    output_dir: Path,
    name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_rmse: float,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "val_rmse": val_rmse,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(checkpoint, output_dir / name)


def build_model(args: argparse.Namespace) -> nn.Module:
    if args.task == "homography":
        return MultiScaleHomographyCNN()
    return MultiScaleRegistrationCNN()


def best_checkpoint_name(args: argparse.Namespace) -> str:
    if args.task == "homography":
        return "multiscale_homography_best.pth"
    return "multiscale_best.pth"


def should_promote_best_checkpoint(args: argparse.Namespace, output_dir: Path) -> bool:
    if args.task != "homography":
        return False
    if args.skip_promotion:
        return False
    if args.max_pairs and args.max_pairs > 0:
        return False
    if "smoke" in output_dir.name.lower():
        return False
    return True


def maybe_promote_best_checkpoint(output_dir: Path, args: argparse.Namespace) -> Path | None:
    if not should_promote_best_checkpoint(args, output_dir):
        return None

    source_path = output_dir / best_checkpoint_name(args)
    if not source_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found for promotion: {source_path}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    target_path = MODEL_DIR / source_path.name
    shutil.copy2(source_path, target_path)
    return target_path


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    configure_runtime(device)
    use_amp = (device.type == "cuda") and not args.disable_amp
    use_channels_last = device.type == "cuda"
    fixed_sample_ids = parse_sample_ids(args.fixed_eval_samples)
    train_loader, val_loader, train_pairs, val_pairs, cache_images, val_seed, val_dataset = build_dataloaders(args)
    train_count = len(train_pairs)
    val_count = len(val_pairs)
    split_manifest_path, validation_manifest_path = write_validation_manifest(
        output_dir,
        args,
        train_pairs,
        val_pairs,
        val_dataset,
        val_seed,
    )

    model = build_model(args).to(device)
    if use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    load_checkpoint_if_needed(model, args.init_checkpoint)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp and device.type == "cuda")
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_rmse = math.inf
    best_epoch = 0
    best_fixed_eval: dict | None = None
    history: list[dict[str, object]] = []

    run_meta = {
        "device": str(device),
        "task": args.task,
        "train_pairs": train_count,
        "val_pairs": val_count,
        "dataset_root": str(Path(args.dataset_root).resolve()),
        "use_amp": use_amp,
        "grad_accum_steps": max(1, args.grad_accum_steps),
        "num_workers": args.num_workers,
        "prefetch_factor": max(2, args.prefetch_factor) if args.num_workers > 0 else 0,
        "cache_images": cache_images,
        "channels_last": use_channels_last,
        "cpu_count": os.cpu_count(),
        "val_seed": val_seed,
        "fixed_eval_samples": fixed_sample_ids,
        "data_split_path": str(split_manifest_path),
        "validation_manifest_path": str(validation_manifest_path),
    }
    print(json.dumps(run_meta, ensure_ascii=False))

    for epoch in range(1, args.epochs + 1):
        train_started_at = time.perf_counter()
        train_metrics = compute_epoch_metrics(
            model,
            train_loader,
            criterion,
            device,
            optimizer=optimizer,
            scaler=scaler,
            grad_accum_steps=max(1, args.grad_accum_steps),
            use_amp=use_amp,
            use_channels_last=use_channels_last,
        )
        train_seconds = time.perf_counter() - train_started_at
        val_started_at = time.perf_counter()
        val_metrics = compute_epoch_metrics(
            model,
            val_loader,
            criterion,
            device,
            optimizer=None,
            scaler=None,
            grad_accum_steps=1,
            use_amp=use_amp,
            use_channels_last=use_channels_last,
        )
        val_seconds = time.perf_counter() - val_started_at
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_mae": train_metrics["mae"],
            "train_rmse": train_metrics["rmse"],
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "train_seconds": train_seconds,
            "val_seconds": val_seconds,
            "train_samples_per_sec": train_metrics["samples"] / train_seconds if train_seconds > 0 else 0.0,
            "val_samples_per_sec": val_metrics["samples"] / val_seconds if val_seconds > 0 else 0.0,
        }
        save_checkpoint(output_dir, "last.pth", model, optimizer, epoch, val_metrics["rmse"], args)
        if val_metrics["rmse"] < best_rmse:
            best_rmse = val_metrics["rmse"]
            best_epoch = epoch
            best_checkpoint_path = output_dir / best_checkpoint_name(args)
            save_checkpoint(output_dir, best_checkpoint_name(args), model, optimizer, epoch, best_rmse, args)
            fixed_eval_result = run_fixed_sample_compare(args, best_checkpoint_path, output_dir, epoch)
            if fixed_eval_result is not None:
                epoch_metrics["fixed_eval"] = fixed_eval_result
                if fixed_eval_result.get("success"):
                    aggregate = fixed_eval_result.get("aggregate", {})
                    epoch_metrics["fixed_eval_target_avg_difference_mean"] = aggregate.get("target_avg_difference_mean")
                    epoch_metrics["fixed_eval_baseline_avg_difference_mean"] = aggregate.get("baseline_avg_difference_mean")
                    best_fixed_eval = fixed_eval_result
                else:
                    epoch_metrics["fixed_eval_error"] = fixed_eval_result.get("error")

        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

    with (output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    summary = {
        "task": args.task,
        "best_epoch": best_epoch,
        "best_val_rmse": best_rmse,
        "output_dir": str(output_dir),
        "data_split_path": str(split_manifest_path),
        "validation_manifest_path": str(validation_manifest_path),
    }
    if best_fixed_eval is not None:
        summary["best_fixed_eval"] = best_fixed_eval
    promoted_path = maybe_promote_best_checkpoint(output_dir, args)
    if promoted_path is not None:
        summary["promoted_model_path"] = str(promoted_path)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
