from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.dense_field import DenseFieldRegistrationCNN, local_ncc_loss, smoothness_loss, warp_tensor
from training_dataset import SyntheticDenseRegistrationDataset, discover_pairs, split_pairs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an experimental dense-field nonrigid registration model.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--sar-dir", default="")
    parser.add_argument("--optical-dir", default="")
    parser.add_argument(
        "--resolution-subset",
        default="",
        choices=["", "High-Resolution", "Low-Resolution"],
        help="Use a MultiResSAR subset such as High-Resolution or Low-Resolution.",
    )
    parser.add_argument("--output-dir", default="training_runs/dense_field")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--input-size", type=int, default=256)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-pairs", type=int, default=0)
    parser.add_argument("--max-translation", type=int, default=12)
    parser.add_argument("--max-displacement", type=float, default=8.0)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--supervision-weight", type=float, default=1.0)
    parser.add_argument("--similarity-weight", type=float, default=0.2)
    parser.add_argument("--smoothness-weight", type=float, default=0.05)
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
        return subset_root / "R", subset_root / "L"
    sar_dir = Path(args.sar_dir).resolve() if args.sar_dir else root / "sar"
    optical_dir = Path(args.optical_dir).resolve() if args.optical_dir else root / "optical"
    return sar_dir, optical_dir


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, int, int]:
    sar_dir, optical_dir = resolve_data_dirs(args)
    pairs = discover_pairs(sar_dir, optical_dir)
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]
    if len(pairs) < 2:
        raise RuntimeError(
            "At least 2 paired samples are required to start dense-field training. "
            f"Current pair count: {len(pairs)}. SAR dir: {sar_dir}. Optical dir: {optical_dir}."
        )

    train_pairs, val_pairs = split_pairs(pairs, train_ratio=args.train_ratio, seed=args.seed)
    if not val_pairs:
        raise RuntimeError("Validation split is empty. Please add more paired samples.")

    train_dataset = SyntheticDenseRegistrationDataset(
        train_pairs,
        input_size=args.input_size,
        max_translation=args.max_translation,
        max_displacement=args.max_displacement,
        grid_size=args.grid_size,
    )
    val_dataset = SyntheticDenseRegistrationDataset(
        val_pairs,
        input_size=args.input_size,
        max_translation=args.max_translation,
        max_displacement=args.max_displacement,
        grid_size=args.grid_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, len(train_pairs), len(val_pairs)


def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    total_flow_mae = 0.0
    total_similarity = 0.0
    total_samples = 0
    supervised_loss = nn.SmoothL1Loss()

    for batch in loader:
        sar = batch["sar"].to(device)
        optical = batch["optical"].to(device)
        target_flow = batch["target_flow"].to(device)
        reference_sar = batch["reference_sar"].to(device)

        with torch.set_grad_enabled(training):
            predicted_flow = model(sar, optical)
            warped_sar = warp_tensor(sar, predicted_flow)

            loss_supervised = supervised_loss(predicted_flow, target_flow)
            loss_similarity = 0.5 * local_ncc_loss(warped_sar, optical) + 0.5 * local_ncc_loss(warped_sar, reference_sar)
            loss_smooth = smoothness_loss(predicted_flow)
            loss = (
                args.supervision_weight * loss_supervised
                + args.similarity_weight * loss_similarity
                + args.smoothness_weight * loss_smooth
            )

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        batch_size = sar.size(0)
        flow_mae = (predicted_flow.detach() - target_flow).abs().mean().item()
        total_loss += loss.item() * batch_size
        total_flow_mae += flow_mae * batch_size
        total_similarity += loss_similarity.detach().item() * batch_size
        total_samples += batch_size

    return {
        "loss": total_loss / total_samples,
        "flow_mae": total_flow_mae / total_samples,
        "similarity": total_similarity / total_samples,
    }


def save_checkpoint(
    output_dir: Path,
    name: str,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    args: argparse.Namespace,
) -> None:
    checkpoint = {
        "epoch": epoch,
        "best_metric": best_metric,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "args": vars(args),
    }
    torch.save(checkpoint, output_dir / name)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, train_count, val_count = build_dataloaders(args)

    model = DenseFieldRegistrationCNN().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    best_flow_mae = math.inf
    history: list[dict[str, float]] = []
    print(
        json.dumps(
            {
                "device": str(device),
                "train_pairs": train_count,
                "val_pairs": val_count,
                "dataset_root": str(Path(args.dataset_root).resolve()),
            },
            ensure_ascii=False,
        )
    )

    for epoch in range(1, args.epochs + 1):
        train_metrics = compute_metrics(model, train_loader, device, args, optimizer=optimizer)
        val_metrics = compute_metrics(model, val_loader, device, args)
        scheduler.step()

        epoch_metrics = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train_loss": train_metrics["loss"],
            "train_flow_mae": train_metrics["flow_mae"],
            "train_similarity": train_metrics["similarity"],
            "val_loss": val_metrics["loss"],
            "val_flow_mae": val_metrics["flow_mae"],
            "val_similarity": val_metrics["similarity"],
        }
        history.append(epoch_metrics)
        print(json.dumps(epoch_metrics, ensure_ascii=False))

        save_checkpoint(output_dir, "last_dense_field.pth", model, optimizer, epoch, val_metrics["flow_mae"], args)
        if val_metrics["flow_mae"] < best_flow_mae:
            best_flow_mae = val_metrics["flow_mae"]
            save_checkpoint(output_dir, "dense_field_best.pth", model, optimizer, epoch, best_flow_mae, args)

    with (output_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    print(json.dumps({"best_val_flow_mae": best_flow_mae, "output_dir": str(output_dir)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
