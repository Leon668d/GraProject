from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageStat


WEB_APP_DIR = Path(__file__).resolve().parents[1]
CODEX_ROOT = WEB_APP_DIR.parents[2]
DEFAULT_PAIRS_CSV = CODEX_ROOT / "workspace" / "acd_sen12_csv" / "val_pairs.csv"
DEFAULT_OUTPUT_DIR = CODEX_ROOT / "workspace" / "diffusion_lightglue_param_sweep"


def resolve_data_path(raw_path: str) -> Path:
    """Resolve Windows and WSL-style paths without mutating the source CSV."""
    path = Path(raw_path)
    if path.exists():
        return path

    text = raw_path.strip()
    mnt_match = re.match(r"^/mnt/([a-zA-Z])/(.*)$", text)
    if os.name == "nt" and mnt_match:
        candidate = Path(f"{mnt_match.group(1).upper()}:/{mnt_match.group(2)}")
        if candidate.exists():
            return candidate

    win_match = re.match(r"^([a-zA-Z]):[\\/](.*)$", text)
    if os.name != "nt" and win_match:
        candidate = Path(f"/mnt/{win_match.group(1).lower()}/{win_match.group(2).replace(chr(92), '/')}")
        if candidate.exists():
            return candidate

    return path


def read_pairs(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {"sar_path", "opt_path"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    return rows


def gray_texture_metrics(image: Image.Image, metric_size: int) -> dict[str, float]:
    gray = image.convert("L").resize((metric_size, metric_size), Image.BILINEAR)
    edge = gray.filter(ImageFilter.FIND_EDGES)
    return {
        "edge_mean": float(ImageStat.Stat(edge).mean[0]),
        "gray_std": float(ImageStat.Stat(gray).stddev[0]),
    }


def water_proxy(image: Image.Image, sample_size: int) -> float:
    rgb = image.convert("RGB").resize((sample_size, sample_size), Image.BILINEAR)
    data = np.asarray(rgb, dtype=np.int16)
    red = data[..., 0]
    green = data[..., 1]
    blue = data[..., 2]

    # Conservative open-water proxy for Sentinel-2 RGB previews. It catches large
    # blue/green smooth water regions without trying to be a full water classifier.
    water_like = (blue > red + 8) & (green > red + 4) & (blue > 70)
    return float(water_like.mean())


def compute_row_metrics(row: dict[str, str], metric_size: int, water_sample_size: int) -> dict[str, Any]:
    sar_path = resolve_data_path(row["sar_path"])
    opt_path = resolve_data_path(row["opt_path"])
    metrics: dict[str, Any] = {
        "sar_resolved_path": str(sar_path),
        "opt_resolved_path": str(opt_path),
    }

    if not sar_path.exists():
        metrics["missing_sar"] = True
    if not opt_path.exists():
        metrics["missing_optical"] = True
    if metrics.get("missing_sar") or metrics.get("missing_optical"):
        return metrics

    with Image.open(opt_path) as optical_image:
        optical_texture = gray_texture_metrics(optical_image, metric_size)
        metrics["opt_edge_mean"] = optical_texture["edge_mean"]
        metrics["opt_gray_std"] = optical_texture["gray_std"]
        metrics["opt_water_ratio"] = water_proxy(optical_image, water_sample_size)

    with Image.open(sar_path) as sar_image:
        sar_texture = gray_texture_metrics(sar_image, metric_size)
        metrics["sar_edge_mean"] = sar_texture["edge_mean"]
        metrics["sar_gray_std"] = sar_texture["gray_std"]

    return metrics


def reject_reasons(metrics: dict[str, Any], args: argparse.Namespace) -> list[str]:
    reasons: list[str] = []
    if metrics.get("missing_sar"):
        reasons.append("missing_sar")
    if metrics.get("missing_optical"):
        reasons.append("missing_optical")
    if reasons:
        return reasons

    if metrics["opt_edge_mean"] < args.min_edge or metrics["opt_gray_std"] < args.min_std:
        reasons.append("low_texture")
    if metrics["opt_water_ratio"] > args.max_water:
        reasons.append("water_like")
    if metrics["sar_edge_mean"] < args.min_sar_edge or metrics["sar_gray_std"] < args.min_sar_std:
        reasons.append("sar_low_texture")
    return reasons


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_summary(
    *,
    input_count: int,
    kept_rows: list[dict[str, Any]],
    rejected_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    filtered_path: Path,
    rejected_path: Path,
) -> dict[str, Any]:
    by_season: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "kept": 0, "rejected": 0})
    reason_counter: Counter[str] = Counter()

    for row in kept_rows:
        season = row.get("season") or "unknown"
        by_season[season]["kept"] += 1
    for row in rejected_rows:
        season = row.get("season") or "unknown"
        by_season[season]["rejected"] += 1
        for reason in str(row.get("reject_reasons", "")).split(";"):
            if reason:
                reason_counter[reason] += 1
    for season, counts in by_season.items():
        counts["input"] = counts["kept"] + counts["rejected"]

    return {
        "input_count": input_count,
        "kept_count": len(kept_rows),
        "rejected_count": len(rejected_rows),
        "kept_ratio": round(len(kept_rows) / max(input_count, 1), 6),
        "thresholds": {
            "min_edge": args.min_edge,
            "min_std": args.min_std,
            "max_water": args.max_water,
            "min_sar_edge": args.min_sar_edge,
            "min_sar_std": args.min_sar_std,
            "metric_size": args.metric_size,
            "water_sample_size": args.water_sample_size,
        },
        "by_season": dict(sorted(by_season.items())),
        "rejection_reasons": dict(sorted(reason_counter.items())),
        "outputs": {
            "filtered_pairs_csv": str(filtered_path),
            "rejected_pairs_csv": str(rejected_path),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter SAR/optical registration pairs before Diffusion + LightGlue parameter search."
    )
    parser.add_argument("--pairs-csv", type=Path, default=DEFAULT_PAIRS_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--min-edge", type=float, default=14.0)
    parser.add_argument("--min-std", type=float, default=18.0)
    parser.add_argument("--max-water", type=float, default=0.35)
    parser.add_argument("--min-sar-edge", type=float, default=5.0)
    parser.add_argument("--min-sar-std", type=float, default=4.0)
    parser.add_argument("--metric-size", type=int, default=256)
    parser.add_argument("--water-sample-size", type=int, default=48)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_pairs(args.pairs_csv)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metric_columns = [
        "quality_pass",
        "reject_reasons",
        "opt_edge_mean",
        "opt_gray_std",
        "opt_water_ratio",
        "sar_edge_mean",
        "sar_gray_std",
        "sar_resolved_path",
        "opt_resolved_path",
    ]
    original_fields = list(rows[0].keys())
    fieldnames = original_fields + [column for column in metric_columns if column not in original_fields]

    kept_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    for row in rows:
        metrics = compute_row_metrics(row, args.metric_size, args.water_sample_size)
        reasons = reject_reasons(metrics, args)
        output_row: dict[str, Any] = {**row, **metrics}
        output_row["quality_pass"] = not reasons
        output_row["reject_reasons"] = ";".join(reasons)
        for key in ("opt_edge_mean", "opt_gray_std", "opt_water_ratio", "sar_edge_mean", "sar_gray_std"):
            if key in output_row and isinstance(output_row[key], float):
                output_row[key] = round(output_row[key], 6)
        if reasons:
            rejected_rows.append(output_row)
        else:
            kept_rows.append(output_row)

    filtered_path = args.output_dir / "filtered_pairs.csv"
    rejected_path = args.output_dir / "rejected_pairs.csv"
    summary_path = args.output_dir / "filter_summary.json"
    write_csv(filtered_path, kept_rows, fieldnames)
    write_csv(rejected_path, rejected_rows, fieldnames)

    summary = build_summary(
        input_count=len(rows),
        kept_rows=kept_rows,
        rejected_rows=rejected_rows,
        args=args,
        filtered_path=filtered_path,
        rejected_path=rejected_path,
    )
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"success": True, **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
