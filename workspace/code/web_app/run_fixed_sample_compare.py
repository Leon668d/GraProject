from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from app import register_cnn


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-sample Web-style model comparison.")
    parser.add_argument("--model-path", required=True, help="Target checkpoint to evaluate.")
    parser.add_argument("--output-dir", required=True, help="Directory where comparison artifacts are written.")
    parser.add_argument("--dataset-root", required=True, help="Dataset root containing SAR/optical folders or MultiRes subsets.")
    parser.add_argument("--sample-ids", required=True, help="Comma-separated sample ids, e.g. 1,10,12,77,100.")
    parser.add_argument("--baseline-model", default="", help="Optional baseline checkpoint to compare against.")
    parser.add_argument("--resolution-subset", default="", choices=["", "High-Resolution", "Low-Resolution"])
    parser.add_argument("--sar-dir", default="", help="Optional override for the SAR image directory.")
    parser.add_argument("--optical-dir", default="", help="Optional override for the optical image directory.")
    return parser.parse_args()


def parse_sample_ids(sample_text: str) -> list[int]:
    sample_ids: list[int] = []
    for token in sample_text.split(","):
        stripped = token.strip()
        if stripped:
            sample_ids.append(int(stripped))
    return sample_ids


def resolve_data_dirs(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(args.dataset_root).resolve()
    if args.resolution_subset:
        subset_root = root / args.resolution_subset
        return subset_root / "R", subset_root / "L"

    sar_dir = Path(args.sar_dir).resolve() if args.sar_dir else root / "sar"
    optical_dir = Path(args.optical_dir).resolve() if args.optical_dir else root / "optical"
    return sar_dir, optical_dir


def pair_key(path: Path) -> str:
    stem = path.stem.lower()
    match = re.search(r"(\d+)$", stem)
    if match:
        return match.group(1)
    return stem


def build_pair_lookup(sar_dir: Path, optical_dir: Path) -> dict[str, tuple[Path, Path]]:
    lookup: dict[str, tuple[Path, Path]] = {}
    sar_images = {
        pair_key(path): path
        for path in sar_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    optical_images = {
        pair_key(path): path
        for path in optical_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }
    for key in sorted(set(sar_images) & set(optical_images)):
        lookup[key] = (sar_images[key], optical_images[key])
    return lookup


def run_model_on_samples(
    model_path: Path,
    label: str,
    sample_ids: list[int],
    pair_lookup: dict[str, tuple[Path, Path]],
    output_dir: Path,
) -> tuple[list[dict], list[float]]:
    sample_results: list[dict] = []
    difference_means: list[float] = []

    for sample_id in sample_ids:
        lookup_key = str(sample_id)
        if lookup_key not in pair_lookup:
            raise FileNotFoundError(f"Sample id {sample_id} not found in dataset.")

        sar_path, optical_path = pair_lookup[lookup_key]
        result_dir = output_dir / f"sample_{sample_id}" / label
        if result_dir.exists():
            shutil.rmtree(result_dir)
        result_dir.mkdir(parents=True, exist_ok=True)

        output = register_cnn(
            str(sar_path),
            str(optical_path),
            str(result_dir),
            model_path=model_path,
        )
        metrics = output.get("metrics", {})
        difference_mean = float(metrics.get("difference_mean", 0.0))
        difference_means.append(difference_mean)
        sample_results.append(
            {
                "sample_id": sample_id,
                "sar_path": str(sar_path),
                "optical_path": str(optical_path),
                "model_name": model_path.name,
                "prediction_type": output.get("prediction_type"),
                "metrics": metrics,
                "timings": output.get("timings", {}),
                "model_info": output.get("model_info", {}),
                "result_dir": str(result_dir),
            }
        )

    return sample_results, difference_means


def main() -> None:
    args = parse_args()
    sample_ids = parse_sample_ids(args.sample_ids)
    if not sample_ids:
        raise ValueError("At least one sample id is required.")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sar_dir, optical_dir = resolve_data_dirs(args)
    pair_lookup = build_pair_lookup(sar_dir, optical_dir)

    target_model = Path(args.model_path).resolve()
    target_results, target_difference_mean = run_model_on_samples(
        target_model,
        "target",
        sample_ids,
        pair_lookup,
        output_dir,
    )

    summary: dict[str, object] = {
        "sample_ids": sample_ids,
        "target_model_name": target_model.name,
        "target_results": target_results,
    }
    aggregate: dict[str, object] = {
        "sample_ids": sample_ids,
        "target_model_name": target_model.name,
        "target_difference_mean": target_difference_mean,
        "target_avg_difference_mean": round(sum(target_difference_mean) / len(target_difference_mean), 3),
    }

    if args.baseline_model:
        baseline_model = Path(args.baseline_model).resolve()
        baseline_results, baseline_difference_mean = run_model_on_samples(
            baseline_model,
            "baseline",
            sample_ids,
            pair_lookup,
            output_dir,
        )
        summary["baseline_model_name"] = baseline_model.name
        summary["baseline_results"] = baseline_results
        aggregate["baseline_model_name"] = baseline_model.name
        aggregate["baseline_difference_mean"] = baseline_difference_mean
        aggregate["baseline_avg_difference_mean"] = round(
            sum(baseline_difference_mean) / len(baseline_difference_mean),
            3,
        )

    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with (output_dir / "aggregate.json").open("w", encoding="utf-8") as file:
        json.dump(aggregate, file, ensure_ascii=False, indent=2)

    print(json.dumps(aggregate, ensure_ascii=False))


if __name__ == "__main__":
    main()
