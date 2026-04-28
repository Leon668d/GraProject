from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


RECOMMENDED_CASES = [
    ("success_summer", "showcase_summer_strong", "success"),
    ("success_winter", "showcase_winter_strong", "success"),
    ("success_rescue_spring", "cascade_rescue_spring", "success_rescue"),
    ("boundary_winter", "boundary_review_winter", "boundary"),
]


@dataclass
class InputCase:
    case_id: str
    sample_id: str
    group: str
    variant: str
    sar_path: Path
    optical_path: Path


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resize_image(source: Path, target: Path, size: tuple[int, int]) -> Path:
    image = Image.open(source).convert("RGB")
    image.resize(size, Image.BILINEAR).save(target)
    return target


def crop_image(source: Path, target: Path, box: tuple[int, int, int, int]) -> Path:
    image = Image.open(source).convert("RGB")
    image.crop(box).save(target)
    return target


def build_cases(demo_dir: Path, generated_input_dir: Path, include_crop_stress: bool) -> list[InputCase]:
    cases: list[InputCase] = []
    for case_id, sample_id, group in RECOMMENDED_CASES:
        sample_dir = demo_dir / sample_id
        cases.append(
            InputCase(
                case_id=case_id,
                sample_id=sample_id,
                group=group,
                variant="square",
                sar_path=sample_dir / "sar_condition.png",
                optical_path=sample_dir / "real_optical_resized.png",
            )
        )

    aspect_specs = [
        ("success_summer_aspect", "showcase_summer_strong", "success_aspect"),
        ("boundary_winter_aspect", "boundary_review_winter", "boundary_aspect"),
    ]
    for case_id, sample_id, group in aspect_specs:
        case_dir = ensure_dir(generated_input_dir / case_id)
        sample_dir = demo_dir / sample_id
        cases.append(
            InputCase(
                case_id=case_id,
                sample_id=sample_id,
                group=group,
                variant="aspect_stress",
                sar_path=resize_image(sample_dir / "sar_condition.png", case_dir / "sar_256x192.png", (256, 192)),
                optical_path=resize_image(sample_dir / "real_optical_resized.png", case_dir / "optical_208x256.png", (208, 256)),
            )
        )

    if include_crop_stress:
        crop_specs = [
            ("success_summer_crop", "showcase_summer_strong", "success_crop"),
            ("boundary_winter_crop", "boundary_review_winter", "boundary_crop"),
        ]
        for case_id, sample_id, group in crop_specs:
            case_dir = ensure_dir(generated_input_dir / case_id)
            sample_dir = demo_dir / sample_id
            cases.append(
                InputCase(
                    case_id=case_id,
                    sample_id=sample_id,
                    group=group,
                    variant="crop_stress",
                    sar_path=crop_image(sample_dir / "sar_condition.png", case_dir / "sar_256x192.png", (0, 32, 256, 224)),
                    optical_path=crop_image(sample_dir / "real_optical_resized.png", case_dir / "optical_208x256.png", (24, 0, 232, 256)),
                )
            )
    return cases


def run_worker(case: InputCase, resize_mode: str, args, worker_script: Path) -> dict:
    run_dir = ensure_dir(args.output_dir / case.case_id / resize_mode)
    command = [
        str(args.python_exe),
        str(worker_script),
        "--sar",
        str(case.sar_path),
        "--optical",
        str(case.optical_path),
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        str(run_dir),
        "--steps",
        str(args.steps),
        "--max-keypoints",
        str(args.max_keypoints),
        "--extractor",
        args.extractors[0],
        "--extractor-policy",
        args.extractor_policy,
        "--match-preprocess",
        args.match_preprocess,
        "--resize-mode",
        resize_mode,
        "--device",
        args.device,
        "--dtype",
        args.dtype,
        "--extractors",
        *args.extractors,
    ]
    started = time.perf_counter()
    process = subprocess.run(
        command,
        cwd=str(args.web_app_dir),
        capture_output=True,
        text=True,
        timeout=args.timeout_seconds,
        check=False,
    )
    elapsed_s = round(time.perf_counter() - started, 2)
    if process.returncode != 0:
        return {
            "case_id": case.case_id,
            "sample_id": case.sample_id,
            "group": case.group,
            "variant": case.variant,
            "resize_mode": resize_mode,
            "ok": False,
            "elapsed_s": elapsed_s,
            "error": (process.stderr.strip() or process.stdout.strip() or "worker_failed")[:500],
        }

    payload = json.loads((run_dir / "diffusion_lightglue_result.json").read_text(encoding="utf-8"))
    metrics = payload.get("metrics", {})
    artifacts = payload.get("artifacts", {})
    return {
        "case_id": case.case_id,
        "sample_id": case.sample_id,
        "group": case.group,
        "variant": case.variant,
        "resize_mode": resize_mode,
        "ok": True,
        "elapsed_s": elapsed_s,
        "selected_extractor": metrics.get("selected_extractor") or metrics.get("extractor"),
        "registration_reliable": metrics.get("registration_reliable"),
        "match_count": metrics.get("match_count"),
        "raw_match_count": metrics.get("raw_match_count"),
        "padding_filtered_count": metrics.get("padding_filtered_count"),
        "inliers": metrics.get("inliers"),
        "inlier_ratio": metrics.get("inlier_ratio"),
        "rmse": metrics.get("rmse"),
        "bad_homography_shape": metrics.get("bad_homography_shape"),
        "poor_spatial_coverage": metrics.get("poor_spatial_coverage"),
        "coordinate_mode": metrics.get("coordinate_mode"),
        "has_h_matrix_original": bool(metrics.get("h_matrix_original")),
        "difference_mean": metrics.get("difference_mean"),
        "total_ms": payload.get("timings", {}).get("total_ms"),
        "registered_preview": artifacts.get("registered_preview"),
        "checkerboard": artifacts.get("checkerboard"),
        "matches": artifacts.get("matches"),
        "error": "",
    }


def write_csv(rows: list[dict], output_path: Path) -> None:
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def metric_value(row: dict, key: str) -> str:
    value = row.get(key)
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def build_markdown(rows: list[dict], output_path: Path) -> None:
    headers = [
        "case_id",
        "group",
        "variant",
        "resize_mode",
        "registration_reliable",
        "selected_extractor",
        "match_count",
        "padding_filtered_count",
        "inliers",
        "inlier_ratio",
        "rmse",
        "bad_homography_shape",
        "coordinate_mode",
    ]
    lines = [
        "# Stretch vs Letterbox Comparison",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(metric_value(row, header) for header in headers) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_contact_sheet(rows: list[dict], output_path: Path) -> None:
    ok_rows = [row for row in rows if row.get("ok")]
    thumb_w = 256
    thumb_h = 192
    row_h = thumb_h + 58
    sheet = Image.new("RGB", (thumb_w * 3, row_h * len(ok_rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(ok_rows):
        y = idx * row_h
        title = (
            f"{row['case_id']} | {row['resize_mode']} | rel={row.get('registration_reliable')} | "
            f"inl={row.get('inliers')} | rmse={row.get('rmse')} | filt={row.get('padding_filtered_count')}"
        )
        draw.text((8, y + 4), title, fill=(20, 20, 20))
        for col_idx, key in enumerate(("registered_preview", "checkerboard", "matches")):
            x = col_idx * thumb_w
            image_path = row.get(key)
            if image_path and Path(image_path).exists():
                image = Image.open(image_path).convert("RGB")
                image.thumbnail((thumb_w, thumb_h - 28), Image.LANCZOS)
                canvas = Image.new("RGB", (thumb_w, thumb_h - 28), (240, 244, 248))
                canvas.paste(image, ((thumb_w - image.width) // 2, ((thumb_h - 28) - image.height) // 2))
                sheet.paste(canvas, (x, y + 30))
            draw.text((x + 8, y + thumb_h + 34), key, fill=(70, 70, 70))
    sheet.save(output_path, quality=92)


def parse_args() -> argparse.Namespace:
    web_app_dir = Path(__file__).resolve().parents[1]
    workspace_root = web_app_dir.parents[1]
    parser = argparse.ArgumentParser(description="Compare stretch vs letterbox resize modes on built-in demo samples.")
    parser.add_argument("--web-app-dir", type=Path, default=web_app_dir)
    parser.add_argument("--demo-dir", type=Path, default=web_app_dir / "static" / "demo_samples")
    parser.add_argument("--output-dir", type=Path, default=workspace_root / "letterbox_comparison_20260428")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--python-exe", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--max-keypoints", type=int, default=2048)
    parser.add_argument("--extractor-policy", default="cascade", choices=["single", "cascade"])
    parser.add_argument("--extractors", nargs="+", default=["superpoint", "aliked"])
    parser.add_argument("--match-preprocess", default="rgb", choices=["rgb", "structure"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--skip-crop-stress", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker_script = args.web_app_dir / "scripts" / "diffusion_lightglue_worker.py"
    generated_input_dir = ensure_dir(args.output_dir / "generated_inputs")
    cases = build_cases(args.demo_dir, generated_input_dir, include_crop_stress=not args.skip_crop_stress)

    rows: list[dict] = []
    for case in cases:
        for resize_mode in ("stretch", "letterbox"):
            row = run_worker(case, resize_mode, args, worker_script)
            rows.append(row)
            print(
                f"{case.case_id} {resize_mode}: ok={row.get('ok')} "
                f"reliable={row.get('registration_reliable')} inliers={row.get('inliers')} rmse={row.get('rmse')}"
            )

    csv_path = args.output_dir / "stretch_vs_letterbox_results.csv"
    md_path = args.output_dir / "stretch_vs_letterbox_summary.md"
    sheet_path = args.output_dir / "stretch_vs_letterbox_contact_sheet.jpg"
    write_csv(rows, csv_path)
    build_markdown(rows, md_path)
    build_contact_sheet(rows, sheet_path)

    print(f"CSV {csv_path}")
    print(f"MD {md_path}")
    print(f"SHEET {sheet_path}")


if __name__ == "__main__":
    main()
