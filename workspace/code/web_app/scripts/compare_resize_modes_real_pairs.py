from __future__ import annotations

import argparse
import csv
import json
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


@dataclass
class RealPairCase:
    case_id: str
    stem: str
    season: str
    tier: str
    variant: str
    sar_path: Path
    optical_path: Path
    opt_edge_mean: float
    opt_gray_std: float


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resize_image(source: Path, target: Path, size: tuple[int, int]) -> Path:
    image = Image.open(source).convert("RGB")
    image.resize(size, Image.BILINEAR).save(target)
    return target


def structure_score(row: dict) -> float:
    return float(row["opt_edge_mean"]) + 0.35 * float(row["opt_gray_std"])


def pick_tier_rows(rows: list[dict], picks_per_season: int) -> list[dict]:
    by_season: dict[str, list[dict]] = {}
    for row in rows:
        sar_path = Path(row["sar_resolved_path"] or row["sar_path"])
        opt_path = Path(row["opt_resolved_path"] or row["opt_path"])
        if sar_path.exists() and opt_path.exists():
            by_season.setdefault(row["season"], []).append(row)

    selected: list[dict] = []
    for season in sorted(by_season):
        season_rows = sorted(by_season[season], key=structure_score)
        if len(season_rows) < picks_per_season:
            raise RuntimeError(f"{season} only has {len(season_rows)} valid rows, need {picks_per_season}")
        for tier_idx in range(picks_per_season):
            pos = round((len(season_rows) - 1) * tier_idx / max(1, picks_per_season - 1))
            row = season_rows[pos]
            row = dict(row)
            row["tier"] = f"tier_{tier_idx + 1}_of_{picks_per_season}"
            selected.append(row)
    return selected


def build_cases(rows: list[dict], generated_input_dir: Path) -> list[RealPairCase]:
    cases: list[RealPairCase] = []
    aspect_sizes = [
        ((256, 192), (208, 256)),
        ((192, 256), (256, 208)),
        ((256, 176), (224, 256)),
    ]
    for idx, row in enumerate(rows):
        stem = row["stem"]
        season = row["season"]
        tier = row["tier"]
        sar_path = Path(row["sar_resolved_path"] or row["sar_path"])
        optical_path = Path(row["opt_resolved_path"] or row["opt_path"])
        base_case_id = f"{season}_{tier}_{stem}"
        cases.append(
            RealPairCase(
                case_id=base_case_id,
                stem=stem,
                season=season,
                tier=tier,
                variant="square",
                sar_path=sar_path,
                optical_path=optical_path,
                opt_edge_mean=float(row["opt_edge_mean"]),
                opt_gray_std=float(row["opt_gray_std"]),
            )
        )

        sar_size, optical_size = aspect_sizes[idx % len(aspect_sizes)]
        case_dir = ensure_dir(generated_input_dir / base_case_id)
        cases.append(
            RealPairCase(
                case_id=f"{base_case_id}_aspect",
                stem=stem,
                season=season,
                tier=tier,
                variant="aspect_stress",
                sar_path=resize_image(sar_path, case_dir / f"sar_{sar_size[0]}x{sar_size[1]}.png", sar_size),
                optical_path=resize_image(
                    optical_path,
                    case_dir / f"optical_{optical_size[0]}x{optical_size[1]}.png",
                    optical_size,
                ),
                opt_edge_mean=float(row["opt_edge_mean"]),
                opt_gray_std=float(row["opt_gray_std"]),
            )
        )
    return cases


def run_worker(case: RealPairCase, resize_mode: str, args, worker_script: Path) -> dict:
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
            "stem": case.stem,
            "season": case.season,
            "tier": case.tier,
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
        "stem": case.stem,
        "season": case.season,
        "tier": case.tier,
        "variant": case.variant,
        "resize_mode": resize_mode,
        "ok": True,
        "elapsed_s": elapsed_s,
        "opt_edge_mean": case.opt_edge_mean,
        "opt_gray_std": case.opt_gray_std,
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


def summarize(rows: list[dict]) -> list[dict]:
    groups: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        if not row.get("ok"):
            continue
        groups.setdefault((row["variant"], row["resize_mode"]), []).append(row)

    summary: list[dict] = []
    for (variant, resize_mode), group_rows in sorted(groups.items()):
        success_count = sum(1 for row in group_rows if row.get("registration_reliable"))
        total = len(group_rows)
        inliers = [float(row["inliers"]) for row in group_rows if row.get("inliers") is not None]
        ratios = [float(row["inlier_ratio"]) for row in group_rows if row.get("inlier_ratio") is not None]
        rmses = [float(row["rmse"]) for row in group_rows if row.get("rmse") is not None]
        summary.append(
            {
                "variant": variant,
                "resize_mode": resize_mode,
                "success_count": success_count,
                "total": total,
                "success_rate": round(success_count / total, 4) if total else 0.0,
                "avg_inliers": round(sum(inliers) / len(inliers), 2) if inliers else None,
                "avg_inlier_ratio": round(sum(ratios) / len(ratios), 4) if ratios else None,
                "avg_rmse": round(sum(rmses) / len(rmses), 4) if rmses else None,
            }
        )
    return summary


def build_markdown(rows: list[dict], summary_rows: list[dict], output_path: Path, selected_rows: list[dict]) -> None:
    headers = [
        "case_id",
        "season",
        "tier",
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
    summary_headers = ["variant", "resize_mode", "success_count", "total", "success_rate", "avg_inliers", "avg_inlier_ratio", "avg_rmse"]
    lines = [
        "# Real Non-Square Stretch vs Letterbox Comparison",
        "",
        "## Sample Selection",
        "",
        f"- Total selected real pairs: {len(selected_rows)}",
        f"- Seasons: {', '.join(sorted({row['season'] for row in selected_rows}))}",
        f"- Tiers per season: {len({row['tier'] for row in selected_rows if row['season'] == selected_rows[0]['season']}) if selected_rows else 0}",
        "",
        "## Aggregate Summary",
        "",
        "| " + " | ".join(summary_headers) + " |",
        "| " + " | ".join(["---"] * len(summary_headers)) + " |",
    ]
    for row in summary_rows:
        lines.append("| " + " | ".join(metric_value(row, header) for header in summary_headers) + " |")
    lines.extend(
        [
            "",
            "## Per-Case Results",
            "",
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
    )
    for row in rows:
        lines.append("| " + " | ".join(metric_value(row, header) for header in headers) + " |")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_contact_sheet(rows: list[dict], output_path: Path) -> None:
    ok_rows = [row for row in rows if row.get("ok")]
    thumb_w = 220
    thumb_h = 172
    row_h = thumb_h + 56
    sheet = Image.new("RGB", (thumb_w * 3, row_h * len(ok_rows)), "white")
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(ok_rows):
        y = idx * row_h
        title = (
            f"{row['season']} {row['tier']} {row['variant']} {row['resize_mode']} "
            f"| rel={row.get('registration_reliable')} inl={row.get('inliers')} rmse={row.get('rmse')}"
        )
        draw.text((8, y + 4), title, fill=(20, 20, 20))
        for col_idx, key in enumerate(("registered_preview", "checkerboard", "matches")):
            x = col_idx * thumb_w
            image_path = row.get(key)
            if image_path and Path(image_path).exists():
                image = Image.open(image_path).convert("RGB")
                image.thumbnail((thumb_w, thumb_h - 26), Image.LANCZOS)
                canvas = Image.new("RGB", (thumb_w, thumb_h - 26), (240, 244, 248))
                canvas.paste(image, ((thumb_w - image.width) // 2, ((thumb_h - 26) - image.height) // 2))
                sheet.paste(canvas, (x, y + 28))
            draw.text((x + 8, y + thumb_h + 30), key, fill=(70, 70, 70))
    sheet.save(output_path, quality=92)


def parse_args() -> argparse.Namespace:
    web_app_dir = Path(__file__).resolve().parents[1]
    workspace_root = web_app_dir.parents[1]
    parser = argparse.ArgumentParser(description="Compare stretch vs letterbox on real sampled pairs from filtered_pairs.csv.")
    parser.add_argument("--web-app-dir", type=Path, default=web_app_dir)
    parser.add_argument("--pairs-csv", type=Path, default=workspace_root / "diffusion_lightglue_param_sweep" / "filtered_pairs.csv")
    parser.add_argument("--output-dir", type=Path, default=workspace_root / "letterbox_real_pairs_20260428")
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
    parser.add_argument("--picks-per-season", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    worker_script = args.web_app_dir / "scripts" / "diffusion_lightglue_worker.py"
    selected_rows = pick_tier_rows(
        list(csv.DictReader(args.pairs_csv.open(encoding="utf-8-sig"))),
        picks_per_season=args.picks_per_season,
    )
    generated_input_dir = ensure_dir(args.output_dir / "generated_inputs")
    cases = build_cases(selected_rows, generated_input_dir)

    rows: list[dict] = []
    for case in cases:
        for resize_mode in ("stretch", "letterbox"):
            row = run_worker(case, resize_mode, args, worker_script)
            rows.append(row)
            print(
                f"{case.case_id} {resize_mode}: ok={row.get('ok')} "
                f"reliable={row.get('registration_reliable')} inliers={row.get('inliers')} rmse={row.get('rmse')}"
            )

    summary_rows = summarize(rows)
    csv_path = args.output_dir / "real_pairs_stretch_vs_letterbox_results.csv"
    md_path = args.output_dir / "real_pairs_stretch_vs_letterbox_summary.md"
    sheet_path = args.output_dir / "real_pairs_stretch_vs_letterbox_contact_sheet.jpg"
    write_csv(rows, csv_path)
    build_markdown(rows, summary_rows, md_path, selected_rows)
    build_contact_sheet(rows, sheet_path)

    print(f"CSV {csv_path}")
    print(f"MD {md_path}")
    print(f"SHEET {sheet_path}")


if __name__ == "__main__":
    main()
