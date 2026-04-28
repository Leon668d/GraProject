from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


CODEX_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_SWEEP_DIR = CODEX_ROOT / "workspace" / "diffusion_lightglue_param_sweep"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def parse_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def reason_key(row: dict[str, str]) -> str:
    if not parse_bool(row.get("worker_success")):
        return "worker_failed"
    reasons = [reason.strip() for reason in row.get("reliability_reasons", "").split(";") if reason.strip()]
    if not reasons:
        return "unreliable_unknown"
    if "inliers < 8" in reasons:
        return "few_inliers"
    if "inlier_ratio < 0.25" in reasons:
        return "low_inlier_ratio"
    if "rmse > 5px" in reasons:
        return "high_rmse"
    if "poor_spatial_coverage" in reasons:
        return "poor_spatial_coverage"
    if "bad_homography_shape" in reasons:
        return "bad_homography_shape"
    return "_".join(reasons).replace(" ", "_").replace("<", "lt").replace(">", "gt")


def load_image(path: Path, size: tuple[int, int]) -> Image.Image:
    if not path.exists():
        canvas = Image.new("RGB", size, (235, 235, 235))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 8), f"missing\n{path.name}", fill=(180, 40, 40))
        return canvas
    return Image.open(path).convert("RGB").resize(size, Image.BILINEAR)


def caption(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    x, y = xy
    draw.rectangle((x - 3, y - 2, x + 520, y + 42), fill=(255, 255, 255))
    draw.text((x, y), text, fill=(10, 10, 10), font=font)


def build_tile(row: dict[str, str], thumb: int, font: ImageFont.ImageFont) -> Image.Image:
    result_dir = Path(row["result_dir"])
    images = [
        ("SAR", result_dir / "sar_condition.png"),
        ("FakeOpt", result_dir / "fake_optical.png"),
        ("RealOpt", result_dir / "real_optical_resized.png"),
        ("Matches", result_dir / "lightglue_matches.png"),
        ("SAR warp", result_dir / "sar_registered.png"),
    ]
    tile_w = thumb * len(images)
    header_h = 70
    label_h = 22
    tile = Image.new("RGB", (tile_w, header_h + thumb + label_h), (250, 250, 250))
    draw = ImageDraw.Draw(tile)

    title = (
        f"{row.get('phase', '')} | {row.get('matcher_strategy') or row.get('extractor', 'superpoint')} | "
        f"selected={row.get('selected_extractor', '')} | pre={row.get('match_preprocess', 'rgb')} | "
        f"steps={row.get('steps', '')}, k={row.get('max_keypoints', '')} | "
        f"{row.get('season', '')}/{row.get('stem', '')}"
    )
    metrics = (
        f"reason={reason_key(row)} | matches={row.get('match_count', '')}, "
        f"inliers={row.get('inliers', '')}, ratio={row.get('inlier_ratio', '')}, rmse={row.get('rmse', '')}"
    )
    draw.text((8, 8), title[:130], fill=(0, 0, 0), font=font)
    draw.text((8, 34), metrics[:130], fill=(120, 30, 30), font=font)

    for idx, (label, path) in enumerate(images):
        image = load_image(path, (thumb, thumb))
        x = idx * thumb
        tile.paste(image, (x, header_h))
        draw.rectangle((x, header_h, x + thumb - 1, header_h + thumb - 1), outline=(220, 220, 220))
        draw.text((x + 6, header_h + thumb + 3), label, fill=(0, 0, 0), font=font)
    return tile


def make_sheet(rows: list[dict[str, str]], output_path: Path, thumb: int, columns: int, title: str) -> None:
    if not rows:
        return
    font = ImageFont.load_default()
    tiles = [build_tile(row, thumb, font) for row in rows]
    tile_w, tile_h = tiles[0].size
    title_h = 36
    rows_count = (len(tiles) + columns - 1) // columns
    sheet = Image.new("RGB", (tile_w * columns, title_h + tile_h * rows_count), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    draw.text((10, 10), title, fill=(0, 0, 0), font=font)
    for idx, tile in enumerate(tiles):
        x = (idx % columns) * tile_w
        y = title_h + (idx // columns) * tile_h
        sheet.paste(tile, (x, y))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create contact sheets for failed Diffusion + LightGlue sweep samples.")
    parser.add_argument("--all-runs-csv", type=Path, default=DEFAULT_SWEEP_DIR / "all_runs.csv")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SWEEP_DIR / "failure_contact_sheets")
    parser.add_argument("--phase", default="verify", help="Use one phase, or 'all'.")
    parser.add_argument("--extractor", default="", help="Optional extractor filter.")
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--per-reason-limit", type=int, default=30)
    parser.add_argument("--thumb", type=int, default=160)
    parser.add_argument("--columns", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = read_csv(args.all_runs_csv)
    failures = []
    for row in rows:
        if args.phase != "all" and row.get("phase") != args.phase:
            continue
        if args.extractor and row.get("extractor", "superpoint") != args.extractor:
            continue
        if not parse_bool(row.get("registration_reliable")):
            failures.append(row)

    failures.sort(
        key=lambda row: (
            row.get("extractor", ""),
            reason_key(row),
            -float(row.get("match_count") or 0),
            row.get("stem", ""),
        )
    )

    outputs: dict[str, str] = {}
    make_sheet(
        failures[: args.limit],
        args.output_dir / "failures_all.png",
        args.thumb,
        args.columns,
        f"All failures, n={len(failures)}",
    )
    if failures:
        outputs["all"] = str(args.output_dir / "failures_all.png")

    by_reason: dict[str, list[dict[str, str]]] = defaultdict(list)
    by_extractor: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in failures:
        by_reason[reason_key(row)].append(row)
        by_extractor[row.get("extractor", "superpoint")].append(row)

    for reason, reason_rows in sorted(by_reason.items()):
        path = args.output_dir / f"failures_reason_{reason}.png"
        make_sheet(
            reason_rows[: args.per_reason_limit],
            path,
            args.thumb,
            args.columns,
            f"Failure reason: {reason}, n={len(reason_rows)}",
        )
        outputs[f"reason:{reason}"] = str(path)

    for extractor, extractor_rows in sorted(by_extractor.items()):
        path = args.output_dir / f"failures_extractor_{extractor}.png"
        make_sheet(
            extractor_rows[: args.per_reason_limit],
            path,
            args.thumb,
            args.columns,
            f"Extractor: {extractor}, failures={len(extractor_rows)}",
        )
        outputs[f"extractor:{extractor}"] = str(path)

    rescued = [row for row in rows if parse_bool(row.get("cascade_rescued"))]
    rescued.sort(key=lambda row: (row.get("matcher_strategy", ""), row.get("stem", "")))
    rescued_path = args.output_dir / "cascade_rescued.png"
    make_sheet(
        rescued[: args.limit],
        rescued_path,
        args.thumb,
        args.columns,
        f"Cascade rescued samples, n={len(rescued)}",
    )
    if rescued:
        outputs["cascade_rescued"] = str(rescued_path)

    summary = {
        "success": True,
        "all_runs_csv": str(args.all_runs_csv),
        "failure_count": len(failures),
        "by_reason": dict(Counter(reason_key(row) for row in failures)),
        "by_extractor": dict(Counter(row.get("extractor", "superpoint") for row in failures)),
        "cascade_rescued_count": len(rescued),
        "outputs": outputs,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "failure_contact_sheet_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
