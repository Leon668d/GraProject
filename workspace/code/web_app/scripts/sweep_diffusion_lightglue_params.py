from __future__ import annotations

import argparse
import csv
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path
from statistics import median
from typing import Any


WEB_APP_DIR = Path(__file__).resolve().parents[1]
CODEX_ROOT = WEB_APP_DIR.parents[2]
SCRIPTS_DIR = WEB_APP_DIR / "scripts"
WORKER_SCRIPT = SCRIPTS_DIR / "diffusion_lightglue_worker.py"
FILTER_SCRIPT = SCRIPTS_DIR / "filter_registration_pairs.py"
CONTACT_SHEET_SCRIPT = SCRIPTS_DIR / "make_failure_contact_sheet.py"
DEFAULT_OUTPUT_DIR = CODEX_ROOT / "workspace" / "diffusion_lightglue_param_sweep"
DEFAULT_RAW_PAIRS_CSV = CODEX_ROOT / "workspace" / "acd_sen12_csv" / "val_pairs.csv"
DEFAULT_FILTERED_PAIRS_CSV = DEFAULT_OUTPUT_DIR / "filtered_pairs.csv"
DEFAULT_CHECKPOINT = Path(r"E:\checkPoint\checkPoint\checkpoint-25001-lcm-adv\model.safetensors")
DEFAULT_RUNTIME_PYTHON = Path(r"E:\Anaconda3\envs\sar_diff\python.exe")
TORCH_CACHE_DIR = WEB_APP_DIR / "runtime_cache" / "torch"


def path_default_runtime() -> str:
    return str(DEFAULT_RUNTIME_PYTHON if DEFAULT_RUNTIME_PYTHON.exists() else Path(sys.executable))


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


def balanced_sample(rows: list[dict[str, str]], limit: int, seed: int) -> list[dict[str, str]]:
    if limit <= 0:
        return []
    if limit >= len(rows):
        return list(rows)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row.get("season") or "unknown"].append(row)

    rng = random.Random(seed)
    for group_rows in groups.values():
        rng.shuffle(group_rows)

    sampled: list[dict[str, str]] = []
    seasons = sorted(groups)
    while len(sampled) < limit:
        progressed = False
        for season in seasons:
            if groups[season]:
                sampled.append(groups[season].pop())
                progressed = True
                if len(sampled) >= limit:
                    break
        if not progressed:
            break
    return sampled


def run_filter_if_needed(args: argparse.Namespace) -> None:
    default_filtered_for_output = args.output_dir / "filtered_pairs.csv"
    should_filter = not args.skip_filter and (
        args.refresh_filter or not args.pairs_csv.exists() or args.pairs_csv.resolve() == default_filtered_for_output.resolve()
    )
    if not should_filter:
        return

    cmd = [
        args.runtime_python,
        str(FILTER_SCRIPT),
        "--pairs-csv",
        str(args.raw_pairs_csv),
        "--output-dir",
        str(args.output_dir),
        "--min-edge",
        str(args.min_edge),
        "--min-std",
        str(args.min_std),
        "--max-water",
        str(args.max_water),
        "--min-sar-edge",
        str(args.min_sar_edge),
        "--min-sar-std",
        str(args.min_sar_std),
    ]
    completed = subprocess.run(cmd, cwd=str(WEB_APP_DIR), text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "Filtering failed before parameter sweep.\n"
            f"command: {' '.join(cmd)}\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )


def safe_name(value: str) -> str:
    keep = []
    for char in value:
        keep.append(char if char.isalnum() or char in ("-", "_", ".") else "_")
    return "".join(keep)[:140] or "sample"


def load_worker_json(result_dir: Path) -> dict[str, Any]:
    result_json = result_dir / "diffusion_lightglue_result.json"
    if not result_json.exists():
        raise FileNotFoundError(f"Worker result JSON not found: {result_json}")
    return json.loads(result_json.read_text(encoding="utf-8"))


def run_worker(
    *,
    args: argparse.Namespace,
    row: dict[str, str],
    phase: str,
    sample_index: int,
    steps: int,
    keypoints: int,
    strategy: dict[str, Any],
) -> dict[str, Any]:
    stem = row.get("stem") or Path(row["sar_path"]).stem
    strategy_name = strategy["name"]
    result_dir = (
        args.output_dir
        / "runs"
        / phase
        / f"strategy_{safe_name(strategy_name)}_steps_{steps}_keypoints_{keypoints}"
        / f"{sample_index:04d}_{safe_name(stem)}"
    )
    result_json = result_dir / "diffusion_lightglue_result.json"
    started = time.perf_counter()

    if args.skip_existing and result_json.exists():
        try:
            payload = load_worker_json(result_dir)
            return flatten_result(
                payload=payload,
                row=row,
                phase=phase,
                sample_index=sample_index,
                steps=steps,
                keypoints=keypoints,
                strategy=strategy,
                result_dir=result_dir,
                reused=True,
            )
        except Exception:
            pass

    result_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        args.runtime_python,
        str(WORKER_SCRIPT),
        "--sar",
        row["sar_path"],
        "--optical",
        row["opt_path"],
        "--checkpoint",
        str(args.checkpoint),
        "--output-dir",
        str(result_dir),
        "--steps",
        str(steps),
        "--seed",
        str(args.seed),
        "--max-keypoints",
        str(keypoints),
        "--extractor",
        strategy["extractors"][0],
        "--extractor-policy",
        strategy["policy"],
        "--match-preprocess",
        strategy["match_preprocess"],
        "--ransac-threshold",
        str(args.ransac_threshold),
        "--device",
        args.device,
        "--dtype",
        args.dtype,
    ]
    cmd.append("--extractors")
    cmd.extend(strategy["extractors"])
    if args.sar_log:
        cmd.append("--sar-log")
    if args.cascade_stop_on_reliable:
        cmd.append("--cascade-stop-on-reliable")

    env = os.environ.copy()
    env.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))
    completed = subprocess.run(
        cmd,
        cwd=str(WEB_APP_DIR),
        env=env,
        text=True,
        capture_output=True,
        timeout=args.timeout_seconds,
        check=False,
    )
    if completed.returncode != 0:
        return {
            "phase": phase,
            "sample_index": sample_index,
            "season": row.get("season", ""),
            "stem": stem,
            "sar_path": row["sar_path"],
            "opt_path": row["opt_path"],
            "steps": steps,
            "max_keypoints": keypoints,
            "extractor": strategy["label"],
            "extractor_policy": strategy["policy"],
            "extractors": "+".join(strategy["extractors"]),
            "match_preprocess": strategy["match_preprocess"],
            "matcher_strategy": strategy_name,
            "selected_extractor": "",
            "worker_success": False,
            "registration_reliable": False,
            "error": (completed.stderr or completed.stdout).strip()[-2000:],
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "result_dir": str(result_dir),
            "reused": False,
        }

    try:
        payload = load_worker_json(result_dir)
        return flatten_result(
            payload=payload,
            row=row,
            phase=phase,
            sample_index=sample_index,
            steps=steps,
            keypoints=keypoints,
            strategy=strategy,
            result_dir=result_dir,
            reused=False,
        )
    except Exception as exc:
        return {
            "phase": phase,
            "sample_index": sample_index,
            "season": row.get("season", ""),
            "stem": stem,
            "sar_path": row["sar_path"],
            "opt_path": row["opt_path"],
            "steps": steps,
            "max_keypoints": keypoints,
            "extractor": strategy["label"],
            "extractor_policy": strategy["policy"],
            "extractors": "+".join(strategy["extractors"]),
            "match_preprocess": strategy["match_preprocess"],
            "matcher_strategy": strategy_name,
            "selected_extractor": "",
            "worker_success": False,
            "registration_reliable": False,
            "error": f"Could not parse worker output: {exc}",
            "elapsed_ms": round((time.perf_counter() - started) * 1000, 2),
            "result_dir": str(result_dir),
            "reused": False,
        }


def flatten_result(
    *,
    payload: dict[str, Any],
    row: dict[str, str],
    phase: str,
    sample_index: int,
    steps: int,
    keypoints: int,
    strategy: dict[str, Any],
    result_dir: Path,
    reused: bool,
) -> dict[str, Any]:
    metrics = payload.get("metrics", {})
    timings = payload.get("timings", {})
    matcher_candidates = metrics.get("matcher_candidates") or []
    return {
        "phase": phase,
        "sample_index": sample_index,
        "season": row.get("season", ""),
        "stem": row.get("stem") or Path(row["sar_path"]).stem,
        "sar_path": row["sar_path"],
        "opt_path": row["opt_path"],
        "steps": steps,
        "max_keypoints": keypoints,
        "extractor": strategy["label"],
        "extractor_policy": metrics.get("extractor_policy", strategy["policy"]),
        "extractors": "+".join(metrics.get("extractors_tried") or strategy["extractors"]),
        "match_preprocess": metrics.get("match_preprocess", strategy["match_preprocess"]),
        "matcher_strategy": strategy["name"],
        "selected_extractor": metrics.get("selected_extractor") or metrics.get("extractor"),
        "worker_success": bool(payload.get("success")),
        "registration_reliable": bool(metrics.get("registration_reliable")),
        "match_count": metrics.get("match_count"),
        "inliers": metrics.get("inliers"),
        "inlier_ratio": metrics.get("inlier_ratio"),
        "rmse": metrics.get("rmse"),
        "mean_score": metrics.get("mean_score"),
        "difference_mean": metrics.get("difference_mean"),
        "dx": metrics.get("dx"),
        "dy": metrics.get("dy"),
        "reliability_reasons": ";".join(metrics.get("reliability_reasons") or []),
        "poor_spatial_coverage": bool(metrics.get("poor_spatial_coverage")),
        "bad_homography_shape": bool(metrics.get("bad_homography_shape")),
        "cascade_rescued": bool(metrics.get("cascade_rescued")),
        "matcher_candidates_json": json.dumps(matcher_candidates, ensure_ascii=False),
        "generation_ms": timings.get("generation_ms"),
        "registration_ms": timings.get("registration_ms"),
        "total_ms": timings.get("total_ms"),
        "error": "",
        "result_dir": str(result_dir),
        "reused": reused,
    }


def numeric_values(rows: list[dict[str, Any]], key: str, *, reliable_only: bool = False) -> list[float]:
    values: list[float] = []
    for row in rows:
        if reliable_only and not row.get("registration_reliable"):
            continue
        value = row.get(key)
        if value in ("", None):
            continue
        try:
            values.append(float(value))
        except (TypeError, ValueError):
            continue
    return values


def median_or_none(values: list[float]) -> float | None:
    return round(float(median(values)), 6) if values else None


def summarize_phase(rows: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["phase"] == phase:
            grouped[
                (
                    str(row.get("matcher_strategy") or row.get("extractor") or "superpoint"),
                    str(row.get("extractor_policy") or "single"),
                    str(row.get("match_preprocess") or "rgb"),
                    int(row["steps"]),
                    int(row["max_keypoints"]),
                )
            ].append(row)

    summaries: list[dict[str, Any]] = []
    for (strategy, policy, match_preprocess, steps, keypoints), group_rows in sorted(grouped.items()):
        n = len(group_rows)
        worker_success_count = sum(bool(row.get("worker_success")) for row in group_rows)
        reliable_count = sum(bool(row.get("registration_reliable")) for row in group_rows)
        failure_count = n - worker_success_count
        selected_counts = Counter(str(row.get("selected_extractor") or "") for row in group_rows if row.get("selected_extractor"))
        summaries.append(
            {
                "phase": phase,
                "matcher_strategy": strategy,
                "extractor": strategy,
                "extractor_policy": policy,
                "match_preprocess": match_preprocess,
                "selected_extractor_counts": json.dumps(dict(sorted(selected_counts.items())), ensure_ascii=False),
                "steps": steps,
                "max_keypoints": keypoints,
                "n": n,
                "worker_success_count": worker_success_count,
                "worker_failure_count": failure_count,
                "worker_success_rate": round(worker_success_count / max(n, 1), 6),
                "reliable_count": reliable_count,
                "success_rate": round(reliable_count / max(n, 1), 6),
                "median_rmse_reliable": median_or_none(numeric_values(group_rows, "rmse", reliable_only=True)),
                "median_rmse_all": median_or_none(numeric_values(group_rows, "rmse")),
                "median_inliers": median_or_none(numeric_values(group_rows, "inliers")),
                "median_inlier_ratio": median_or_none(numeric_values(group_rows, "inlier_ratio")),
                "median_match_count": median_or_none(numeric_values(group_rows, "match_count")),
                "median_total_ms": median_or_none(numeric_values(group_rows, "total_ms")),
                "fail_few_inliers": sum("inliers < 8" in str(row.get("reliability_reasons", "")) for row in group_rows),
                "fail_low_inlier_ratio": sum(
                    "inlier_ratio < 0.25" in str(row.get("reliability_reasons", "")) for row in group_rows
                ),
                "fail_high_rmse": sum("rmse > 5px" in str(row.get("reliability_reasons", "")) for row in group_rows),
                "poor_spatial_coverage_count": sum(
                    "poor_spatial_coverage" in str(row.get("reliability_reasons", "")) for row in group_rows
                ),
                "bad_homography_count": sum(
                    "bad_homography_shape" in str(row.get("reliability_reasons", "")) for row in group_rows
                ),
                "cascade_rescued_count": sum(bool(row.get("cascade_rescued")) for row in group_rows),
            }
        )
    return summaries


def rank_key(summary: dict[str, Any]) -> tuple[float, float, float, float, float]:
    rmse = summary.get("median_rmse_reliable")
    median_time = summary.get("median_total_ms")
    return (
        -float(summary.get("success_rate") or 0.0),
        float(rmse) if rmse is not None else 1e9,
        -float(summary.get("median_inlier_ratio") or 0.0),
        float(median_time) if median_time is not None else 1e9,
        float(summary.get("worker_failure_count") or 0.0),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_paper_table(path: Path, summaries: list[dict[str, Any]]) -> None:
    lines = [
        "| Phase | Strategy | Preprocess | Steps | Keypoints | N | Success Rate | Median RMSE | Median Inliers | Median Time (ms) |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(summaries, key=lambda item: (item["phase"], rank_key(item))):
        rmse = row["median_rmse_reliable"]
        lines.append(
            f"| {row['phase']} | {row['matcher_strategy']} | {row['match_preprocess']} | "
            f"{row['steps']} | {row['max_keypoints']} | {row['n']} | "
            f"{row['success_rate']:.3f} | {rmse if rmse is not None else 'NA'} | "
            f"{row['median_inliers']} | {row['median_total_ms']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_strategies(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.strategy_preset == "cascade-study":
        return [
            {
                "name": "single-superpoint-rgb",
                "label": "superpoint",
                "policy": "single",
                "extractors": ["superpoint"],
                "match_preprocess": "rgb",
            },
            {
                "name": "single-aliked-rgb",
                "label": "aliked",
                "policy": "single",
                "extractors": ["aliked"],
                "match_preprocess": "rgb",
            },
            {
                "name": "cascade-superpoint+aliked-rgb",
                "label": "superpoint+aliked",
                "policy": "cascade",
                "extractors": ["superpoint", "aliked"],
                "match_preprocess": "rgb",
            },
            {
                "name": "cascade-superpoint+aliked-structure",
                "label": "superpoint+aliked",
                "policy": "cascade",
                "extractors": ["superpoint", "aliked"],
                "match_preprocess": "structure",
            },
        ]

    strategies: list[dict[str, Any]] = []
    for policy in args.extractor_policies:
        for match_preprocess in args.match_preprocesses:
            if policy == "single":
                for extractor in args.extractors:
                    strategies.append(
                        {
                            "name": f"single-{extractor}-{match_preprocess}",
                            "label": extractor,
                            "policy": "single",
                            "extractors": [extractor],
                            "match_preprocess": match_preprocess,
                        }
                    )
            else:
                label = "+".join(args.extractors)
                strategies.append(
                    {
                        "name": f"cascade-{label}-{match_preprocess}",
                        "label": label,
                        "policy": "cascade",
                        "extractors": list(args.extractors),
                        "match_preprocess": match_preprocess,
                    }
                )
    return strategies


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep Diffusion + LightGlue parameters on filtered registration pairs.")
    parser.add_argument("--pairs-csv", type=Path, default=DEFAULT_FILTERED_PAIRS_CSV)
    parser.add_argument("--raw-pairs-csv", type=Path, default=DEFAULT_RAW_PAIRS_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--runtime-python", default=path_default_runtime())
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--verify-limit", type=int, default=100)
    parser.add_argument("--steps", type=int, nargs="+", default=[6, 8, 12])
    parser.add_argument("--keypoints", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--extractors", nargs="+", default=["superpoint"], choices=["superpoint", "disk", "aliked"])
    parser.add_argument("--extractor-policies", nargs="+", default=["single"], choices=["single", "cascade"])
    parser.add_argument("--match-preprocesses", nargs="+", default=["rgb"], choices=["rgb", "structure"])
    parser.add_argument("--strategy-preset", default="custom", choices=["custom", "cascade-study"])
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    parser.add_argument("--cascade-stop-on-reliable", action="store_true")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--sar-log", action="store_true")
    parser.add_argument("--skip-filter", action="store_true")
    parser.add_argument("--refresh-filter", action="store_true")
    parser.add_argument("--min-edge", type=float, default=14.0)
    parser.add_argument("--min-std", type=float, default=18.0)
    parser.add_argument("--max-water", type=float, default=0.35)
    parser.add_argument("--min-sar-edge", type=float, default=5.0)
    parser.add_argument("--min-sar-std", type=float, default=4.0)
    parser.add_argument("--no-contact-sheet", action="store_true")
    parser.add_argument("--contact-sheet-limit", type=int, default=80)
    pairs_csv_supplied = "--pairs-csv" in sys.argv[1:]
    args = parser.parse_args()
    if not pairs_csv_supplied and args.pairs_csv == DEFAULT_FILTERED_PAIRS_CSV:
        args.pairs_csv = args.output_dir / "filtered_pairs.csv"
    return args


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_filter_if_needed(args)

    rows = read_pairs(args.pairs_csv)
    coarse_rows = balanced_sample(rows, args.limit, args.seed)
    verify_rows = balanced_sample(rows, args.verify_limit, args.seed + 1009)
    all_results: list[dict[str, Any]] = []

    strategies = build_strategies(args)
    param_grid = list(product(strategies, args.steps, args.keypoints))
    for strategy, steps, keypoints in param_grid:
        for sample_index, row in enumerate(coarse_rows):
            print(
                f"[coarse] strategy={strategy['name']} steps={steps} keypoints={keypoints} "
                f"sample={sample_index + 1}/{len(coarse_rows)}"
            )
            all_results.append(
                run_worker(
                    args=args,
                    row=row,
                    phase="coarse",
                    sample_index=sample_index,
                    steps=steps,
                    keypoints=keypoints,
                    strategy=strategy,
                )
            )

    coarse_summaries = summarize_phase(all_results, "coarse")
    top_summaries = sorted(coarse_summaries, key=rank_key)[: max(args.top_k, 1)]
    verify_params = [
        (str(row.get("matcher_strategy") or row.get("extractor") or "single-superpoint-rgb"), int(row["steps"]), int(row["max_keypoints"]))
        for row in top_summaries
    ]
    strategy_by_name = {strategy["name"]: strategy for strategy in strategies}

    for strategy_name, steps, keypoints in verify_params:
        strategy = strategy_by_name[strategy_name]
        for sample_index, row in enumerate(verify_rows):
            print(
                f"[verify] strategy={strategy['name']} steps={steps} keypoints={keypoints} "
                f"sample={sample_index + 1}/{len(verify_rows)}"
            )
            all_results.append(
                run_worker(
                    args=args,
                    row=row,
                    phase="verify",
                    sample_index=sample_index,
                    steps=steps,
                    keypoints=keypoints,
                    strategy=strategy,
                )
            )

    summaries = coarse_summaries + summarize_phase(all_results, "verify")
    verify_summaries = [row for row in summaries if row["phase"] == "verify"]
    best_source = verify_summaries or coarse_summaries
    best = sorted(best_source, key=rank_key)[0] if best_source else None

    all_runs_path = args.output_dir / "all_runs.csv"
    summary_path = args.output_dir / "summary_by_params.csv"
    best_path = args.output_dir / "best_params.json"
    paper_table_path = args.output_dir / "paper_table.md"

    write_csv(all_runs_path, all_results)
    write_csv(summary_path, summaries)
    write_paper_table(paper_table_path, summaries)

    contact_sheet_summary = None
    if not args.no_contact_sheet:
        cmd = [
            args.runtime_python,
            str(CONTACT_SHEET_SCRIPT),
            "--all-runs-csv",
            str(all_runs_path),
            "--output-dir",
            str(args.output_dir / "failure_contact_sheets"),
            "--phase",
            "all",
            "--limit",
            str(args.contact_sheet_limit),
        ]
        completed = subprocess.run(cmd, cwd=str(WEB_APP_DIR), text=True, capture_output=True, check=False)
        if completed.returncode == 0:
            try:
                contact_sheet_summary = json.loads(completed.stdout.strip().splitlines()[-1])
            except Exception:
                contact_sheet_summary = {"success": True, "stdout": completed.stdout.strip()}
        else:
            contact_sheet_summary = {
                "success": False,
                "error": (completed.stderr or completed.stdout).strip()[-2000:],
            }

    best_payload = {
        "success": best is not None,
        "best_params": {
            "matcher_strategy": str(best["matcher_strategy"]) if best else None,
            "extractor": str(best["extractor"]) if best else None,
            "extractor_policy": str(best["extractor_policy"]) if best else None,
            "match_preprocess": str(best["match_preprocess"]) if best else None,
            "steps": int(best["steps"]) if best else None,
            "max_keypoints": int(best["max_keypoints"]) if best else None,
        },
        "selection_phase": best["phase"] if best else None,
        "selection_rule": "max success_rate, then min median_rmse_reliable, then max median_inlier_ratio, then min median_total_ms",
        "best_summary": best,
        "inputs": {
            "pairs_csv": str(args.pairs_csv),
            "checkpoint": str(args.checkpoint),
            "coarse_limit": args.limit,
            "verify_limit": args.verify_limit,
            "steps": args.steps,
            "keypoints": args.keypoints,
            "extractors": args.extractors,
            "extractor_policies": args.extractor_policies,
            "match_preprocesses": args.match_preprocesses,
            "strategy_preset": args.strategy_preset,
        },
        "outputs": {
            "all_runs_csv": str(all_runs_path),
            "summary_by_params_csv": str(summary_path),
            "paper_table_md": str(paper_table_path),
            "failure_contact_sheets": str(args.output_dir / "failure_contact_sheets"),
        },
        "contact_sheet": contact_sheet_summary,
    }
    best_path.write_text(json.dumps(best_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(best_payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
