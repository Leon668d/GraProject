"""
SAR-光学影像配准 Web 系统

当前版本基于传统 ORB 特征点匹配实现配准流程，并补充了：
1. 日志记录
2. 耗时统计
3. 差分图
4. 轮廓叠加图
"""

from __future__ import annotations

import json
import csv
import sqlite3
import subprocess
import os
import sys
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import rasterio
from flask import Flask, g, jsonify, redirect, render_template, request, send_file, session, url_for
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024
app.config["SECRET_KEY"] = "graduation-design-demo-secret-key"

BASE_DIR = Path(__file__).resolve().parent
WORKSPACE_ROOT = BASE_DIR.parents[1]
UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULTS_FOLDER = BASE_DIR / "results"
LOG_FOLDER = BASE_DIR / "logs"
LOG_FILE = LOG_FOLDER / "operations.jsonl"
DB_PATH = BASE_DIR / "app_data.db"
MODEL_DIR = BASE_DIR / "models"
DEFAULT_CNN_MODEL = MODEL_DIR / "multiscale_highres_ft_best.pth"
MODEL_INPUT_SIZE = 512
DEFAULT_CHECKERBOARD_CELLS = 8
CNN_RUNTIME_PYTHON = BASE_DIR / ".cnn_runtime" / "Scripts" / "python.exe"
CNN_WORKER_SCRIPT = BASE_DIR / "cnn_worker.py"
DEFAULT_DIFFUSION_RUNTIME_PYTHON = Path(r"E:\Anaconda3\envs\sar_diff\python.exe")
DIFFUSION_RUNTIME_PYTHON = Path(
    os.environ.get(
        "DIFFUSION_RUNTIME_PYTHON",
        str(DEFAULT_DIFFUSION_RUNTIME_PYTHON if DEFAULT_DIFFUSION_RUNTIME_PYTHON.exists() else sys.executable),
    )
)
DIFFUSION_WORKER_SCRIPT = BASE_DIR / "scripts" / "diffusion_lightglue_worker.py"
DIFFUSION_DEFAULT_STEPS = int(os.environ.get("DIFFUSION_DEFAULT_STEPS", "8"))
DIFFUSION_DEFAULT_MAX_KEYPOINTS = int(os.environ.get("DIFFUSION_DEFAULT_MAX_KEYPOINTS", "2048"))
DIFFUSION_DEFAULT_EXTRACTOR_POLICY = os.environ.get("DIFFUSION_DEFAULT_EXTRACTOR_POLICY", "cascade")
DIFFUSION_DEFAULT_EXTRACTORS = os.environ.get("DIFFUSION_DEFAULT_EXTRACTORS", "superpoint aliked").split()
DIFFUSION_DEFAULT_MATCH_PREPROCESS = os.environ.get("DIFFUSION_DEFAULT_MATCH_PREPROCESS", "rgb")
DIFFUSION_DEFAULT_RESIZE_MODE = os.environ.get("DIFFUSION_DEFAULT_RESIZE_MODE", "stretch")
DIFFUSION_TIMEOUT_SECONDS = int(os.environ.get("DIFFUSION_TIMEOUT_SECONDS", "900"))
DIFFUSION_TORCH_HOME = BASE_DIR / "runtime_cache" / "torch"
UNSUPPORTED_MODEL_KEYWORDS = ("complex",)
SESSION_META_NAME = "session_meta.json"
DIFFUSION_BASELINE_DIR = WORKSPACE_ROOT / "acd_pretrained_registration_baseline"
DIFFUSION_BASELINE_CHECKPOINT = Path(r"E:\checkPoint\checkPoint\checkpoint-25001-lcm-adv\model.safetensors")
DIFFUSION_BASELINE_ARTIFACTS = {
    "contact_sheet": "contact_sheet.png",
    "metrics_csv": "lightglue_eval.csv",
    "manifest_csv": "manifest.csv",
}
DIFFUSION_PARAM_SWEEP_DIR = WORKSPACE_ROOT / "diffusion_lightglue_param_sweep"
DIFFUSION_CASCADE_STUDY_DIR = WORKSPACE_ROOT / "diffusion_lightglue_cascade_study_100"
DEMO_SAMPLES_DIR = BASE_DIR / "static" / "demo_samples"
EXPERIMENT_ARTIFACT_ROOTS = {
    "param_sweep": DIFFUSION_PARAM_SWEEP_DIR,
    "cascade_study": DIFFUSION_CASCADE_STUDY_DIR,
}
MODEL_RUNTIME = {
    "checked": False,
    "available": False,
    "error": "CNN runtime not checked yet",
    "device": "cpu",
    "model_name": DEFAULT_CNN_MODEL.name,
}
DIFFUSION_RUNTIME = {
    "checked": False,
    "available": False,
    "error": "Diffusion runtime not checked yet",
    "device": "cpu",
}

for folder in (UPLOAD_FOLDER, RESULTS_FOLDER, LOG_FOLDER):
    folder.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["RESULTS_FOLDER"] = str(RESULTS_FOLDER)


def get_db() -> sqlite3.Connection:
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(error):
    del error
    db = g.pop("db", None)
    if db is not None:
        db.close()


def count_image_files(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for path in folder.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff"})


def summarize_lightglue_csv(csv_path: Path) -> dict:
    if not csv_path.exists():
        return {"available": False, "rows": [], "candidates": {}, "primary": None}

    rows = []
    candidates: dict[str, dict] = {}
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candidate = row.get("candidate") or "pretrained"
            metric_row = {
                "candidate": candidate,
                "file": row.get("file") or "",
                "match_count": float(row.get("match_count") or 0),
                "mean_score": float(row.get("mean_score") or 0),
                "inliers": float(row.get("inliers") or 0),
                "inlier_ratio": float(row.get("inlier_ratio") or 0),
            }
            rows.append(metric_row)
            bucket = candidates.setdefault(
                candidate,
                {
                    "sample_count": 0,
                    "match_count_sum": 0.0,
                    "mean_score_sum": 0.0,
                    "inliers_sum": 0.0,
                    "inlier_ratio_sum": 0.0,
                },
            )
            bucket["sample_count"] += 1
            bucket["match_count_sum"] += metric_row["match_count"]
            bucket["mean_score_sum"] += metric_row["mean_score"]
            bucket["inliers_sum"] += metric_row["inliers"]
            bucket["inlier_ratio_sum"] += metric_row["inlier_ratio"]

    for candidate, bucket in candidates.items():
        sample_count = max(bucket["sample_count"], 1)
        candidates[candidate] = {
            "sample_count": bucket["sample_count"],
            "avg_match_count": round(bucket["match_count_sum"] / sample_count, 2),
            "avg_mean_score": round(bucket["mean_score_sum"] / sample_count, 4),
            "avg_inliers": round(bucket["inliers_sum"] / sample_count, 2),
            "avg_inlier_ratio": round(bucket["inlier_ratio_sum"] / sample_count, 4),
        }

    primary = "pretrained" if "pretrained" in candidates else (next(iter(candidates), None))
    return {
        "available": True,
        "rows": rows[:12],
        "candidates": candidates,
        "primary": primary,
    }


def coerce_experiment_value(value: str):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return ""
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        number = float(text)
    except ValueError:
        return text
    if number.is_integer():
        return int(number)
    return number


def read_csv_dicts(csv_path: Path, limit: int | None = None) -> list[dict]:
    if not csv_path.exists():
        return []
    rows: list[dict] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({key: coerce_experiment_value(value) for key, value in row.items()})
            if limit is not None and len(rows) >= limit:
                break
    return rows


def safe_json_load(path: Path, default):
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def artifact_url(root_key: str, relative_path: str) -> str | None:
    root = EXPERIMENT_ARTIFACT_ROOTS.get(root_key)
    if root is None:
        return None
    candidate = root / relative_path
    return url_for("experiment_artifact", root_key=root_key, relative_path=relative_path) if candidate.exists() else None


def static_url_if_exists(relative_path: str) -> str | None:
    candidate = BASE_DIR / "static" / relative_path
    return url_for("static", filename=relative_path.replace("\\", "/")) if candidate.exists() else None


def load_demo_manifest() -> dict:
    manifest_path = DEMO_SAMPLES_DIR / "manifest.json"
    if not manifest_path.exists():
        return {"version": 1, "samples": []}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def demo_sample_static_url(sample_id: str, filename: str) -> str:
    return url_for("static", filename=f"demo_samples/{sample_id}/{filename}")


def build_demo_task(sample: dict) -> dict:
    sample_id = sample["id"]
    artifacts = sample.get("artifacts", {})
    preview_urls = {
        "checkerboard": demo_sample_static_url(sample_id, artifacts.get("checkerboard", "checkerboard.png")),
        "overlay": demo_sample_static_url(sample_id, artifacts.get("overlay", "false_color_overlay.png")),
        "difference": demo_sample_static_url(sample_id, artifacts.get("difference", "difference_map.png")),
        "contour": demo_sample_static_url(sample_id, artifacts.get("contour", "contour_overlay.png")),
        "fake_optical": demo_sample_static_url(sample_id, artifacts.get("fake_optical", "fake_optical.png")),
        "matches": demo_sample_static_url(sample_id, artifacts.get("matches", "lightglue_matches.png")),
        "sar_transferred_matches": demo_sample_static_url(
            sample_id,
            artifacts.get("sar_transferred_matches", "sar_transferred_matches.png"),
        ),
        "optical_transferred_points": demo_sample_static_url(
            sample_id,
            artifacts.get("optical_transferred_points", "optical_transferred_points.png"),
        ),
        "registered_preview": demo_sample_static_url(sample_id, artifacts.get("sar_registered", "sar_registered.png")),
        "sar_condition": demo_sample_static_url(sample_id, artifacts.get("sar_condition", "sar_condition.png")),
        "real_optical": demo_sample_static_url(sample_id, artifacts.get("real_optical", "real_optical_resized.png")),
        "download": demo_sample_static_url(sample_id, artifacts.get("download", "sar_registered.png")),
    }
    return {
        "session_id": sample_id,
        "sar_name": f"{sample.get('stem', sample_id)} SAR",
        "optical_name": f"{sample.get('stem', sample_id)} Optical",
        "method": sample.get("method", "diffusion_lightglue_demo"),
        "model_name": sample.get("model_name", "ACD_LCM_ADV + Cascade LightGlue"),
        "status": "success",
        "created_at": "built-in demo",
        "timings": sample.get("timings", {}),
        "metrics": sample.get("metrics", {}),
        "preview_urls": preview_urls,
        "demo": {
            "id": sample_id,
            "title": sample.get("title", sample_id),
            "season": sample.get("season", ""),
            "purpose": sample.get("purpose", ""),
            "tag": sample.get("tag", ""),
            "primary": bool(sample.get("primary", False)),
        },
    }


def init_db() -> None:
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS tasks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            session_id TEXT NOT NULL,
            sar_name TEXT,
            optical_name TEXT,
            method TEXT,
            model_name TEXT,
            status TEXT NOT NULL,
            upload_ms REAL,
            load_ms REAL,
            preprocess_ms REAL,
            registration_ms REAL,
            visualization_ms REAL,
            total_ms REAL,
            matches_used INTEGER,
            inliers INTEGER,
            difference_mean REAL,
            dx REAL,
            dy REAL,
            created_at TEXT NOT NULL,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
        """
    )
    columns = {row[1] for row in cursor.execute("PRAGMA table_info(tasks)").fetchall()}
    if "model_name" not in columns:
        try:
            cursor.execute("ALTER TABLE tasks ADD COLUMN model_name TEXT")
        except sqlite3.OperationalError as error:
            if "duplicate column name" not in str(error).lower():
                raise
    db.commit()
    db.close()


def get_current_user() -> dict | None:
    user_id = session.get("user_id")
    username = session.get("username")
    if not user_id or not username:
        return None
    return {
        "id": user_id,
        "username": username,
    }


def save_task_record(
    session_id: str,
    status: str,
    method: str | None = None,
    model_name: str | None = None,
    timings: dict | None = None,
    metrics: dict | None = None,
    files: dict | None = None,
) -> None:
    current_user = get_current_user()
    db = get_db()
    db.execute(
        """
        INSERT INTO tasks (
            user_id, session_id, sar_name, optical_name, method, model_name, status,
            upload_ms, load_ms, preprocess_ms, registration_ms, visualization_ms, total_ms,
            matches_used, inliers, difference_mean, dx, dy, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            current_user["id"] if current_user else None,
            session_id,
            (files or {}).get("sar_name"),
            (files or {}).get("optical_name"),
            method,
            model_name,
            status,
            (timings or {}).get("upload_ms"),
            (timings or {}).get("load_ms"),
            (timings or {}).get("preprocess_ms"),
            (timings or {}).get("registration_ms"),
            (timings or {}).get("visualization_ms"),
            (timings or {}).get("total_ms"),
            (metrics or {}).get("matches_used"),
            (metrics or {}).get("inliers"),
            (metrics or {}).get("difference_mean"),
            (metrics or {}).get("dx"),
            (metrics or {}).get("dy"),
            now_text(),
        ),
    )
    db.commit()


def get_cnn_runtime() -> dict:
    global MODEL_RUNTIME

    if MODEL_RUNTIME.get("checked"):
        return MODEL_RUNTIME

    runtime = {
        "checked": True,
        "available": False,
        "error": None,
        "device": "cpu",
        "model_name": DEFAULT_CNN_MODEL.name,
        "python_path": str(CNN_RUNTIME_PYTHON),
        "input_size": MODEL_INPUT_SIZE,
        "warnings": [],
    }

    try:
        if not DEFAULT_CNN_MODEL.exists():
            raise FileNotFoundError(f"未找到模型权重: {DEFAULT_CNN_MODEL}")
        if not CNN_RUNTIME_PYTHON.exists():
            raise FileNotFoundError(f"未找到本地 CNN 运行环境: {CNN_RUNTIME_PYTHON}")
        if not CNN_WORKER_SCRIPT.exists():
            raise FileNotFoundError(f"未找到 CNN worker 脚本: {CNN_WORKER_SCRIPT}")

        command = [
            str(CNN_RUNTIME_PYTHON),
            "-c",
            "import torch, json; print(json.dumps({'torch_version': torch.__version__, 'cuda': torch.cuda.is_available()}))",
        ]
        result = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "无法启动本地 CNN 运行环境")

        runtime_info = json.loads(result.stdout.strip())
        device = "cuda" if runtime_info.get("cuda") else "cpu"

        runtime.update(
            {
                "available": True,
                "error": None,
                "device": device,
                "torch_version": runtime_info.get("torch_version"),
            }
        )
    except Exception as error:
        runtime["error"] = str(error)

    MODEL_RUNTIME = runtime
    return MODEL_RUNTIME


def get_diffusion_runtime() -> dict:
    global DIFFUSION_RUNTIME

    if DIFFUSION_RUNTIME.get("checked"):
        return DIFFUSION_RUNTIME

    superpoint_cache = DIFFUSION_TORCH_HOME / "hub" / "checkpoints" / "superpoint_v1.pth"
    lightglue_cache = DIFFUSION_TORCH_HOME / "hub" / "checkpoints" / "superpoint_lightglue_v0-1_arxiv.pth"
    runtime = {
        "checked": True,
        "available": False,
        "error": None,
        "device": "cpu",
        "python_path": str(DIFFUSION_RUNTIME_PYTHON),
        "worker_path": str(DIFFUSION_WORKER_SCRIPT),
        "checkpoint_path": str(DIFFUSION_BASELINE_CHECKPOINT),
        "checkpoint_exists": DIFFUSION_BASELINE_CHECKPOINT.exists(),
        "worker_exists": DIFFUSION_WORKER_SCRIPT.exists(),
        "superpoint_cache_exists": superpoint_cache.exists(),
        "lightglue_cache_exists": lightglue_cache.exists(),
        "warnings": [],
    }

    try:
        if not DIFFUSION_RUNTIME_PYTHON.exists():
            raise FileNotFoundError(f"未找到扩散运行环境: {DIFFUSION_RUNTIME_PYTHON}")
        if not DIFFUSION_WORKER_SCRIPT.exists():
            raise FileNotFoundError(f"未找到扩散推理 worker: {DIFFUSION_WORKER_SCRIPT}")
        if not DIFFUSION_BASELINE_CHECKPOINT.exists():
            raise FileNotFoundError(f"未找到扩散模型权重: {DIFFUSION_BASELINE_CHECKPOINT}")
        if not superpoint_cache.exists() or not lightglue_cache.exists():
            runtime["warnings"].append("LightGlue/SuperPoint 权重缓存未完全就绪，首次运行可能需要联网下载。")

        command = [
            str(DIFFUSION_RUNTIME_PYTHON),
            "-c",
            (
                "import json, torch, diffusers, safetensors, cv2, kornia; "
                "print(json.dumps({'torch_version': torch.__version__, "
                "'cuda': torch.cuda.is_available(), "
                "'diffusers_version': diffusers.__version__}))"
            ),
        ]
        result = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            env={**os.environ, "TORCH_HOME": str(DIFFUSION_TORCH_HOME)},
            capture_output=True,
            text=True,
            timeout=45,
            check=False,
        )
        if result.returncode != 0:
            raise RuntimeError(result.stderr.strip() or result.stdout.strip() or "无法启动扩散运行环境")

        runtime_info = json.loads(result.stdout.strip())
        runtime.update(
            {
                "available": True,
                "error": None,
                "device": "cuda" if runtime_info.get("cuda") else "cpu",
                "torch_version": runtime_info.get("torch_version"),
                "diffusers_version": runtime_info.get("diffusers_version"),
            }
        )
    except Exception as error:
        runtime["error"] = str(error)

    DIFFUSION_RUNTIME = runtime
    return DIFFUSION_RUNTIME


def is_supported_model_file(model_path: Path) -> bool:
    name = model_path.name.lower()
    if model_path.suffix.lower() != ".pth":
        return False
    return not any(keyword in name for keyword in UNSUPPORTED_MODEL_KEYWORDS)


def get_model_profile(model_name: str | Path | None) -> dict:
    name = Path(model_name).name.lower() if model_name else ""
    experimental = "experimental" in name or "smoke" in name

    def finalize(profile: dict) -> dict:
        return {
            **profile,
            "experimental": experimental,
            "status_tag": "experimental" if experimental else "stable",
        }

    if "dense" in name or "nonrigid" in name:
        return finalize({
            "family": "dense_field_nonrigid",
            "display_name": "Dense-Field Nonrigid CNN",
            "registration_type": "dense_field",
            "coarse_to_fine": True,
            "supports_dense_field": True,
            "supports_nonrigid": True,
            "supports_homography": False,
        })
    if "homography" in name:
        return finalize({
            "family": "multiscale_homography",
            "display_name": "Multi-scale Homography CNN",
            "registration_type": "global_homography",
            "coarse_to_fine": True,
            "supports_dense_field": False,
            "supports_nonrigid": False,
            "supports_homography": True,
        })
    if "multiscale" in name:
        return finalize({
            "family": "multiscale_translation",
            "display_name": "Multi-scale Coarse-to-Fine CNN",
            "registration_type": "global_translation",
            "coarse_to_fine": True,
            "supports_dense_field": False,
            "supports_nonrigid": False,
            "supports_homography": False,
        })

    return finalize({
        "family": "baseline_translation",
        "display_name": "Baseline Translation CNN",
        "registration_type": "global_translation",
        "coarse_to_fine": False,
        "supports_dense_field": False,
        "supports_nonrigid": False,
        "supports_homography": False,
    })


def get_supported_model_paths() -> list[Path]:
    discovered_by_name: dict[str, Path] = {}
    if MODEL_DIR.exists():
        for path in MODEL_DIR.glob("*.pth"):
            if is_supported_model_file(path):
                discovered_by_name.setdefault(path.name, path)

    def sort_key(path: Path):
        name = path.name.lower()
        return (
            0 if name.endswith("_best.pth") else 1,
            0 if path.name == DEFAULT_CNN_MODEL.name else 1,
            0 if "homography" in name else 1,
            0 if "multiscale" in name else 1,
            0 if "dense" in name else 1,
            0 if "baseline_best" in name else 1,
            name,
        )

    return sorted(discovered_by_name.values(), key=sort_key)


def resolve_model_path(model_name: str | None) -> Path:
    if not model_name:
        return DEFAULT_CNN_MODEL

    candidate = MODEL_DIR / model_name
    if candidate.exists() and is_supported_model_file(candidate):
        return candidate
    raise FileNotFoundError(f"不支持或不存在的模型文件: {model_name}")


def now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def resolve_model_path(model_name: str | None) -> Path:
    if not model_name:
        return DEFAULT_CNN_MODEL
    for candidate in get_supported_model_paths():
        if candidate.name == model_name:
            return candidate
    raise FileNotFoundError(f"Unsupported or missing model file: {model_name}")


def normalize_to_float(image: np.ndarray, use_log: bool = False) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if use_log:
        array = np.log1p(np.clip(array, a_min=0, a_max=None))

    min_value = float(array.min())
    max_value = float(array.max())
    if max_value - min_value < 1e-10:
        return np.zeros_like(array, dtype=np.float32)

    return (array - min_value) / (max_value - min_value)


def make_json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def session_meta_path(session_dir: Path) -> Path:
    return session_dir / SESSION_META_NAME


def load_session_metadata(session_dir: Path) -> dict:
    meta_path = session_meta_path(session_dir)
    if not meta_path.exists():
        return {}
    with meta_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_session_metadata(session_dir: Path, data: dict) -> None:
    meta_path = session_meta_path(session_dir)
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(make_json_safe(data), file, ensure_ascii=False, indent=2)


def append_operation_log(action: str, status: str = "success", **details) -> dict:
    entry = {
        "timestamp": now_text(),
        "action": action,
        "status": status,
        **make_json_safe(details),
    }
    with LOG_FILE.open("a", encoding="utf-8") as file:
        file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def user_owns_session(session_id: str, current_user: dict | None = None) -> bool:
    current_user = current_user or get_current_user()
    if not current_user:
        return False

    metadata = load_session_metadata(UPLOAD_FOLDER / session_id)
    owner = metadata.get("user")
    if isinstance(owner, dict) and owner.get("id") == current_user.get("id"):
        return True

    db = get_db()
    row = db.execute(
        "SELECT 1 FROM tasks WHERE user_id = ? AND session_id = ? LIMIT 1",
        (current_user["id"], session_id),
    ).fetchone()
    return row is not None


def build_preview_urls(session_id: str) -> dict:
    result_dir = RESULTS_FOLDER / session_id
    download_name = (
        "registered_output_original.png"
        if (result_dir / "registered_output_original.png").exists()
        else ("registered_output.png" if (result_dir / "registered_output.png").exists() else "registered_output.tif")
    )
    return {
        "checkerboard": (
            f"/api/results/{session_id}/checkerboard_original.png"
            if (result_dir / "checkerboard_original.png").exists()
            else f"/api/results/{session_id}/checkerboard.png"
        ),
        "overlay": (
            f"/api/results/{session_id}/false_color_overlay_original.png"
            if (result_dir / "false_color_overlay_original.png").exists()
            else f"/api/results/{session_id}/false_color_overlay.png"
        ),
        "difference": (
            f"/api/results/{session_id}/difference_map_original.png"
            if (result_dir / "difference_map_original.png").exists()
            else f"/api/results/{session_id}/difference_map.png"
        ),
        "contour": (
            f"/api/results/{session_id}/contour_overlay_original.png"
            if (result_dir / "contour_overlay_original.png").exists()
            else f"/api/results/{session_id}/contour_overlay.png"
        ),
        "deformation": (
            f"/api/results/{session_id}/deformation_heatmap.png"
            if (result_dir / "deformation_heatmap.png").exists()
            else None
        ),
        "fake_optical": (
            f"/api/results/{session_id}/fake_optical.png"
            if (result_dir / "fake_optical.png").exists()
            else None
        ),
        "matches": (
            f"/api/results/{session_id}/lightglue_matches.png"
            if (result_dir / "lightglue_matches.png").exists()
            else None
        ),
        "sar_transferred_matches": (
            f"/api/results/{session_id}/sar_transferred_matches.png"
            if (result_dir / "sar_transferred_matches.png").exists()
            else None
        ),
        "optical_transferred_points": (
            f"/api/results/{session_id}/optical_transferred_points.png"
            if (result_dir / "optical_transferred_points.png").exists()
            else None
        ),
        "fake_registered": (
            f"/api/results/{session_id}/fake_optical_registered.png"
            if (result_dir / "fake_optical_registered.png").exists()
            else None
        ),
        "sar_registered": (
            f"/api/results/{session_id}/sar_registered_original.png"
            if (result_dir / "sar_registered_original.png").exists()
            else (f"/api/results/{session_id}/sar_registered.png" if (result_dir / "sar_registered.png").exists() else None)
        ),
        "registered_preview": (
            f"/api/results/{session_id}/sar_registered_original.png"
            if (result_dir / "sar_registered_original.png").exists()
            else (f"/api/results/{session_id}/sar_registered.png" if (result_dir / "sar_registered.png").exists() else None)
        ),
        "sar_registered_canvas": (
            f"/api/results/{session_id}/sar_registered.png"
            if (result_dir / "sar_registered.png").exists()
            else None
        ),
        "sar_condition": (
            f"/api/results/{session_id}/sar_condition.png"
            if (result_dir / "sar_condition.png").exists()
            else None
        ),
        "real_optical": (
            f"/api/results/{session_id}/real_optical_resized.png"
            if (result_dir / "real_optical_resized.png").exists()
            else None
        ),
        "download": f"/api/download/{session_id}/{download_name}",
    }


def query_task_history(limit: int = 12) -> list[dict]:
    current_user = get_current_user()
    if not current_user:
        return []

    db = get_db()
    rows = db.execute(
        """
        SELECT
            session_id, sar_name, optical_name, method, model_name, status,
            upload_ms, load_ms, preprocess_ms, registration_ms, visualization_ms, total_ms,
            matches_used, inliers, difference_mean, dx, dy, created_at
        FROM tasks
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT ?
        """,
        (current_user["id"], limit),
    ).fetchall()

    history = []
    for row in rows:
        metadata = load_session_metadata(UPLOAD_FOLDER / row["session_id"])
        full_metrics = {
            "matches_used": row["matches_used"],
            "inliers": row["inliers"],
            "difference_mean": row["difference_mean"],
            "dx": row["dx"],
            "dy": row["dy"],
        }
        full_metrics.update(metadata.get("last_registration", {}).get("metrics", {}))
        history.append(
            {
                "session_id": row["session_id"],
                "sar_name": row["sar_name"],
                "optical_name": row["optical_name"],
                "method": row["method"],
                "model_name": row["model_name"],
                "status": row["status"],
                "created_at": row["created_at"],
                "timings": {
                    "upload_ms": row["upload_ms"],
                    "load_ms": row["load_ms"],
                    "preprocess_ms": row["preprocess_ms"],
                    "registration_ms": row["registration_ms"],
                    "visualization_ms": row["visualization_ms"],
                    "total_ms": row["total_ms"],
                },
                "metrics": {
                    **full_metrics,
                },
                "preview_urls": build_preview_urls(row["session_id"]),
            }
        )
    return history


def query_task_detail(session_id: str) -> dict | None:
    current_user = get_current_user()
    if not current_user:
        return None

    db = get_db()
    row = db.execute(
        """
        SELECT
            session_id, sar_name, optical_name, method, model_name, status,
            upload_ms, load_ms, preprocess_ms, registration_ms, visualization_ms, total_ms,
            matches_used, inliers, difference_mean, dx, dy, created_at
        FROM tasks
        WHERE user_id = ? AND session_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (current_user["id"], session_id),
    ).fetchone()

    if row is None:
        return None

    metadata = load_session_metadata(UPLOAD_FOLDER / session_id)
    full_metrics = {
        "matches_used": row["matches_used"],
        "inliers": row["inliers"],
        "difference_mean": row["difference_mean"],
        "dx": row["dx"],
        "dy": row["dy"],
    }
    full_metrics.update(metadata.get("last_registration", {}).get("metrics", {}))

    return {
        "session_id": row["session_id"],
        "sar_name": row["sar_name"],
        "optical_name": row["optical_name"],
        "method": row["method"],
        "model_name": row["model_name"],
        "status": row["status"],
        "created_at": row["created_at"],
        "timings": {
            "upload_ms": row["upload_ms"],
            "load_ms": row["load_ms"],
            "preprocess_ms": row["preprocess_ms"],
            "registration_ms": row["registration_ms"],
            "visualization_ms": row["visualization_ms"],
            "total_ms": row["total_ms"],
        },
        "metrics": {
            **full_metrics,
        },
        "preview_urls": build_preview_urls(row["session_id"]),
    }


def query_task_summary() -> dict:
    current_user = get_current_user()
    if not current_user:
        return {
            "total_tasks": 0,
            "success_tasks": 0,
            "failed_tasks": 0,
            "avg_total_ms": None,
            "latest_created_at": None,
            "latest_model_name": None,
        }

    db = get_db()
    aggregate = db.execute(
        """
        SELECT
            COUNT(*) AS total_tasks,
            SUM(CASE WHEN status = 'success' THEN 1 ELSE 0 END) AS success_tasks,
            SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed_tasks,
            AVG(total_ms) AS avg_total_ms
        FROM tasks
        WHERE user_id = ?
        """,
        (current_user["id"],),
    ).fetchone()
    latest = db.execute(
        """
        SELECT created_at, model_name
        FROM tasks
        WHERE user_id = ?
        ORDER BY id DESC
        LIMIT 1
        """,
        (current_user["id"],),
    ).fetchone()

    return {
        "total_tasks": aggregate["total_tasks"] or 0,
        "success_tasks": aggregate["success_tasks"] or 0,
        "failed_tasks": aggregate["failed_tasks"] or 0,
        "avg_total_ms": round(float(aggregate["avg_total_ms"]), 2) if aggregate["avg_total_ms"] is not None else None,
        "latest_created_at": latest["created_at"] if latest else None,
        "latest_model_name": latest["model_name"] if latest else None,
    }


def read_recent_logs(limit: int = 8) -> list[dict]:
    if not LOG_FILE.exists():
        return []

    with LOG_FILE.open("r", encoding="utf-8") as file:
        lines = [line.strip() for line in file if line.strip()]

    recent_lines = lines[-limit:]
    return [json.loads(line) for line in reversed(recent_lines)]


def read_recent_logs(limit: int = 8, current_user: dict | None = None) -> list[dict]:
    if not LOG_FILE.exists():
        return []

    current_user = current_user or get_current_user()
    if not current_user:
        return []

    with LOG_FILE.open("r", encoding="utf-8") as file:
        entries = [json.loads(line) for line in file if line.strip()]

    visible_entries = []
    for entry in entries:
        entry_user = entry.get("user")
        if isinstance(entry_user, dict) and entry_user.get("id") == current_user.get("id"):
            visible_entries.append(entry)
            continue
        if entry.get("username") == current_user.get("username"):
            visible_entries.append(entry)

    return list(reversed(visible_entries[-limit:]))


def normalize_to_uint8(image: np.ndarray, use_log: bool = False) -> np.ndarray:
    array = np.asarray(image, dtype=np.float32)
    if use_log:
        array = np.log1p(np.clip(array, a_min=0, a_max=None))

    min_value = float(array.min())
    max_value = float(array.max())
    if max_value - min_value < 1e-10:
        return np.zeros_like(array, dtype=np.uint8)

    normalized = (array - min_value) / (max_value - min_value)
    return (normalized * 255).clip(0, 255).astype(np.uint8)


def to_gray_uint8(image: np.ndarray, use_log: bool = False) -> np.ndarray:
    if image.ndim == 2:
        return normalize_to_uint8(image, use_log=use_log)

    rgb_uint8 = normalize_to_uint8(image, use_log=use_log)
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2GRAY)


def to_bgr_uint8(image: np.ndarray, use_log: bool = False) -> np.ndarray:
    if image.ndim == 2:
        gray = normalize_to_uint8(image, use_log=use_log)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    rgb_uint8 = normalize_to_uint8(image, use_log=use_log)
    return cv2.cvtColor(rgb_uint8, cv2.COLOR_RGB2BGR)


def find_uploaded_files(session_dir: Path) -> tuple[Path | None, Path | None]:
    files = [file for file in session_dir.iterdir() if file.is_file() and file.name != SESSION_META_NAME]
    sar_path = None
    optical_path = None

    for file in files:
        suffix = file.suffix.lower()
        if suffix not in [".tif", ".tiff", ".npy", ".jpg", ".jpeg", ".png"]:
            continue

        lower_name = file.name.lower()
        if sar_path is None and ("sar" in lower_name or lower_name.startswith("r") or suffix == ".npy"):
            sar_path = file
            continue

        if optical_path is None and ("optical" in lower_name or "opt" in lower_name or lower_name.startswith("l")):
            optical_path = file

    image_files = [file for file in files if file.suffix.lower() in [".tif", ".tiff", ".npy", ".jpg", ".jpeg", ".png"]]
    if sar_path is None and image_files:
        sar_path = image_files[0]
    if optical_path is None and len(image_files) > 1:
        optical_path = image_files[1]

    return sar_path, optical_path


def read_sar_image(sar_path: str) -> np.ndarray:
    lower_path = sar_path.lower()
    if lower_path.endswith(".npy"):
        return np.load(sar_path)

    if lower_path.endswith((".jpg", ".jpeg", ".png")):
        image = cv2.imread(sar_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"无法读取 SAR 文件: {sar_path}")
        return image

    with rasterio.open(sar_path) as src:
        return src.read(1)


def read_optical_image(optical_path: str) -> tuple[np.ndarray, dict]:
    lower_path = optical_path.lower()

    if lower_path.endswith((".jpg", ".jpeg", ".png")):
        image = cv2.imread(optical_path, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"无法读取光学文件: {optical_path}")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        profile = {
            "driver": "GTiff",
            "height": rgb_image.shape[0],
            "width": rgb_image.shape[1],
            "count": 1,
            "dtype": "float32",
        }
        return rgb_image, profile

    with rasterio.open(optical_path) as src:
        count = min(src.count, 3)
        optical = src.read(list(range(1, count + 1)))
        optical = np.transpose(optical, (1, 2, 0))
        profile = src.profile.copy()
    return optical, profile


def create_checkerboard_fixed(
    sar: np.ndarray,
    optical: np.ndarray,
    board_cells: int = DEFAULT_CHECKERBOARD_CELLS,
) -> np.ndarray:
    if optical.ndim == 3:
        height, width = optical.shape[:2]
    else:
        height, width = optical.shape

    resize_interpolation = cv2.INTER_AREA if (
        sar.shape[0] > height or sar.shape[1] > width
    ) else cv2.INTER_LINEAR
    sar_resized = cv2.resize(sar, (width, height), interpolation=resize_interpolation)

    sar_gray = to_gray_uint8(sar_resized, use_log=True)
    optical_gray = to_gray_uint8(optical)

    # Mild smoothing makes the checkerboard more readable for SAR/optical pairs
    # without hiding large-scale structural misalignment.
    sar_gray = cv2.GaussianBlur(sar_gray, (0, 0), sigmaX=1.2, sigmaY=1.2)
    optical_gray = cv2.GaussianBlur(optical_gray, (0, 0), sigmaX=1.2, sigmaY=1.2)

    sar_bgr = cv2.cvtColor(sar_gray, cv2.COLOR_GRAY2BGR)
    optical_bgr = cv2.cvtColor(optical_gray, cv2.COLOR_GRAY2BGR)

    rows = max(2, board_cells)
    cols = max(2, int(round(board_cells * width / max(height, 1))))

    y_coords = np.arange(height, dtype=np.float32)[:, None]
    x_coords = np.arange(width, dtype=np.float32)[None, :]
    cell_h = height / float(rows)
    cell_w = width / float(cols)

    row_idx = np.minimum((y_coords / cell_h).astype(np.int32), rows - 1)
    col_idx = np.minimum((x_coords / cell_w).astype(np.int32), cols - 1)
    parity_mask = ((row_idx + col_idx) % 2 == 0).astype(np.float32)

    y_local = np.mod(y_coords, cell_h)
    x_local = np.mod(x_coords, cell_w)
    dist_to_vertical = np.minimum(x_local, cell_w - x_local)
    dist_to_horizontal = np.minimum(y_local, cell_h - y_local)
    dist_to_seam = np.minimum(dist_to_vertical, dist_to_horizontal)

    feather = max(6.0, min(cell_w, cell_h) * 0.18)
    blend_zone = np.clip(dist_to_seam / feather, 0.0, 1.0)
    blend_zone = blend_zone * blend_zone * (3.0 - 2.0 * blend_zone)
    soft_mask = np.where(
        dist_to_seam < feather,
        0.5 + (parity_mask - 0.5) * blend_zone,
        parity_mask,
    )
    soft_mask = soft_mask[..., None].astype(np.float32)
    checkerboard = sar_bgr.astype(np.float32) * soft_mask + optical_bgr.astype(np.float32) * (1.0 - soft_mask)
    return np.clip(checkerboard, 0, 255).astype(np.uint8)


def create_false_color_overlay_fixed(sar: np.ndarray, optical: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    if optical.ndim == 3:
        height, width = optical.shape[:2]
    else:
        height, width = optical.shape

    sar_gray = to_gray_uint8(cv2.resize(sar, (width, height), interpolation=cv2.INTER_LINEAR), use_log=True)
    optical_bgr = to_bgr_uint8(optical)

    sar_color = np.zeros((height, width, 3), dtype=np.uint8)
    sar_color[:, :, 2] = sar_gray

    optical_color = np.zeros((height, width, 3), dtype=np.uint8)
    optical_color[:, :, 0] = optical_bgr[:, :, 0]
    optical_color[:, :, 1] = optical_bgr[:, :, 1]

    return cv2.addWeighted(sar_color, alpha, optical_color, 1.0 - alpha, 0)


def create_difference_map(sar: np.ndarray, optical: np.ndarray) -> np.ndarray:
    if optical.ndim == 3:
        height, width = optical.shape[:2]
    else:
        height, width = optical.shape

    sar_gray = to_gray_uint8(cv2.resize(sar, (width, height), interpolation=cv2.INTER_LINEAR), use_log=True)
    optical_gray = to_gray_uint8(optical)

    sar_gradient_x = cv2.Sobel(sar_gray, cv2.CV_32F, 1, 0, ksize=3)
    sar_gradient_y = cv2.Sobel(sar_gray, cv2.CV_32F, 0, 1, ksize=3)
    optical_gradient_x = cv2.Sobel(optical_gray, cv2.CV_32F, 1, 0, ksize=3)
    optical_gradient_y = cv2.Sobel(optical_gray, cv2.CV_32F, 0, 1, ksize=3)

    sar_magnitude = cv2.magnitude(sar_gradient_x, sar_gradient_y)
    optical_magnitude = cv2.magnitude(optical_gradient_x, optical_gradient_y)

    sar_grad_uint8 = normalize_to_uint8(sar_magnitude)
    optical_grad_uint8 = normalize_to_uint8(optical_magnitude)
    difference = cv2.absdiff(sar_grad_uint8, optical_grad_uint8)
    difference = cv2.GaussianBlur(difference, (5, 5), 0)
    return difference


def create_contour_overlay(sar: np.ndarray, optical: np.ndarray) -> np.ndarray:
    if optical.ndim == 3:
        height, width = optical.shape[:2]
    else:
        height, width = optical.shape

    sar_gray = to_gray_uint8(cv2.resize(sar, (width, height), interpolation=cv2.INTER_LINEAR), use_log=True)
    optical_gray = to_gray_uint8(optical)

    sar_edges = cv2.Canny(cv2.GaussianBlur(sar_gray, (5, 5), 0), 60, 160)
    optical_edges = cv2.Canny(cv2.GaussianBlur(optical_gray, (5, 5), 0), 60, 160)

    kernel = np.ones((2, 2), np.uint8)
    sar_edges = cv2.dilate(sar_edges, kernel, iterations=1)
    optical_edges = cv2.dilate(optical_edges, kernel, iterations=1)

    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    overlay[:, :, 2] = optical_edges
    overlay[:, :, 1] = sar_edges
    return overlay


def create_deformation_heatmap(flow: np.ndarray) -> np.ndarray:
    magnitude = np.sqrt(np.square(flow[0]) + np.square(flow[1])).astype(np.float32)
    magnitude_uint8 = normalize_to_uint8(magnitude)
    return cv2.applyColorMap(magnitude_uint8, cv2.COLORMAP_TURBO)


def warp_with_dense_flow(image: np.ndarray, flow: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    map_x = grid_x + flow[0].astype(np.float32)
    map_y = grid_y + flow[1].astype(np.float32)
    return cv2.remap(
        image.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    ).astype(image.dtype)


def homography_vector_to_matrix(vector: np.ndarray, input_size: int) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    if values.size != 8:
        raise ValueError(f"Expected 8 homography parameters, received {values.size}.")

    size = float(input_size - 1)
    source = np.array(
        [[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]],
        dtype=np.float32,
    )
    destination = source + values.reshape(4, 2)
    h_matrix = cv2.getPerspectiveTransform(source, destination)
    scale = float(h_matrix[2, 2]) if abs(float(h_matrix[2, 2])) > 1e-8 else 1.0
    return h_matrix / scale


def scale_homography_to_image_space(
    h_matrix_model: np.ndarray,
    width: int,
    height: int,
    input_size: int,
) -> np.ndarray:
    scale = np.array(
        [
            [input_size / float(width), 0.0, 0.0],
            [0.0, input_size / float(height), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    scale_inv = np.array(
        [
            [float(width) / input_size, 0.0, 0.0],
            [0.0, float(height) / input_size, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    h_matrix_image = scale_inv @ h_matrix_model @ scale
    scale_factor = float(h_matrix_image[2, 2]) if abs(float(h_matrix_image[2, 2])) > 1e-8 else 1.0
    return h_matrix_image / scale_factor


def transform_points_with_homography(points: np.ndarray, h_matrix: np.ndarray) -> np.ndarray:
    reshaped = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(reshaped, h_matrix.astype(np.float32)).reshape(-1, 2)


def save_registration_outputs(
    sar_warped: np.ndarray,
    optical: np.ndarray,
    output_profile: dict,
    result_dir_path: Path,
    dense_flow: np.ndarray | None = None,
    checkerboard_cells: int = DEFAULT_CHECKERBOARD_CELLS,
) -> dict:
    height, width = optical.shape[:2]

    registered_path = result_dir_path / "registered_output.tif"
    checkerboard_path = result_dir_path / "checkerboard.png"
    overlay_path = result_dir_path / "false_color_overlay.png"
    difference_path = result_dir_path / "difference_map.png"
    contour_path = result_dir_path / "contour_overlay.png"
    deformation_path = result_dir_path / "deformation_heatmap.png"
    dense_flow_path = result_dir_path / "dense_flow.npy"

    output_profile = output_profile.copy()
    output_profile.update(
        {
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "dtype": str(sar_warped.dtype),
        }
    )
    output_profile.pop("nodata", None)

    with rasterio.open(registered_path, "w", **output_profile) as dst:
        dst.write(sar_warped, 1)

    checkerboard = create_checkerboard_fixed(sar_warped, optical, board_cells=checkerboard_cells)
    overlay = create_false_color_overlay_fixed(sar_warped, optical, alpha=0.5)
    difference_map = create_difference_map(sar_warped, optical)
    contour_overlay = create_contour_overlay(sar_warped, optical)

    cv2.imwrite(str(checkerboard_path), checkerboard)
    cv2.imwrite(str(overlay_path), overlay)
    cv2.imwrite(str(difference_path), difference_map)
    cv2.imwrite(str(contour_path), contour_overlay)

    deformation_heatmap = None
    if dense_flow is not None:
        deformation_heatmap = create_deformation_heatmap(dense_flow)
        cv2.imwrite(str(deformation_path), deformation_heatmap)
        np.save(dense_flow_path, dense_flow.astype(np.float32))

    return {
        "artifacts": {
            "checkerboard_path": checkerboard_path,
            "overlay_path": overlay_path,
            "difference_path": difference_path,
            "contour_path": contour_path,
            "registered_path": registered_path,
            "deformation_path": deformation_path if dense_flow is not None else None,
            "dense_flow_path": dense_flow_path if dense_flow is not None else None,
        },
        "difference_map": difference_map,
        "deformation_heatmap": deformation_heatmap,
    }


def register_orb(sar_path: str, optical_path: str, result_dir: str) -> dict:
    result_dir_path = Path(result_dir)
    timings = {}
    pipeline_start = time.perf_counter()

    load_start = time.perf_counter()
    sar = read_sar_image(sar_path)
    optical, optical_profile = read_optical_image(optical_path)
    timings["load_ms"] = round((time.perf_counter() - load_start) * 1000, 2)

    preprocess_start = time.perf_counter()
    height, width = optical.shape[:2]
    sar = cv2.resize(sar, (width, height), interpolation=cv2.INTER_LINEAR)

    sar_uint8 = to_gray_uint8(sar, use_log=True)
    optical_gray = to_gray_uint8(optical)
    timings["preprocess_ms"] = round((time.perf_counter() - preprocess_start) * 1000, 2)

    registration_start = time.perf_counter()
    orb = cv2.ORB_create(nfeatures=1000, scaleFactor=1.2, nlevels=8)
    keypoints_sar, descriptors_sar = orb.detectAndCompute(sar_uint8, None)
    keypoints_optical, descriptors_optical = orb.detectAndCompute(optical_gray, None)

    matches = []
    if descriptors_sar is not None and descriptors_optical is not None:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(descriptors_sar, descriptors_optical)
        matches = sorted(matches, key=lambda item: item.distance)

    good_matches = matches[: min(100, len(matches))]

    homography = None
    inlier_count = 0
    dx = None
    dy = None
    sar_warped = sar.copy()

    if len(good_matches) > 10:
        src_points = np.float32([keypoints_sar[match.queryIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        dst_points = np.float32([keypoints_optical[match.trainIdx].pt for match in good_matches]).reshape(-1, 1, 2)
        homography, inlier_mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)

        if homography is not None:
            sar_warped = cv2.warpPerspective(
                sar,
                homography,
                (width, height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REFLECT,
            )
            dx = float(homography[0, 2])
            dy = float(homography[1, 2])
            if inlier_mask is not None:
                inlier_count = int(inlier_mask.sum())

    timings["registration_ms"] = round((time.perf_counter() - registration_start) * 1000, 2)

    visualization_start = time.perf_counter()
    output_bundle = save_registration_outputs(sar_warped, optical, optical_profile, result_dir_path)
    difference_map = output_bundle["difference_map"]
    timings["visualization_ms"] = round((time.perf_counter() - visualization_start) * 1000, 2)
    timings["total_ms"] = round((time.perf_counter() - pipeline_start) * 1000, 2)

    metrics = {
        "matches_total": len(matches),
        "matches_used": len(good_matches),
        "inliers": inlier_count,
        "dx": dx,
        "dy": dy,
        "difference_mean": round(float(difference_map.mean()), 2),
    }

    print(f"检测到 {len(keypoints_sar)} 个 SAR 特征点，{len(keypoints_optical)} 个光学特征点")
    print(f"匹配到 {len(matches)} 对，使用 {len(good_matches)} 对最佳匹配")

    return {
        "method": "orb",
        "timings": timings,
        "metrics": metrics,
        "artifacts": output_bundle["artifacts"],
    }


def register_cnn(
    sar_path: str,
    optical_path: str,
    result_dir: str,
    model_path: Path | None = None,
    checkerboard_cells: int = DEFAULT_CHECKERBOARD_CELLS,
) -> dict:
    runtime = get_cnn_runtime()
    if not runtime.get("available"):
        raise RuntimeError(runtime.get("error") or "CNN runtime unavailable")

    selected_model_path = model_path or DEFAULT_CNN_MODEL
    model_profile = get_model_profile(selected_model_path)
    device = runtime["device"]
    input_size = runtime["input_size"]
    result_dir_path = Path(result_dir)
    timings = {}
    pipeline_start = time.perf_counter()

    load_start = time.perf_counter()
    sar = read_sar_image(sar_path)
    optical, optical_profile = read_optical_image(optical_path)
    timings["load_ms"] = round((time.perf_counter() - load_start) * 1000, 2)

    preprocess_start = time.perf_counter()
    height, width = optical.shape[:2]
    sar_resized = cv2.resize(sar, (width, height), interpolation=cv2.INTER_LINEAR)
    optical_gray = to_gray_uint8(optical)

    sar_model = cv2.resize(sar_resized, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    optical_model = cv2.resize(optical_gray, (input_size, input_size), interpolation=cv2.INTER_LINEAR)

    sar_npy_path = result_dir_path / "_cnn_sar.npy"
    optical_npy_path = result_dir_path / "_cnn_optical.npy"
    flow_npy_path = result_dir_path / "_cnn_flow.npy"
    np.save(sar_npy_path, normalize_to_float(sar_model, use_log=True))
    np.save(optical_npy_path, normalize_to_float(optical_model))
    timings["preprocess_ms"] = round((time.perf_counter() - preprocess_start) * 1000, 2)

    inference_start = time.perf_counter()
    try:
        command = [
            str(CNN_RUNTIME_PYTHON),
            str(CNN_WORKER_SCRIPT),
            "--sar-npy",
            str(sar_npy_path),
            "--optical-npy",
            str(optical_npy_path),
            "--model-path",
            str(selected_model_path),
        ]
        if model_profile["family"] == "dense_field_nonrigid":
            command.extend(["--flow-out", str(flow_npy_path)])
        process = subprocess.run(
            command,
            cwd=str(BASE_DIR),
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if process.returncode != 0:
            raise RuntimeError(process.stderr.strip() or process.stdout.strip() or "CNN worker execution failed")

        worker_output = json.loads(process.stdout.strip())
        if not worker_output.get("success"):
            raise RuntimeError(worker_output.get("error") or "CNN worker returned an unsuccessful status")
    finally:
        for temp_file in (sar_npy_path, optical_npy_path):
            if temp_file.exists():
                temp_file.unlink()

    dense_flow = None
    flow_magnitude = None
    homography_model = None
    homography_image = None
    homography_params = None
    corner_mapping = None
    center_mapping = None
    dx = None
    dy = None
    dx_model = None
    dy_model = None
    prediction_type = worker_output.get("prediction_type")

    if prediction_type == "dense_flow":
        if not flow_npy_path.exists():
            raise RuntimeError("Dense-field prediction finished without producing a flow file.")
        dense_flow = np.load(flow_npy_path).astype(np.float32)
        scale_x = width / float(input_size)
        scale_y = height / float(input_size)
        flow_x = cv2.resize(dense_flow[0], (width, height), interpolation=cv2.INTER_LINEAR) * scale_x
        flow_y = cv2.resize(dense_flow[1], (width, height), interpolation=cv2.INTER_LINEAR) * scale_y
        dense_flow = np.stack([flow_x, flow_y], axis=0).astype(np.float32)
        sar_warped = warp_with_dense_flow(sar_resized, dense_flow)
        dx_model = float(dense_flow[0].mean())
        dy_model = float(dense_flow[1].mean())
        dx = dx_model
        dy = dy_model
        flow_magnitude = np.sqrt(np.square(dense_flow[0]) + np.square(dense_flow[1]))
    elif prediction_type == "homography":
        homography_params = np.asarray(worker_output.get("prediction", []), dtype=np.float32).reshape(-1)
        if homography_params.size != 8:
            raise RuntimeError(f"Homography model returned {homography_params.size} values instead of 8.")

        homography_model = homography_vector_to_matrix(homography_params, input_size)
        homography_image = scale_homography_to_image_space(homography_model, width, height, input_size)
        sar_warped = cv2.warpPerspective(
            sar_resized.astype(np.float32),
            homography_image.astype(np.float32),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        ).astype(sar_resized.dtype, copy=False)

        image_corners = np.array(
            [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
            dtype=np.float32,
        )
        warped_corners = transform_points_with_homography(image_corners, homography_image)
        corner_labels = ["top_left", "top_right", "bottom_right", "bottom_left"]
        corner_mapping = {
            label: [round(float(point[0]), 3), round(float(point[1]), 3)]
            for label, point in zip(corner_labels, warped_corners)
        }

        center_point = np.array([[width * 0.5, height * 0.5]], dtype=np.float32)
        center_after = transform_points_with_homography(center_point, homography_image)[0]
        dx = float(center_after[0] - center_point[0, 0])
        dy = float(center_after[1] - center_point[0, 1])
        center_mapping = {
            "source": [round(float(center_point[0, 0]), 3), round(float(center_point[0, 1]), 3)],
            "mapped": [round(float(center_after[0]), 3), round(float(center_after[1]), 3)],
        }

        model_center = np.array([[input_size * 0.5, input_size * 0.5]], dtype=np.float32)
        model_center_after = transform_points_with_homography(model_center, homography_model)[0]
        dx_model = float(model_center_after[0] - model_center[0, 0])
        dy_model = float(model_center_after[1] - model_center[0, 1])
    else:
        prediction = np.asarray(worker_output.get("prediction", []), dtype=np.float32).reshape(-1)

        dx_model = float(prediction[0]) if prediction.size > 0 else 0.0
        dy_model = float(prediction[1]) if prediction.size > 1 else 0.0

        scale_x = width / float(input_size)
        scale_y = height / float(input_size)
        dx = dx_model * scale_x
        dy = dy_model * scale_y

        translation_matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        sar_warped = cv2.warpAffine(
            sar_resized,
            translation_matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
    timings["registration_ms"] = round((time.perf_counter() - inference_start) * 1000, 2)

    visualization_start = time.perf_counter()
    output_bundle = save_registration_outputs(
        sar_warped,
        optical,
        optical_profile,
        result_dir_path,
        dense_flow=dense_flow,
        checkerboard_cells=checkerboard_cells,
    )
    difference_map = output_bundle["difference_map"]
    timings["visualization_ms"] = round((time.perf_counter() - visualization_start) * 1000, 2)
    timings["total_ms"] = round((time.perf_counter() - pipeline_start) * 1000, 2)

    metrics = {
        "matches_total": None,
        "matches_used": None,
        "inliers": None,
        "dx": round(dx, 4) if dx is not None else None,
        "dy": round(dy, 4) if dy is not None else None,
        "difference_mean": round(float(difference_map.mean()), 2),
        "raw_dx_model": round(dx_model, 4) if dx_model is not None else None,
        "raw_dy_model": round(dy_model, 4) if dy_model is not None else None,
        "flow_mean_magnitude": round(float(flow_magnitude.mean()), 4) if flow_magnitude is not None else None,
        "flow_max_magnitude": round(float(flow_magnitude.max()), 4) if flow_magnitude is not None else None,
        "checkerboard_cells": checkerboard_cells,
        "prediction_type": prediction_type,
        "h_matrix": np.round(homography_image, 6).tolist() if homography_image is not None else None,
        "raw_h_matrix_model": np.round(homography_model, 6).tolist() if homography_model is not None else None,
        "raw_h_params_model": np.round(homography_params, 6).tolist() if homography_params is not None else None,
        "corner_mapping": corner_mapping,
        "center_mapping": center_mapping,
    }

    if flow_npy_path.exists():
        flow_npy_path.unlink()

    return {
        "method": "cnn",
        "timings": timings,
        "metrics": metrics,
        "artifacts": output_bundle["artifacts"],
        "model_info": {
            "model_name": selected_model_path.name,
            "model_profile": get_model_profile(selected_model_path),
            "device": device,
            "input_size": input_size,
            "warnings": runtime.get("warnings", []),
            "torch_version": runtime.get("torch_version"),
            "worker_model_family": worker_output.get("model_family"),
            "prediction_type": worker_output.get("prediction_type"),
            "worker_missing_keys": worker_output.get("missing_keys"),
            "worker_unexpected_keys": worker_output.get("unexpected_keys"),
        },
    }


def register_diffusion_lightglue(
    sar_path: str,
    optical_path: str,
    result_dir: str,
    *,
    steps: int = DIFFUSION_DEFAULT_STEPS,
    max_keypoints: int = DIFFUSION_DEFAULT_MAX_KEYPOINTS,
    extractor_policy: str = DIFFUSION_DEFAULT_EXTRACTOR_POLICY,
    extractors: list[str] | None = None,
    match_preprocess: str = DIFFUSION_DEFAULT_MATCH_PREPROCESS,
    resize_mode: str = DIFFUSION_DEFAULT_RESIZE_MODE,
) -> dict:
    if not DIFFUSION_WORKER_SCRIPT.exists():
        raise FileNotFoundError(f"未找到扩散推理 worker: {DIFFUSION_WORKER_SCRIPT}")
    if not DIFFUSION_BASELINE_CHECKPOINT.exists():
        raise FileNotFoundError(f"未找到扩散模型权重: {DIFFUSION_BASELINE_CHECKPOINT}")

    result_dir_path = Path(result_dir)
    result_dir_path.mkdir(parents=True, exist_ok=True)
    requested_policy = extractor_policy if extractor_policy in {"single", "cascade"} else DIFFUSION_DEFAULT_EXTRACTOR_POLICY
    requested_extractors = extractors or DIFFUSION_DEFAULT_EXTRACTORS or ["superpoint"]
    requested_extractors = [item for item in requested_extractors if item in {"superpoint", "disk", "aliked"}] or ["superpoint"]
    requested_preprocess = match_preprocess if match_preprocess in {"rgb", "structure"} else DIFFUSION_DEFAULT_MATCH_PREPROCESS
    requested_resize_mode = resize_mode if resize_mode in {"stretch", "letterbox"} else DIFFUSION_DEFAULT_RESIZE_MODE

    command = [
        str(DIFFUSION_RUNTIME_PYTHON),
        str(DIFFUSION_WORKER_SCRIPT),
        "--sar",
        str(sar_path),
        "--optical",
        str(optical_path),
        "--checkpoint",
        str(DIFFUSION_BASELINE_CHECKPOINT),
        "--output-dir",
        str(result_dir_path),
        "--steps",
        str(max(1, min(int(steps), 50))),
        "--max-keypoints",
        str(max(128, min(int(max_keypoints), 4096))),
        "--extractor",
        requested_extractors[0],
        "--extractor-policy",
        requested_policy,
        "--match-preprocess",
        requested_preprocess,
        "--resize-mode",
        requested_resize_mode,
        "--extractors",
        *requested_extractors,
    ]

    worker_env = os.environ.copy()
    worker_env.setdefault("TORCH_HOME", str(DIFFUSION_TORCH_HOME))
    process = subprocess.run(
        command,
        cwd=str(BASE_DIR),
        env=worker_env,
        capture_output=True,
        text=True,
        timeout=DIFFUSION_TIMEOUT_SECONDS,
        check=False,
    )
    if process.returncode != 0:
        stderr = process.stderr.strip()
        stdout = process.stdout.strip()
        raise RuntimeError(stderr or stdout or "Diffusion + LightGlue worker execution failed")

    result_json = result_dir_path / "diffusion_lightglue_result.json"
    if result_json.exists():
        payload = json.loads(result_json.read_text(encoding="utf-8"))
    else:
        payload = json.loads(process.stdout.strip().splitlines()[-1])
    if not payload.get("success"):
        raise RuntimeError(payload.get("error") or "Diffusion + LightGlue worker returned an unsuccessful status")
    return payload


@app.route("/")
def index():
    if get_current_user():
        return redirect(url_for("dashboard"))
    return redirect(url_for("login_page"))


@app.route("/login")
def login_page():
    if get_current_user():
        return redirect(url_for("dashboard"))
    return render_template("login.html")


@app.route("/dashboard")
def dashboard():
    current_user = get_current_user()
    if current_user is None:
        return redirect(url_for("login_page"))
    return render_template("dashboard.html", current_user=current_user)


@app.route("/history/task/<session_id>")
def task_detail_page(session_id: str):
    current_user = get_current_user()
    if current_user is None:
        return redirect(url_for("login_page"))
    return render_template("task_detail.html", current_user=current_user, session_id=session_id)


@app.route("/api/auth/status")
def auth_status():
    current_user = get_current_user()
    return jsonify(
        {
            "success": True,
            "logged_in": current_user is not None,
            "user": current_user,
        }
    )


@app.route("/api/auth/register", methods=["POST"])
def auth_register():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    if len(username) < 3:
        return jsonify({"error": "用户名至少需要 3 个字符"}), 400
    if len(password) < 6:
        return jsonify({"error": "密码至少需要 6 个字符"}), 400

    db = get_db()
    existing_user = db.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()
    if existing_user:
        return jsonify({"error": "用户名已存在"}), 400

    cursor = db.execute(
        "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
        (username, generate_password_hash(password), now_text()),
    )
    db.commit()

    session["user_id"] = cursor.lastrowid
    session["username"] = username

    append_operation_log("register_user", username=username)
    return jsonify(
        {
            "success": True,
            "message": "注册成功",
            "user": get_current_user(),
        }
    )


@app.route("/api/auth/login", methods=["POST"])
def auth_login():
    data = request.get_json(silent=True) or {}
    username = (data.get("username") or "").strip()
    password = data.get("password") or ""

    db = get_db()
    user = db.execute(
        "SELECT id, username, password_hash FROM users WHERE username = ?",
        (username,),
    ).fetchone()

    if user is None or not check_password_hash(user["password_hash"], password):
        append_operation_log("login", status="failed", username=username, error="用户名或密码错误")
        return jsonify({"error": "用户名或密码错误"}), 400

    session["user_id"] = user["id"]
    session["username"] = user["username"]
    append_operation_log("login", username=username)

    return jsonify(
        {
            "success": True,
            "message": "登录成功",
            "user": get_current_user(),
        }
    )


@app.route("/api/auth/logout", methods=["POST"])
def auth_logout():
    current_user = get_current_user()
    if current_user:
        append_operation_log("logout", username=current_user["username"])
    session.clear()
    return jsonify({"success": True, "message": "已退出登录"})


@app.route("/api/upload", methods=["POST"])
def upload_files():
    upload_start = time.perf_counter()

    try:
        current_user = get_current_user()
        if current_user is None:
            return jsonify({"error": "请先登录"}), 401

        if "sar" not in request.files or "optical" not in request.files:
            return jsonify({"error": "请上传 SAR 和光学影像文件"}), 400

        sar_file = request.files["sar"]
        optical_file = request.files["optical"]

        if sar_file.filename == "" or optical_file.filename == "":
            return jsonify({"error": "请选择文件"}), 400

        session_id = str(uuid.uuid4())
        session_dir = UPLOAD_FOLDER / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        sar_safe_name = secure_filename(sar_file.filename) or "sar_input"
        optical_safe_name = secure_filename(optical_file.filename) or "optical_input"
        sar_path = session_dir / f"sar_{sar_safe_name}"
        optical_path = session_dir / f"optical_{optical_safe_name}"

        sar_file.save(str(sar_path))
        optical_file.save(str(optical_path))

        upload_ms = round((time.perf_counter() - upload_start) * 1000, 2)
        metadata = {
            "session_id": session_id,
            "created_at": now_text(),
            "user": current_user,
            "files": {
                "sar_name": sar_file.filename,
                "optical_name": optical_file.filename,
                "sar_path": str(sar_path),
                "optical_path": str(optical_path),
                "sar_size_bytes": sar_path.stat().st_size,
                "optical_size_bytes": optical_path.stat().st_size,
            },
            "timings": {
                "upload_ms": upload_ms,
            },
        }
        save_session_metadata(session_dir, metadata)

        append_operation_log(
            "upload",
            session_id=session_id,
            user=metadata["user"],
            files=metadata["files"],
            timings=metadata["timings"],
        )

        return jsonify(
            {
                "success": True,
                "session_id": session_id,
                "sar_path": str(sar_path),
                "optical_path": str(optical_path),
                "timings": metadata["timings"],
            }
        )
    except Exception as error:
        append_operation_log("upload", status="failed", error=str(error))
        return jsonify({"error": str(error)}), 500


@app.route("/api/register", methods=["POST"])
def register_images():
    return diffusion_register_images()

    session_id = None

    try:
        current_user = get_current_user()
        if current_user is None:
            return jsonify({"error": "请先登录"}), 401

        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id")
        selected_model_name = data.get("model_name")
        checkerboard_cells = int(data.get("checkerboard_cells", DEFAULT_CHECKERBOARD_CELLS))
        checkerboard_cells = max(2, min(checkerboard_cells, 20))

        if not session_id:
            return jsonify({"error": "缺少 session_id"}), 400

        session_dir = UPLOAD_FOLDER / session_id
        if not session_dir.exists():
            return jsonify({"error": "无效的 session_id"}), 400

        if not user_owns_session(session_id, current_user):
            return jsonify({"error": "无权访问该任务"}), 403

        sar_path, optical_path = find_uploaded_files(session_dir)
        if not sar_path or not optical_path:
            files = [file.name for file in session_dir.iterdir() if file.is_file()]
            return jsonify({"error": f"未找到有效文件，当前文件：{files}"}), 400

        result_dir = RESULTS_FOLDER / session_id
        result_dir.mkdir(parents=True, exist_ok=True)

        metadata = load_session_metadata(session_dir)
        runtime = get_cnn_runtime()
        if not runtime.get("available"):
            return jsonify({"error": f"CNN 当前不可用：{runtime.get('error')}"}), 400

        selected_model_path = resolve_model_path(selected_model_name)
        result_payload = register_cnn(
            str(sar_path),
            str(optical_path),
            str(result_dir),
            model_path=selected_model_path,
            checkerboard_cells=checkerboard_cells,
        )

        metrics = result_payload["metrics"]
        timings = result_payload["timings"]
        actual_model_name = result_payload.get("model_info", {}).get("model_name")

        metadata["last_registration"] = {
            "finished_at": now_text(),
            "method": result_payload["method"],
            "model_name": actual_model_name,
            "checkerboard_cells": checkerboard_cells,
            "timings": timings,
            "metrics": metrics,
        }
        save_session_metadata(session_dir, metadata)

        save_task_record(
            session_id=session_id,
            status="success",
            method=result_payload["method"],
            model_name=actual_model_name,
            timings={
                "upload_ms": metadata.get("timings", {}).get("upload_ms"),
                **timings,
            },
            metrics=metrics,
            files=metadata.get("files", {}),
        )

        log_entry = append_operation_log(
            "register",
            session_id=session_id,
            method=result_payload["method"],
            model_name=actual_model_name,
            user=metadata.get("user"),
            files=metadata.get("files", {}),
            timings={
                "upload_ms": metadata.get("timings", {}).get("upload_ms"),
                **timings,
            },
            metrics=metrics,
        )

        return jsonify(
            {
                "success": True,
                "results": {
                    "method": result_payload["method"],
                    "model_name": actual_model_name,
                    "metrics": metrics,
                    "checkerboard_cells": checkerboard_cells,
                },
                "timings": {
                    "upload_ms": metadata.get("timings", {}).get("upload_ms"),
                    **timings,
                },
                "model_info": result_payload.get("model_info"),
                "checkerboard_url": f"/api/results/{session_id}/checkerboard.png",
                "overlay_url": f"/api/results/{session_id}/false_color_overlay.png",
                "difference_url": f"/api/results/{session_id}/difference_map.png",
                "contour_url": f"/api/results/{session_id}/contour_overlay.png",
                "deformation_url": (
                    f"/api/results/{session_id}/deformation_heatmap.png"
                    if (result_dir / "deformation_heatmap.png").exists()
                    else None
                ),
                "registered_url": f"/api/download/{session_id}/registered_output.tif",
                "log_entry": log_entry,
            }
        )
    except Exception as error:
        print(f"配准错误: {error}")
        print(traceback.format_exc())
        if session_id:
            save_task_record(session_id=session_id, status="failed")
        append_operation_log("register", status="failed", session_id=session_id, error=str(error))
        return jsonify({"error": str(error)}), 500


@app.route("/api/diffusion-register", methods=["POST"])
def diffusion_register_images():
    session_id = None

    try:
        current_user = get_current_user()
        if current_user is None:
            return jsonify({"error": "请先登录"}), 401

        data = request.get_json(silent=True) or {}
        session_id = data.get("session_id")
        steps = int(data.get("steps", DIFFUSION_DEFAULT_STEPS))
        max_keypoints = int(data.get("max_keypoints", DIFFUSION_DEFAULT_MAX_KEYPOINTS))
        extractor_policy = str(data.get("extractor_policy", DIFFUSION_DEFAULT_EXTRACTOR_POLICY))
        extractors = data.get("extractors", DIFFUSION_DEFAULT_EXTRACTORS)
        if isinstance(extractors, str):
            extractors = extractors.split()
        match_preprocess = str(data.get("match_preprocess", DIFFUSION_DEFAULT_MATCH_PREPROCESS))
        resize_mode = str(data.get("resize_mode", DIFFUSION_DEFAULT_RESIZE_MODE))

        if not session_id:
            return jsonify({"error": "缺少 session_id"}), 400

        session_dir = UPLOAD_FOLDER / session_id
        if not session_dir.exists():
            return jsonify({"error": "无效的 session_id"}), 400

        if not user_owns_session(session_id, current_user):
            return jsonify({"error": "无权访问该任务"}), 403

        sar_path, optical_path = find_uploaded_files(session_dir)
        if not sar_path or not optical_path:
            files = [file.name for file in session_dir.iterdir() if file.is_file()]
            return jsonify({"error": f"未找到有效文件，当前文件：{files}"}), 400

        result_dir = RESULTS_FOLDER / session_id
        result_dir.mkdir(parents=True, exist_ok=True)

        metadata = load_session_metadata(session_dir)
        result_payload = register_diffusion_lightglue(
            str(sar_path),
            str(optical_path),
            str(result_dir),
            steps=steps,
            max_keypoints=max_keypoints,
            extractor_policy=extractor_policy,
            extractors=extractors,
            match_preprocess=match_preprocess,
            resize_mode=resize_mode,
        )

        metrics = result_payload["metrics"]
        timings = {
            "upload_ms": metadata.get("timings", {}).get("upload_ms"),
            **result_payload.get("timings", {}),
        }
        model_info = result_payload.get("model_info", {})
        model_name = f"ACD_LCM_ADV:{Path(model_info.get('generator_checkpoint', DIFFUSION_BASELINE_CHECKPOINT)).name}+LightGlue"

        metadata["last_registration"] = {
            "finished_at": now_text(),
            "method": result_payload["method"],
            "model_name": model_name,
            "timings": timings,
            "metrics": metrics,
            "artifacts": result_payload.get("artifacts", {}),
        }
        save_session_metadata(session_dir, metadata)

        save_task_record(
            session_id=session_id,
            status="success",
            method=result_payload["method"],
            model_name=model_name,
            timings=timings,
            metrics=metrics,
            files=metadata.get("files", {}),
        )

        log_entry = append_operation_log(
            "diffusion_register",
            session_id=session_id,
            method=result_payload["method"],
            model_name=model_name,
            user=metadata.get("user"),
            files=metadata.get("files", {}),
            timings=timings,
            metrics=metrics,
        )

        return jsonify(
            {
                "success": True,
                "results": {
                    "method": result_payload["method"],
                    "model_name": model_name,
                    "metrics": metrics,
                },
                "timings": timings,
                "model_info": model_info,
                "checkerboard_url": (
                    f"/api/results/{session_id}/checkerboard_original.png"
                    if (result_dir / "checkerboard_original.png").exists()
                    else f"/api/results/{session_id}/checkerboard.png"
                ),
                "overlay_url": (
                    f"/api/results/{session_id}/false_color_overlay_original.png"
                    if (result_dir / "false_color_overlay_original.png").exists()
                    else f"/api/results/{session_id}/false_color_overlay.png"
                ),
                "difference_url": (
                    f"/api/results/{session_id}/difference_map_original.png"
                    if (result_dir / "difference_map_original.png").exists()
                    else f"/api/results/{session_id}/difference_map.png"
                ),
                "contour_url": (
                    f"/api/results/{session_id}/contour_overlay_original.png"
                    if (result_dir / "contour_overlay_original.png").exists()
                    else f"/api/results/{session_id}/contour_overlay.png"
                ),
                "fake_optical_url": f"/api/results/{session_id}/fake_optical.png",
                "sar_condition_url": f"/api/results/{session_id}/sar_condition.png",
                "real_optical_url": f"/api/results/{session_id}/real_optical_resized.png",
                "match_url": f"/api/results/{session_id}/lightglue_matches.png",
                "sar_transfer_match_url": f"/api/results/{session_id}/sar_transferred_matches.png",
                "optical_points_url": f"/api/results/{session_id}/optical_transferred_points.png",
                "fake_registered_url": f"/api/results/{session_id}/fake_optical_registered.png",
                "sar_registered_url": (
                    f"/api/results/{session_id}/sar_registered_original.png"
                    if (result_dir / "sar_registered_original.png").exists()
                    else f"/api/results/{session_id}/sar_registered.png"
                ),
                "registered_preview_url": (
                    f"/api/results/{session_id}/sar_registered_original.png"
                    if (result_dir / "sar_registered_original.png").exists()
                    else f"/api/results/{session_id}/sar_registered.png"
                ),
                "registered_url": (
                    f"/api/download/{session_id}/registered_output_original.png"
                    if (result_dir / "registered_output_original.png").exists()
                    else f"/api/download/{session_id}/registered_output.png"
                ),
                "log_entry": log_entry,
            }
        )
    except Exception as error:
        print(f"扩散 + LightGlue 配准错误: {error}")
        print(traceback.format_exc())
        if session_id:
            save_task_record(session_id=session_id, status="failed")
        append_operation_log("diffusion_register", status="failed", session_id=session_id, error=str(error))
        return jsonify({"error": str(error)}), 500


@app.route("/api/results/<session_id>/<filename>")
def get_result(session_id: str, filename: str):
    current_user = get_current_user()
    if current_user is None:
        return jsonify({"error": "请先登录"}), 401
    if not user_owns_session(session_id, current_user):
        return jsonify({"error": "无权访问该任务结果"}), 403

    result_path = RESULTS_FOLDER / session_id / filename
    if not result_path.exists():
        return jsonify({"error": "文件不存在"}), 404
    return send_file(str(result_path))


@app.route("/api/download/<session_id>/<filename>")
def download_result(session_id: str, filename: str):
    current_user = get_current_user()
    if current_user is None:
        return jsonify({"error": "请先登录"}), 401
    if not user_owns_session(session_id, current_user):
        return jsonify({"error": "无权下载该任务结果"}), 403

    result_path = RESULTS_FOLDER / session_id / filename
    if not result_path.exists():
        return jsonify({"error": "文件不存在"}), 404

    append_operation_log(
        "download",
        session_id=session_id,
        user=current_user,
        filename=filename,
        path=str(result_path),
    )
    return send_file(str(result_path), as_attachment=True, download_name=filename)


@app.route("/api/logs")
def get_logs():
    current_user = get_current_user()
    if current_user is None:
        return jsonify({"error": "请先登录"}), 401

    try:
        limit = int(request.args.get("limit", 8))
    except ValueError:
        limit = 8

    limit = max(1, min(limit, 50))
    return jsonify({"success": True, "logs": read_recent_logs(limit, current_user=current_user)})


@app.route("/api/history")
def get_history():
    current_user = get_current_user()
    if current_user is None:
        return jsonify({"success": True, "logged_in": False, "history": []})

    try:
        limit = int(request.args.get("limit", 12))
    except ValueError:
        limit = 12
    limit = max(1, min(limit, 30))

    return jsonify(
        {
            "success": True,
            "logged_in": True,
            "user": current_user,
            "summary": query_task_summary(),
            "history": query_task_history(limit),
        }
    )


@app.route("/api/history/<session_id>")
def get_history_detail(session_id: str):
    current_user = get_current_user()
    if current_user is None:
        return jsonify({"error": "请先登录"}), 401

    detail = query_task_detail(session_id)
    if detail is None:
        return jsonify({"error": "未找到对应历史任务"}), 404

    return jsonify(
        {
            "success": True,
            "logged_in": True,
            "user": current_user,
            "task": detail,
        }
    )


@app.route("/api/diffusion-baseline/status")
def diffusion_baseline_status():
    current_user = get_current_user()
    if current_user is None:
        return jsonify({"error": "Please login first"}), 401

    metrics_path = DIFFUSION_BASELINE_DIR / "lightglue_eval.csv"
    contact_sheet_path = DIFFUSION_BASELINE_DIR / "contact_sheet.png"
    manifest_path = DIFFUSION_BASELINE_DIR / "manifest.csv"
    metrics = summarize_lightglue_csv(metrics_path)

    return jsonify(
        {
            "success": True,
            "generator": {
                "name": "ACD_S2ODPM LCM_ADV pretrained generator",
                "checkpoint_path": str(DIFFUSION_BASELINE_CHECKPOINT),
                "checkpoint_exists": DIFFUSION_BASELINE_CHECKPOINT.exists(),
                "decision": "Use this pretrained generator as the current baseline; stop the short fine-tuning branch.",
            },
            "pipeline": [
                {
                    "step": "SAR -> Fake Optical",
                    "detail": "Sentinel-1 SAR is translated by the pretrained LCM_ADV generator.",
                },
                {
                    "step": "Fake Optical + Real Optical",
                    "detail": "SuperPoint extracts keypoints and LightGlue performs matching.",
                },
                {
                    "step": "RANSAC / Homography",
                    "detail": "Match count, confidence score, inliers and inlier ratio are used for registration quality checks.",
                },
            ],
            "artifacts": {
                "baseline_dir": str(DIFFUSION_BASELINE_DIR),
                "baseline_dir_exists": DIFFUSION_BASELINE_DIR.exists(),
                "contact_sheet_exists": contact_sheet_path.exists(),
                "metrics_csv_exists": metrics_path.exists(),
                "manifest_csv_exists": manifest_path.exists(),
                "contact_sheet_url": "/api/diffusion-baseline/artifact/contact_sheet" if contact_sheet_path.exists() else None,
                "metrics_csv_url": "/api/diffusion-baseline/artifact/metrics_csv" if metrics_path.exists() else None,
                "manifest_csv_url": "/api/diffusion-baseline/artifact/manifest_csv" if manifest_path.exists() else None,
                "fake_optical_count": count_image_files(DIFFUSION_BASELINE_DIR / "gen_opt"),
                "real_optical_count": count_image_files(DIFFUSION_BASELINE_DIR / "gt_opt"),
                "sar_condition_count": count_image_files(DIFFUSION_BASELINE_DIR / "cond_sar"),
            },
            "lightglue": metrics,
        }
    )


@app.route("/api/diffusion-baseline/artifact/<artifact_name>")
def diffusion_baseline_artifact(artifact_name: str):
    current_user = get_current_user()
    if current_user is None:
        return jsonify({"error": "Please login first"}), 401

    filename = DIFFUSION_BASELINE_ARTIFACTS.get(artifact_name)
    if filename is None:
        return jsonify({"error": "Unknown artifact"}), 404

    artifact_path = DIFFUSION_BASELINE_DIR / filename
    if not artifact_path.exists():
        return jsonify({"error": "Artifact not found"}), 404

    return send_file(
        str(artifact_path),
        as_attachment=artifact_path.suffix.lower() == ".csv",
        download_name=artifact_path.name,
    )


@app.route("/api/demo-samples")
def demo_samples():
    current_user = get_current_user()
    if current_user is None:
        return jsonify({"error": "Please login first"}), 401

    manifest = load_demo_manifest()
    samples = []
    for sample in manifest.get("samples", []):
        metrics = sample.get("metrics", {})
        samples.append(
            {
                "id": sample.get("id"),
                "title": sample.get("title"),
                "season": sample.get("season"),
                "stem": sample.get("stem"),
                "purpose": sample.get("purpose"),
                "tag": sample.get("tag"),
                "primary": bool(sample.get("primary", False)),
                "metrics": {
                    "inliers": metrics.get("inliers"),
                    "inlier_ratio": metrics.get("inlier_ratio"),
                    "rmse": metrics.get("rmse"),
                    "selected_extractor": metrics.get("selected_extractor"),
                    "cascade_rescued": metrics.get("cascade_rescued"),
                    "registration_reliable": metrics.get("registration_reliable"),
                },
            }
        )

    return jsonify(
        {
            "success": True,
            "default_sample_id": manifest.get("default_sample_id"),
            "samples": samples,
        }
    )


@app.route("/api/demo-samples/<sample_id>")
def demo_sample_detail(sample_id: str):
    current_user = get_current_user()
    if current_user is None:
        return jsonify({"error": "Please login first"}), 401

    manifest = load_demo_manifest()
    sample = next((item for item in manifest.get("samples", []) if item.get("id") == sample_id), None)
    if sample is None:
        return jsonify({"error": "Demo sample not found"}), 404

    return jsonify({"success": True, "task": build_demo_task(sample)})


@app.route("/api/experiment-summary")
def experiment_summary():
    current_user = get_current_user()
    if current_user is None:
        return jsonify({"error": "Please login first"}), 401

    best_params_path = DIFFUSION_CASCADE_STUDY_DIR / "best_params.json"
    summary_csv_path = DIFFUSION_CASCADE_STUDY_DIR / "summary_by_params.csv"
    all_runs_csv_path = DIFFUSION_CASCADE_STUDY_DIR / "all_runs.csv"
    paper_table_path = DIFFUSION_CASCADE_STUDY_DIR / "paper_table.md"
    filter_summary_path = DIFFUSION_PARAM_SWEEP_DIR / "filter_summary.json"
    contact_summary_path = DIFFUSION_CASCADE_STUDY_DIR / "failure_contact_sheets" / "failure_contact_sheet_summary.json"

    best_payload = safe_json_load(best_params_path, {})
    strategy_rows = read_csv_dicts(summary_csv_path)
    filter_summary = safe_json_load(filter_summary_path, {})
    contact_summary = safe_json_load(contact_summary_path, {})

    if not best_payload and strategy_rows:
        best_payload = {
            "best_params": {
                "matcher_strategy": strategy_rows[0].get("matcher_strategy"),
                "extractor_policy": strategy_rows[0].get("extractor_policy"),
                "match_preprocess": strategy_rows[0].get("match_preprocess"),
                "steps": strategy_rows[0].get("steps"),
                "max_keypoints": strategy_rows[0].get("max_keypoints"),
            },
            "best_summary": strategy_rows[0],
        }

    return jsonify(
        {
            "success": True,
            "study": {
                "name": "Diffusion + LightGlue cascade study",
                "generator_checkpoint": str(DIFFUSION_BASELINE_CHECKPOINT),
                "study_dir": str(DIFFUSION_CASCADE_STUDY_DIR),
                "param_sweep_dir": str(DIFFUSION_PARAM_SWEEP_DIR),
                "summary_csv_exists": summary_csv_path.exists(),
                "best_params_exists": best_params_path.exists(),
                "filter_summary_exists": filter_summary_path.exists(),
            },
            "best_params": best_payload.get("best_params", {}),
            "best_summary": best_payload.get("best_summary", {}),
            "selection_rule": best_payload.get(
                "selection_rule",
                "max success_rate, then min median_rmse_reliable, then max median_inlier_ratio, then min median_total_ms",
            ),
            "strategy_rows": strategy_rows,
            "filter_summary": filter_summary,
            "contact_sheet": contact_summary,
            "artifacts": {
                "summary_csv_url": artifact_url("cascade_study", "summary_by_params.csv"),
                "all_runs_csv_url": artifact_url("cascade_study", "all_runs.csv"),
                "paper_table_url": artifact_url("cascade_study", "paper_table.md") if paper_table_path.exists() else None,
                "failures_all_url": artifact_url("cascade_study", "failure_contact_sheets/failures_all.png"),
                "failures_all_preview_url": static_url_if_exists(
                    "demo_samples/experiment_artifacts/boundary_cases_preview.png"
                ),
                "cascade_rescued_url": artifact_url("cascade_study", "failure_contact_sheets/cascade_rescued.png"),
                "cascade_rescued_preview_url": static_url_if_exists(
                    "demo_samples/experiment_artifacts/cascade_rescued_preview.png"
                ),
                "few_inliers_url": artifact_url("cascade_study", "failure_contact_sheets/failures_reason_few_inliers.png"),
                "low_inlier_ratio_url": artifact_url(
                    "cascade_study",
                    "failure_contact_sheets/failures_reason_low_inlier_ratio.png",
                ),
                "bad_homography_url": artifact_url(
                    "cascade_study",
                    "failure_contact_sheets/failures_reason_bad_homography_shape.png",
                ),
                "filtered_pairs_url": artifact_url("param_sweep", "filtered_pairs.csv"),
                "rejected_pairs_url": artifact_url("param_sweep", "rejected_pairs.csv"),
                "filter_summary_url": artifact_url("param_sweep", "filter_summary.json"),
            },
        }
    )


@app.route("/api/experiment-artifact/<root_key>/<path:relative_path>")
def experiment_artifact(root_key: str, relative_path: str):
    current_user = get_current_user()
    if current_user is None:
        return jsonify({"error": "Please login first"}), 401

    root = EXPERIMENT_ARTIFACT_ROOTS.get(root_key)
    if root is None:
        return jsonify({"error": "Unknown artifact root"}), 404

    try:
        root_resolved = root.resolve()
        requested = (root / relative_path).resolve()
        requested.relative_to(root_resolved)
    except Exception:
        return jsonify({"error": "Invalid artifact path"}), 400

    if not requested.exists() or not requested.is_file():
        return jsonify({"error": "Artifact not found"}), 404

    return send_file(
        str(requested),
        as_attachment=requested.suffix.lower() in {".csv", ".json", ".md"},
        download_name=requested.name,
    )


@app.route("/api/model/status")
def model_status():
    diffusion_runtime = get_diffusion_runtime()
    diffusion_available = diffusion_runtime.get("available", False)
    default_model = "ACD_LCM_ADV:model.safetensors"
    return jsonify(
        {
            "model_loaded": diffusion_available,
            "device": diffusion_runtime.get("device", "cpu"),
            "method": "Diffusion + LightGlue",
            "description": "Web 端当前统一使用 ACD_S2ODPM 预训练扩散生成器 + SuperPoint/LightGlue 配准链路。",
            "cnn_available": False,
            "cnn_model_name": None,
            "cnn_error": "CNN route disabled; this app now uses Diffusion + LightGlue only.",
            "cnn_warnings": [],
            "torch_version": diffusion_runtime.get("torch_version"),
            "diffusion_available": diffusion_available,
            "diffusion_device": diffusion_runtime.get("device", "cpu"),
            "diffusion_error": diffusion_runtime.get("error"),
            "diffusion_warnings": diffusion_runtime.get("warnings", []),
            "diffusion_runtime_python": diffusion_runtime.get("python_path"),
            "diffusion_torch_version": diffusion_runtime.get("torch_version"),
            "diffusion_diffusers_version": diffusion_runtime.get("diffusers_version"),
            "diffusion_generator_checkpoint": str(DIFFUSION_BASELINE_CHECKPOINT),
            "diffusion_generator_exists": diffusion_runtime.get("checkpoint_exists", False),
            "diffusion_worker_exists": diffusion_runtime.get("worker_exists", False),
            "diffusion_superpoint_cache_exists": diffusion_runtime.get("superpoint_cache_exists", False),
            "diffusion_lightglue_cache_exists": diffusion_runtime.get("lightglue_cache_exists", False),
            "supported_models": [default_model],
            "supported_model_options": [
                {
                    "name": default_model,
                    "label": default_model,
                    "profile": {
                        "family": "diffusion_lightglue",
                        "display_name": "ACD_S2ODPM LCM_ADV + LightGlue",
                        "status_tag": "baseline",
                        "experimental": False,
                    },
                }
            ],
            "default_model": default_model,
            "default_model_profile": {
                "family": "diffusion_lightglue",
                "display_name": "ACD_S2ODPM LCM_ADV + LightGlue",
                "status_tag": "baseline",
                "experimental": False,
            },
            "supported_model_profiles": {
                default_model: {
                    "family": "diffusion_lightglue",
                    "display_name": "ACD_S2ODPM LCM_ADV + LightGlue",
                    "status_tag": "baseline",
                    "experimental": False,
                }
            },
            "supported_outputs": [
                "checkerboard",
                "false_color_overlay",
                "difference_map",
                "contour_overlay",
                "deformation_heatmap",
                "fake_optical",
                "lightglue_matches",
                "fake_optical_registered",
            ],
        }
    )


with app.app_context():
    init_db()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("SAR-光学影像配准 Web 系统")
    print("当前算法: ORB 特征点匹配")
    print("新增模块: 日志记录 / 耗时统计 / 差分图 / 轮廓叠加图")
    print("=" * 60)
    print("访问地址: http://localhost:5000")
    print("=" * 60 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
