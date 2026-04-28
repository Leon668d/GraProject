from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from torchvision import transforms

from diffusers import LCMScheduler


WEB_APP_DIR = Path(__file__).resolve().parents[1]
CODEX_ROOT = WEB_APP_DIR.parents[2]
ACD_ROOT = CODEX_ROOT / "ACD_S2ODPM"
LIGHTGLUE_ROOT = CODEX_ROOT / "LightGlue"
TORCH_CACHE_DIR = WEB_APP_DIR / "runtime_cache" / "torch"
TORCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("TORCH_HOME", str(TORCH_CACHE_DIR))

for repo_path in (ACD_ROOT, LIGHTGLUE_ROOT):
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

from models import SAR2OptUNetv3  # noqa: E402
from lightglue import ALIKED, DISK, LightGlue, SuperPoint  # noqa: E402
from lightglue.utils import load_image, rbd  # noqa: E402

MODEL_SIZE = 256


def now_ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 2)


# 生成器权重以 safetensors 形式保存，因此这里显式读取权重文件，
# 这样后续推理流程只依赖 PyTorch 张量，不依赖训练时的额外封装。
def safe_load(model_path: Path) -> dict:
    state_dict = {}
    with safe_open(str(model_path), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            state_dict[key] = handle.get_tensor(key)
    return state_dict


# 这里构建的是扩散生成器的固定网络结构，必须与预训练 ACD 权重严格一致；
# Web 端只负责推理，不在运行时动态改网络配置。
def build_model() -> SAR2OptUNetv3:
    return SAR2OptUNetv3(
        sample_size=256,
        in_channels=4,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )


def resolve_checkpoint(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [
        path / "model.safetensors",
        path / "diffusion_pytorch_model.safetensors",
        path / "model_1.safetensors",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No generator checkpoint found at {path}")


# 所有输入影像都会先压到统一的 8-bit 可视化空间，
# 这样 tif/npy 等不同来源的数据都能走同一套预处理、推理和网页展示流程。
def normalize_to_uint8(array: np.ndarray, *, log_sar: bool = False) -> np.ndarray:
    data = np.asarray(array)
    if data.ndim == 3 and data.shape[0] <= 8 and data.shape[0] < data.shape[-1]:
        data = np.moveaxis(data, 0, -1)
    if data.ndim == 3 and data.shape[-1] == 1:
        data = data[..., 0]
    data = data.astype(np.float32, copy=False)
    if log_sar:
        positive = data > 0
        if positive.any():
            data = np.where(positive, 10.0 * np.log10(np.maximum(data, 1e-6)), data)
    finite = np.isfinite(data)
    if not finite.any():
        return np.zeros(data.shape[:2], dtype=np.uint8)
    valid = data[finite]
    lo, hi = np.percentile(valid, [2, 98])
    if not math.isfinite(float(lo)) or not math.isfinite(float(hi)) or hi <= lo:
        lo, hi = float(valid.min()), float(valid.max())
    if hi <= lo:
        return np.zeros(data.shape[:2], dtype=np.uint8)
    scaled = (np.clip(data, lo, hi) - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)


# 这个辅助函数屏蔽 tif/png/jpg/npy 之间的格式差异，
# 始终返回下游 transforms 期望的 PIL 图像格式。
def read_any_image(path: Path, *, mode: str, log_sar: bool = False) -> Image.Image:
    if path.suffix.lower() == ".npy":
        array = np.load(path)
    else:
        try:
            with Image.open(path) as image:
                if mode == "L":
                    return image.convert("L")
                return image.convert("RGB")
        except Exception:
            array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if array is None:
                raise
            if array.ndim == 3:
                array = cv2.cvtColor(array, cv2.COLOR_BGR2RGB)

    if mode == "L":
        if array.ndim == 3:
            if array.shape[0] <= 8 and array.shape[0] < array.shape[-1]:
                array = np.moveaxis(array, 0, -1)
            array = array[..., 0]
        return Image.fromarray(normalize_to_uint8(array, log_sar=log_sar)).convert("L")

    if array.ndim == 2:
        rgb = np.repeat(normalize_to_uint8(array)[..., None], 3, axis=2)
    else:
        if array.shape[0] <= 8 and array.shape[0] < array.shape[-1]:
            array = np.moveaxis(array, 0, -1)
        rgb = normalize_to_uint8(array[..., :3])
        if rgb.ndim == 2:
            rgb = np.repeat(rgb[..., None], 3, axis=2)
    return Image.fromarray(rgb).convert("RGB")


def tensor_to_rgb_uint8(tensor: torch.Tensor) -> np.ndarray:
    image = tensor.detach().float().cpu()[0]
    image = ((image + 1.0) / 2.0).clamp(0, 1)
    image = image.permute(1, 2, 0).numpy()
    return np.clip(image * 255.0, 0, 255).astype(np.uint8)


def transform_matrix(meta: dict) -> np.ndarray:
    return np.array(
        [
            [float(meta["scale_x"]), 0.0, float(meta["pad_x"])],
            [0.0, float(meta["scale_y"]), float(meta["pad_y"])],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def stretch_image(image: Image.Image, *, size: int, mode: str) -> tuple[Image.Image, dict]:
    width, height = image.size
    resized = image.resize((size, size), Image.BILINEAR).convert(mode)
    meta = {
        "mode": "stretch",
        "original_width": width,
        "original_height": height,
        "canvas_width": size,
        "canvas_height": size,
        "scale_x": size / max(float(width), 1.0),
        "scale_y": size / max(float(height), 1.0),
        "pad_x": 0.0,
        "pad_y": 0.0,
        "content_box": [0.0, 0.0, float(size), float(size)],
    }
    return resized, meta


def letterbox_image(
    image: Image.Image,
    *,
    size: int,
    mode: str,
    fill: int | tuple[int, int, int] = 0,
) -> tuple[Image.Image, dict]:
    width, height = image.size
    scale = min(size / max(float(width), 1.0), size / max(float(height), 1.0))
    resized_width = max(1, min(size, int(round(width * scale))))
    resized_height = max(1, min(size, int(round(height * scale))))
    pad_x = (size - resized_width) // 2
    pad_y = (size - resized_height) // 2
    resized = image.resize((resized_width, resized_height), Image.BILINEAR).convert(mode)
    canvas = Image.new(mode, (size, size), fill)
    canvas.paste(resized, (pad_x, pad_y))
    meta = {
        "mode": "letterbox",
        "original_width": width,
        "original_height": height,
        "canvas_width": size,
        "canvas_height": size,
        "scale_x": scale,
        "scale_y": scale,
        "pad_x": float(pad_x),
        "pad_y": float(pad_y),
        "content_box": [float(pad_x), float(pad_y), float(pad_x + resized_width), float(pad_y + resized_height)],
    }
    return canvas, meta


def prepare_input_canvas(image: Image.Image, *, mode: str, resize_mode: str) -> tuple[Image.Image, dict]:
    if resize_mode == "letterbox":
        fill = (0, 0, 0) if mode == "RGB" else 0
        return letterbox_image(image, size=MODEL_SIZE, mode=mode, fill=fill)
    return stretch_image(image, size=MODEL_SIZE, mode=mode)


def points_inside_content(points: np.ndarray, meta: dict) -> np.ndarray:
    if not len(points):
        return np.zeros((0,), dtype=bool)
    x0, y0, x1, y1 = [float(value) for value in meta.get("content_box", [0.0, 0.0, MODEL_SIZE, MODEL_SIZE])]
    return (points[:, 0] >= x0) & (points[:, 0] < x1) & (points[:, 1] >= y0) & (points[:, 1] < y1)


def save_rgb(path: Path, image: np.ndarray) -> None:
    Image.fromarray(image.astype(np.uint8)).save(path)


def to_gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)


def checkerboard(image_a: np.ndarray, image_b: np.ndarray, cells: int = 8) -> np.ndarray:
    height, width = image_a.shape[:2]
    output = image_a.copy()
    cell_h = max(1, height // cells)
    cell_w = max(1, width // cells)
    for row in range(cells):
        for col in range(cells):
            if (row + col) % 2 == 1:
                y0, y1 = row * cell_h, height if row == cells - 1 else (row + 1) * cell_h
                x0, x1 = col * cell_w, width if col == cells - 1 else (col + 1) * cell_w
                output[y0:y1, x0:x1] = image_b[y0:y1, x0:x1]
    return output


def save_visualizations(registered: np.ndarray, optical: np.ndarray, result_dir: Path, *, suffix: str = "") -> dict:
    result_dir.mkdir(parents=True, exist_ok=True)
    registered_gray = to_gray(registered)
    optical_gray = to_gray(optical)
    diff = cv2.absdiff(registered_gray, optical_gray)
    diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_MAGMA)
    diff_color = cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB)

    overlay = np.zeros((*registered_gray.shape, 3), dtype=np.uint8)
    overlay[..., 0] = registered_gray
    overlay[..., 1] = optical_gray
    overlay[..., 2] = ((registered_gray.astype(np.uint16) + optical_gray.astype(np.uint16)) // 2).astype(np.uint8)

    contour = optical.copy()
    registered_edges = cv2.Canny(registered_gray, 80, 160)
    optical_edges = cv2.Canny(optical_gray, 80, 160)
    contour[registered_edges > 0] = np.array([230, 65, 55], dtype=np.uint8)
    contour[optical_edges > 0] = np.array([30, 170, 105], dtype=np.uint8)

    paths = {
        "checkerboard": result_dir / f"checkerboard{suffix}.png",
        "overlay": result_dir / f"false_color_overlay{suffix}.png",
        "difference": result_dir / f"difference_map{suffix}.png",
        "contour": result_dir / f"contour_overlay{suffix}.png",
    }
    save_rgb(paths["checkerboard"], checkerboard(registered, optical))
    save_rgb(paths["overlay"], overlay)
    save_rgb(paths["difference"], diff_color)
    save_rgb(paths["contour"], contour)
    return {
        "difference_mean": round(float(diff.mean()), 4),
        "paths": {key: str(value) for key, value in paths.items()},
    }


def draw_matches(
    fake: np.ndarray,
    optical: np.ndarray,
    points0: np.ndarray,
    points1: np.ndarray,
    inlier_mask: np.ndarray | None,
    output_path: Path,
    max_lines: int = 120,
) -> None:
    height = max(fake.shape[0], optical.shape[0])
    width = fake.shape[1] + optical.shape[1]
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    canvas[: fake.shape[0], : fake.shape[1]] = fake
    canvas[: optical.shape[0], fake.shape[1] :] = optical

    if points0.size:
        count = min(len(points0), max_lines)
        if inlier_mask is not None and len(inlier_mask) == len(points0):
            order = np.argsort(-inlier_mask.astype(np.int32))
        else:
            order = np.arange(len(points0))
        for idx in order[:count]:
            p0 = tuple(np.round(points0[idx]).astype(int))
            p1_raw = np.round(points1[idx]).astype(int)
            p1 = (int(p1_raw[0] + fake.shape[1]), int(p1_raw[1]))
            is_inlier = bool(inlier_mask[idx]) if inlier_mask is not None and idx < len(inlier_mask) else False
            color = (40, 190, 110) if is_inlier else (238, 142, 45)
            cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, p0, 3, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, p1, 3, color, -1, cv2.LINE_AA)

    save_rgb(output_path, canvas)


def draw_points_on_image(
    image: np.ndarray,
    points: np.ndarray,
    inlier_mask: np.ndarray | None,
    output_path: Path,
    max_points: int = 200,
) -> None:
    canvas = image.copy()
    if points.size:
        if inlier_mask is not None and len(inlier_mask) == len(points):
            order = np.argsort(-inlier_mask.astype(np.int32))
        else:
            order = np.arange(len(points))
        for idx in order[: min(len(points), max_points)]:
            point = tuple(np.round(points[idx]).astype(int))
            is_inlier = bool(inlier_mask[idx]) if inlier_mask is not None and idx < len(inlier_mask) else False
            color = (40, 190, 110) if is_inlier else (238, 142, 45)
            cv2.circle(canvas, point, 4, color, -1, cv2.LINE_AA)
            cv2.circle(canvas, point, 7, color, 1, cv2.LINE_AA)
    save_rgb(output_path, canvas)


# 第 1 阶段：利用预训练扩散模型把 SAR 翻译成 Fake Optical。
# 输出图与 resize 后的 SAR 条件图保持像素对齐，后面才能把
# Fake Optical 上的匹配点直接当作 SAR 坐标来复用。
def run_generation(args, result_dir: Path, device: torch.device, dtype: torch.dtype) -> dict:
    start = time.perf_counter()
    checkpoint_path = resolve_checkpoint(Path(args.checkpoint))
    model = build_model()
    model.load_state_dict(safe_load(checkpoint_path), strict=True)
    model.eval().to(device=device, dtype=dtype)

    transform_sar = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    transform_opt = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    sar_image = read_any_image(Path(args.sar), mode="L", log_sar=args.sar_log)
    optical_image = read_any_image(Path(args.optical), mode="RGB")
    sar_canvas, sar_transform = prepare_input_canvas(sar_image, mode="L", resize_mode=args.resize_mode)
    optical_canvas, optical_transform = prepare_input_canvas(optical_image, mode="RGB", resize_mode=args.resize_mode)
    sar_tensor = transform_sar(sar_canvas).unsqueeze(0).to(device=device, dtype=dtype)
    optical_tensor = transform_opt(optical_canvas).unsqueeze(0)

    scheduler = LCMScheduler(num_train_timesteps=1000)
    scheduler.set_timesteps(args.steps, device=device)

    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)
    pred = torch.randn((1, 3, MODEL_SIZE, MODEL_SIZE), device=device, dtype=dtype, generator=generator)
    denoised = None
    with torch.no_grad():
        for timestep in scheduler.timesteps:
            model_input = torch.cat((pred, sar_tensor), dim=1)
            model_pred = model(model_input, timestep)
            pred, denoised = scheduler.step(model_pred, timestep, pred, return_dict=False)
    if denoised is None:
        raise RuntimeError("Diffusion scheduler produced no output")

    fake = tensor_to_rgb_uint8(denoised)
    sar_np = np.asarray(sar_canvas.convert("L"))
    sar_rgb = np.repeat(sar_np[..., None], 3, axis=2)
    optical_rgb = tensor_to_rgb_uint8(optical_tensor)
    sar_original_np = np.asarray(sar_image.convert("L"))
    sar_original_rgb = np.repeat(sar_original_np[..., None], 3, axis=2)
    optical_original_rgb = np.asarray(optical_image.convert("RGB"))

    fake_path = result_dir / "fake_optical.png"
    sar_path = result_dir / "sar_condition.png"
    optical_path = result_dir / "real_optical_resized.png"
    save_rgb(fake_path, fake)
    save_rgb(sar_path, sar_rgb)
    save_rgb(optical_path, optical_rgb)
    return {
        "checkpoint": str(checkpoint_path),
        "fake": fake,
        "sar": sar_rgb,
        "optical": optical_rgb,
        "sar_original": sar_original_rgb,
        "optical_original": optical_original_rgb,
        "sar_transform": sar_transform,
        "optical_transform": optical_transform,
        "paths": {
            "fake_optical": str(fake_path),
            "sar_condition": str(sar_path),
            "real_optical": str(optical_path),
        },
        "generation_ms": now_ms(start),
    }


# 这里把特征提取器做成可配置项，是因为实验里需要比较
# 单一提取器和 SuperPoint -> ALIKED 级联两种策略。
def build_extractor(name: str, max_keypoints: int):
    if name == "superpoint":
        return SuperPoint(max_num_keypoints=max_keypoints)
    if name == "disk":
        return DISK(max_num_keypoints=max_keypoints)
    if name == "aliked":
        return ALIKED(max_num_keypoints=max_keypoints)
    raise ValueError(f"Unsupported extractor: {name}")


# 结构增强模式会压低 Fake Optical 的颜色和风格噪声，
# 在 RGB 直接匹配不稳定时，突出边缘和几何结构。
def enhance_structure(image: np.ndarray) -> np.ndarray:
    gray = to_gray(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(gray)
    blurred = cv2.GaussianBlur(contrast, (0, 0), 1.0)
    sharpened = cv2.addWeighted(contrast, 1.45, blurred, -0.45, 0)
    edges = cv2.Canny(sharpened, 60, 140)
    enhanced = cv2.addWeighted(sharpened, 0.85, edges, 0.15, 0)
    return np.repeat(enhanced[..., None], 3, axis=2).astype(np.uint8)


# 匹配既可以直接在 RGB 图上做，也可以在结构增强图上做。
# 这里选中的图像版本，也会直接用于后续匹配证据图的绘制。
def prepare_match_images(args, fake: np.ndarray, optical: np.ndarray, result_dir: Path) -> dict:
    if args.match_preprocess == "rgb":
        return {
            "fake_path": result_dir / "fake_optical.png",
            "optical_path": result_dir / "real_optical_resized.png",
            "fake_display": fake,
            "optical_display": optical,
            "fake_meta": args.sar_transform,
            "optical_meta": args.optical_transform,
        }

    fake_structure = enhance_structure(fake)
    optical_structure = enhance_structure(optical)
    fake_path = result_dir / "match_fake_structure.png"
    optical_path = result_dir / "match_real_optical_structure.png"
    save_rgb(fake_path, fake_structure)
    save_rgb(optical_path, optical_structure)
    return {
        "fake_path": fake_path,
        "optical_path": optical_path,
        "fake_display": fake_structure,
        "optical_display": optical_structure,
        "fake_meta": args.sar_transform,
        "optical_meta": args.optical_transform,
    }


# 一个可用的单应矩阵不仅要有内点，还要求内点在空间上分布开。
# 否则即使局部区域数值上看起来不错，整体 warp 仍然可能很差。
def inlier_spatial_quality(
    points0: np.ndarray,
    points1: np.ndarray,
    inlier_mask: np.ndarray | None,
    shape: tuple[int, int, int],
    *,
    min_coverage: float,
    min_quadrants: int,
) -> tuple[bool, dict]:
    if inlier_mask is None or not inlier_mask.any():
        return False, {"coverage0": 0.0, "coverage1": 0.0, "quadrants0": 0, "quadrants1": 0}

    height, width = shape[:2]
    image_area = max(float(width * height), 1.0)

    def stats(points: np.ndarray) -> tuple[float, int]:
        inlier_points = points[inlier_mask]
        if len(inlier_points) == 0:
            return 0.0, 0
        mins = inlier_points.min(axis=0)
        maxs = inlier_points.max(axis=0)
        coverage = float(max(0.0, maxs[0] - mins[0]) * max(0.0, maxs[1] - mins[1]) / image_area)
        quadrants = set()
        for x, y in inlier_points:
            quadrants.add((int(x >= width / 2), int(y >= height / 2)))
        return coverage, len(quadrants)

    coverage0, quadrants0 = stats(points0)
    coverage1, quadrants1 = stats(points1)
    ok = max(coverage0, coverage1) >= min_coverage and max(quadrants0, quadrants1) >= min_quadrants
    return ok, {
        "coverage0": round(coverage0, 6),
        "coverage1": round(coverage1, 6),
        "quadrants0": quadrants0,
        "quadrants1": quadrants1,
    }


# 即使内点数足够，估计出的变换仍可能不合理。
# 这里额外检查面积变化和角点位移，提前拦截异常 Homography。
def homography_shape_quality(
    homography: np.ndarray | None,
    shape: tuple[int, int, int],
    *,
    min_area_ratio: float,
    max_area_ratio: float,
    max_corner_shift_ratio: float,
) -> tuple[bool, dict]:
    if homography is None:
        return False, {"area_ratio": None, "max_corner_shift_ratio": None}

    height, width = shape[:2]
    corners = np.array(
        [[[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]],
        dtype=np.float32,
    )
    warped = cv2.perspectiveTransform(corners, homography.astype(np.float64)).reshape(-1, 2)
    if not np.isfinite(warped).all():
        return False, {"area_ratio": None, "max_corner_shift_ratio": None}

    area = abs(float(cv2.contourArea(warped.astype(np.float32))))
    area_ratio = area / max(float(width * height), 1.0)
    shifts = np.linalg.norm(warped - corners.reshape(-1, 2), axis=1)
    max_shift_ratio = float(shifts.max() / max(float(np.hypot(width, height)), 1.0))
    ok = min_area_ratio <= area_ratio <= max_area_ratio and max_shift_ratio <= max_corner_shift_ratio
    return ok, {
        "area_ratio": round(area_ratio, 6),
        "max_corner_shift_ratio": round(max_shift_ratio, 6),
    }


# 这是最终的质量门控，网页端和实验脚本都依赖这一步判断：
# 当前配准结果应被视为“可信”，还是应进入复核。
def evaluate_reliability(
    *,
    inliers: int,
    inlier_ratio: float,
    rmse: float | None,
    spatial_ok: bool,
    homography_ok: bool,
) -> tuple[bool, list[str]]:
    reasons = []
    if inliers < 8:
        reasons.append("inliers < 8")
    if inlier_ratio < 0.25:
        reasons.append("inlier_ratio < 0.25")
    if rmse is None:
        reasons.append("rmse unavailable")
    elif rmse > 5.0:
        reasons.append("rmse > 5px")
    if not spatial_ok:
        reasons.append("poor_spatial_coverage")
    if not homography_ok:
        reasons.append("bad_homography_shape")
    return not reasons, reasons


# 一个 candidate 就代表“一种提取器 + 一次 LightGlue 匹配 + 一次
# Homography 估计”。级联模式本质上就是重复跑多个 candidate，再择优。
def run_match_candidate(
    args,
    extractor_name: str,
    match_inputs: dict,
    result_dir: Path,
    device: torch.device,
) -> dict:
    start = time.perf_counter()
    try:
        extractor = build_extractor(extractor_name, args.max_keypoints).eval().to(device)
        matcher = LightGlue(features=extractor_name).eval().to(device)
        image0 = load_image(str(match_inputs["fake_path"])).to(device)
        image1 = load_image(str(match_inputs["optical_path"])).to(device)
        feats0 = extractor.extract(image0)
        feats1 = extractor.extract(image1)
        matches01 = matcher({"image0": feats0, "image1": feats1})
        feats0, feats1, matches01 = [rbd(item) for item in [feats0, feats1, matches01]]
    except Exception as exc:
        return {
            "extractor": extractor_name,
            "error": str(exc),
            "elapsed_ms": now_ms(start),
            "homography": None,
            "inlier_mask": None,
            "points0": np.empty((0, 2), dtype=np.float32),
            "points1": np.empty((0, 2), dtype=np.float32),
            "metrics": {
                "extractor": extractor_name,
                "match_count": 0,
                "raw_match_count": 0,
                "padding_filtered_count": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "mean_score": 0.0,
                "rmse": None,
                "registration_reliable": False,
                "reliability_reasons": ["matcher_error"],
                "poor_spatial_coverage": True,
                "bad_homography_shape": True,
            },
        }

    matches = matches01["matches"]
    scores = matches01.get("scores")
    raw_match_count = int(matches.shape[0])

    points0 = np.empty((0, 2), dtype=np.float32)
    points1 = np.empty((0, 2), dtype=np.float32)
    score_values = np.empty((0,), dtype=np.float32)
    homography = None
    inlier_mask = None
    rmse = None
    if raw_match_count:
        points0 = feats0["keypoints"][matches[..., 0]].detach().cpu().numpy().astype(np.float32)
        points1 = feats1["keypoints"][matches[..., 1]].detach().cpu().numpy().astype(np.float32)
        if scores is not None and len(scores) > 0:
            score_values = scores.detach().cpu().numpy().astype(np.float32)
        if args.resize_mode == "letterbox":
            valid_mask = points_inside_content(points0, match_inputs["fake_meta"]) & points_inside_content(
                points1,
                match_inputs["optical_meta"],
            )
            points0 = points0[valid_mask]
            points1 = points1[valid_mask]
            if len(score_values) == len(valid_mask):
                score_values = score_values[valid_mask]

    match_count = int(len(points0))
    padding_filtered_count = raw_match_count - match_count
    mean_score = float(score_values.mean()) if len(score_values) > 0 else 0.0
    if match_count >= 4:
        homography, mask = cv2.findHomography(points0, points1, cv2.USAC_MAGSAC, args.ransac_threshold)
        if homography is None:
            homography, mask = cv2.findHomography(points0, points1, cv2.RANSAC, args.ransac_threshold)
        if mask is not None:
            inlier_mask = mask.reshape(-1).astype(bool)
        if homography is not None:
            projected = cv2.perspectiveTransform(points0.reshape(-1, 1, 2), homography).reshape(-1, 2)
            errors = np.linalg.norm(projected - points1, axis=1)
            if inlier_mask is not None and inlier_mask.any():
                errors = errors[inlier_mask]
            rmse = float(np.sqrt(np.mean(np.square(errors)))) if len(errors) else None

    inliers = int(inlier_mask.sum()) if inlier_mask is not None else 0
    inlier_ratio = float(inliers / max(match_count, 1))
    spatial_ok, spatial_stats = inlier_spatial_quality(
        points0,
        points1,
        inlier_mask,
        match_inputs["fake_display"].shape,
        min_coverage=args.min_spatial_coverage,
        min_quadrants=args.min_inlier_quadrants,
    )
    homography_ok, homography_stats = homography_shape_quality(
        homography,
        match_inputs["fake_display"].shape,
        min_area_ratio=args.min_homography_area_ratio,
        max_area_ratio=args.max_homography_area_ratio,
        max_corner_shift_ratio=args.max_corner_shift_ratio,
    )
    registration_reliable, reliability_reasons = evaluate_reliability(
        inliers=inliers,
        inlier_ratio=inlier_ratio,
        rmse=rmse,
        spatial_ok=spatial_ok,
        homography_ok=homography_ok,
    )
    h_matrix = np.round(homography, 6).tolist() if homography is not None else None
    dx = float(homography[0, 2]) if homography is not None else None
    dy = float(homography[1, 2]) if homography is not None else None

    return {
        "extractor": extractor_name,
        "error": "",
        "elapsed_ms": now_ms(start),
        "homography": homography,
        "inlier_mask": inlier_mask,
        "points0": points0,
        "points1": points1,
        "metrics": {
            "extractor": extractor_name,
            "match_count": match_count,
            "raw_match_count": raw_match_count,
            "padding_filtered_count": padding_filtered_count,
            "matches_total": match_count,
            "matches_used": match_count,
            "correct_matches": inliers,
            "inliers": inliers,
            "inlier_ratio": round(inlier_ratio, 4),
            "mean_score": round(mean_score, 6),
            "rmse": round(float(rmse), 4) if rmse is not None else None,
            "registration_reliable": registration_reliable,
            "reliability_reasons": reliability_reasons,
            "dx": round(dx, 4) if dx is not None else None,
            "dy": round(dy, 4) if dy is not None else None,
            "h_matrix": h_matrix,
            "poor_spatial_coverage": not spatial_ok,
            "bad_homography_shape": not homography_ok,
            **spatial_stats,
            **homography_stats,
        },
    }


# 这里的排序策略是保守的：先保证“可信”优先于“不可信”，
# 然后再比较内点数、内点率和 RMSE。
def candidate_score(candidate: dict) -> tuple:
    metrics = candidate["metrics"]
    rmse = metrics.get("rmse")
    return (
        int(bool(metrics.get("registration_reliable"))),
        int(metrics.get("inliers") or 0),
        float(metrics.get("inlier_ratio") or 0.0),
        -float(rmse if rmse is not None else 1e9),
        float(metrics.get("mean_score") or 0.0),
    )


def summarize_candidate(candidate: dict) -> dict:
    metrics = candidate["metrics"]
    return {
        "extractor": candidate["extractor"],
        "elapsed_ms": candidate["elapsed_ms"],
        "error": candidate.get("error", ""),
        "match_count": metrics.get("match_count"),
        "raw_match_count": metrics.get("raw_match_count"),
        "padding_filtered_count": metrics.get("padding_filtered_count"),
        "inliers": metrics.get("inliers"),
        "inlier_ratio": metrics.get("inlier_ratio"),
        "rmse": metrics.get("rmse"),
        "registration_reliable": metrics.get("registration_reliable"),
        "reliability_reasons": metrics.get("reliability_reasons"),
        "poor_spatial_coverage": metrics.get("poor_spatial_coverage"),
        "bad_homography_shape": metrics.get("bad_homography_shape"),
    }


# 第 2 阶段：先在 Fake Optical -> Real Optical 之间估计几何关系，
# 再把选中的 Homography 迁移回 SAR，因为 Fake Optical 与 SAR
# 共享同一个 resize 后的像素坐标系。
def run_lightglue(args, generation: dict, result_dir: Path, device: torch.device) -> dict:
    start = time.perf_counter()
    fake = generation["fake"]
    sar = generation["sar"]
    optical = generation["optical"]
    sar_original = generation["sar_original"]
    optical_original = generation["optical_original"]
    args.sar_transform = generation["sar_transform"]
    args.optical_transform = generation["optical_transform"]
    match_inputs = prepare_match_images(args, fake, optical, result_dir)
    extractor_names = args.extractors if args.extractor_policy == "cascade" else [args.extractor]
    candidates = []
    for extractor_name in extractor_names:
        candidate = run_match_candidate(args, extractor_name, match_inputs, result_dir, device)
        candidates.append(candidate)
        if args.extractor_policy == "cascade" and args.cascade_stop_on_reliable:
            if candidate["metrics"].get("registration_reliable"):
                break

    selected = max(candidates, key=candidate_score)
    selected_metrics = selected["metrics"]
    points0 = selected["points0"]
    points1 = selected["points1"]
    inlier_mask = selected["inlier_mask"]
    homography = selected["homography"]
    h_canvas = homography
    h_original = None
    if h_canvas is not None:
        try:
            h_original = np.linalg.inv(transform_matrix(generation["optical_transform"])) @ h_canvas @ transform_matrix(
                generation["sar_transform"],
            )
            if abs(float(h_original[2, 2])) > 1e-12:
                h_original = h_original / h_original[2, 2]
            if not np.isfinite(h_original).all():
                h_original = None
        except np.linalg.LinAlgError:
            h_original = None

    if homography is None:
        fake_registered = fake.copy()
        sar_registered = sar.copy()
    else:
        fake_registered = cv2.warpPerspective(
            fake,
            homography.astype(np.float32),
            (optical.shape[1], optical.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        sar_registered = cv2.warpPerspective(
            sar,
            homography.astype(np.float32),
            (optical.shape[1], optical.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

    fake_registered_path = result_dir / "fake_optical_registered.png"
    sar_registered_path = result_dir / "sar_registered.png"
    download_path = result_dir / "registered_output.png"
    match_path = result_dir / "lightglue_matches.png"
    sar_transfer_match_path = result_dir / "sar_transferred_matches.png"
    optical_points_path = result_dir / "optical_transferred_points.png"
    save_rgb(fake_registered_path, fake_registered)
    save_rgb(sar_registered_path, sar_registered)
    save_rgb(download_path, sar_registered)
    draw_matches(match_inputs["fake_display"], match_inputs["optical_display"], points0, points1, inlier_mask, match_path)
    draw_matches(sar, optical, points0, points1, inlier_mask, sar_transfer_match_path)
    draw_points_on_image(optical, points1, inlier_mask, optical_points_path)
    visual = save_visualizations(sar_registered, optical, result_dir)
    original_paths = {}
    original_visual = None
    if args.resize_mode == "letterbox" and h_original is not None:
        sar_registered_original = cv2.warpPerspective(
            sar_original,
            h_original.astype(np.float32),
            (optical_original.shape[1], optical_original.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        sar_registered_original_path = result_dir / "sar_registered_original.png"
        download_original_path = result_dir / "registered_output_original.png"
        save_rgb(sar_registered_original_path, sar_registered_original)
        save_rgb(download_original_path, sar_registered_original)
        original_visual = save_visualizations(sar_registered_original, optical_original, result_dir, suffix="_original")
        original_paths = {
            "sar_registered_original": str(sar_registered_original_path),
            "registered_preview": str(sar_registered_original_path),
            "download": str(download_original_path),
            **original_visual["paths"],
        }

    candidate_summaries = [summarize_candidate(candidate) for candidate in candidates]
    cascade_rescued = bool(
        args.extractor_policy == "cascade"
        and selected["extractor"] != extractor_names[0]
        and selected_metrics.get("registration_reliable")
    )

    return {
        "lightglue_ms": now_ms(start),
        "paths": {
            "matches": str(match_path),
            "sar_transferred_matches": str(sar_transfer_match_path),
            "optical_transferred_points": str(optical_points_path),
            "fake_registered": str(fake_registered_path),
            "sar_registered": str(sar_registered_path),
            "registered_preview": str(sar_registered_path),
            "download": str(download_path),
            **visual["paths"],
            **original_paths,
        },
        "metrics": {
            "prediction_type": "sar_via_fakeoptical_lightglue_homography",
            **selected_metrics,
            "resize_mode": args.resize_mode,
            "coordinate_mode": (
                "letterbox_with_original_backprojection"
                if args.resize_mode == "letterbox" and h_original is not None
                else ("letterbox_canvas_no_homography" if args.resize_mode == "letterbox" else "stretched_canvas")
            ),
            "h_matrix_canvas": np.round(h_canvas, 6).tolist() if h_canvas is not None else None,
            "h_matrix_original": np.round(h_original, 6).tolist() if h_original is not None else None,
            "sar_transform": generation["sar_transform"],
            "optical_transform": generation["optical_transform"],
            "reliability_rule": "reliable iff inliers >= 8, inlier_ratio >= 0.25, rmse <= 5px, spatial coverage is sufficient, homography shape is plausible",
            "difference_mean": original_visual["difference_mean"] if original_visual else visual["difference_mean"],
            "geometry_source": "Fake Optical -> Real Optical",
            "geometry_applied_to": "SAR -> Real Optical",
            "point_transfer": "Fake Optical keypoints are reused as SAR coordinates in the 256x256 canvas; letterbox mode back-projects the selected homography to original image coordinates.",
            "max_keypoints": args.max_keypoints,
            "extractor": selected["extractor"],
            "selected_extractor": selected["extractor"],
            "extractor_policy": args.extractor_policy,
            "extractors_tried": extractor_names,
            "match_preprocess": args.match_preprocess,
            "cascade_rescued": cascade_rescued,
            "matcher_candidates": candidate_summaries,
        },
    }


# main 是一个很薄的推理入口：负责解析参数、执行生成、执行配准，
# 最后把结果整理成单个 JSON，供 Flask 网页端读取。
def main() -> None:
    parser = argparse.ArgumentParser(description="Single-pair ACD_S2ODPM + LightGlue web worker.")
    parser.add_argument("--sar", required=True)
    parser.add_argument("--optical", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--max-keypoints", type=int, default=1024)
    parser.add_argument("--extractor", default="superpoint", choices=["superpoint", "disk", "aliked"])
    parser.add_argument("--extractors", nargs="+", choices=["superpoint", "disk", "aliked"], default=None)
    parser.add_argument("--extractor-policy", default="single", choices=["single", "cascade"])
    parser.add_argument("--cascade-stop-on-reliable", action="store_true")
    parser.add_argument("--match-preprocess", default="rgb", choices=["rgb", "structure"])
    parser.add_argument("--resize-mode", default="stretch", choices=["stretch", "letterbox"])
    parser.add_argument("--ransac-threshold", type=float, default=3.0)
    parser.add_argument("--min-spatial-coverage", type=float, default=0.015)
    parser.add_argument("--min-inlier-quadrants", type=int, default=2)
    parser.add_argument("--min-homography-area-ratio", type=float, default=0.2)
    parser.add_argument("--max-homography-area-ratio", type=float, default=5.0)
    parser.add_argument("--max-corner-shift-ratio", type=float, default=1.5)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="fp16", choices=["fp16", "fp32"])
    parser.add_argument("--sar-log", action="store_true")
    args = parser.parse_args()
    if args.extractors is None:
        args.extractors = [args.extractor]
    if args.extractor_policy == "single":
        args.extractor = args.extractors[0] if args.extractors else args.extractor
        args.extractors = [args.extractor]

    total_start = time.perf_counter()
    result_dir = Path(args.output_dir)
    result_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    requested_device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    device = torch.device(requested_device)
    dtype = torch.float16 if args.dtype == "fp16" and device.type == "cuda" else torch.float32

    generation = run_generation(args, result_dir, device, dtype)
    lightglue = run_lightglue(args, generation, result_dir, device)

    timings = {
        "generation_ms": generation["generation_ms"],
        "registration_ms": lightglue["lightglue_ms"],
        "total_ms": now_ms(total_start),
    }
    payload = {
        "success": True,
        "method": "diffusion_lightglue",
        "model_info": {
            "generator_checkpoint": generation["checkpoint"],
            "device": str(device),
            "dtype": str(dtype).replace("torch.", ""),
            "steps": args.steps,
            "max_keypoints": args.max_keypoints,
            "extractor": args.extractor,
            "extractor_policy": args.extractor_policy,
            "extractors": args.extractors,
            "match_preprocess": args.match_preprocess,
            "resize_mode": args.resize_mode,
        },
        "timings": timings,
        "metrics": {
            **lightglue["metrics"],
            "generator_steps": args.steps,
        },
        "artifacts": {
            **generation["paths"],
            **lightglue["paths"],
        },
    }
    result_json = result_dir / "diffusion_lightglue_result.json"
    result_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
