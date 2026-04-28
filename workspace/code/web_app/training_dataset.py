from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class PairRecord:
    sar_path: Path
    optical_path: Path


def _list_images(directory: Path) -> dict[str, Path]:
    files = {}
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            files[_pair_key(path)] = path
    return files


def _pair_key(path: Path) -> str:
    stem = path.stem.lower()
    match = re.search(r"(\d+)$", stem)
    if match:
        return match.group(1)
    return re.sub(r"^[a-z]+", "", stem) or stem


def discover_pairs(sar_dir: Path, optical_dir: Path) -> list[PairRecord]:
    if not sar_dir.exists():
        raise FileNotFoundError(f"SAR directory not found: {sar_dir}")
    if not optical_dir.exists():
        raise FileNotFoundError(f"Optical directory not found: {optical_dir}")

    sar_images = _list_images(sar_dir)
    optical_images = _list_images(optical_dir)
    shared_keys = sorted(set(sar_images) & set(optical_images))
    return [PairRecord(sar_images[key], optical_images[key]) for key in shared_keys]


def split_pairs(
    pairs: list[PairRecord],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> tuple[list[PairRecord], list[PairRecord]]:
    shuffled = pairs[:]
    random.Random(seed).shuffle(shuffled)
    train_count = max(1, int(len(shuffled) * train_ratio))
    if train_count >= len(shuffled):
        train_count = max(1, len(shuffled) - 1)
    return shuffled[:train_count], shuffled[train_count:]


def generate_random_dense_flow(
    height: int,
    width: int,
    grid_size: int = 5,
    max_displacement: float = 8.0,
) -> torch.Tensor:
    coarse_flow = torch.empty(1, 2, grid_size, grid_size).uniform_(-max_displacement, max_displacement)
    dense_flow = TF.resize(
        coarse_flow,
        [height, width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=False,
    )
    return dense_flow.squeeze(0)


def transform_points_with_homography(points: np.ndarray, h_matrix: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    homogeneous = np.concatenate([points, np.ones((points.shape[0], 1), dtype=np.float32)], axis=1)
    mapped = homogeneous @ h_matrix.T
    scale = np.clip(mapped[:, 2:3], 1e-8, None)
    return mapped[:, :2] / scale


def homography_matrix_to_vector(h_matrix: np.ndarray, input_size: int) -> torch.Tensor:
    size = float(input_size - 1)
    corners = np.array(
        [[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]],
        dtype=np.float32,
    )
    mapped_corners = transform_points_with_homography(corners, np.asarray(h_matrix, dtype=np.float32))
    corner_offsets = mapped_corners - corners
    return torch.from_numpy(corner_offsets.reshape(-1).astype(np.float32))


def solve_homography_matrix(source: np.ndarray, destination: np.ndarray) -> np.ndarray:
    source = np.asarray(source, dtype=np.float32)
    destination = np.asarray(destination, dtype=np.float32)
    if source.shape != (4, 2) or destination.shape != (4, 2):
        raise ValueError("Homography estimation expects exactly 4 source and 4 destination points.")

    rows = []
    targets = []
    for (x, y), (u, v) in zip(source, destination):
        rows.append([x, y, 1.0, 0.0, 0.0, 0.0, -u * x, -u * y])
        rows.append([0.0, 0.0, 0.0, x, y, 1.0, -v * x, -v * y])
        targets.extend([u, v])

    coefficients = np.linalg.solve(np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.float32))
    return np.array(
        [
            [coefficients[0], coefficients[1], coefficients[2]],
            [coefficients[3], coefficients[4], coefficients[5]],
            [coefficients[6], coefficients[7], 1.0],
        ],
        dtype=np.float32,
    )


def build_centered_affine_matrix(
    center_x: float,
    center_y: float,
    angle_deg: float,
    scale: float,
) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    cos_value = float(np.cos(angle_rad) * scale)
    sin_value = float(np.sin(angle_rad) * scale)

    translate_to_origin = np.array(
        [[1.0, 0.0, -center_x], [0.0, 1.0, -center_y], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    rotate_scale = np.array(
        [[cos_value, -sin_value, 0.0], [sin_value, cos_value, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    translate_back = np.array(
        [[1.0, 0.0, center_x], [0.0, 1.0, center_y], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return translate_back @ rotate_scale @ translate_to_origin


def warp_image_with_homography(image: np.ndarray, h_matrix: np.ndarray, output_size: int) -> np.ndarray:
    inverse_h = np.linalg.inv(h_matrix.astype(np.float32))
    inverse_h = inverse_h / inverse_h[2, 2]
    coefficients = tuple(float(value) for value in inverse_h.reshape(-1)[:8])
    pil_image = Image.fromarray(image.astype(np.float32), mode="F")
    warped = pil_image.transform(
        (output_size, output_size),
        Image.Transform.PERSPECTIVE,
        coefficients,
        resample=Image.BILINEAR,
        fillcolor=0.0,
    )
    return np.asarray(warped, dtype=np.float32)


def _index_seed(base_seed: int, index: int) -> int:
    return int(base_seed + index)


class SyntheticShiftRegistrationDataset(Dataset):
    """
    Build supervised translation labels from roughly aligned image pairs.

    The dataset assumes SAR and optical images correspond to the same scene.
    During training we add a random translation to the SAR image and ask the
    model to predict the inverse shift required to align it back to the
    optical image.
    """

    def __init__(
        self,
        pairs: list[PairRecord],
        input_size: int = 512,
        max_shift: int = 32,
        sar_log_scale: float = 255.0,
        cache_images: bool = False,
        deterministic: bool = False,
        base_seed: int = 42,
    ) -> None:
        self.pairs = pairs
        self.input_size = input_size
        self.max_shift = max_shift
        self.sar_log_scale = sar_log_scale
        self.cache_images = cache_images
        self._pair_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.deterministic = deterministic
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_gray_tensor(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("L")
        image = image.resize((self.input_size, self.input_size), resample=Image.BILINEAR)
        tensor = TF.to_tensor(image)
        return tensor

    def _normalize_sar(self, sar: torch.Tensor) -> torch.Tensor:
        scaled = torch.log1p(sar * self.sar_log_scale)
        return scaled / torch.log1p(torch.tensor(self.sar_log_scale))

    def _random_shift(self, index: int | None = None) -> tuple[int, int]:
        if self.deterministic and index is not None:
            generator = np.random.default_rng(_index_seed(self.base_seed, index))
            dx = int(generator.integers(-self.max_shift, self.max_shift + 1))
            dy = int(generator.integers(-self.max_shift, self.max_shift + 1))
            return dx, dy

        dx = random.randint(-self.max_shift, self.max_shift)
        dy = random.randint(-self.max_shift, self.max_shift)
        return dx, dy

    def _get_pair_tensors(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_images and index in self._pair_cache:
            sar, optical = self._pair_cache[index]
            return sar.clone(), optical.clone()

        pair = self.pairs[index]
        sar = self._normalize_sar(self._load_gray_tensor(pair.sar_path))
        optical = self._load_gray_tensor(pair.optical_path)
        if self.cache_images:
            self._pair_cache[index] = (sar, optical)
        return sar.clone(), optical.clone()

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sar, optical = self._get_pair_tensors(index)

        dx, dy = self._random_shift(index=index)
        shifted_sar = TF.affine(
            sar,
            angle=0.0,
            translate=[dx, dy],
            scale=1.0,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )

        target = torch.tensor([-float(dx), -float(dy)], dtype=torch.float32)
        return {
            "sar": shifted_sar,
            "optical": optical,
            "target": target,
        }

    def describe_sample(self, index: int) -> dict:
        dx, dy = self._random_shift(index=index)
        pair = self.pairs[index]
        return {
            "index": index,
            "seed": _index_seed(self.base_seed, index) if self.deterministic else None,
            "sar_path": str(pair.sar_path),
            "optical_path": str(pair.optical_path),
            "transform": {
                "type": "translation",
                "dx": dx,
                "dy": dy,
            },
            "target": [-float(dx), -float(dy)],
        }


class SyntheticHomographyRegistrationDataset(Dataset):
    """
    Build supervised homography labels from roughly aligned image pairs.

    We synthetically warp the SAR image with a random perspective transform and
    ask the model to predict the inverse homography that maps the warped SAR
    image back to the optical reference.
    """

    def __init__(
        self,
        pairs: list[PairRecord],
        input_size: int = 512,
        max_corner_shift: float = 32.0,
        max_translation: float = 32.0,
        max_rotation_deg: float = 10.0,
        max_scale_jitter: float = 0.08,
        sar_log_scale: float = 255.0,
        cache_images: bool = False,
        deterministic: bool = False,
        base_seed: int = 42,
    ) -> None:
        self.pairs = pairs
        self.input_size = input_size
        self.max_corner_shift = max_corner_shift
        self.max_translation = max_translation
        self.max_rotation_deg = max_rotation_deg
        self.max_scale_jitter = max_scale_jitter
        self.sar_log_scale = sar_log_scale
        self.cache_images = cache_images
        self._pair_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.deterministic = deterministic
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_gray_tensor(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("L")
        image = image.resize((self.input_size, self.input_size), resample=Image.BILINEAR)
        return TF.to_tensor(image)

    def _normalize_sar(self, sar: torch.Tensor) -> torch.Tensor:
        scaled = torch.log1p(sar * self.sar_log_scale)
        normalizer = torch.log1p(torch.tensor(self.sar_log_scale))
        return scaled / normalizer

    def _get_pair_tensors(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cache_images and index in self._pair_cache:
            sar, optical = self._pair_cache[index]
            return sar.clone(), optical.clone()

        pair = self.pairs[index]
        sar = self._normalize_sar(self._load_gray_tensor(pair.sar_path))
        optical = self._load_gray_tensor(pair.optical_path)
        if self.cache_images:
            self._pair_cache[index] = (sar, optical)
        return sar.clone(), optical.clone()

    def _sample_forward_homography(
        self,
        rng: np.random.Generator | None = None,
        return_metadata: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, dict]:
        size = float(self.input_size - 1)
        source = np.array(
            [[0.0, 0.0], [size, 0.0], [size, size], [0.0, size]],
            dtype=np.float32,
        )
        center_x = size * 0.5
        center_y = size * 0.5
        generator = rng or np.random.default_rng()
        for _ in range(12):
            angle_deg = float(generator.uniform(-self.max_rotation_deg, self.max_rotation_deg))
            scale = float(generator.uniform(1.0 - self.max_scale_jitter, 1.0 + self.max_scale_jitter))
            affine_matrix = build_centered_affine_matrix(center_x, center_y, angle_deg, scale)
            affine_destination = transform_points_with_homography(source, affine_matrix)
            global_shift = generator.uniform(-self.max_translation, self.max_translation, size=(1, 2)).astype(np.float32)
            jitter = generator.uniform(-self.max_corner_shift, self.max_corner_shift, size=(4, 2)).astype(np.float32)
            destination = affine_destination + global_shift + jitter
            destination[:, 0] = np.clip(destination[:, 0], 0.0, size)
            destination[:, 1] = np.clip(destination[:, 1], 0.0, size)
            try:
                h_matrix = solve_homography_matrix(source, destination)
            except np.linalg.LinAlgError:
                continue
            if abs(np.linalg.det(h_matrix)) > 1e-6:
                matrix = h_matrix.astype(np.float32)
                metadata = {
                    "type": "homography",
                    "angle_deg": round(angle_deg, 6),
                    "scale": round(scale, 6),
                    "global_shift": global_shift.reshape(-1).astype(np.float32).round(6).tolist(),
                    "jitter": jitter.astype(np.float32).round(6).tolist(),
                    "destination": destination.astype(np.float32).round(6).tolist(),
                    "used_identity_fallback": False,
                }
                if return_metadata:
                    return matrix, metadata
                return matrix

        identity = np.eye(3, dtype=np.float32)
        metadata = {
            "type": "homography",
            "angle_deg": 0.0,
            "scale": 1.0,
            "global_shift": [0.0, 0.0],
            "jitter": np.zeros((4, 2), dtype=np.float32).tolist(),
            "destination": source.astype(np.float32).tolist(),
            "used_identity_fallback": True,
        }
        if return_metadata:
            return identity, metadata
        return identity

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        sar, optical = self._get_pair_tensors(index)

        sar_np = sar.squeeze(0).numpy().astype(np.float32)
        rng = np.random.default_rng(_index_seed(self.base_seed, index)) if self.deterministic else None
        forward_h = self._sample_forward_homography(rng=rng)
        moved_sar = warp_image_with_homography(sar_np, forward_h, self.input_size)
        inverse_h = np.linalg.inv(forward_h)
        inverse_h = inverse_h / inverse_h[2, 2]

        return {
            "sar": torch.from_numpy(moved_sar.copy()).unsqueeze(0),
            "optical": optical,
            "target": homography_matrix_to_vector(inverse_h, self.input_size),
        }

    def describe_sample(self, index: int) -> dict:
        pair = self.pairs[index]
        rng = np.random.default_rng(_index_seed(self.base_seed, index))
        forward_h, transform = self._sample_forward_homography(rng=rng, return_metadata=True)
        inverse_h = np.linalg.inv(forward_h)
        inverse_h = inverse_h / inverse_h[2, 2]
        target = homography_matrix_to_vector(inverse_h, self.input_size)
        return {
            "index": index,
            "seed": _index_seed(self.base_seed, index) if self.deterministic else None,
            "sar_path": str(pair.sar_path),
            "optical_path": str(pair.optical_path),
            "transform": transform,
            "target": target.tolist(),
        }


class SyntheticDenseRegistrationDataset(Dataset):
    """
    Synthetic nonrigid dataset built from roughly aligned SAR/optical pairs.

    We create a smooth dense deformation on the SAR image and train the model to
    predict the approximate inverse field while also encouraging warped SAR and
    optical images to become more similar.
    """

    def __init__(
        self,
        pairs: list[PairRecord],
        input_size: int = 256,
        max_translation: int = 12,
        max_displacement: float = 8.0,
        grid_size: int = 5,
        sar_log_scale: float = 255.0,
    ) -> None:
        self.pairs = pairs
        self.input_size = input_size
        self.max_translation = max_translation
        self.max_displacement = max_displacement
        self.grid_size = grid_size
        self.sar_log_scale = sar_log_scale

    def __len__(self) -> int:
        return len(self.pairs)

    def _load_gray_tensor(self, path: Path) -> torch.Tensor:
        image = Image.open(path).convert("L")
        image = image.resize((self.input_size, self.input_size), resample=Image.BILINEAR)
        return TF.to_tensor(image)

    def _normalize_sar(self, sar: torch.Tensor) -> torch.Tensor:
        scaled = torch.log1p(sar * self.sar_log_scale)
        normalizer = torch.log1p(torch.tensor(self.sar_log_scale))
        return scaled / normalizer

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        from models.dense_field import warp_tensor

        pair = self.pairs[index]
        sar = self._normalize_sar(self._load_gray_tensor(pair.sar_path))
        optical = self._load_gray_tensor(pair.optical_path)

        flow = generate_random_dense_flow(
            self.input_size,
            self.input_size,
            grid_size=self.grid_size,
            max_displacement=self.max_displacement,
        )
        dx = random.randint(-self.max_translation, self.max_translation)
        dy = random.randint(-self.max_translation, self.max_translation)
        flow[0] += dx
        flow[1] += dy

        moving_sar = warp_tensor(sar.unsqueeze(0), -flow.unsqueeze(0)).squeeze(0)
        return {
            "sar": moving_sar,
            "optical": optical,
            "target_flow": flow,
            "reference_sar": sar,
        }
