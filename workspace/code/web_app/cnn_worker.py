"""
CNN inference worker.
Runs inside the local .cnn_runtime environment and returns predictions for the
selected registration model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from models.baseline import BaselineCNN
from models.dense_field import DenseFieldRegistrationCNN
from models.multiscale import MultiScaleHomographyCNN, MultiScaleRegistrationCNN


def load_state_dict(model_path: Path):
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        checkpoint = torch.load(model_path, map_location="cpu")
    return checkpoint.get("model_state_dict", checkpoint)


def build_model(model_path: Path):
    name = model_path.name.lower()
    if "dense" in name or "nonrigid" in name:
        return DenseFieldRegistrationCNN(), "dense_field_nonrigid"
    if "homography" in name:
        return MultiScaleHomographyCNN(), "multiscale_homography"
    if "multiscale" in name:
        return MultiScaleRegistrationCNN(), "multiscale_translation"
    return BaselineCNN(), "baseline_translation"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sar-npy", required=True)
    parser.add_argument("--optical-npy", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--flow-out", default="")
    args = parser.parse_args()

    sar_array = np.load(args.sar_npy).astype(np.float32)
    optical_array = np.load(args.optical_npy).astype(np.float32)

    sar_tensor = torch.from_numpy(sar_array).unsqueeze(0).unsqueeze(0)
    optical_tensor = torch.from_numpy(optical_array).unsqueeze(0).unsqueeze(0)

    model_path = Path(args.model_path)
    model, model_family = build_model(model_path)
    state_dict = load_state_dict(model_path)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    if missing_keys or unexpected_keys:
        raise RuntimeError(
            f"Model state mismatch for {model_path.name}: "
            f"missing={missing_keys}, unexpected={unexpected_keys}"
        )
    model.eval()

    with torch.no_grad():
        prediction = model(sar_tensor, optical_tensor)

    prediction_array = prediction.detach().cpu().numpy()
    result = {
        "success": True,
        "missing_keys": len(missing_keys),
        "unexpected_keys": len(unexpected_keys),
        "model_family": model_family,
        "torch_version": torch.__version__,
    }

    if model_family == "dense_field_nonrigid":
        if not args.flow_out:
            raise RuntimeError("Dense-field model requires --flow-out to save the predicted displacement field.")
        flow_path = Path(args.flow_out)
        np.save(flow_path, prediction_array.squeeze(0).astype(np.float32))
        flow = prediction_array.squeeze(0)
        magnitude = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
        result.update(
            {
                "prediction_type": "dense_flow",
                "flow_path": str(flow_path),
                "flow_mean_magnitude": float(magnitude.mean()),
                "flow_max_magnitude": float(magnitude.max()),
                "prediction": None,
            }
        )
    elif model_family == "multiscale_homography":
        result.update(
            {
                "prediction_type": "homography",
                "prediction": prediction_array.reshape(-1).tolist(),
            }
        )
    else:
        result.update(
            {
                "prediction_type": "translation",
                "prediction": prediction_array.reshape(-1).tolist(),
            }
        )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
