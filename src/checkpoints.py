from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Union

import torch


DEPLOYABLE_MODEL_FORMAT = "rfdetr_deployable_model"


def _preferred_weight_names() -> List[str]:
    return [
        "checkpoint_best_regular.pth",
        "checkpoint_best_ema.pth",
        "checkpoint_last_regular.pth",
        "checkpoint_last_total.pth",
        "checkpoint_best_total.pth",
    ]


def find_trained_weights(output_dir: Union[str, Path]) -> Path:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        raise FileNotFoundError(f"Training output directory not found: {output_dir}")

    for file_name in _preferred_weight_names():
        candidate = output_dir / file_name
        if candidate.is_file():
            return candidate

    candidates = sorted(
        path
        for path in output_dir.rglob("*.pth")
        if path.name not in {
            "rfdetr_detection_model.pth",
            "rfdetr_segmentation_model.pth",
        }
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "Could not find the trained RF-DETR weights checkpoint in "
        f"{output_dir}. Expected one of {_preferred_weight_names()}."
    )


def save_deployable_model(
    path: Union[str, Path],
    trained_weights_path: Union[str, Path],
    class_names: List[str],
    task: str,
    model_config: Dict[str, Any],
    metrics: Optional[Dict[str, Any]] = None,
) -> None:
    trained_weights_path = Path(trained_weights_path)
    if not trained_weights_path.is_file():
        raise FileNotFoundError(f"Trained weights checkpoint not found: {trained_weights_path}")

    payload = {
        "format": DEPLOYABLE_MODEL_FORMAT,
        "class_names": class_names,
        "task": task,
        "model": model_config,
        "metrics": metrics or {},
        "weights_checkpoint_name": trained_weights_path.name,
        "weights_checkpoint": torch.load(trained_weights_path, map_location="cpu", weights_only=False),
    }

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_deployable_model(path: Union[str, Path]) -> Dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        raise TypeError("Deployable model must be a dict")
    if obj.get("format") != DEPLOYABLE_MODEL_FORMAT:
        raise ValueError("Unsupported model artifact format")
    if "weights_checkpoint" not in obj:
        raise ValueError("Deployable model is missing embedded weights")
    return obj


def extract_class_names(bundle: Dict[str, Any]) -> List[str]:
    class_names = bundle.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("Model artifact does not contain valid class_names")
    return class_names


def extract_task(bundle: Dict[str, Any]) -> str:
    task = bundle.get("task")
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Model artifact does not contain a valid task")
    return task


def extract_model_config(bundle: Dict[str, Any]) -> Dict[str, Any]:
    model_config = bundle.get("model", {})
    if model_config is None:
        return {}
    if not isinstance(model_config, dict):
        raise ValueError("Model artifact does not contain a valid model config")
    return model_config


def with_embedded_weights_file(bundle: Dict[str, Any], callback):
    checkpoint_name = str(bundle.get("weights_checkpoint_name") or "embedded_weights.pth")
    with TemporaryDirectory(prefix="rfdetr-weights-") as temp_dir:
        weights_path = Path(temp_dir) / checkpoint_name
        torch.save(bundle["weights_checkpoint"], weights_path)
        return callback(weights_path)
