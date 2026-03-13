from pathlib import Path
from typing import Any, Dict, List, Optional

from rfdetr import RFDETRBase, RFDETRSegPreview


SUPPORTED_TASKS = {"detection", "segmentation"}


def normalize_task(task: str) -> str:
    normalized = str(task or "detection").lower().strip()
    if normalized not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task: {task}. Expected one of {sorted(SUPPORTED_TASKS)}"
        )
    return normalized


def build_model(
    task: str,
    model_cfg: Optional[Dict[str, Any]] = None,
    class_names: Optional[List[str]] = None,
):
    normalized_task = normalize_task(task)
    model_cfg = model_cfg or {}

    init_kwargs = {}
    pretrained_weights = model_cfg.get("pretrained_weights")
    if pretrained_weights:
        init_kwargs["pretrain_weights"] = str(pretrained_weights)
    if class_names:
        init_kwargs["num_classes"] = len(class_names)

    if normalized_task == "segmentation":
        return RFDETRSegPreview(**init_kwargs)
    return RFDETRBase(**init_kwargs)


def build_train_kwargs(cfg: Dict[str, Any], output_dir: Path) -> Dict[str, Any]:
    train_cfg = cfg.get("train", {})
    model_cfg = cfg.get("model", {})

    kwargs: Dict[str, Any] = {
        "dataset_dir": cfg["dataset"]["root_dir"],
        "epochs": int(train_cfg["num_epochs"]),
        "batch_size": int(train_cfg["batch_size"]),
        "grad_accum_steps": int(train_cfg.get("grad_accum_steps", 1)),
        "lr": float(train_cfg["lr"]),
        "output_dir": str(output_dir),
    }

    optional_train_args = {
        "weight_decay": train_cfg.get("weight_decay"),
        "num_workers": train_cfg.get("num_workers"),
        "device": train_cfg.get("device", cfg.get("device")),
        "resolution": train_cfg.get("resolution"),
        "amp": train_cfg.get("amp"),
        "tensorboard": train_cfg.get("tensorboard"),
        "early_stopping": train_cfg.get("early_stopping"),
    }
    for key, value in optional_train_args.items():
        if value is not None:
            kwargs[key] = value

    if normalize_task(cfg.get("task", "detection")) == "segmentation":
        kwargs["segmentation_head"] = bool(model_cfg.get("segmentation_head", True))

    return kwargs


def resolve_output_dir(cfg: Dict[str, Any]) -> Path:
    task = normalize_task(cfg.get("task", "detection"))
    return Path(cfg["output"]["dir"]) / cfg["output"].get("run_name", task)
