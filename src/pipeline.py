from pathlib import Path
from typing import Any, Dict, List, Optional

import rfdetr


SUPPORTED_TASKS = {"detection", "segmentation"}
DEFAULT_MODEL_VARIANTS = {"detection": "base", "segmentation": "seg-preview"}
AUTO_PRETRAINED_WEIGHTS_VALUES = {"", "auto", "default", "none", "null"}

DETECTION_MODEL_CLASSES = {
    "nano": "RFDETRNano",
    "n": "RFDETRNano",
    "small": "RFDETRSmall",
    "s": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "m": "RFDETRMedium",
    "large": "RFDETRLarge",
    "l": "RFDETRLarge",
    "xlarge": "RFDETRXLarge",
    "xl": "RFDETRXLarge",
    "2xlarge": "RFDETR2XLarge",
    "2xl": "RFDETR2XLarge",
    "base": "RFDETRBase",
}

SEGMENTATION_MODEL_CLASSES = {
    "nano": "RFDETRSegNano",
    "n": "RFDETRSegNano",
    "small": "RFDETRSegSmall",
    "s": "RFDETRSegSmall",
    "medium": "RFDETRSegMedium",
    "m": "RFDETRSegMedium",
    "large": "RFDETRSegLarge",
    "l": "RFDETRSegLarge",
    "xlarge": "RFDETRSegXLarge",
    "xl": "RFDETRSegXLarge",
    "2xlarge": "RFDETRSeg2XLarge",
    "2xl": "RFDETRSeg2XLarge",
    "seg-preview": "RFDETRSegPreview",
}


def normalize_task(task: str) -> str:
    normalized = str(task or "detection").lower().strip()
    if normalized not in SUPPORTED_TASKS:
        raise ValueError(
            f"Unsupported task: {task}. Expected one of {sorted(SUPPORTED_TASKS)}"
        )
    return normalized


def normalize_model_variant(task: str, variant: Optional[str]) -> str:
    normalized_task = normalize_task(task)
    normalized_variant = str(
        variant or DEFAULT_MODEL_VARIANTS[normalized_task]
    ).lower().strip()
    normalized_variant = normalized_variant.replace("_", "").replace("-", "")

    model_classes = (
        SEGMENTATION_MODEL_CLASSES if normalized_task == "segmentation" else DETECTION_MODEL_CLASSES
    )
    canonical_variants = {
        key.lower().replace("_", "").replace("-", ""): key
        for key in model_classes
    }
    class_name_variants = {
        value.lower().replace("_", "").replace("-", ""): key
        for key, value in model_classes.items()
    }

    resolved_variant = canonical_variants.get(normalized_variant) or class_name_variants.get(
        normalized_variant
    )
    if resolved_variant is None:
        raise ValueError(
            f"Unsupported {normalized_task} model variant: {variant}. "
            f"Expected one of {sorted(model_classes)}"
        )
    return resolved_variant


def resolve_model_class(task: str, variant: Optional[str]):
    normalized_task = normalize_task(task)
    normalized_variant = normalize_model_variant(normalized_task, variant)
    model_classes = (
        SEGMENTATION_MODEL_CLASSES if normalized_task == "segmentation" else DETECTION_MODEL_CLASSES
    )
    class_name = model_classes[normalized_variant]

    try:
        return getattr(rfdetr, class_name)
    except AttributeError as exc:
        raise ValueError(
            f"The installed rfdetr package does not expose {class_name}. "
            f"Requested task={normalized_task}, variant={normalized_variant}."
        ) from exc


def build_model(
    task: str,
    model_cfg: Optional[Dict[str, Any]] = None,
    class_names: Optional[List[str]] = None,
):
    normalized_task = normalize_task(task)
    model_cfg = model_cfg or {}
    model_class = resolve_model_class(normalized_task, model_cfg.get("variant"))

    init_kwargs = {}
    pretrained_weights = model_cfg.get("pretrained_weights")
    if pretrained_weights and str(pretrained_weights).strip().lower() not in AUTO_PRETRAINED_WEIGHTS_VALUES:
        init_kwargs["pretrain_weights"] = str(pretrained_weights)
    if class_names:
        init_kwargs["num_classes"] = len(class_names)
    return model_class(**init_kwargs)


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


def override_train_dataset_dir(train_kwargs: Dict[str, Any], dataset_dir: str) -> Dict[str, Any]:
    updated_kwargs = dict(train_kwargs)
    updated_kwargs["dataset_dir"] = dataset_dir
    return updated_kwargs


def resolve_output_dir(cfg: Dict[str, Any]) -> Path:
    task = normalize_task(cfg.get("task", "detection"))
    return Path(cfg["output"]["dir"]) / cfg["output"].get("run_name", task)
