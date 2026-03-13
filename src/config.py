from pathlib import Path
from copy import deepcopy
from typing import Any, Dict, Optional, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            data,
            f,
            sort_keys=False,
            allow_unicode=True,
        )


def _resolve_path(value: Optional[Union[str, Path]], base_dir: Path) -> Optional[str]:
    if value in (None, ""):
        return None

    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()

    return str(path)


def resolve_runtime_config(
    cfg: Dict[str, Any],
    config_path: Union[str, Path],
) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    config_dir = config_path.parent

    dataset_cfg = cfg.setdefault("dataset", {})
    root_dir = _resolve_path(dataset_cfg.get("root_dir"), config_dir)
    if root_dir is None:
        raise ValueError("Config is missing dataset.root_dir")

    dataset_cfg["root_dir"] = root_dir
    dataset_root = Path(root_dir)

    split_folder_names = {"train": "train", "val": "valid", "test": "test"}
    for split_name, default_folder in split_folder_names.items():
        split_cfg = dataset_cfg.get(split_name)
        if not split_cfg:
            continue

        img_dir = split_cfg.get("img_dir", default_folder)
        ann_file = split_cfg.get("ann_file", f"{default_folder}/_annotations.coco.json")

        split_cfg["img_dir"] = _resolve_path(img_dir, dataset_root)
        split_cfg["ann_file"] = _resolve_path(ann_file, dataset_root)

    output_cfg = cfg.setdefault("output", {})
    output_dir = output_cfg.get("dir", "./outputs")
    output_cfg["dir"] = _resolve_path(output_dir, config_dir)

    model_cfg = cfg.setdefault("model", {})
    pretrained_weights = model_cfg.get("pretrained_weights")
    if pretrained_weights:
        model_cfg["pretrained_weights"] = _resolve_path(pretrained_weights, config_dir)

    cfg["_meta"] = {
        "config_path": str(config_path),
        "config_dir": str(config_dir),
    }
    return cfg


def strip_internal_metadata(cfg: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = deepcopy(cfg)
    cleaned.pop("_meta", None)
    return cleaned
