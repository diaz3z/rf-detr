import json
import logging
import random
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import torch


def setup_logging(log_path: Optional[Union[str, Path]] = None) -> logging.Logger:
    logger = logging.getLogger("rfdetr")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_path is not None:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_cfg: str) -> str:
    if str(device_cfg).lower() == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(device_cfg)


def ensure_exists(path: Union[str, Path], kind: str = "path") -> None:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"{kind} not found: {p}")


def ensure_file(path: Union[str, Path], kind: str = "file") -> None:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"{kind} not found: {p}")


def ensure_dir(path: Union[str, Path], kind: str = "directory") -> None:
    p = Path(path)
    if not p.is_dir():
        raise FileNotFoundError(f"{kind} not found: {p}")


def save_json(data: Any, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)


def save_text(text: str, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")
