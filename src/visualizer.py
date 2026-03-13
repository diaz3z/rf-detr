from pathlib import Path
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def draw_detections(
    image: Image.Image,
    detections,
    class_names: List[str],
    score_threshold: float,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> int:
    image_np = np.array(image)

    fig = plt.figure(figsize=(12, 10))
    plt.imshow(image_np)
    ax = plt.gca()

    kept = 0

    boxes = detections.xyxy if hasattr(detections, "xyxy") else []
    scores = detections.confidence if hasattr(detections, "confidence") else []
    labels = detections.class_id if hasattr(detections, "class_id") else []

    for box, label, score in zip(boxes, labels, scores):
        if float(score) < score_threshold:
            continue

        x1, y1, x2, y2 = box
        class_name = class_names[int(label)] if 0 <= int(label) < len(class_names) else f"Class_{label}"

        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1, f"{class_name} {float(score):.2f}")
        kept += 1

    plt.axis("off")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return kept


def draw_segmentations(
    image: Image.Image,
    detections,
    class_names: List[str],
    score_threshold: float,
    mask_threshold: float,
    output_path: Optional[Path] = None,
    show: bool = False,
) -> int:
    image_np = np.array(image).copy()

    boxes = detections.xyxy if hasattr(detections, "xyxy") else []
    scores = detections.confidence if hasattr(detections, "confidence") else []
    labels = detections.class_id if hasattr(detections, "class_id") else []
    masks = detections.mask if hasattr(detections, "mask") else None

    kept = 0

    fig = plt.figure(figsize=(12, 10))
    plt.imshow(image_np)
    ax = plt.gca()

    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if float(score) < score_threshold:
            continue

        x1, y1, x2, y2 = box
        class_name = class_names[int(label)] if 0 <= int(label) < len(class_names) else f"Class_{label}"

        if masks is not None and idx < len(masks):
            mask = masks[idx]
            mask = np.array(mask)
            if mask.ndim > 2:
                mask = mask.squeeze()
            binary_mask = mask > mask_threshold
            colored = np.zeros((*binary_mask.shape, 4), dtype=np.float32)
            colored[..., 0] = 1.0
            colored[..., 3] = binary_mask.astype(np.float32) * 0.35
            ax.imshow(colored)

        rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, linewidth=2)
        ax.add_patch(rect)
        ax.text(x1, y1, f"{class_name} {float(score):.2f}")
        kept += 1

    plt.axis("off")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return kept