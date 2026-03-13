import argparse
from pathlib import Path

import supervision as sv
from PIL import Image
from supervision.metrics import MeanAveragePrecision
from tqdm import tqdm

from checkpoints import (
    extract_class_names,
    extract_model_config,
    extract_task,
    load_deployable_model,
    with_embedded_weights_file,
)
from config import load_yaml, resolve_runtime_config
from pipeline import build_model, normalize_task, resolve_output_dir
from utils import ensure_dir, ensure_file, save_json


def load_detection_dataset(images_path: str, annotations_path: str):
    return sv.DetectionDataset.from_coco(
        images_directory_path=images_path,
        annotations_path=annotations_path,
    )


def run_validation(
    config_path: str,
    model_path: str,
    threshold: float = None,
) -> Path:
    cfg = resolve_runtime_config(load_yaml(config_path), config_path)
    ensure_file(model_path, "model")
    bundle = load_deployable_model(model_path)

    threshold = 0.5 if threshold is None else float(threshold)
    class_names = extract_class_names(bundle)
    task = normalize_task(extract_task(bundle))
    model_config = extract_model_config(bundle)

    test_cfg = cfg["dataset"]["test"]
    ensure_dir(test_cfg["img_dir"], "test img_dir")
    ensure_file(test_cfg["ann_file"], "test ann_file")

    ds = load_detection_dataset(
        images_path=test_cfg["img_dir"],
        annotations_path=test_cfg["ann_file"],
    )

    predictions = []
    targets = []

    def _run_with_model(weights_path: Path) -> None:
        runtime_model_config = dict(model_config)
        runtime_model_config["pretrained_weights"] = str(weights_path)
        model = build_model(task, runtime_model_config, class_names=class_names)

        for path, _, annotations in tqdm(ds, desc=f"Validating {task}"):
            image = Image.open(path).convert("RGB")
            detections = model.predict(image, threshold=threshold)
            predictions.append(detections)
            targets.append(annotations)

    with_embedded_weights_file(bundle, _run_with_model)

    metric = MeanAveragePrecision()
    result = metric.update(predictions, targets).compute()

    print("Validation done")
    print(result)

    output_dir = resolve_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)

    confusion_matrix = sv.ConfusionMatrix.from_detections(
        predictions=predictions,
        targets=targets,
        classes=class_names,
    )
    plot = confusion_matrix.plot()
    plot_path = output_dir / "confusion_matrix.png"
    plot.figure.savefig(plot_path, bbox_inches="tight")

    metrics_path = output_dir / "validation_metrics.json"
    save_json(result, metrics_path)

    print(f"Saved metrics to {metrics_path}")
    print(f"Saved confusion matrix to {plot_path}")
    return metrics_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate an RF-DETR run against the test split in the config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detection.yaml",
        help="Path to the YAML config used for training.",
    )
    parser.add_argument("--model", type=str, required=True, help="Path to the deployable model produced by training.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override the score threshold used during prediction.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_validation(
        config_path=args.config,
        model_path=args.model,
        threshold=args.threshold,
    )


if __name__ == "__main__":
    main()
