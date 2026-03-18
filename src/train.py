import argparse
import json
import shutil
import threading
import time
from datetime import timedelta
from pathlib import Path

import torch

from checkpoints import find_trained_weights, save_deployable_model
from coco_utils import (
    has_segmentation_annotations,
    prepare_rfdetr_dataset,
    resolve_class_names,
    validate_coco_annotation_file,
    validate_rfdetr_category_ids,
)
from config import load_yaml, resolve_runtime_config, save_yaml, strip_internal_metadata
from pipeline import (
    build_model,
    build_train_kwargs,
    normalize_task,
    override_train_dataset_dir,
    resolve_output_dir,
)
from utils import (
    clear_directory,
    ensure_dir,
    ensure_file,
    resolve_device,
    save_json,
    set_seed,
    setup_logging,
)


def validate_config_paths(cfg: dict) -> None:
    ensure_dir(cfg["dataset"]["root_dir"], "dataset root_dir")
    ensure_dir(cfg["dataset"]["train"]["img_dir"], "train img_dir")
    ensure_file(cfg["dataset"]["train"]["ann_file"], "train ann_file")

    val_cfg = cfg["dataset"].get("val")
    if val_cfg:
        ensure_dir(val_cfg["img_dir"], "val img_dir")
        ensure_file(val_cfg["ann_file"], "val ann_file")

    test_cfg = cfg["dataset"].get("test")
    if test_cfg:
        ensure_dir(test_cfg["img_dir"], "test img_dir")
        ensure_file(test_cfg["ann_file"], "test ann_file")


def resolve_resume_checkpoint(cfg: dict, output_dir: Path) -> Path | None:
    resume_cfg = cfg.get("train", {}).get("resume")
    if not resume_cfg:
        return None

    if str(resume_cfg).lower() == "auto":
        auto_checkpoint = output_dir / "checkpoint.pth"
        return auto_checkpoint if auto_checkpoint.is_file() else None

    resume_path = Path(resume_cfg)
    ensure_file(resume_path, "resume checkpoint")
    return resume_path


def prepare_output_dir(cfg: dict, output_dir: Path) -> None:
    exist_ok = bool(cfg.get("output", {}).get("exist_ok", False))
    resume_cfg = cfg.get("train", {}).get("resume")

    if output_dir.exists() and exist_ok:
        if resume_cfg:
            raise ValueError(
                "output.exist_ok=true cannot be used together with train.resume. "
                "Use one of: overwrite the run folder, or resume from its checkpoint."
            )
        clear_directory(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)


def _is_cuda_engine_runtime_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    return "unable to find an engine" in message or "get was unable to find an engine" in message


def _parse_duration_to_seconds(value: str) -> float | None:
    if not value:
        return None
    parts = str(value).split(":")
    if len(parts) != 3:
        return None

    hours, minutes, seconds = parts
    try:
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except ValueError:
        return None


def _format_seconds(value: float | None) -> str:
    if value is None:
        return "unknown"
    return str(timedelta(seconds=int(round(value))))


def _extract_epoch_class_map(entry: dict, task: str) -> list[dict]:
    if task == "segmentation":
        metrics = entry.get("test_results_json_masks") or entry.get("ema_test_results_json_masks") or {}
    else:
        metrics = entry.get("test_results_json") or entry.get("ema_test_results_json") or {}
    return list(metrics.get("class_map") or [])


def _log_epoch_summary(logger, entry: dict, task: str, total_epochs: int, epoch_durations: list[float]) -> None:
    epoch_number = int(entry.get("epoch", -1)) + 1
    epoch_time_seconds = _parse_duration_to_seconds(entry.get("epoch_time"))
    if epoch_time_seconds is not None:
        epoch_durations.append(epoch_time_seconds)

    average_epoch_seconds = (
        sum(epoch_durations) / len(epoch_durations) if epoch_durations else None
    )
    estimated_total_seconds = (
        average_epoch_seconds * total_epochs if average_epoch_seconds is not None else None
    )
    estimated_remaining_seconds = (
        average_epoch_seconds * max(total_epochs - epoch_number, 0)
        if average_epoch_seconds is not None
        else None
    )

    logger.info(
        "Epoch %d/%d completed | epoch_time=%s | estimated_total=%s | remaining=%s",
        epoch_number,
        total_epochs,
        entry.get("epoch_time", "unknown"),
        _format_seconds(estimated_total_seconds),
        _format_seconds(estimated_remaining_seconds),
    )

    class_map = _extract_epoch_class_map(entry, task)
    if not class_map:
        logger.info("Epoch %d/%d | per-class mAP@50:95 unavailable", epoch_number, total_epochs)
        return

    all_entry = next(
        (item for item in class_map if str(item.get("class", "")).lower() == "all"),
        None,
    )
    if all_entry is not None:
        logger.info(
            "Epoch %d/%d | class=all | mAP@50:95=%.4f",
            epoch_number,
            total_epochs,
            float(all_entry.get("map@50:95", 0.0)),
        )

    for item in class_map:
        if str(item.get("class", "")).lower() == "all":
            continue
        logger.info(
            "Epoch %d/%d | class=%s | mAP@50:95=%.4f",
            epoch_number,
            total_epochs,
            item.get("class", "unknown"),
            float(item.get("map@50:95", 0.0)),
        )


def monitor_training_progress(
    log_file: Path,
    task: str,
    total_epochs: int,
    logger,
    stop_event: threading.Event,
) -> None:
    processed_epochs: set[int] = set()
    epoch_durations: list[float] = []
    file_position = 0

    while not stop_event.is_set():
        if not log_file.exists():
            time.sleep(2.0)
            continue

        with log_file.open("r", encoding="utf-8") as f:
            f.seek(file_position)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                epoch_index = entry.get("epoch")
                if epoch_index is None or int(epoch_index) in processed_epochs:
                    continue

                processed_epochs.add(int(epoch_index))
                _log_epoch_summary(logger, entry, task, total_epochs, epoch_durations)

            file_position = f.tell()

        time.sleep(2.0)


def log_final_class_map(output_dir: Path, task: str, logger) -> None:
    results_file = output_dir / ("results_mask.json" if task == "segmentation" else "results.json")
    if not results_file.is_file():
        logger.info("Final per-class mAP@50:95 file not found at %s", results_file)
        return

    try:
        results = json.loads(results_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.info("Final per-class mAP@50:95 file could not be parsed: %s", results_file)
        return

    class_map = results.get("class_map", {})
    split_results = class_map.get("test") or class_map.get("valid") or []
    if not split_results:
        logger.info("Final per-class mAP@50:95 unavailable in %s", results_file)
        return

    all_entry = next(
        (item for item in split_results if str(item.get("class", "")).lower() == "all"),
        None,
    )
    if all_entry is not None:
        logger.info(
            "Final | class=all | mAP@50:95=%.4f",
            float(all_entry.get("map@50:95", 0.0)),
        )

    logger.info("Final per-class mAP@50:95")
    for item in split_results:
        if str(item.get("class", "")).lower() == "all":
            continue
        logger.info(
            "Final | class=%s | mAP@50:95=%.4f",
            item.get("class", "unknown"),
            float(item.get("map@50:95", 0.0)),
        )


def train_with_runtime_fallback(
    task: str,
    model_cfg: dict,
    class_names: list[str],
    train_kwargs: dict,
    logger,
) -> None:
    try:
        model = build_model(task, model_cfg, class_names=class_names)
        model.train(**train_kwargs)
        return
    except RuntimeError as exc:
        if not _is_cuda_engine_runtime_error(exc):
            raise

        logger.warning("RF-DETR training hit a CUDA/cuDNN engine error: %s", exc)
        logger.warning("Retrying once with cuDNN disabled and AMP disabled")

        retry_kwargs = dict(train_kwargs)
        retry_kwargs["amp"] = False
        previous_cudnn_enabled = torch.backends.cudnn.enabled
        torch.backends.cudnn.enabled = False

        try:
            retry_model_cfg = dict(model_cfg or {})
            retry_model = build_model(task, retry_model_cfg, class_names=class_names)
            retry_model.train(**retry_kwargs)
        finally:
            torch.backends.cudnn.enabled = previous_cudnn_enabled


def run_training(config_path: str) -> Path:
    cfg = resolve_runtime_config(load_yaml(config_path), config_path)
    task = normalize_task(cfg.get("task", "detection"))

    set_seed(int(cfg.get("seed", 42)))
    validate_config_paths(cfg)

    train_ann = cfg["dataset"]["train"]["ann_file"]
    validate_coco_annotation_file(train_ann)

    if task == "segmentation" and not has_segmentation_annotations(train_ann):
        raise ValueError(
            "Segmentation task selected but no segmentation annotations were found "
            "in the training COCO file"
        )

    output_dir = resolve_output_dir(cfg)
    prepare_output_dir(cfg, output_dir)
    persisted_cfg = strip_internal_metadata(cfg)

    logger = setup_logging(output_dir / "train.log")
    logger.info("Starting RF-DETR training")
    logger.info("Task = %s", task)

    if cfg["output"].get("save_config_copy", True):
        save_yaml(persisted_cfg, output_dir / "config_resolved.yaml")

    class_names, cat_id_to_contig, contig_to_cat_id = resolve_class_names(
        ann_file=train_ann,
        cfg_class_names=cfg.get("model", {}).get("class_names"),
    )

    logger.info("Dataset root_dir = %s", cfg["dataset"]["root_dir"])
    logger.info("Classes = %s", class_names)
    logger.info("Class count = %d", len(class_names))

    device = resolve_device(cfg["train"].get("device", cfg.get("device", "auto")))
    logger.info("Device = %s", device)

    train_kwargs = build_train_kwargs(cfg, output_dir)
    total_epochs = int(cfg.get("train", {}).get("num_epochs", 1))
    resume_checkpoint = resolve_resume_checkpoint(cfg, output_dir)
    prepared_dataset_dir = prepare_rfdetr_dataset(
        cfg["dataset"],
        output_dir / "_prepared_rfdetr_dataset",
    )
    train_kwargs = override_train_dataset_dir(train_kwargs, prepared_dataset_dir)
    if resume_checkpoint is not None:
        train_kwargs["resume"] = str(resume_checkpoint)
    train_kwargs["device"] = device
    logger.info("Train kwargs = %s", train_kwargs)
    logger.info("Prepared RF-DETR dataset = %s", prepared_dataset_dir)
    if resume_checkpoint is not None:
        logger.info("Resuming training from %s", resume_checkpoint)
    elif str(cfg.get("train", {}).get("resume", "")).lower() == "auto":
        logger.info("Auto resume enabled, but no checkpoint.pth was found in %s", output_dir)

    try:
        validate_rfdetr_category_ids(train_ann)
    except ValueError:
        logger.info(
            "Source COCO category ids are not RF-DETR ready; using normalized prepared dataset"
        )

    progress_stop_event = threading.Event()
    progress_thread = threading.Thread(
        target=monitor_training_progress,
        args=(output_dir / "log.txt", task, total_epochs, logger, progress_stop_event),
        daemon=True,
    )
    progress_thread.start()

    try:
        train_with_runtime_fallback(
            task=task,
            model_cfg=cfg.get("model") or {},
            class_names=class_names,
            train_kwargs=train_kwargs,
            logger=logger,
        )
    finally:
        progress_stop_event.set()
        progress_thread.join(timeout=5.0)

    final_model_path = output_dir / cfg["output"].get(
        "final_model_name",
        f"rfdetr_{task}_model.pth",
    )
    trained_weights_path = find_trained_weights(output_dir)

    save_deployable_model(
        path=final_model_path,
        trained_weights_path=trained_weights_path,
        class_names=class_names,
        task=task,
        model_config=persisted_cfg.get("model", {}),
        metrics={"status": "completed", "task": task},
    )

    save_json(
        {
            "task": task,
            "class_names": class_names,
            "cat_id_to_contig": cat_id_to_contig,
            "contig_to_cat_id": contig_to_cat_id,
            "dataset_root": cfg["dataset"]["root_dir"],
            "prepared_dataset_root": None,
            "prepared_dataset_deleted": True,
        },
        output_dir / "label_metadata.json",
    )

    shutil.rmtree(prepared_dataset_dir, ignore_errors=True)

    logger.info("Training finished")
    logger.info("Trained weights discovered at %s", trained_weights_path)
    logger.info("Deployable model saved to %s", final_model_path)
    log_final_class_map(output_dir, task, logger)
    logger.info("Prepared RF-DETR dataset removed from %s", prepared_dataset_dir)
    logger.info("RF-DETR training artifacts are saved by the library under %s", output_dir)
    return final_model_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an RF-DETR detection or segmentation model from a YAML config.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/detection.yaml",
        help="Path to a training config YAML file.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
