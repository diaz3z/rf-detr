import argparse
import shutil
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

    train_with_runtime_fallback(
        task=task,
        model_cfg=cfg.get("model") or {},
        class_names=class_names,
        train_kwargs=train_kwargs,
        logger=logger,
    )

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
