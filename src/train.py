import argparse
import shutil
from pathlib import Path

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
    output_dir.mkdir(parents=True, exist_ok=True)
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

    model = build_model(task, cfg.get("model"), class_names=class_names)

    train_kwargs = build_train_kwargs(cfg, output_dir)
    prepared_dataset_dir = prepare_rfdetr_dataset(
        cfg["dataset"],
        output_dir / "_prepared_rfdetr_dataset",
    )
    train_kwargs = override_train_dataset_dir(train_kwargs, prepared_dataset_dir)
    train_kwargs["device"] = device
    logger.info("Train kwargs = %s", train_kwargs)
    logger.info("Prepared RF-DETR dataset = %s", prepared_dataset_dir)

    try:
        validate_rfdetr_category_ids(train_ann)
    except ValueError:
        logger.info(
            "Source COCO category ids are not RF-DETR ready; using normalized prepared dataset"
        )

    model.train(**train_kwargs)

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
