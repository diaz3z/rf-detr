import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union


PLACEHOLDER_SUPERCATEGORIES = {"", "none", "null", None}


def load_coco_json(ann_file: Union[str, Path]) -> Dict[str, Any]:
    ann_path = Path(ann_file)
    if not ann_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {ann_path}")

    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Invalid COCO annotation format in {ann_path}")

    return data


def read_coco_categories(ann_file: Union[str, Path]) -> List[Dict[str, Any]]:
    data = load_coco_json(ann_file)
    categories = data.get("categories", [])

    if not categories:
        raise ValueError(f"No categories found in {ann_file}")

    if not all(isinstance(cat, dict) for cat in categories):
        raise ValueError(f"Invalid categories structure in {ann_file}")

    return categories


def select_leaf_categories(categories: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not categories:
        return []

    has_any_supercategory = any(
        cat.get("supercategory", "none") not in PLACEHOLDER_SUPERCATEGORIES
        for cat in categories
    )
    if not has_any_supercategory:
        return categories

    parent_names = {
        cat.get("supercategory")
        for cat in categories
        if cat.get("supercategory", "none") not in PLACEHOLDER_SUPERCATEGORIES
    }
    has_children = {
        str(cat["name"])
        for cat in categories
        if str(cat.get("name")) in parent_names
    }

    leaf_categories = [cat for cat in categories if str(cat["name"]) not in has_children]
    return leaf_categories or categories


def build_label_maps_from_coco(
    ann_file: Union[str, Path],
) -> Tuple[List[str], Dict[int, int], Dict[int, int]]:
    categories = select_leaf_categories(read_coco_categories(ann_file))
    categories_sorted = sorted(categories, key=lambda x: int(x["id"]))

    class_names = [str(cat["name"]) for cat in categories_sorted]

    cat_id_to_contig = {}
    contig_to_cat_id = {}

    for idx, cat in enumerate(categories_sorted):
        cat_id = int(cat["id"])
        cat_id_to_contig[cat_id] = idx
        contig_to_cat_id[idx] = cat_id

    return class_names, cat_id_to_contig, contig_to_cat_id


def build_category_id_remap(ann_file: Union[str, Path]) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    categories = select_leaf_categories(read_coco_categories(ann_file))
    categories_sorted = sorted(categories, key=lambda x: int(x["id"]))

    remap: Dict[int, int] = {}
    normalized_categories: List[Dict[str, Any]] = []

    for idx, category in enumerate(categories_sorted):
        source_id = int(category["id"])
        remap[source_id] = idx

        normalized_category = dict(category)
        normalized_category["id"] = idx
        normalized_categories.append(normalized_category)

    return normalized_categories, remap


def resolve_class_names(
    ann_file: Union[str, Path],
    cfg_class_names: List[str] = None,
) -> Tuple[List[str], Dict[int, int], Dict[int, int]]:
    coco_class_names, cat_id_to_contig, contig_to_cat_id = build_label_maps_from_coco(
        ann_file
    )

    if cfg_class_names:
        cfg_names = [str(x) for x in cfg_class_names]

        if len(cfg_names) != len(coco_class_names):
            raise ValueError(
                "class_names length mismatch "
                f"cfg={len(cfg_names)} coco={len(coco_class_names)}"
            )

        class_names = cfg_names
    else:
        class_names = coco_class_names

    return class_names, cat_id_to_contig, contig_to_cat_id


def get_num_classes(ann_file: Union[str, Path]) -> int:
    class_names, _, _ = build_label_maps_from_coco(ann_file)
    return len(class_names)


def validate_coco_annotation_file(ann_file: Union[str, Path]) -> None:
    data = load_coco_json(ann_file)

    required_top_level_keys = ["images", "annotations", "categories"]
    for key in required_top_level_keys:
        if key not in data:
            raise ValueError(f"Missing key '{key}' in {ann_file}")

    if not isinstance(data["images"], list):
        raise ValueError(f"'images' must be a list in {ann_file}")

    if not isinstance(data["annotations"], list):
        raise ValueError(f"'annotations' must be a list in {ann_file}")

    if not isinstance(data["categories"], list):
        raise ValueError(f"'categories' must be a list in {ann_file}")

    if not data["categories"]:
        raise ValueError(f"No categories found in {ann_file}")

    leaf_categories = select_leaf_categories(data["categories"])
    leaf_category_ids = {int(cat["id"]) for cat in leaf_categories}
    invalid_annotation_ids = sorted(
        {
            int(ann["category_id"])
            for ann in data["annotations"]
            if int(ann["category_id"]) not in leaf_category_ids
        }
    )
    if invalid_annotation_ids:
        raise ValueError(
            "Annotation category ids do not map to trainable leaf categories in "
            f"{ann_file}: {invalid_annotation_ids}"
        )


def validate_rfdetr_category_ids(ann_file: Union[str, Path]) -> None:
    categories = select_leaf_categories(read_coco_categories(ann_file))
    category_ids = sorted(int(cat["id"]) for cat in categories)
    expected_ids = list(range(len(category_ids)))

    if category_ids != expected_ids:
        raise ValueError(
            "RF-DETR expects trainable category ids to be contiguous and zero-based. "
            f"Found category ids {category_ids} in {ann_file}, expected {expected_ids}. "
            "Re-export the dataset with category ids 0..N-1 or remap the COCO categories "
            "before training."
        )


def normalize_coco_annotation_data(ann_file: Union[str, Path]) -> Dict[str, Any]:
    data = load_coco_json(ann_file)
    normalized_categories, remap = build_category_id_remap(ann_file)

    normalized_annotations: List[Dict[str, Any]] = []
    for annotation in data.get("annotations", []):
        normalized_annotation = dict(annotation)
        category_id = int(normalized_annotation["category_id"])
        if category_id not in remap:
            raise ValueError(
                f"Annotation category_id {category_id} in {ann_file} does not map "
                "to a trainable leaf category"
            )
        normalized_annotation["category_id"] = remap[category_id]
        normalized_annotations.append(normalized_annotation)

    normalized_data = dict(data)
    normalized_data["categories"] = normalized_categories
    normalized_data["annotations"] = normalized_annotations
    return normalized_data


def _link_or_copy_file(src: Path, dst: Path) -> None:
    if dst.exists():
        dst.unlink()

    try:
        os.link(src, dst)
        return
    except OSError:
        pass

    try:
        os.symlink(src, dst)
        return
    except OSError:
        pass

    shutil.copy2(src, dst)


def _mirror_split_images(src_dir: Path, dst_dir: Path, ann_file_name: str) -> None:
    for item in src_dir.iterdir():
        if not item.is_file():
            continue
        if item.name == ann_file_name:
            continue
        _link_or_copy_file(item, dst_dir / item.name)


def _dataset_prep_manifest(dataset_cfg: Dict[str, Any]) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {"splits": {}}

    for split_name in ("train", "val", "test"):
        split_cfg = dataset_cfg.get(split_name)
        if not split_cfg:
            continue

        img_dir = Path(split_cfg["img_dir"])
        ann_file = Path(split_cfg["ann_file"])
        image_files = sorted(
            item.name
            for item in img_dir.iterdir()
            if item.is_file() and item.name != ann_file.name
        )
        manifest["splits"][split_name] = {
            "img_dir": str(img_dir.resolve()),
            "ann_file": str(ann_file.resolve()),
            "ann_mtime_ns": ann_file.stat().st_mtime_ns,
            "ann_size": ann_file.stat().st_size,
            "image_count": len(image_files),
            "image_names": image_files,
        }

    return manifest


def prepare_rfdetr_dataset(dataset_cfg: Dict[str, Any], prepared_root: Union[str, Path]) -> str:
    prepared_root = Path(prepared_root)
    manifest_path = prepared_root / ".prep_manifest.json"
    current_manifest = _dataset_prep_manifest(dataset_cfg)

    if prepared_root.exists() and manifest_path.exists():
        try:
            cached_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if cached_manifest == current_manifest:
                return str(prepared_root)
        except (json.JSONDecodeError, OSError):
            pass

    if prepared_root.exists():
        shutil.rmtree(prepared_root)
    prepared_root.mkdir(parents=True, exist_ok=True)

    split_folder_names = {"train": "train", "val": "valid", "test": "test"}

    for split_name, prepared_split_name in split_folder_names.items():
        split_cfg = dataset_cfg.get(split_name)
        if not split_cfg:
            continue

        src_img_dir = Path(split_cfg["img_dir"])
        src_ann_file = Path(split_cfg["ann_file"])
        dst_split_dir = prepared_root / prepared_split_name
        dst_split_dir.mkdir(parents=True, exist_ok=True)

        _mirror_split_images(src_img_dir, dst_split_dir, src_ann_file.name)

        normalized_ann = normalize_coco_annotation_data(src_ann_file)
        dst_ann_file = dst_split_dir / "_annotations.coco.json"
        with dst_ann_file.open("w", encoding="utf-8") as f:
            json.dump(normalized_ann, f, indent=2, ensure_ascii=False)

    manifest_path.write_text(
        json.dumps(current_manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return str(prepared_root)


def has_segmentation_annotations(ann_file: Union[str, Path]) -> bool:
    data = load_coco_json(ann_file)
    annotations = data.get("annotations", [])

    for ann in annotations:
        segmentation = ann.get("segmentation")
        if segmentation:
            return True

    return False


def summarize_coco_dataset(ann_file: Union[str, Path]) -> Dict[str, Any]:
    data = load_coco_json(ann_file)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    categories = data.get("categories", [])

    class_names, cat_id_to_contig, contig_to_cat_id = build_label_maps_from_coco(
        ann_file
    )

    return {
        "num_images": len(images),
        "num_annotations": len(annotations),
        "num_categories": len(categories),
        "class_names": class_names,
        "cat_id_to_contig": cat_id_to_contig,
        "contig_to_cat_id": contig_to_cat_id,
        "has_segmentation": has_segmentation_annotations(ann_file),
    }
