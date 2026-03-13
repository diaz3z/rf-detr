import json
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
