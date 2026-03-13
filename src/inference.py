import argparse
from pathlib import Path
from typing import List, Tuple


import cv2
import numpy as np
from PIL import Image

from checkpoints import (
    extract_class_names,
    extract_model_config,
    extract_task,
    load_deployable_model,
    with_embedded_weights_file,
)
from pipeline import build_model, normalize_task
from utils import ensure_file
from visualizer import draw_detections, draw_segmentations


IMAGE_EXTENSIONS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def collect_images(single_image: str, image_dir: str) -> List[Path]:
    images: List[Path] = []

    if single_image:
        images.append(Path(single_image))

    if image_dir:
        image_dir = Path(image_dir)
        for ext in IMAGE_EXTENSIONS:
            images.extend(sorted(image_dir.glob(ext)))

    deduped = []
    seen = set()
    for path in images:
        key = str(path.resolve())
        if key not in seen:
            deduped.append(path)
            seen.add(key)

    return deduped


def _resolve_thresholds(score_threshold: float, mask_threshold: float) -> Tuple[float, float]:
    resolved_score_threshold = 0.5 if score_threshold is None else float(score_threshold)
    resolved_mask_threshold = 0.5 if mask_threshold is None else float(mask_threshold)
    return resolved_score_threshold, resolved_mask_threshold


def _annotate_frame(
    frame_bgr: np.ndarray,
    detections,
    class_names: List[str],
    task: str,
    score_threshold: float,
    mask_threshold: float,
) -> Tuple[np.ndarray, int]:
    annotated = frame_bgr.copy()
    boxes = detections.xyxy if hasattr(detections, "xyxy") else []
    scores = detections.confidence if hasattr(detections, "confidence") else []
    labels = detections.class_id if hasattr(detections, "class_id") else []
    masks = detections.mask if hasattr(detections, "mask") else None

    kept = 0
    for idx, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if float(score) < score_threshold:
            continue

        x1, y1, x2, y2 = [int(v) for v in box]
        class_index = int(label)
        class_name = (
            class_names[class_index]
            if 0 <= class_index < len(class_names)
            else f"Class_{class_index}"
        )

        if task == "segmentation" and masks is not None and idx < len(masks):
            mask = np.array(masks[idx])
            if mask.ndim > 2:
                mask = mask.squeeze()
            binary_mask = mask > mask_threshold
            if binary_mask.shape[:2] != annotated.shape[:2]:
                binary_mask = cv2.resize(
                    binary_mask.astype(np.uint8),
                    (annotated.shape[1], annotated.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            overlay = annotated.copy()
            overlay[binary_mask] = (0, 0, 255) # Red color for mask overlay
            annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            annotated,
            f"{class_name} {float(score):.2f}",
            (x1, max(y1 - 10, 0)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        kept += 1

    return annotated, kept


def _run_image_inference(
    model,
    task: str,
    class_names: List[str],
    image_paths: List[Path],
    output_dir: Path,
    score_threshold: float,
    mask_threshold: float,
    show: bool,
    save: bool,
) -> None:
    for image_path in image_paths:
        ensure_file(image_path, "image")
        pil_image = Image.open(image_path).convert("RGB")
        detections = model.predict(pil_image, threshold=score_threshold)

        out_path = output_dir / f"{image_path.stem}_pred{image_path.suffix}"
        save_path = out_path if save else None

        if task == "segmentation":
            kept = draw_segmentations(
                image=pil_image,
                detections=detections,
                class_names=class_names,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
                output_path=save_path,
                show=bool(show),
            )
        else:
            kept = draw_detections(
                image=pil_image,
                detections=detections,
                class_names=class_names,
                score_threshold=score_threshold,
                output_path=save_path,
                show=bool(show),
            )

        destination = out_path if save else "<not saved>"
        print(f"{image_path} -> {destination} | predictions_above_threshold={kept}")


def _build_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    safe_fps = fps if fps and fps > 0 else 30.0
    return cv2.VideoWriter(str(output_path), fourcc, safe_fps, (width, height))


def _run_video_inference(
    model,
    task: str,
    class_names: List[str],
    source,
    output_path: Path,
    score_threshold: float,
    mask_threshold: float,
    show: bool,
    save: bool,
    max_frames: int = 0,
) -> Path:
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise ValueError(f"Unable to open video source: {source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
    writer = _build_video_writer(output_path, fps, width, height) if save else None

    frame_count = 0
    kept_total = 0

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            detections = model.predict(pil_image, threshold=score_threshold)

            annotated_frame, kept = _annotate_frame(
                frame_bgr=frame_bgr,
                detections=detections,
                class_names=class_names,
                task=task,
                score_threshold=score_threshold,
                mask_threshold=mask_threshold,
            )
            if writer is not None:
                writer.write(annotated_frame)

            frame_count += 1
            kept_total += kept

            if show:
                cv2.imshow("RF-DETR Inference", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if max_frames and frame_count >= max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if show:
            cv2.destroyAllWindows()

    destination = output_path if save else "<not saved>"
    print(
        f"{source} -> {destination} | frames_processed={frame_count} | "
        f"predictions_above_threshold={kept_total}"
    )
    return output_path


def run_inference(
    model_path: str,
    image: str = "",
    image_dir: str = "",
    video: str = "",
    webcam: bool = False,
    webcam_index: int = 0,
    output_dir: str = "",
    score_threshold: float = None,
    mask_threshold: float = None,
    save: bool = False,
    show: bool = False,
    max_frames: int = 0,
) -> Path:
    modes_selected = sum(bool(value) for value in (image, image_dir, video)) + int(webcam)
    if modes_selected == 0:
        raise ValueError("Provide one input source: --image, --image-dir, --video, or --webcam")
    if modes_selected > 1:
        raise ValueError("Use only one input source at a time")

    ensure_file(model_path, "model")
    bundle = load_deployable_model(model_path)
    class_names = extract_class_names(bundle)
    task = normalize_task(extract_task(bundle))
    model_config = dict(extract_model_config(bundle))

    resolved_output_dir = Path(output_dir or (Path(model_path).resolve().parent / "predictions"))
    resolved_score_threshold, resolved_mask_threshold = _resolve_thresholds(
        score_threshold=score_threshold,
        mask_threshold=mask_threshold,
    )

    def _run_with_model(weights_path: Path) -> Path:
        runtime_model_config = dict(model_config)
        runtime_model_config["pretrained_weights"] = str(weights_path)
        model = build_model(task, runtime_model_config, class_names=class_names)
        if image or image_dir:
            image_paths = collect_images(image, image_dir)
            if not image_paths:
                raise ValueError("No images found")
            _run_image_inference(
                model=model,
                task=task,
                class_names=class_names,
                image_paths=image_paths,
                output_dir=resolved_output_dir,
                score_threshold=resolved_score_threshold,
                mask_threshold=resolved_mask_threshold,
                show=show,
                save=save,
            )
            return resolved_output_dir

        if video:
            video_path = Path(video)
            ensure_file(video_path, "video")
            output_path = resolved_output_dir / f"{video_path.stem}_pred.mp4"
            return _run_video_inference(
                model=model,
                task=task,
                class_names=class_names,
                source=str(video_path),
                output_path=output_path,
                score_threshold=resolved_score_threshold,
                mask_threshold=resolved_mask_threshold,
                show=show,
                save=save,
                max_frames=max_frames,
            )

        output_path = resolved_output_dir / f"webcam_{webcam_index}_pred.mp4"
        return _run_video_inference(
            model=model,
            task=task,
            class_names=class_names,
            source=webcam_index,
            output_path=output_path,
            score_threshold=resolved_score_threshold,
            mask_threshold=resolved_mask_threshold,
            show=show,
            save=save,
            max_frames=max_frames,
        )
    return with_embedded_weights_file(bundle, _run_with_model)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run RF-DETR inference from one deployable model file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the deployable model produced by training.",
    )
    parser.add_argument("--image", type=str, default="", help="Path to a single image.")
    parser.add_argument(
        "--image-dir",
        type=str,
        default="",
        help="Path to a directory of images. Supported extensions: jpg, jpeg, png, bmp, webp.",
    )
    parser.add_argument("--video", type=str, default="", help="Path to a video file.")
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Use a webcam as the input source.",
    )
    parser.add_argument(
        "--webcam-index",
        type=int,
        default=0,
        help="Webcam device index to open when --webcam is used.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Directory where prediction images or videos will be saved. Defaults to <model_dir>/predictions.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="Score threshold used during prediction. Defaults to 0.5.",
    )
    parser.add_argument(
        "--mask-threshold",
        type=float,
        default=None,
        help="Mask threshold used for segmentation overlays. Defaults to 0.5.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save output images or videos to disk.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Optional frame limit for video or webcam inference. Use 0 for no limit.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display video/webcam frames live. Press q to stop.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_inference(
        model_path=args.model,
        image=args.image,
        image_dir=args.image_dir,
        video=args.video,
        webcam=args.webcam,
        webcam_index=args.webcam_index,
        output_dir=args.output_dir,
        score_threshold=args.score_threshold,
        mask_threshold=args.mask_threshold,
        save=args.save,
        show=args.show,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
