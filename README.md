# RF-DETR Training Project

This repository provides a Windows-first workflow for training, validating, and running inference with RF-DETR on a COCO-format dataset. The project is organized so a new user can install dependencies, update one config file, train with an explicit detection or segmentation script, and use a single deployable model file for inference.

## What this repo does

- Train an RF-DETR detection model or segmentation model from YAML config.
- Validate the trained run on the `test` split from the same config.
- Run inference on one image or a directory of images.
- Save logs, resolved config, validation metrics, and deployable model files to `outputs/`.

## Repository layout

```text
rf-detr/
|-- requirements.txt
|-- configs/
|   |-- detection.yaml
|   `-- segmentation.yaml
|-- src/
|   |-- train.py
|   |-- train_detection.py
|   |-- train_segmentation.py
|   |-- validate.py
|   |-- inference.py
|   |-- pipeline.py
|   `-- ...
`-- outputs/
```

## Prerequisites

- Windows 10/11
- Python 3.10 or newer
- A COCO-format dataset with this structure:

```text
your-dataset/
|-- train/
|   |-- _annotations.coco.json
|   `-- images...
|-- valid/
|   |-- _annotations.coco.json
|   `-- images...
`-- test/
    |-- _annotations.coco.json
    `-- images...
```

## 1. Create a virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
```

## 2. Install PyTorch

Install the correct PyTorch build for your machine first. Use the official PyTorch selector if you need CUDA-specific commands.

CPU example:

```powershell
pip install torch torchvision
```

## 3. Install project dependencies

```powershell
pip install -r requirements.txt
```

## 4. Prepare your dataset

Place your dataset under `data/` or point the config to any other folder on disk.

Recommended local structure:

```text
rf-detr/
`-- data/
    `-- person.v1i.coco/
        |-- train/
        |-- valid/
        `-- test/
```

## 5. Update the config

Edit one of these files:

- `configs/detection.yaml`
- `configs/segmentation.yaml`

You usually only need to update:

- `dataset.root_dir`
- `model.class_names`
- `train.num_epochs`
- `train.batch_size`
- `train.device`
- `train.resolution`
- `output.run_name`

Important notes:

- `dataset.root_dir` can be relative or absolute.
- Split paths are resolved relative to `dataset.root_dir`.
- Detection uses `rf-detr-base.pth` by default.
- Segmentation uses `rf-detr-seg-preview.pt` by default.
- The final model file already contains the metadata needed for inference.
- For detection, use a `train.resolution` divisible by `56`.
- Roboflow COCO exports with a parent category like `objects` are supported; the code uses only leaf categories for training.

## 6. Train

Detection:

```powershell
python src\train_detection.py
```

Segmentation:

```powershell
python src\train_segmentation.py
```

Optional custom config override:

```powershell
python src\train_detection.py --config configs/detection.yaml
python src\train_segmentation.py --config configs/segmentation.yaml
```

Training outputs are written under:

```text
outputs/<run_name>/
```

Expected files include:

- `train.log`
- `config_resolved.yaml`
- `rfdetr_*_model.pth`

## 7. Validate

Validation uses the `test` split from the config and saves metrics plus a confusion matrix under the run output directory.

```powershell
python src\validate.py --config configs/segmentation.yaml --model outputs\rfdetr_face_segmentation\rfdetr_segmentation_model.pth
```

Optional:

```powershell
python src\validate.py --config configs/segmentation.yaml --model outputs\rfdetr_face_segmentation\rfdetr_segmentation_model.pth --threshold 0.4
```

Generated files:

- `outputs/<run_name>/validation_metrics.json`
- `outputs/<run_name>/confusion_matrix.png`

## 8. Run inference

Single image:

```powershell
python src\inference.py --model outputs\rfdetr_face_segmentation\rfdetr_segmentation_model.pth --image path\to\image.jpg --save
```

Without `--save`, inference runs but does not write files to disk.

Directory of images:

```powershell
python src\inference.py --model outputs\rfdetr_face_segmentation\rfdetr_segmentation_model.pth --image-dir path\to\images --save
```

Custom output folder:

```powershell
python src\inference.py --model outputs\rfdetr_face_segmentation\rfdetr_segmentation_model.pth --image-dir path\to\images --output-dir outputs\predictions
```

Video:

```powershell
python src\inference.py --model outputs\rfdetr_face_segmentation\rfdetr_segmentation_model.pth --video path\to\video.mp4 --save
```

Webcam:

```powershell
python src\inference.py --model outputs\rfdetr_face_segmentation\rfdetr_segmentation_model.pth --webcam --show
```

Save webcam output:

```powershell
python src\inference.py --model outputs\rfdetr_face_segmentation\rfdetr_segmentation_model.pth --webcam --show --save
```

## Command summary

```powershell
python src\train_detection.py
python src\validate.py --config configs/detection.yaml --model outputs\rfdetr_face_detection\rfdetr_detection_model.pth
python src\inference.py --model outputs\rfdetr_face_detection\rfdetr_detection_model.pth --image path\to\image.jpg
```

## Config reference

### Shared fields

```yaml
task: detection | segmentation
seed: 42
dataset:
  root_dir: ../data/person.v1i.coco
  train:
    img_dir: train
    ann_file: train/_annotations.coco.json
  val:
    img_dir: valid
    ann_file: valid/_annotations.coco.json
  test:
    img_dir: test
    ann_file: test/_annotations.coco.json
model:
  class_names:
    - class_1
    - class_2
  pretrained_weights: ../rf-detr-base.pth
train:
  num_epochs: 5
  batch_size: 4
  grad_accum_steps: 1
  lr: 0.0001
  weight_decay: 0.0001
  num_workers: 2
  device: cuda
  resolution: 560
  amp: true
output:
  dir: ../outputs
  run_name: my_run
  final_model_name: rfdetr_detection_model.pth
  save_config_copy: true
```

### Segmentation-only fields

```yaml
model:
  pretrained_weights: ../rf-detr-seg-preview.pt
  segmentation_head: true
```

## Troubleshooting

### `ModuleNotFoundError: No module named 'torch'`

Install PyTorch before `pip install -r requirements.txt`.

### `dataset root_dir not found`

Check `dataset.root_dir` in your config. Relative paths are resolved from the config file location.

### `Backbone requires input shape to be divisible by 56`

Use a detection `train.resolution` divisible by `56`, such as `448` or `560`.

### `Segmentation task selected but no segmentation annotations were found`

Your COCO annotation file does not contain segmentation polygons or masks. Use detection mode or export a segmentation dataset.

### Validation or inference cannot find the model file

Train first, or pass `--model` explicitly.

## GitHub upload guidance

This repository now includes a `.gitignore` that excludes:

- model weights
- generated outputs
- local datasets
- editor files
- Python cache files

If you need to publish weights, use a separate model registry, release asset, or cloud storage link instead of committing them to Git.
