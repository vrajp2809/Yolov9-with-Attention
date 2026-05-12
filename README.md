# YOLOv9 with Channel Attention

This repository contains a notebook-based implementation of **YOLOv9 enhanced with Channel Attention (CBAM-style)** for object detection on Pascal VOC.

The project workflow is implemented in:

- `yolov9 with attention.ipynb`

---

## Overview

The notebook performs the full pipeline:

1. Downloads and prepares Pascal VOC data.
2. Converts VOC XML annotations to YOLO label format.
3. Clones the official YOLOv9 repository.
4. Adds a custom `ChannelAttention` module.
5. Integrates attention into a custom model config (`gelan-c-ca.yaml`).
6. Trains the modified model (single GPU or DDP).

---

## Attention Changes Implemented

The notebook introduces these key modifications to YOLOv9:

- Adds `models/attention.py` with:
  - `ChannelAttention`
  - `C2f_CA` compatibility wrapper
- Updates `models/yolo.py` imports so attention modules can be referenced in YAML.
- Creates a custom model config (`models/detect/gelan-c-ca.yaml`) that inserts **ChannelAttention** after deeper backbone stages.
- Adjusts `nc` from 80 to 20 for Pascal VOC experiments.
- Includes fixes related to deterministic training behavior when using max-based attention operations.

---

## Repository Structure

```text
.
├── yolov9 with attention.ipynb   # Main end-to-end notebook
└── README.md                     # Project documentation
```

> Note: Most code changes are generated and applied inside the notebook after cloning YOLOv9 in the runtime environment.

---

## How to Run

### 1) Open the notebook

Run `yolov9 with attention.ipynb` in an environment with GPU support (Kaggle/Colab/local CUDA setup).

### 2) Install dependencies

The notebook installs required packages, including YOLOv9 requirements and PyTorch.

### 3) Prepare dataset

The notebook:

- Downloads VOC 2007/2012 data
- Converts annotations to YOLO format
- Generates class files and data YAML

### 4) Build and train the model

Training is launched via `train_dual.py` using the custom config:

- Config: `models/detect/gelan-c-ca.yaml`
- Data YAML: `voc_ca.yaml`
- Supports multi-GPU (`torchrun`) and single-GPU fallback

---

## Model Notes

- Base architecture: YOLOv9 GELAN-C
- Attention type: Channel Attention (CBAM-inspired)
- Intended dataset in this notebook: Pascal VOC (20 classes)

---

## Important Notes

- This repository currently stores the workflow as a notebook, not as a standalone packaged training codebase.
- Generated files (patched YOLOv9 modules/configs, run outputs) are created during notebook execution.
- Paths in the notebook are tailored for Kaggle-style environments and may need adjustment elsewhere.

---

## Future Improvements

- Export notebook modifications into standalone Python files/scripts.
- Add reproducible training commands for local execution.
- Add evaluation/inference examples and result snapshots.
- Add dependency lock/version pinning for reproducibility.

