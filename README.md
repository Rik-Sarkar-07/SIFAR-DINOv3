# DINOv3 Base and Small Model Finetuning on Video Datasets

This repository focuses on finetuning **DINOv3 (Small and Base variants)** on video classification datasets such as **Kinetics-400** and **Something-Something-v2 (SSv2)** using **SIFAR-style input representation** and optional **MSN-style pretraining**.

---

##  Overview

**SIFAR (Super Image for Action Recognition)** adapts standard **image classification models** for video understanding by transforming video frames into a **single Super Image**.

Instead of sequential frame processing, frames are spatially rearranged into a grid, enabling the use of standard image backbones for video tasks.

In this project, traditional CNN/ViT backbones are replaced with **DINOv3**, a self-supervised Vision Transformer backbone developed by Meta AI.

---

##  Architecture

**Pipeline Overview:**

```
Video Frames
     │
     ▼
Frame Sampling
     │
     ▼
Super Image Construction (SIFAR)
     │
     ▼
DINOv3 Backbone
     │
     ▼
Classification Head
     │
     ▼
Action Prediction
```

---

##  Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/Rik-Sarkar-07/SIFAR-DINOv3.git
cd SIFAR-DINOv3
```

### 2. Environment Setup

(Works for both pretraining and finetuning)

```bash
conda env create --file env.yaml
conda activate sifar_msn
```

> **Note:** You may also use `requirements.txt` as an alternative.

---

##  A. Dataset Preparation

### A1. Dataset Download

* **Annotations (Kinetics-400 & SSv2):**
  [https://drive.google.com/drive/folders/1iVenj2jUaqbZfK9mHnyLfogkLF8nXq-O](https://drive.google.com/drive/folders/1iVenj2jUaqbZfK9mHnyLfogkLF8nXq-O)

* **Video Datasets:**
  [https://drive.google.com/drive/folders/1ObLuUCHZ2xDa2TCOpo4ew7Ne8vG_SuBD](https://drive.google.com/drive/folders/1ObLuUCHZ2xDa2TCOpo4ew7Ne8vG_SuBD)

#### Download using gdown:

```bash
pip install gdown
gdown --fuzzy "<FULL_GOOGLE_DRIVE_URL>"
```

---

### A2. Annotation Setup

1. Create two files:

   * `train.txt`
   * `val.txt`

2. Each line should follow the format:

```
/path/to/video.mp4 label start_frame end_frame
```

**Example:**

```
/raid/abircs/Datasets/Kinetics400/train_256/playing_drums/GJJUUAxgIYo_000007_000017.mp4 1 300 230
```

3. Provide the **directory containing these files** via:

```
--data_dir <path_to_annotation_folder>
```

4. Update dataset configurations in:

```
video_dataset_config.py
```

---

##  B. Training Details

### General Arguments

| Argument          | Kinetics-400 | SSv2       | Description              |
| ----------------- | ------------ | ---------- | ------------------------ |
| `--class_numbers` | 400          | 174        | Number of classes        |
| `--model`         | dino_small   | dino_small | or `dino_base`           |
| `--duration`      | 16           | 16         | Number of sampled frames |

---

### 1. Standard SIFAR-style DINOv3 Finetuning

* Use: `--dino_model_path`
* Do **NOT** use: `--msn_pretraining`

#### Example: DINOv3 Small

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=28529 main.py \
  --data_dir /path/to/Kinetics400_sifar \
  --use_pyav --dataset kinetics400 \
  --opt adamw --lr 5e-4 --epochs 30 --sched cosine \
  --duration 16 --batch-size 4 --super_img_rows 4 \
  --num_workers 16 --disable_scaleup \
  --mixup 0.8 --cutmix 1.0 --drop-path 0.05 \
  --pretrained --warmup-epochs 5 --no-amp \
  --model dino_small \
  --output_dir /path/to/output \
  --weight-decay 0.01 --clip-grad 1.0 \
  --class_numbers 400 \
  --dino_model_path /path/to/dinov3_vits16_pretrain.pth
```

#### Example: DINOv3 Base

```bash
--model dino_base \
--dino_model_path /path/to/dinov3_vitb16_pretrain.pth
```

---

### 2. MSN-style Finetuning

* Use: `--msn_model_path`
* Enable: `--msn_pretraining True`

#### Example: MSN-pretrained DINOv3 Small

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py \
  --data_dir /path/to/Kinetics400_sifar \
  --use_pyav --dataset kinetics400 \
  --opt adamw --lr 5e-4 --epochs 30 --sched cosine \
  --duration 16 --batch-size 4 --super_img_rows 4 \
  --num_workers 16 --disable_scaleup \
  --mixup 0.8 --cutmix 1.0 --drop-path 0.05 \
  --pretrained --warmup-epochs 5 --no-amp \
  --model dino_small \
  --output_dir /path/to/output \
  --weight-decay 0.01 --clip-grad 1.0 \
  --class_numbers 400 \
  --msn_pretraining True \
  --msn_model_path /path/to/msn_checkpoint.pth
```

---

##  C. Pretrained Checkpoints

* **DINOv3 Base (LVD Pretraining):**
  [https://drive.google.com/drive/folders/1e1PLmTIKrMzgnhplKvWIleIFo5iknFI6](https://drive.google.com/drive/folders/1e1PLmTIKrMzgnhplKvWIleIFo5iknFI6)

* **MSN-pretrained DINOv3:**
  [https://drive.google.com/drive/folders/14y_jOugOtlDFJOaRJaehSfcXvxRG6lvb](https://drive.google.com/drive/folders/14y_jOugOtlDFJOaRJaehSfcXvxRG6lvb)
  *(Continuously updated)*

---

##  D. Acknowledgements

This work builds upon:

* **SIFAR: Super Image for Action Recognition**
* **DINOv3: Self-supervised Vision Transformer (LVD pretraining)**

---

##  E. Contact

## Contact

**Author:** Sudipta Sarkar  

**Date:** 20 March 2026  

**Email:** [sudiptasarkar3600@gmail.com](mailto:sudiptasarkar3600@gmail.com)  

**Website:** [sudipta-rkmrc.github.io](https://sudipta-rkmrc.github.io/website/)
---
