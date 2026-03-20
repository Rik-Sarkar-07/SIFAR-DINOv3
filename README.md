
# DinoV3 Base and Small Model Finetuning on Video Datasets

Finetuning **DinoV3** (small and base) pretrained checkpoints on video classification datasets (Kinetics-400, Something-Something-v2, etc.) using SIFAR-style input or MSN-style pretraining.

## A. Dataset Preparation

1. Create annotation files: `train.txt` and `val.txt`
2. Format of each line:
   ```
   /raid/abircs/Datasets/Kinetics400/train_256/playing_drums/GJJUUAxgIYo_000007_000017.mp4 1 300 230
   ```
   → `video_path` `label` `start_frame` `end_frame`

3. Pass the **folder containing** these txt files (folder path) to `--data_dir`

4. Update `video_dataset_config.py` with your dataset name and paths

### Note:- For conda environment use requirements.txt or env.yaml files.

## B. Training Details

### General Arguments

| Argument          | Kinetics-400       | SSv2              | Notes                              |
|-------------------|--------------------|-------------------|------------------------------------|
| `--class_numbers` | 400                | 174               | Number of classes                  |
| `--model`         | `dino_small`       | `dino_small`      | or `dino_base`                     |
| `--duration`      | 16                 | 16                | Number of frames sampled           |

### 1. Standard SIFAR-style DinoV3 Finetuning

Use `--dino_model_path` + **do not** use `--msn_pretraining`

**Example – DinoV3 Small (SIFAR)**

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=28529 main.py \
  --data_dir /home/dasabir/orcd/scratch/datasets/Kinetics400_sifar \
  --use_pyav \
  --dataset kinetics400 \
  --opt adamw --lr 5e-4 --epochs 30 --sched cosine \
  --duration 16 --batch-size 4 --super_img_rows 4 \
  --num_workers 16 --disable_scaleup \
  --mixup 0.8 --cutmix 1.0 --drop-path 0.05 \
  --pretrained --warmup-epochs 5 --no-amp \
  --model dino_small \
  --output_dir /home/dasabir/orcd/scratch/sudipta/workspace/Sudipta_MSN_Dino/fine/Dino_MSN_Finetune/output/test_1 \
  --weight-decay 0.01 --clip-grad 1.0 \
  --class_numbers 400 \
  --dino_model_path /home/dasabir/orcd/scratch/sudipta/workspace/Sudipta_MSN_Dino/Dino_MSN_Finetune/dino_repo/dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

**Example – DinoV3 Base (SIFAR)**

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=28529 main.py \
  --data_dir /home/dasabir/orcd/scratch/datasets/Kinetics400_sifar \
  --use_pyav --dataset kinetics400 \
  --opt adamw --lr 5e-4 --epochs 30 --sched cosine \
  --duration 16 --batch-size 4 --super_img_rows 4 \
  --num_workers 16 --disable_scaleup \
  --mixup 0.8 --cutmix 1.0 --drop-path 0.05 \
  --pretrained --warmup-epochs 5 --no-amp \
  --model dino_base \
  --output_dir /home/dasabir/orcd/scratch/sudipta/workspace/Sudipta_MSN_Dino/fine/Dino_MSN_Finetune/output/test_1 \
  --weight-decay 0.01 --clip-grad 1.0 \
  --class_numbers 400 \
  --dino_model_path /home/dasabir/orcd/scratch/sudipta/workspace/Sudipta_MSN_Dino/Dino_MSN_Finetune/dino_repo/dinov3_vitb16_pretrain.pth
```

### 2. MSN-style Finetuning of DinoV3

Use `--msn_model_path` + `--msn_pretraining True`

**Example – MSN-pretrained DinoV3 Small**

```bash
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=28529 main.py \
  --data_dir /home/dasabir/orcd/scratch/datasets/Kinetics400_sifar \
  --use_pyav --dataset kinetics400 \
  --opt adamw --lr 5e-4 --epochs 30 --sched cosine \
  --duration 16 --batch-size 4 --super_img_rows 4 \
  --num_workers 16 --disable_scaleup \
  --mixup 0.8 --cutmix 1.0 --drop-path 0.05 \
  --pretrained --warmup-epochs 5 --no-amp \
  --model dino_small \
  --output_dir /home/dasabir/orcd/scratch/sudipta/workspace/Sudipta_MSN_Dino/fine/Dino_MSN_Finetune/output/test_1 \
  --weight-decay 0.01 --clip-grad 1.0 \
  --class_numbers 400 \
  --msn_pretraining True \
  --msn_model_path /home/dasabir/orcd/scratch/sudipta/workspace/Sudipta_MSN_Dino/Dino_MSN_Pretraining/output/Dino_small_K400_with_lr_1e-6_and_fview_2/checkpoint.pth
```

## C. Pretrained Checkpoints

- **DinoV3 Base** (LVD pretraining): [[link](https://drive.google.com/drive/folders/1e1PLmTIKrMzgnhplKvWIleIFo5iknFI6)]
- **MSN-pretrained DinoV3 checkpoints**: [[link](https://drive.google.com/drive/folders/14y_jOugOtlDFJOaRJaehSfcXvxRG6lvb)] *(continuously updated)*

## Contact

**Author:** Sudipta Sarkar  
**Date:** 16 Feb 2026  
**Email:** sudiptasarkar3600@gmail.com


