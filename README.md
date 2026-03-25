# FMDP

Official implementation of **"FMDP: Leveraging a Foundation Model for Dual-Pixel Disparity Estimation"** (MVA 2025)

Zhuofeng Wu, Doehyung Lee, Zihua Liu, Kazunori Yoshizaki, Yusuke Monno, Masatoshi Okutomi

## Environment Setup

```bash
conda env create -f env.yml
conda activate mono-qpd
```

## Pretrained Checkpoints

### Depth Anything V2

Download the ViT-Large checkpoint from [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) and place it at:

```
mono_qpd/Depth_Anything_V2/checkpoints/depth_anything_v2_vitl.pth
```

### FMDP

Place the pretrained FMDP checkpoint at:

```
result/train/Exp0825FMDP/checkpoints/<checkpoint_file>.pth
```

## Dataset

Download the QP dataset following the instructions in [QPDNet](https://github.com/Zhuofeng-Wu/QPDNet).

Place or symlink the dataset:

```bash
ln -s /path/to/QP-Data datasets/QP-Data
```

## Training

```bash
python train_mono_qpd.py --exp_name Exp0825FMDP
```

To resume from a checkpoint:

```bash
python train_mono_qpd.py --exp_name Exp0825FMDP --restore_ckpt result/train/Exp0825FMDP/checkpoints/latest.pth
```

## Evaluation

```bash
python evaluate_mono_qpd.py --exp_name Exp0825FMDP --ckpt_epoch latest --eval_datasets QPD-Test --save_result
```

---

## Appendix: Internal Handover Only

> **This section is for internal handover purposes only. Remove before public release.**

### QPD + QPDv2 Dataset

Place or symlink the combined dataset:

```bash
ln -s /path/to/QP-Data-v2 datasets/QP-Data-v2
```

The `QP-Data-v2` directory should contain both QPD and QPDv2 data.

### QPD + QPDv2 Pretrained Checkpoint

Place the QPD+QPDv2 mixed-trained checkpoint at:

```
result/train/Exp0121MixedDataset/checkpoints/<checkpoint_file>.pth
```

### Training with QPD + QPDv2

```bash
python train_mono_qpd.py --exp_name Exp0121MixedDataset
```

The training steps are set to 400K to account for the doubled dataset size (see `run_settings.py`).

To resume:

```bash
python train_mono_qpd.py --exp_name Exp0121MixedDataset --restore_ckpt result/train/Exp0121MixedDataset/checkpoints/latest.pth
```

### Evaluation with QPD + QPDv2 Model

```bash
python evaluate_mono_qpd.py --exp_name Exp0121MixedDataset --ckpt_epoch latest --eval_datasets QPD-Test --save_result
```

### Running on TSUBAME Supercomputer

Jobs on TSUBAME are managed via the `runsync` system.

#### Training

```bash
python runsync/runsync_script.py --type train --run_setting_name Exp0825FMDP --run_script True
```

#### Evaluation

```bash
python runsync/runsync_script.py \
  --type eval \
  --run_setting_name Exp0825FMDP \
  --run_script True \
  --eval_run_time 01:30:00 \
  --eval_datasets QPD-Test DPD_Disp
```

#### Evaluation Time Reference (TSUBAME)

| Dataset | Approx. Time | Recommended Time Limit |
|---------|-------------|----------------------|
| QPD-Test | ~5 min | 10 min |
| DPD-Disp | ~3 min | 10 min |
| QPD-Test + DPD-Disp | ~8 min | 15 min |

#### Experiment Configurations

Experiment configs are defined as dataclasses in `runsync/presets/run_settings.py`.

- `Exp*`: For TSUBAME (large batch size)
- `LocalExp*`: For local testing (small batch size, debug mode)
