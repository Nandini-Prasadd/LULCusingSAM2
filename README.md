# LULC Segmentation Using SAM 2

This repository contains the code used for **Land Use and Land Cover Segmentation through Efficient Feature Modeling using SAM 2**. The project adapts SAM 2 for prompt-free semantic segmentation of high-resolution remote-sensing imagery from OpenEarthMap.

The main approach trains two complementary SAM 2 branches:

- **SAM 2 Tiny**, which is lighter and useful for texture-heavy classes.
- **SAM 2 Base+**, which is stronger for larger structures and global context.

At inference time, the two branches are combined with class-aware late fusion and optional four-view test-time augmentation (TTA).

## Paper

The project paper is included as:

```text
paper.pdf
```

## What This Project Does

- Fine-tunes SAM 2 Tiny and SAM 2 Base+ for semantic LULC segmentation.
- Uses a multi-scale convolutional preprocessor before the SAM 2 encoder.
- Uses channel attention to reweight multi-scale features.
- Uses LoRA adapters for parameter-efficient SAM 2 encoder adaptation.
- Uses a SegFormer-style fusion head to produce dense class logits.
- Evaluates single models, TTA variants, class-aware fusion, and efficiency metrics.
- Includes baseline and ablation scripts for comparison.

## Repository Layout

```text
.
|-- README.md
|-- paper.pdf
|-- kaggle_cell_class_distribution.py
|-- sam2_tiny_hybrid_train.py
|-- sam2_baseplus_hybrid_train.py
|-- sam2_tiny_baseplus_ensemble_eval.py
|-- baseline/
|   |-- kaggle_cell_deeplab_v3_plus_resnet-50.py
|   |-- kaggle_cell_deeplabv3plus_mobilenetv2.py
|   |-- kaggle_cell_linknet_resnet34.py
|   |-- kaggle_cell_manet_resnet50.py
|   |-- kaggle_cell_mask2former_swin_tiny.py
|   |-- kaggle_cell_segformer_b0_baseline.py
|   |-- kaggle_cell_segformer_b2_baseline.py
|   |-- kaggle_cell_unetplusplus_efficientnet_b3.py
|   |-- kaggle_cell_upernet_swin_small.py
|   `-- kaggle_cell_upernet_swin_tiny_baseline.py
|-- ablationstudy/
|   |-- kaggle_cell_baseplus_frozen_linear.py
|   |-- kaggle_cell_baseplus_lora_linear.py
|   |-- kaggle_cell_baseplus_lora_segformer.py
|   |-- kaggle_cell_baseplus_full_lora_r16.py
|   |-- kaggle_cell_baseplus_full_lora_r32.py
|   |-- kaggle_cell_baseplus_full_lora_r64.py
|   |-- kaggle_cell_baseplus_msp_no_attention.py
|   `-- kaggle_cell_base_plus_msp_no_attention_seed42_30ep_final.py
`-- evaluation/
    |-- kaggle_cell_fusion_comparison_without_tta.py
    |-- kaggle_cell_fusion_comparison_with_four-view_tta.py
    |-- kaggle_cell_held-out_fusion_bareland_and_efficiency.py
    |-- kaggle_cell_parameters_flops_memory_and_latency.py
    |-- kaggle_cell_qualitative_comparison_and_error_maps.py
    `-- kaggle_cell_sam2_base_plus_single_view_and_tta.py
```

Some files in `baseline/`, `ablationstudy/`, and `evaluation/` are currently zero-byte placeholders. The populated scripts are the runnable ones.

## Dataset

The scripts are written for the OpenEarthMap dataset on Kaggle and expect this path:

```text
/kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap
```

Expected structure:

```text
global-land-cover-mapping-openearthmap/
|-- images/
|   |-- train/
|   `-- val/
`-- label/
    |-- train/
    `-- val/
```

The segmentation labels are:

| ID | Class |
|---:|---|
| 0 | Unknown |
| 1 | Bareland |
| 2 | Rangeland |
| 3 | Developed |
| 4 | Road |
| 5 | Tree |
| 6 | Water |
| 7 | Agriculture |
| 8 | Building |

`Unknown` is used in masks and pixel accuracy, but is excluded from mean class metrics such as mIoU.

## Environment

The code is Kaggle-oriented. Most scripts are intended to be copied into or run as Kaggle notebook cells, with GPU enabled.

Typical dependencies:

```bash
pip install torch torchvision opencv-python-headless albumentations tqdm peft matplotlib numpy
```

Some baseline and evaluation scripts install additional dependencies inside the script, including packages such as `transformers`, `timm`, `segmentation-models-pytorch`, `safetensors`, `scipy`, and Hydra/OmegaConf for SAM 2 configs.

SAM 2 is expected under:

```text
/kaggle/working/sam2
```

Basic setup:

```bash
cd /kaggle/working
git clone https://github.com/facebookresearch/sam2.git
mkdir -p /kaggle/working/sam2/checkpoints
```

Download or attach these SAM 2 checkpoints:

```text
sam2_hiera_tiny.pt
sam2_hiera_base_plus.pt
```

The training scripts look for checkpoints in:

```text
/kaggle/working/sam2/checkpoints/
/kaggle/working/checkpoints/
```

The evaluation scripts generally search under `/kaggle/input` for attached model checkpoints and trained weights.

## Model Architecture

Each branch follows the same high-level design:

1. Read RGB `.tif` images and grayscale `.tif` masks.
2. Resize inputs to `1024 x 1024`.
3. Normalize images with ImageNet mean and standard deviation.
4. Extract multi-scale local features with convolution kernels of size `3`, `5`, `11`, and `19`.
5. Apply channel attention over the concatenated features.
6. Map the preprocessed features back to 3 channels and add them to the image.
7. Feed the result into a LoRA-adapted SAM 2 image encoder.
8. Fuse SAM 2 FPN features with the high-resolution stem using a SegFormer-style MLP head.
9. Predict 9-class semantic segmentation logits.

Main LoRA settings:

```text
r = 128
lora_alpha = 256
target_modules = ["qkv", "proj", "lin1", "lin2"]
lora_dropout = 0.05
```

## Training

Train SAM 2 Tiny:

```bash
python sam2_tiny_hybrid_train.py
```

Train SAM 2 Base+:

```bash
python sam2_baseplus_hybrid_train.py
```

Default settings for both main training scripts:

| Setting | Value |
|---|---:|
| Image size | 1024 |
| Classes | 9 |
| Batch size | 2 |
| Gradient accumulation | 8 |
| Epochs | 30 |
| Learning rate | 3e-4 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Loss | Focal + Dice |

Saved Tiny weights:

```text
/kaggle/working/checkpoints_hybrid_attention_focal_tiny
```

Saved Base+ weights:

```text
/kaggle/working/checkpoints_hybrid_attention_focal
```

Each branch saves:

```text
hybrid_stem.pth
hybrid_stem_adapter.pth
hybrid_segformer_head.pth
hybrid_sam2_backbone/
```

## Ensemble Evaluation

Run the main ensemble evaluator:

```bash
python sam2_tiny_baseplus_ensemble_eval.py
```

Before running outside the original Kaggle setup, update these constants:

```python
SAM2_LARGE_CKPT = "path/to/sam2_hiera_base_plus.pt"
SAM2_TINY_CKPT = "path/to/sam2_hiera_tiny.pt"
WEIGHTS_LARGE = "path/to/base_plus_weights"
WEIGHTS_TINY = "path/to/tiny_weights"
```

The evaluator reports:

- mIoU
- mDice/F1
- mean precision
- mean recall
- pixel accuracy
- mean class accuracy
- frequency-weighted IoU
- Cohen's kappa
- per-class IoU, Dice, precision, and recall

It writes these plots:

```text
aggregate_metrics.png
per_class_iou_heatmap.png
ensemble_spider_chart.png
qualitative_comparison.png
```

## Class-Aware Fusion

The ensemble combines Base+ and Tiny softmax probabilities with fixed class-specific weights.

| Class | Base+ weight | Tiny weight |
|---|---:|---:|
| Unknown | 0.50 | 0.50 |
| Bareland | 0.70 | 0.30 |
| Rangeland | 0.50 | 0.50 |
| Developed | 0.60 | 0.40 |
| Road | 0.75 | 0.25 |
| Tree | 0.35 | 0.65 |
| Water | 0.65 | 0.35 |
| Agriculture | 0.25 | 0.75 |
| Building | 0.60 | 0.40 |

For each class `c`:

```text
P_ensemble(c) = w_base(c) * P_base(c) + w_tiny(c) * P_tiny(c)
```

The final label is the `argmax` over fused probabilities.

## Evaluation Scripts

The `evaluation/` folder contains extra Kaggle cells for reviewer-style analysis:

| Script | Purpose |
|---|---|
| `kaggle_cell_sam2_base_plus_single_view_and_tta.py` | Compares SAM 2 Base+ single-view inference against four-view TTA. |
| `kaggle_cell_fusion_comparison_without_tta.py` | Evaluates Base+/Tiny fusion without TTA. |
| `kaggle_cell_fusion_comparison_with_four-view_tta.py` | Evaluates Base+/Tiny fusion with four-view TTA. |
| `kaggle_cell_held-out_fusion_bareland_and_efficiency.py` | Runs held-out fusion and Bareland-focused/efficiency analysis. |
| `kaggle_cell_parameters_flops_memory_and_latency.py` | Reports parameter count, FLOPs, memory, and latency. |
| `kaggle_cell_qualitative_comparison_and_error_maps.py` | Placeholder for qualitative and error-map visualizations. |

Most populated evaluation scripts normalize attached Kaggle model folders under:

```text
/kaggle/working/normalized_models
```

and write result archives under:

```text
/kaggle/working/results
```

## Baselines

The `baseline/` folder contains comparison experiments. Populated baseline scripts include:

| Script | Model |
|---|---|
| `kaggle_cell_segformer_b0_baseline.py` | SegFormer B0 |
| `kaggle_cell_segformer_b2_baseline.py` | SegFormer B2 |
| `kaggle_cell_upernet_swin_tiny_baseline.py` | UPerNet Swin-Tiny |
| `kaggle_cell_deeplab_v3_plus_resnet-50.py` | DeepLabV3+ ResNet-50 |
| `kaggle_cell_unetplusplus_efficientnet_b3.py` | U-Net++ EfficientNet-B3 |

Zero-byte placeholder baseline files are present for:

```text
kaggle_cell_deeplabv3plus_mobilenetv2.py
kaggle_cell_linknet_resnet34.py
kaggle_cell_manet_resnet50.py
kaggle_cell_mask2former_swin_tiny.py
kaggle_cell_upernet_swin_small.py
```

## Ablation Studies

The `ablationstudy/` folder contains SAM 2 Base+ ablations for:

- frozen SAM 2 encoder with a linear head
- LoRA with a linear head
- LoRA with a SegFormer-style head
- full LoRA at ranks 16, 32, and 64
- multi-scale preprocessing without attention

The file `kaggle_cell_base_plus_msp_no_attention_seed42_30ep_final.py` is currently an empty placeholder.

## Class Distribution

Run:

```bash
python kaggle_cell_class_distribution.py
```

This counts OpenEarthMap label pixels and writes:

```text
/kaggle/working/results/class_distribution/class_distribution.csv
/kaggle/working/results/class_distribution/class_distribution.json
/kaggle/working/results/class_distribution/class_distribution.png
```

## Reported Results

Reported OpenEarthMap validation metrics from the paper:

| Metric | Base+ | Tiny | Ensemble |
|---|---:|---:|---:|
| mIoU | 67.47% | 66.24% | 68.69% |
| FW IoU | 66.70% | 66.40% | 68.15% |
| Mean Precision | 80.05% | 79.97% | 81.12% |
| Mean Recall | 80.46% | 78.49% | 80.96% |
| Pixel Accuracy | 79.94% | 79.65% | 80.94% |
| Mean Class Accuracy | 80.46% | 78.49% | 80.96% |
| mDice | 80.12% | 79.13% | 80.98% |
| Cohen's Kappa | 0.7596 | 0.7553 | 0.7711 |

`mIoU` excludes the `Unknown` class.

Reported ablation summary:

| Method | mIoU | mDice | Pixel Accuracy |
|---|---:|---:|---:|
| Tiny | 61.18 | 75.08 | 76.62 |
| Base+ | 63.83 | 77.12 | 78.53 |
| Ensemble | 64.05 | 77.32 | 78.52 |
| Tiny + MSP | 66.24 | 79.13 | 79.65 |
| Large + MSP | 67.47 | 80.12 | 79.94 |
| Ensemble + MSP | 68.69 | 80.98 | 80.94 |

## Notes

- The dataset and trained model weights are not included in this repository.
- The scripts are mostly path-driven, not CLI-driven. Update hard-coded Kaggle paths before using a different environment.
- Use a Kaggle GPU accelerator for the training and evaluation scripts.
- The main scripts expect SAM 2 config files from the cloned `facebookresearch/sam2` repository.
- Several scripts are designed as complete Kaggle cells and may install missing packages at runtime.

## Acknowledgements

This work uses the OpenEarthMap dataset and Meta's SAM 2 codebase and pretrained checkpoints.
