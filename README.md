# Land Use and Land Cover Segmentation Using SAM 2

This repository contains the training, evaluation, baseline, and ablation scripts for the paper:

**Land Use and Land Cover Segmentation through Efficient Feature Modeling using SAM 2**

The project adapts Meta's SAM 2 image encoder for prompt-free semantic segmentation of high-resolution satellite imagery from the OpenEarthMap benchmark. The final system trains two complementary branches, SAM 2 Tiny and SAM 2 Base+, and combines their class probabilities through a class-aware late-fusion ensemble with four-view test-time augmentation.

## Highlights

- Prompt-free LULC segmentation for OpenEarthMap.
- Dual SAM 2 Tiny and SAM 2 Base+ branches.
- Multi-scale convolutional preprocessing with channel attention.
- LoRA-based adaptation of the SAM 2 image encoder.
- SegFormer-style MLP fusion head for dense pixel prediction.
- Four-view test-time augmentation at inference.
- Class-aware ensemble fusion across land-cover classes.

## Repository Structure

```text
.
|-- sam2_tiny_hybrid_train.py
|-- sam2_baseplus_hybrid_train.py
|-- sam2_tiny_baseplus_ensemble_eval.py
|-- kaggle_cell_class_distribution.py
|-- baseline/
|   |-- kaggle_cell_segformer_b0_baseline.py
|   `-- kaggle_cell_upernet_swin_tiny_baseline.py
`-- ablationstudy/
    |-- kaggle_cell_baseplus_frozen_linear.py
    |-- kaggle_cell_baseplus_lora_linear.py
    |-- kaggle_cell_baseplus_lora_segformer.py
    |-- kaggle_cell_baseplus_full_lora_r16.py
    |-- kaggle_cell_baseplus_full_lora_r32.py
    |-- kaggle_cell_baseplus_full_lora_r64.py
    `-- kaggle_cell_baseplus_msp_no_attention.py
```

## Dataset

The scripts use the OpenEarthMap dataset with the following Kaggle path:

```text
/kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap
```

The project uses 9 label IDs:

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

`Unknown` is included during training and pixel accuracy computation, but excluded from mean class metrics such as mIoU.

## Environment

The main scripts are written as Kaggle-ready Python scripts/notebook cells. They assume GPU execution and hard-coded Kaggle paths.

Core dependencies:

```bash
pip install torch torchvision opencv-python-headless albumentations tqdm peft matplotlib numpy
```

SAM 2 must also be available on the Python path. On Kaggle, the scripts expect it at:

```text
/kaggle/working/sam2
```

Clone SAM 2 and download checkpoints before running the main training scripts:

```bash
cd /kaggle/working
git clone https://github.com/facebookresearch/sam2.git
mkdir -p /kaggle/working/sam2/checkpoints
```

Place the following checkpoints under `/kaggle/working/sam2/checkpoints/`:

```text
sam2_hiera_tiny.pt
sam2_hiera_base_plus.pt
```

## Model Overview

Each SAM 2 branch follows the same hybrid structure:

1. Resize RGB imagery and masks to `1024 x 1024`.
2. Normalize images using ImageNet mean and standard deviation.
3. Extract local and multi-scale texture features using convolution kernels of sizes `3`, `5`, `11`, and `19`.
4. Apply channel attention over the concatenated multi-scale features.
5. Convert the processed stem features back to 3 channels and add them to the normalized image.
6. Pass the image through a LoRA-adapted SAM 2 image encoder.
7. Fuse high-resolution stem features and SAM 2 FPN features with a SegFormer-style MLP head.
8. Produce 9-class semantic segmentation logits.

The LoRA configuration used by the main branches is:

```text
r = 128
lora_alpha = 256
target_modules = ["qkv", "proj", "lin1", "lin2"]
lora_dropout = 0.05
```

## Training

Train the SAM 2 Tiny branch:

```bash
python sam2_tiny_hybrid_train.py
```

Default Tiny settings:

| Setting | Value |
|---|---:|
| Image size | 1024 |
| Batch size | 2 |
| Gradient accumulation | 8 |
| Epochs | 30 |
| Learning rate | 3e-4 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Loss | Focal + Dice |
| Output | `/kaggle/working/checkpoints_hybrid_attention_focal_tiny` |

Train the SAM 2 Base+ branch:

```bash
python sam2_baseplus_hybrid_train.py
```

Default Base+ settings:

| Setting | Value |
|---|---:|
| Image size | 1024 |
| Batch size | 2 |
| Gradient accumulation | 8 |
| Epochs | 30 |
| Learning rate | 3e-4 |
| Optimizer | AdamW |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingLR |
| Loss | Focal + Dice |
| Output | `/kaggle/working/checkpoints_hybrid_attention_focal` |

Each trained branch saves:

```text
hybrid_stem.pth
hybrid_stem_adapter.pth
hybrid_segformer_head.pth
hybrid_sam2_backbone/
```

## Ensemble Evaluation

Run the ensemble evaluator:

```bash
python sam2_tiny_baseplus_ensemble_eval.py
```

Before running, update these constants in `sam2_tiny_baseplus_ensemble_eval.py` if your checkpoints are stored elsewhere:

```python
SAM2_LARGE_CKPT = "..."
SAM2_TINY_CKPT = "..."
WEIGHTS_LARGE = "..."
WEIGHTS_TINY = "..."
```

The evaluator computes:

- mIoU
- mDice/F1
- mean precision
- mean recall
- pixel accuracy
- mean class accuracy
- frequency-weighted IoU
- Cohen's kappa

It also writes visual outputs:

```text
aggregate_metrics.png
per_class_iou_heatmap.png
ensemble_spider_chart.png
qualitative_comparison.png
```

## Class-Aware Fusion Weights

The final ensemble combines Base+ and Tiny probability maps with fixed per-class weights:

| Class | Base+ | Tiny |
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

For each class `c`, the ensemble probability is:

```text
P_ens(c) = w_base(c) * P_base(c) + w_tiny(c) * P_tiny(c)
```

The final segmentation label is selected with `argmax` over the fused probabilities.

## Reported Validation Results

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

`mIoU` is computed excluding the `Unknown` class.

## Baselines

The `baseline/` folder contains Kaggle-ready comparisons using Hugging Face Transformers:

- `kaggle_cell_segformer_b0_baseline.py`: SegFormer B0 baseline.
- `kaggle_cell_upernet_swin_tiny_baseline.py`: UPerNet Swin-Tiny baseline.

Both baselines use the same OpenEarthMap train/validation split, `1024 x 1024` resizing, focal+dice loss, AdamW optimization, and 30 training epochs. Outputs are written under `/kaggle/working/runs/`.

## Ablation Studies

The `ablationstudy/` folder contains Kaggle-cell scripts for testing Base+ design choices:

- frozen encoder with linear head
- LoRA with linear head
- LoRA with SegFormer head
- full LoRA at ranks 16, 32, and 64
- multi-scale preprocessor without attention

These scripts clone SAM 2, download the Base+ checkpoint when needed, write a generated experiment runner under `/kaggle/working/`, and launch the configured ablation.

## Class Distribution Utility

Run:

```bash
python kaggle_cell_class_distribution.py
```

This produces OpenEarthMap class distribution summaries under:

```text
/kaggle/working/results/class_distribution
```

Outputs include:

```text
class_distribution.csv
class_distribution.json
class_distribution.png
```

## Notes

- The repository currently does not include trained weights or the OpenEarthMap dataset.
- The scripts are path-driven rather than CLI-driven; edit constants directly for a new environment.
- The main training scripts require SAM 2 config files from the cloned SAM 2 repository.
- GPU memory requirements are high because images are trained at `1024 x 1024`; the default scripts use mixed precision and gradient accumulation.

## Acknowledgements

This work builds on OpenEarthMap for global high-resolution land-cover mapping and Meta's SAM 2 checkpoints and codebase.
