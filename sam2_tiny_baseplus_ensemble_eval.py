import os, sys
import logging
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
warnings.filterwarnings("ignore")
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

SAFE_DIR = "/kaggle/working/training_workspace"
if os.path.exists(SAFE_DIR) and os.getcwd() != SAFE_DIR:
    os.chdir(SAFE_DIR)
import cv2, torch, torch.nn as nn, torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from peft import PeftModel
from sam2.build_sam import build_sam2

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SAM2_LARGE_CKPT = "/kaggle/input/datasets/abhijaatoff/sambase/results/checkpoints/sam2_hiera_base_plus.pt"
SAM2_TINY_CKPT  = "/kaggle/input/datasets/abhijaatoff/focaltinyr128/results/checkpoints/sam2_hiera_tiny.pt"
WEIGHTS_LARGE   = "/kaggle/input/datasets/abhijaatoff/sambase/results/checkpoints_hybrid_attention_focal"
WEIGHTS_TINY    = "/kaggle/input/datasets/abhijaatoff/focaltinyr128/results/checkpoints_hybrid_attention_focal"

# ---------------------------------------------------------------------------
# Class-aware weights
# ---------------------------------------------------------------------------
CLASS_WEIGHTS = torch.tensor([
    [0.5,  0.5 ],  # 0 Unknown
    [0.7,  0.3 ],  # 1 Bareland
    [0.5,  0.5 ],  # 2 Rangeland
    [0.6,  0.4 ],  # 3 Developed
    [0.75, 0.25],  # 4 Road
    [0.35, 0.65],  # 5 Tree
    [0.65, 0.35],  # 6 Water
    [0.25, 0.75],  # 7 Agriculture
    [0.6,  0.4 ],  # 8 Building
], dtype=torch.float32)

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class EarthMapValDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=1024):
        self.image_paths = sorted(os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif'))
        self.mask_paths  = sorted(os.path.join(mask_dir,  f) for f in os.listdir(mask_dir)  if f.endswith('.tif'))
        self.image_size  = image_size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std  = np.array([0.229, 0.224, 0.225])

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        raw   = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(raw, cv2.COLOR_BGR2RGB)
        orig  = cv2.resize(image, (self.image_size, self.image_size))
        mask  = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask  = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        norm  = (orig / 255.0 - self.mean) / self.std
        return (torch.tensor(norm, dtype=torch.float32).permute(2, 0, 1),
                torch.tensor(mask, dtype=torch.long), orig)

# ---------------------------------------------------------------------------
# Architecture
# ---------------------------------------------------------------------------
class MultiScalePreprocessor(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        self.conv11 = nn.Conv2d(in_channels, 20, kernel_size=11, padding='same')
        self.conv19 = nn.Conv2d(in_channels, 20, kernel_size=19, padding='same')
        self.conv3  = nn.Conv2d(in_channels, 20, kernel_size=3,  padding='same')
        self.conv5  = nn.Conv2d(in_channels, 20, kernel_size=5,  padding='same')
        self.attention_pool = nn.AdaptiveAvgPool2d(1)
        self.attention_fc   = nn.Sequential(
            nn.Linear(80, 20), nn.ReLU(), nn.Linear(20, 80), nn.Sigmoid()
        )
        self.conv1x1 = nn.Conv2d(80, out_channels, kernel_size=1)
        self.bn      = nn.BatchNorm2d(out_channels)
        self.relu    = nn.ReLU()

    def forward(self, x):
        concat_out = torch.cat([
            self.relu(self.conv11(x)), self.relu(self.conv19(x)),
            self.relu(self.conv3(x)),  self.relu(self.conv5(x))
        ], dim=1)
        b, c, _, _ = concat_out.size()
        attn = self.attention_pool(concat_out).view(b, c)
        attn = self.attention_fc(attn).view(b, c, 1, 1)
        return self.relu(self.bn(self.conv1x1(concat_out * attn)))

class SegFormerMLPHead_Hybrid(nn.Module):
    def __init__(self, in_channels_list=(32, 256, 256, 256), embed_dim=64, num_classes=9):
        super().__init__()
        self.projs = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.ReLU())
            for c in in_channels_list
        )
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels_list), 256, kernel_size=1),
            nn.BatchNorm2d(256), nn.ReLU(), nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, features):
        target_size = next(iter(features)).shape[-2:]
        outs = []
        for proj, f in zip(self.projs, features):
            x = proj(f)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            outs.append(x)
        return self.fusion(torch.cat(outs, dim=1))

class HybridSAM2(nn.Module):
    def __init__(self, sam2_ckpt, model_cfg, weights_dir, num_classes=9, device='cuda'):
        super().__init__()
        self.preprocessor  = MultiScalePreprocessor(in_channels=3, out_channels=32)
        self.stem_to_sam   = nn.Conv2d(32, 3, kernel_size=1)
        base = build_sam2(model_cfg, sam2_ckpt, device=device)
        self.image_encoder = PeftModel.from_pretrained(
            base.image_encoder,
            os.path.join(weights_dir, 'hybrid_sam2_backbone')
        )
        self.segmentation_head = SegFormerMLPHead_Hybrid(
            in_channels_list=(32, 256, 256, 256), embed_dim=64, num_classes=num_classes
        )

    def forward(self, x):
        high_res_skip = self.preprocessor(x)
        backbone_out  = self.image_encoder(self.stem_to_sam(high_res_skip) + x)
        f0, f1, f2    = backbone_out.get('backbone_fpn')
        return self.segmentation_head((high_res_skip, f0, f1, f2))

def load_model(ckpt, cfg, weights_dir, num_classes, device):
    m = HybridSAM2(ckpt, cfg, weights_dir, num_classes, device)
    m.preprocessor.load_state_dict(
        torch.load(os.path.join(weights_dir, 'hybrid_stem.pth'), map_location=device))
    m.stem_to_sam.load_state_dict(
        torch.load(os.path.join(weights_dir, 'hybrid_stem_adapter.pth'), map_location=device))
    m.segmentation_head.load_state_dict(
        torch.load(os.path.join(weights_dir, 'hybrid_segformer_head.pth'), map_location=device))
    m = m.to(device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
        m = nn.DataParallel(m)
    return m.eval()

# ---------------------------------------------------------------------------
# TTA
# ---------------------------------------------------------------------------
@torch.no_grad()
def tta_probs(model, images, device):
    amp = torch.autocast(device_type='cuda' if 'cuda' in device else 'cpu', dtype=torch.float16)
    with amp:
        l0  = model(images)
        lh  = torch.flip(model(torch.flip(images,  [3])),    [3])
        lv  = torch.flip(model(torch.flip(images,  [2])),    [2])
        lhv = torch.flip(model(torch.flip(images, [2, 3])), [2, 3])
    return torch.softmax((l0 + lh + lv + lhv).float() / 4.0, dim=1)

# ---------------------------------------------------------------------------
# Class-aware ensemble
# ---------------------------------------------------------------------------
def class_aware_ensemble(probs_large, probs_tiny, class_weights, device):
    w       = class_weights.to(device)
    w_large = w[:, 0].view(1, -1, 1, 1)
    w_tiny  = w[:, 1].view(1, -1, 1, 1)
    blended = w_large * probs_large + w_tiny * probs_tiny
    return torch.argmax(blended, dim=1), blended

# ---------------------------------------------------------------------------
# Metric accumulator — tracks everything needed for all metrics
# Stores per-class: TP, FP, FN, TN, target_count, total_pixels
# ---------------------------------------------------------------------------
class SegMetrics:
    """
    Accumulates per-class statistics batch-by-batch, then computes:
      IoU / mIoU, Dice / mDice, Precision, Recall, F1 (same as Dice),
      Pixel Accuracy, Mean Class Accuracy, Frequency-Weighted IoU, Cohen's Kappa.
    Class 0 (Unknown) is excluded from all mean metrics.
    """
    def __init__(self, num_classes: int, exclude_class: int = 0):
        self.C  = num_classes
        self.ex = exclude_class
        self.tp    = np.zeros(num_classes, dtype=np.float64)  # TP per class
        self.fp    = np.zeros(num_classes, dtype=np.float64)  # FP per class
        self.fn    = np.zeros(num_classes, dtype=np.float64)  # FN per class
        self.gt    = np.zeros(num_classes, dtype=np.float64)  # ground-truth pixels per class
        self.total = 0                                          # total pixels seen

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """preds / targets: 1-D flattened integer tensors on any device."""
        preds   = preds.view(-1).cpu().numpy()
        targets = targets.view(-1).cpu().numpy()
        self.total += targets.size
        for cls in range(self.C):
            p = (preds   == cls)
            t = (targets == cls)
            tp = int((p & t).sum())
            fp = int((p & ~t).sum())
            fn = int((~p & t).sum())
            self.tp[cls] += tp
            self.fp[cls] += fp
            self.fn[cls] += fn
            self.gt[cls] += int(t.sum())

    # ---- helpers -----------------------------------------------------------
    def _valid_mask(self) -> np.ndarray:
        mask = self.gt > 0
        mask[self.ex] = False
        return mask

    def iou_per_class(self) -> np.ndarray:
        denom = self.tp + self.fp + self.fn
        return np.where(denom > 0, self.tp / denom, 0.0)

    def dice_per_class(self) -> np.ndarray:
        denom = 2 * self.tp + self.fp + self.fn
        return np.where(denom > 0, 2 * self.tp / denom, 0.0)

    def precision_per_class(self) -> np.ndarray:
        denom = self.tp + self.fp
        return np.where(denom > 0, self.tp / denom, 0.0)

    def recall_per_class(self) -> np.ndarray:
        denom = self.tp + self.fn          # == gt
        return np.where(denom > 0, self.tp / denom, 0.0)

    # ---- aggregate metrics -------------------------------------------------
    def miou(self) -> float:
        v = self._valid_mask()
        return float(self.iou_per_class()[v].mean()) if v.any() else 0.0

    def mdice(self) -> float:
        v = self._valid_mask()
        return float(self.dice_per_class()[v].mean()) if v.any() else 0.0

    def mean_precision(self) -> float:
        v = self._valid_mask()
        return float(self.precision_per_class()[v].mean()) if v.any() else 0.0

    def mean_recall(self) -> float:
        v = self._valid_mask()
        return float(self.recall_per_class()[v].mean()) if v.any() else 0.0

    def pixel_accuracy(self) -> float:
        """Overall: correct pixels / total pixels (includes class 0)."""
        return float(self.tp.sum() / self.total) if self.total > 0 else 0.0

    def mean_class_accuracy(self) -> float:
        """Average per-class recall (excluding Unknown)."""
        return self.mean_recall()

    def fw_iou(self) -> float:
        """Frequency-weighted IoU — weights each class IoU by its GT pixel fraction."""
        v    = self._valid_mask()
        freq = self.gt[v] / (self.gt[v].sum() + 1e-9)
        return float((freq * self.iou_per_class()[v]).sum())

    def cohen_kappa(self) -> float:
        """
        Cohen's Kappa = (p_o − p_e) / (1 − p_e)
        p_o = pixel accuracy
        p_e = sum_c (pred_c / N) * (gt_c / N)
        """
        N = self.total
        if N == 0:
            return 0.0
        pred_counts = self.tp + self.fp
        p_o = self.tp.sum() / N
        p_e = float(np.sum((pred_counts / N) * (self.gt / N)))
        return float((p_o - p_e) / (1.0 - p_e + 1e-9))

    def summary(self) -> dict:
        return dict(
            mIoU          = self.miou(),
            mDice         = self.mdice(),
            mPrecision    = self.mean_precision(),
            mRecall       = self.mean_recall(),
            PixelAcc      = self.pixel_accuracy(),
            mClassAcc     = self.mean_class_accuracy(),
            FWIoU         = self.fw_iou(),
            CohenKappa    = self.cohen_kappa(),
            iou_per_class       = self.iou_per_class(),
            dice_per_class      = self.dice_per_class(),
            precision_per_class = self.precision_per_class(),
            recall_per_class    = self.recall_per_class(),
        )

# ---------------------------------------------------------------------------
# Pretty printing helpers
# ---------------------------------------------------------------------------
def print_summary_table(results: dict, class_names: tuple):
    """Prints the overall aggregate metrics side by side for all three models."""
    aggregate_keys = ['mIoU', 'mDice', 'mPrecision', 'mRecall',
                      'PixelAcc', 'mClassAcc', 'FWIoU', 'CohenKappa']
    labels = {
        'mIoU'       : 'mIoU (excl. Unknown)',
        'mDice'      : 'mDice / mF1',
        'mPrecision' : 'Mean Precision',
        'mRecall'    : 'Mean Recall',
        'PixelAcc'   : 'Pixel Accuracy',
        'mClassAcc'  : 'Mean Class Accuracy',
        'FWIoU'      : 'Freq-Weighted IoU',
        'CohenKappa' : "Cohen's Kappa",
    }
    models = ['Large solo', 'Tiny solo', 'Ensemble']
    keys   = ['large', 'tiny', 'ensemble']

    print('\n' + '='*72)
    print('  AGGREGATE METRICS')
    print('='*72)
    print(f"  {'Metric':<24} {'Large solo':>12} {'Tiny solo':>12} {'Ensemble':>12}")
    print(f"  {'-'*60}")
    for k in aggregate_keys:
        row = f"  {labels[k]:<24}"
        for key in keys:
            val = results[key][k]
            # Kappa is already 0-1 but doesn't multiply well—show as raw float
            if k == 'CohenKappa':
                row += f" {val:>12.4f}"
            else:
                row += f" {val*100:>11.2f}%"
        print(row)
    print('='*72)

def print_per_class_table(results: dict, class_names: tuple, metric: str, label: str):
    """Prints a per-class breakdown for IoU, Dice, Precision, or Recall."""
    key_map = {
        'iou'  : 'iou_per_class',
        'dice' : 'dice_per_class',
        'prec' : 'precision_per_class',
        'rec'  : 'recall_per_class',
    }
    arr_key = key_map[metric]
    print(f"\n  PER-CLASS {label.upper()}")
    print(f"  {'Class':<14} {'Large':>9} {'Tiny':>9} {'Ensemble':>11} {'vs best solo':>14}")
    print(f"  {'-'*60}")
    for i, name in enumerate(class_names):
        if i == 0:
            print(f"  {name:<14}  (excluded)")
            continue
        vl = results['large'][arr_key][i]
        vt = results['tiny'][arr_key][i]
        ve = results['ensemble'][arr_key][i]
        best = max(vl, vt)
        gap  = ve - best
        flag = '+' if gap >= 0 else '-'
        print(f"  {name:<14} {vl*100:>8.2f}% {vt*100:>8.2f}% {ve*100:>10.2f}%"
              f"  [{flag}{abs(gap)*100:.2f}%]")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    VAL_IMAGE_DIR = '/kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap/images/val'
    VAL_MASK_DIR  = '/kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap/label/val'
    DEVICE      = 'cuda' if torch.cuda.is_available() else 'cpu'
    NUM_CLASSES = 9
    BATCH_SIZE  = 4

    class_names  = ('Unknown', 'Bareland', 'Rangeland', 'Developed', 'Road',
                    'Tree', 'Water', 'Agriculture', 'Building')
    class_colors = [(0,0,0),(0,1,0),(.6,.4,.2),(.5,.5,.5),(1,0,0),
                    (.5,0,.5),(0,.5,1),(1,.5,.8),(1,.6,0)]
    cmap = ListedColormap(class_colors)

    print('Loading Large model...')
    model_large = load_model(SAM2_LARGE_CKPT, 'sam2_hiera_b+.yaml', WEIGHTS_LARGE, NUM_CLASSES, DEVICE)
    print('Loading Tiny model...')
    model_tiny  = load_model(SAM2_TINY_CKPT,  'sam2_hiera_t.yaml',  WEIGHTS_TINY,  NUM_CLASSES, DEVICE)

    val_dataset = EarthMapValDataset(VAL_IMAGE_DIR, VAL_MASK_DIR, image_size=1024)
    val_loader  = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    # One SegMetrics object per model variant
    metrics_large    = SegMetrics(NUM_CLASSES, exclude_class=0)
    metrics_tiny     = SegMetrics(NUM_CLASSES, exclude_class=0)
    metrics_ensemble = SegMetrics(NUM_CLASSES, exclude_class=0)

    print('Running class-aware ensemble validation (4x TTA per model)...')
    with torch.no_grad():
        for images, masks, _ in tqdm(val_loader, desc='Evaluating'):
            images = images.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE,  non_blocking=True)

            probs_l = tta_probs(model_large, images, DEVICE)
            probs_t = tta_probs(model_tiny,  images, DEVICE)

            ens_preds, _   = class_aware_ensemble(probs_l, probs_t, CLASS_WEIGHTS, DEVICE)
            solo_large     = torch.argmax(probs_l, dim=1)
            solo_tiny      = torch.argmax(probs_t, dim=1)

            flat_masks = masks.view(-1)
            metrics_large.update(solo_large.view(-1),    flat_masks)
            metrics_tiny.update(solo_tiny.view(-1),      flat_masks)
            metrics_ensemble.update(ens_preds.view(-1),  flat_masks)

    # -----------------------------------------------------------------------
    # Collect summaries
    # -----------------------------------------------------------------------
    results = {
        'large'    : metrics_large.summary(),
        'tiny'     : metrics_tiny.summary(),
        'ensemble' : metrics_ensemble.summary(),
    }

    # -----------------------------------------------------------------------
    # Print tables
    # -----------------------------------------------------------------------
    print_summary_table(results, class_names)

    for metric, label in [('iou',  'IoU'),
                           ('dice', 'Dice / F1'),
                           ('prec', 'Precision'),
                           ('rec',  'Recall')]:
        print_per_class_table(results, class_names, metric, label)

    print('\n' + '='*72)

    # -----------------------------------------------------------------------
    # Bar-chart — aggregate metrics comparison
    # -----------------------------------------------------------------------
    agg_keys  = ['mIoU', 'mDice', 'mPrecision', 'mRecall',
                 'PixelAcc', 'mClassAcc', 'FWIoU']
    agg_labels = ['mIoU', 'mDice/F1', 'mPrec', 'mRecall',
                  'PixAcc', 'mClsAcc', 'FWIoU']
    vals_large  = [results['large'][k]    * 100 for k in agg_keys]
    vals_tiny   = [results['tiny'][k]     * 100 for k in agg_keys]
    vals_ens    = [results['ensemble'][k] * 100 for k in agg_keys]
    kappa_vals  = [results['large']['CohenKappa'],
                   results['tiny']['CohenKappa'],
                   results['ensemble']['CohenKappa']]

    x = np.arange(len(agg_keys))
    w = 0.25
    fig, axes = plt.subplots(1, 2, figsize=(20, 6),
                             gridspec_kw={'width_ratios': [4, 1]})

    # Left: main metric bars
    ax = axes[0]
    ax.bar(x - w,   vals_large, w, label='Large solo',  color='#4C72B0', alpha=0.85)
    ax.bar(x,       vals_tiny,  w, label='Tiny solo',   color='#DD8452', alpha=0.85)
    ax.bar(x + w,   vals_ens,   w, label='Ensemble',    color='#55A868', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(agg_labels, fontsize=11)
    ax.set_ylabel('Score (%)', fontsize=12)
    ax.set_ylim(0, 105)
    ax.set_title('Aggregate Metrics — Large vs Tiny vs Class-Aware Ensemble', fontsize=13)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for bar_group, vals in [(x - w, vals_large), (x, vals_tiny), (x + w, vals_ens)]:
        for bx, v in zip(bar_group, vals):
            ax.text(bx, v + 0.5, f'{v:.1f}', ha='center', va='bottom', fontsize=7.5)

    # Right: Kappa
    ax2 = axes[1]
    ax2.bar(['Large', 'Tiny', 'Ensemble'], kappa_vals,
            color=['#4C72B0', '#DD8452', '#55A868'], alpha=0.85)
    ax2.set_title("Cohen's Kappa", fontsize=13)
    ax2.set_ylabel('Kappa', fontsize=12)
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    for bx, v in enumerate(kappa_vals):
        ax2.text(bx, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('aggregate_metrics.png', dpi=150, bbox_inches='tight')
    plt.show()

    # -----------------------------------------------------------------------
    # Per-class IoU heatmap — all three models
    # -----------------------------------------------------------------------
    iou_matrix = np.array([
        results['large']['iou_per_class'][1:],
        results['tiny']['iou_per_class'][1:],
        results['ensemble']['iou_per_class'][1:],
    ]) * 100   # shape: (3, 8)

    fig2, ax3 = plt.subplots(figsize=(14, 3.5))
    im = ax3.imshow(iou_matrix, aspect='auto', cmap='RdYlGn', vmin=0, vmax=100)
    ax3.set_xticks(range(8));  ax3.set_xticklabels(class_names[1:], fontsize=11)
    ax3.set_yticks(range(3));  ax3.set_yticklabels(['Large', 'Tiny', 'Ensemble'], fontsize=11)
    ax3.set_title('Per-Class IoU Heatmap (%)', fontsize=13)
    for r in range(3):
        for c in range(8):
            ax3.text(c, r, f'{iou_matrix[r, c]:.1f}', ha='center', va='center',
                     fontsize=9, color='black')
    plt.colorbar(im, ax=ax3, label='IoU (%)')
    plt.tight_layout()
    plt.savefig('per_class_iou_heatmap.png', dpi=150, bbox_inches='tight')
    plt.show()

    # -----------------------------------------------------------------------
    # Per-class Precision / Recall spider chart for the ensemble
    # -----------------------------------------------------------------------
    categories = list(class_names[1:])   # exclude Unknown
    N = len(categories)
    prec_ens = results['ensemble']['precision_per_class'][1:] * 100
    rec_ens  = results['ensemble']['recall_per_class'][1:]    * 100
    iou_ens  = results['ensemble']['iou_per_class'][1:]       * 100

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]   # close polygon

    fig3, ax4 = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for vals, color, label in [
        (prec_ens, '#4C72B0', 'Precision'),
        (rec_ens,  '#DD8452', 'Recall'),
        (iou_ens,  '#55A868', 'IoU'),
    ]:
        v = vals.tolist() + vals[:1].tolist()
        ax4.plot(angles, v, '-o', color=color, linewidth=2, label=label)
        ax4.fill(angles, v, alpha=0.08, color=color)

    ax4.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=11)
    ax4.set_ylim(0, 100)
    ax4.set_title('Ensemble — Per-class Precision / Recall / IoU', fontsize=13, pad=20)
    ax4.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=11)
    plt.tight_layout()
    plt.savefig('ensemble_spider_chart.png', dpi=150, bbox_inches='tight')
    plt.show()

    # -----------------------------------------------------------------------
    # Qualitative visualisation (unchanged)
    # -----------------------------------------------------------------------
    vis_loader = DataLoader(val_dataset, batch_size=2, shuffle=True, num_workers=2)
    images, masks, orig_images = next(iter(vis_loader))
    images = images.to(DEVICE)
    with torch.no_grad():
        pl = tta_probs(model_large, images, DEVICE)
        pt = tta_probs(model_tiny,  images, DEVICE)
        preds, _ = class_aware_ensemble(pl, pt, CLASS_WEIGHTS, DEVICE)
        preds = preds.cpu().numpy()

    masks       = masks.numpy()
    orig_images = orig_images.numpy()

    fig4, axes = plt.subplots(2, 3, figsize=(18, 12))
    for orig, mask, pred, row in zip(orig_images, masks, preds, axes):
        row[0].imshow(orig);                             row[0].set_title('Original Image'); row[0].axis('off')
        row[1].imshow(mask, cmap=cmap, vmin=0, vmax=8); row[1].set_title('Ground Truth');   row[1].axis('off')
        row[2].imshow(pred, cmap=cmap, vmin=0, vmax=8); row[2].set_title('Ensemble Pred');  row[2].axis('off')
    patches = [mpatches.Patch(color=class_colors[c], label=class_names[c]) for c in range(NUM_CLASSES)]
    fig4.legend(handles=patches, loc='lower center', ncol=9, fontsize=12)
    plt.tight_layout(rect=(0, 0.05, 1, 1))
    plt.savefig('qualitative_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
