import os, sys, subprocess, pathlib, json, random, logging, warnings

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
os.environ["PIP_ROOT_USER_ACTION"] = "ignore"
os.environ["OPENCV_LOG_LEVEL"] = "SILENT"
os.environ["CPL_LOG"] = "/dev/null"
os.environ["CPL_DEBUG"] = "OFF"
warnings.filterwarnings("ignore")
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

MODEL_KIND = 'deeplabv3plus_resnet50'
NOTEBOOK_TITLE = 'RSASE Baseline - DeepLabV3+ ResNet-50'

import csv
import json
import logging
import os
import pathlib
import random
import shutil
import subprocess
import sys
import warnings

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
os.environ["PIP_ROOT_USER_ACTION"] = "ignore"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
warnings.filterwarnings("ignore")
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

DATA_ROOT = pathlib.Path("/kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap")
TRAIN_IMAGES = DATA_ROOT / "images" / "train"
TRAIN_MASKS = DATA_ROOT / "label" / "train"
VAL_IMAGES = DATA_ROOT / "images" / "val"
VAL_MASKS = DATA_ROOT / "label" / "val"

IMAGE_SIZE = 1024
NUM_CLASSES = 9
BATCH_SIZE = 1
ACCUMULATION_STEPS = 16
EPOCHS = 25
LR = 6e-5
WEIGHT_DECAY = 1e-4
SEED = 42
WORKERS = 2

IS_SEGFORMER = MODEL_KIND in {"segformer_b2", "segformer_b3"}
IS_MASK2FORMER = MODEL_KIND == "mask2former_swin_tiny"
IS_SMP = MODEL_KIND in {
    "deeplabv3plus_resnet50",
    "unetplusplus_efficientnet_b3",
    "pspnet_resnet50",
}

if MODEL_KIND == "segformer_b2":
    MODEL_NAME = "nvidia/segformer-b2-finetuned-ade-512-512"
    RUN_NAME = "segformer_b2"
elif MODEL_KIND == "segformer_b3":
    MODEL_NAME = "nvidia/segformer-b3-finetuned-ade-512-512"
    RUN_NAME = "segformer_b3"
elif MODEL_KIND == "mask2former_swin_tiny":
    MODEL_NAME = "facebook/mask2former-swin-tiny-ade-semantic"
    RUN_NAME = "mask2former_swin_tiny"
elif MODEL_KIND == "deeplabv3plus_resnet50":
    MODEL_NAME = "DeepLabV3+ with ImageNet-pretrained ResNet-50 encoder"
    RUN_NAME = "deeplabv3plus_resnet50"
elif MODEL_KIND == "unetplusplus_efficientnet_b3":
    MODEL_NAME = "U-Net++ with ImageNet-pretrained EfficientNet-B3 encoder"
    RUN_NAME = "unetplusplus_efficientnet_b3"
elif MODEL_KIND == "pspnet_resnet50":
    MODEL_NAME = "PSPNet with ImageNet-pretrained ResNet-50 encoder"
    RUN_NAME = "pspnet_resnet50"
else:
    raise ValueError(MODEL_KIND)

OUT_DIR = pathlib.Path("/kaggle/working/runs") / RUN_NAME
ZIP_BASE = pathlib.Path("/kaggle/working") / f"rsase_{RUN_NAME}"

print("Notebook:", NOTEBOOK_TITLE)
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator before running this cell.")
print("GPU:", torch.cuda.get_device_name(0))


def ensure_package(import_name, pip_name=None):
    pip_name = pip_name or import_name
    try:
        __import__(import_name)
        return
    except Exception:
        pass
    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", pip_name
        ])
    except subprocess.CalledProcessError:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--no-deps", pip_name
        ])
    __import__(import_name)


package_specs = [
    ("transformers", "transformers>=4.45,<5"),
    ("timm", "timm"),
    ("albumentations", "albumentations"),
    ("cv2", "opencv-python-headless"),
    ("tqdm", "tqdm"),
    ("scipy", "scipy"),
    ("safetensors", "safetensors"),
]
if IS_SMP:
    package_specs.append(("segmentation_models_pytorch", "segmentation-models-pytorch"))
for import_name, pip_name in package_specs:
    ensure_package(import_name, pip_name)

import albumentations as A
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers.utils import logging as transformers_logging

try:
    if IS_SEGFORMER:
        from transformers import SegformerForSemanticSegmentation
    elif IS_MASK2FORMER:
        from transformers import Mask2FormerForUniversalSegmentation
    else:
        import segmentation_models_pytorch as smp
except ImportError:
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--upgrade", "transformers>=4.45,<5"
    ])
    if IS_SEGFORMER:
        from transformers import SegformerForSemanticSegmentation
    elif IS_MASK2FORMER:
        from transformers import Mask2FormerForUniversalSegmentation
    else:
        import segmentation_models_pytorch as smp

transformers_logging.set_verbosity_error()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

CLASS_NAMES = (
    "Unknown", "Bareland", "Rangeland", "Developed", "Road",
    "Tree", "Water", "Agriculture", "Building",
)


def list_tifs(directory):
    return sorted(
        path for path in pathlib.Path(directory).iterdir()
        if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
    )


class OpenEarthMapDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=1024, is_train=True):
        image_paths = list_tifs(image_dir)
        mask_by_stem = {path.stem: path for path in list_tifs(mask_dir)}
        self.samples = [(path, mask_by_stem[path.stem]) for path in image_paths if path.stem in mask_by_stem]
        if len(self.samples) != len(image_paths):
            missing = [path.name for path in image_paths if path.stem not in mask_by_stem]
            raise ValueError(f"Missing masks for {len(missing)} images; examples: {missing[:5]}")
        if not self.samples:
            raise ValueError(f"No paired TIFF samples found in {image_dir} and {mask_dir}")
        self.image_size = image_size
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=45,
                border_mode=cv2.BORDER_REFLECT_101,
                p=0.3,
            ),
            A.RandomBrightnessContrast(p=0.2),
        ]) if is_train else None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, mask_path = self.samples[index]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if image is None or mask is None:
            raise ValueError(f"Could not read pair: {image_path}, {mask_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed["image"], transformed["mask"]
        if mask.min() < 0 or mask.max() >= NUM_CLASSES:
            raise ValueError(f"Mask labels outside [0, {NUM_CLASSES - 1}] in {mask_path}")
        image = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        return (
            torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float(),
            torch.from_numpy(np.ascontiguousarray(mask)).long(),
        )


class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, num_classes=9):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.num_classes = num_classes

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        focal = (self.alpha * (1.0 - pt).pow(self.gamma) * ce).mean()
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice = 1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)
        return focal + self.dice_weight * dice.mean()


class SegMetrics:
    def __init__(self, num_classes=9, exclude_class=0):
        self.num_classes = num_classes
        self.exclude_class = exclude_class
        self.confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    def update(self, predictions, targets):
        predictions = predictions.detach().view(-1).cpu().numpy().astype(np.int64)
        targets = targets.detach().view(-1).cpu().numpy().astype(np.int64)
        valid = (targets >= 0) & (targets < self.num_classes)
        encoded = self.num_classes * targets[valid] + predictions[valid]
        self.confusion += np.bincount(encoded, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )

    def summary(self):
        matrix = self.confusion.astype(np.float64)
        tp = np.diag(matrix)
        gt = matrix.sum(axis=1)
        predicted = matrix.sum(axis=0)
        fp = predicted - tp
        fn = gt - tp
        iou = np.divide(tp, tp + fp + fn, out=np.zeros_like(tp), where=(tp + fp + fn) > 0)
        dice = np.divide(2 * tp, 2 * tp + fp + fn, out=np.zeros_like(tp), where=(2 * tp + fp + fn) > 0)
        precision = np.divide(tp, tp + fp, out=np.zeros_like(tp), where=(tp + fp) > 0)
        recall = np.divide(tp, tp + fn, out=np.zeros_like(tp), where=(tp + fn) > 0)
        valid_classes = gt > 0
        valid_classes[self.exclude_class] = False
        total = matrix.sum()
        pixel_acc = float(tp.sum() / max(total, 1.0))
        known_total = gt[1:].sum()
        pixel_acc_known = float(tp[1:].sum() / max(known_total, 1.0))
        frequencies = np.divide(gt[valid_classes], max(gt[valid_classes].sum(), 1.0))
        po = pixel_acc
        pe = float((gt * predicted).sum() / max(total * total, 1.0))
        kappa = float((po - pe) / max(1.0 - pe, 1e-12))
        return {
            "mIoU": float(iou[valid_classes].mean()),
            "mDice": float(dice[valid_classes].mean()),
            "mPrecision": float(precision[valid_classes].mean()),
            "mRecall": float(recall[valid_classes].mean()),
            "PixelAcc": pixel_acc,
            "PixelAccExUnknown": pixel_acc_known,
            "mClassAcc": float(recall[valid_classes].mean()),
            "FWIoU": float((frequencies * iou[valid_classes]).sum()),
            "CohenKappa": kappa,
            "iou_per_class": iou.tolist(),
            "dice_per_class": dice.tolist(),
            "precision_per_class": precision.tolist(),
            "recall_per_class": recall.tolist(),
            "class_names": list(CLASS_NAMES),
        }


def make_model():
    id2label = {index: name for index, name in enumerate(CLASS_NAMES)}
    label2id = {name: index for index, name in id2label.items()}
    if IS_SEGFORMER:
        model = SegformerForSemanticSegmentation.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_CLASSES,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    elif IS_MASK2FORMER:
        model = Mask2FormerForUniversalSegmentation.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_CLASSES,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    elif MODEL_KIND == "deeplabv3plus_resnet50":
        model = smp.DeepLabV3Plus(
            encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES
        )
    elif MODEL_KIND == "unetplusplus_efficientnet_b3":
        model = smp.UnetPlusPlus(
            encoder_name="efficientnet-b3", encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES
        )
    elif MODEL_KIND == "pspnet_resnet50":
        model = smp.PSPNet(
            encoder_name="resnet50", encoder_weights="imagenet", in_channels=3, classes=NUM_CLASSES
        )
    else:
        raise ValueError(MODEL_KIND)
    if getattr(model, "supports_gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
    return model
    
def freeze_smp_batchnorm(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.SyncBatchNorm)):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False

def mask2former_targets(masks):
    mask_labels = []
    class_labels = []
    for semantic_mask in masks:
        classes = torch.unique(semantic_mask, sorted=True)
        class_labels.append(classes.long())
        mask_labels.append(torch.stack([(semantic_mask == class_id) for class_id in classes]).float())
    return mask_labels, class_labels


def semantic_scores(outputs, output_size):
    if IS_SEGFORMER:
        return F.interpolate(outputs.logits.float(), size=output_size, mode="bilinear", align_corners=False)
    if IS_MASK2FORMER:
        class_probabilities = outputs.class_queries_logits.float().softmax(dim=-1)[..., :-1]
        mask_probabilities = outputs.masks_queries_logits.float().sigmoid()
        scores = torch.einsum("bqc,bqhw->bchw", class_probabilities, mask_probabilities)
        return F.interpolate(scores, size=output_size, mode="bilinear", align_corners=False)
    return F.interpolate(outputs.float(), size=output_size, mode="bilinear", align_corners=False)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metrics = SegMetrics(NUM_CLASSES, exclude_class=0)
    for images, masks in tqdm(loader, desc="Validate", leave=False):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(pixel_values=images) if not IS_SMP else model(images)
        scores = semantic_scores(outputs, masks.shape[-2:])
        metrics.update(scores.argmax(dim=1), masks)
    return metrics.summary()


def write_json(path, value):
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(value, handle, indent=2)


def write_per_class_csv(path, summary):
    with pathlib.Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["class_id", "class_name", "iou", "dice", "precision", "recall"],
        )
        writer.writeheader()
        for index, name in enumerate(summary["class_names"]):
            writer.writerow({
                "class_id": index,
                "class_name": name,
                "iou": summary["iou_per_class"][index],
                "dice": summary["dice_per_class"][index],
                "precision": summary["precision_per_class"][index],
                "recall": summary["recall_per_class"][index],
            })


for required_path in [TRAIN_IMAGES, TRAIN_MASKS, VAL_IMAGES, VAL_MASKS]:
    if not required_path.exists():
        raise FileNotFoundError(f"Missing OpenEarthMap path: {required_path}")

train_dataset = OpenEarthMapDataset(TRAIN_IMAGES, TRAIN_MASKS, IMAGE_SIZE, is_train=True)
val_dataset = OpenEarthMapDataset(VAL_IMAGES, VAL_MASKS, IMAGE_SIZE, is_train=False)
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True,
    num_workers=WORKERS,
    pin_memory=True,
    persistent_workers=WORKERS > 0,
)
val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=WORKERS,
    pin_memory=True,
    persistent_workers=WORKERS > 0,
)

device = "cuda"
model = make_model().to(device)

if IS_SMP:
    freeze_smp_batchnorm(model)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LR,
    weight_decay=WEIGHT_DECAY,
)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=5e-6)
criterion = FocalDiceLoss(num_classes=NUM_CLASSES) if not IS_MASK2FORMER else None
scaler = torch.amp.GradScaler("cuda", enabled=True)

OUT_DIR.mkdir(parents=True, exist_ok=True)
config = {
    "notebook_title": NOTEBOOK_TITLE,
    "model_kind": MODEL_KIND,
    "model": MODEL_NAME,
    "data_root": str(DATA_ROOT),
    "train_samples": len(train_dataset),
    "validation_samples": len(val_dataset),
    "image_size": IMAGE_SIZE,
    "num_classes": NUM_CLASSES,
    "epochs": EPOCHS,
    "batch_size": BATCH_SIZE,
    "accumulation_steps": ACCUMULATION_STEPS,
    "effective_batch_size": BATCH_SIZE * ACCUMULATION_STEPS,
    "learning_rate": LR,
    "weight_decay": WEIGHT_DECAY,
    "loss": "native_mask2former_set_prediction_loss" if IS_MASK2FORMER else "focal_dice",
    "seed": SEED,
    "unknown_class": "included during training; excluded from mean class metrics",
}
write_json(OUT_DIR / "config.json", config)

best_miou = -1.0
training_log = []
optimizer.zero_grad(set_to_none=True)

for epoch in range(1, EPOCHS + 1):
    model.train()
    if IS_SMP:
        freeze_smp_batchnorm(model)
    running_loss = 0.0
    for step, (images, masks) in tqdm(
        enumerate(train_loader, start=1),
        total=len(train_loader),
        desc=f"Epoch {epoch}/{EPOCHS}",
    ):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            if not IS_MASK2FORMER:
                outputs = model(pixel_values=images) if not IS_SMP else model(images)
                scores = semantic_scores(outputs, masks.shape[-2:])
                loss = criterion(scores, masks)
            else:
                mask_labels, class_labels = mask2former_targets(masks)
                outputs = model(
                    pixel_values=images,
                    mask_labels=mask_labels,
                    class_labels=class_labels,
                )
                loss = outputs.loss
            scaled_loss = loss / ACCUMULATION_STEPS
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch {epoch}, step {step}: {loss.item()}")
        scaler.scale(scaled_loss).backward()
        running_loss += float(loss.detach().item())
        if step % ACCUMULATION_STEPS == 0 or step == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

    scheduler.step()
    summary = evaluate(model, val_loader, device)
    row = {
        "epoch": epoch,
        "avg_loss": running_loss / max(len(train_loader), 1),
        "lr": float(scheduler.get_last_lr()[0]),
        **summary,
    }
    training_log.append(row)
    write_json(OUT_DIR / "training_log.json", training_log)
    print({
        "epoch": epoch,
        "loss": round(row["avg_loss"], 6),
        "mIoU": round(row["mIoU"], 6),
        "mDice": round(row["mDice"], 6),
        "PixelAcc": round(row["PixelAcc"], 6),
    })

    if summary["mIoU"] > best_miou:
        best_miou = summary["mIoU"]
        if IS_SMP:
            best_dir = OUT_DIR / "best_model"
            best_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_dir / "model_state.pth")
        else:
            model.save_pretrained(OUT_DIR / "best_model", safe_serialization=True)
        write_json(OUT_DIR / "best_metrics.json", row)
        write_per_class_csv(OUT_DIR / "per_class_best.csv", summary)

archive_path = shutil.make_archive(str(ZIP_BASE), "zip", root_dir=OUT_DIR)
print("Best validation mIoU:", best_miou)
print("Results folder:", OUT_DIR)
print("ZIP archive:", archive_path)