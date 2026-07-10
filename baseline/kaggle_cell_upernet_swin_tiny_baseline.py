# ============================================================
# RSASE Baseline - UPerNet Swin-Tiny
# Paste this whole file into one Kaggle notebook cell.
# Warnings are suppressed for clean notebook output.
# ============================================================
import os, sys, subprocess, pathlib, json, random, logging, warnings

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
os.environ["PIP_ROOT_USER_ACTION"] = "ignore"
warnings.filterwarnings("ignore")
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

DATA_ROOT = "/kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap"
TRAIN_IMAGES = f"{DATA_ROOT}/images/train"
TRAIN_MASKS = f"{DATA_ROOT}/label/train"
VAL_IMAGES = f"{DATA_ROOT}/images/val"
VAL_MASKS = f"{DATA_ROOT}/label/val"
OUT_DIR = "/kaggle/working/runs/upernet_swin_tiny"

IMAGE_SIZE = 1024
NUM_CLASSES = 9
BATCH_SIZE = 2
ACCUMULATION_STEPS = 8
EPOCHS = 30
LR = 6e-5
WEIGHT_DECAY = 1e-4
SEED = 42
MODEL_NAME = "openmmlab/upernet-swin-tiny"

print("Checking Kaggle PyTorch...")
import torch
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
try:
    import torchvision
    print("Torchvision:", torchvision.__version__)
except Exception as exc:
    print("Torchvision check skipped:", repr(exc))
def ensure_package(import_name, pip_name=None):
    pip_name = pip_name or import_name
    try:
        __import__(import_name)
        print("OK", import_name)
        return
    except Exception as exc:
        print(f"Installing {pip_name}; import {import_name} failed: {exc!r}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", pip_name])
    except subprocess.CalledProcessError:
        print(f"Normal install failed for {pip_name}; retrying without dependency resolution.")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", pip_name])
    __import__(import_name)
    print("OK", import_name)

print("Checking/installing remaining packages...")
for import_name, pip_name in [
    ("transformers", "transformers"),
    ("timm", "timm"),
    ("albumentations", "albumentations"),
    ("cv2", "opencv-python-headless"),
    ("tqdm", "tqdm"),
    ("safetensors", "safetensors"),
]:
    ensure_package(import_name, pip_name)

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import UperNetForSemanticSegmentation
from transformers.utils import logging as transformers_logging

transformers_logging.set_verbosity_error()

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.benchmark = True

CLASS_NAMES = (
    "Unknown", "Bareland", "Rangeland", "Developed", "Road",
    "Tree", "Water", "Agriculture", "Building"
)

class OpenEarthMapDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=1024, is_train=True):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".tif")])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".tif")])
        if len(self.image_paths) != len(self.mask_paths):
            raise ValueError(f"Image/mask count mismatch: {len(self.image_paths)} vs {len(self.mask_paths)}")
        if not self.image_paths:
            raise ValueError(f"No .tif images found in {image_dir}")
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
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]
        image = (image.astype(np.float32) / 255.0 - self.mean) / self.std
        return torch.from_numpy(image).permute(2, 0, 1).float(), torch.from_numpy(mask).long()

class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, num_classes=9):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.num_classes = num_classes

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()
        probs = F.softmax(logits, dim=1)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        intersection = (probs * one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + one_hot.sum(dim=(2, 3))
        dice_loss = (1.0 - (2.0 * intersection + 1e-5) / (union + 1e-5)).mean()
        return focal_loss + self.dice_weight * dice_loss

class SegMetrics:
    def __init__(self, num_classes=9, exclude_class=0):
        self.num_classes = num_classes
        self.exclude_class = exclude_class
        self.tp = np.zeros(num_classes, dtype=np.float64)
        self.fp = np.zeros(num_classes, dtype=np.float64)
        self.fn = np.zeros(num_classes, dtype=np.float64)
        self.gt = np.zeros(num_classes, dtype=np.float64)
        self.total = 0

    def update(self, preds, targets):
        preds = preds.detach().view(-1).cpu().numpy()
        targets = targets.detach().view(-1).cpu().numpy()
        self.total += targets.size
        for c in range(self.num_classes):
            p = preds == c
            t = targets == c
            self.tp[c] += np.logical_and(p, t).sum()
            self.fp[c] += np.logical_and(p, ~t).sum()
            self.fn[c] += np.logical_and(~p, t).sum()
            self.gt[c] += t.sum()

    def summary(self):
        denom = self.tp + self.fp + self.fn
        iou = np.divide(self.tp, denom, out=np.zeros_like(self.tp), where=denom > 0)
        dice_denom = 2 * self.tp + self.fp + self.fn
        dice = np.divide(2 * self.tp, dice_denom, out=np.zeros_like(self.tp), where=dice_denom > 0)
        valid = self.gt > 0
        valid[self.exclude_class] = False
        return {
            "mIoU": float(iou[valid].mean()) if valid.any() else 0.0,
            "mDice": float(dice[valid].mean()) if valid.any() else 0.0,
            "PixelAcc": float(self.tp.sum() / max(1, self.total)),
            "iou_per_class": iou.tolist(),
            "dice_per_class": dice.tolist(),
            "class_names": list(CLASS_NAMES),
        }

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    metrics = SegMetrics(NUM_CLASSES, exclude_class=0)
    for images, masks in tqdm(loader, desc="Validate"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(pixel_values=images).logits
        logits = F.interpolate(logits.float(), size=masks.shape[-2:], mode="bilinear", align_corners=False)
        metrics.update(logits.argmax(dim=1), masks)
    return metrics.summary()

def make_model():
    id2label = {i: name for i, name in enumerate(CLASS_NAMES)}
    label2id = {name: i for i, name in id2label.items()}
    return UperNetForSemanticSegmentation.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch:", torch.__version__)
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")
for p in [TRAIN_IMAGES, TRAIN_MASKS, VAL_IMAGES, VAL_MASKS]:
    if not pathlib.Path(p).exists():
        raise FileNotFoundError(p)

train_ds = OpenEarthMapDataset(TRAIN_IMAGES, TRAIN_MASKS, IMAGE_SIZE, is_train=True)
val_ds = OpenEarthMapDataset(VAL_IMAGES, VAL_MASKS, IMAGE_SIZE, is_train=False)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)

model = make_model().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=5e-6)
criterion = FocalDiceLoss(num_classes=NUM_CLASSES)
scaler = torch.amp.GradScaler("cuda")

out = pathlib.Path(OUT_DIR)
out.mkdir(parents=True, exist_ok=True)
best_miou = -1.0
training_log = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    epoch_loss = 0.0
    for step, (images, masks) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{EPOCHS}"):
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(pixel_values=images).logits
            logits = F.interpolate(logits, size=masks.shape[-2:], mode="bilinear", align_corners=False)
            loss = criterion(logits, masks) / ACCUMULATION_STEPS
        scaler.scale(loss).backward()
        epoch_loss += loss.item() * ACCUMULATION_STEPS
        if (step + 1) % ACCUMULATION_STEPS == 0 or (step + 1) == len(train_loader):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    scheduler.step()
    summary = evaluate(model, val_loader, device)
    row = {"epoch": epoch + 1, "avg_loss": float(epoch_loss / len(train_loader)), "lr": float(scheduler.get_last_lr()[0]), **summary}
    training_log.append(row)
    print(row)
    if summary["mIoU"] > best_miou:
        best_miou = summary["mIoU"]
        model.save_pretrained(out / "best_model")
        with open(out / "best_metrics.json", "w") as f:
            json.dump(row, f, indent=2)

model.save_pretrained(out / "final_model")
with open(out / "training_log.json", "w") as f:
    json.dump(training_log, f, indent=2)
with open(out / "config.json", "w") as f:
    json.dump({
        "model": MODEL_NAME,
        "data_root": DATA_ROOT,
        "image_size": IMAGE_SIZE,
        "num_classes": NUM_CLASSES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "accumulation_steps": ACCUMULATION_STEPS,
        "learning_rate": LR,
        "weight_decay": WEIGHT_DECAY,
        "loss": "focal_dice",
        "output_dir": OUT_DIR,
        "seed": SEED,
        "unknown_class": "included in training labels; excluded from mean metrics",
    }, f, indent=2)
print("DONE. UPerNet Swin-Tiny outputs:", OUT_DIR)
