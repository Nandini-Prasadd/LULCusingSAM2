import sys
import os
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

# ===========================================================================
# 1. THE ULTIMATE KAGGLE SHADOWING FIX
# ===========================================================================
repo_dir = "/kaggle/working/sam2"
if os.path.exists(repo_dir):
    os.chdir(repo_dir)

if repo_dir not in sys.path:
    sys.path.insert(0, repo_dir)

sys.path = [p for p in sys.path if p not in ["/kaggle/working", ""]]
# ===========================================================================

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import albumentations as A
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

from sam2.build_sam import build_sam2

# ---------------------------------------------------------------------------
# 2. Dataset Loader 
# ---------------------------------------------------------------------------
class PromptlessEarthMapDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=1024, is_train=True):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.tif')])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('.tif')])
        self.image_size = image_size
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        
        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5), 
            A.VerticalFlip(p=0.5), 
            A.RandomRotate90(p=0.5), 
            A.Transpose(p=0.5),      
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.3), 
            A.RandomBrightnessContrast(p=0.2),
        ]) if is_train else None

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        raw_img = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.image_size, self.image_size))
        
        mask = cv2.imread(self.mask_paths[idx], cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image, mask = augmented['image'], augmented['mask']
        
        image = (image / 255.0 - self.mean) / self.std
        return torch.tensor(image, dtype=torch.float32).permute(2, 0, 1), torch.tensor(mask, dtype=torch.long)

# ---------------------------------------------------------------------------
# 3. SOTA Architecture: Hybrid UNet + SAM 2 + Channel Attention
# ---------------------------------------------------------------------------
class MultiScalePreprocessor(nn.Module):
    def __init__(self, in_channels=3, out_channels=32):
        super().__init__()
        
        self.conv11 = nn.Conv2d(in_channels, 20, kernel_size=11, padding='same')
        self.conv19 = nn.Conv2d(in_channels, 20, kernel_size=19, padding='same')
        self.conv3 = nn.Conv2d(in_channels, 20, kernel_size=3, padding='same')
        self.conv5 = nn.Conv2d(in_channels, 20, kernel_size=5, padding='same')
        
        self.attention_pool = nn.AdaptiveAvgPool2d(1)
        self.attention_fc = nn.Sequential(
            nn.Linear(80, 20),
            nn.ReLU(),
            nn.Linear(20, 80),
            nn.Sigmoid()
        )
        
        self.conv1x1 = nn.Conv2d(80, out_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x11 = self.relu(self.conv11(x))
        x19 = self.relu(self.conv19(x))
        x3 = self.relu(self.conv3(x))
        x5 = self.relu(self.conv5(x))
        
        concat_out = torch.cat([x11, x19, x3, x5], dim=1)
        
        b, c, _, _ = concat_out.size()
        attn = self.attention_pool(concat_out).view(b, c)
        attn = self.attention_fc(attn).view(b, c, 1, 1)
        
        attended_features = concat_out * attn
        return self.relu(self.bn(self.conv1x1(attended_features)))

class SegFormerMLPHead_Hybrid(nn.Module):
    def __init__(self, in_channels_list=[32, 256, 256, 256], embed_dim=64, num_classes=9):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(c, embed_dim, kernel_size=1), nn.BatchNorm2d(embed_dim), nn.ReLU())
            for c in in_channels_list
        ])
        
        self.fusion = nn.Sequential(
            nn.Conv2d(embed_dim * len(in_channels_list), 256, kernel_size=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.1),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, features):
        target_size = features[0].shape[-2:] 
        outs = []
        for i, f in enumerate(features):
            x = self.projs[i](f)
            if x.shape[-2:] != target_size:
                x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
            outs.append(x)
            
        out = torch.cat(outs, dim=1)
        return self.fusion(out)

class UNet_SAM2_Hybrid(nn.Module):
    def __init__(self, sam2_checkpoint, model_cfg, num_classes=9, device="cuda"):
        super().__init__()
        
        self.preprocessor = MultiScalePreprocessor(in_channels=3, out_channels=32)
        self.stem_to_sam = nn.Conv2d(32, 3, kernel_size=1) 
        
        base_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
        self.image_encoder = base_model.image_encoder
        
        lora_config = LoraConfig(
            r=128, 
            lora_alpha=256, 
            target_modules=["qkv", "proj", "lin1", "lin2"], 
            lora_dropout=0.05, 
            bias="none"
        )
        self.image_encoder = get_peft_model(self.image_encoder, lora_config)
        self.segmentation_head = SegFormerMLPHead_Hybrid(in_channels_list=[32, 256, 256, 256], embed_dim=64, num_classes=num_classes)

    def forward(self, images):
        high_res_skip = self.preprocessor(images)
        sam_input = self.stem_to_sam(high_res_skip) + images 
        backbone_out = self.image_encoder(sam_input)
        fpn_features = backbone_out["backbone_fpn"]
        
        logits = self.segmentation_head([high_res_skip, fpn_features[0], fpn_features[1], fpn_features[2]])
        return logits

# ---------------------------------------------------------------------------
# 4. Focal + Dice Loss 
# ---------------------------------------------------------------------------
class FocalDiceLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, dice_weight=0.5, num_classes=9):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.dice_weight = dice_weight
        self.num_classes = num_classes

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (self.alpha * (1 - pt) ** self.gamma * ce_loss).mean()

        probs = F.softmax(logits, dim=1)
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        
        intersection = (probs * targets_one_hot).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets_one_hot.sum(dim=(2, 3))
        dice_loss = (1.0 - (2. * intersection + 1e-5) / (union + 1e-5)).mean()

        return focal_loss + (self.dice_weight * dice_loss)

# ---------------------------------------------------------------------------
# 5. Ultra-Safe Training Execution Loop (MULTI-GPU READY)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    IMAGE_DIR = "/kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap/images/train"
    MASK_DIR  = "/kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap/label/train"

    # Because you are using 2 GPUs now, an effective batch size of 2 means 
    # 1 image per GPU. If VRAM permits with the Base+ model, you might be able to bump this up.
    BATCH_SIZE = 2 
    ACCUMULATION_STEPS = 8 
    EPOCHS = 30 
    LEARNING_RATE = 3e-4

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = PromptlessEarthMapDataset(IMAGE_DIR, MASK_DIR, image_size=1024, is_train=True)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True, num_workers=4, pin_memory=True)

    print("Building Hybrid Attention SAM 2 Architecture (r=128)...")
    
    SAM2_CHECKPOINT = "/kaggle/working/sam2/checkpoints/sam2_hiera_base_plus.pt"
    if not os.path.exists(SAM2_CHECKPOINT):
        SAM2_CHECKPOINT = "/kaggle/working/checkpoints/sam2_hiera_base_plus.pt"

    model = UNet_SAM2_Hybrid(
        sam2_checkpoint=SAM2_CHECKPOINT, 
        model_cfg="configs/sam2/sam2_hiera_b+.yaml",
        num_classes=9,
        device=DEVICE
    )

    model = model.to(DEVICE)

    # --- THE MULTI-GPU ACTIVATION CODE ---
    if torch.cuda.device_count() > 1:
        print(f"🔥 Dual-GPU Activated! Utilizing {torch.cuda.device_count()} GPUs via DataParallel.")
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=5e-5)
    
    criterion = FocalDiceLoss(alpha=0.25, gamma=2.0, dice_weight=0.5)
    scaler = torch.amp.GradScaler('cuda')

    print(f"🚀 Starting Hybrid Attention Training on {DEVICE} for {EPOCHS} Epochs...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for batch_idx, (images, masks) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}"):
            images = images.to(DEVICE, non_blocking=True)
            masks  = masks.to(DEVICE, non_blocking=True)

            with torch.autocast(device_type="cuda" if "cuda" in DEVICE else "cpu", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, masks) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()
            epoch_loss += loss.item() * ACCUMULATION_STEPS

            if (batch_idx + 1) % ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(dataloader):
                scaler.unscale_(optimizer) 
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

        current_lr = scheduler.get_last_lr()[0]
        print(f"✅ Epoch {epoch+1} | Avg Loss: {epoch_loss/len(dataloader):.4f} | LR: {current_lr:.6f}")
        scheduler.step()

    # ---------------------------------------------------------------------------
    # 6. Saving Hybrid Weights (Unwrapping from DataParallel)
    # ---------------------------------------------------------------------------
    rescue_dir = "/kaggle/working/checkpoints_hybrid_attention_focal"
    os.makedirs(rescue_dir, exist_ok=True)

    # Safely unpack the original model from the DataParallel wrapper so loading it later works perfectly
    model_to_save = model.module if torch.cuda.device_count() > 1 else model

    torch.save(model_to_save.preprocessor.state_dict(), f"{rescue_dir}/hybrid_stem.pth")
    model_to_save.image_encoder.save_pretrained(f"{rescue_dir}/hybrid_sam2_backbone")
    torch.save(model_to_save.segmentation_head.state_dict(), f"{rescue_dir}/hybrid_segformer_head.pth")
    torch.save(model_to_save.stem_to_sam.state_dict(), f"{rescue_dir}/hybrid_stem_adapter.pth")

    print(f"🎉 Hybrid Architecture weights successfully saved to {rescue_dir}")
