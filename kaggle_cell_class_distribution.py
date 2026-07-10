# ============================================================
# RSASE Analysis - OpenEarthMap Class Distribution
# Paste this whole file into one Kaggle notebook cell. CPU is enough.
# ============================================================
import os, json, pathlib, logging, warnings

os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")
logging.captureWarnings(True)
logging.getLogger("py.warnings").setLevel(logging.ERROR)
import cv2
import numpy as np
import matplotlib.pyplot as plt

DATA_ROOT = "/kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap"
OUT_DIR = pathlib.Path("/kaggle/working/results/class_distribution")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = (
    "Unknown", "Bareland", "Rangeland", "Developed", "Road",
    "Tree", "Water", "Agriculture", "Building"
)

def count_split(split):
    mask_dir = pathlib.Path(DATA_ROOT) / "label" / split
    if not mask_dir.exists():
        raise FileNotFoundError(mask_dir)
    counts = np.zeros(len(CLASS_NAMES), dtype=np.int64)
    files = sorted(mask_dir.glob("*.tif"))
    for path in files:
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise RuntimeError(f"Could not read {path}")
        bincount = np.bincount(mask.reshape(-1), minlength=len(CLASS_NAMES))
        counts += bincount[: len(CLASS_NAMES)]
    total = int(counts.sum())
    rows = []
    for i, name in enumerate(CLASS_NAMES):
        rows.append({
            "split": split,
            "class_id": i,
            "class_name": name,
            "pixels": int(counts[i]),
            "percent": float(100.0 * counts[i] / total) if total else 0.0,
        })
    return rows, counts

all_rows = []
split_counts = {}
for split in ["train", "val"]:
    rows, counts = count_split(split)
    all_rows.extend(rows)
    split_counts[split] = counts

csv_path = OUT_DIR / "class_distribution.csv"
with open(csv_path, "w") as f:
    f.write("split,class_id,class_name,pixels,percent\n")
    for row in all_rows:
        f.write(f"{row['split']},{row['class_id']},{row['class_name']},{row['pixels']},{row['percent']:.8f}\n")

json_path = OUT_DIR / "class_distribution.json"
with open(json_path, "w") as f:
    json.dump(all_rows, f, indent=2)

x = np.arange(1, len(CLASS_NAMES))
width = 0.36
train_pct = 100 * split_counts["train"][1:] / max(1, split_counts["train"].sum())
val_pct = 100 * split_counts["val"][1:] / max(1, split_counts["val"].sum())

plt.figure(figsize=(12, 5))
plt.bar(x - width / 2, train_pct, width, label="Train")
plt.bar(x + width / 2, val_pct, width, label="Validation")
plt.xticks(x, CLASS_NAMES[1:], rotation=30, ha="right")
plt.ylabel("Pixel share (%)")
plt.title("OpenEarthMap class distribution excluding Unknown")
plt.legend()
plt.tight_layout()
plot_path = OUT_DIR / "class_distribution.png"
plt.savefig(plot_path, dpi=180)

print("DONE")
print("CSV:", csv_path)
print("JSON:", json_path)
print("Plot:", plot_path)
