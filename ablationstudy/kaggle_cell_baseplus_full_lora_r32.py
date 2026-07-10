# ============================================================
# RSASE Ablation - Base+ Full LoRA r32
# Paste this whole file into one Kaggle notebook cell.
# Warnings are suppressed for clean notebook output.
# ============================================================
RUN_VARIANT = "full"
LORA_R = 32
NOTEBOOK_TITLE = 'RSASE Ablation - Base+ Full LoRA r32'

import os, sys, subprocess, pathlib, urllib.request, base64, zlib, logging, warnings

os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
os.environ["PIP_ROOT_USER_ACTION"] = "ignore"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["SAM2_BUILD_CUDA"] = "0"
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

SAM2_DIR = pathlib.Path("/kaggle/working/sam2")
CKPT_DIR = SAM2_DIR / "checkpoints"
CKPT_PATH = CKPT_DIR / "sam2_hiera_base_plus.pt"
CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
RUNNER_PATH = pathlib.Path("/kaggle/working/sam2_lulc_experiments.py")
OUT_DIR = f"/kaggle/working/runs/base_plus_full_lora_r{LORA_R}"

print("Notebook:", NOTEBOOK_TITLE)
print("Variant:", RUN_VARIANT)
print("LoRA rank:", LORA_R)
print("Checking Kaggle PyTorch...")
import torch
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
try:
    import torchvision
    print("Torchvision:", torchvision.__version__)
except Exception as exc:
    print("Torchvision check skipped:", repr(exc))
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], check=False)

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

def ensure_hydra_stack():
    try:
        import hydra  # noqa: F401
        import omegaconf  # noqa: F401
        print("OK hydra")
        print("OK omegaconf")
        return
    except Exception as exc:
        print(f"Installing pinned Hydra stack; import failed: {exc!r}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "PyYAML"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "antlr4-python3-runtime==4.9.3"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "omegaconf==2.3.0"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "hydra-core==1.3.2"])
    import hydra  # noqa: F401
    import omegaconf  # noqa: F401
    print("OK hydra")
    print("OK omegaconf")

print("Checking/installing remaining packages...")
ensure_hydra_stack()
for import_name, pip_name in [
    ("transformers", "transformers"),
    ("accelerate", "accelerate"),
    ("peft", "peft"),
    ("albumentations", "albumentations"),
    ("cv2", "opencv-python-headless"),
    ("tqdm", "tqdm"),
    ("iopath", "iopath"),
    ("portalocker", "portalocker"),
    ("safetensors", "safetensors"),
]:
    ensure_package(import_name, pip_name)

print("Cloning/installing SAM2...")
if not SAM2_DIR.exists():
    subprocess.check_call(["git", "clone", "https://github.com/facebookresearch/sam2.git", str(SAM2_DIR)])
subprocess.check_call([sys.executable, "-m", "pip", "install", "--no-cache-dir", "--no-deps", "-e", str(SAM2_DIR)])

print("Downloading Base+ checkpoint...")
CKPT_DIR.mkdir(parents=True, exist_ok=True)
if not CKPT_PATH.exists():
    urllib.request.urlretrieve(CKPT_URL, CKPT_PATH)

print("Writing experiment runner...")
RUNNER_PATH.write_bytes(zlib.decompress(base64.b64decode('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')))

print("Checking paths...")
for p in [TRAIN_IMAGES, TRAIN_MASKS, VAL_IMAGES, VAL_MASKS, CKPT_PATH, RUNNER_PATH]:
    if not pathlib.Path(p).exists():
        raise FileNotFoundError(p)
    print("OK", p)

print("CUDA smoke test...")
subprocess.check_call([
    sys.executable, "-c",
    "import warnings; warnings.filterwarnings('ignore'); import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"
])

cmd = [
    sys.executable, str(RUNNER_PATH), "train",
    "--sam2-repo", str(SAM2_DIR),
    "--model-size", "base_plus",
    "--model-cfg", "configs/sam2/sam2_hiera_b+.yaml",
    "--checkpoint", str(CKPT_PATH),
    "--variant", RUN_VARIANT,
    "--lora-r", str(LORA_R),
    "--lora-alpha", str(LORA_R * 2),
    "--train-images", TRAIN_IMAGES,
    "--train-masks", TRAIN_MASKS,
    "--val-images", VAL_IMAGES,
    "--val-masks", VAL_MASKS,
    "--out-dir", OUT_DIR,
    "--epochs", "30",
    "--batch-size", "1",
    "--accumulation-steps", "16",
    "--eval-batch-size", "1",
    "--workers", "2",
]
print("Starting training:")
print(" ".join(cmd))
subprocess.check_call(cmd)
print("DONE. Outputs saved under:", OUT_DIR)
