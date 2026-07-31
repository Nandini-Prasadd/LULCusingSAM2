
NOTEBOOK_TITLE = 'Fusion Comparison Without TTA'
NEED_BASE = True
NEED_TINY = True

import base64
import json
import logging
import os
import pathlib
import shutil
import subprocess
import sys
import warnings
import zlib

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

DATA_ROOT = pathlib.Path("/kaggle/input/datasets/aletbm/global-land-cover-mapping-openearthmap")
VAL_IMAGES = DATA_ROOT / "images" / "val"
VAL_MASKS = DATA_ROOT / "label" / "val"


INPUT_ROOT = pathlib.Path("/kaggle/input")


def locate_unique(filename):
    matches = [path for path in INPUT_ROOT.rglob(filename) if "checkpoints" in path.parts]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} anywhere under {INPUT_ROOT}. "
            "Expand the Models input in Kaggle and confirm the file was uploaded."
        )
    if len(matches) > 1:
        print(f"Multiple matches for {filename}; using {matches[0]}")
    return matches[0]


def locate_weights(checkpoint):
    model_root = checkpoint.parent.parent
    expected = model_root / "checkpoints_hybrid_attention_focal"
    if expected.exists():
        return expected
    matches = [
        path.parent.parent
        for path in model_root.rglob("adapter_model.safetensors")
        if path.parent.name == "hybrid_sam2_backbone"
    ]
    if not matches:
        raise FileNotFoundError(f"Could not locate trained hybrid weights under {model_root}")
    return matches[0]


BASE_CKPT = locate_unique("sam2_hiera_base_plus.pt") if NEED_BASE else None
TINY_CKPT = locate_unique("sam2_hiera_tiny.pt") if NEED_TINY else None
BASE_WEIGHTS = locate_weights(BASE_CKPT) if NEED_BASE else None
TINY_WEIGHTS = locate_weights(TINY_CKPT) if NEED_TINY else None

if NEED_BASE:
    print("Detected Base+ checkpoint:", BASE_CKPT)
    print("Detected Base+ weights:", BASE_WEIGHTS)
if NEED_TINY:
    print("Detected Tiny checkpoint:", TINY_CKPT)
    print("Detected Tiny weights:", TINY_WEIGHTS)

SAM2_DIR = pathlib.Path("/kaggle/working/sam2")
RUNNER_PATH = pathlib.Path("/kaggle/working/sam2_lulc_experiments.py")
NORMALIZED_ROOT = pathlib.Path("/kaggle/working/normalized_models")
BASE_RUN = NORMALIZED_ROOT / "base_plus_full"
TINY_RUN = NORMALIZED_ROOT / "tiny_full"

print("Notebook:", NOTEBOOK_TITLE)
import torch
print("Torch:", torch.__version__)
print("CUDA:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise RuntimeError("Enable a Kaggle GPU accelerator before running this cell.")
print("GPU:", torch.cuda.get_device_name(0))

subprocess.run(
    [sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
    check=False,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
)


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


def ensure_hydra_stack():
    try:
        import hydra  # noqa: F401
        import omegaconf  # noqa: F401
        return
    except Exception:
        pass
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "PyYAML"])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "antlr4-python3-runtime==4.9.3"
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--no-deps", "omegaconf==2.3.0"
    ])
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--no-deps", "hydra-core==1.3.2"
    ])

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
    ("matplotlib", "matplotlib"),
]:
    ensure_package(import_name, pip_name)

if not SAM2_DIR.exists():
    subprocess.check_call(["git", "clone", "-q", "https://github.com/facebookresearch/sam2.git", str(SAM2_DIR)])
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q", "--no-cache-dir", "--no-deps", "-e", str(SAM2_DIR)
])

RUNNER_PATH.write_bytes(zlib.decompress(base64.b64decode('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')))


def require(path):
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Kaggle input: {path}\n"
            "Add both model datasets and the OpenEarthMap dataset to this notebook."
        )
    return path


required_paths = [VAL_IMAGES, VAL_MASKS]
if NEED_BASE:
    required_paths.extend([BASE_CKPT, BASE_WEIGHTS])
if NEED_TINY:
    required_paths.extend([TINY_CKPT, TINY_WEIGHTS])
for required_path in required_paths:
    require(required_path)


def run_runner(*args):
    cmd = [sys.executable, str(RUNNER_PATH), *map(str, args)]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def normalize_model(model_size):
    if model_size == "base_plus":
        run_dir = BASE_RUN
        checkpoint = BASE_CKPT
        weights = BASE_WEIGHTS
        model_cfg = "configs/sam2/sam2_hiera_b+.yaml"
    elif model_size == "tiny":
        run_dir = TINY_RUN
        checkpoint = TINY_CKPT
        weights = TINY_WEIGHTS
        model_cfg = "configs/sam2/sam2_hiera_t.yaml"
    else:
        raise ValueError(model_size)
    if not (run_dir / "config.json").exists():
        run_runner(
            "import-existing",
            "--sam2-repo", SAM2_DIR,
            "--model-size", model_size,
            "--model-cfg", model_cfg,
            "--checkpoint", checkpoint,
            "--variant", "full",
            "--weights-dir", weights,
            "--out-dir", run_dir,
            "--symlink",
        )
    return run_dir


def archive_results(out_dir, archive_name):
    archive_base = pathlib.Path("/kaggle/working") / archive_name
    archive_path = shutil.make_archive(str(archive_base), "zip", root_dir=out_dir)
    print("Results folder:", out_dir)
    print("ZIP archive:", archive_path)
    print("Files:")
    for path in sorted(pathlib.Path(out_dir).rglob("*")):
        if path.is_file():
            print(" -", path)

base_run = normalize_model("base_plus")
tiny_run = normalize_model("tiny")
out_dir = pathlib.Path("/kaggle/working/results/03_fusion_without_tta")
run_runner(
    "fusion", "--sam2-repo", SAM2_DIR,
    "--base-run-dir", base_run, "--tiny-run-dir", tiny_run,
    "--val-images", VAL_IMAGES, "--val-masks", VAL_MASKS,
    "--out-dir", out_dir, "--eval-batch-size", "1", "--workers", "2",
    "--alpha-steps", "21",
)
archive_results(out_dir, "rsase_fusion_without_tta")