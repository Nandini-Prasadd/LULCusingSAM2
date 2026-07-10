# ============================================================
# RSASE Ablation - Base+ Frozen Linear
# Paste this whole file into one Kaggle notebook cell.
# ============================================================
RUN_VARIANT = "frozen_linear"
NOTEBOOK_TITLE = "RSASE Ablation - Base+ Frozen Linear"

import os, sys, subprocess, pathlib, urllib.request, base64, zlib, logging, warnings

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

SAM2_DIR = pathlib.Path("/kaggle/working/sam2")
CKPT_DIR = SAM2_DIR / "checkpoints"
CKPT_PATH = CKPT_DIR / "sam2_hiera_base_plus.pt"
CKPT_URL = "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
RUNNER_PATH = pathlib.Path("/kaggle/working/sam2_lulc_experiments.py")
OUT_DIR = "/kaggle/working/runs/base_plus_frozen_linear"

print("Notebook:", NOTEBOOK_TITLE)
print("Variant:", RUN_VARIANT)
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

print("Writing frozen-linear experiment runner...")
RUNNER_PATH.write_bytes(zlib.decompress(base64.b64decode('eNrtfX9z20iO6P/+FFxdvVdkQsu2EudmVE9b58kks6lLZvKS7My98nOxaImSuaZILkk59qZ8n/0A9C90synJTvbutm6qZmKRDaDRaDQajUY3/+kPR5u2ObrMy6OsvAnqu+6qKp8djEajg39NV6siO2yydHEXZLd11uTrrOyCZlOWWRMsqyborrLg49m7YBL8Umflq7Tprt6lddBkN3mbV+X44OBNF7Sbuq6arp0eHAaf8vLu6Ie0zZ4GXZPmZV6ugs95dxXMq3KZrzZNellkAfyTdoAf3KRNnpZdOwbUjwAL7KyrRVYE2U1abAQMoQMv+LfadMGnT2dju6blBpmBKtY10Gurchpkf92kRRws89tsEcyLtG0P089pk8WCkUVWzuF3Wi4OgmBVVJdpEaRFfZUG7ecsq5H+27QDoLs4WGfrqrkj4ADop+usy5rDebUBUV0CyNU6ba6Bd0T6v1Br3gHfN1lAzc1awX9e1psuDn76FBPjcUCcx8Grss3WIBJBPmsaaOg6BQYOXkGHzDvgXnRT0GbdpkYRWz2xSDsg1AVrZAeAN+UCuu7omnCOqFaSLfVhk9VVm3fQmiDtNNDnitg/atP1JEZB1yCt4PAQnw8RxRCYX2Xz67rKocd0jYCwqD6XRZUu/PUjhFPXmPTvYNlU6yBJlpsOBJUkQb5GPQJJlFVHfd8eHKh3zQpk32bqed7eqJ/Yfer3X6Dz1e+qVb8akG21Vk/t1abLC/10p8E6UH/BE0qVlAa6TzHQLvI59KAuEpB12l0V+aWCeg+PoqC7q1H35fsfCfcNKE5Knf02b+H5lxobiXr6EfRVqOSnTV1kutnzm4n6uU67uqg6qOzgwPweb9osHJ2tVqOoDzgG7qDLWmA+WMvfPqg7/IVAddGp8nKzru/wXVlr+VTN/Mp6GJclgZTu2/FyU85F4xDgtZAJq3NeFVWjhYviyBYv8R0ovxQgkcKuascodC1K+P0Wla2J6Tdov0T462KtgPD3wUHX3E1hfAe6D4vLDVo4oVvI2NlBdjvP6i54RX/gtUA4C2bBz1WZMRqiu7Nlp5mumvQlWbU4eA/v36HhioNV1iUIlpAh69HHWuGdoAkWss2CDzCQQPVe4eAPR2/KtkuLQlR1mYEVzsgigzZNgzqvwZQYgFEk+AKKBwcHL9+effyY/Hz27tVH4D+kKkZ/Lq9LGJ+jWDz+AEawgPGgnj+k5cp68WN2kxVVnRkIELb6/anJMvX7NzCQjXo4WzX5fFPgQNY1bfJiAVzDc6R4e/nL218+GObC4/FxHKh/opi9POm9fIHPz/GfiXl5is/iH/XyxE/zVL88tSqikhMDeaJfHo+/c16+UDShQa/f/NurHxPRrN9evfnpT5+wXUJrYeZooS8J95z+pV9AVdA+voiD4J8C2TUc4J8J4JkEUJ01TEJ3Hwd5QSDPJYjuUKseat/kVFKBLualz6j0hSzFTrfIU+kzWUpqwIsnVPzPspgpxjCLSlUI4kLIfAEWNJsJeS5hdumeTUjuB/9iTDD9G/wqnIiPMF+KYVXCHD0N2k6wBRYyWbf1NLisqkK/STvoJBrw5v0VeEIJViuQDw5+Pfvw5uznTx+nZMDP4WXMa7uADv8ilB1G4d+yMinyMkub0ZRDhU5hrDiavU6LNottdtRLzctsJPGkLo7ASKYDFfGib1JNm63AAK0zf02m9MGVGVRVHyAnZWWQ3Bp75abOT83mUVUuN0XR6yt8t520eDdE+R70ZpEtwV/LFgkMvQY8blDtEJ+nYLy7KDj8I00uag5A52SMxQQTCRWux/4CMSDWaQmeZuItm28WKQdIYLLoAV2m8+usXLQIDdO19mNBo7F5sg3pYpGgG5h0VYKeTkgP6BPSCHFakkOjVTm5s1WLHsjVOLuF+b01yBGVGlhw90Aw6IoR/FQbCvVmDHNe1nQhGA1DhKDmnxfAMlQE0y78DiPFCTyMsYHofYcjcm5Hol5EeUiNAB9JeYiZP7lEc0WCcQUiCA2LTXaB8im0X4EAY01WeRimHoLe6koMuBO6jBT+ZbUpZOOl94v1Bi8L6MMj5VcIV3+ZN+ii4iIwu03X4JRO/385ssmtcvBQETe46rq6nR4dwZurDXp366NlOs/Asl7DEgjMyvyKumCMKL6FR48293QOMy+OwWBeEMkBlkpNycUnuo/alJA24oicgzbkMJlk7VQ74GjjL0ivlXNOb6QLCMLQSKg/jILmZS197hktTMb4j6kpBt7mm6aF9SGZkcioAS38ABXpShpTSySg1M6AIrDIhmLtp2IuEOHRkijg/xRm5SRfp6ssWeRN2NawcjXD2tN8ScWRopnVLT6WI2sReCQXqu1RWmTd5fpILLoP0XWBtTSYyUPw/HHJdAiuCk5EHdij+oj4a4++EHf30rUcquPvQfPJk78XWb6Of/Jg7D3h1Xj55s1ghB/VEIY/iHFBvyJHZddpe/3fWGOL9DIrvq3CPoTk/h39QKp2Lz8QeT/whyvrg+k+phUMfQjBr6kmXBZSMCBp879lWxV2mRcZrl5avbKgyRDU9g4c1RH5EVd5Br43vhrXHeNh1K5T8mc5GL1z4C5BrZO62LQOrH7vwBdps8ocWHpn4O4t7oF53ZBz0/KLrx+T0AVfFOndHYb8HrGY5UNwH4k2xKCtIEIm8+UqgWmfZONVEPgr1EIEzweUQhaK1nINGd+l68KrIEMobQ/F0pUhtMunPTytM0M4BUe555ohMSzFEWJrwEnLm0z41OS322MoDj5fpZ1HfuA6IbhvUYK/mQslecC3LEL3Gnr056p7XW3KhfCrl6N3edtihPcL1nk/Dj5k8wzcOljjfUHs+xF2tghQvAfntO6KrG2V6ZGBy1D+lQxgG5MkL/MuScyIaLNiaSSrHTZqZszcTjErOq8FuFAr0GXQoJPjyXNW3ia0TyOiIFAs1s2OPJKqyVc+EGcFqNgdi2pRDqi0LSw1YKFq/GHVB3+B8RXqFsXB6Mm4y5ejKIpsctS4/agpOXiJgR7A0AxdFqPgDzNTYCpzXGuhC7+mxca3uBL24A1SPUISgdghWuct+eGgFt6q70UXtcCvh9wXH1f31Nnt2EawWolLPLeqHY1Zjn6uApSYZAhWJKDtuBr5onsIldrTz6he0DHmwQZiOgRQ7Mnp5Cwtobysx2nTpHfh+fH4+XcUS3x+KmK/xy9giIu4IEDJoKDDUdstHCKTyfcU65yI2DHFPHcQgSFRthjSUdsATLRqxJAxOYNHErY9BryEzsYvwQxUbdbXm/PeG9qEGP8JpPS3qoRF8Osir8N6ZoLXfeBfs6bL53uBfqDQ0gfcZcu+P94B/AnbQHxvh/t4lS+7j8BAJgiHXjgSDUImRb7OO6B4/GJyGg/DIkENezIM2FClEvL5FoqXVbPIGtqemc1vJuMffvnw46sPyYdXr9++evkpOTneUglK4Jm/eIewfwB9v+pKmANeQoc2aduROCcetAtnXLO5AcwBTA2oWmR5waL3pi6vleE0VlmXd9la0omDfHEr4pI8kPQZFBaFk68xPaBH8ByQYBghxJt3H16d/Sh2diwbhERgdPRHhsf0mMAUVicsCVhMb7XcCFGZZHV+09EOYggVC9aIp+SHnz5MPvz0gx+pycj7oldxEDpGLXatXBSjpLKmrkQCAynQm58/gf68ffPzqzMmAZoD+kI0drwvw58+nP2/jy/P3r6y5EiEHilIxFVydGrmYmS8SoHgm6+TB0rj1cdPVlMcmzhoO9PNCvdpMzTlNo7oqZnsL+Ryhv9EdqTMlAIBTex8RAUjEDp7h1AjM+TkLEWQ43lV34WGdikMueBhDEMYJpGQTSLBUTA5PR0fB4dmRsOXamJyvDKxS+huGmIlsW/3KxrXWbPegGGdwEQWByd2/w2QEx3JyRVVuer3Cnf0DjzBRM5yzCuMSWSx9qtxxYCrH7/NiA72pau953ewlM5pXnnfZHVTzcGGopzK8btqsSkyn/MsLVsJS/C0LLOiVd7vM2B00/VfT2LfrqDcDfF5uRvojDAa6xodD8IiBmSsZxsU1js3JyfotJTgIZQ3k0XIGI+DCfT1ddaUcjE0OzmJYX2ywP3SGS7JM9cpI4LfP4Dg93sQfLY/vWd7kDvdn9zpMDlQXrvX+h6YLktq0aFQ69kirTFF6+xm9R5eAgMn0TbM5VzgiU2CLk+Lvm8DxW9pAzX87hjb4JnXAeRD9vbPob9IYmPrvxvA/piv1lW+cAn4uv/WVijkieu9owAOhctSIP+AC5efwR4BBY4duf59sREIonnG2QCL/TltFnI4gpshLNAnGuA0qvgL03vQhHnaJZjtp8wZPNtC7/vMmpmQjavwNvII0wf6/d6gz/aGPO1DXtiPi3w9Yw5n3zBvUfBL8B7iIIH/cPLWMhvTDG5rNFAo1WRqD4nQIEbjmzz7HCLZfbCX8xALGBZMSta81OtL9vCEqLrTgSPDyzLkOs15jUx85WO2ek377+/evv8TuD1/urtsYJg8bIaAlUPbsY1AmBjAUcjWl9kClr5rNVO8gFVkuVknMj1Qvf7+EXMEzGV/acXAEXxiGtwuJd9hhiSIHPZz1gB3yMdDuHzUa+xh8CGTFvXDGbh9igEFV+oH/tWPIy6Z5bvVEpuma85B0XBJ5NYJnusEwwrbhOIIA+D7AL7mw+sfm6oGLQUsWLT2iyWTxAHTpmF2Bo3qMksxt4pvYXObekF2vwbjLWKAlF6Kuk2O+8UWG9xhBLdTsR1Vzfnxxbi9Suvs/HAyZV7zpkNNPr+w9rNRwQWH2Ot/y+vQ6L3h3Imy3QIdhAixPHJ3wG9Z7Ri1Y1z298GR1OuxWaBk4S2sYFC0DC2maPxsdJnrxKm0yFegLhV0RdOKTCKbEWzuOK3rrFyAgddFRbXKSQ5MX0MzfSFWLAy+E5MkPKdlvNusynUtdtPEa9k+hvzQ9qkwApEzJjbHFJDX739G+/pI39vV9q+xnUQiX+ZZM+xIbhlYg6Ophmnt7zWibPUwLQh5rTDAen3xbTtadamYJT+evZts78+BjQjayzG7ZO52hNrhct7LAx9W0p0p9SmHKV1kN/lcbI9BwQjz3EYOPzoTCwG8OUMx644mTT5nGJZrxc6JtZkkQ78OfKO3USbfOUV0foTrul28EDPCNKDlPIAcj49Ph3dRhvXfpDVRnGIwI80ZMVLygCN/2cVM9DiizJPrs5gluBscl2GObI15b5jG1g+ey6rHKie3v+pyKhhY/MMSs7dkn3Hi+q1nYWezaIwHUnzmtxQkd9wKFaeTZqwTQq3qsTsmYqmyM/HHsfmO9vX5FKERMEDVgsycPmQwxmy3BCRFGxHZIjSs2UhxrxrDQ1aw/hC79rNeHvNupoaqPnA9Pzo6RUmXPTJjfayqDT3pbFQ6lpu/bbJq0oXahWStaXvTJLRc7CUDtDms0feZm5kY2n0f1ozrmfnZB5PexJpMaDs7H/31+gas7wjdGPwL0jyRfyeji4FqpH2Y8Yc+6GWetrNRCcNqNBQEGOwq+2TKLqURomP6z9RF5z6TzgwrS5ut9HmbBJGAC9uTcH2CaEuPqpkdHU3CU/9cuF7iVjPjkno2uQie6jd7NsK/4jS+h15+zJwF48xq7IAbInZg9w6XMCdAha5bn5eufAzKOsGWTOnkle3e2G48WLREnOJThHsBCstis5C+SPJ2De7wful1XivfiJMU0fbW0W/GlltFiJQi6FKHX68ElBdPOMzUz68vgUMZteiPplAzwDJ4mRcHSLjIDDmh85F+AtDRRWTvKlOyMziboV/pYnvgRN4gvR/V8i8t7zU62JOAKzaQ7haiysN8XcHE/SPMfG+rtt1n0SBdKOMh4VmiVbpes5cTPIq1AKJyWuPgp99sfSEOA88ER3YR8YPmFP/aRYwtAGBP+/paA/ZAuPy2PYjltLO/mQBuCugIWjnOG/gF2tw1uMWllhSSImapLzZ0flPONUY6tQnJZrd1eCiJ8kR26HRVUciE+SQIT4JDIBAFT55wUT5RnEW0Y8a6AozApeC3rZbdOr3VnIpVtBOsaBMcXFdVRyjyd6gbxa2w2w9mc+2YnMATmFgisfPG+KFlWJvN5aZOKPh74lYfjdvNOkQWcavuGVvsb0qBSYguFAwqh9AgHVItJeMT2nEMYWAAKxaLT4OT7PAUtyBDUbN80RO0Sso0ffe0r9FPTLU89PoO3NB83g4OandExnhOo9gsMvFWjdLjoVyy+bb1SIYxHouek8lTi2ygv2Wg8KEVEbBzgF48d8ONj8csH4u56h6L2VVdiuuSY8/rhIR0bIzLpsazKLJ7YK5dPMC02B1EyElZk0rDz/Ei69L5VSj3BA5PovG83sAjHScP+wOWUOXD3sisxU9njNS4n3umBQCAmIcdsor/MFM6JAZsZJ/GmYN7CMuVBk/Yyl0IZ9bVzaZmzBDDKu+CmdXQPoTU0XN4f6FZrIP/HXQ9lphi9sH/fRt8acP/+w76q86C7xSgGd83aZEv6ByGSUUC6ZcLyrmbumklSrX/yLQTi86l9C96qzl9gKm9NtXm1QaWL40YEtsrXmRltda5I7UyZkvzq7fxBGQ+X2VNFgrcP2KGhUI/EgTF4W/DEFnDB3I0CZ58LVecxABnoJRzuqDlqwX2DcQEvKRF8bWcfIsOq/PbrEjS+XzTpPM7wweZ1F4inZj8JUkxCFQuD1mVyGQ0kSnCBStWOFRfIqapvFwlG3Hwf5CDedWA1DojBln9oTYYauj4mVb4R7YRdDhGs9hnel5dZWVyDeuidJA/vRlMdKzEY7Ryx97FCdbC542EEqTbrVpXVz0ZHAVGFWramqIml6I85JQBNEK3UxkgfI4iv8hCqAqc04ycJeFRAXXyl77nxg8qWacNUx5zM0B1+RcQ+sWUBbvBTqoGcKvJPMpqoxeYln2zfT0F45gc7inLEc+W0D0TwJuOY1LB9kYomwib7K/GhJ9TGy5QRPYb2TdSXK6Ev9hHe9dvqj+P5LIthFYrGsInjUQAB96M0xJmfa2isUMFF5aaDArmkXTeK0lpYlp2j6T4geSpyQnxPo7WezQhZ/M5EBPdapuwaAD81a269sWL5zFFLqX1S9QEUfPXN+P1b7zPQ9KqJ4Hpe+WN7EnuJdqof0UTpdrHrZbbFGtgAQI8g+Gi+IwLao8vgMUXQ8CeIQYY+u0QmjvaAEe8GkIgqIRO0wEsgbCbfhi0unpiU35u0lqGeenfqUkyETOverLOyYpgsAhjO0GptbjcCDMb0i59nzbAb1bIDhI3HclrL9IbGVVO6rTBQ+pO/SIRTZ9Ukse9pn076iw3RCgOL9gKJYFIvR+vr/E8MNSXgdmX93PQ+a6kumbn7NGv6MtGlIlb5uqsROpg4OTBtTHeKIax+894e0QbLI11x5LxApYm4VzeAkXb3uB5dDN5oGTt7HOQcNi+DTRXVXdFEWRxbYSKE44ifmUHIofrfnRu3NLBB7yeDFQncOj1oOvuaqRvyVgPRm9ZIgivnYNvr7jL1m5dO+PAdkMN9M6akhRTOnEniWoUqij26IQu0o4ZLMXaEP+J99z8JRU0++TyMKbe38P4IFAbszd4G17/9C9BmfOMUZ+OdbSR7yAuR18c7PveZXxSxvPlSnGk9ySRIc9pUy9DZodY3Xt0TnDyvXV8lyUPDGUEzFgz+nkBM9wwddMC1BauNyNgRszwoIiTGSAAxG9PXoAo1o/DiQAz94WbAyAouZuGbLPQlDs7htZOn4GytvuU+uKNiom0FNJeNZuSWU7E91lzYY5IlZTFlIiRa9gO2HlBhqYOx0bu3TLbDsIyunRqUNapzpkYA8sqcm2q3q8l44oSCJcC3dFYkUsKsOcj81Ie4+BKy+DkGw5kB+JxpMpGrLIuHLHSUdxDiHp8ifHHKej3MHk8YCC6RsVrZng9zsgTte1rh+yUhv3M5Qxnrp5mbZvDaMt4SCW3TlZcRzUVj4buWYO4qcshLzyffv00Btnko07QgF7q6vDUSp0U1VycgGJGSDadJqhdjKnpUl4gRbPZPkjWzKenWmrO7qm9D7etxbodO1usWdg54/fAtjHAxbKTB6XWooauCjmI4pLdmCZKROgAVh5/DE5oA1tggc+L7iw5vQ7zIoWIe8XcrbQ8a7zUOFTmHc+WJWAGiry8DttmPqVujoMFZqGLn+3dGkvtg/f9S9cAQw8HGvjwnLeJxA57txvoy9okeZNHANOjRAKGiBW1FZDkNO7BL6+auxmUjsWLsBdP0dQZDNvRoct36WBd12SZrkdQsXM+GOyEAcq7IGA4yOQ3fYmInhCNeJjN0gOJu1rUvXyqD0aKnMINdLuVLZPrD0WQaLhrEnzYd13yzTxKNXuyW0LY7Dh1J1F2X4eaHKfW7Bm7dHAqmz7OxeT3kDjT1TTwuYrW/DsNhr2/kTmPquDYkVVWbbVp9G4m9faIPKmQ58f1ZZK0dYZREHEDdOj3jSP7IhPSMAz9mQMcYeifH2N73qSLUwZL9XUsbH6Nhybe2Fe3mmdiexZR9/n0CvilQFdWZXqBx+qxAhfh4DwVe6ewYSYUgMuMHScZrO3AOaN/oW/zA6siAitkW+gn7fnJDmS2qKFt6Oy2C7m6YEgZcZCWQtY00Qp6gCNtrEGatMq0Uqqgoi2Hvbc54JhzVS2DL5qBe/LDGQf8zLc7/+i4DO6eSFnoKUiuncSDjIaTA8TkMqAFfQH0UkOV5Pqpmds9w3gbvOvnxfsT54q4ZxX7ojgV2Qq9d2Xb0HpHZraq4IHn+gqT9q79JN3XPdXcrZa81URoSCstjVQ1crXcoQ0DykpK2OTgT+KiUsXcFEm+Go5UNFIg1Q3OtHinT01XDgU9zwDKq+CLpCZufPoXeQV+RXnKIGAKitKnLRJMa0go9ccTFfXlfMb8zMNAZle6rhMBpQ9FYI+pX6UkISK1o3m9GZnluKCWbjpwo9sulC4wZZoYsp7LCE5eYBW8Zl01VSPvGJawkecIjIhn8NRO6SnLEKCV7qXysHTa14CYO/TS/0eL99grWiq60nl7S7wgSECxF+oaqvNnFzgx4R+DfLMv8kQgTzjy1f7YcSBrF7+GFSOEpj7FRj1F5vDXjU7Xg2H9XCSJSmVB3aDk98fqhjhpAuollmIDqqIYtbUw1hd8ybWnurUA4ER3DliHHqJaONJ2EH3NJsnk52NEJiLNC24zRAleFuiWSCu2V/4o3gck4XEtbaqzVtSf6RWCnk/j4PhCJHKdxMHhCT8a/pnYUWAnQ2BSnpLok8BUCh0uaTxh7dMLa/WBn0QcwcQdl8UjJDQgCqRPdETunGoyEwkqKEtQJYSOvvpDCLLxploXwdJ4keSiKw3+OOsRZMzgW33LpEjMxMBKU31uQ5m6MLTVRkcBekXqJl0gYE4GoNOWx9pvy2BZljWUUiiqOLf2LC94BALoqJx7y1H50vd9BI18gVu28VAxVgEA5K32YfJqg2s8xZW9E3xxnnsO49AeMEdy9oQHsGqWRaBRfTvEA/iNyhnQyL194j7mvXNWXioOClnqAHN8xIWZ1LcYz5rKHnYCJu/V3ZrRWMQt9g1imPi+CM9t3TFFBpz9Us7vvL3h7GJ7pkP6+a34lxFmqssX0vK0D0ZA9hlPP81GvbZSQzBEBG0ZI9e/0YsQDxnnWbGgoTGjzXysEs/VX2d3uCiMHBpj+oNuLp5Y8BbS8MZ/9NBPrzFRm1AOhq4N9VwZ6lwXKl5e4h0I7ks8VJs1LXuz7SrRHdeItleb5bLI3CKRX6W/9yQzzOUXz2ZbrlVlV4qa+0D5NWaK15n6EXMWZ+y3NbAML8Z4SX5idnhIiWtmfsY8qoltncm/bBOzqWpYRradYcraBpUSn8m/prDOy0R8pE5ot7WLiHFSIuaLjrrfKBELJ/OpEMRL5HWgzh41gVoALFBpviowIhCKG4lbM8Xleyykymqia023VCTKWT36KvheNZR67NZykxamNTJ8V2xnHwAYsmJQ427hyGB+y3Axk5YY3ujwu4Pd7bvYeUtcm5eDEVMq8OmwiJq7eqgHldFCrvBcN5VAdRv0QXGRAKY6BXdgjJw91zcb2GiomHAjKytzh+xsXYl77x35bZWhLsTNn8QnzGGBWkJ1LmbmknWKpJbs3teN/su2xaq6y9cgBXNdIb3Bm9nWv5mOWOYFzplFur5cpEE9DWr7GHisdjDZ+XEWyCpUpgbL0hCLGFhYz9O7Gdv9EW+4frbzqwzXhy6PRZPoovHLClZv2VlZZik4Aqu3H0LdtDj4BJpyKyrJ6mp+heeCQWLrXO5Xyge5SqCJXGTw2gcnWU6JOKAlkkrEAUT+nl5YhyPlvqh5oZqW8nal63r8E4iT7lkA0yVu1gC3Hj9WueB7q5iI1XTyo0oEpnKjf9/D+k/ZwzIGRIH5TMoIE33XG3EnK26I1Lr2fglDE1qqQKXOskv9G1XER9SIDyAF0B9UogKh8boG8ciroKQqXY2TYjUyOVUWiJNlNeJJVRZg716FEZ8nAdQ/bY7YvKmBnFlgZGYMAPFNHyM9d0iAPoVHbjjuCm8nNGx7QW7h3WfgX6zFOYTDE3lGo6hWydZVlx0LIE0xh+WY9kRuqofwRc0yRptLOvQoorngTmPQAk8eO1/JEudW5MFXfqAEuRAjIV/cxoEOKYrJH1nDT8LaU7wJW3CHCuRDtczw1jinYJG189ly9Iqa+0W0+mlwcn/0hTX5nn2XbNq/l7g1VzTYsy+sJmFIXhbVHG8rchqu1msUO6fb/x+IPBDH3m7b3ai2nkQ90e1eSGMg3N2H+7rwt7P7NfXcrjK452CDkVLpaVgfNZcqdDRkPw9619U3Y/oTijPtuCFPZ/ntOpkqP1WHmPBpjJez41GIJ3tViBt8WvFRGaPgfw0h0tksnOpcjJn4/IWl7X1ByrZtSnEnv/F0+sLU34EW32+eF3lNQzvB262TsO+1oZhvqVS4GwSMaNEQG9iibSwobsVZ5375wwxPerNSZod13FFAm1R9Q8Fv1NW+InHMLtQBA2v5OWYKBuOvjYuT5KAqonMjiisHhuZpUy9ufWBcISmaMDo/vuAHRqxLc/S6yL3FJQw1O1q9EJxCB1qrNAy+8c8BagUlg5x4dh8eN/ThBvqAPD+YETOW4sAydj1HSu2o+G5rBDkrNfiypEn4y/U9TsE0a1xDNTg/MLZoCGIynX0G5gbnFcrJx8EaRfe9rXYxihmlc3HY7QIXTnqi7Y8sPgcP0+iruPesTRw4PgBSd/e2LWs+6D0QJv6QWwnag2DsWddfJjy+D7+dcCUFdp06SKHxJBqgj6EcyStKbBTt11RUc19TxVa+jj9TsJQFnwd3lP3aaZbXfBtROk4UqeRfqT9w7i4UL3q3YugvU+ttRhUA9R8wtVJK6YXoI3E7lbyUw75DwronY3Ycmb0c7i3FwRNYX3faZ1IDkJyf0SshEfyuOxvXHs9mD7+k59DsgaOuhHH2c93NUm0QIvuuCrPmbVZou4lE7zoZpe7SaoirLdSlFFJM+tlO9JWY6pgwD8MqZfJFYq3QZD/+uV+cMqYlBn5oFsMI/kgrD2P669kV07Rr8QZaveE1tiRiq5/eWtcfLbMiZHaCt+ecDAHLhHl5TkbEPfabdvaccuiNpWbfOjG43SyXOV7eMoIqjL9uZQ+MRPrAaPtCcKlMePJFEL3Xttyy40NmejnS+5CGgrTWA/vMEc+iYqZXFjvmVw0SeWvy70PkWwyRb77z0D+Fxb4QedC7JXVoaBKQZ3x6yNNXLsWuB/zaSZmAHMpsdgR54XkKM6WaqfLCjjkSi21VVKMpn0/7VsCdVGP7C51fSwP0DyO+4Gin9EHNx9KBAYsXY5jcoa+g1cuweRwtsfqg2F1ivqSOkaUG80jxsvZJRP7JLTojZT0uwBev0zld0nYiFV7g0/o2umAUjTsETr/4nLCIEyZf0ul4srx/KNPESYqcuCzfH/g9qQEX6rWQ2Xb3ad/YzqODQiybintTZvwal6o/D3qcK5XMZFEzY3Yvas69YpK5vsMmM54cr00gSR48SCJjyuvqkV04Z4P+wvL9RFrXdgewR8yMfocYpXwNEzM34OLQF/dautlv/BWlvnk5sG2H5sISDcEoqezDFRkSDJRtT0PkaXVx8PrNv736MRG3Vvz26s1Pf/r0cUBqfTvl55vgHsK3MVrqkO62HEHO/gCnfSuoOTVFe0l0yK44X/spYFFNok9ddZAXFqWRXyl6RvF8wCL6RS0r3lfYwr/LxTfMMc1p2l8V6WMf2nvE9jPxqhAMnyMeT9hqvE2aQi+iXH66HJvsVIiXwt/N5DY0+A9O8fn1hR2kMSUUQkm4sEe2o/HkiUvM4eiid4INJi4Hhs+nrG5MH+xV7VS3fdkQjuTwAPMsFgwDixAJJtYicmmh1xa5vuq6n0YqjxnJhRken1Ioqp+m1v1i6qL6Xtum7lenYAhsskclnoKiXFWL4axSeYGVE5/zAPLbmxSwfrcNwbq/ycU0hT4S6l4swx298IK+9yWtsrdepA+9TFX1ygeubn7SwOKFD9S+1UnBs7c+JD0kVOxWXIgg3sI6zD0l6KbMDi14Lc3Hhe5eik8r4khkrA4sf7HIs/alBBtKplm3/jubzKIFvzMz3XmjUe/ytZHYuRB1jCjqiIvxsEa/F3CF8azJEtspNZG7FY6JIY8ghDJ0UngivRgYCMHiYu4SptIr6N3r3+MC/1WhM+gvaBC5Tq6icoDzkRweNNdgFDvsLxgUKEtmuVBZQw73FjzLkdHw7KPzMkqwztgUs3OL2718ipLeoEL6VkZ6LRNr6UIKdVxSnn7w5BiwjIL+su8HpcRgIb7J0g93p4L/I7s3bdYb50MXybYw+R5rOu88ip+uwf1OFHOEh1GUwrjeagPS41j7doPTFe0dSK2pSvuTlkQAV3nABt7cvhQJilaC/Nc2/1vznBVp3ZL/7uEar3bFCow2ooCVw6JQj/THPZD0cWQPJ0Sh4yJyTGRzGib6flS6slF0HFliMVRoGtPpt0O0ukWfHrzcRe4BMlQ181G3umQ1MgFjroAEAb8D74CBxRumaJwcT54/efJMepvfOhy/xU3WM9ROT9lAus6ykMCQ20CF++xZWhOmWZr/PnN+i5nzHzIAbs3fTowbE1f5jG7YdyPZLqRhx4pXS31LHGdzqJJz2yulTzEN1OKCslqV0zH1uBxbs2cHs/L3SMi9/0dxOALVKf9Znkd9+S1iyvpzM4+OJ7vekD9meRlDNTtilP/gvtC36ZBv0xlf1xG/e2W/e2WP8MrMtLS3e6ZRvt5Po49BFJXwasL0FodGM2Xfv4ihn7pCXTYxX6e1yLzPFi+romrg2XHd0luYjtqr6jPIT2LM8B/wcvB40TH+TW9n38WB+XSw+JxXljaUD6jIUM4rVh7SvzHMNiBu+vjpRAOlIPZwVC2Xo61uJ+4l5Xgvz83/AH9zn4PI+7qfzik/6zCy0fTf0zkUZVR1/KqrNUTkrfgvf3n7y4ePkWXp2hS/nZjAoASDx2Zn+QLdNxzHtyzdwIM3bmtQ73AUj2SmgX1fpSEmjvXTWZi1PHlNuQWCHN6nC26D1J8oUhtnNbqhgpm1/D1+j3/DObZwxpuHV0EERXqZFTP2LQB4K1qQm8M4WBX/XEDErvBYUEv7UkHNFG5mHKAOyqs+ZkrnzwHzYsCRxPMBf91kGU2F3gOm/90cEn6Li502cKn2PPF6Bv+3yPiNLjZytycyzDMu7oPdo91VkX2jr63hr15x1jTig4aSoT/MFEY0Tls8oYO+xQYGyXdWLgCoRnpLnV8X+KWwS5zmWrzL5wVedLGiiSScPI+DU5YngSjIqpzCzD0LvNBMTKM3+KXVkWdq4vB8jtICZpMuQJ1cxKpZYOB/ojSjAOwh3R6HNmUQc3LBLtgBXHEF/ye0Zjswn11Yt/RIzB/g4eku1OcKFXoEMF+pVZyLRsCnWpyZ9ghGH7IF3l1u+QQnO3wCTpH1AV3rF8DctrUfTv39gFfsFdkKveartFzgV7KlhcNsW1irFtXnrAnmGTKG96qAxZt9zys6tom1m1rwtRz9auZpYV+DL2Ce7qciR+DeYve5TaWjQ6pFegcTJ37fppthZtnx+ETklx2Pv+daS/WmNxl+RJxnxjKvJ8Gqp8fPF/fJF5qjxW2LeCXp/bguV8DMos5nJ99BBZeX1S3MLCiE2Yg4sRS3G8+LqoUG0klNcrHSBX7aar2GhTmd4YcJvsXDBvBAP8dnzWqDNyO+pwLH+xLQYySSSrhwdHiIB5kP8YMIyJtwhmajo+t0BT7vEca+wPs9QiDJ3QAZYWI5DX3GkC0RwM9Jb9K8wP3DkB8v3EobjNWhuYifjgvSORxV1fdbsWkmOKRAjg8ZFydb8WX8z4u8HRNvUvGjTbbLEmbZQ3WvAn56QH6KtwVJwtwF/tt2edEUqFo8v6pwdp+JhDTcEm/XuFkfc18PP06PF5iOLlgHmuI9asNT9QbXXGs7gMROzj8AS53wN61q6WZOfUY6YuwvN0WxnXM8En7Y+LVi8t1uVJF44kOfnL7Yja4OqEsCtJ42JI7Hx6dbaZC7fChPnD9AiAJPnEN/kOyLR1SGWA+uCoRyiLdz7zBISbEp5on48njrfPVHUA7JBA6YR24VxW6yDyrEWGqT08eBZkmyqOZJopK80IdlTUDvh55aREILCLYaZrtRrC+U5ouvWix48Ot6m0siIJlWlxnJT0E4Nl9gRZxET4AU0B42eZPt2BgT/0oS/Ozxobh/wkflux2MiOsofJjPjrejFs3AsHqWHT7fjiqurjgU11n4iZzsJJKBBceLLvz4p/gR7q34GFg5xPPPQxyMdwiAboSxLVTPwExO96FBt8oM0JjsYgNXllKig2zs4AK1kQ4a+w21Hk3iaxCe4SQKDtXtzYMDS8BFFrkB7WilfXLG9VZMeSH10HwukXHwedqgzqsNMo8AEaPRq77ZlNuY9uLsYwmeb6UAi+1D3Jjb3miRROdptkysHmq0KI4sKh5jCDzsaP0AKnpMj0R9kOgGaOwQ3lZcGvlbTO/EDBwKeXuEf2kSlgbkTxARJ/NQtfMjPUh4fhJ7yc6PKnY0vZWe7mzs8HTFMWXQZ1Dih2bHdpvokUzk0n3UABjG3mcMDGM/vCe9ZPbvTS/63j3qxd6nV+VRHbdDWWhgsCcRJmJkHtV/XsR9us6LuJe4vZi4VJdR7u0enxdbYB7KgDT3/9VcIZOahXz1Zbd4o5SzPway1XcfqsXAmH4IuUf7XICqtg+kI08HD4RzznZI7QtW5b6AD9P1Q2wa3o9ZbaGmPQKbjH3JwBZ8dXLTwuanr7fgmpnBRnfytPehYCzdACkng20LTT7YbGK9fUnP7o34kMmvIL1MfMGE08fFJVSZ0GmbJKHqkgQVL0lkXUILD/4D/F++pg==')))

print("Checking paths...")
for p in [TRAIN_IMAGES, TRAIN_MASKS, VAL_IMAGES, VAL_MASKS, CKPT_PATH, RUNNER_PATH]:
    if not pathlib.Path(p).exists():
        raise FileNotFoundError(p)
    print("OK", p)

print("CUDA smoke test...")
subprocess.check_call([
    sys.executable, "-c",
    "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0)); print(torch.ones(1, device='cuda'))"
])

cmd = [
    sys.executable, str(RUNNER_PATH), "train",
    "--sam2-repo", str(SAM2_DIR),
    "--model-size", "base_plus",
    "--model-cfg", "configs/sam2/sam2_hiera_b+.yaml",
    "--checkpoint", str(CKPT_PATH),
    "--variant", RUN_VARIANT,
    "--train-images", TRAIN_IMAGES,
    "--train-masks", TRAIN_MASKS,
    "--val-images", VAL_IMAGES,
    "--val-masks", VAL_MASKS,
    "--out-dir", OUT_DIR,
    "--epochs", "30",
    "--batch-size", "2",
    "--accumulation-steps", "8",
    "--eval-batch-size", "1",
    "--workers", "2",
]
print("Starting training:")
print(" ".join(cmd))
subprocess.check_call(cmd)
print("DONE. Outputs saved under:", OUT_DIR)
