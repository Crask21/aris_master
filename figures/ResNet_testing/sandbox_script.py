import random
import numpy as np
from pathlib import Path
import sys

import torch
from PIL import Image
from matplotlib import pyplot as plt
project_root = next(
	(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "src").exists()),
	Path.cwd()
)

sys.path.insert(0, str(project_root))
import json
from src.utils.add_note import add_note
from src.waste_diffuser.augmentations import build_image_augmentations
config_path = project_root / "queue/TEST_DATA_AUGMENTATION.json"
with config_path.open("r", encoding="utf-8") as f:
    config = json.load(f)
print(config["augmentation"])
# Create augmentations
augmentations = build_image_augmentations(config)
print("Successfully created augmentations pipeline from config")
def _resolve_path(p):
    p = Path(p)
    return p if p.is_absolute() else (project_root / p)

def _find_train_dirs(cfg):
    dirs = []

    # Common top-level keys
    for key in ("train_dir", "train_data_dir", "data_dir"):
        v = cfg.get("data", {}).get(key)
        if isinstance(v, str):
            rp = _resolve_path(v)
            if rp.exists():
                dirs.append(rp)

    # Class-level directories
    classes_cfg = cfg.get("data", {}).get("classes", {})
    if isinstance(classes_cfg, dict):
        for _, class_info in classes_cfg.items():
            if not isinstance(class_info, dict):
                continue
            for key in ("train_dir", "train_data_dir", "real_data_dir", "data_dir"):
                v = class_info.get(key)
                if isinstance(v, str):
                    rp = _resolve_path(v)
                    if rp.exists():
                        dirs.append(rp)

    # De-duplicate while preserving order
    seen = set()
    unique_dirs = []
    for d in dirs:
        if d not in seen:
            seen.add(d)
            unique_dirs.append(d)

    return unique_dirs

def _get_train_transform(augs):
    if isinstance(augs, dict):
        for k in ("train", "training", "train_transform"):
            if k in augs:
                return augs[k]
    return augs

train_dirs = _find_train_dirs(config)
if not train_dirs:
    raise FileNotFoundError("No train directory found in config.")

image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
image_paths = []
for d in train_dirs:
    image_paths.extend([p for p in d.rglob("*") if p.suffix.lower() in image_exts])

if not image_paths:
    raise FileNotFoundError(f"No images found in train directories: {train_dirs}")

transform = _get_train_transform(augmentations)
print(f"Using transform: {transform}")

n_images = min(4, len(image_paths))
n_augments = 1
#selected = random.sample(image_paths, n_images)
selected = [image_paths[163], image_paths[1000], image_paths[-6600], image_paths[-2500]]
selected = [Path("/home/ap/cloud/master/dataset/datasets/wood/all_wood_categories/train/impregnated_wood/images/img_bold-eagle_2024-11-04T16-40-15-319.png"),
            Path("/home/ap/cloud/master/dataset/datasets/plastic/train/hard_plastic/images/img_blushing-lion_2025-10-16T11-23-40-051.png"),
            Path("/home/ap/cloud/master/dataset/datasets/plastic/train/soft_plastic/images/img_blushing-lion_2025-11-26T08-34-21-732.png"),
            Path("/home/ap/cloud/master/dataset/datasets/mineral_wool/all_mineral_wool_categories/train/glass_wool_brown/images/img_gallant-stag_2025-10-15T10-04-44-454.png")]
fig, axes = plt.subplots(n_augments + 1, n_images, figsize=(3.2 * n_images, 3.2 * (n_augments + 1)))
if n_augments + 1 == 1:
    axes = [axes]

augmentation = "combined"  
for r, img_path in enumerate(selected):
    img = np.array(Image.open(img_path).convert("RGB"))

    # Original
    axes[0][r].imshow(img)
    axes[0][r].set_title(f"Original\n{img_path.name}", fontsize=9)
    axes[0][r].axis("off")

    # Augmented variants
    for c in range(1, n_augments + 1):
        out = transform(img)
        aug = out["image"] if isinstance(out, dict) and "image" in out else out

        if torch.is_tensor(aug):
            aug = aug.detach().cpu()
            if aug.ndim == 3 and aug.shape[0] in (1, 3, 4):
                aug = aug.permute(1, 2, 0)
            aug = aug.numpy()

        if isinstance(aug, np.ndarray) and aug.dtype != np.uint8:
            if aug.min() < 0.0:
                aug = (((aug + 1.0) / 2.0) * 255.0).clip(0, 255).astype(np.uint8)
            elif aug.max() <= 1.0:
                aug = (aug * 255.0).clip(0, 255).astype(np.uint8)
            else:
                aug = aug.clip(0, 255).astype(np.uint8)

        axes[c][r].imshow(aug)
        axes[c][r].set_title(f"Aug: {augmentation}", fontsize=9)
        axes[c][r].axis("off")

plt.tight_layout()
image_output_path = project_root / f"{augmentation.lower()}_augmentation_samples.png"
plt.savefig(image_output_path, dpi=100)
print(f"Saved augmentation samples to {image_output_path}")
add_note(notes_path = "/home/ap/cloud/Master/aris_master/figures/ResNet testing/notes.md", title=f"{augmentation} Augmentation", content = image_output_path)
# plt.show()

print("Train directories used:")
for d in train_dirs:
    print("-", d)
print(f"Images found: {len(image_paths)}")  