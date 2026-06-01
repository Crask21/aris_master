from pathlib import Path
import random
from PIL import Image
import matplotlib.pyplot as plt

seed = 1
image_path = Path("/media/aris/Data/master2025dev/Figures/Image matrices")
models = ["Diff-Aug", "Diff-Mix", "Diff-II", "DA-Fusion", "Real_Guidance", "SDEdit"]
models = ["Real_Guidance", "SDEdit"]
# models = ["Real_Guidance"]

valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
real_root = image_path / "Real"


def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in valid_ext


def list_images(folder: Path) -> list[Path]:
    return sorted([path for path in folder.iterdir() if is_image_file(path)])


def pick_images(rng: random.Random, images: list[Path], count: int) -> list[Path]:
    if len(images) <= count:
        return images
    return rng.sample(images, count)


if not real_root.is_dir():
    raise FileNotFoundError(f"Real folder not found: {real_root}")

class_dirs = sorted(
    [folder for folder in real_root.iterdir() if folder.is_dir() and any(is_image_file(path) for path in folder.iterdir())]
)
if len(class_dirs) < 4:
    raise RuntimeError(f"Need at least 4 class folders in {real_root}")

rng = random.Random(seed)
selected_classes = rng.sample(class_dirs, 4)

for model_name in models:
    model_root = image_path / model_name
    if not model_root.is_dir():
        print(f"Skipping missing model folder: {model_root}")
        continue

    import matplotlib.gridspec as gridspec

    # Create a 4x6 GridSpec where column 1 is a narrow spacer between the real and generated columns.
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(4, 6, width_ratios=[1, 0.05, 1, 1, 1, 1])

    # Build axes as a 2D list for indexing: axes[row][col]
    axes = [[fig.add_subplot(gs[r, c]) for c in range(6)] for r in range(4)]

    # Titles: Real in first visible column, Generated in the following four visible columns
    axes[0][0].set_title("Real", fontsize=20)
    axes[0][2].set_title("Generated", fontsize=20)
    axes[0][3].set_title("Generated", fontsize=20)
    axes[0][4].set_title("Generated", fontsize=20)
    axes[0][5].set_title("Generated", fontsize=20)

    for row_idx, class_dir in enumerate(selected_classes):
        class_name = class_dir.name
        real_images = list_images(class_dir)
        if not real_images:
            for ax in axes[row_idx]:
                ax.axis("off")
            continue

        real_image = rng.choice(real_images)
        real_stem = real_image.stem

        generated_class_dir = model_root / class_name
        generated_matches = []
        if generated_class_dir.is_dir():
            generated_matches = [path for path in list_images(generated_class_dir) if real_stem in path.stem]
            generated_matches = pick_images(rng, generated_matches, 4)

        with Image.open(real_image) as image:
            axes[row_idx][0].imshow(image.convert("RGB"))
        axes[row_idx][0].axis("off")

        # Place generated images into columns 2..5 (index offset by +1 because column 1 is spacer)
        for col in range(1, 5):
            ax = axes[row_idx][col + 1]
            match_index = col - 1
            if match_index < len(generated_matches):
                generated_image = generated_matches[match_index]
                with Image.open(generated_image) as image:
                    ax.imshow(image.convert("RGB"))
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", va="center")
            ax.axis("off")

    plt.tight_layout()
    save_path = image_path / f"{model_name}_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.show()
