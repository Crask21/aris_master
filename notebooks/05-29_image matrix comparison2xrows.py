from pathlib import Path
import random

from PIL import Image
import matplotlib.pyplot as plt


SEED = 2
IMAGE_PATH = Path("/media/aris/Data/master2025dev/Figures/Image matrices")
MODELS = ["Diff-Aug", "Diff-Mix", "Diff-II", "DA-Fusion", "Real_Guidance", "SDEdit"]
MODELS_WITHOUT_IMAGE_GUIDANCE = ["Conditional", "masked"]
VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def is_image_file(path: Path) -> bool:
	return path.is_file() and path.suffix.lower() in VALID_EXT


def list_images(folder: Path) -> list[Path]:
	return sorted([path for path in folder.iterdir() if is_image_file(path)])


def pick_one(rng: random.Random, images: list[Path]) -> Path | None:
	if not images:
		return None
	return rng.choice(images)


def pick_images(rng: random.Random, images: list[Path], count: int) -> list[Path]:
    if len(images) <= count:
        return images
    return rng.sample(images, count)


def build_class_dirs(real_root: Path) -> list[Path]:
	if not real_root.is_dir():
		raise FileNotFoundError(f"Real folder not found: {real_root}")

	class_dirs = [
		folder
		for folder in sorted(real_root.iterdir())
		if folder.is_dir() and any(is_image_file(path) for path in folder.iterdir())
	]
	if not class_dirs:
		raise RuntimeError(f"No class folders with images found in {real_root}")
	return class_dirs


def find_matching_images(model_class_dir: Path, real_stem: str) -> list[Path]:
	if not model_class_dir.is_dir():
		return []
	return [path for path in list_images(model_class_dir) if real_stem in path.stem]


def main() -> None:
    rng = random.Random(SEED)
    real_root = IMAGE_PATH / "Real"
    class_dirs = build_class_dirs(real_root)
    

    # for model_name in MODELS:
    # 	model_root = IMAGE_PATH / model_name
    # 	if not model_root.is_dir():
    # 		print(f"Skipping missing model folder: {model_root}")
    # 		continue

    # Two rows per class
    row_count = len(class_dirs) * 2

    # Real column + spacer + one column per model.
    # fig = plt.figure(figsize=(3.2 * (len(MODELS) + 2), max(6, row_count * 2.2)))
    # gs = fig.add_gridspec(
    #     row_count,
    #     len(MODELS) + 2,
    #     width_ratios=[1, 0.06] + [1] * len(MODELS),
    # )

    # axes = [[fig.add_subplot(gs[row_idx, col_idx]) for col_idx in range(len(MODELS) + 2)] for row_idx in range(row_count)]
    fig, axes = plt.subplots(8, 9, figsize=(16,14))
    # Hide the spacer column.
    for row_idx in range(row_count):
        axes[row_idx][1].axis("off")

    # axes[0][0].set_title("Real", fontsize=18)
    # for model_idx, model in enumerate(MODELS, start=2):
    #     axes[0][model_idx].set_title(model, fontsize=18)

    for class_i, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        real_images = list_images(class_dir)
        chosen = pick_images(rng, real_images, 2)

        for subrow in range(2):
            row_idx = class_i * 2 + subrow

            if subrow < len(chosen):
                real_image = chosen[subrow]
            else:
                real_image = None

            if real_image is None:
                for ax in axes[row_idx]:
                    ax.axis("off")
                continue

            real_stem = real_image.stem

            with Image.open(real_image) as image:
                axes[row_idx][0].imshow(image.convert("RGB"))
            axes[row_idx][0].axis("off")

            # if subrow == 0:
            #     axes[row_idx][0].set_title(class_name, fontsize=12)

            for model_idx, model_name in enumerate(MODELS, start=1):
                ax = axes[row_idx][model_idx]
                model_class_dir = IMAGE_PATH / model_name / class_name
                matching_images = find_matching_images(model_class_dir, real_stem)
                selected_image = pick_one(rng, matching_images)

                if selected_image is not None:
                    with Image.open(selected_image) as image:
                        ax.imshow(image.convert("RGB"))
                else:
                    ax.text(0.5, 0.5, "Missing", ha="center", va="center")
                ax.axis("off")
            
            # For the MODELS_WITHOUT_IMAGE_GUIDANCE, we can just pick any 4 images from the class folder since they won't have matching stems.
            for model_idx, model_name in enumerate(MODELS_WITHOUT_IMAGE_GUIDANCE, start=len(MODELS)+1):
                ax = axes[row_idx][model_idx]
                model_class_dir = IMAGE_PATH / model_name / class_name
                all_images = list_images(model_class_dir)
                selected_images = pick_images(rng, all_images, 4)
                if selected_images:
                    # Just show the first one since we only have one cell per model in this layout.
                    with Image.open(selected_images[0]) as image:
                        ax.imshow(image.convert("RGB"))
                else:
                    ax.text(0.5, 0.5, "Missing", ha="center", va="center")
                ax.axis("off")

    plt.tight_layout()
    save_path = IMAGE_PATH / f"{model_name}_comparison.png"
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
	main()
