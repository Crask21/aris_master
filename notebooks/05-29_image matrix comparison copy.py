from pathlib import Path
import random

from PIL import Image
import matplotlib.pyplot as plt


SEED = 1
IMAGE_PATH = Path("/media/aris/Data/master2025dev/Figures/Image matrices")
MODELS = ["Diff-Aug", "Diff-Mix", "Diff-II", "DA-Fusion", "Real_Guidance", "SDEdit"]
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

    row_count = len(class_dirs)
    col_count = len(MODELS) + 2

    fig = plt.figure(figsize=(3.6 * col_count, max(4, row_count * 3.4)))
    outer_gs = fig.add_gridspec(
        row_count,
        col_count,
        width_ratios=[1, 0.08] + [1] * len(MODELS),
        wspace=0.15,
        hspace=0.28,
    )

    # Column 0 is the real image anchor. Column 1 is a spacer. Columns 2..7 are 2x2 comparison grids.
    real_axes = []
    model_titles = []

    for row_idx, class_dir in enumerate(class_dirs):
        class_name = class_dir.name
        real_images = list_images(class_dir)
        real_image = pick_one(rng, real_images)

        real_ax = fig.add_subplot(outer_gs[row_idx, 0])
        real_axes.append(real_ax)

        spacer_ax = fig.add_subplot(outer_gs[row_idx, 1])
        spacer_ax.axis("off")

        if real_image is None:
            real_ax.axis("off")
            real_ax.text(0.5, 0.5, "Missing", ha="center", va="center")
        else:
            with Image.open(real_image) as image:
                real_ax.imshow(image.convert("RGB"))
            real_ax.axis("off")

        if row_idx == 0:
            real_ax.set_title("Real", fontsize=18)

        real_ax.text(-0.02, 0.5, class_name, transform=real_ax.transAxes, ha="right", va="center", fontsize=11)

        for model_idx, model_name in enumerate(MODELS, start=2):
            model_cell = outer_gs[row_idx, model_idx].subgridspec(2, 2, wspace=0.02, hspace=0.02)
            cell_axes = [fig.add_subplot(model_cell[r, c]) for r in range(2) for c in range(2)]

            if row_idx == 0:
                model_titles.append((model_idx, model_name))

            if real_image is None:
                for ax in cell_axes:
                    ax.axis("off")
                continue

            real_stem = real_image.stem
            model_class_dir = IMAGE_PATH / model_name / class_name
            matching_images = pick_images(rng, find_matching_images(model_class_dir, real_stem), 4)

            for ax, image_path in zip(cell_axes, matching_images):
                with Image.open(image_path) as image:
                    ax.imshow(image.convert("RGB"))
                ax.axis("off")

            for ax in cell_axes[len(matching_images):]:
                ax.axis("off")
                ax.text(0.5, 0.5, "Missing", ha="center", va="center")

    for model_idx, model_name in model_titles:
        # Put the model title on the top-left tile of each 2x2 cell so it stays attached to the panel.
        title_ax = fig.axes[row_count + 2 * (model_idx - 2)] if False else None

    # Add titles by walking the first row's sub-axes in plotting order.
    first_row_offset = 2
    for model_idx, model_name in enumerate(MODELS, start=2):
        fig.axes[first_row_offset + (model_idx - 2) * 4].set_title(model_name, fontsize=18)

    plt.tight_layout()
    save_path = IMAGE_PATH / "comparison_2x2_matrix.png"
    plt.savefig(save_path, dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
