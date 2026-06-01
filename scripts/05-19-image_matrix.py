
#!/usr/bin/env python3
"""Create an image matrix (collage) from a dataset arranged by class directories.

The script builds an image with `n_rows` x `n_cols` tiles where each class
occupies `m_rows_per_class` rows. Images are sampled per-class from subfolders
of `path`.

Example:
  python scripts/05-19-image_matrix.py \
	/path/to/dataset 8 8 2 --seed 42 --thumb 256 --output collage.png

The dataset directory should have one subdirectory per class, each containing
image files.
"""

from pathlib import Path
import random
import argparse
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import sys

VALID_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_class_dirs(root: Path):
	if not root.exists():
		raise FileNotFoundError(f"Path not found: {root}")
	dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and not p.name.startswith('.')]
	return dirs


def collect_images(cls_dir: Path):
	# Recursively collect image files in the class directory, skipping any 'masks' subfolders
	imgs = []
	for p in cls_dir.rglob('*'):
		if not p.is_file():
			continue
		if p.suffix.lower() not in VALID_EXT:
			continue
		# skip files under any directory named 'masks'
		if any(part.lower() == 'masks' for part in p.parts):
			continue
		imgs.append(p)
	return sorted(imgs)


def center_crop(im: Image.Image):
	w, h = im.size
	s = min(w, h)
	left = (w - s) // 2
	top = (h - s) // 2
	return im.crop((left, top, left + s, top + s))


def build_collage_class_label(path: Path, n_rows: int, n_cols: int, m_rows_per_class: int, seed: int, thumb: int, output: Path):
	rng = random.Random(seed)
	classes = list_class_dirs(path)
	k = len(classes)
	if k == 0:
		raise SystemExit(f"No class subdirectories found in {path}")

	if n_rows != m_rows_per_class * k:
		raise SystemExit(f"n_rows ({n_rows}) must equal m_rows_per_class ({m_rows_per_class}) * num_classes ({k})")

	# How many images per class we need
	per_class_needed = m_rows_per_class * n_cols

	# prepare figure
	tile = int(thumb)
	dpi = 100
	fig_w = (n_cols * tile) / dpi
	fig_h = (n_rows * tile) / dpi
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)
	# ensure axes is 2D array
	axes = np.atleast_2d(axes)
	plt.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.01, hspace=0.01, wspace=0.01)

	# For each class pick images
	class_image_lists = []
	for cls in classes:
		imgs = collect_images(cls)
		if len(imgs) == 0:
			class_image_lists.append([None] * per_class_needed)
			continue
		if len(imgs) >= per_class_needed:
			chosen = rng.sample(imgs, per_class_needed)
		else:
			# allow repeats if not enough images
			chosen = imgs.copy()
			chosen += [rng.choice(imgs) for _ in range(per_class_needed - len(imgs))]
		rng.shuffle(chosen)
		class_image_lists.append(chosen)

	# Fill axes with images
	for class_idx, chosen in enumerate(class_image_lists):
		for row_in_class in range(m_rows_per_class):
			global_row = class_idx * m_rows_per_class + row_in_class
			for col in range(n_cols):
				img_idx = row_in_class * n_cols + col
				src = chosen[img_idx]
				ax = axes[global_row, col]
				ax.axis('off')
				if src is None:
					# show blank white
					blank = np.ones((tile, tile, 3), dtype=np.uint8) * 255
					ax.imshow(blank)
					continue
				try:
					with Image.open(src) as im:
						im = im.convert("RGB")
						im = center_crop(im)
						im = im.resize((tile, tile), Image.LANCZOS)
						arr = np.array(im)
						ax.imshow(arr)
				except Exception:
					blank = np.ones((tile, tile, 3), dtype=np.uint8) * 255
					ax.imshow(blank)

	# Draw class labels on left side (figure coordinates)
	for class_idx, cls in enumerate(classes):
		start_row = class_idx * m_rows_per_class
		mid_row = start_row + m_rows_per_class / 2.0
		# figure y coordinate: top=1, bottom=0 -> convert
		y_fig = 1.0 - (mid_row / n_rows)
		fig.text(0.02, y_fig, cls.name, va='center', ha='left', fontsize=10)

	fig.savefig(output, bbox_inches='tight')
	plt.close(fig)
	print(f"Saved collage to {output}")

def build_collage(path: Path, n_rows: int, n_cols: int, m_rows_per_class: int, seed: int, thumb: int, output: Path):
	rng = random.Random(seed)
	classes = list_class_dirs(path)
	k = len(classes)
	if k == 0:
		raise SystemExit(f"No class subdirectories found in {path}")

	if n_rows != m_rows_per_class * k:
		raise SystemExit(f"n_rows ({n_rows}) must equal m_rows_per_class ({m_rows_per_class}) * num_classes ({k})")

	# How many images per class we need
	per_class_needed = m_rows_per_class * n_cols

	# prepare figure
	tile = int(thumb)
	dpi = 100
	fig_w = (n_cols * tile) / dpi
	fig_h = (n_rows * tile) / dpi
	fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), dpi=dpi)
	# ensure axes is 2D array
	axes = np.atleast_2d(axes)
	plt.subplots_adjust(left=0.15, right=0.99, top=0.99, bottom=0.01, hspace=0.01, wspace=0.01)

	# For each class pick images
	class_image_lists = []
	for cls in classes:
		imgs = collect_images(cls)
		if len(imgs) == 0:
			class_image_lists.append([None] * per_class_needed)
			continue
		if len(imgs) >= per_class_needed:
			chosen = rng.sample(imgs, per_class_needed)
		else:
			# allow repeats if not enough images
			chosen = imgs.copy()
			chosen += [rng.choice(imgs) for _ in range(per_class_needed - len(imgs))]
		rng.shuffle(chosen)
		class_image_lists.append(chosen)

	# Fill axes with images
	for class_idx, chosen in enumerate(class_image_lists):
		for row_in_class in range(m_rows_per_class):
			global_row = class_idx * m_rows_per_class + row_in_class
			for col in range(n_cols):
				img_idx = row_in_class * n_cols + col
				src = chosen[img_idx]
				ax = axes[global_row, col]
				ax.axis('off')
				if src is None:
					# show blank white
					blank = np.ones((tile, tile, 3), dtype=np.uint8) * 255
					ax.imshow(blank)
					continue
				try:
					with Image.open(src) as im:
						im = im.convert("RGB")
						im = center_crop(im)
						im = im.resize((tile, tile), Image.LANCZOS)
						arr = np.array(im)
						ax.imshow(arr)
				except Exception:
					blank = np.ones((tile, tile, 3), dtype=np.uint8) * 255
					ax.imshow(blank)
     
    # # Draw class labels on left side (figure coordinates)
	# for class_idx, cls in enumerate(classes):
	# 	start_row = class_idx * m_rows_per_class
	# 	mid_row = start_row + m_rows_per_class / 2.0
	# 	# figure y coordinate: top=1, bottom=0 -> convert
	# 	y_fig = 1.0 - (mid_row / n_rows)
	# 	fig.text(0.02, y_fig, cls.name, va='center', ha='left', fontsize=10)
  
	plt.tight_layout()
	fig.savefig(output, bbox_inches='tight')
	plt.close(fig)
	print(f"Saved collage to {output}")


def parse_args():
	p = argparse.ArgumentParser(description="Create an image matrix (collage) from dataset folders (one folder per class).")
	p.add_argument("--path", type=Path, help="Path to dataset root (contains one subfolder per class)")
	p.add_argument("--n_rows", type=int, default= 4, help="Total number of rows in output grid")
	p.add_argument("--n_cols", type=int, default = 5, help="Number of columns in output grid")
	p.add_argument("--m_rows_per_class", type=int, default=1, help="Number of rows allocated per class")
	p.add_argument("--seed", type=int, default=1, help="Random seed for sampling")
	p.add_argument("--thumb", type=int, default=128, help="Tile size in pixels (square)")
	p.add_argument("--output", type=Path, default=Path("collage.png"), help="Output filename")
	return p.parse_args()


def main():
	args = parse_args()
	build_collage(args.path, args.n_rows, args.n_cols, args.m_rows_per_class, args.seed, args.thumb, args.output)


if __name__ == "__main__":
	main()
