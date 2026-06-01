import argparse
import random
import shutil
from pathlib import Path


DEFAULT_MASTER_PATH = Path("/media/aris/Data/master2025dev/datasets/aris_4_class_master")
DEFAULT_OUTPUT_PARENT = Path("/media/aris/Data/master2025dev/datasets")
# DEFAULT_OUTPUT_PARENT = Path("/media/aris/Data/master2025dev/aris_master/temp")
DEFAULT_SEEDS = [1, 2, 3, 4]
# DEFAULT_SEEDS = [1]
DEFAULT_TRAIN_SIZES = [10, 25, 50, 100, 250, 500, 1000]
VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args():
	parser = argparse.ArgumentParser(
		description=(
			"Generate seeded nested train splits under aris_4_class_<seed>/"
			"<n>_real/real/(train|val|test) from the master dataset."
		)
	)
	parser.add_argument("--master", type=Path, default=DEFAULT_MASTER_PATH)
	parser.add_argument("--output-parent", type=Path, default=DEFAULT_OUTPUT_PARENT)
	parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
	parser.add_argument("--train-sizes", type=int, nargs="+", default=DEFAULT_TRAIN_SIZES)
	parser.add_argument(
		"--keep-existing",
		action="store_true",
		help="Do not delete existing aris_4_class_<seed> folders before writing.",
	)
	return parser.parse_args()


def _strip_prefix(stem: str, prefix: str) -> str:
	return stem[len(prefix):] if stem.startswith(prefix) else stem


def _get_files(folder: Path):
	if not folder.is_dir():
		return []
	return [f for f in folder.iterdir() if f.is_file() and f.suffix.lower() in VALID_EXTENSIONS.union({".json", ".pt"})]


def collect_class_items(class_dir: Path):
	images = {_strip_prefix(f.stem, "img_"): f for f in _get_files(class_dir / "images") if f.suffix.lower() in VALID_EXTENSIONS}
	annots = {_strip_prefix(f.stem, "annot_"): f for f in _get_files(class_dir / "annots") if f.suffix.lower() == ".json"}
	masks = {_strip_prefix(f.stem, "mask_"): f for f in _get_files(class_dir / "masks") if f.suffix.lower() == ".pt"}

	keys = sorted(images.keys())
	return [
		{
			"key": key,
			"image": images[key],
			"annot": annots.get(key),
			"mask": masks.get(key),
		}
		for key in keys
	]


def ensure_split_dirs(root: Path, class_names):
	for split in ["train", "val", "test"]:
		for class_name in class_names:
			class_root = root / split / class_name
			(class_root / "images").mkdir(parents=True, exist_ok=True)
			(class_root / "annots").mkdir(parents=True, exist_ok=True)
			(class_root / "masks").mkdir(parents=True, exist_ok=True)


def copy_items(items, split_root: Path, class_name: str):
	images_dir = split_root / class_name / "images"
	annots_dir = split_root / class_name / "annots"
	masks_dir = split_root / class_name / "masks"

	for item in items:
		shutil.copy2(item["image"], images_dir / item["image"].name)
		if item["annot"] is not None:
			shutil.copy2(item["annot"], annots_dir / item["annot"].name)
		if item["mask"] is not None:
			shutil.copy2(item["mask"], masks_dir / item["mask"].name)


def generate_seed_dataset(master_path: Path, output_parent: Path, seed: int, train_sizes, keep_existing: bool):
	class_dirs = sorted([p for p in master_path.iterdir() if p.is_dir()])
	class_names = [p.name for p in class_dirs]

	output_root = output_parent / f"aris_4_class_{seed}"
	if output_root.exists() and not keep_existing:
		shutil.rmtree(output_root)
	output_root.mkdir(parents=True, exist_ok=True)

	rng = random.Random(seed)
	per_class_items = {}
	for class_dir in class_dirs:
		items = collect_class_items(class_dir)
		rng.shuffle(items)
		per_class_items[class_dir.name] = items

	for n in sorted(train_sizes):
		real_root = output_root / f"{n}_real" / "real"
		ensure_split_dirs(real_root, class_names)

		for class_name in class_names:
			items = per_class_items[class_name]
			max_available = len(items)
			train_count = min(n, max_available)

			if train_count < n:
				print(
					f"[seed={seed}] class '{class_name}' requested {n} train images, "
					f"but only {max_available} are available. Using {train_count}."
				)
			train_items = items[:train_count]
			remaining_items = items[train_count:]
			val_count = len(remaining_items) // 2
			val_items = remaining_items[:val_count]
			test_items = remaining_items[val_count:]
   
			# Use the seeded random order so smaller train sets stay nested subsets.
			# Copy the remaining images to both val and test.
			copy_items(train_items, real_root / "train", class_name)
			copy_items(val_items, real_root / "val", class_name)
			copy_items(test_items, real_root / "test", class_name)

		print(f"Created seed={seed}, n={n} at {real_root}")


def main():
	args = parse_args()
	master_path = args.master
	output_parent = args.output_parent

	if not master_path.is_dir():
		raise FileNotFoundError(f"Master dataset path not found: {master_path}")
	output_parent.mkdir(parents=True, exist_ok=True)

	for seed in args.seeds:
		generate_seed_dataset(
			master_path=master_path,
			output_parent=output_parent,
			seed=seed,
			train_sizes=args.train_sizes,
			keep_existing=args.keep_existing,
		)

	print("Done generating seeded datasets.")


if __name__ == "__main__":
	main()

