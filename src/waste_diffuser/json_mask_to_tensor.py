"""Utilities for converting JSON run-length masks to binary image masks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

try:
	import torch
except ImportError:  # pragma: no cover - torch is available in the training env, but keep this import-safe.
	torch = None


def _load_json_payload(mask_json: str | Path | dict[str, Any]) -> dict[str, Any]:
	if isinstance(mask_json, (str, Path)):
		with open(mask_json, "r") as handle:
			return json.load(handle)
	return mask_json


def _iter_annotations(payload: dict[str, Any]) -> Iterable[dict[str, Any]]:
	annotations = payload.get("annotations", [])
	if isinstance(annotations, dict):
		return [annotations]
	return annotations


def json_mask_to_binary_mask(
	mask_json: str | Path | dict[str, Any],
	image_size: tuple[int, int] = (1200, 1920),
	one_based_indices: bool = False,
) -> np.ndarray:
	"""Convert a JSON mask with run-length segmentation into a binary mask.

	Args:
		mask_json: Path to a JSON file or an already loaded JSON dictionary.
		image_size: Image size as (height, width).
		one_based_indices: Set to True if start_positions are 1-based instead of 0-based.

	Returns:
		A uint8 numpy array of shape (height, width) with values in {0, 1}.
	"""

	payload = _load_json_payload(mask_json)
	height, width = image_size
	mask = np.zeros((height, width), dtype=np.uint8)
	total_pixels = height * width

	for annotation in _iter_annotations(payload):
		segmentation = annotation.get("segmentation") or {}
		start_positions = segmentation.get("start_positions", [])
		run_lengths = segmentation.get("run_lengths", [])

		if len(start_positions) != len(run_lengths):
			raise ValueError(
				"Invalid segmentation: start_positions and run_lengths must have the same length."
			)

		for start, run_length in zip(start_positions, run_lengths):
			flat_start = int(start) - 1 if one_based_indices else int(start)
			flat_end = flat_start + int(run_length)

			if flat_start < 0 or flat_start >= total_pixels:
				continue

			flat_end = min(flat_end, total_pixels)
			mask.reshape(-1)[flat_start:flat_end] = 1

	return mask


def center_crop_and_resize_mask(
	mask: np.ndarray,
	output_size: tuple[int, int] = (128, 128),
) -> np.ndarray:
	"""Center-crop a binary mask to a square and resize it with nearest-neighbor interpolation."""

	if mask.ndim != 2:
		raise ValueError(f"Expected a 2D mask, got shape {mask.shape}")

	height, width = mask.shape
	crop_size = min(height, width)
	top = (height - crop_size) // 2
	left = (width - crop_size) // 2
	cropped = mask[top : top + crop_size, left : left + crop_size]

	pil_mask = Image.fromarray((cropped > 0).astype(np.uint8) * 255, mode="L")
	resized = pil_mask.resize((output_size[1], output_size[0]), resample=Image.Resampling.NEAREST)
	return (np.array(resized) > 0).astype(np.uint8)


def json_mask_to_tensor(
	mask_json: str | Path | dict[str, Any],
	image_size: tuple[int, int] = (1200, 1920),
	output_size: tuple[int, int] = (128, 128),
	one_based_indices: bool = False,
):
	"""Convert a JSON mask directly to a downscaled binary torch tensor."""

	if torch is None:
		raise ImportError("torch is required to convert the mask to a tensor")

	mask = json_mask_to_binary_mask(
		mask_json=mask_json,
		image_size=image_size,
		one_based_indices=one_based_indices,
	)
	mask = center_crop_and_resize_mask(mask, output_size=output_size)
	return torch.from_numpy(mask.astype(np.uint8))

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Convert a JSON mask to a binary tensor and save as a .pt file.")
    parser.add_argument("--input_json", type=str, help="Path to the input JSON mask file.", default="/media/aris/Data/master2025dev/datasets/aris_4_class/10_real/real/train/hard_plastic/annots/annot_brave-panther_2024-11-13T14-47-25-769.json")
    args = parser.parse_args()

    output_tensor = json_mask_to_tensor(args.input_json)
    #Import image and overlap with tensor for verification
    from PIL import Image
    image_path = args.input_json.replace("annots", "images").replace("annot", "img").replace(".json", ".png")
    image = Image.open(image_path).convert("RGB").resize((128, 128))
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)  # Convert to (C, H, W)
    
    #Overlay the mask on the image for visualization
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image_tensor.permute(1, 2, 0))  # Convert
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.title("Mask Overlay")
    plt.imshow(image_tensor.permute(1, 2, 0))  # Original
    plt.imshow(output_tensor, alpha=0.3, cmap="Reds")  # Overlay mask
    plt.axis("off")
    plt.show()