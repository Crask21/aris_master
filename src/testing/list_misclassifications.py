#!/usr/bin/env python3
"""List misclassified samples for a checkpoint on a chosen dataset split."""

import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm

from evaluate_resnet18 import load_model
from resnet_dataloader import ResNetDataloader

logger = logging.getLogger(__name__)


class SplitImageDataset(Dataset):
    """Simple dataset for deterministic split evaluation with filepath tracking."""

    def __init__(self, samples, class_lut, transform):
        self.samples = samples
        self.class_lut = class_lut
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image = Image.open(sample["filepath"]).convert("RGB")
        image_tensor = self.transform(image)
        label_idx = self.class_lut[sample["class"]]
        return {
            "image": image_tensor,
            "class": label_idx,
            "filepath": sample["filepath"],
            "true_label_name": sample["class"],
        }


def get_loader_for_split(dataloader_instance, split):
    """Build a deterministic loader for the requested split."""
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Invalid split: {split}. Choose from train, val, test.")

    split_samples = dataloader_instance.generate_data_split(split=split)
    if len(split_samples) == 0:
        raise ValueError(f"No samples found for split '{split}'.")

    dataset = SplitImageDataset(
        samples=split_samples,
        class_lut=dataloader_instance.class_LUT,
        transform=dataloader_instance.val_augmentations(),
    )

    loader = DataLoader(
        dataset,
        batch_size=dataloader_instance.batch_size,
        shuffle=False,
        num_workers=dataloader_instance.num_workers,
        persistent_workers=dataloader_instance.num_workers > 0,
    )
    return loader


def collect_misclassifications(model, dataloader, device, class_names):
    """Run inference and return all misclassified samples."""
    model.eval()
    misclassified = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Scanning split"):
            images = batch["image"].to(device)
            labels = batch["class"].to(device)
            filepaths = batch["filepath"]

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            confs, preds = torch.max(probs, dim=1)

            mismatch_mask = preds != labels
            mismatch_indices = torch.where(mismatch_mask)[0]

            for idx in mismatch_indices:
                i = int(idx.item())
                true_idx = int(labels[i].item())
                pred_idx = int(preds[i].item())
                misclassified.append(
                    {
                        "filepath": filepaths[i],
                        "true_label": class_names[true_idx],
                        "predicted_label": class_names[pred_idx],
                        "confidence": float(confs[i].item()),
                    }
                )

    return misclassified


def main():
    parser = ArgumentParser(
        description="List all misclassifications for a checkpoint on one split"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        required=True,
        help="Path to checkpoint file (.ckpt)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate (default: train)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save misclassifications JSON",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable debug logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[%(levelname)s] %(message)s",
    )

    ckpt_path = Path(args.ckpt)
    config_path = Path(args.config)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    dataloader_instance = ResNetDataloader(str(config_path), preview=False)
    eval_loader = get_loader_for_split(dataloader_instance, args.split)

    class_names = dataloader_instance.classes
    num_classes = len(class_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    logger.info(f"Evaluating split: {args.split} ({len(eval_loader.dataset)} samples)")

    model, _ = load_model(str(ckpt_path), num_classes, device)
    misclassified = collect_misclassifications(model, eval_loader, device, class_names)

    logger.info(
        "Misclassified %d/%d samples",
        len(misclassified),
        len(eval_loader.dataset),
    )

    print(json.dumps(misclassified, indent=2))

    if args.output is not None:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(misclassified, f, indent=2)
        logger.info(f"Saved misclassifications to: {output_path}")


if __name__ == "__main__":
    main()
