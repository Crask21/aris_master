

import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class _RandomMaskDataset(Dataset):
    def __init__(self, mask_files, transform=None):
        self.mask_files = list(mask_files)
        self.transform = transform

    def __len__(self):
        return len(self.mask_files)

    def _load_mask_file(self, file_path: Path):
        suffix = file_path.suffix.lower()

        if suffix == ".pt":
            mask = torch.load(file_path)
            if isinstance(mask, torch.Tensor):
                if mask.ndim == 3 and mask.shape[0] == 1:
                    mask = mask.squeeze(0)
                return mask.float()
            raise TypeError(f"Unsupported .pt mask type in {file_path}: {type(mask)}")

        if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
            image = Image.open(file_path).convert("L")
            mask_np = np.array(image, dtype=np.float32)
            return torch.from_numpy(mask_np)

        if suffix == ".npy":
            mask_np = np.load(file_path)
            return torch.from_numpy(mask_np).float()

        raise ValueError(f"Unsupported mask file extension: {file_path}")

    def __getitem__(self, idx):
        file_path = self.mask_files[idx]
        mask = self._load_mask_file(file_path)

        # Convert to binary [0, 1] if data appears to be in [0, 255] or logits.
        if mask.max() > 1:
            mask = (mask > 127).float()
        else:
            mask = (mask > 0.5).float()

        if mask.ndim == 2:
            mask = mask.unsqueeze(0)

        if self.transform is not None:
            mask = self.transform(mask)

        return {
            "mask": mask,
            "mask_path": str(file_path),
        }


class MaskDataLoader:
    def __init__(
        self,
        dataset_path="/media/aris/Data/master2025dev/datasets/aris_4_class/500_real/real/train",
        classes=None,
        resolution=128,
        batch_size=16,
        num_workers=0,
        shuffle=True,
    ):
        self.dataset_path = Path(dataset_path)
        self.classes = classes
        self.resolution = resolution
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset path does not exist: {self.dataset_path}")

        self.mask_files = self._discover_mask_files()
        if len(self.mask_files) == 0:
            raise FileNotFoundError(
                f"No mask files found under {self.dataset_path}. Expected files in 'masks' folders with extensions "
                "(.pt, .png, .jpg, .jpeg, .bmp, .tif, .tiff, .npy)."
            )

        self.mask_transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.resolution, self.resolution),
                    interpolation=transforms.InterpolationMode.NEAREST,
                    antialias=False,
                )
            ]
        )

        self.dataset = _RandomMaskDataset(self.mask_files, transform=self.mask_transform)
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
        )

    def _discover_mask_files(self):
        supported_ext = {".pt", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"}
        candidates = []

        for class_dir in sorted(self.dataset_path.iterdir()):
            if not class_dir.is_dir():
                continue

            if self.classes is not None and class_dir.name not in self.classes:
                continue

            for root, _, files in __import__("os").walk(class_dir):
                root_path = Path(root)
                if root_path.name != "masks":
                    continue

                for file_name in files:
                    file_path = root_path / file_name
                    if file_path.suffix.lower() in supported_ext:
                        candidates.append(file_path)

        return candidates

    def get_random_mask(self):
        random_file = random.choice(self.mask_files)
        sample = _RandomMaskDataset([random_file], transform=self.mask_transform)[0]
        return sample

    def get_dataloader(self):
        return self.dataloader
    

if __name__ == "__main__":
    dataloader = MaskDataLoader(
        dataset_path="/media/aris/Data/master2025dev/datasets/aris_4_class/500_real/real/train",
        classes=None,
        resolution=128,
        batch_size=16,
        num_workers=0,
        shuffle=True,
    ).get_dataloader()

    #plot an array of 4 masks
    import matplotlib.pyplot as plt
    for batch in dataloader:
        masks = batch["mask"]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        for i in range(4):
            axes[i].imshow(masks[i].squeeze(0), cmap="gray")
            axes[i].axis("off")
        plt.show()
        break