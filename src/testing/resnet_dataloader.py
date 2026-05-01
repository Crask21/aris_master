# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import json
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from datasets import load_dataset
import PIL.Image as Image
from pathlib import Path
import logging
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

# Add scripts directory to path for imports
# scripts_path = str(Path(__file__).resolve().parent.parent.parent / "scripts")
# if scripts_path not in sys.path:
#     sys.path.insert(0, scripts_path)
from src.waste_diffuser.dataloader_interface import dataloaderInterface
from src.waste_diffuser.augmentations import build_image_augmentations
from src.summarize_training.generate_config_summary import generate_data_summary_from_config


# ---------------------------------------------------------------------------- #
#                                     Class                                    #
# ---------------------------------------------------------------------------- #
class ResNetDataloader(dataloaderInterface):
    def __init__(self, config_path, real_image_count=None, synthetic_image_count=None, run_number=None, **kwargs):
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        data_config = config["data"]
        # Extract "augmentation.use_data_augmentations_real" from the config file
        self.data_augmentations_real = config.get("augmentation", {}).get("use_data_augmentations_real", False)
        self.data_augmentations_synthetic = config.get("augmentation", {}).get("use_data_augmentations_synthetic", False)
        aug_cfg = config.get("augmentation", {})
        self.mixup_config = aug_cfg.get("mixup", {}) if isinstance(aug_cfg.get("mixup", {}), dict) else {}
        self.cutmix_config = aug_cfg.get("cutmix", {}) if isinstance(aug_cfg.get("cutmix", {}), dict) else {}
        print(f"[INFO] Data augmentations enabled for real images: {self.data_augmentations_real}")
        print(f"[INFO] Data augmentations enabled for synthetic images: {self.data_augmentations_synthetic}")
        self.real_image_count = data_config["real_image_count"]
        self.synthetic_image_count = data_config["synthetic_image_count"]
        self.prefetch_factor = config.get("hyperparameters", {}).get("dataloader_prefetch_factor", 6)
        
        if real_image_count is not None:
            self.real_image_count = real_image_count
        if synthetic_image_count is not None:
            self.synthetic_image_count = synthetic_image_count

        if not isinstance(self.prefetch_factor, int) or self.prefetch_factor < 1:
            print(f"[WARNING] Invalid dataloader_prefetch_factor={self.prefetch_factor}. Falling back to 6.")
            self.prefetch_factor = 6

        print(f"[INFO] Dataloader prefetch factor set to: {self.prefetch_factor}")
        
        # [Assertion] Assert that either real_image_count or synthetic_image_count is specified in the config file
        assert self.real_image_count is not None or self.synthetic_image_count is not None, \
            f"Either real_image_count or synthetic_image_count must be specified in the config file. Please check the config file and specify at least one of them.\n Config file: {Path(config_path).resolve()}"
        
        # Use deterministic seed: if run_number is provided, use it as seed
        # Otherwise fall back to random behavior (for backwards compatibility)
        if run_number is not None:
            self.seed = run_number
            print(f"[INFO] Using deterministic seed based on run_number: {self.seed}")
        else:
            self.seed = np.random.randint(0, 100000)
            print(f"[INFO] Random seed for this run: {self.seed}")
        
        super().__init__(config_path, **kwargs)
        
        

        # synthetic_train_split = self.generate_synth_split(split="train", image_count=synthetic_image_count)


    def set_training_augmentations(self):
        return super().set_training_augmentations()

    def _mixup_enabled(self) -> bool:
        return bool(self.mixup_config.get("enabled", False) and self.mixup_config.get("p", 0.0) > 0.0)

    def _cutmix_enabled(self) -> bool:
        return bool(self.cutmix_config.get("enabled", False) and self.cutmix_config.get("p", 0.0) > 0.0)

    @staticmethod
    def _sample_lambda(alpha: float, device: torch.device) -> float:
        if alpha <= 0.0:
            return 1.0
        dist = torch.distributions.Beta(alpha, alpha)
        return float(dist.sample().to(device=device).item())

    @staticmethod
    def _rand_bbox(size, lam):
        _, _, h, w = size
        cut_ratio = float(np.sqrt(1.0 - lam))
        cut_w = int(w * cut_ratio)
        cut_h = int(h * cut_ratio)

        cx = int(torch.randint(0, w, (1,)).item())
        cy = int(torch.randint(0, h, (1,)).item())

        x1 = np.clip(cx - cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        return x1, y1, x2, y2

    def _apply_mixup(self, inputs: torch.Tensor, labels: torch.Tensor):
        alpha = float(self.mixup_config.get("alpha", 0.2))
        lam = self._sample_lambda(alpha, inputs.device)
        index = torch.randperm(inputs.size(0), device=inputs.device)
        mixed_inputs = lam * inputs + (1.0 - lam) * inputs[index]
        labels_a, labels_b = labels, labels[index]
        return mixed_inputs, labels_a, labels_b, lam

    def _apply_cutmix(self, inputs: torch.Tensor, labels: torch.Tensor):
        alpha = float(self.cutmix_config.get("alpha", 1.0))
        lam = self._sample_lambda(alpha, inputs.device)
        index = torch.randperm(inputs.size(0), device=inputs.device)
        labels_a, labels_b = labels, labels[index]

        x1, y1, x2, y2 = self._rand_bbox(inputs.size(), lam)
        mixed_inputs = inputs.clone()
        mixed_inputs[:, :, y1:y2, x1:x2] = inputs[index, :, y1:y2, x1:x2]

        patch_area = float((x2 - x1) * (y2 - y1))
        total_area = float(inputs.size(-1) * inputs.size(-2))
        lam_adjusted = 1.0 - (patch_area / total_area if total_area > 0 else 0.0)
        return mixed_inputs, labels_a, labels_b, lam_adjusted

    def apply_batch_mixing(self, inputs: torch.Tensor, labels: torch.Tensor):
        if inputs.size(0) < 2:
            return inputs, labels, labels, 1.0, "none"

        cutmix_p = float(self.cutmix_config.get("p", 0.0)) if self._cutmix_enabled() else 0.0
        mixup_p = float(self.mixup_config.get("p", 0.0)) if self._mixup_enabled() else 0.0

        if cutmix_p <= 0.0 and mixup_p <= 0.0:
            return inputs, labels, labels, 1.0, "none"

        r = torch.rand(1).item()
        if cutmix_p > 0.0 and r < cutmix_p:
            mixed_inputs, labels_a, labels_b, lam = self._apply_cutmix(inputs, labels)
            return mixed_inputs, labels_a, labels_b, lam, "cutmix"

        if mixup_p > 0.0 and r < (cutmix_p + mixup_p):
            mixed_inputs, labels_a, labels_b, lam = self._apply_mixup(inputs, labels)
            return mixed_inputs, labels_a, labels_b, lam, "mixup"

        return inputs, labels, labels, 1.0, "none"

    @staticmethod
    def mixed_loss(criterion, outputs, labels_a, labels_b, lam):
        return lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)

    def _save_preview_grid(
        self,
        images: torch.Tensor,
        labels_a: torch.Tensor,
        labels_b: torch.Tensor | None,
        output_name: str,
        show: bool,
        lam: float | None = None,
    ) -> None:
        num_images = min(16, images.size(0))
        images = images[:num_images]
        labels_a = labels_a[:num_images]
        labels_b = labels_b[:num_images] if labels_b is not None else None

        # Convert from training range [-1, 1] to display range [0, 1].
        images = (images + 1.0) / 2.0
        images = torch.clamp(images, 0.0, 1.0)

        grid_size = int(np.ceil(np.sqrt(num_images)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = np.array(axes).flatten()

        for idx, ax in enumerate(axes):
            if idx < num_images:
                img = images[idx].detach().cpu().permute(1, 2, 0).numpy()
                ax.imshow(img)
                class_a = self.classes[int(labels_a[idx])]
                if labels_b is None:
                    ax.set_title(class_a)
                else:
                    class_b = self.classes[int(labels_b[idx])]
                    if lam is None:
                        ax.set_title(f"{class_a} | {class_b}")
                    else:
                        ax.set_title(f"{class_a} | {class_b} ({lam:.2f})")
                ax.axis("off")
            else:
                ax.axis("off")

        plt.tight_layout()
        output_path = os.path.join(self.output_dir, output_name)
        plt.savefig(output_path)
        print(f"[INFO] Preview saved: {output_path}")
        if show:
            plt.show()
        else:
            plt.close(fig)

    def preview_dataloader(self, dataloader, num_images=16, labels=True, show=True):
        super().preview_dataloader(dataloader, num_images=num_images, labels=labels, show=show)

        if not (self._mixup_enabled() or self._cutmix_enabled()):
            return

        batch = next(iter(dataloader))
        base_images = batch["image"]
        base_labels = batch["class"]

        if self._mixup_enabled():
            mixup_images, labels_a, labels_b, lam = self._apply_mixup(base_images.clone(), base_labels.clone())
            self._save_preview_grid(
                images=mixup_images,
                labels_a=labels_a,
                labels_b=labels_b,
                output_name="dataloader_preview_mixup.png",
                show=show,
                lam=lam,
            )

        if self._cutmix_enabled():
            cutmix_images, labels_a, labels_b, lam = self._apply_cutmix(base_images.clone(), base_labels.clone())
            self._save_preview_grid(
                images=cutmix_images,
                labels_a=labels_a,
                labels_b=labels_b,
                output_name="dataloader_preview_cutmix.png",
                show=show,
                lam=lam,
            )

# ----------------- Generate data dictionary from config file ---------------- #
    def generate_data_split(self, split, image_count=None):
        """Generate a data dictionary from the config file. The data dictionary will contain the filepaths, labels and split for each image in the dataset.
        arguments:
            split: str: the split to generate the data dictionary for (train, val, test)
        returns:        
            data_dict: list: a list of dictionaries containing the filepaths, labels and split for each image in the dataset
        """
        
            # --- Image count handling --- #
        if image_count is not None:
            print(f"[INFO] Generating data dictionary for split: {split} with image count: {image_count}")
            class_image_count = {}
            for class_name in self.classes:
                class_image_count[class_name] = image_count // len(self.classes)
            # Handle the case where image_count is not perfectly divisible by the number of classes
            remaining_images = image_count % len(self.classes)
            for i in range(remaining_images):
                class_name = self.classes[i % len(self.classes)]
                class_image_count[class_name] += 1
            print(f"[INFO] Class image count: {class_image_count}")
        
        data_dict = []
        # --- Generate data dictionary --- #
        for category, details in self.data_config["classes"].items():
            if split == "synth":
                data_dir = self.data_config["synthetic_data_dir"]
                all_sub_categories = [category]
                
                if data_dir is None:
                    print(f"[WARNING] synthetic_data_dir is not specified in the config file. Skipping synthetic data. Please check the config file and specify synthetic_data_dir.")
                    break
            else:
                data_dir = details["data_dir"] + "/" + split
                all_sub_categories = details["sub_categories"]
            
            # [Assertion] Assert that the data_dir directory exists
            assert Path(data_dir).exists(), f"data_dir directory does not exist. Please check the config file and the data_dir directory.\n   data_dir: {data_dir}\n  Config file: {self.config_path}"
            category_images = []
            for sub_category in all_sub_categories:
                # [Assertion] Assert that the sub-category exists
                if not Path(data_dir + "/" + sub_category).exists():
                    print(f"[ERROR] Sub-category {sub_category} does not exist in data_dir directory \nSkipping this sub-category. Please check the config file and the data_dir directory.\n  data_dir: {data_dir}\n  Config file: {self.config_path}")
                    sys.exit(1)

                # Find all .png files in the data_dir directory for the sub-category
                sub_category_path = os.path.join(data_dir, sub_category)
                for root, dirs, files in os.walk(sub_category_path):
                    for file in files:
                        if file.endswith(".png") or file.endswith(".JPEG") or file.endswith(".jpg") or file.endswith(".jpeg"):
                            
                            image_file = os.path.join(root, file)
                            sample = {"filepath": image_file, "class": category, "split": split}
                            category_images.append(sample)
            
            # --- Handle image count for the category --- #
            if image_count is not None:
                
                # Check if there are enough images for the category
                if len(category_images) < class_image_count[category]:
                    pass
                    #raise ValueError(f"Not enough images for category {category}. Required: {class_image_count[category]}, Available: {len(category_images)}. Please check the config file and the data_dir directory.")
                # Shuffle the data dictionary for the category
                np.random.shuffle(category_images)
                # Keep only the specified number of images for the category
                category_images = category_images[:class_image_count[category]]
                
            data_dict.extend(category_images)
            
        return data_dict
    
# --------------------------- Define augmentations --------------------------- #
    def training_augmentations(self, data_augmentations=False, data_type = None):
            # Preprocessing the datasets and DataLoaders creation.
        if data_type is not None:
            if data_type == "real":
                print(f"[INFO] Setting up real image training augmentations.")
            elif data_type == "synth":
                print(f"[INFO] Setting up synthetic image training augmentations.")
            else:
                print(f"[INFO] Setting up training augmentations.")
        else:
                print(f"[INFO] Setting up training augmentations. (No data_type given)")


        if data_augmentations:

            print("[INFO] Classic data augmentations is used")
            augmentations = build_image_augmentations(
                config=self.config
            )
        else:
            print("[INFO] No data augmentations is used")

            augmentations = self.base_augmentations() 
        print(f"[INFO] Training augmentations set: {augmentations}")
        return augmentations
    
# -------------------- Base transform (for synthetic data) ------------------- #
    def base_augmentations(self):
        """Minimal transform for synthetic data: resize + center crop + normalize.
        No random augmentations (flips, random crops, color jitter, etc.)."""
        self.base_aug = transforms.Compose([
            #transforms.CenterCrop(1200) if self.center_crop else transforms.Lambda(lambda x: x),
            #transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip() if self.random_flip else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        return self.base_aug
    
# ------------------------- Validation augmentations ------------------------- #
    def val_augmentations(self):
            # Preprocessing the datasets and DataLoaders creation.
        spatial_augmentations = [
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution),
        ]

        augmentations = transforms.Compose(
            spatial_augmentations
            + [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        print(f"[INFO] Validation augmentations set: {augmentations}")
        return augmentations

# ------------- Transform images using the defined augmentations ------------- #
    def train_transform(self, examples):
        processed = []
        for filepath, split in zip(examples["filepath"], examples["split"]):
            # Import as image using PIL
                # print(f"[INFO] Loading image from disk: {filepath}")
            image = Image.open(filepath).convert("RGB")
            if split == "synth":
                # Synthetic images: only resize + center crop + normalize (no random augmentations)
                processed.append(self.train_synthetic_aug(image))
            else:
                # Real images: full training augmentations
                processed.append(self.train_aug(image))
        class_indices = [self.class_LUT[cls] for cls in examples["class"]]
        return {"image": processed, "class": class_indices}
    
# ----------------------- Validation transform function ---------------------- #
    def val_transform(self, examples):
        processed = []
        for filepath in examples["filepath"]:
            # Import as image using PIL
            image = Image.open(filepath)
            processed.append(self.val_aug(image.convert("RGB")))
        class_indices = [self.class_LUT[cls] for cls in examples["class"]]
        return {"image": processed, "class": class_indices}
    
# ------------------------------ Get dataloader ------------------------------ #
    def get_dataloader(self, split="train"):
        # ----- Load data ----- #
        
        if self.data_file is not None:
            print(f"[INFO] Loading dataset from provided data file: {self.data_file}")
            with open(self.data_file, 'r') as f:
                real_train_split = json.load(f)
            # Keep only the images from the 'train' split
            real_train_split = [sample for sample in real_train_split if sample["split"] == "train"]
        else:
            real_train_split = self.generate_data_split(split="train", image_count=self.real_image_count)
        synthetic_train_split = self.generate_data_split(split="synth", image_count=self.synthetic_image_count)
        val_split = self.generate_data_split(split="val")
        metrics = self.config["evaluation"].get("metrics", [])
        # Check if test is within any of the metric entries
        test_split = []
        use_test_split = False
        print(f"[INFO] Evaluation metrics specified in config: {metrics}")
        for metric in metrics:
            if "test" in metric:
                print(f"[INFO] Test split will be generated because 'test' is found in the evaluation metrics: {metrics}")
                use_test_split = True
                break
        if use_test_split:
            test_split = self.generate_data_split(split="test")
        
        data_dict = real_train_split + val_split + synthetic_train_split + test_split

        generate_data_summary_from_config(config_path=self.config_path, data_dict=data_dict) # Generate notes.md summary of the dataset based on the config file and the generated data dictionary
        # Print dataset summary
        self.print_dataset_summary(data_dict)
        self.data_json_path = os.path.join(self.output_dir, "data_file.json")
        
        # Dump data files to json file
        with open(self.data_json_path, 'w') as f:
            json.dump(data_dict, f, indent=4)
            print(f"[INFO] Data dictionary saved to {self.data_json_path}")
            
        # Load dataset from json file
        dataset = load_dataset("json", data_files=self.data_json_path)
        # Filter dataset into train and val splits
        train_dataset = dataset["train"].filter(lambda x: x["split"] == "train" or x["split"] == "synth")
        val_dataset = dataset["train"].filter(lambda x: x["split"] == "val")
        if use_test_split:
            test_dataset = dataset["train"].filter(lambda x: x["split"] == "test")
        
        print(f"[INFO] Dataset loaded from {self.config_path} with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")


        
        # --- Define augmentations --- #
        self.val_aug = self.val_augmentations()
        self.train_aug = self.training_augmentations(data_augmentations=self.data_augmentations_real, data_type = "real")
        self.train_synthetic_aug = self.training_augmentations(data_augmentations=self.data_augmentations_synthetic, data_type="synth")
        
        train_dataset.set_transform(self.train_transform)
        val_dataset.set_transform(self.val_transform)
        if use_test_split:
            test_dataset.set_transform(self.val_transform)

        
        train_loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "generator": torch.Generator().manual_seed(self.seed),
            "persistent_workers": self.num_workers > 0,
        }
        val_loader_kwargs = {
            "batch_size": self.batch_size,
            "shuffle": True,
            "num_workers": self.num_workers,
            "generator": torch.Generator().manual_seed(self.seed),
            "persistent_workers": self.num_workers > 0,
        }
        if self.num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = self.prefetch_factor
            val_loader_kwargs["prefetch_factor"] = self.prefetch_factor
        
        self.train_loader = torch.utils.data.DataLoader(train_dataset, **train_loader_kwargs)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, **val_loader_kwargs)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, **val_loader_kwargs) if use_test_split else None
        
        self.preview_dataloader(self.train_loader, show=self.preview)
            
        return self.train_loader


# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":    # Load config
    config_path = "/home/ap/cloud/Master/aris_master/queue/scheduled/TEST_DATA_AUGMENTATION.json"
    print(f"Loading config from: {config_path}")
    
    dataloader = ResNetDataloader(config_path=config_path, preview=True, real_image_count=1000, synthetic_image_count=1000)