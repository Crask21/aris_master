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
    def __init__(self, config_path,real_image_count=None, synthetic_image_count=None, **kwargs):
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        data_config = config["data"]
        # Extract "augmentation.use_data_augmentations_real" from the config file
        self.data_augmentations_real = config.get("augmentation", {}).get("use_data_augmentations_real", False)
        self.data_augmentations_synthetic = config.get("augmentation", {}).get("use_data_augmentations_synthetic", False)
        print(f"[INFO] Data augmentations enabled for real images: {self.data_augmentations_real}")
        print(f"[INFO] Data augmentations enabled for synthetic images: {self.data_augmentations_synthetic}")
        self.real_image_count = data_config["real_image_count"]
        self.synthetic_image_count = data_config["synthetic_image_count"]
        
        if real_image_count is not None:
            self.real_image_count = real_image_count
        if synthetic_image_count is not None:
            self.synthetic_image_count = synthetic_image_count
        
        
        # [Assertion] Assert that either real_image_count or synthetic_image_count is specified in the config file
        assert self.real_image_count is not None or self.synthetic_image_count is not None, \
            f"Either real_image_count or synthetic_image_count must be specified in the config file. Please check the config file and specify at least one of them.\n Config file: {Path(config_path).resolve()}"
        
        # Make a random seed each run
        self.seed = np.random.randint(0, 100000)
        print(f"[INFO] Random seed for this run: {self.seed}")
        
        super().__init__(config_path, **kwargs)
        
        

        # synthetic_train_split = self.generate_synth_split(split="train", image_count=synthetic_image_count)


    def set_training_augmentations(self):
        return super().set_training_augmentations()

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
                        if file.endswith(".png"):
                            
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
    def training_augmentations(self, data_augmentations=False):
            # Preprocessing the datasets and DataLoaders creation.
        print(f"[INFO] Setting up training augmentations. Using data augmentations: {data_augmentations}")
        if data_augmentations:

            augmentations = build_image_augmentations(
                config=self.config
            )
        else:

            augmentations = self.base_augmentations() 
        print(f"[INFO] Training augmentations set: {augmentations}")
        return augmentations
    
# -------------------- Base transform (for synthetic data) ------------------- #
    def base_augmentations(self):
        """Minimal transform for synthetic data: resize + center crop + normalize.
        No random augmentations (flips, random crops, color jitter, etc.)."""
        self.base_aug = transforms.Compose([
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.resolution) if self.center_crop else transforms.Lambda(lambda x: x),
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
        
        data_dict = real_train_split + val_split + synthetic_train_split

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
        
        print(f"[INFO] Dataset loaded from {self.config_path} with {len(train_dataset)} training samples and {len(val_dataset)} validation samples.")


        
        # --- Define augmentations --- #
        self.val_aug = self.val_augmentations()
        self.train_aug = self.training_augmentations(data_augmentations=self.data_augmentations_real)
        self.train_synthetic_aug = self.training_augmentations(data_augmentations=self.data_augmentations_synthetic)
                  
        train_dataset.set_transform(self.train_transform)
        val_dataset.set_transform(self.val_transform)
        
        self.train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, generator=torch.Generator().manual_seed(self.seed)) 
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,generator=torch.Generator().manual_seed(self.seed)) 
        
        
        self.preview_dataloader(self.train_loader, show=self.preview)
            
        return self.train_loader


# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":    # Load config
    config_path = "/home/ap/cloud/Master/aris_master/queue/scheduled/TEST_DATA_AUGMENTATION.json"
    print(f"Loading config from: {config_path}")
    
    dataloader = ResNetDataloader(config_path=config_path, preview=True, real_image_count=1000, synthetic_image_count=1000)