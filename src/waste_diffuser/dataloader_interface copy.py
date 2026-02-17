# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from datasets import load_dataset
import PIL.Image as Image
from pathlib import Path
import logging      
    

# ---------------------------------------------------------------------------- #
#                                     Class                                    #
# ---------------------------------------------------------------------------- #
class dataloaderInterface:
    # ----------------------------------- Init ----------------------------------- #
    def __init__(self, config: str, 
                 output_dir: str = None, 
                 resolution: int = None, 
                 center_crop: bool = None, 
                 random_flip: bool = None, 
                 preview: bool = True, 
                 batch_size: int = None, 
                 num_workers: int = None,
                 use_synthetic: bool = False):
        
        self.config_path = config
        self.output_dir = output_dir
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.preview = preview
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.use_synthetic = use_synthetic
                
        # Initialize config as json object
        with open(config, 'r') as f:            
            config = json.load(f)
            self.data_config = config["data"]
               
        # Set parameters from config file if not provided as arguments
        if output_dir is None:
            self.output_dir = config["logging"]["output_dir"]
        if batch_size is None:
            self.batch_size = config["hyperparameters"]["batch_size"]
        if num_workers is None:
            self.num_workers = config["hyperparameters"]["dataloader_num_workers"]
            
        # Set classes and number of classes from config file
        self.classes = list(self.data_config["classes"].keys())
        self.class_LUT = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)
        
            

        # ----- Augmentations ----- #
        # Check if augmentations args are provided, if not, use the ones from the config file
        if self.resolution is None:
            self.resolution = self.data_config["resolution"]
        if self.center_crop is None:
            self.center_crop = self.data_config["center_crop"]
        if self.random_flip is None:
            self.random_flip = self.data_config["random_flip"]

        self.dataloader = self.get_dataloader()
        
        
        # Save config file to output directory
        output_config_path = os.path.join(self.output_dir, "config.json")
        with open(output_config_path, 'w') as f:
            json.dump(config, f, indent=4)
        print(f"[INFO] Config file saved to {output_config_path}")
        
# ----------------- Generate data dictionary from config file ---------------- #
    def generate_data_split(self, split,image_count=None):
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
                
            else:
                data_dir = details["data_dir"] + "/" + split
                all_sub_categories = details["sub_categories"]
            
            # [Assertion] Assert that the data_dir directory exists
            assert Path(data_dir).exists(), f"data_dir directory does not exist. Please check the config file and the data_dir directory.\n   data_dir: {data_dir}\n  Config file: {self.config_path}"
            category_images = []
            for sub_category in all_sub_categories:
                # [Assertion] Assert that the sub-category exists
                if not Path(data_dir + "/" + sub_category).exists():
                    print(f"[WARNING] Sub-category {sub_category} does not exist in data_dir directory \nSkipping this sub-category. Please check the config file and the data_dir directory.\n  data_dir: {data_dir}\n  Config file: {self.config_path}")
                    continue
                
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
                    raise ValueError(f"Not enough images for category {category}. Required: {class_image_count[category]}, Available: {len(category_images)}. Please check the config file and the data_dir directory.")
                # Shuffle the data dictionary for the category
                np.random.shuffle(category_images)
                # Keep only the specified number of images for the category
                category_images = category_images[:class_image_count[category]]
                
            data_dict.extend(category_images)
            
        return data_dict


# --------------------------- Define augmentations --------------------------- #
    def training_augmentations(self):
            # Preprocessing the datasets and DataLoaders creation.
        spatial_augmentations = [
            transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(self.resolution) if self.center_crop else transforms.RandomCrop(self.resolution),
            transforms.RandomHorizontalFlip() if self.random_flip else transforms.Lambda(lambda x: x),
        ]

        self.augmentations = transforms.Compose(
            spatial_augmentations
            + [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        return self.augmentations
    

# ------------- Transform images using the defined augmentations ------------- #
    def train_transform(self, examples):
        processed = []
        for filepath in examples["filepath"]:
            # Import as image using PIL
            image = Image.open(filepath)
            processed.append(self.train_aug(image.convert("RGB")))
        class_indices = [self.class_LUT[cls] for cls in examples["class"]]
        return {"image": processed, "class": class_indices}
    

# ------------------------------ Get dataloader ------------------------------ #
    def get_dataloader(self):
        # ----- Load data ----- #
        real_train_split = self.generate_data_split(split="train")
        real_val_split = self.generate_data_split(split="val")
        data_dict = real_train_split + real_val_split
        
        # Print dataset summary
        self.print_dataset_summary(data_dict)
        
        
        # Dump data files to json file
        self.data_json_path = os.path.join(self.output_dir, "data_file.json")
        with open(self.data_json_path, 'w') as f:
            json.dump(data_dict, f, indent=4)
            print(f"[INFO] Data dictionary saved to {self.data_json_path}")
            
        # Load dataset from json file
        dataset = load_dataset("json", data_files=self.data_json_path)
        # Filter dataset into train and val splits
        train_dataset = dataset["train"].filter(lambda x: x["split"] == "train") 
        
        print(f"[INFO] Dataset loaded from {self.config_path} with {len(train_dataset)} training samples.")

        
        # --- Define augmentations --- #
        self.train_aug = self.training_augmentations()
                  
        train_dataset.set_transform(self.train_transform)
        self.dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers) 
        
        if self.preview == True:
            self.preview_dataloader(self.dataloader)
        return self.dataloader
        
    
# ---------------------------- Preview dataloader ---------------------------- #
    def preview_dataloader(self, dataloader, num_images=16, labels=True):
        """
        Preview images from the dataloader in a grid.
        
        Args:
            dataloader: PyTorch DataLoader object
            num_images: Number of images to display (default: 8)
        """
        # Get a batch from the dataloader
        batch = next(iter(dataloader))
        images = batch["image"]
        class_labels = batch["class"]
        class_labels = [self.classes[idx] for idx in class_labels]
        
        # Limit to num_images
        images = images[:num_images]
        class_labels = class_labels[:num_images]
        
        # Denormalize images from [-1, 1] to [0, 1]
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        
        # Create grid
        grid_size = int(np.ceil(np.sqrt(num_images)))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = axes.flatten()
        
        for idx, ax in enumerate(axes):
            if idx < len(images):
                # Convert to numpy and transpose from CxHxW to HxWxC
                img = images[idx].permute(1, 2, 0).numpy()
                ax.imshow(img)
                if labels:
                    ax.set_title(class_labels[idx])
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.tight_layout()
        # Save figure to output directory
        output_path = os.path.join(self.output_dir, "dataloader_preview.png")
        plt.savefig(output_path)
        print(f"[INFO] Dataloader preview saved to {output_path}")
        plt.show()


# ------------------------- Dataset summary function ------------------------- #
    def print_dataset_summary(self, data_dict):
        """Print a formatted summary of the dataset with split and class distributions.
        
        Arguments:
            data_dict: list of dictionaries containing 'filepath', 'class', and 'split' keys
        """
        from collections import Counter
        
        # Count splits (map 'synth' to 'train' for main split table)
        split_counts = Counter()
        for item in data_dict:
            split = item['split']
            if split == 'synth':
                split_counts['train'] += 1
            else:
                split_counts[split] += 1
        
        # Map 'val' to 'validation' for display
        display_split_counts = {}
        for split, count in split_counts.items():
            display_name = 'validation' if split == 'val' else split
            display_split_counts[display_name] = count
        
        total_images = sum(display_split_counts.values())
        
        print("### Dataset summary:\n")
        
        # Split distribution table
        print("| Split | Count | Share of overall |")
        print("|-------|-------|------------------|")
        for split in sorted(display_split_counts.keys()):
            count = display_split_counts[split]
            percentage = (count / total_images) * 100
            print(f"| {split:<20} | {count:<20} | {percentage:.2f}%{' ' * 14} |")
        print(f"| {'TOTAL':<20} | {total_images:<20} | {100.00:.2f}%{' ' * 14} |")
        print()
        
        # Class distribution per split
        class_split_counts = {}
        for item in data_dict:
            cls = item['class']
            split = item['split']
            # Map synth to train for this table
            if split == 'synth':
                split = 'train'
            elif split == 'val':
                split = 'validation'
            
            if cls not in class_split_counts:
                class_split_counts[cls] = {}
            if split not in class_split_counts[cls]:
                class_split_counts[cls][split] = 0
            class_split_counts[cls][split] += 1
        
        print("| Class             | Count (train) | Share of train | Count (validation) | Share of validation |")
        print("|------------------|---------------|----------------|--------------------|----------------------|")
        
        train_total = display_split_counts.get('train', 0)
        val_total = display_split_counts.get('validation', 0)
        
        for cls in sorted(class_split_counts.keys()):
            train_count = class_split_counts[cls].get('train', 0)
            val_count = class_split_counts[cls].get('validation', 0)
            train_pct = (train_count / train_total * 100) if train_total > 0 else 0
            val_pct = (val_count / val_total * 100) if val_total > 0 else 0
            
            print(f"| {cls:<20} | {train_count:<20} | {train_pct:.2f}%{' ' * 14} | {val_count:<20} | {val_pct:.2f}%{' ' * 14} |")
        
        print(f"| {'TOTAL':<20} | {train_total:<20} | {100.00:.2f}%{' ' * 14} | {val_total:<20} | {100.00:.2f}%{' ' * 14} |")
        print()
        
        if self.use_synthetic:
            # Real vs. synthetic training split
            print("#### Real vs. synthetic training split:\n")
            print("| Class             | Count (train, real) | Share of train (real) | Count (train, synthetic) | Share of train (synthetic) |")
            print("|------------------|---------------------|-----------------------|--------------------------|----------------------------|")
            
            real_synth_counts = {}
            for item in data_dict:
                if item['split'] in ['train', 'synth']:
                    cls = item['class']
                    split_type = 'real' if item['split'] == 'train' else 'synthetic'
                    
                    if cls not in real_synth_counts:
                        real_synth_counts[cls] = {'real': 0, 'synthetic': 0}
                    real_synth_counts[cls][split_type] += 1
            
            total_real = 0
            total_synthetic = 0
            
            for cls in sorted(real_synth_counts.keys()):
                real_count = real_synth_counts[cls]['real']
                synth_count = real_synth_counts[cls]['synthetic']
                total_real += real_count
                total_synthetic += synth_count
                
                class_total = real_count + synth_count
                real_pct = (real_count / class_total * 100) if class_total > 0 else 0
                synth_pct = (synth_count / class_total * 100) if class_total > 0 else 0
                
                print(f"| {cls:<20} | {real_count:<20} | {real_pct:.2f}%{' ' * 14} | {synth_count:<24} | {synth_pct:.2f}%{' ' * 14} |")
            
            train_total_all = total_real + total_synthetic
            real_pct_total = (total_real / train_total_all * 100) if train_total_all > 0 else 0
            synth_pct_total = (total_synthetic / train_total_all * 100) if train_total_all > 0 else 0
            
            print(f"| {'TOTAL':<20} | {total_real:<20} | {real_pct_total:.2f}%{' ' * 14} | {total_synthetic:<24} | {synth_pct_total:.2f}%{' ' * 14} |")
            print()

    
    
# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
def main():        
    print("Testing dataloader interface...")
    config_path = "/media/aris/Data/master2025dev/aris_master/training/template/config_org.json"
    dl_interface = dataloaderInterface(config = config_path)
    print("Dataloader interface test completed successfully.")
    
if __name__ == "__main__":      
    main()