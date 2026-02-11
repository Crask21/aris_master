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
    
from src.waste_diffuser.dataloader_interface import dataloaderInterface


# ---------------------------------------------------------------------------- #
#                                     Class                                    #
# ---------------------------------------------------------------------------- #
class ResNetDataloader(dataloaderInterface):
    def __init__(self, config):
        
        with open(config, 'r') as f:
            data_config = json.load(f)["data"]
        real_image_count = data_config["real_image_count"]
        synthetic_image_count = data_config["synthetic_image_count"]
        
        if real_image_count is not None and synthetic_image_count is not None:
            self.real_image_count = real_image_count
            self.synthetic_image_count = synthetic_image_count
            self.image_count = real_image_count + synthetic_image_count
        else:
            raise ValueError("Both real_image_count and synthetic_image_count must be specified in the config file. Please check the config file and specify both values.")
        
        
        super().__init__(config)
        
        

        # synthetic_train_split = self.generate_synth_split(split="train", image_count=synthetic_image_count)

        
        
    def generate_data_split(self, split="train", image_count=None):
        """Generate a data dictionary from the config file. The data dictionary will contain the filepaths, labels and split for each image in the dataset. The data split uses the real_image_count and synthetic_image_count from the config file to determine how many real and synthetic images to include in the data dictionary for each class. 
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


# ------------------------------ Get dataloader ------------------------------ #
    def get_dataloader(self, split="train"):
        # ----- Load data ----- #
        real_train_split = self.generate_data_split(split="train", image_count=self.real_image_count)
        val_split = self.generate_data_split(split="val")
        # synthetic_train_split = self.generate_synth_split(split="train", image_count=self.synthetic_image_count)
        
        data_dict = real_train_split + val_split # + synthetic_train_split
        self.data_json_path = os.path.join(self.output_dir, "data_file.json")
        
        # Dump data files to json file
        with open(self.data_json_path, 'w') as f:
            json.dump(data_dict, f, indent=4)
            print(f"[INFO] Data dictionary saved to {self.data_json_path}")
            
        # Load dataset from json file
        dataset = load_dataset("json", data_files=self.data_json_path)
        # Filter dataset into train and val splits
        self.train_ds = dataset["train"].filter(lambda x: x["split"] == "train")
        self.val_ds = dataset["train"].filter(lambda x: x["split"] == "val")
        
        print(f"[INFO] Dataset loaded from {self.config_path} with {len(self.train_ds)} training samples and {len(self.val_ds)} validation samples.")
        if split == "train":
            dataset = self.train_ds
        elif split == "val":
            dataset = self.val_ds
        else:
            raise ValueError("Invalid split. Must be 'train' or 'val'.")
        
        # --- Define augmentations --- #
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
                  
        dataset.set_transform(self.transform_images)
        self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        if self.preview == True:
            self.preview_dataloader(self.dataloader)
        return self.dataloader




# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":    # Load config
    config_path = "/media/aris/Data/master2025dev/aris_master/testing/config.json"
    print(f"Loading config from: {config_path}")
    
    dataloader = ResNetDataloader(config_path)