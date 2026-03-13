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
logger = logging.getLogger(__name__)

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
                 synthetic: bool = False,
                 vae_latents: bool = None,
                 data_file: str = None):
        
        self.config_path = config
        self.output_dir = output_dir
        self.resolution = resolution
        self.center_crop = center_crop
        self.random_flip = random_flip
        self.preview = preview
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.synthetic = synthetic
        self.vae_latents = vae_latents
        self.data_file = data_file
        
        # Setup logger
        if not logger.hasHandlers():
            #"%(levelname)s - %(message)s"
            logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
        
        # Initialize config as json object
        with open(config, 'r') as f:            
            self.config = json.load(f)
            self.data_config = self.config["data"]
               
        # Set parameters from config file if not provided as arguments
        if output_dir is None:
            self.output_dir = self.config["logging"]["output_dir"]
        if batch_size is None:
            self.batch_size = self.config["hyperparameters"]["batch_size"]
        if num_workers is None:
            self.num_workers = self.config["hyperparameters"]["dataloader_num_workers"]
        # Check if vae
        if vae_latents is None:
            self.vae_latents = self.config["vae"]["use_vae"]  # Default to False if not specified in config
        
        # limit the number of images per class if max_images_per_class is specified in the config file
        self.max_images_per_class = self.data_config["max_images_per_class"]
            
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
        self.output_config_path = os.path.join(self.output_dir, "config.json")
        with open(self.output_config_path, 'w') as f:
            json.dump(self.config, f, indent=4)
        
        logger.info(f"Config file saved to {self.output_config_path}")
        
# ----------------- Generate data dictionary from config file ---------------- #
    def generate_data_split(self, split: str):
        """Generate a data dictionary from the config file. The data dictionary will contain the filepaths, labels and split for each image in the dataset.
        arguments:
            split: str: the split to generate the data dictionary for (train, val, test)
        returns:        
            data_dict: list: a list of dictionaries containing the filepaths, labels and split for each image in the dataset
        """
        
        data_dict = []
        
        
        for category, details in self.data_config["classes"].items():
            data_dir = details["data_dir"] + "/" + split
            # Assert that the data_dir directory exists
            assert Path(data_dir).exists(), f"data_dir directory does not exist. Please check the config file and the data_dir directory.\n   data_dir: {data_dir}\n  Config file: {self.config_path}"
            all_sub_categories = details["sub_categories"]
            
            
            for sub_category in all_sub_categories:
                # Assert that the sub-category exists
                if not Path(data_dir + "/" + sub_category).exists():
                    logger.warning(f"Sub-category {sub_category} does not exist in data_dir directory \nSkipping this sub-category. Please check the config file and the data_dir directory.\n  data_dir: {data_dir}\n  Config file: {self.config_path}")
                    continue
                
                # Find all .png files in the data_dir directory for the sub-category
                sub_category_path = os.path.join(data_dir, sub_category)
                
                # image count variable for limiting the number of images per class if max_images_per_class is specified in the config file
                image_count = 0
                # Look globally in the folder in recursive way for .png files
                for root, dirs, files in os.walk(sub_category_path):
                    if self.vae_latents == True:
                        if self.data_config["custom_dataset_name"] is not None:
                            if root.endswith(self.data_config["custom_dataset_name"]):
                                print(f"Found custom dataset folder: {root}")
                                for file in files:
                                    if file.endswith(".pt"):
                                        latent_file = os.path.join(root, file)
                                        sample = {"filepath": latent_file, "class": category, "split": split}
                                        data_dict.append(sample)
                                        image_count += 1
                                    if self.max_images_per_class is not None and image_count >= self.max_images_per_class:
                                        break
                                    
                        elif root.endswith(f"latents_{self.resolution}") and self.data_config["custom_dataset_name"] is None:
                            print(f"Found latents folder: {root}")
                            for file in files:
                                if file.endswith(".pt"):
                                    latent_file = os.path.join(root, file)
                                    sample = {"filepath": latent_file, "class": category, "split": split}
                                    data_dict.append(sample)
                                    image_count += 1
                                if self.max_images_per_class is not None and image_count >= self.max_images_per_class:
                                    break
                    else:
                        print(f"Looking for images in: {root}")
                        for file in files:
                            if file.endswith(".png"):
                                image_file = os.path.join(root, file)
                                sample = {"filepath": image_file, "class": category, "split": split}
                                data_dict.append(sample) 
                                image_count += 1
                            if self.max_images_per_class is not None and image_count >= self.max_images_per_class:
                                break
                                
                    if self.max_images_per_class is not None and image_count >= self.max_images_per_class:
                        logger.info(f"Reached max_images_per_class limit for class {category}. Stopping search for this class.")
                        break 
        
        print(f"Generated data dictionary for split '{split}' with {len(data_dict)} samples.")
        return data_dict


# ------------- Transform images using the defined augmentations ------------- #
    def transform_images(self, examples):
        processed = []
        for filepath in examples["filepath"]:
            # Import as image using PIL
            image = Image.open(filepath)
            processed.append(self.augmentations(image.convert("RGB")))
        class_indices = [self.class_LUT[cls] for cls in examples["class"]]
        return {"image": processed, "class": class_indices}
    
    def load_latent(self, examples):
        latents = []
        for filepath in examples["filepath"]:
            latent = torch.load(filepath)
            latents.append(latent)
        class_indices = [self.class_LUT[cls] for cls in examples["class"]]
        return {"image": latents, "class": class_indices}
    
# ------------------------------ Get dataloader ------------------------------ #
    def get_dataloader(self, split="train"):
        # ----- Load data ----- #
        if self.data_file is not None:
            logger.info(f"Loading dataset from provided data file: {self.data_file}")
            with open(self.data_file, 'r') as f:
                train_dict = json.load(f)
            train_dict = [sample for sample in train_dict if sample["split"] == "train"]
            logger.info(f"Loaded {len(train_dict)} training samples from data file.")
        else:
            train_dict = self.generate_data_split(split="train")
            
        # if self.max_images_per_class is not None:
        #     logger.info(f"Limiting to {self.max_images_per_class} images per class.")
        #     class_counts = {cls: 0 for cls in self.classes}
        #     limited_train_dict = []
        #     for sample in train_dict:
        #         cls = sample["class"]
        #         if class_counts[cls] < self.max_images_per_class:
        #             limited_train_dict.append(sample)
        #             class_counts[cls] += 1
        #     train_dict = limited_train_dict
        #     logger.info(f"After limiting, {len(train_dict)} training samples remain.")
                
        val_dict = self.generate_data_split(split="val")
        
        data_dict = train_dict + val_dict
        self.data_json_path = os.path.join(self.output_dir, "data_file.json")
        
        # Check if the data_json file exists
        if not os.path.exists(self.data_json_path):
            # If the file does not exist, create the folder path to it
            os.makedirs(os.path.dirname(self.data_json_path), exist_ok=True)
            
        
        # Dump data files to json file
        with open(self.data_json_path, 'w') as f:
            json.dump(data_dict, f, indent=4)
            logger.info(f"Data dictionary saved to {self.data_json_path}")
            
        # Load dataset from json file
        dataset = load_dataset("json", data_files=self.data_json_path)
        # Filter dataset into train and val splits
        self.train_ds = dataset["train"].filter(lambda x: x["split"] == "train")
        self.val_ds = dataset["train"].filter(lambda x: x["split"] == "val")
        
        logger.info(f"Dataset loaded from {self.config_path} with {len(self.train_ds)} training samples and {len(self.val_ds)} validation samples.")
        if split == "train":
            dataset = self.train_ds
        elif split == "val":
            dataset = self.val_ds
        else:
            raise ValueError("Invalid split. Must be 'train' or 'val'.")
        
        if not self.vae_latents:
            logger.info(f"Using image dataset with resolution {self.resolution} and augmentations: center_crop={self.center_crop}, random_flip={self.random_flip}")
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
                self.print_dataset_summary(data_dict)
        else:
            logger.info(f"Using VAE latent dataset with resolution {self.resolution}")
            # For VAE latents, we don't need to apply augmentations, but we still need to load the data and create a dataloader
            
            
            dataset.set_transform(self.load_latent)
            self.dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
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
        logger.info(f"Dataloader preview saved to {output_path}")
        plt.show()

# ------------------------- Dataset summary function ------------------------- #
    def _format_percentage(self, numerator, denominator):
        if denominator <= 0:
            return "0.00%"
        return f"{(numerator / denominator) * 100:.2f}%"

    def _log_markdown_table(self, headers, rows):
        string_rows = [[str(cell) for cell in row] for row in rows]
        widths = []
        for idx, header in enumerate(headers):
            max_row_width = max((len(row[idx]) for row in string_rows), default=0)
            widths.append(max(len(str(header)), max_row_width))

        def _format_row(cells):
            return "| " + " | ".join(str(cells[i]).ljust(widths[i]) for i in range(len(widths))) + " |"

        separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"

        logger.info(_format_row(headers))
        logger.info(separator)
        for row in string_rows:
            logger.info(_format_row(row))
        logger.info("")


    def print_dataset_summary(self, data_dict):
        """Print a formatted summary of the dataset with split and class distributions.
        
        Arguments:
            data_dict: list of dictionaries containing 'filepath', 'class', and 'split' keys
        """
        from collections import Counter

        # Count splits (map 'synth' to 'train' for the main tables)
        split_counts = Counter()
        class_split_counts = Counter()
        real_synth_counts = Counter()

        classes_in_data = set()
        for item in data_dict:
            cls = item['class']
            split = item['split']
            classes_in_data.add(cls)

            mapped_split = 'train' if split == 'synth' else split
            display_split = 'validation' if mapped_split == 'val' else mapped_split

            split_counts[display_split] += 1
            class_split_counts[(cls, display_split)] += 1

            if split in ['train', 'synth']:
                split_type = 'real' if split == 'train' else 'synthetic'
                real_synth_counts[(cls, split_type)] += 1

        total_images = sum(split_counts.values())

        logger.info("### Dataset summary:\n")

        # Split distribution table
        split_rows = []
        for split_name in sorted(split_counts.keys()):
            count = split_counts[split_name]
            split_rows.append([
                split_name,
                count,
                self._format_percentage(count, total_images),
            ])
        split_rows.append([
            'TOTAL',
            total_images,
            self._format_percentage(total_images, total_images),
        ])
        self._log_markdown_table(["Split", "Count", "Share of overall"], split_rows)

        # Class distribution for train/validation
        train_total = split_counts.get('train', 0)
        val_total = split_counts.get('validation', 0)

        class_rows = []
        for cls in sorted(classes_in_data):
            train_count = class_split_counts[(cls, 'train')]
            val_count = class_split_counts[(cls, 'validation')]
            class_rows.append([
                cls,
                train_count,
                self._format_percentage(train_count, train_total),
                val_count,
                self._format_percentage(val_count, val_total),
            ])

        class_rows.append([
            'TOTAL',
            train_total,
            self._format_percentage(train_total, train_total),
            val_total,
            self._format_percentage(val_total, val_total),
        ])

        self._log_markdown_table(
            ["Class", "Count (train)", "Share of train", "Count (validation)", "Share of validation"],
            class_rows,
        )

        # Real vs synthetic training split
        logger.info("#### Real vs. synthetic training split:\n")

        total_real = sum(real_synth_counts[(cls, 'real')] for cls in classes_in_data)
        total_synthetic = sum(real_synth_counts[(cls, 'synthetic')] for cls in classes_in_data)

        real_synth_rows = []
        for cls in sorted(classes_in_data):
            real_count = real_synth_counts[(cls, 'real')]
            synthetic_count = real_synth_counts[(cls, 'synthetic')]
            real_synth_rows.append([
                cls,
                real_count,
                self._format_percentage(real_count, total_real),
                synthetic_count,
                self._format_percentage(synthetic_count, total_synthetic),
            ])

        real_synth_rows.append([
            'TOTAL',
            total_real,
            self._format_percentage(total_real, total_real),
            total_synthetic,
            self._format_percentage(total_synthetic, total_synthetic),
        ])

        self._log_markdown_table(
            [
                "Class",
                "Count (train, real)",
                "Share of train (real)",
                "Count (train, synthetic)",
                "Share of train (synthetic)",
            ],
            real_synth_rows,
        )

    
    
# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
def main():
    print("Testing dataloader interface...")
    config_path = "/home/ap/cloud/Master/aris_master/queue/scheduled/resolution_224.json"
    dl_interface = dataloaderInterface(config = config_path)
    print("Dataloader interface test completed successfully.")
    
if __name__ == "__main__":      
    main()