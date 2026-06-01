'''
DiffDA_pipeline.py
This file is used for the DiffDA pipeline. It is used to import different pipelines from the other DiffDA venv.

'''
import sys
import os
sys.path.append("/media/aris/Data/master2025dev/DiffDA-Eval/")
from augmentation import DiffMixGenerator
from data.wrapper import get_dataset
import omegaconf
from torchvision import transforms

class_LUT = []
class DiffDA_pipeline:
    def __init__(self, **kwargs):
        # -------------------------------- Parameters -------------------------------- #
        
        cfg_path = kwargs.get("cfg_path", "/media/aris/Data/master2025dev/DiffDA-Eval/configs/generation/aris/GEN_diff_mix.yaml")
        dataset_name = kwargs.get("dataset_name", "100_real")
        augmentation_ratio = kwargs.get("augmentation_ratio", 5)
        seed = kwargs.get("seed", 0)
        self.output_dir = kwargs.get("output_dir", "/media/aris/Data/master2025dev/aris_master/temp/")
        resolution = kwargs.get("resolution", 128)
        self.save_to_output_dir = kwargs.get("save_to_output_dir", False)
        
        # ---------------------------- Modification of cfg --------------------------- #
        cfg = omegaconf.OmegaConf.load(cfg_path)
        cfg.data.dataset = os.path.join(dataset_name, "real")
        cfg.generation.root_dir = self.output_dir
        cfg.generation.augmentation_ratio = augmentation_ratio
        dset, _ = get_dataset(cfg, split="train", return_pure_image=True)
        

        ti_embeds = cfg.generation.ti_embeds
        
        #TEMP
        ti_embeds += "_temp_renaiming"
        print(f"Looking for TI embeds in {ti_embeds}...")
        
        
        list_files = os.listdir(ti_embeds)
        file_name = [f for f in list_files if dataset_name in f and "seed" + str(seed) in f][0]
        cfg.generation.ti_embeds = os.path.join(ti_embeds, file_name)

        db_lora_weights = cfg.generation.db_lora_weights
        
        #TEMP
        db_lora_weights += "_temp_renaming"
        print(f"Looking for DB LoRA weights in {db_lora_weights}...")
        
        
        list_files = os.listdir(db_lora_weights)
        file_name = [f for f in list_files if dataset_name in f and "seed" + str(seed) in f][0]
        cfg.generation.db_lora_weights = os.path.join(db_lora_weights, file_name)
        
        # ---------------------------- Prepare output_dir ---------------------------- #
        if self.save_to_output_dir:
            os.makedirs(self.output_dir, exist_ok=True)
            for class_name in dset.class_names.values():
                print(f"Creating directory for class {class_name}...")
                os.makedirs(os.path.join(self.output_dir, class_name), exist_ok=True)
        self.num_existing_augmented_samples_per_class = {
            class_name: len(
                [
                    file_name
                    for file_name in os.listdir(os.path.join(self.output_dir, class_name))
                    if "augmented" in file_name
                ]
            )
            for class_name in dset.class_names.values()
        }
        
        
        # ---------------------------- Initialize pipeline --------------------------- #
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution)),
        ])
        self.gen_images = DiffMixGenerator(cfg, dset)
    def __call__(self, image):
        samples, grid_to_show, target_classes = self.gen_images.augment_one_sample(image)
        samples = [self.transform(sample) for sample in samples]
        
        if self.save_to_output_dir:
            self.save_samples(samples, target_classes)
        return samples, grid_to_show, target_classes
    def save_samples(self, samples, target_classes):
        for (sample, target_class) in zip(samples, target_classes):
            sample.save(os.path.join(self.output_dir, target_class, f"augmented_{self.num_existing_augmented_samples_per_class[target_class]}.png"))
            self.num_existing_augmented_samples_per_class[target_class] += 1
            

if __name__ == "__main__":
    output_dir = "/media/aris/Data/master2025dev/aris_master/temp/"
    pipeline = DiffDA_pipeline(save_to_output_dir=True, output_dir=output_dir)
    image = pipeline.gen_images.dset[0]
    samples, grid_to_show, target_classes = pipeline(image)
    
    import matplotlib.pyplot as plt
    plt.imshow(grid_to_show)
    plt.axis('off')
    plt.show()
        