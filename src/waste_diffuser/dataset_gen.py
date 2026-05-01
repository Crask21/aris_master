
import argparse
import torch
from diffusers import UNet2DModel, AutoencoderKL, DDIMPipeline
# Add the project root to sys.path to allow imports from src/   
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.waste_diffuser.pipeline import Pipeline
from src.SDEdit.sdedit import SDEdit_gen_dataset
from src.testing.compute_fid import compute_fid
import json
import os
from PIL import Image
from datetime import datetime
def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using diffusion models")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="/media/aris/Data/master2025dev/aris_master/training/ddim-ema-impregnated-wood-128-2x_self_attention",
        help="Path to the trained model directory"
    )
    parser.add_argument(
        "--vae",
        action="store_true",
        help="Use VAE for decoding latents"
    )
    parser.add_argument(
        "--vae_dir",
        type=str,
        default="/media/aris/Data/master2025dev/aris_master/models/VAE/vae-ft-mse-840000-ema-pruned",
        help="Path to the VAE model directory (used when --vae is set)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Number of images to generate"
    )
    parser.add_argument(
        "--num_inference_steps",
        type=int,
        default=50,
        help="Number of denoising steps"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/media/aris/Data/master2025dev/datasets/synthetic/00-00-default_output_path",
        help="Output directory for the generated images"
    )
    parser.add_argument(
        "--image_num",
        type=int,
        default=1,
        help="Number of images to generate for each class"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to the model config file (used to extract class names if model is conditional)"
    )
    parser.add_argument(
        "--compute_fid",
        type=int,
        default=0,
        help="Number of times to compute FID. If > 1, prints mean and std.",
    )
    return parser.parse_args()

class Dataset_gen:
    #TODO: import the .config file and extract the class names for later use
    def __init__(self):
        # ----------------------------- Intialize config ----------------------------- #
        self.args = parse_args()
        # ISO-style timestamp for this run (safe for filenames)
        self.run_timestamp = datetime.utcnow().isoformat(timespec='milliseconds').replace(':','-')
        self.config = None
        if self.args.config is not None:
            with open(self.args.config, "r") as f:
                raw_config = json.load(f)
                self.config = raw_config["generate"]
            self.args.image_num = self.config.get("num_images_per_class", 1)
            self.args.batch_size = self.config["batch_size"]
            self.args.model_dir = self.config["model_dir"]
            self.args.output_dir = self.config["output_dir"]
            self.args.method = self.config["method"]
            self.classes = self.config["classes"]
            if self.args.method == "unconditional":
                assert len(self.classes) == 1, "For unconditional generation, there should be only one class specified in the config."
                self.args.output_dir = os.path.join(self.args.output_dir, self.classes[0])
        
        # ------------------------------- Generate data ------------------------------ #
        if self.args.method == "unconditional":
            print("Running unconditional generation...")
            self.classic_diffusion()
        elif self.args.method == "conditional":
            print("Running conditional generation...")
            self.classic_diffusion()
        elif self.args.method.lower() == "sdedit":
            print("Running SDEdit generation...")
            _class = self.config.get("class_label", 0)
            if type(_class) == str:
                # Index of _class on the classes list in the config file
                class_label = self.classes.index(_class)
                print(f"Using class label {class_label} for class {_class}")
            else:
                class_label = _class
                print(f"Using class label {class_label} from config file")
            SDEdit_gen_dataset(
                model_dir=self.args.model_dir,
                output_dir=self.args.output_dir,
                guide_image_folder=self.config["guide_image_folder"],
                synth_images_per_guide_image=self.config["synth_images_per_guide_image"],
                batch_size=self.args.batch_size,
                num_inference_steps=self.args.num_inference_steps,
                strength=self.config.get("strength", 0.9),
                class_label=class_label,
            )

        if self.args.method == "conditional" and len(self.classes) == 1:
            self.args.output_dir = os.path.join(self.args.output_dir, self.classes[0],"images")
            print(f"Only one class specified in config, setting output directory to {self.args.output_dir} for FID computation.")
        # -------------------------------- Compute FID ------------------------------- #
        if self.args.compute_fid or (self.config is not None and self.config.get("compute_fid", False)):
            
            real_path = self.config["real_image_path"] if self.config is not None else input("Enter path to real images for FID computation: ")
            # Look for fid_results*.json in the model directory to potentially reuse dataset hashes
            previous_results = None
            for filename in Path(self.args.model_dir).glob("fid_results_*.json"):
                previous_results = filename
                print(f"Found previous FID results JSON: {previous_results}. Will attempt to reuse dataset hashes if compatible.")
                break
            fid_dict = compute_fid(
                real=real_path,
                fake=self.args.output_dir,
                cuda=True,
                search_deep=True,
                batch_size=self.args.batch_size,
            )
            # Save the FID results to a JSON file in the model directory
            fid_results_path = os.path.join(self.args.model_dir, f"fid_results_{self.run_timestamp}.json")
            with open(fid_results_path, "w") as f:
                json.dump(fid_dict, f, indent=4)
            print(f"FID results saved to {fid_results_path}")



    def classic_diffusion(self):
        pipeline = Pipeline.from_pretrained(self.args.model_dir)
        self.pipeline = pipeline.to("cuda")
        self.pipeline.unet.eval()
        
        #Make sure output directory exists
        os.makedirs(self.args.output_dir, exist_ok=True)
        
        print(self.pipeline.unet.config)
        print("number of class embeds:", self.pipeline.unet.config.num_class_embeds)
        if self.pipeline.unet.config.num_class_embeds is None:
            print("Model is unconditional. Generating images without class labels.")
            self.generate_images_unconditional()
        else:
            print(f"Model is conditional with {self.pipeline.unet.config.num_class_embeds} class embeds. Generating images for each class.")
            self.generate_images_conditional()
            
                
    def generate_images_unconditional(self):
        #Check amount of images already in output directory
        existing_images = len([f for f in os.listdir(self.args.output_dir) if f.endswith(".png")])
        print(f"Found {existing_images} existing images in output directory.")
        if existing_images >= self.args.image_num:
            print(f"Already have {existing_images} images, which is >= requested {self.args.image_num}. Skipping generation.")
            return
        else:
            print(f"Generating {self.args.image_num - existing_images} new images...")
            n_images_to_generate = self.args.image_num - existing_images
            for i in range(n_images_to_generate//self.args.batch_size):
                print(f"Generating batch {i+1}/{n_images_to_generate//self.args.batch_size}..." )
                image = self.pipeline(num_inference_steps=self.args.num_inference_steps, batch_size=self.args.batch_size).images
                # Save images to output directory
                for j, img in enumerate(image):
                        # Save image with ISO timestamp prefix
                        img.save(f"{self.args.output_dir}/{self.run_timestamp}_{i*self.args.batch_size + j:04d}.png")
            
            # Handle remaining images if image_num is not divisible by batch_size
            remaining = n_images_to_generate % self.args.batch_size
            if remaining > 0:
                print(f"Generating remaining {remaining} images...")
                image = self.pipeline(num_inference_steps=self.args.num_inference_steps, batch_size=remaining).images
                for j, img in enumerate(image):
                    img.save(f"{self.args.output_dir}/{self.run_timestamp}_{(self.args.image_num - remaining) + j:04d}.png")
    
    def generate_images_conditional(self):
        for class_label in range(self.pipeline.unet.config.num_class_embeds):
            print(f"Generating images for class {class_label}...")
            #Check amount of images already in output directory
            if self.config is not None:
                output_dir_class = os.path.join(self.args.output_dir, self.classes[class_label],"images")
            else:
                output_dir_class = os.path.join(self.args.output_dir, f"class_{class_label}", "images")
            os.makedirs(output_dir_class, exist_ok=True)
            existing_images = len([f for f in os.listdir(output_dir_class) if f.endswith(".png")])
            print(f"Found {existing_images} existing images in directory {output_dir_class}.")
            if existing_images >= self.args.image_num:
                print(f"Already have {existing_images} images, which is >= requested {self.args.image_num}. Skipping generation.")
                continue
            else:
                print(f"Generating {self.args.image_num - existing_images} new images...")
                n_images_to_generate = self.args.image_num - existing_images
                for i in range(n_images_to_generate//self.args.batch_size):
                    print(f"Generating batch {i+1}/{n_images_to_generate//self.args.batch_size}..." )
                    image = self.pipeline(num_inference_steps=self.args.num_inference_steps, batch_size=self.args.batch_size, class_labels=torch.tensor([class_label]*self.args.batch_size).to("cuda")).images
                    # Save images to output directory
                    for j, img in enumerate(image):
                        # Image is a tensor, convert to PIL image before saving
                        image_png = (img / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                        image_png = (image_png * 255).round().astype("uint8")
                        image_png = Image.fromarray(image_png)
                        # Save image with ISO timestamp prefix
                        image_png.save(f"{output_dir_class}/{self.run_timestamp}_{i*self.args.batch_size + j:04d}.png")
                        #img.save(f"{output_dir_class}/{i*self.args.batch_size + j:04d}.png")
                
                # Handle remaining images if image_num is not divisible by batch_size
                remaining = n_images_to_generate % self.args.batch_size
                if remaining > 0:
                    print(f"Generating remaining {remaining} images...")
                    image = self.pipeline(num_inference_steps=self.args.num_inference_steps, batch_size=remaining, class_labels=torch.tensor([class_label]*remaining).to("cuda")).images
                    for j, img in enumerate(image):
                        image_png = (img / 2 + 0.5).clamp(0, 1).cpu().permute(1, 2, 0).numpy()
                        image_png = (image_png * 255).round().astype("uint8")
                        image_png = Image.fromarray(image_png)
                        # Save image with ISO timestamp prefix
                        image_png.save(f"{output_dir_class}/{self.run_timestamp}_{(self.args.image_num - remaining) + j:04d}.png")
                

if __name__ == "__main__":
    # Check if parse_args are empty
    if len(sys.argv) == 1:
        print("No arguments provided. Using default arguments for testing...")
        
        # Default arguments for testing
        sys.argv.extend([
            "--model_dir", "/media/aris/Data/master2025dev/aris_master/training/02-03_impregnated-wood_128_2x-self-attention",
            "--output_dir", "/media/aris/Data/master2025dev/datasets/synthetic/02-03_impregnated-wood_128_2x-self-attention",
            #"--vae",
            "--batch_size", "16",
            "--num_inference_steps", "50",
            "--image_num", "1000"
        ])
    dataset_gen = Dataset_gen()