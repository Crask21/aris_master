import torch
import argparse
import math
from PIL import Image
from diffusers import DDIMPipeline, DDPMPipeline, AutoencoderKL
import sys
import os

# ---------------------------------------------------------------------------- #
#                                   Functions                                  #
# ---------------------------------------------------------------------------- #
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
        "--output_name",
        type=str,
        default="samples_grid.png",
        help="Output filename for the grid"
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        help="Number of columns in the output grid"
    )
    parser.add_argument(
        "--image_num",
        type=int,
        default=1,
        help="Number of images to generate"
    )
    return parser.parse_args()

# ---------------------------------------------------------------------------- #
#                                     Main                                     #
# ---------------------------------------------------------------------------- #
def main():
    args = parse_args()
    
    # Load the diffusion model
    if args.vae:
        pipe = DDIMPipeline.from_pretrained(args.model_dir)
    else:
        pipe = DDIMPipeline.from_pretrained(args.model_dir)
    
    pipe = pipe.to("cuda")
    
    # (Optional) speed + reduce memory on Ampere/Ada
    pipe.unet.to(memory_format=torch.channels_last)
    
    for i in range(args.image_num):
        print(f"Generating image {i+1}/{args.image_num}..." )
    
        # Generate images
        # ------------------------------------ VAE ----------------------------------- #
        if args.vae:
            print(f"Generating {args.batch_size} images with VAE decoding...")
            
            # Load VAE
            vae = AutoencoderKL.from_pretrained(args.vae_dir)
            vae = vae.to("cuda")
            
            # Generate latents
            images = pipe(
                batch_size=args.batch_size,
                num_inference_steps=args.num_inference_steps,
                output_type="latent"
            ).images
            
            # Permute images from (B, H, W, C) to (B, C, H, W)
            images = torch.tensor(images).permute(0, 3, 1, 2).to("cuda")
            print("Generated latents shape:", images.shape)
            
            # Decode with VAE
            with torch.no_grad():
                images = vae.decode(images / vae.config.scaling_factor).sample
            
            print("Decoded images shape:", images.shape)
            
            # Convert to PIL
            images = (images / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
            images = (images * 255).round().astype("uint8")
            
            # Rotate RGB channels
            images = images[..., [0, 2, 1]]
            
            pil_images = [Image.fromarray(image) for image in images]
            images = pil_images
            
            # Print image shapes
            for i, img in enumerate(images):
                print(f"Image {i} shape: {img.size}")
        # ---------------------------------- NO VAE ---------------------------------- #
        else:
            print(f"Generating {args.batch_size} images...")
            images = pipe(
                batch_size=args.batch_size,
                num_inference_steps=args.num_inference_steps
            ).images
        
        # -------------------------------- Save images ------------------------------- #
        cols = args.cols
        rows = math.ceil(len(images) / cols)
        w, h = images[0].size
        grid = Image.new("RGB", (cols * w, rows * h))
        
        for i, img in enumerate(images):
            grid.paste(img, (i % cols * w, i // cols * h))
        
        # Append a number to the filename if it already exists
        if os.path.exists(f"{args.model_dir}/{args.output_name}"):
            base, ext = os.path.splitext(args.output_name)
            count = 1
            while os.path.exists(f"{args.model_dir}/{base}_{count}{ext}"):
                count += 1
            args.output_name = f"{base}_{count}{ext}"
        
        # Save the sample grid
        output_path = f"{args.model_dir}/{args.output_name}"
        grid.save(output_path)
        print(f"Saved {output_path}")
        
        # Add the image to the ## Sampling Results section in notes.md
        notes_path = f"{args.model_dir}/notes.md"
        if os.path.exists(notes_path):
            with open(notes_path, "r") as f:
                content = f.read()
            
            # Check if Sampling Results section exists
            if "## Sampling Results" in content:
                # Append image markdown to existing Sampling Results section
                content = content.replace(
                    "## Sampling Results",
                    f"## Sampling Results\n\n![{args.output_name}]({args.output_name})",
                    1
                )
            else:
                # Create Sampling Results section
                content += f"\n## Sampling Results\n\n![{args.output_name}]({args.output_name})\n"
            
            with open(notes_path, "w") as f:
                f.write(content)
            
            print(f"Updated {notes_path} with the generated image.")

# ---------------------------------------------------------------------------- #
#                               Manual execution                               #
# ---------------------------------------------------------------------------- #
if __name__ == "__main__":
    
    # Check if parse_args are empty
    if len(sys.argv) == 1:
        print("No arguments provided. Using default arguments for testing...")
        
        # Default arguments for testing
        sys.argv.extend([
            "--model_dir", "/media/aris/Data/master2025dev/aris_master/training/ddim-ema-normal-wood-128-2x_self_attention",
            #"--vae",
            "--batch_size", "16",
            "--num_inference_steps", "50",
            "--image_num", "1"
        ])
        
    main()