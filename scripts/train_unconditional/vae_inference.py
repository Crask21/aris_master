import torch
from diffusers import DDPMPipeline, AutoencoderKL

model_dir = "/media/aris/Data/master2025dev/aris_master/training/ddim-ema-drink-can-128"

pipe = DDPMPipeline.from_pretrained(model_dir)
pipe = pipe.to("cuda")

vae_dir = "/media/aris/Data/master2025dev/aris_master/models/VAE/vae-ft-mse-840000-ema-pruned"
vae = AutoencoderKL.from_pretrained(vae_dir)
vae = vae.to("cuda")


# (Optional) speed + reduce memory on Ampere/Ada
pipe.unet.to(memory_format=torch.channels_last)

# Generate a batch of images
images = pipe(batch_size=16, num_inference_steps=50, output_type="latent").images
#permute images from (B, H, W, C) to (B, C, H, W)
images = torch.tensor(images).permute(0, 3, 1, 2).to("cuda")
print("Generated images shape:", images.shape)


# Decode with VAE
with torch.no_grad():
    images = vae.decode(images / vae.config.scaling_factor).sample

print("Decoded images shape:", images.shape)

# convert to PIL
from PIL import Image
images = (images / 2 + 0.5).clamp(0, 1).cpu().permute(0, 2, 3, 1).numpy()
images = (images * 255).round().astype("uint8")

#Rotate RGB channels
images = images[..., [0, 2, 1]]

pil_images = [Image.fromarray(image) for image in images]
images = pil_images

# print image shapes
for i, img in enumerate(images):
    print(f"Image {i} shape: {img.size}")
# Save a grid
from PIL import Image
import math

cols = 4
rows = math.ceil(len(images) / cols)
w, h = images[0].size
grid = Image.new("RGB", (cols * w, rows * h))

for i, img in enumerate(images):
    grid.paste(img, (i % cols * w, i // cols * h))

grid.save(model_dir + "/samples_grid.png")
print("Saved " + model_dir + "/samples_grid.png")