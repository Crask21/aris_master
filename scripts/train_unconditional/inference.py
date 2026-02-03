import torch
from diffusers import DDIMPipeline

model_dir = "/media/aris/Data/master2025dev/aris_master/training/ddim-ema-impregnated-wood-128-2x_self_attention"

pipe = DDIMPipeline.from_pretrained(model_dir)
pipe = pipe.to("cuda")

# (Optional) speed + reduce memory on Ampere/Ada
pipe.unet.to(memory_format=torch.channels_last)

# Generate a batch of images
images = pipe(batch_size=16, num_inference_steps=50).images

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