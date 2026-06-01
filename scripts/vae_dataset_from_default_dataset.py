
'''
Goal of this file: transform the dataset of images into a dataset of latents, which can be used for training the VAE.
'''

def main(resolution=128, dataset_dir="/media/aris/Data/master2025dev/datasets/wood/4_main_categories_512x512/train"):
    import torch
    from diffusers import AutoencoderKL
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import os

    # Load the pretrained VAE model
    vae_dir = "/media/aris/Data/master2025dev/aris_master/models/VAE/vae-ft-mse-840000-ema-pruned"
    vae = AutoencoderKL.from_pretrained(vae_dir)
    vae = vae.to("cuda")

    
    # Define transform for HuggingFace datasets (applied to dict with "image" key)
    def transform_batch(examples):
        transform_fn = transforms.Compose([
            transforms.Resize((resolution, resolution)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        examples["image"] = [transform_fn(img.convert("RGB")) for img in examples["image"]]
        return examples
    
    from datasets import load_dataset

    dataset = load_dataset("pcuenq/lsun-bedrooms", split="train")
    dataset.set_transform(transform_batch)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False
    )
    
    # Create output directory for latents
    latents_dir = os.path.join(dataset_dir, f"latents_{resolution}")
    os.makedirs(latents_dir, exist_ok=True)

    # Process the images in batches and save the latents using tqdm
    from tqdm import tqdm
    for i, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
        # Limit to 100000 images for training
        if i * dataloader.batch_size >= 100000:
            break
        # batch["image"] is a list of tensors, stack them into a batch
        images = batch["image"]
        if isinstance(images, list):
            images = torch.stack(images)
        images = images.to("cuda")
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
        
        # Save the latents as .pt files
        for j in range(latents.size(0)):
            img_idx = i * dataloader.batch_size + j
            original_filename = f"{img_idx:06d}"
            
            latent_path = os.path.join(latents_dir, f"{original_filename}.pt")
            torch.save(latents[j].cpu(), latent_path)
            
    
    #Quality check: load a random latent and decode it back to an image
    latent_files = os.listdir(latents_dir)
    if latent_files:
        random_latent_path = os.path.join(latents_dir, latent_files[0])
        latent = torch.load(random_latent_path).unsqueeze(0).to("cuda")
        with torch.no_grad():
            decoded_image = vae.decode(latent / vae.config.scaling_factor).sample
        decoded_image = (decoded_image.cpu().clamp(-1, 1) + 1) / 2  # Denormalize to [0, 1]
        print(f"Decoded image shape: {decoded_image.shape}")
        import matplotlib.pyplot as plt
        plt.imshow(decoded_image.squeeze().permute(1, 2, 0))
        plt.title("Decoded Image from Latent")
        plt.axis("off")
        plt.show()
            

if __name__ == "__main__":
    resolution = 256
    main(resolution, dataset_dir=f"/media/aris/Data/master2025dev/datasets/lsun/train")