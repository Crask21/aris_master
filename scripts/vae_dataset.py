
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

    
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])
    
    dataset = datasets.ImageFolder(dataset_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    
        

    # Process the images in batches and save the latents
    for i, (images, labels) in enumerate(dataloader):
        images = images.to("cuda")
        with torch.no_grad():
            latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
        
        # Save the latents as .pt files with original filenames in class folders
        for j in range(latents.size(0)):
            # Get the original file path and class name
            img_idx = i * dataloader.batch_size + j
            original_path, class_idx = dataset.imgs[img_idx]
            class_name = dataset.classes[class_idx]
            original_filename = os.path.splitext(os.path.basename(original_path))[0]
            
            # Create class folder if it doesn't exist
            class_latents_dir = os.path.join(dataset_dir, class_name, f"latents_{resolution}")
            os.makedirs(class_latents_dir, exist_ok=True)
            
            latent_path = os.path.join(class_latents_dir, f"{original_filename}.pt")
            torch.save(latents[j].cpu(), latent_path)
            print(f"Saved {latent_path}")
    
    #Quality check: load a random latent and decode it back to an image
    random_latent_path = os.path.join(dataset_dir, dataset.classes[0], f"latents_{resolution}", os.listdir(os.path.join(dataset_dir, dataset.classes[0], f"latents_{resolution}"))[0])
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
    resolution = 512
    main(resolution, dataset_dir=f"/media/aris/Data/master2025dev/datasets/wood/4_main_categories/train")