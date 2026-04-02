# ---------------------------------------------------------------------------- #
#                                    Imports                                   #
# ---------------------------------------------------------------------------- #
import argparse
import tkinter as tk
import torch
from pathlib import Path
from diffusers import UNet2DModel, AutoencoderKL, DDIMPipeline
from tqdm import tqdm
from src.waste_diffuser.pipeline import Pipeline
import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageTk, ImageFilter
import torchvision.transforms as transforms
import glob
import re
import math

from typing import List, Optional, Union, Union

from diffusers.schedulers import DDIMScheduler
from diffusers.pipelines import ImagePipelineOutput

# ---------------------------------------------------------------------------- #
#                                   Functions                                  #
# ---------------------------------------------------------------------------- #
# ------------------------------ SDEdit Pipeline ----------------------------- #
class SDEdit(Pipeline):
    def __init__(self, unet: UNet2DModel, scheduler: DDIMScheduler):
        super().__init__(unet=unet, scheduler=scheduler)

    def _target_size(self):
        sample_size = self.unet.config.sample_size
        if isinstance(sample_size, int):
            return sample_size, sample_size
        return tuple(sample_size)

    def _prepare_guide_tensor(self, guide_image, batch_size: int):
        if guide_image is None:
            return None

        if isinstance(guide_image, (str, Path)):
            guide_image = Image.open(guide_image).convert("RGB")

        target_height, target_width = self._target_size()
        guide_image = guide_image.resize((target_width, target_height))

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])
        guide_tensor = transform(guide_image).unsqueeze(0)
        guide_tensor = guide_tensor.repeat(batch_size, 1, 1, 1)
        return guide_tensor.to(device=self._execution_device, dtype=self.unet.dtype)

    def _prepare_preserve_mask(self, mask_image, batch_size: int):
        if mask_image is None:
            return None

        if isinstance(mask_image, (str, Path)):
            mask_image = Image.open(mask_image).convert("L")

        target_height, target_width = self._target_size()
        mask_image = mask_image.resize((target_width, target_height), Image.NEAREST)
        mask_tensor = transforms.ToTensor()(mask_image).unsqueeze(0)
        mask_tensor = (mask_tensor > 0.5).to(dtype=self.unet.dtype)
        return mask_tensor.repeat(batch_size, 1, 1, 1).to(device=self._execution_device)

    def _show_preview(self, image: torch.Tensor, title: str, save_folder: Optional[str] = None):
        img = image[0].detach().float().cpu()

        if img.shape[0] == 1:
            vis = img[0]
            vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
            plt.imshow(vis.numpy(), cmap="gray")
        else:
            if img.shape[0] >= 3:
                vis = img[:3].permute(1, 2, 0)
            else:
                vis = img.mean(dim=0)
            vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
            plt.imshow(vis.numpy())

        plt.title(title)
        plt.axis("off")
        plt.show()

    def _save_preview(self, image: torch.Tensor, filename: str, save_folder: Optional[str] = None):
        img = image[0].detach().float().cpu()

        if img.shape[0] == 1:
            vis = img[0]
            vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
            plt.imshow(vis.numpy(), cmap="gray")
        else:
            if img.shape[0] >= 3:
                vis = img[:3].permute(1, 2, 0)
            else:
                vis = img.mean(dim=0)
            vis = (vis - vis.min()) / (vis.max() - vis.min() + 1e-8)
            plt.imshow(vis.numpy())

        plt.title("")
        plt.axis("off")
        save_folder = save_folder or '.'
        os.makedirs(save_folder, exist_ok=True)
        fullpath = os.path.join(save_folder, filename)
        plt.savefig(fullpath)

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        eta: float = 0.0,
        num_inference_steps: int = 50,
        use_clipped_model_output: Optional[bool] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        preview: bool = False,
        guide_image: Optional[Union[str, Path, Image.Image]] = None,
        preserve_mask: Optional[Union[str, Path, Image.Image]] = None,
        strength: float = 0.8,
        class_labels: Optional[torch.Tensor] = None,
        save_images: bool = False,
        save_images_dir: Optional[str] = None,
    ):
        preview_steps = 45
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch "
                f"size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if class_labels is not None:
            class_labels = class_labels.to(self._execution_device)

        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps.to(self._execution_device)

        noise = torch.randn(
            image_shape,
            generator=generator,
            device=self._execution_device,
            dtype=self.unet.dtype,
        )

        guide_tensor = self._prepare_guide_tensor(guide_image, batch_size)
        preserve_mask_tensor = self._prepare_preserve_mask(preserve_mask, batch_size)

        if guide_tensor is None:
            image = noise
            active_timesteps = timesteps
        else:
            strength = float(max(0.0, min(1.0, strength)))
            init_timestep = min(max(int(num_inference_steps * strength), 1), num_inference_steps)
            t_start = max(num_inference_steps - init_timestep, 0)
            active_timesteps = timesteps[t_start:]

            noise_timestep = active_timesteps[0].repeat(batch_size)
            image = self.scheduler.add_noise(guide_tensor, noise, noise_timestep)

            if preview:
                self._show_preview(guide_tensor, "Guide image")
                self._save_preview(guide_tensor, "guide_image.png", save_folder="previews")
                self._show_preview(image, f"Noised guide at t={int(active_timesteps[0])}")
                self._save_preview(image, f"Noised_guide_t{int(active_timesteps[0])}.png", save_folder="Noised guide")
        #print(f"Using {len(active_timesteps)} active timesteps out of {len(timesteps)} total timesteps.")
        total_steps = len(active_timesteps)

        for step_index, t in enumerate(self.progress_bar(active_timesteps)):
            if class_labels is None:
                model_output = self.unet(image, t).sample
            else:
                model_output = self.unet(image, t, class_labels=class_labels).sample

            step_output = self.scheduler.step(
                model_output,
                t,
                image,
                eta=eta,
                use_clipped_model_output=use_clipped_model_output,
                generator=generator,
            )
            image = step_output.prev_sample
            predicted_final = getattr(step_output, "pred_original_sample", None)

            if guide_tensor is not None and preserve_mask_tensor is not None:
                if step_index + 1 < total_steps:
                    next_timestep = active_timesteps[step_index + 1].repeat(batch_size)
                    reference = self.scheduler.add_noise(guide_tensor, noise, next_timestep)
                else:
                    reference = guide_tensor

                image = preserve_mask_tensor * reference + (1.0 - preserve_mask_tensor) * image

                if predicted_final is not None:
                    predicted_final = preserve_mask_tensor * guide_tensor + (1.0 - preserve_mask_tensor) * predicted_final

            if preview and step_index % max(1, total_steps // preview_steps) == 0:
                if predicted_final is not None:
                    self._show_preview(
                        predicted_final,
                        f"Predicted final image: step {step_index + 1}/{total_steps}",
                    )
                    self._show_preview(image, f"Noised image: step {step_index + 1}/{total_steps}")
                else:
                    self._show_preview(image, f"Step {step_index + 1}/{total_steps}")
            # Save intermediate images if requested (names match GIF assembly patterns)
            if save_images:
                out_dir = save_images_dir or '.'
                try:
                    if predicted_final is not None:
                        pred_name = f"Predicted_final_image_step_{step_index + 1}_{total_steps}.png"
                        self._save_preview(predicted_final, pred_name, save_folder=out_dir)
                    noised_name = f"Noised_image_step_{step_index + 1}_{total_steps}.png"
                    self._save_preview(image, noised_name, save_folder=out_dir)
                except Exception as e:
                    print(f"Warning: failed saving intermediate images: {e}")
            # print(f"Completed step {step_index + 1}/{total_steps} (t={int(t)})")

        if preview:
            self._show_preview(image, "Final output")
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.cpu().permute(0, 2, 3, 1).numpy()

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)

def draw_preserve_mask(
    guide_image_path,
    preserve_mask_path,
    brush_size=18,
    target_size=None,
    overwrite=False,
    ):
    if preserve_mask_path is None:
        return None

    preserve_mask_path = Path(preserve_mask_path)
    if preserve_mask_path.exists() and not overwrite:
        print(f"Using existing preserve mask: {preserve_mask_path}")
        return str(preserve_mask_path)

    guide_image = Image.open(guide_image_path).convert("RGB")
    if target_size is not None:
        if isinstance(target_size, int):
            target_size = (target_size, target_size)
        target_height, target_width = target_size
        guide_image = guide_image.resize((target_width, target_height))

    mask_image = Image.new("L", guide_image.size, 0)
    photo_image = None

    root = tk.Tk()
    root.title("Draw Preserve Mask")

    instruction = tk.Label(
        root,
        text="Left drag draws preserve regions, right drag erases, Save writes the mask and closes.",
    )
    instruction.pack(padx=8, pady=(8, 4))

    controls = tk.Frame(root)
    controls.pack(fill="x", padx=8, pady=(0, 8))

    brush_var = tk.IntVar(value=max(1, int(brush_size)))
    tk.Label(controls, text="Brush size").pack(side="left")
    tk.Scale(
        controls,
        from_=1,
        to=64,
        orient="horizontal",
        variable=brush_var,
        length=180,
    ).pack(side="left", padx=(8, 16))

    canvas = tk.Canvas(root, width=guide_image.width, height=guide_image.height, cursor="crosshair")
    canvas.pack(padx=8, pady=(0, 8))

    def refresh_canvas():
        nonlocal photo_image
        guide_array = np.asarray(guide_image).astype(np.float32)
        mask_array = np.asarray(mask_image).astype(np.float32) / 255.0
        overlay = guide_array.copy()
        overlay[..., 0] = overlay[..., 0] * (1.0 - 0.35 * mask_array) + 255.0 * (0.35 * mask_array)
        overlay[..., 1] = overlay[..., 1] * (1.0 - 0.35 * mask_array)
        overlay[..., 2] = overlay[..., 2] * (1.0 - 0.35 * mask_array)
        preview_image = Image.fromarray(overlay.astype(np.uint8))
        photo_image = ImageTk.PhotoImage(preview_image)
        canvas.delete("all")
        canvas.create_image(0, 0, anchor="nw", image=photo_image)

    def paint(event, fill_value):
        radius = max(1, int(brush_var.get()))
        x0 = max(0, event.x - radius)
        y0 = max(0, event.y - radius)
        x1 = min(guide_image.width - 1, event.x + radius)
        y1 = min(guide_image.height - 1, event.y + radius)
        draw = ImageDraw.Draw(mask_image)
        draw.ellipse((x0, y0, x1, y1), fill=fill_value)
        refresh_canvas()

    def draw_preserve(event):
        paint(event, 255)

    def erase_preserve(event):
        paint(event, 0)

    def clear_mask():
        mask_image.paste(0, (0, 0, guide_image.width, guide_image.height))
        refresh_canvas()
        print("Cleared preserve mask.")

    def save_and_close():
        preserve_mask_path.parent.mkdir(parents=True, exist_ok=True)
        mask_image.save(preserve_mask_path)
        print(f"Saved preserve mask to: {preserve_mask_path}")
        root.destroy()

    button_row = tk.Frame(root)
    button_row.pack(fill="x", padx=8, pady=(0, 8))
    tk.Button(button_row, text="Clear", command=clear_mask).pack(side="left")
    tk.Button(button_row, text="Save", command=save_and_close).pack(side="right")

    canvas.bind("<B1-Motion>", draw_preserve)
    canvas.bind("<Button-1>", draw_preserve)
    canvas.bind("<B3-Motion>", erase_preserve)
    canvas.bind("<Button-3>", erase_preserve)

    refresh_canvas()
    root.mainloop()

    return str(preserve_mask_path)


def generate_images(
    model_dir,
    output_dir,
    num_images=5,
    batch_size=16,
    num_inference_steps=50,
    class_label=0,
    preview=False,
    guide_image_path=None,
    preserve_mask_path=None,
    strength=0.8,
    save_images=False,
    verbose=False,
    force_generate=False,
    ):
    pipeline = SDEdit.from_pretrained(model_dir)
    pipeline.to("cuda")
    pipeline.unet.eval()

    if preserve_mask_path is not None and guide_image_path is not None:
        sample_size = pipeline.unet.config.sample_size
        target_size = (sample_size, sample_size) if isinstance(sample_size, int) else tuple(sample_size)
        draw_preserve_mask(
            guide_image_path=guide_image_path,
            preserve_mask_path=preserve_mask_path,
            target_size=target_size,
        )

    if not Path(output_dir).exists():
        os.makedirs(output_dir, exist_ok=True)
        print(f"Created output directory at: {Path(output_dir).resolve()}")
    # print("Embeds",pipeline.unet.config.num_class_embeds)
    if pipeline.unet.config.num_class_embeds is None:
        unconditional = True
    else:
        unconditional = False
    
    if verbose:
        print(f"Generating images for class {class_label}...")

    if not unconditional:
        output_dir_class = output_dir
        #os.path.join(output_dir, f"class_{class_label}")
    else:
        output_dir_class = output_dir

    os.makedirs(output_dir_class, exist_ok=True)
    existing_images = len([f for f in os.listdir(output_dir_class) if f.endswith(".png")])
    if verbose:
        print(f"Found {existing_images} existing images in directory {output_dir_class}.")

    if existing_images >= num_images and not force_generate:
        if verbose:
            print(f"Already have {existing_images} images, which is >= requested {num_images}. Skipping generation.")
        return
    
    if force_generate and existing_images > 0:
        if verbose:
            print(f"Force generate is True, but found {existing_images} existing images. New images will be numbered starting from {existing_images}.")
        n_images_to_generate = num_images
    else:
        n_images_to_generate = max(0, num_images - existing_images)
    full_batches = n_images_to_generate // batch_size

    if verbose:
        print(f"Generating {n_images_to_generate} new images...")

    for i in range(full_batches):
        if verbose:
            print(f"Generating batch {i + 1}/{full_batches}...")
        call_kwargs = dict(
            num_inference_steps=num_inference_steps,
            batch_size=batch_size,
            preview=preview,
            guide_image=guide_image_path,
            preserve_mask=preserve_mask_path,
            strength=strength,
                save_images=save_images,
                save_images_dir=output_dir_class,
        )
        
        if not unconditional:
            call_kwargs["class_labels"] = torch.tensor([class_label] * batch_size).to("cuda")
        # print(f"Calling pipeline with kwargs: {call_kwargs}")
        images = pipeline(**call_kwargs).images

        for j, img in enumerate(images):
            img.save(f"{output_dir_class}/{Path(guide_image_path).name}_{i * batch_size + j:04d}.png")

    remaining = n_images_to_generate % batch_size
    if remaining > 0:
        print(f"Generating remaining {remaining} images...")
        call_kwargs = dict(
            num_inference_steps=num_inference_steps,
            batch_size=remaining,
            preview=preview,
            guide_image=guide_image_path,
            preserve_mask=preserve_mask_path,
            strength=strength,
            save_images=save_images,
            save_images_dir=output_dir_class,
        )

        if not unconditional:
            call_kwargs["class_labels"] = torch.tensor([class_label] * remaining).to("cuda")
        # print(f"Calling pipeline with kwargs: {call_kwargs}")

        images = pipeline(**call_kwargs).images

        for j, img in enumerate(images):
            img.save(f"{output_dir_class}/{Path(guide_image_path).name}_{(num_images - remaining) + j:04d}.png")

    print(f"Finished generating images to: {Path(output_dir_class).resolve()}")


def _extract_number_key(path: str):
    """Return a tuple of integers found in the filename for stable numeric sorting."""
    nums = re.findall(r"\d+", os.path.basename(path))
    if not nums:
        return (0,)
    return tuple(int(x) for x in nums)


def make_gif_from_pattern(
    directory: str,
    glob_pattern: str,
    output_path: str,
    frame_duration_ms: Optional[int] = None,
    total_duration_ms: Optional[int] = None,
    last_frame_duration_ms: Optional[int] = None,
    reverse_pingpong: bool = False,
    loop: int = 0,
):
    """Create an animated GIF from images matching a glob pattern in directory.

    - If `total_duration_ms` is provided, it will be distributed evenly across all frames
        (including the reverse frames if `reverse_pingpong=True`). Otherwise `frame_duration_ms`
        is used for each frame.
    - `last_frame_duration_ms` overrides the duration of the final forward frame (and its
        mirrored counterpart when ping-ponging).
    """
    search = os.path.join(directory, glob_pattern)
    files = glob.glob(search)
    if not files:
        raise FileNotFoundError(f"No files match: {search}")

    # stable numeric sort using all integer groups in filename
    files = sorted(files, key=_extract_number_key)

    # load images and ensure same size/mode
    imgs = []
    for p in files:
        im = Image.open(p).convert("RGBA")
        imgs.append(im)

    # resize all to first image size if needed
    w, h = imgs[0].size
    for i, im in enumerate(imgs):
        if im.size != (w, h):
            imgs[i] = im.resize((w, h), Image.Resampling.LANCZOS)

    # build frame sequence (with optional ping-pong reverse)
    indices = list(range(len(imgs)))
    if reverse_pingpong and len(imgs) > 1:
        indices = indices + list(range(len(imgs) - 2, -1, -1))

    total_frames = len(indices)

    # compute per-frame duration
    if total_duration_ms is not None:
        per_frame = max(1, int(math.floor(float(total_duration_ms) / total_frames)))
    elif frame_duration_ms is not None:
        per_frame = int(frame_duration_ms)
    else:
        per_frame = 333  # default ~3 fps

    # build durations list
    durations = [per_frame] * total_frames

    # if last_frame_duration_ms provided, find forward last-frame index(es)
    if last_frame_duration_ms is not None:
        # forward final frame index in sequence is len(imgs)-1 at position (len(imgs)-1)
        # in the indices list, its first occurrence index is imgs_len-1
        forward_pos = indices.index(len(imgs) - 1)
        durations[forward_pos] = int(last_frame_duration_ms)
        if reverse_pingpong and len(imgs) > 1:
            # mirrored occurrence of final frame appears earlier in reverse part as well
            # the mirrored forward frame index is the first occurrence from the end where index==len(imgs)-1
            # find all occurrences and set them
            for k, idx in enumerate(indices):
                if idx == len(imgs) - 1:
                    durations[k] = int(last_frame_duration_ms)

    # assemble frames list in order
    frames = [imgs[i].convert('P', palette=Image.ADAPTIVE) for i in indices]

    # save GIF
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=loop,
        optimize=False,
    )

def SDEdit_gen_dataset(model_dir, output_dir, guide_image_folder, synth_images_per_guide_image, batch_size = 16, num_inference_steps = 50, preview = False, strength = 0.9, save_images = False, class_label = 0, verbose = False):
    # Print the core parameters for this dataset generation run
    print(f"Starting SDEdit dataset generation with parameters:")
    print(f"  Model directory: {model_dir}")
    print(f"  Output directory: {output_dir}")
    print(f"  Guide image folder: {guide_image_folder}")
    print(f"  Class label: {class_label}")
    print(f"  Synthetic images per guide image: {synth_images_per_guide_image}")
    print(f"  Batch size: {batch_size}")
    print(f"  Strength: {strength}")
    img_patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    guide_files = []
    for pat in img_patterns:
        guide_files.extend(glob.glob(os.path.join(guide_image_folder, pat)))
    guide_files = sorted(set(guide_files), key=_extract_number_key)

    os.makedirs(output_dir, exist_ok=True)
    # Number of existing .png files in the output directory (to avoid overwriting)
    existing_images = len([f for f in os.listdir(output_dir) if f.endswith(".png")])
    print(f"Found {existing_images} existing .png images in output directory: {output_dir}. New images will be numbered starting from {existing_images}.")
    if not guide_files:
        print(f"No guide images found in: {guide_image_folder}")
    else:
        for idx, guide_path in tqdm(enumerate(guide_files), total=len(guide_files)-existing_images):
            print(f"[{idx+1}/{len(guide_files)}] Using guide image: {guide_path}")
            # use a subdirectory per guide to avoid filename collisions
                # guide_stem = Path(guide_path).stem
                # out_dir = os.path.join(output_dir, guide_stem)
            generate_images(
                model_dir=model_dir,
                output_dir=output_dir,
                num_images=synth_images_per_guide_image,
                batch_size=batch_size,
                num_inference_steps=num_inference_steps,
                preview=preview,
                guide_image_path=guide_path,
                preserve_mask_path=None,
                strength=strength,
                save_images=save_images,
                class_label=class_label, 
                verbose=False,
                force_generate = True,  # always generate since we're doing per-guide generation and want synth_images_per_guide_image per guide
            )


def main():
    parser = argparse.ArgumentParser(description="SDEdit Image Generation")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the pretrained model directory")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save generated images")
    parser.add_argument("--num_images", type=int, default=1, help="Number of images to generate per class")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for generation")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Number of diffusion steps")
    parser.add_argument("--class_labels", type=int, default=None, help="Number of class labels (if using a class-conditional"
                        " model). If not specified, will use the number from the model config or default to 1.")
    parser.add_argument("--preview", action="store_true", help="Whether to show preview images during generation")
    parser.add_argument("--save_images", action="store_true", help="Whether to save intermediate images during generation")
    parser.add_argument("--guide_image_path", type=str, default=None, help="Path to guide image for SDEdit")
    parser.add_argument("--guide_image_folder", type=str, default=None, help="Folder with guide images for batch per-guide generation")
    parser.add_argument("--synth_images_per_guide_image", type=int, default=None, help="Number of synthetic images to generate per guide image (when --guide_image_folder is set)")
    parser.add_argument("--strength", type=float, default=0.9, help="Strength of the edit (0.0 = no change, 1.0 = full noise)")
    # GIF options
    parser.add_argument("--make_gif", action="store_true", help="After generation, assemble step images into a GIF")
    parser.add_argument("--include_noised", action="store_true", help="Also create GIF from noised images pattern (Noised_image_step_*)")
    parser.add_argument("--gif_output_name", type=str, default=None, help="Filename for the generated GIF (defaults per-pattern)")
    parser.add_argument("--frame_duration_ms", type=int, default=None, help="Duration per frame in milliseconds (overridden by --total_duration_ms)")
    parser.add_argument("--total_duration_ms", type=int, default=None, help="Total duration for the whole GIF in milliseconds (distributed evenly across frames)")
    parser.add_argument("--last_frame_duration_ms", type=int, default=None, help="Duration in ms for the final forward frame (can be longer)")
    parser.add_argument("--reverse_pingpong", action="store_true", help="Play GIF forward then in reverse (ping-pong)")

    args = parser.parse_args()

    # If a guide image folder and per-guide count are provided, generate per-guide images
    if args.guide_image_folder is not None and args.synth_images_per_guide_image is not None:
        SDEdit_gen_dataset(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            guide_image_folder=args.guide_image_folder,
            synth_images_per_guide_image=args.synth_images_per_guide_image,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            preview=args.preview,
            strength=args.strength,
            save_images=args.save_images,
        )
    else:
        generate_images(
            model_dir=args.model_dir,
            output_dir=args.output_dir,
            num_images=args.num_images,
            batch_size=args.batch_size,
            num_inference_steps=args.num_inference_steps,
            preview=args.preview,
            guide_image_path=args.guide_image_path,
            preserve_mask_path=None,
            strength=args.strength,
            save_images=args.save_images,
        )
    # Optional GIF generation
    if getattr(args, 'make_gif', False):
        # default patterns in the output directory
        patterns = []
        if getattr(args, 'include_noised', False):
            patterns.append(('Noised', 'Noised_image_step_*_*.png'))
        patterns.append(('Predicted', 'Predicted_final_image_step_*_*.png'))

        for name, pat in patterns:
            try:
                out_name = args.gif_output_name or f"{name}_steps.gif"
                out_path = os.path.join(args.output_dir, out_name)
                make_gif_from_pattern(
                    directory=args.output_dir,
                    glob_pattern=pat,
                    output_path=out_path,
                    frame_duration_ms=args.frame_duration_ms,
                    total_duration_ms=args.total_duration_ms,
                    last_frame_duration_ms=args.last_frame_duration_ms,
                    reverse_pingpong=args.reverse_pingpong,
                    loop=0,
                )
                print(f"Saved GIF: {out_path}")
            except FileNotFoundError as e:
                print(f"Skipping pattern {pat}: {e}")
if __name__ == "__main__":
    main()