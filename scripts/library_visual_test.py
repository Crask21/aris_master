import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


def generate_field(size: int) -> torch.Tensor:
    torch.manual_seed(42)
    grid = torch.linspace(-3, 3, steps=size)
    x, y = torch.meshgrid(grid, grid, indexing="ij")
    base = torch.sin(x) * torch.cos(y) + 0.3 * torch.cos(2 * x + y)
    gaussian = torch.exp(-(x**2 + y**2) / 4)
    noise = 0.15 * torch.randn_like(base)
    return base + gaussian + noise


def plot_field(field: torch.Tensor, output: Path) -> None:
    field_np = field.detach().cpu().numpy()
    center_line = field_np[field_np.shape[0] // 2]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    im = axes[0].imshow(field_np, cmap="viridis", origin="lower")
    axes[0].set_title("Torch → NumPy field")
    plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].plot(center_line)
    axes[1].set_title("Center line profile")
    axes[1].grid(True, alpha=0.3)

    axes[2].hist(field_np.flatten(), bins=40, color="#4c72b0")
    axes[2].set_title("Value distribution")
    axes[2].grid(True, alpha=0.3)

    fig.suptitle("Matplotlib + Torch + NumPy sanity check")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    print(f"Saved visualization to {output}")


def main(output_path: str | None = None, size: int = 200) -> None:
    output = Path(output_path) if output_path else Path("outputs/visualizations/library_test.png")
    field = generate_field(size)
    mean = field.mean().item()
    std = field.std().item()
    print(f"Torch tensor stats — mean: {mean:.4f}, std: {std:.4f}")
    plot_field(field, output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check matplotlib, torch, and numpy.")
    parser.add_argument("--output", type=str, default=None, help="Where to save the visualization PNG.")
    parser.add_argument("--size", type=int, default=200, help="Grid size for the synthetic field.")
    args = parser.parse_args()
    main(args.output, args.size)
