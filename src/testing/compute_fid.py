from pathlib import Path
from torch_fidelity import calculate_metrics
import argparse


def main():
    parser = argparse.ArgumentParser(description="Compute FID between real and synthetic image folders")
    parser.add_argument("--real", type=str, required=True, help="Path to real images")
    parser.add_argument("--fake", type=str, required=True, help="Path to synthetic images")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--search_deep", action="store_true", help="Search for all images in subdirectories")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    real_path = Path(args.real)
    fake_path = Path(args.fake)

    if not real_path.exists():
        raise FileNotFoundError(f"Real path does not exist: {real_path}")
    if not fake_path.exists():
        raise FileNotFoundError(f"Fake path does not exist: {fake_path}")

    metrics = calculate_metrics(
        input1=str(fake_path),   # generated / synthetic
        input2=str(real_path),   # reference / real
        cuda=args.cuda,
        fid=True,
        isc=False,
        kid=False,
        prc=False,
        batch_size=args.batch_size,
        samples_find_deep=args.search_deep,
        verbose=True,
    )

    print(f"\nFID: {metrics['frechet_inception_distance']:.4f}")


if __name__ == "__main__":
    main()