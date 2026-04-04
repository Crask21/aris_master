from pathlib import Path
from torch_fidelity import calculate_metrics
import argparse
from statistics import mean, stdev
from torch import manual_seed
from torch import manual_seed
import numpy as np


def compute_fid(
    real,
    fake,
    cuda=False,
    search_deep=False,
    batch_size=64,
):
    real_path = Path(real)
    fake_path = Path(fake)

    if not real_path.exists():
        raise FileNotFoundError(f"Real path does not exist: {real_path}")
    if not fake_path.exists():
        raise FileNotFoundError(f"Fake path does not exist: {fake_path}")


    metrics = calculate_metrics(
        input1=str(fake_path),   # generated / synthetic
        input2=str(real_path),   # reference / real
        cuda=cuda, 
        fid=True,
        isc=False,
        kid=False,
        prc=False,
        batch_size=batch_size,
        samples_find_deep=search_deep,
        verbose=True,
    )
    fid = metrics["frechet_inception_distance"]

    print(f"\nFID: {fid:.4f}")
    return {"fid": fid}


def main():
    parser = argparse.ArgumentParser(description="Compute FID between real and synthetic image folders")
    parser.add_argument("--real", type=str, required=True, help="Path to real images")
    parser.add_argument("--fake", type=str, required=True, help="Path to synthetic images")
    parser.add_argument("--dont_search_deep", action="store_false", help="Search for all images in subdirectories")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    compute_fid(
        real=args.real,
        fake=args.fake,
        cuda=True,
        search_deep=args.dont_search_deep,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()