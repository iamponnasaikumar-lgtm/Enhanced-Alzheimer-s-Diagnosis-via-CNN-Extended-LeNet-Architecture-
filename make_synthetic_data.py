"""Generate a small SYNTHETIC brain-like dataset for smoke-testing the pipeline.

These images are NOT real MRI scans. They only let you check that the code runs
end to end before you download real data. Never report results from them.

Example:
    python make_synthetic_data.py --out data_synthetic --per_class 60
"""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image

CLASSES = {"CN": 0, "MCI": 1, "AD": 2}     # severity level drives ventricle size / cortical thinning


def make_image(severity, rng, size=176):
    yy, xx = np.mgrid[0:size, 0:size]
    cx = size / 2 + rng.uniform(-6, 6)
    cy = size / 2 + rng.uniform(-6, 6)
    brain = ((xx - cx) / (size * 0.38)) ** 2 + ((yy - cy) / (size * 0.45)) ** 2 <= 1
    inner = ((xx - cx) / (size * (0.34 - 0.02 * severity))) ** 2 + \
            ((yy - cy) / (size * (0.41 - 0.02 * severity))) ** 2 <= 1

    img = np.zeros((size, size), dtype=float)
    img[brain] = 0.35                                   # cortical rim
    img[inner] = 0.60 + rng.uniform(-0.05, 0.05)        # white matter
    for dx in (-0.09 * size, 0.09 * size):              # ventricles grow with severity
        vw = size * (0.04 + 0.025 * severity)
        vh = size * (0.09 + 0.035 * severity)
        vent = ((xx - (cx + dx)) / vw) ** 2 + ((yy - cy) / vh) ** 2 <= 1
        img[vent] = 0.08
    img += rng.normal(0, 0.04, img.shape)
    return (np.clip(img, 0, 1) * 255).astype("uint8")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="data_synthetic")
    p.add_argument("--per_class", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    n_test = max(1, int(args.per_class * 0.2))
    for cname, sev in CLASSES.items():
        for split in ("train", "test"):
            (out / split / cname).mkdir(parents=True, exist_ok=True)
        for i in range(args.per_class):
            split = "test" if i < n_test else "train"
            Image.fromarray(make_image(sev, rng)).save(out / split / cname / f"{cname}_{i:04d}.png")
    print(f"Synthetic dataset written to {out.resolve()}")


if __name__ == "__main__":
    main()
