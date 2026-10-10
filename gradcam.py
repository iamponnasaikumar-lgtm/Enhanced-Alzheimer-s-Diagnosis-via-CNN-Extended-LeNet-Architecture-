"""Grad-CAM visual explanations and a simple region-of-interest attention analysis.

Example:
    python gradcam.py --model results/extended_lenet/best.keras --data_dir data
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import config
import data


def make_gradcam(model, img, class_index=None, layer_name="last_conv"):
    """Return (heatmap in [0,1] at IMG_SIZE, predicted/explained class, probabilities)."""
    grad_model = tf.keras.Model(model.inputs,
                                [model.get_layer(layer_name).output, model.output])
    x = tf.convert_to_tensor(img[None, ...], dtype=tf.float32)
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(x, training=False)
        if class_index is None:
            class_index = int(tf.argmax(preds[0]))
        score = preds[:, class_index]
    grads = tape.gradient(score, conv_out)
    weights = tf.reduce_mean(grads, axis=(1, 2))                    # (1, C)
    cam = tf.reduce_sum(conv_out * weights[:, None, None, :], axis=-1)[0]
    cam = tf.nn.relu(cam)
    cam = tf.image.resize(cam[..., None], config.IMG_SIZE).numpy()[..., 0]
    cam = cam / (cam.max() + 1e-8)
    return cam, class_index, preds[0].numpy()


def roi_attention_share(cam, roi):
    """Fraction of total Grad-CAM activation that falls inside a normalised box."""
    h, w = cam.shape
    x0, y0, x1, y1 = roi
    inside = cam[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)].sum()
    return float(inside / (cam.sum() + 1e-8))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True, help="Path to a trained .keras model")
    p.add_argument("--data_dir", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--n_per_class", type=int, default=3)
    p.add_argument("--out_dir", default=None, help="Default: <model folder>/gradcam")
    p.add_argument("--roi", type=float, nargs=4, default=list(config.DEFAULT_ROI),
                   metavar=("X0", "Y0", "X1", "Y1"),
                   help="Normalised ROI box (0-1) used for the attention-share analysis")
    return p.parse_args()


def main():
    args = parse_args()
    data.set_seed()
    model = tf.keras.models.load_model(args.model)
    splits, class_names = data.make_splits(args.data_dir)
    paths, labels = splits[args.split]

    out_dir = Path(args.out_dir) if args.out_dir else Path(args.model).parent / "gradcam"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(config.SEED)
    chosen = []
    for c in range(len(class_names)):
        idx = np.where(labels == c)[0]
        if len(idx):
            chosen += list(rng.choice(idx, size=min(args.n_per_class, len(idx)), replace=False))

    rows = []
    cols = 3
    fig_rows = int(np.ceil(len(chosen) / cols))
    fig, axes = plt.subplots(fig_rows, cols, figsize=(4 * cols, 4 * fig_rows), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")

    x0, y0, x1, y1 = args.roi
    for ax, i in zip(axes.ravel(), chosen):
        img = data.load_image_array(paths[i])
        cam, pred, probs = make_gradcam(model, img)
        share = roi_attention_share(cam, args.roi)
        h, w = cam.shape
        ax.imshow(img[..., 0], cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.4)
        ax.add_patch(patches.Rectangle((x0 * w, y0 * h), (x1 - x0) * w, (y1 - y0) * h,
                                       fill=False, edgecolor="white", linestyle="--"))
        ax.set_title(f"true: {class_names[labels[i]]} | pred: {class_names[pred]}\n"
                     f"ROI attention: {share:.0%}", fontsize=9)
        rows.append({"image": paths[i], "true": class_names[labels[i]],
                     "pred": class_names[pred], "confidence": float(probs[pred]),
                     "roi_attention_share": share})

    fig.tight_layout()
    fig.savefig(out_dir / "gradcam_samples.png", dpi=150)
    plt.close(fig)

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "roi_analysis.csv", index=False)
    summary = df.groupby("true")["roi_attention_share"].mean().round(4)
    summary.to_csv(out_dir / "roi_summary_by_class.csv")
    print("Mean share of Grad-CAM attention inside the ROI box, by true class:")
    print(summary.to_string())
    print(f"\nSaved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
