"""Classify a single MRI image with a trained model.

Example:
    python predict.py --model results/extended_lenet/best.keras \
        --class_names results/class_names.json --image path/to/scan.jpg --gradcam out.png
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf

import data
from gradcam import make_gradcam


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--class_names", required=True, help="class_names.json written by train.py")
    p.add_argument("--image", required=True)
    p.add_argument("--gradcam", default=None, help="Optional path to save a Grad-CAM overlay")
    args = p.parse_args()

    model = tf.keras.models.load_model(args.model)
    class_names = json.load(open(args.class_names))
    img = data.load_image_array(args.image)
    cam, pred, probs = make_gradcam(model, img)

    print(f"Prediction: {class_names[pred]}  (confidence {probs[pred]:.1%})")
    for name, pr in sorted(zip(class_names, probs), key=lambda t: -t[1]):
        print(f"  {name:20s} {pr:.1%}")
    print("Research/educational use only - not a medical diagnosis.")

    if args.gradcam:
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(img[..., 0], cmap="gray")
        ax.imshow(cam, cmap="jet", alpha=0.4)
        ax.set_title(f"Predicted: {class_names[pred]}")
        ax.axis("off")
        fig.savefig(args.gradcam, dpi=150, bbox_inches="tight")
        print(f"Grad-CAM saved to {args.gradcam}")


if __name__ == "__main__":
    main()
