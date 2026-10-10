"""Train and compare the baseline and extended LeNet models.

Example:
    python train.py --data_dir data --epochs 30
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

import config
import data
import models
from evaluate import evaluate_model, plot_history


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--data_dir", required=True, help="Dataset folder (see README for layout)")
    p.add_argument("--output_dir", default="results")
    p.add_argument("--epochs", type=int, default=config.EPOCHS)
    p.add_argument("--batch_size", type=int, default=config.BATCH_SIZE)
    p.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    p.add_argument("--patience", type=int, default=config.PATIENCE)
    p.add_argument("--models", nargs="+", default=config.MODEL_NAMES,
                   choices=list(models.BUILDERS), help="Models to train")
    return p.parse_args()


def plot_comparison(df, path):
    metrics = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    x = np.arange(len(df))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, m in enumerate(metrics):
        ax.bar(x + (i - 1.5) * width, df[m], width, label=m)
    ax.set_xticks(x)
    ax.set_xticklabels(df["model"])
    ax.set_ylim(0, 1.05)
    ax.set_title("Model comparison on the test set")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    data.set_seed()
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    splits, class_names = data.make_splits(args.data_dir)
    n_classes = len(class_names)
    (out_root / "class_names.json").write_text(json.dumps(class_names))

    print(f"Classes: {class_names}")
    for name, (paths, labels) in splits.items():
        counts = {class_names[i]: int((labels == i).sum()) for i in range(n_classes)}
        print(f"  {name:5s}: {len(paths):5d} images  {counts}")

    augmenter = data.build_augmenter()
    train_ds = data.make_dataset(*splits["train"], batch_size=args.batch_size,
                                 training=True, augment=augmenter)
    val_ds = data.make_dataset(*splits["val"], batch_size=args.batch_size)
    test_ds = data.make_dataset(*splits["test"], batch_size=args.batch_size)
    class_weights = data.compute_weights(splits["train"][1], n_classes)
    print(f"Class weights: {class_weights}")

    rows = []
    for name in args.models:
        print(f"\n{'=' * 70}\nTraining {name}\n{'=' * 70}")
        out_dir = out_root / name
        out_dir.mkdir(parents=True, exist_ok=True)

        tf.keras.backend.clear_session()
        data.set_seed()
        model = models.BUILDERS[name](n_classes)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(args.lr),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        model.summary()

        callbacks = [
            tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience,
                                             restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                                 patience=3, min_lr=1e-6),
            tf.keras.callbacks.ModelCheckpoint(str(out_dir / "best.keras"),
                                               monitor="val_loss", save_best_only=True),
            tf.keras.callbacks.CSVLogger(str(out_dir / "training_log.csv")),
        ]
        history = model.fit(train_ds, validation_data=val_ds, epochs=args.epochs,
                            class_weight=class_weights, callbacks=callbacks,
                            shuffle=False, verbose=2)
        plot_history(history, out_dir / "training_curves.png")

        metrics = evaluate_model(model, test_ds, class_names, out_dir)
        metrics.update(model=name, parameters=int(model.count_params()),
                       epochs_run=len(history.history["loss"]))
        rows.append(metrics)

    df = pd.DataFrame(rows)
    cols = ["model", "parameters", "epochs_run", "accuracy", "precision_macro",
            "recall_macro", "f1_macro", "roc_auc_macro", "test_samples"]
    df = df[cols]
    if "lenet_baseline" in df["model"].values:
        base = float(df.loc[df["model"] == "lenet_baseline", "accuracy"].iloc[0])
        df["accuracy_gain_vs_lenet_pts"] = ((df["accuracy"] - base) * 100).round(2)
    df.to_csv(out_root / "model_comparison.csv", index=False)
    plot_comparison(df, out_root / "model_comparison.png")

    print("\nModel comparison (test set):")
    print(df.round(4).to_string(index=False))
    print(f"\nAll outputs saved to: {out_root.resolve()}")


if __name__ == "__main__":
    main()
