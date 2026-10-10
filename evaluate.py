"""Evaluation: accuracy, precision, recall, F1, confusion matrix and ROC analysis."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


def predict_dataset(model, ds):
    """Return (y_true, class probabilities) for an un-shuffled dataset."""
    y_true = np.concatenate([y.numpy() for _, y in ds])
    probs = model.predict(ds, verbose=0)
    return y_true, probs


def plot_history(history, path):
    hist = history.history
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(hist["loss"], label="train")
    axes[0].plot(hist["val_loss"], label="validation")
    axes[0].set(title="Loss", xlabel="Epoch")
    axes[1].plot(hist["accuracy"], label="train")
    axes[1].plot(hist["val_accuracy"], label="validation")
    axes[1].set(title="Accuracy", xlabel="Epoch")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_confusion(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title="Confusion matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_roc(y_true, probs, class_names, path):
    """One-vs-rest ROC curves; returns the macro-average AUC (NaN if undefined)."""
    n = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n)))
    if n == 2:                                   # label_binarize returns 1 column
        y_bin = np.hstack([1 - y_bin, y_bin])

    fig, ax = plt.subplots(figsize=(6, 5))
    aucs = []
    for i, name in enumerate(class_names):
        positives = y_bin[:, i].sum()
        if positives == 0 or positives == len(y_bin):
            continue                              # AUC undefined for this class
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        class_auc = auc(fpr, tpr)
        aucs.append(class_auc)
        ax.plot(fpr, tpr, label=f"{name} (AUC = {class_auc:.2f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="ROC curves (one-vs-rest)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return float(np.mean(aucs)) if aucs else float("nan")


def evaluate_model(model, ds, class_names, out_dir):
    """Compute metrics on `ds`, write reports/plots to `out_dir`, return a metrics dict."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    y_true, probs = predict_dataset(model, ds)
    y_pred = probs.argmax(axis=1)
    labels = list(range(len(class_names)))

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=labels, target_names=class_names, zero_division=0
    )
    (out_dir / "classification_report.txt").write_text(report)

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    np.savetxt(out_dir / "confusion_matrix.csv", cm, fmt="%d", delimiter=",")
    _plot_confusion(cm, class_names, out_dir / "confusion_matrix.png")
    macro_auc = _plot_roc(y_true, probs, class_names, out_dir / "roc_curve.png")

    metrics = {
        "accuracy": float(accuracy),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
        "f1_macro": float(f1),
        "roc_auc_macro": macro_auc,
        "test_samples": int(len(y_true)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(report)
    return metrics
