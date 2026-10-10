"""Dataset discovery, splitting, preprocessing and augmentation."""
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import config

AUTOTUNE = tf.data.AUTOTUNE


def set_seed(seed: int = config.SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# --------------------------------------------------------------------------- #
# Discovery and splitting
# --------------------------------------------------------------------------- #
def _scan_folder(folder, class_names=None):
    """Read images from folder/<class_name>/*.{jpg,png,...}."""
    folder = Path(folder)
    if class_names is None:
        class_names = sorted(d.name for d in folder.iterdir() if d.is_dir())
    if not class_names:
        raise ValueError(f"No class sub-folders found in {folder}")

    paths, labels = [], []
    for idx, cname in enumerate(class_names):
        cdir = folder / cname
        if not cdir.is_dir():
            raise FileNotFoundError(f"Missing class folder: {cdir}")
        files = sorted(
            p for p in cdir.rglob("*") if p.suffix.lower() in config.IMAGE_EXTENSIONS
        )
        paths += [str(p) for p in files]
        labels += [idx] * len(files)

    if not paths:
        raise ValueError(f"No images found under {folder}")
    return paths, np.array(labels, dtype="int32"), class_names


def make_splits(data_dir, seed: int = config.SEED):
    """Return ({'train','val','test'} -> (paths, labels), class_names).

    Layout A: data_dir/train/<class>/... and data_dir/test/<class>/...
    Layout B: data_dir/<class>/...   (a stratified train/val/test split is made)
    """
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {data_dir}")

    if (data_dir / "train").is_dir() and (data_dir / "test").is_dir():
        tr_p, tr_l, classes = _scan_folder(data_dir / "train")
        te_p, te_l, _ = _scan_folder(data_dir / "test", classes)
        tr_p, va_p, tr_l, va_l = train_test_split(
            tr_p, tr_l, test_size=config.VAL_FRACTION, stratify=tr_l, random_state=seed
        )
    else:
        paths, labels, classes = _scan_folder(data_dir)
        tr_p, te_p, tr_l, te_l = train_test_split(
            paths, labels, test_size=config.TEST_FRACTION, stratify=labels, random_state=seed
        )
        val_rel = config.VAL_FRACTION / (1.0 - config.TEST_FRACTION)
        tr_p, va_p, tr_l, va_l = train_test_split(
            tr_p, tr_l, test_size=val_rel, stratify=tr_l, random_state=seed
        )

    splits = {
        "train": (list(tr_p), np.asarray(tr_l, dtype="int32")),
        "val": (list(va_p), np.asarray(va_l, dtype="int32")),
        "test": (list(te_p), np.asarray(te_l, dtype="int32")),
    }
    return splits, classes


def compute_weights(labels, n_classes):
    """Balanced class weights to counter class imbalance."""
    w = compute_class_weight("balanced", classes=np.arange(n_classes), y=labels)
    return {i: float(x) for i, x in enumerate(w)}


# --------------------------------------------------------------------------- #
# Preprocessing
# --------------------------------------------------------------------------- #
def _load_image(path, label):
    """Decode -> grayscale -> resize -> normalise to [0,1] -> contrast stretch."""
    raw = tf.io.read_file(path)
    img = tf.io.decode_image(raw, channels=1, expand_animations=False)
    img.set_shape([None, None, 1])
    img = tf.image.resize(img, config.IMG_SIZE)
    img = tf.cast(img, tf.float32) / 255.0                 # normalisation
    lo, hi = tf.reduce_min(img), tf.reduce_max(img)
    img = (img - lo) / (hi - lo + 1e-6)                    # per-image contrast stretching
    return img, label


def load_image_array(path) -> np.ndarray:
    """Preprocess a single image exactly like the training pipeline. Shape (H, W, 1)."""
    img, _ = _load_image(str(path), 0)
    return img.numpy()


# --------------------------------------------------------------------------- #
# Augmentation + tf.data
# --------------------------------------------------------------------------- #
def build_augmenter():
    """Rotation and contrast augmentation (applied to training batches only)."""
    layers = tf.keras.layers
    return tf.keras.Sequential(
        [
            layers.RandomRotation(config.ROTATION_FACTOR, fill_mode="constant", fill_value=0.0),
            layers.RandomContrast(config.CONTRAST_FACTOR, value_range=(0.0, 1.0)),
        ],
        name="augmentation",
    )


def make_dataset(paths, labels, batch_size=config.BATCH_SIZE, training=False,
                 augment=None, seed=config.SEED):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(_load_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size)
    if training and augment is not None:
        ds = ds.map(
            lambda x, y: (tf.clip_by_value(augment(x, training=True), 0.0, 1.0), y),
            num_parallel_calls=AUTOTUNE,
        )
    return ds.prefetch(AUTOTUNE)
