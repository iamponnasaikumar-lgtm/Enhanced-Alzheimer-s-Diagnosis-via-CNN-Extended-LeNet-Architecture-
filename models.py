"""Model definitions: LeNet baseline, plain CNN baseline and extended LeNet."""

import keras
from keras import layers

import config


def _input():
    return layers.Input(
        shape=(*config.IMG_SIZE, config.CHANNELS),
        name="mri"
    )


# ---------------------------------------------------------
# 1. LeNet Baseline
# ---------------------------------------------------------
def build_lenet_baseline(n_classes: int) -> keras.Model:
    """Original LeNet-style baseline model."""

    inp = _input()

    x = layers.Conv2D(
        6,
        5,
        padding="same",
        activation="relu"
    )(inp)

    x = layers.AveragePooling2D(2)(x)

    x = layers.Conv2D(
        16,
        5,
        activation="relu",
        name="last_conv"
    )(x)

    x = layers.AveragePooling2D(2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(
        120,
        activation="relu"
    )(x)

    x = layers.Dense(
        84,
        activation="relu"
    )(x)

    out = layers.Dense(
        n_classes,
        activation="softmax"
    )(x)

    return keras.Model(
        inp,
        out,
        name="lenet_baseline"
    )


# ---------------------------------------------------------
# 2. Plain CNN Baseline
# ---------------------------------------------------------
def build_plain_cnn(n_classes: int) -> keras.Model:
    """Deeper CNN baseline with MaxPooling and no Dropout."""

    inp = _input()

    x = layers.Conv2D(
        32,
        3,
        padding="same",
        activation="relu"
    )(inp)

    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(
        64,
        3,
        padding="same",
        activation="relu"
    )(x)

    x = layers.MaxPooling2D(2)(x)

    x = layers.Conv2D(
        128,
        3,
        padding="same",
        activation="relu",
        name="last_conv"
    )(x)

    x = layers.MaxPooling2D(2)(x)

    x = layers.Flatten()(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    out = layers.Dense(
        n_classes,
        activation="softmax"
    )(x)

    return keras.Model(
        inp,
        out,
        name="plain_cnn"
    )


# ---------------------------------------------------------
# 3. Extended LeNet
# ---------------------------------------------------------
def build_extended_lenet(n_classes: int) -> keras.Model:
    """
    Extended LeNet with MaxPooling and moderate Dropout.

    Architecture:
    Conv32 -> MaxPool -> Dropout
    Conv64 -> MaxPool -> Dropout
    Conv128 -> MaxPool -> Dropout
    Flatten -> Dense128 -> Dropout -> Output
    """

    inp = _input()

    # -----------------------------------------------------
    # Convolution Block 1
    # -----------------------------------------------------
    x = layers.Conv2D(
        32,
        5,
        padding="same",
        activation="relu"
    )(inp)

    x = layers.MaxPooling2D(2)(x)

    x = layers.Dropout(0.10)(x)

    # -----------------------------------------------------
    # Convolution Block 2
    # -----------------------------------------------------
    x = layers.Conv2D(
        64,
        5,
        padding="same",
        activation="relu"
    )(x)

    x = layers.MaxPooling2D(2)(x)

    x = layers.Dropout(0.10)(x)

    # -----------------------------------------------------
    # Convolution Block 3
    # -----------------------------------------------------
    x = layers.Conv2D(
        128,
        3,
        padding="same",
        activation="relu",
        name="last_conv"
    )(x)

    x = layers.MaxPooling2D(2)(x)

    x = layers.Dropout(0.10)(x)

    # -----------------------------------------------------
    # Classifier
    # -----------------------------------------------------
    x = layers.Flatten()(x)

    x = layers.Dense(
        128,
        activation="relu"
    )(x)

    x = layers.Dropout(0.25)(x)

    # -----------------------------------------------------
    # Output Layer
    # -----------------------------------------------------
    out = layers.Dense(
        n_classes,
        activation="softmax"
    )(x)

    return keras.Model(
        inp,
        out,
        name="extended_lenet"
    )


# ---------------------------------------------------------
# Model Registry
# ---------------------------------------------------------
BUILDERS = {
    "lenet_baseline": build_lenet_baseline,
    "plain_cnn": build_plain_cnn,
    "extended_lenet": build_extended_lenet,
}