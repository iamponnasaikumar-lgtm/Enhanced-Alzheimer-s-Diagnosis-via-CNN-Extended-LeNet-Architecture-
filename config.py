"""Central configuration for the Alzheimer's MRI classification project."""

SEED = 42

# Input
IMG_SIZE = (128, 128)          # (height, width) every MRI slice is resized to
CHANNELS = 1                   # grayscale MRI
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Splits
VAL_FRACTION = 0.15            # carved out of the training split
TEST_FRACTION = 0.15           # only used when the dataset has no test/ folder

# Training
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
PATIENCE = 6                   # early-stopping patience (epochs)

# Augmentation
ROTATION_FACTOR = 0.04         # fraction of 2*pi -> roughly +/- 14 degrees
CONTRAST_FACTOR = 0.2          # random contrast change of +/- 20 %

# Models compared in train.py (baselines first, proposed model last)
MODEL_NAMES = ["lenet_baseline", "plain_cnn", "extended_lenet"]

# Region-of-interest box for Grad-CAM analysis, normalised (x0, y0, x1, y1).
# This is a placeholder central box; see README ("Region-specific analysis").
DEFAULT_ROI = (0.25, 0.35, 0.75, 0.75)
