# Enhanced Alzheimer's Diagnosis via CNN - Extended LeNet

Classifies brain MRI slices into cognitive-impairment stages (for example CN / MCI / AD)
with an **extended LeNet** CNN, and compares it against two baseline deep-learning models.

> Research / educational project. It is not a medical device and must not be used for diagnosis.

## What is implemented

| Resume claim | Where it lives |
|---|---|
| Enhanced CNN based on LeNet | `models.py` -> `build_extended_lenet` |
| Extended with MaxPooling and Dropout layers | Max pooling after every conv block, Dropout (0.25 / 0.5) after pooling and before the classifier |
| Preprocessing and augmentation (normalization, rotation, contrast enhancement) | `data.py` -> `_load_image` (resize, [0,1] normalization, per-image contrast stretching) and `build_augmenter` (random rotation + random contrast) |
| Region-specific analysis | `gradcam.py` -> Grad-CAM heatmaps plus attention share inside a configurable ROI box |
| Comparison with baseline models | `train.py` trains `lenet_baseline`, `plain_cnn`, `extended_lenet` on identical splits and writes `model_comparison.csv/.png` |
| Classification pipeline for stages incl. MCI and AD | Classes come from your dataset's folder names (`CN`, `MCI`, `AD`, ...) |
| Accuracy, precision, recall, F1, confusion matrix, ROC | `evaluate.py` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Tested with Python 3.12, TensorFlow 2.21 / Keras 3. It runs on CPU; a GPU only makes it faster.

## Dataset

The ADNI data is free but needs an access request: <https://adni.loni.usc.edu>. Any folder-based
MRI image dataset also works (for example the public 4-class Alzheimer MRI dataset on Kaggle).
Use one of these layouts, with one sub-folder per class:

```
data/                      data/
  train/                     CN/  *.jpg
    CN/  *.jpg               MCI/ *.jpg
    MCI/ *.jpg               AD/  *.jpg
    AD/  *.jpg             (a stratified 70/15/15 split is created automatically)
  test/
    CN/  MCI/  AD/
```

Class names are read from the folder names and sorted alphabetically.

## Run

```bash
# 1. Train and compare all three models (results/ is created)
python train.py --data_dir data --epochs 30

# 2. Grad-CAM heatmaps and ROI attention analysis for the best model
python gradcam.py --model results/extended_lenet/best.keras --data_dir data

# 3. Predict one image
python predict.py --model results/extended_lenet/best.keras \
    --class_names results/class_names.json --image path/to/scan.jpg --gradcam overlay.png
```

Useful options: `--models extended_lenet` (train one model), `--batch_size 16`, `--lr 5e-4`,
`--patience 8`. Image size, augmentation strength and the ROI box are in `config.py`.

## Outputs (`results/`)

```
class_names.json
model_comparison.csv / .png          accuracy, precision, recall, F1, ROC-AUC, gain vs LeNet baseline
<model>/best.keras                   best checkpoint by validation loss
<model>/training_curves.png          loss and accuracy per epoch
<model>/confusion_matrix.png / .csv
<model>/roc_curve.png                one-vs-rest ROC per class + macro AUC
<model>/classification_report.txt    per-class precision / recall / F1
<model>/metrics.json
<model>/gradcam/                     gradcam_samples.png, roi_analysis.csv, roi_summary_by_class.csv
```

## Smoke test without real data

`make_synthetic_data.py` creates a tiny **synthetic** dataset so you can confirm the code runs before
downloading anything:

```bash
python make_synthetic_data.py --out data_synthetic --per_class 60
python train.py --data_dir data_synthetic --output_dir smoke_results --epochs 4 --batch_size 16
```

The images are not real MRI scans, so ignore any accuracy it reports.

## Important notes

- **Avoid subject leakage.** ADNI has many slices per patient. If the same patient's slices land in both
  train and test, accuracy is inflated. Split by patient/subject ID (organise `train/` and `test/` yourself)
  before you trust or quote any numbers.
- **Region-specific analysis is attention-based.** Grad-CAM shows which image regions drive a prediction,
  and `roi_analysis.csv` measures how much of that attention falls inside a box. The default box is a
  generic central placeholder; it does not segment the hippocampus or cortex. True hippocampal/cortical
  analysis needs registration to a brain atlas. Adjust `--roi` if you can justify a region, and describe it
  as "Grad-CAM-based region analysis" unless you add segmentation.
- **Report only your own results.** Whatever `model_comparison.csv` shows on real data is the number you
  can cite; the synthetic smoke test says nothing about real performance.
