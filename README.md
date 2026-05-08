# Skin Lesion Classification

A machine learning project for classifying dermoscopic skin lesion images and exploring melanoma risk prediction using public medical image datasets.

This project investigates two related tasks:

1. **Multi-class skin lesion classification** across seven lesion categories.
2. **Binary malignancy classification** to distinguish malignant and benign lesion groups.

The workflow uses TensorFlow/Keras, EfficientNetB0 transfer learning, image preprocessing, leakage-aware train/validation/test splitting, class imbalance handling, and clinically relevant evaluation metrics.

> **Disclaimer:** This project is for educational and research purposes only. It is not intended for clinical diagnosis, medical decision-making, or replacing assessment by a qualified healthcare professional.

## Project Motivation

Melanoma is a serious form of skin cancer where early detection can significantly affect outcomes. This project explores how machine learning models can be trained on dermoscopic image datasets to classify skin lesions and evaluate malignancy-related risk signals.

The goal is not to build a deployable medical product, but to demonstrate a careful machine learning workflow involving image preprocessing, model evaluation, class imbalance awareness, and discussion of limitations.

## Project Overview

The current workflow includes:

- Dataset acquisition using KaggleHub
- ISIC 2019 and HAM10000-style dataset preprocessing
- Label mapping into readable lesion categories
- Combined tidy metadata table creation
- Class distribution analysis
- Selective undersampling of the overrepresented melanocytic nevus class
- Image preprocessing and augmentation
- Baseline model comparison using traditional machine learning models
- EfficientNetB0 transfer learning for seven-class classification
- Binary malignant-vs-benign classification
- Threshold selection with recall sensitivity in mind
- Confusion matrix, F1-score, classification report, ROC-AUC, and top-3 accuracy evaluation

## Results Summary

Detailed results are available in [`reports/model_results.md`](reports/model_results.md).

### Subproblem A: Seven-Class Classification

| Model | Top-1 Accuracy | Top-3 Accuracy | Macro-F1 | Macro-AUC |
|---|---:|---:|---:|---:|
| EfficientNetB0 CNN | 78.7% | 97-98% | 0.65 | 0.95 |

The top-3 output is especially useful because it gives a shortlist of likely lesion types rather than forcing a single overconfident prediction.

### Subproblem B: Binary Malignancy Classification

| Model | ROC-AUC | Accuracy | Malignant Recall | Precision | NPV |
|---|---:|---:|---:|---:|---:|
| EfficientNetB0 CNN | 0.89-0.91 | 71-73% | 0.94-0.95 | 0.49 | 0.97 |

The binary classifier uses a safety-first threshold that prioritises high malignant recall. This intentionally increases false positives to reduce false negatives, which is more appropriate for a screening-oriented medical context.

## Tech Stack

- Python
- TensorFlow / Keras
- EfficientNetB0
- scikit-learn
- OpenCV
- NumPy / Pandas
- Matplotlib / Seaborn
- KaggleHub
- Jupyter Notebook / Google Colab

## Repository Structure

```text
skin-lesion-classification/
├── data/                 # Dataset instructions only; raw data is not committed
├── notebooks/            # Jupyter notebooks for experiments and analysis
├── reports/              # Final report and generated figures
│   ├── model_results.md  # Summary of evaluation results
│   └── figures/          # Evaluation plots, confusion matrices, etc.
├── src/                  # Reusable Python source code
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── tests/                # Test files and testing notes
├── requirements.txt
├── .gitignore
└── README.md
```

## Dataset

This project is designed around public dermoscopic skin lesion datasets, including:

- ISIC 2019-style dermoscopic image data
- HAM10000 / ISIC 2018-style dermoscopic image data

Large datasets and medical images are not included in this repository. See [`data/README.md`](data/README.md) for instructions on how dataset files should be organised locally.

## Methodology

The intended workflow is:

1. Load dermoscopic image metadata and image files.
2. Convert labels into a consistent set of readable skin lesion classes.
3. Explore class distribution and identify imbalance.
4. Apply a balancing strategy to reduce dominance of common benign classes.
5. Preprocess and augment images for model input.
6. Train baseline models and an EfficientNetB0 transfer-learning model.
7. Evaluate the model using classification metrics and visualisations.
8. Discuss limitations, bias, uncertainty, and responsible use.

## Evaluation Metrics

For medical image classification, raw accuracy is not enough. This project considers metrics such as:

- Accuracy
- Precision
- Recall / sensitivity
- F1-score
- Confusion matrix
- ROC-AUC for binary classification
- Top-3 accuracy for multi-class classification

In melanoma-related classification, recall/sensitivity is especially important because false negatives can be more harmful than false positives.

## How to Run

Clone the repository:

```bash
git clone https://github.com/nour0603/skin-lesion-classification.git
cd skin-lesion-classification
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Prepare the dataset index:

```bash
python src/data.py
```

The training utilities can then be called from `src/train.py` using the prepared CSV index.

## Report

The key report findings are summarised in [`reports/model_results.md`](reports/model_results.md).

The full written report can be added manually as:

```text
reports/final-report.pdf
```

## Current Status

This repository has been refactored from the original notebook workflow into reusable Python modules. The next steps are to add final figures, optional tests, and the full report PDF.

## Future Improvements

- Add the full written report PDF
- Add model performance figures to `reports/figures/`
- Add automated tests for preprocessing utilities
- Add a trained-model card explaining limitations
- Add clearer comparison between baseline and EfficientNetB0 models
- Add a lightweight prediction demo
- Add Grad-CAM explainability visualisations
