# Skin Lesion Classification

A machine learning project for classifying dermoscopic skin lesion images and exploring melanoma risk prediction using medical image datasets.

This project focuses on the end-to-end workflow of a medical image classification task, including data preprocessing, exploratory analysis, model training, evaluation, and responsible interpretation of results.

> **Disclaimer:** This project is for educational and research purposes only. It is not intended for clinical diagnosis, medical decision-making, or replacing assessment by a qualified healthcare professional.

## Project Motivation

Melanoma is a serious form of skin cancer where early detection can significantly affect outcomes. This project explores how machine learning models can be trained on dermoscopic image datasets to classify skin lesions and evaluate malignancy-related risk signals.

The goal is not to build a deployable medical product, but to demonstrate a careful machine learning workflow involving image preprocessing, model evaluation, class imbalance awareness, and discussion of limitations.

## Features

- Dermoscopic image preprocessing pipeline
- Skin lesion classification workflow
- Model training and evaluation scripts
- Support for experiment notebooks
- Responsible medical-AI framing and limitations
- Space for final report, figures, and evaluation outputs

## Repository Structure

```text
skin-lesion-classification/
├── data/                 # Dataset instructions only; raw data is not committed
├── notebooks/            # Jupyter notebooks for experiments and analysis
├── reports/              # Final report and generated figures
│   └── figures/          # Evaluation plots, confusion matrices, etc.
├── src/                  # Reusable Python source code
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

This project is designed for dermoscopic skin lesion image datasets such as HAM10000 or ISIC-style datasets.

Large datasets and medical images are not included in this repository. See [`data/README.md`](data/README.md) for instructions on how dataset files should be organised locally.

## Methodology

The intended workflow is:

1. Load dermoscopic image metadata and image files.
2. Explore class distribution and identify imbalance.
3. Preprocess images for model input.
4. Train an image classification model.
5. Evaluate the model using classification metrics and visualisations.
6. Discuss limitations, bias, uncertainty, and responsible use.

## Evaluation Metrics

For medical image classification, raw accuracy is not enough. This project should consider metrics such as:

- Accuracy
- Precision
- Recall / sensitivity
- F1-score
- Confusion matrix
- ROC-AUC, where applicable

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

Run scripts from the project root:

```bash
python src/preprocessing.py
python src/train.py
python src/evaluate.py
```

## Report

A written project report can be added under:

```text
reports/final-report.pdf
```

Once added, this README can link to it directly.

## Current Status

This repository is being prepared as a polished portfolio project. The next steps are to add the existing notebook/model code, include evaluation results, and attach the final report.

## Future Improvements

- Add the cleaned experiment notebook
- Add model performance results and figures
- Add automated tests for preprocessing utilities
- Add a trained-model card explaining limitations
- Add clearer comparison between baseline and improved models
- Add a lightweight prediction script for demonstration purposes
