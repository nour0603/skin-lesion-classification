# Model Results

This page summarises the headline results from the project report.

## Subproblem A: Seven-Class Skin Lesion Classification

The multi-class classifier predicts the top three most likely lesion categories from seven target classes:

- Actinic keratosis
- Basal cell carcinoma
- Benign keratosis
- Dermatofibroma
- Melanoma
- Melanocytic nevus
- Vascular lesion

### Baseline Comparison

| Model | Accuracy | F1 Score | Main Limitation |
|---|---:|---:|---|
| Logistic Regression using RGB histograms | 0.30 | 0.33 weighted | Linear model using crude colour histograms; confused visually similar lesions. |
| Naive Bayes | 0.32 | 0.39 weighted | Independence assumption and weak texture modelling. |
| KNN | 0.64 | 0.68 weighted | Sensitive to scale/noise and weak generalisation, especially for melanoma recall. |
| EfficientNetB0 CNN | 0.787 | 0.65 macro | Stronger spatial feature learning, substantially better ranking and top-3 performance. |

### EfficientNetB0 Results

| Metric | Result |
|---|---:|
| Top-1 accuracy | 78.7% |
| Top-3 accuracy | 97-98% |
| Macro-F1 | 0.65 |
| Macro-AUC | 0.95 |

The report highlights that top-3 accuracy is the most clinically useful output for this task because it provides a shortlist of likely diagnoses rather than a single overconfident prediction.

## Subproblem B: Binary Malignancy Classification

The binary classifier predicts whether a lesion is likely benign or malignant. A safety-first threshold was selected using the validation precision-recall curve to prioritise high malignant recall and reduce false negatives.

### EfficientNetB0 Binary Results

| Metric | Result |
|---|---:|
| ROC-AUC | 0.89-0.91 |
| Accuracy | 71-73% |
| Malignant recall | 0.94-0.95 |
| Precision | 0.49 |
| Negative predictive value | 0.97 |

### Confusion Matrix at Safety-First Threshold

| True Class | Predicted Benign | Predicted Malignant |
|---|---:|---:|
| Benign | 2170 | 1306 |
| Malignant | 73 | 1238 |

The key trade-off is intentional: the model over-flags more benign lesions as suspicious in exchange for missing fewer malignant cases. This aligns with a screening context where false negatives are more concerning than false positives.

## Key Issues and Limitations

- Significant class imbalance, especially the dominance of melanocytic nevus images.
- Visual overlap between melanoma, nevus, and benign keratosis.
- Compute and I/O constraints from training EfficientNetB0 in Google Colab.
- Lower precision caused by the deliberate safety-first threshold choice.
- Need for future validation on more diverse datasets and clinical workflows.

## Future Work

- Add larger and more diverse datasets, including better skin tone diversity.
- Add explainability outputs such as Grad-CAM heatmaps.
- Improve melanoma recall in the seven-class classifier.
- Calibrate binary risk thresholds for different clinical settings.
- Conduct clinical validation before any real-world medical use.
