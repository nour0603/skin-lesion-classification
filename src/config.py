"""
Configuration constants for the skin lesion classification project.
"""

from pathlib import Path

SEED = 42
IMG_SIZE = 380
BATCH_SIZE = 32

KEEP7 = {
    "Melanoma",
    "Melanocytic nevus",
    "Basal cell carcinoma",
    "Actinic keratosis",
    "Benign keratosis",
    "Dermatofibroma",
    "Vascular lesion",
}

MALIGNANT_CLASSES = {
    "Melanoma",
    "Basal cell carcinoma",
    "Actinic keratosis",
}

ISIC_LABEL_MAP = {
    "MEL": "Melanoma",
    "NV": "Melanocytic nevus",
    "BCC": "Basal cell carcinoma",
    "AK": "Actinic keratosis",
    "BKL": "Benign keratosis",
    "DF": "Dermatofibroma",
    "VASC": "Vascular lesion",
    "SCC": "Squamous cell carcinoma",
}

HAM_LABEL_MAP = {
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratosis",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "vasc": "Vascular lesion",
    "scc": "Squamous cell carcinoma",
}

DEFAULT_OUTPUT_DIR = Path("data/processed")
