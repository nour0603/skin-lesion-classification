"""
Dataset preparation utilities.

The original project notebook used KaggleHub to download ISIC 2019 and
HAM10000-style dermoscopic image datasets, converted their labels into a
shared readable format, combined them into one index table, then applied
leakage-aware train/validation/test splits.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

try:
    from .config import (
        DEFAULT_OUTPUT_DIR,
        HAM_LABEL_MAP,
        ISIC_LABEL_MAP,
        KEEP7,
        SEED,
    )
except ImportError:  # Allows running this file directly with `python src/data.py`.
    from config import DEFAULT_OUTPUT_DIR, HAM_LABEL_MAP, ISIC_LABEL_MAP, KEEP7, SEED


def download_datasets() -> tuple[Path, Path]:
    """
    Download the public Kaggle datasets used in the project.

    Returns:
        Tuple of local paths: (isic_2019_dir, ham10000_dir).
    """
    isic_dir = Path(kagglehub.dataset_download("andrewmvd/isic-2019"))
    ham_dir = Path(kagglehub.dataset_download("nightfury007/ham10000-isic2018-raw"))
    return isic_dir, ham_dir


def _resolve_isic_image_dir(isic_dir: Path) -> Path:
    """Handle Kaggle mirrors where the images folder may be nested twice."""
    candidate = isic_dir / "ISIC_2019_Training_Input"
    nested = candidate / "ISIC_2019_Training_Input"
    return nested if nested.exists() else candidate


def load_isic2019(isic_dir: str | Path) -> pd.DataFrame:
    """
    Load and tidy ISIC 2019 metadata.

    The ISIC ground-truth file stores labels as one-hot columns. This function
    converts it into one row per image with a readable `label_name`.
    """
    isic_dir = Path(isic_dir)
    gt_path = isic_dir / "ISIC_2019_Training_GroundTruth.csv"
    meta_path = isic_dir / "ISIC_2019_Training_Metadata.csv"
    image_dir = _resolve_isic_image_dir(isic_dir)

    ground_truth = pd.read_csv(gt_path)
    metadata = pd.read_csv(meta_path)

    label_cols = [col for col in ground_truth.columns if col != "image"]
    labels_long = (
        ground_truth
        .melt(id_vars=["image"], value_vars=label_cols, var_name="raw_label", value_name="is_one")
        .query("is_one == 1")
        .drop(columns="is_one")
    )

    tidy = labels_long.merge(metadata, on="image", how="left")
    tidy["label_name"] = tidy["raw_label"].map(ISIC_LABEL_MAP).fillna("None of the above")
    tidy["image_path"] = tidy["image"].apply(lambda image_id: str(image_dir / f"{image_id}.jpg"))
    tidy["dataset"] = "ISIC2019"

    return tidy


def load_ham10000(ham_dir: str | Path) -> pd.DataFrame:
    """
    Load and tidy HAM10000/ISIC 2018-style metadata.
    """
    ham_dir = Path(ham_dir)
    dataverse_dir = ham_dir / "dataverse_files"
    image_dir = dataverse_dir / "HAM10000_images_combined_600x450"
    metadata_path = dataverse_dir / "HAM10000_metadata"

    try:
        metadata = pd.read_csv(metadata_path)
    except Exception:
        metadata = pd.read_csv(metadata_path, engine="python")

    columns = {column.lower(): column for column in metadata.columns}
    image_col = columns.get("image_id") or columns.get("image")
    dx_col = columns.get("dx") or columns.get("diagnosis")

    if image_col is None or dx_col is None:
        raise ValueError("Could not identify HAM10000 image or diagnosis columns.")

    tidy = metadata.copy()
    tidy["image"] = tidy[image_col]
    tidy["raw_label"] = tidy[dx_col]
    tidy["label_name"] = tidy["raw_label"].map(HAM_LABEL_MAP)
    tidy["image_path"] = tidy["image"].apply(lambda image_id: str(image_dir / f"{image_id}.jpg"))
    tidy["dataset"] = "HAM10000"

    return tidy


def combine_indexes(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """
    Combine dataset indexes and remove missing/duplicated image paths.
    """
    combined = pd.concat(list(frames), ignore_index=True)
    combined = combined[combined["label_name"].notna()].copy()
    combined = combined[combined["image_path"].apply(lambda path: Path(path).exists())]
    combined = combined.drop_duplicates(subset=["image_path"]).reset_index(drop=True)
    return combined


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Add an integer `label_id` column using scikit-learn's LabelEncoder.
    """
    out = df.copy()
    encoder = LabelEncoder()
    out["label_id"] = encoder.fit_transform(out["label_name"])
    return out, encoder


def selective_undersample_nevus(
    df: pd.DataFrame,
    target_count: int = 2500,
    seed: int = SEED,
) -> pd.DataFrame:
    """
    Reduce the dominant Melanocytic nevus class while keeping other classes.

    This mirrors the original notebook's custom balancing strategy: keep the
    clinically important/minority classes and selectively undersample the very
    common benign nevus class.
    """
    major = df[df["label_name"] == "Melanocytic nevus"]
    rest = df[df["label_name"] != "Melanocytic nevus"]

    sampled_major = major.sample(
        n=min(target_count, len(major)),
        random_state=seed,
    )

    balanced = (
        pd.concat([sampled_major, rest], ignore_index=True)
        .sample(frac=1, random_state=seed)
        .reset_index(drop=True)
    )

    return balanced


def add_leakage_aware_splits(
    df: pd.DataFrame,
    label_col: str = "label_name",
    seed: int = SEED,
    test_size: float = 0.15,
    val_size: float = 0.15,
) -> pd.DataFrame:
    """
    Add train/validation/test splits while reducing image leakage.

    If lesion_id or patient_id exists, the split is done at that grouping level
    rather than purely at image level.
    """
    out = df.copy()

    if "lesion_id" in out.columns and out["lesion_id"].notna().any():
        unit_col = "lesion_id"
    elif "patient_id" in out.columns and out["patient_id"].notna().any():
        unit_col = "patient_id"
    else:
        unit_col = "image_path"

    units = out[[unit_col, label_col]].drop_duplicates(unit_col).copy()

    stratify = units[label_col] if units[label_col].value_counts().min() >= 2 else None
    train_units, temp_units = train_test_split(
        units,
        test_size=test_size + val_size,
        random_state=seed,
        stratify=stratify,
    )

    temp_relative_test = test_size / (test_size + val_size)
    temp_stratify = temp_units[label_col] if temp_units[label_col].value_counts().min() >= 2 else None
    val_units, test_units = train_test_split(
        temp_units,
        test_size=temp_relative_test,
        random_state=seed,
        stratify=temp_stratify,
    )

    split_map = {
        **{unit: "train" for unit in train_units[unit_col]},
        **{unit: "val" for unit in val_units[unit_col]},
        **{unit: "test" for unit in test_units[unit_col]},
    }

    out["split"] = out[unit_col].map(split_map)
    return out


def prepare_project_index(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> pd.DataFrame:
    """
    Download datasets, build the combined index, balance it, split it, and save it.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    isic_dir, ham_dir = download_datasets()
    isic = load_isic2019(isic_dir)
    ham = load_ham10000(ham_dir)

    combined = combine_indexes([isic, ham])
    combined = combined[combined["label_name"].isin(KEEP7)].reset_index(drop=True)
    combined, _ = encode_labels(combined)

    balanced = selective_undersample_nevus(combined)
    balanced = add_leakage_aware_splits(balanced)

    output_path = output_dir / "index_balanced_skin_lesions.csv"
    balanced.to_csv(output_path, index=False)
    print(f"Saved prepared index to {output_path}")

    return balanced


if __name__ == "__main__":
    prepare_project_index()
