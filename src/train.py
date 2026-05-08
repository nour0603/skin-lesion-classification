"""
Training utilities for the skin lesion classification project.

This module refactors the core training logic from the original notebook into
reusable functions for:

1. Seven-class skin lesion classification.
2. Binary malignant-vs-benign classification.
3. Simple baseline model comparison using RGB histogram features.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import Model, layers
from tensorflow.keras.applications import EfficientNetB0

try:
    from .config import BATCH_SIZE, IMG_SIZE, KEEP7, MALIGNANT_CLASSES, SEED
    from .preprocessing import make_tf_dataset, rgb_hist_features
except ImportError:
    from config import BATCH_SIZE, IMG_SIZE, KEEP7, MALIGNANT_CLASSES, SEED
    from preprocessing import make_tf_dataset, rgb_hist_features


def load_index(index_csv: str | Path, keep_classes: set[str] = KEEP7) -> pd.DataFrame:
    """
    Load a prepared index CSV and keep valid project classes.
    """
    df = pd.read_csv(index_csv)
    df = df[df["label_name"].isin(keep_classes)].dropna(subset=["image_path"]).reset_index(drop=True)

    if "split" not in df.columns:
        raise ValueError("Index must contain a 'split' column. Run data preparation first.")

    return df


def add_multiclass_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Add integer labels for seven-class classification.
    """
    out = df.copy()
    classes = sorted(out["label_name"].unique())
    label_to_id = {label: idx for idx, label in enumerate(classes)}
    out["y"] = out["label_name"].map(label_to_id).astype(int)
    return out, classes


def build_efficientnet_multiclass(
    n_classes: int,
    image_size: int = IMG_SIZE,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """
    Build an EfficientNetB0 transfer-learning model for multi-class classification.
    """
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
        pooling="avg",
    )

    x = layers.Dropout(0.3)(base.output)
    output = layers.Dense(n_classes, activation="softmax", name="lesion_class")(x)
    model = Model(base.input, output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )

    return model


def build_efficientnet_binary(
    image_size: int = IMG_SIZE,
    learning_rate: float = 1e-4,
) -> tf.keras.Model:
    """
    Build an EfficientNetB0 transfer-learning model for binary malignancy classification.
    """
    base = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3),
        pooling="avg",
    )

    x = layers.Dropout(0.35)(base.output)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.25)(x)
    output = layers.Dense(1, activation="sigmoid", name="malignancy_probability")(x)
    model = Model(base.input, output)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Recall(name="recall"),
            tf.keras.metrics.Precision(name="precision"),
        ],
    )

    return model


def make_multiclass_datasets(
    df: pd.DataFrame,
    classes: list[str],
    batch_size: int = BATCH_SIZE,
    image_size: int = IMG_SIZE,
):
    """
    Create train, validation, and test datasets for seven-class classification.
    """
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]

    train_ds = make_tf_dataset(
        train_df["image_path"],
        train_df["y"].to_numpy(),
        batch_size=batch_size,
        image_size=image_size,
        training=True,
        one_hot_classes=len(classes),
    )
    val_ds = make_tf_dataset(
        val_df["image_path"],
        val_df["y"].to_numpy(),
        batch_size=batch_size,
        image_size=image_size,
        one_hot_classes=len(classes),
    )
    test_ds = make_tf_dataset(
        test_df["image_path"],
        test_df["y"].to_numpy(),
        batch_size=batch_size,
        image_size=image_size,
        one_hot_classes=len(classes),
    )

    return train_ds, val_ds, test_ds


def make_binary_datasets(
    df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
    image_size: int = IMG_SIZE,
):
    """
    Create train, validation, and test datasets for malignant-vs-benign classification.
    """
    out = df.copy()
    out["is_malignant"] = out["label_name"].isin(MALIGNANT_CLASSES).astype(int)

    train_df = out[out["split"] == "train"]
    val_df = out[out["split"] == "val"]
    test_df = out[out["split"] == "test"]

    train_ds = make_tf_dataset(
        train_df["image_path"],
        train_df["is_malignant"].to_numpy(),
        batch_size=batch_size,
        image_size=image_size,
        training=True,
    )
    val_ds = make_tf_dataset(
        val_df["image_path"],
        val_df["is_malignant"].to_numpy(),
        batch_size=batch_size,
        image_size=image_size,
    )
    test_ds = make_tf_dataset(
        test_df["image_path"],
        test_df["is_malignant"].to_numpy(),
        batch_size=batch_size,
        image_size=image_size,
    )

    return train_ds, val_ds, test_ds, out


def compute_multiclass_weights(y: np.ndarray) -> dict[int, float]:
    """
    Compute class weights for imbalanced seven-class classification.
    """
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return dict(zip(classes.astype(int), weights.astype(float)))


def compute_binary_weights(y: np.ndarray) -> dict[int, float]:
    """
    Compute simple binary weights, increasing the weight of the malignant class.
    """
    positives = int(np.sum(y))
    negatives = int(len(y) - positives)

    return {
        0: 1.0,
        1: negatives / max(positives, 1),
    }


def train_multiclass(
    index_csv: str | Path,
    output_model_path: str | Path = "models/efficientnetb0_multiclass.keras",
    epochs: int = 10,
) -> tf.keras.Model:
    """
    Train the seven-class EfficientNetB0 model from a prepared index CSV.
    """
    df = load_index(index_csv)
    df, classes = add_multiclass_labels(df)
    train_ds, val_ds, _ = make_multiclass_datasets(df, classes)

    class_weight = compute_multiclass_weights(df[df["split"] == "train"]["y"].to_numpy())

    model = build_efficientnet_multiclass(n_classes=len(classes))
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
    )

    output_model_path = Path(output_model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model_path)

    return model


def train_binary(
    index_csv: str | Path,
    output_model_path: str | Path = "models/efficientnetb0_binary.keras",
    epochs: int = 10,
) -> tf.keras.Model:
    """
    Train the binary malignant-vs-benign EfficientNetB0 model.
    """
    df = load_index(index_csv)
    train_ds, val_ds, _, binary_df = make_binary_datasets(df)

    y_train = binary_df[binary_df["split"] == "train"]["is_malignant"].to_numpy()
    class_weight = compute_binary_weights(y_train)

    model = build_efficientnet_binary()
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_weight,
    )

    output_model_path = Path(output_model_path)
    output_model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(output_model_path)

    return model


def compare_baselines(df: pd.DataFrame, max_images: int = 1000) -> pd.DataFrame:
    """
    Compare quick baseline models using RGB histogram features.

    This is useful as a sanity check against the EfficientNetB0 model.
    """
    sample = df.sample(n=min(max_images, len(df)), random_state=SEED).copy()

    encoder = LabelEncoder()
    y = encoder.fit_transform(sample["label_name"])

    x = rgb_hist_features(sample["image_path"])

    train_mask = sample["split"].eq("train").to_numpy()
    test_mask = sample["split"].eq("test").to_numpy()

    x_train, y_train = x[train_mask], y[train_mask]
    x_test, y_test = x[test_mask], y[test_mask]

    models = {
        "KNN": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "Naive Bayes": GaussianNB(),
        "Logistic Regression": make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1000),
        ),
    }

    rows = []
    for name, baseline in models.items():
        baseline.fit(x_train, y_train)
        pred = baseline.predict(x_test)

        rows.append(
            {
                "model": name,
                "accuracy": accuracy_score(y_test, pred),
                "macro_f1": f1_score(y_test, pred, average="macro"),
            }
        )

    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


def main() -> None:
    """
    CLI placeholder.

    Example:
        python src/train.py data/processed/index_balanced_skin_lesions.csv
    """
    print("Use train_multiclass(...) or train_binary(...) with a prepared index CSV.")


if __name__ == "__main__":
    main()
