import numpy as np
import pandas as pd
import pytest

from src.data import (
    add_leakage_aware_splits,
    encode_labels,
    selective_undersample_nevus,
)


def _make_df(counts: dict[str, int]) -> pd.DataFrame:
    rows = []
    for label, count in counts.items():
        for i in range(count):
            rows.append({"label_name": label, "image_path": f"/fake/{label}_{i}.jpg"})
    return pd.DataFrame(rows)


class TestSelectiveUndersampleNevus:
    def test_reduces_to_target_count(self):
        df = _make_df({"Melanocytic nevus": 3000, "Melanoma": 500})
        result = selective_undersample_nevus(df, target_count=2500)
        assert (result["label_name"] == "Melanocytic nevus").sum() == 2500

    def test_preserves_other_classes(self):
        df = _make_df({"Melanocytic nevus": 3000, "Melanoma": 500, "Dermatofibroma": 200})
        result = selective_undersample_nevus(df, target_count=2500)
        assert (result["label_name"] == "Melanoma").sum() == 500
        assert (result["label_name"] == "Dermatofibroma").sum() == 200

    def test_keeps_all_when_below_target(self):
        df = _make_df({"Melanocytic nevus": 100, "Melanoma": 50})
        result = selective_undersample_nevus(df, target_count=2500)
        assert (result["label_name"] == "Melanocytic nevus").sum() == 100

    def test_output_is_shuffled_dataframe(self):
        df = _make_df({"Melanocytic nevus": 200, "Melanoma": 50})
        result = selective_undersample_nevus(df, target_count=100)
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 150  # 100 nevus + 50 melanoma


class TestAddLeakageAwareSplits:
    def _make_df(self, n_per_class: int = 40) -> pd.DataFrame:
        classes = ["Melanoma", "Melanocytic nevus", "Basal cell carcinoma"]
        rows = []
        for cls in classes:
            for i in range(n_per_class):
                rows.append({"label_name": cls, "image_path": f"/fake/{cls}_{i}.jpg"})
        return pd.DataFrame(rows)

    def test_all_rows_get_a_split(self):
        df = self._make_df()
        result = add_leakage_aware_splits(df)
        assert result["split"].notna().all()

    def test_only_valid_split_values(self):
        df = self._make_df()
        result = add_leakage_aware_splits(df)
        assert set(result["split"].unique()) <= {"train", "val", "test"}

    def test_no_rows_lost(self):
        df = self._make_df()
        result = add_leakage_aware_splits(df)
        assert len(result) == len(df)

    def test_train_is_largest_split(self):
        df = self._make_df(n_per_class=100)
        result = add_leakage_aware_splits(df)
        counts = result["split"].value_counts()
        assert counts["train"] > counts["val"]
        assert counts["train"] > counts["test"]


class TestEncodeLabels:
    def test_adds_label_id_column(self):
        df = pd.DataFrame({"label_name": ["Melanoma", "Melanocytic nevus", "Melanoma"]})
        result, _ = encode_labels(df)
        assert "label_id" in result.columns

    def test_ids_are_integers(self):
        df = pd.DataFrame({"label_name": ["Melanoma", "Melanocytic nevus"]})
        result, _ = encode_labels(df)
        assert np.issubdtype(result["label_id"].dtype, np.integer)

    def test_encoder_inverse_round_trips(self):
        classes = ["Actinic keratosis", "Melanoma", "Melanocytic nevus"]
        df = pd.DataFrame({"label_name": classes})
        result, encoder = encode_labels(df)
        decoded = list(encoder.inverse_transform(result["label_id"]))
        assert decoded == classes

    def test_distinct_classes_get_distinct_ids(self):
        df = pd.DataFrame({"label_name": ["A", "B", "C", "A", "B"]})
        result, _ = encode_labels(df)
        mapping = dict(zip(result["label_name"], result["label_id"]))
        assert len(set(mapping.values())) == 3
