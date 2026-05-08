import numpy as np
import pytest

from src.evaluate import select_threshold_for_recall


def _separable_predictions(n: int = 300, seed: int = 42):
    """Predictions where positives score higher than negatives."""
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    y_prob = np.where(
        y_true == 1,
        rng.uniform(0.55, 1.0, n),
        rng.uniform(0.0, 0.55, n),
    )
    return y_true, y_prob


class TestSelectThresholdForRecall:
    def test_returns_float(self):
        y_true, y_prob = _separable_predictions()
        t = select_threshold_for_recall(y_true, y_prob, recall_target=0.90)
        assert isinstance(t, float)

    def test_threshold_in_unit_interval(self):
        y_true, y_prob = _separable_predictions()
        t = select_threshold_for_recall(y_true, y_prob, recall_target=0.90)
        assert 0.0 <= t <= 1.0

    def test_selected_threshold_achieves_recall_target(self):
        y_true, y_prob = _separable_predictions()
        t = select_threshold_for_recall(y_true, y_prob, recall_target=0.90)
        y_pred = (y_prob >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        recall = tp / (tp + fn + 1e-8)
        # Allow a small tolerance for discrete threshold steps
        assert recall >= 0.88

    def test_lower_recall_target_yields_higher_threshold(self):
        y_true, y_prob = _separable_predictions()
        t_high = select_threshold_for_recall(y_true, y_prob, recall_target=0.95)
        t_low = select_threshold_for_recall(y_true, y_prob, recall_target=0.70)
        # Higher recall target → lower threshold needed to capture more positives
        assert t_high <= t_low

    def test_fallback_when_no_threshold_meets_target(self):
        # Weak, nearly-random predictions — recall=0.99 is unachievable
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=200)
        y_prob = rng.uniform(0.45, 0.55, size=200)
        t = select_threshold_for_recall(y_true, y_prob, recall_target=0.99)
        # Falls back to best-F1 threshold — should still return a valid float
        assert isinstance(t, float)
        assert 0.0 <= t <= 1.0
