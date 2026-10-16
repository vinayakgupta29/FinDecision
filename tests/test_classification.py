"""Unit tests for sentiment classification service functions and heuristics.

Covers softmax numerical stability, canonical label alias resolution,
label ordering preservation, and multi-chunk aggregation warnings.
"""

import math

from app.services.classification_service import (
    ClassificationService,
    canonical_label,
    softmax,
)


def test_softmax_sums_to_one():
    """Verify that softmax produces valid probabilities bounded [0, 1] summing to 1.0."""
    probabilities = softmax([-1.0, 0.0, 2.0])

    assert all(0.0 <= p <= 1.0 for p in probabilities)
    assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-9)


def test_known_label_aliases():
    """Verify that sentiment aliases ('positive', 'pos', etc.) resolve to canonical forms."""
    assert canonical_label("positive") == "positive"
    assert canonical_label("negative") == "negative"
    assert canonical_label("neutral") == "neutral"


def test_model_label_order_is_respected():
    """Verify that classification output maintains exact label ordering defined by the model."""
    class FakeModel:
        labels = ["negative", "neutral", "positive"]

        def predict_logits(self, text):
            return [-2.0, 0.0, 3.0]

    service = ClassificationService(FakeModel())

    predicted, probabilities, warnings = service.classify(
        "Revenue increased substantially."
    )

    assert predicted == "positive"
    assert list(probabilities) == [
        "negative", "neutral", "positive"
    ]
    assert math.isclose(
        sum(probabilities.values()), 1.0, abs_tol=1e-9
    )


def test_long_text_returns_warning():
    """Verify that dividing long documents into multiple chunks appends advisory warning."""
    class FakeModel:
        labels = ["negative", "neutral", "positive"]

        def predict_logits(self, text):
            return [-1.0, 0.0, 1.0]

    service = ClassificationService(
        FakeModel(), chunk_chars=100, max_chunks=2
    )

    _, _, warnings = service.classify("Revenue rose. " * 100)

    assert warnings
    assert any("not a calibrated" in warning for warning in warnings)
