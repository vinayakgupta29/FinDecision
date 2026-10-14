"""Financial text sentiment classification service.

Handles label canonicalization, numerically stable softmax calculation,
per-chunk logit generation, and length-weighted aggregation of sentiment probabilities.
"""

import math
from collections import defaultdict

from app.services.chunking import split_text


# Standardized sentiment alias lookup table for model output labels.
ALIASES = {
    "positive": "positive",
    "pos": "positive",
    "negative": "negative",
    "neg": "negative",
    "neutral": "neutral",
    "neu": "neutral",
}


def canonical_label(label: str) -> str:
    """Normalize raw model label strings into standard canonical classes.

    Args:
        label: Raw label string from model id2label mapping (e.g. 'POSITIVE', 'pos', 'Neutral').

    Returns:
        str: Canonicalized label ('positive', 'negative', or 'neutral').
    """
    key = label.strip().lower().replace(" ", "_")
    return ALIASES.get(key, key)


def softmax(logits: list[float]) -> list[float]:
    """Compute numerically stable softmax probability distribution from raw model logits.

    Subtracts the maximum logit before exponentiation to prevent numerical overflow
    with large positive logits.

    Args:
        logits: List of float logit values produced by the classification head.

    Returns:
        list[float]: Normalized probability values summing to 1.0.

    Raises:
        ValueError: If logits list is empty or sum of exponentials is invalid/non-finite.
    """
    if not logits:
        raise ValueError("The model returned no logits")

    maximum = max(logits)
    exponentials = [
        math.exp(value - maximum)
        for value in logits
    ]
    total = sum(exponentials)

    if not math.isfinite(total) or total <= 0:
        raise ValueError("Invalid model logits")

    return [value / total for value in exponentials]


class ClassificationService:
    """Orchestrates document chunking and weighted sentiment classification.

    Attributes:
        model_service: Instance of ModelService used to compute inference logits.
        chunk_chars: Maximum characters allocated per chunk.
        max_chunks: Upper limit of text chunks evaluated for a single document.
    """

    def __init__(
        self,
        model_service,
        chunk_chars: int = 2_000,
        max_chunks: int = 12,
    ):
        """Initialize the classification orchestrator.

        Args:
            model_service: ModelService instance providing loaded model and tokenizer.
            chunk_chars: Target character size for splitting input text.
            max_chunks: Maximum chunks to process per document.
        """
        self.model_service = model_service
        self.chunk_chars = chunk_chars
        self.max_chunks = max_chunks

    def classify(
        self, text: str
    ) -> tuple[str, dict[str, float], list[str]]:
        """Classify sentiment of input text across multiple chunks.

        Splits text into chunks, computes softmax probabilities for each chunk,
        and aggregates results using character-length weighting:
            P(label) = sum(P_chunk(label) * len(chunk)) / sum(len(chunk))

        Args:
            text: Financial text or article content to classify.

        Returns:
            tuple[str, dict[str, float], list[str]]:
                - predicted: The class with the highest overall aggregated probability.
                - probabilities: Dictionary mapping canonical labels to normalized scores.
                - warnings: List of warnings regarding chunking or truncation.

        Raises:
            ValueError: If the text is empty or contains no usable content.
            RuntimeError: If model labels are unavailable, ambiguous, or logit counts mismatch.
        """
        # Split text into chunks bounded by sentence boundaries.
        prepared = split_text(
            text, self.chunk_chars, self.max_chunks
        )

        if not prepared.chunks:
            raise ValueError("No usable text was supplied")

        # Resolve model labels into canonical representations.
        labels = [
            canonical_label(label)
            for label in self.model_service.labels
        ]

        if not labels:
            raise RuntimeError("Model labels are unavailable")

        if len(labels) != len(set(labels)):
            raise RuntimeError(
                f"Duplicate or ambiguous label mapping: {labels}"
            )

        totals = defaultdict(float)
        total_weight = 0

        # Perform inference on each chunk and accumulate length-weighted probabilities.
        for chunk in prepared.chunks:
            logits = self.model_service.predict_logits(chunk)

            if len(logits) != len(labels):
                raise RuntimeError(
                    "Logit count does not match the label count"
                )

            probabilities = softmax(logits)
            weight = max(1, len(chunk))

            for label, probability in zip(labels, probabilities):
                totals[label] += probability * weight

            total_weight += weight

        # Compute weighted average probabilities across all chunks.
        probabilities = {
            label: totals[label] / total_weight
            for label in labels
        }

        # Normalize to ensure probabilities strictly sum to 1.0.
        normalizer = sum(probabilities.values())
        probabilities = {
            label: max(0.0, value / normalizer)
            for label, value in probabilities.items()
        }

        # Pick the label with highest probability as the final prediction.
        predicted = max(probabilities, key=probabilities.get)

        return predicted, probabilities, prepared.warnings
