"""Evaluation script for assessing sentiment classification metrics on labeled datasets.

Reads a labeled CSV file (containing 'text' and 'label' columns), runs model
inference, and outputs accuracy, macro F1, per-class metrics, and confusion matrix.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

# Add project root to sys.path to allow standalone script invocation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.services.classification_service import ClassificationService
from app.services.model_service import ModelService


def main():
    """Run model evaluation across labeled CSV dataset and write JSON metrics report."""
    parser = argparse.ArgumentParser(
        description="Evaluate model classification performance on a labeled CSV dataset."
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to CSV file with 'text' and 'label' columns.",
    )
    parser.add_argument(
        "--output", default="reports/evaluation.json",
        help="Destination path for evaluation report JSON (default: reports/evaluation.json).",
    )
    args = parser.parse_args()

    # Load dataset and discard rows with missing text or ground-truth labels.
    dataset = pd.read_csv(args.csv).dropna(
        subset=["text", "label"]
    )

    settings = get_settings()

    # Initialize and load model on CPU.
    model = ModelService(
        settings.model_id,
        settings.model_revision,
        device="cpu",
    )
    model.load()

    classifier = ClassificationService(
        model, settings.chunk_chars, settings.max_chunks
    )

    actual = []
    predicted = []

    # Iterate through each sample and collect predicted vs actual classes.
    for text, label in zip(
        dataset["text"], dataset["label"]
    ):
        prediction, _, _ = classifier.classify(str(text))
        actual.append(str(label).strip().lower())
        predicted.append(prediction)

    labels = sorted(set(actual) | set(predicted))

    # Calculate statistical metrics.
    report = {
        "model": settings.model_id,
        "examples": len(dataset),
        "accuracy": accuracy_score(actual, predicted),
        "macro_f1": f1_score(
            actual, predicted,
            average="macro",
            zero_division=0,
        ),
        "per_class": classification_report(
            actual,
            predicted,
            labels=labels,
            output_dict=True,
            zero_division=0,
        ),
        "labels": labels,
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=labels
        ).tolist(),
        "caveat": (
            "Results are not independent if the evaluation "
            "examples overlap the checkpoint's training data."
        ),
    }

    # Persist report to JSON file.
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
