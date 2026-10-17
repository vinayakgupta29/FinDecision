"""Performance benchmarking script for FinDecision sentiment inference.

Measures cold start model loading time, CPU inference latency distribution across
sample iterations, and peak resident set size (memory consumption).
"""

import json
import platform
import resource
import sys
import time
from pathlib import Path

# Add project root to sys.path to allow standalone script invocation.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings
from app.services.classification_service import ClassificationService
from app.services.model_service import ModelService


def main():
    """Execute benchmark run and output JSON report to reports/benchmark.json."""
    settings = get_settings()

    # Initialize model service configured for CPU device.
    model = ModelService(
        settings.model_id,
        settings.model_revision,
        device="cpu",
    )

    # Measure cold model load duration (download/deserialization into memory).
    start = time.perf_counter()
    model.load()
    cold_start = time.perf_counter() - start

    classifier = ClassificationService(
        model, settings.chunk_chars, settings.max_chunks
    )

    sample = (
        "The company reported stronger quarterly revenue "
        "but weaker operating margins."
    )

    latencies = []

    # Run multiple warmup and test iterations to profile inference latency.
    for _ in range(10):
        start = time.perf_counter()
        classifier.classify(sample)
        latencies.append(
            (time.perf_counter() - start) * 1000
        )

    ordered = sorted(latencies)

    # Compile structured hardware and performance report.
    report = {
        "model": settings.model_id,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cpu": platform.processor() or "unknown",
        "cold_model_load_seconds": round(cold_start, 3),
        "latency_ms": {
            "min": round(min(latencies), 2),
            "median": round(ordered[len(ordered) // 2], 2),
            "max": round(max(latencies), 2),
        },
        "peak_rss": resource.getrusage(
            resource.RUSAGE_SELF
        ).ru_maxrss,
        "note": (
            "On Linux, ru_maxrss is in KiB. "
            "Measure model snapshot size separately."
        ),
    }

    # Save benchmark result to disk.
    output = Path("reports/benchmark.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
