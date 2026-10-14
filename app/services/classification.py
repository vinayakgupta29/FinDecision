"""Backward-compatibility module alias for classification_service.

Re-exports canonical classification classes and utility functions to maintain
compatibility across older imports.
"""

from app.services.classification_service import (
    ALIASES,
    ClassificationService,
    canonical_label,
    softmax,
)

__all__ = [
    "ALIASES",
    "ClassificationService",
    "canonical_label",
    "softmax",
]
