"""Backward-compatibility module alias for model_service.

Re-exports ModelService to preserve backwards compatibility across legacy import paths.
"""

from app.services.model_service import ModelService

__all__ = ["ModelService"]
