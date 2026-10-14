"""Machine learning model management and inference service.

Manages Hugging Face AutoModel and AutoTokenizer lifecycles, memory loading,
thread-safe CPU inference locking, and logit extraction.
"""

import threading
from typing import Any

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)


class ModelService:
    """Encapsulates transformer model loading, tokenization, and forward passes.

    Attributes:
        model_id: Hugging Face model repository identifier.
        revision: Optional git commit SHA or tag for reproducible model downloads.
        device: Target execution torch.device ('cpu' or 'cuda').
        tokenizer: Hugging Face tokenizer instance, or None if uninitialized.
        model: Hugging Face sequence classification model instance, or None.
    """

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        device: str = "cpu",
    ):
        """Initialize ModelService configuration.

        Args:
            model_id: Model repository ID from Hugging Face Hub (e.g. '9mark9/finbert-minilm-sentiment').
            revision: Optional immutable commit hash or branch tag.
            device: Compute device string ('cpu' or 'cuda').
        """
        self.model_id = model_id
        self.revision = revision
        self.device = torch.device(device)

        self.tokenizer = None
        self.model = None
        # Thread lock ensures concurrent requests safely share model forward passes without race conditions.
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        """Indicate whether both the model weights and tokenizer have been loaded into memory."""
        return (
            self.model is not None
            and self.tokenizer is not None
        )

    def load(self) -> None:
        """Download (if needed) and load model weights and tokenizer into system memory.

        Moves the model to the configured compute device and activates evaluation mode.
        """
        kwargs: dict[str, Any] = {}

        if self.revision:
            kwargs["revision"] = self.revision

        # Instantiate Hugging Face tokenizer and classification model.
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, **kwargs
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, **kwargs
        )

        model.to(self.device)
        model.eval()

        self.tokenizer = tokenizer
        self.model = model

    @property
    def labels(self) -> list[str]:
        """Return ordered list of classification label names from model config."""
        if not self.loaded:
            return []

        mapping = self.model.config.id2label

        return [
            str(mapping[i])
            for i in range(self.model.config.num_labels)
        ]

    @property
    def max_length(self) -> int | None:
        """Return maximum sequence length supported by tokenizer, or None if unspecified."""
        if self.tokenizer is None:
            return None

        length = getattr(
            self.tokenizer, "model_max_length", None
        )

        if isinstance(length, int) and 0 < length < 100_000:
            return length

        return None

    def predict_logits(self, text: str) -> list[float]:
        """Tokenize text and execute model forward pass to compute raw unnormalized logits.

        Thread-safe and runs inside torch.inference_mode() to minimize memory overhead
        and disable autograd graph tracking.

        Args:
            text: Text chunk to feed through the transformer encoder.

        Returns:
            list[float]: Output logits corresponding to model labels.

        Raises:
            RuntimeError: If called before model is loaded into memory.
        """
        if not self.loaded:
            raise RuntimeError("Model is not loaded")

        # Encode text chunk into PyTorch tensors with truncation.
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length or 512,
        )

        encoded = {
            key: value.to(self.device)
            for key, value in encoded.items()
        }

        # Lock forward pass to protect concurrent thread safety.
        with self._lock, torch.inference_mode():
            logits = self.model(**encoded).logits[0]

        return logits.detach().float().cpu().tolist()
