"""Data models and request/response schemas for FinDecision API.

Utilizes Pydantic for validation, schema serialization, and documentation.
"""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl


class TextRequest(BaseModel):
    """Payload schema for direct text sentiment classification.

    Attributes:
        text: Non-empty financial string to be classified (up to 50,000 characters).
    """

    model_config = ConfigDict(str_strip_whitespace=True)

    text: str = Field(
        min_length=1,
        max_length=50_000,
        description="Financial text to classify",
    )


class URLRequest(BaseModel):
    """Payload schema for article URL sentiment classification.

    Attributes:
        url: Valid HTTP or HTTPS web address of a financial article.
    """

    url: HttpUrl


class ArticleMetadata(BaseModel):
    """Metadata extracted from a fetched web article.

    Attributes:
        title: Extracted page/article headline, if available.
        source_url: Final resolved URL after any HTTP redirects.
    """

    title: str | None = None
    source_url: str


class PredictionResponse(BaseModel):
    """Output schema returned after sentiment classification.

    Attributes:
        predicted_class: Top predicted sentiment category ('positive', 'negative', 'neutral').
        probabilities: Normalized softmax probability mapping for each class.
        input_type: Source type of input ('text' or 'url').
        warnings: Advisory messages regarding text truncation or chunk-aggregation heuristics.
        model: Hugging Face identifier of the inference model used.
        article: Optional article metadata if classification originated from a URL.
    """

    predicted_class: str
    probabilities: dict[str, float]
    input_type: Literal["text", "url"]
    warnings: list[str] = Field(default_factory=list)
    model: str
    article: ArticleMetadata | None = None


class HealthResponse(BaseModel):
    """Health status schema for server liveness and model readiness probes.

    Attributes:
        status: 'ok' if model is ready for inference; otherwise 'degraded'.
        model_loaded: Boolean flag indicating whether model weights are in memory.
        model: Identifier of configured sentiment classification model.
    """

    status: Literal["ok", "degraded"]
    model_loaded: bool
    model: str


class ModelInfoResponse(BaseModel):
    """Informational schema detailing the active model metadata and operational constraints.

    Attributes:
        model: Hugging Face model identifier.
        supported_labels: List of target classes produced by the model checkpoint.
        max_input_length: Maximum sequence length supported by tokenizer context window.
        model_loaded: True if model weights are presently loaded in memory.
        inference_local: Always True, signifying local on-device CPU execution.
        limitations: Operational constraints, caveats, and heuristics of the model.
    """

    model: str
    supported_labels: list[str]
    max_input_length: int | None
    model_loaded: bool
    inference_local: bool = True
    limitations: list[str]
