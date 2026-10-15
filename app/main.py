"""FastAPI entry point and HTTP API routing for FinDecision.

Provides REST endpoints for sentiment analysis of text and web articles,
health monitoring, model metadata inspection, and serves the static frontend UI.
"""

import asyncio
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.schemas import (
    ArticleMetadata,
    HealthResponse,
    ModelInfoResponse,
    PredictionResponse,
    TextRequest,
    URLRequest,
)
from app.services.article_service import (
    ArticleFetchError,
    URLRejected,
    fetch_article,
)
from app.services.classification_service import ClassificationService
from app.services.model_service import ModelService


# Configure application logger.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("findecision")

settings = get_settings()

# Initialize core model and classification services.
model_service = ModelService(
    settings.model_id,
    settings.model_revision,
    settings.device,
)

classifier = ClassificationService(
    model_service,
    settings.chunk_chars,
    settings.max_chunks,
)

# Concurrency semaphore: serializes CPU inference to prevent CPU/memory thrashing on laptops.
inference_slots = asyncio.Semaphore(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan context manager.

    Handles eager model loading during startup in a background thread to prevent
    blocking the asyncio event loop.
    """
    if settings.load_model_on_startup:
        try:
            await asyncio.to_thread(model_service.load)
            logger.info("Loaded model: %s", settings.model_id)
        except Exception:
            logger.exception("Model loading failed")

    yield


app = FastAPI(
    title="FinDecision",
    version="1.0.0",
    description=(
        "Local financial sentiment classification. "
        "Sentiment is not a prediction of investment returns."
    ),
    lifespan=lifespan,
)


@app.exception_handler(Exception)
async def handle_unexpected_error(
    request: Request,
    exc: Exception,
):
    """Global catch-all exception handler for unhandled internal server errors.

    Logs error traceback and returns a sanitized JSON 500 response.
    """
    logger.exception("Unexpected application error")

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "internal_error",
                "message": "An unexpected internal error occurred.",
            }
        },
    )


def require_model():
    """Verify that the machine learning model is loaded and ready for inference.

    Raises:
        HTTPException: 503 Service Unavailable if the model is not loaded.
    """
    if not model_service.loaded:
        raise HTTPException(
            status_code=503,
            detail={
                "code": "model_unavailable",
                "message": (
                    "The model is not loaded. Check the model ID, "
                    "network access, and server logs."
                ),
            },
        )


def run_classification(
    text: str,
    input_type: str,
    article: ArticleMetadata | None = None,
) -> PredictionResponse:
    """Execute synchronous text chunking and sentiment classification.

    Validates text length limits, runs inference, and packages results into PredictionResponse.

    Args:
        text: Normalized text to classify.
        input_type: Indicator of source format ('text' or 'url').
        article: Optional metadata for article inputs.

    Returns:
        PredictionResponse: Sentiment prediction, probability mapping, and warnings.

    Raises:
        HTTPException: 413 if text is too large, 422 if invalid text, 500 on failure.
    """
    if len(text) > settings.max_text_chars:
        raise HTTPException(
            status_code=413,
            detail={
                "code": "text_too_large",
                "message": "Text exceeds the configured size limit.",
            },
        )

    require_model()

    try:
        predicted, probabilities, warnings = classifier.classify(
            text
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "invalid_text",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        logger.exception("Classification failed")
        raise HTTPException(
            status_code=500,
            detail={
                "code": "classification_failed",
                "message": "Classification failed.",
            },
        ) from exc

    return PredictionResponse(
        predicted_class=predicted,
        probabilities=probabilities,
        input_type=input_type,
        warnings=warnings,
        model=settings.model_id,
        article=article,
    )


@app.get("/", include_in_schema=False)
def frontend():
    """Serve the single-page HTML frontend application."""
    return FileResponse(
        "app/static/index.html",
        media_type="text/html",
    )


# Mount static assets directory for CSS and JavaScript bundles.
app.mount(
    "/static",
    StaticFiles(directory="app/static"),
    name="static",
)


@app.get("/health", response_model=HealthResponse)
def health():
    """Liveness and readiness check endpoint.

    Returns 'ok' if the classification model is loaded and ready, otherwise 'degraded'.
    """
    loaded = model_service.loaded

    return HealthResponse(
        status="ok" if loaded else "degraded",
        model_loaded=loaded,
        model=settings.model_id,
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    """Inspect active model metadata, supported sentiment labels, and system caveats."""
    return ModelInfoResponse(
        model=settings.model_id,
        supported_labels=model_service.labels,
        max_input_length=model_service.max_length,
        model_loaded=model_service.loaded,
        inference_local=True,
        limitations=[
            "Softmax scores are not necessarily calibrated.",
            "Sentiment classification does not predict investment returns.",
            "Long-text probabilities use length-weighted chunk aggregation.",
            "URL extraction may fail on paywalled or dynamic pages.",
        ],
    )


@app.post("/predict/text", response_model=PredictionResponse)
async def predict_text(payload: TextRequest):
    """Classify financial sentiment directly from submitted text content.

    Queues request through the inference semaphore and offloads CPU forward passes
    to an asynchronous thread.
    """
    text = payload.text.strip()

    if not text:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "empty_text",
                "message": "Text must not be empty.",
            },
        )

    # Acquire inference slot to protect CPU from concurrent exhaustion.
    async with inference_slots:
        return await asyncio.to_thread(
            run_classification, text, "text"
        )


@app.post("/predict/url", response_model=PredictionResponse)
async def predict_url(payload: URLRequest):
    """Fetch article content from a web URL, extract text, and classify sentiment.

    Performs SSRF security checks, streams and limits payload size, extracts text
    via Trafilatura, and classifies sentiment.
    """
    try:
        article_text, title, final_url = await asyncio.to_thread(
            fetch_article,
            str(payload.url),
            settings,
        )
    except URLRejected as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "code": "url_rejected",
                "message": str(exc),
            },
        ) from exc
    except ArticleFetchError as exc:
        raise HTTPException(
            status_code=422,
            detail={
                "code": "article_unavailable",
                "message": str(exc),
            },
        ) from exc

    article = ArticleMetadata(
        title=title,
        source_url=final_url,
    )

    # Acquire inference slot for CPU model execution.
    async with inference_slots:
        return await asyncio.to_thread(
            run_classification,
            article_text,
            "url",
            article,
        )
