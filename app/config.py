"""Application configuration module.

Defines environment variables, runtime parameters, and default values
for FinDecision using Pydantic Settings.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global configuration settings for FinDecision.

    Attributes:
        app_name: User-facing title of the application.
        model_id: Hugging Face repository identifier for sentiment model.
        model_revision: Optional immutable commit hash/tag for reproducibility.
        device: PyTorch device ('cpu' or 'cuda').
        max_text_chars: Maximum allowable character length for raw input text.
        chunk_chars: Target character size per text chunk for transformer inputs.
        max_chunks: Maximum number of chunks processed per classification.
        max_url_response_bytes: Maximum allowed byte size for article downloads (5 MB).
        connect_timeout_seconds: Network socket connection timeout in seconds.
        read_timeout_seconds: HTTP response read timeout in seconds.
        max_redirects: Maximum HTTP redirects allowed during article fetching.
        load_model_on_startup: Whether to eagerly load the model during app lifespan startup.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "FinDecision"
    model_id: str = "9mark9/finbert-minilm-sentiment"
    model_revision: str | None = None
    device: str = "cpu"

    max_text_chars: int = 50_000
    chunk_chars: int = 2_000
    max_chunks: int = 12

    max_url_response_bytes: int = 5_000_000
    connect_timeout_seconds: float = 5.0
    read_timeout_seconds: float = 10.0
    max_redirects: int = 3

    load_model_on_startup: bool = True


@lru_cache
def get_settings() -> Settings:
    """Retrieve cached application settings instance.

    Returns:
        Settings: The singleton Settings instance loaded from environment/.env.
    """
    return Settings()
