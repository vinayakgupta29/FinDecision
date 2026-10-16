"""Pytest shared configuration and global test fixtures.

Ensures automated unit and integration tests do not trigger remote network calls
or heavy model downloads.
"""

import pytest

from app.main import settings


@pytest.fixture(autouse=True)
def disable_model_download_during_tests(monkeypatch):
    """Disable eager model loading on FastAPI startup during test sessions.

    Prevents tests from attempting to download weights from Hugging Face Hub.
    """
    monkeypatch.setattr(
        settings, "load_model_on_startup", False
    )
