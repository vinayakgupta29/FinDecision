"""Security regression tests for Server-Side Request Forgery (SSRF) mitigations.

Ensures private IP ranges, cloud metadata services, loopback interfaces,
non-HTTP schemes, and embedded credentials in URLs are rejected.
"""

import pytest

from app.services.article_service import (
    URLRejected,
    validate_public_url,
)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "http://localhost/",
        "http://127.0.0.1/",
        "http://169.254.169.254/latest/meta-data/",
        "http://[::1]/",
    ],
)
def test_unsafe_urls_are_rejected(url):
    """Verify that private, local, and metadata endpoints are blocked by SSRF validation."""
    with pytest.raises(URLRejected):
        validate_public_url(url)


def test_url_credentials_are_rejected():
    """Verify that URLs containing embedded basic auth credentials are systematically rejected."""
    with pytest.raises(URLRejected):
        validate_public_url(
            "https://user:password@example.com/article"
        )


def test_non_http_scheme_is_rejected():
    """Verify that unsupported protocol schemes (such as FTP) raise URLRejected."""
    with pytest.raises(URLRejected):
        validate_public_url("ftp://example.com/article")
