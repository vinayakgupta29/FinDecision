"""Article retrieval and extraction service with robust SSRF mitigations.

Safely fetches web articles from public HTTP/HTTPS URLs, validates against
private network addresses, streams and size-limits responses, and extracts
structured text content using Trafilatura.
"""

import ipaddress
import socket
from urllib.parse import urljoin, urlsplit

import httpx
import trafilatura
from trafilatura.metadata import extract_metadata

from app.config import Settings


class URLRejected(ValueError):
    """Raised when a provided URL violates security constraints or safety policies."""

    pass


class ArticleFetchError(RuntimeError):
    """Raised when an article cannot be fetched, extracted, or parsed."""

    pass


def validate_public_url(url: str) -> str:
    """Validate that a URL points to an accessible, globally routable public internet resource.

    Implements defense-in-depth protections against Server-Side Request Forgery (SSRF):
      1. Scheme constraint: Only 'http' and 'https' are allowed (disallows file://, gopher://, etc.).
      2. Disallows embedded credentials (user:pass@host).
      3. Blocklists local and cloud metadata hostnames (localhost, *.local, metadata.google.internal).
      4. Resolves hostname via DNS and rejects non-global IPs (private, loopback, link-local, multicast).

    Args:
        url: Raw string representation of the target URL.

    Returns:
        str: The validated URL if it passes all security checks.

    Raises:
        URLRejected: If the URL fails scheme, hostname, credential, or IP safety requirements.
    """
    try:
        parsed = urlsplit(url)

        # Enforce scheme restrictions.
        if parsed.scheme.lower() not in {"http", "https"}:
            raise URLRejected("Only HTTP and HTTPS URLs are allowed.")

        # Require a valid hostname.
        if not parsed.hostname:
            raise URLRejected("URL must contain a hostname.")

        # Deny embedded user/password credentials.
        if parsed.username or parsed.password:
            raise URLRejected("Embedded URL credentials are forbidden.")

        host = parsed.hostname.rstrip(".").lower()

        # Reject common cloud and internal local hostnames.
        if (
            host == "localhost"
            or host.endswith((".localhost", ".local"))
            or host == "metadata.google.internal"
        ):
            raise URLRejected("Local hostnames are forbidden.")

        # Attempt to parse as an IP address literal; if not an IP, resolve DNS.
        try:
            addresses = [ipaddress.ip_address(host)]
        except ValueError:
            try:
                results = socket.getaddrinfo(
                    host,
                    parsed.port or (
                        443 if parsed.scheme == "https" else 80
                    ),
                    type=socket.SOCK_STREAM,
                )
            except socket.gaierror as exc:
                raise URLRejected(
                    "Hostname resolution failed."
                ) from exc

            addresses = [
                ipaddress.ip_address(result[4][0])
                for result in results
            ]

        # Enforce that all resolved IP addresses are globally routable.
        if not addresses or any(
            not address.is_global for address in addresses
        ):
            raise URLRejected(
                "Private, reserved, and non-public IP addresses "
                "are forbidden."
            )

        return url

    except URLRejected:
        raise
    except (ValueError, OSError) as exc:
        raise URLRejected("Malformed or unsafe URL.") from exc


def fetch_article(
    url: str,
    settings: Settings,
) -> tuple[str, str | None, str]:
    """Fetch an article over HTTP, follow safe redirects, and extract body text.

    Streams the HTTP response to enforce maximum payload limits before reading
    large bodies into memory. Validates MIME content types and uses Trafilatura
    to extract readable financial text while stripping ads, navigations, and boilerplates.

    Args:
        url: Web URL of the article to retrieve.
        settings: Application Settings containing timeouts, redirect caps, and size limits.

    Returns:
        tuple[str, str | None, str]:
            - article_text: Cleaned text extracted from the webpage.
            - title: Extracted document title if available.
            - final_url: The actual resolved URL following all valid redirects.

    Raises:
        URLRejected: If the initial URL or any subsequent redirect target fails safety checks.
        ArticleFetchError: If the server returns an error, payload exceeds size limits,
            or text cannot be extracted.
    """
    current_url = str(url)

    timeout = httpx.Timeout(
        connect=settings.connect_timeout_seconds,
        read=settings.read_timeout_seconds,
        write=5.0,
        pool=5.0,
    )

    headers = {
        "User-Agent": "FinDecision/1.0",
        "Accept": "text/html,application/xhtml+xml",
    }

    # Follow redirects manually to re-validate each target against SSRF filters.
    for redirect_count in range(settings.max_redirects + 1):
        validate_public_url(current_url)

        try:
            with httpx.Client(
                timeout=timeout,
                follow_redirects=False,
                headers=headers,
                trust_env=False,
            ) as client:
                with client.stream("GET", current_url) as response:
                    # Handle standard HTTP redirect statuses.
                    if response.status_code in {
                        301, 302, 303, 307, 308
                    }:
                        location = response.headers.get("location")

                        if not location:
                            raise ArticleFetchError(
                                "Redirect has no Location header."
                            )

                        if redirect_count >= settings.max_redirects:
                            raise ArticleFetchError(
                                "Maximum redirect count exceeded."
                            )

                        current_url = urljoin(
                            current_url, location
                        )
                        continue

                    # Reject client or server HTTP errors.
                    if response.status_code >= 400:
                        raise ArticleFetchError(
                            f"Article server returned HTTP "
                            f"{response.status_code}."
                        )

                    # Ensure response is an HTML document.
                    content_type = response.headers.get(
                        "content-type", ""
                    ).lower()

                    if not any(
                        kind in content_type
                        for kind in (
                            "text/html",
                            "application/xhtml+xml",
                        )
                    ):
                        raise ArticleFetchError(
                            "The URL did not return HTML."
                        )

                    content_length = response.headers.get(
                        "content-length"
                    )

                    if (
                        content_length
                        and content_length.isdigit()
                        and int(content_length)
                        > settings.max_url_response_bytes
                    ):
                        raise ArticleFetchError(
                            "Article exceeds the response-size limit."
                        )

                    # Stream bytes while enforcing size ceiling to prevent DoS.
                    body = []
                    size = 0

                    for chunk in response.iter_bytes():
                        size += len(chunk)

                        if size > settings.max_url_response_bytes:
                            raise ArticleFetchError(
                                "Article exceeds the response-size limit."
                            )

                        body.append(chunk)

                    html = b"".join(body).decode(
                        response.encoding or "utf-8",
                        errors="replace",
                    )

            # Extract main article body using Trafilatura precision heuristics.
            article_text = trafilatura.extract(
                html,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )

            # Ensure minimum viable text content was obtained.
            if (
                not article_text
                or len(article_text.strip()) < 80
            ):
                raise ArticleFetchError(
                    "Meaningful article text could not be extracted. "
                    "The page may be paywalled, inaccessible, or "
                    "JavaScript-rendered."
                )

            metadata = extract_metadata(html)
            title = metadata.title if metadata else None

            return article_text, title, current_url

        except (URLRejected, ArticleFetchError):
            raise
        except httpx.TimeoutException as exc:
            raise ArticleFetchError(
                "Article request timed out."
            ) from exc
        except httpx.HTTPError as exc:
            raise ArticleFetchError(
                "Unable to fetch the article."
            ) from exc

    raise ArticleFetchError("Unable to resolve article redirects.")
