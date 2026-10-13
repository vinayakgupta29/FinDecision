"""Text chunking and normalization service for long document processing.

Splits lengthy financial reports and articles into manageable chunks that fit
within transformer context windows while preserving sentence boundaries where possible.
"""

from dataclasses import dataclass


@dataclass
class ChunkedText:
    """Container holding normalized text chunks and advisory warnings.

    Attributes:
        chunks: List of split text strings ready for tokenization and model inference.
        warnings: List of advisory warnings detailing chunking heuristics or truncation.
    """

    chunks: list[str]
    warnings: list[str]


def split_text(
    text: str,
    chunk_chars: int,
    max_chunks: int,
) -> ChunkedText:
    """Split normalized text into sentence-aware chunks up to a maximum chunk count.

    Normalizes whitespace and checks length against chunk_chars. If the text exceeds
    the limit, it searches backward from each window boundary for clean break points
    (such as sentence endings or whitespace) in the latter half of the window.

    Args:
        text: Raw input financial text or article body.
        chunk_chars: Maximum character capacity per chunk (e.g. 2,000 characters).
        max_chunks: Upper ceiling on total number of chunks to evaluate (e.g. 12).

    Returns:
        ChunkedText: An object containing the list of text chunks and any operational warnings.
    """
    # Collapse repetitive whitespace, newlines, and tabs into clean single spaces.
    text = " ".join(text.split())

    if not text:
        return ChunkedText([], [])

    # If within single-chunk budget, return directly with no warnings.
    if len(text) <= chunk_chars:
        return ChunkedText([text], [])

    chunks: list[str] = []
    cursor = 0

    # Iteratively extract bounded windows up to max_chunks.
    while cursor < len(text) and len(chunks) < max_chunks:
        end = min(cursor + chunk_chars, len(text))

        if end < len(text):
            # Attempt to break on natural sentence or word delimiters within the chunk.
            boundary = max(
                text.rfind(". ", cursor, end),
                text.rfind("; ", cursor, end),
                text.rfind(" ", cursor, end),
            )

            # Only accept the delimiter boundary if it is in the second half of the chunk
            # to avoid generating unnecessarily tiny fragments.
            if boundary > cursor + chunk_chars // 2:
                end = boundary + 1

        piece = text[cursor:end].strip()

        if piece:
            chunks.append(piece)

        # Advance cursor ensuring forward progress
        cursor = max(end, cursor + 1)

    warnings = [
        f"Long text was divided into {len(chunks)} chunks. "
        "The length-weighted score is not a calibrated "
        "article-level probability."
    ]

    # Flag if text remained beyond the max_chunks budget
    if cursor < len(text):
        warnings.append(
            f"Input was truncated: only the first {len(chunks)} "
            "chunks were evaluated."
        )

    return ChunkedText(chunks, warnings)
