from __future__ import annotations


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split text into overlapping windows.

    Short records stay one chunk. A break on a newline or sentence is used
    only when it sits in the second half of the window, so a chunk does not
    collapse to a few words.
    """
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be greater than overlap")
    text = text.strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            window = text[start:end]
            break_at = max(window.rfind("\n"), window.rfind(". "))
            if break_at >= size // 2:
                end = start + break_at + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = end - overlap
        if next_start <= start:
            next_start = start + 1
        start = next_start
    return chunks
