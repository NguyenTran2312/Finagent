import re
from typing import TypedDict

PAGE_MARKER = re.compile(r"===== PAGE (\d+) =====")


class PageChunk(TypedDict):
    page_index: int
    start: int
    end: int
    text: str


def chunk_document(source_text: str) -> list[PageChunk]:
    """Split source text into page chunks using its page markers."""
    markers = list(PAGE_MARKER.finditer(source_text))
    if not markers:
        raise ValueError("Document does not contain any page markers.")

    pages: list[PageChunk] = []
    for page_index, marker in enumerate(markers):
        start = marker.end()
        if source_text.startswith("\r\n", start):
            start += 2
        elif source_text.startswith("\n", start) or source_text.startswith("\r", start):
            start += 1

        if page_index + 1 < len(markers):
            end = markers[page_index + 1].start()
        else:
            end = len(source_text)

        pages.append(
            {
                "page_index": page_index,
                "start": start,
                "end": end,
                "text": source_text[start:end],
            }
        )

    return pages
