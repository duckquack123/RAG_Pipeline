from dataclasses import asdict, dataclass
from pathlib import Path
import re
from collections.abc import Iterable

try:
    import pymupdf
except ImportError:  # Older environments expose the same library as fitz.
    import fitz as pymupdf


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str
    source: str
    page: int
    start: int
    end: int
    title: str | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def extract_chunks(pdf_path: str | Path, chunk_size: int = 900,
                   overlap: int = 120) -> list[Chunk]:
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller")

    path = Path(pdf_path).resolve()
    chunks: list[Chunk] = []
    document = pymupdf.open(path)
    try:
        for page_number, page in enumerate(document, start=1):
            raw = page.get_text()
            text = _clean(raw)
            if len(text) < 20:
                continue
            paragraphs = [part.strip() for part in re.split(r"\n\s*\n", raw)
                          if part.strip()]
            if not paragraphs:
                paragraphs = [text]
            page_offset = 0
            current = ""
            chunk_start = 0
            for paragraph in paragraphs:
                value = _clean(paragraph)
                if not value:
                    continue
                if len(value) > chunk_size:
                    if current:
                        chunks.append(Chunk(
                            id=f"{path.name}:p{page_number}:c{len(chunks)}",
                            text=current, source=str(path), page=page_number,
                            start=chunk_start, end=chunk_start + len(current),
                        ))
                        current = ""
                    for segment_start in range(0, len(value), chunk_size - overlap):
                        segment = value[segment_start:segment_start + chunk_size]
                        chunks.append(Chunk(
                            id=f"{path.name}:p{page_number}:c{len(chunks)}",
                            text=segment, source=str(path), page=page_number,
                            start=page_offset + segment_start,
                            end=page_offset + segment_start + len(segment),
                        ))
                    page_offset += len(value) + 1
                    continue
                candidate = f"{current} {value}".strip()
                if current and len(candidate) > chunk_size:
                    chunks.append(Chunk(
                        id=f"{path.name}:p{page_number}:c{len(chunks)}",
                        text=current, source=str(path), page=page_number,
                        start=chunk_start, end=chunk_start + len(current),
                    ))
                    overlap_text = current[-overlap:] if overlap else ""
                    chunk_start = max(page_offset - len(overlap_text), 0)
                    current = f"{overlap_text} {value}".strip()
                else:
                    current = candidate
                page_offset += len(value) + 1
            if current:
                chunks.append(Chunk(
                    id=f"{path.name}:p{page_number}:c{len(chunks)}",
                    text=current, source=str(path), page=page_number,
                    start=chunk_start, end=chunk_start + len(current),
                ))
    finally:
        document.close()
    if not chunks:
        raise ValueError(f"No text was extracted from {path}")
    return chunks


def extract_documents(paths: Iterable[str | Path], chunk_size: int = 900,
                      overlap: int = 120) -> list[Chunk]:
    """Extract a collection of PDFs into one metadata-aware corpus."""
    pdf_paths = [Path(path) for path in paths]
    if not pdf_paths:
        raise ValueError("At least one PDF path is required")
    chunks: list[Chunk] = []
    for path in pdf_paths:
        chunks.extend(extract_chunks(path, chunk_size, overlap))
    return chunks
