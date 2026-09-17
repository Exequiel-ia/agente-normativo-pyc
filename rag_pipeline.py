from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass(frozen=True)
class Chunk:
    text: str
    source: str
    page: int


@dataclass(frozen=True)
class SearchResult:
    text: str
    source: str
    page: int
    score: float


@dataclass
class KnowledgeBase:
    chunks: list[Chunk]
    vectorizer: TfidfVectorizer
    matrix: object
    document_count: int
    page_count: int
    document_pages: dict[str, int]


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split(text: str, max_chars: int = 1800, overlap: int = 250) -> list[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip()
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        if len(paragraph) <= max_chars:
            current = paragraph
        else:
            start = 0
            while start < len(paragraph):
                piece = paragraph[start : start + max_chars]
                chunks.append(piece)
                start += max_chars - overlap
            current = ""
    if current:
        chunks.append(current)
    return [c for c in chunks if len(c) >= 80]


def build_knowledge_base(documents_dir: Path) -> KnowledgeBase:
    chunks: list[Chunk] = []
    document_pages: dict[str, int] = {}
    for pdf_path in sorted(documents_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        pages_with_text = 0
        for page_number, page in enumerate(reader.pages, 1):
            text = _clean(page.extract_text() or "")
            if not text:
                continue
            pages_with_text += 1
            for part in _split(text):
                chunks.append(Chunk(text=part, source=pdf_path.name, page=page_number))
        document_pages[pdf_path.name] = pages_with_text

    if not chunks:
        raise RuntimeError("No se pudo extraer contenido de los documentos PDF.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
        sublinear_tf=True,
    )
    matrix = vectorizer.fit_transform([chunk.text for chunk in chunks])
    return KnowledgeBase(
        chunks=chunks,
        vectorizer=vectorizer,
        matrix=matrix,
        document_count=len(document_pages),
        page_count=sum(document_pages.values()),
        document_pages=document_pages,
    )


def search_knowledge_base(kb: KnowledgeBase, question: str, top_k: int = 5) -> list[SearchResult]:
    query = kb.vectorizer.transform([question])
    scores = (kb.matrix @ query.T).toarray().ravel()
    best = np.argsort(scores)[::-1][:top_k]
    return [
        SearchResult(
            text=kb.chunks[i].text,
            source=kb.chunks[i].source,
            page=kb.chunks[i].page,
            score=float(scores[i]),
        )
        for i in best
    ]
