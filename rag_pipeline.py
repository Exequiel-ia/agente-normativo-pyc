from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer


CODE_PATTERN = re.compile(r"\b[A-Z]{1,4}[\s-]?\d{1,4}\b", re.IGNORECASE)


@dataclass(frozen=True)
class Chunk:
    text: str
    indexed_text: str
    source: str
    page: int
    codes: tuple[str, ...] = ()
    section_code: str = ""
    metadata: dict[str, str] | None = None


@dataclass(frozen=True)
class SearchResult:
    text: str
    source: str
    page: int
    score: float
    codes: tuple[str, ...] = ()
    section_code: str = ""


@dataclass
class KnowledgeBase:
    chunks: list[Chunk]
    word_vectorizer: TfidfVectorizer
    word_matrix: Any
    char_vectorizer: TfidfVectorizer
    char_matrix: Any
    document_count: int
    page_count: int
    document_pages: dict[str, int]
    document_metadata: dict[str, dict[str, str]]


@dataclass(frozen=True)
class DocumentInspection:
    name: str
    size_bytes: int
    sha256: str
    total_pages: int
    pages_with_text: int
    characters: int
    codes: tuple[str, ...]
    warnings: tuple[str, ...]


def normalize_code(value: str) -> str:
    return re.sub(r"[\s-]+", "", value).upper()


def extract_codes(text: str) -> tuple[str, ...]:
    return tuple(sorted({normalize_code(match.group(0)) for match in CODE_PATTERN.finditer(text)}))


def _clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _split(text: str, max_chars: int = 1600, overlap: int = 300) -> list[str]:
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
                chunks.append(paragraph[start : start + max_chars])
                start += max_chars - overlap
            current = ""
    if current:
        chunks.append(current)
    return [chunk for chunk in chunks if len(chunk) >= 80]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_pdf(path: Path) -> DocumentInspection:
    warnings: list[str] = []
    try:
        reader = PdfReader(str(path))
    except Exception as exc:
        raise ValueError(f"No se pudo abrir el PDF: {exc}") from exc

    texts: list[str] = []
    pages_with_text = 0
    for page in reader.pages:
        text = _clean(page.extract_text() or "")
        texts.append(text)
        if text:
            pages_with_text += 1
    if not pages_with_text:
        warnings.append("El documento no contiene texto extraíble; puede requerir OCR.")
    elif pages_with_text < len(reader.pages):
        warnings.append(f"{len(reader.pages) - pages_with_text} página(s) no contienen texto extraíble.")

    all_text = "\n".join(texts)
    return DocumentInspection(
        name=path.name,
        size_bytes=path.stat().st_size,
        sha256=file_sha256(path),
        total_pages=len(reader.pages),
        pages_with_text=pages_with_text,
        characters=len(all_text),
        codes=extract_codes(all_text),
        warnings=tuple(warnings),
    )


def load_metadata(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def save_metadata(path: Path, metadata: dict[str, dict[str, str]]) -> None:
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")


def build_knowledge_base(
    documents_dir: Path,
    metadata_path: Path | None = None,
    max_chars: int = 1600,
    overlap: int = 300,
) -> KnowledgeBase:
    chunks: list[Chunk] = []
    document_pages: dict[str, int] = {}
    document_metadata = load_metadata(metadata_path) if metadata_path else {}

    for pdf_path in sorted(documents_dir.glob("*.pdf")):
        reader = PdfReader(str(pdf_path))
        pages_with_text = 0
        current_section_code = ""
        doc_metadata = document_metadata.get(pdf_path.name, {})
        metadata_prefix = " | ".join(f"{k}: {v}" for k, v in doc_metadata.items() if v)
        for page_number, page in enumerate(reader.pages, 1):
            text = _clean(page.extract_text() or "")
            if not text:
                continue
            pages_with_text += 1
            heading = re.search(r"(?:ARCHIVO|Archivo)\s+([A-Z]{1,4}[\s-]?\d{1,4})", text)
            if heading:
                current_section_code = normalize_code(heading.group(1))
            page_codes = set(extract_codes(text))
            if current_section_code:
                page_codes.add(current_section_code)
            for part in _split(text, max_chars=max_chars, overlap=overlap):
                codes = tuple(sorted(page_codes | set(extract_codes(part))))
                prefix = " | ".join(filter(None, [metadata_prefix, " ".join(codes)]))
                chunks.append(
                    Chunk(
                        text=part,
                        indexed_text=f"{prefix}\n{part}".strip(),
                        source=pdf_path.name,
                        page=page_number,
                        codes=codes,
                        section_code=current_section_code,
                        metadata=doc_metadata,
                    )
                )
        document_pages[pdf_path.name] = pages_with_text

    if not chunks:
        raise RuntimeError("No se pudo extraer contenido de los documentos PDF.")

    indexed_texts = [chunk.indexed_text for chunk in chunks]
    word_vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
        min_df=1, max_df=0.99, sublinear_tf=True,
    )
    char_vectorizer = TfidfVectorizer(
        lowercase=True, strip_accents="unicode", analyzer="char_wb",
        ngram_range=(2, 5), min_df=1, sublinear_tf=True,
    )
    word_matrix = word_vectorizer.fit_transform(indexed_texts)
    char_matrix = char_vectorizer.fit_transform(indexed_texts)
    return KnowledgeBase(
        chunks=chunks,
        word_vectorizer=word_vectorizer,
        word_matrix=word_matrix,
        char_vectorizer=char_vectorizer,
        char_matrix=char_matrix,
        document_count=len(document_pages),
        page_count=sum(document_pages.values()),
        document_pages=document_pages,
        document_metadata=document_metadata,
    )


def search_knowledge_base(kb: KnowledgeBase, question: str, top_k: int = 5) -> list[SearchResult]:
    word_query = kb.word_vectorizer.transform([question])
    char_query = kb.char_vectorizer.transform([question])
    word_scores = (kb.word_matrix @ word_query.T).toarray().ravel()
    char_scores = (kb.char_matrix @ char_query.T).toarray().ravel()
    scores = (0.65 * word_scores) + (0.35 * char_scores)

    query_codes = set(extract_codes(question))
    if query_codes:
        normalized_question = question.lower()
        schedule_intent = any(
            term in normalized_question
            for term in ("periodicidad", "cuándo", "cuando", "plazo", "enviar", "envío", "envio", "fecha")
        )
        section_bonus = 0.45 if schedule_intent else 1.20
        for index, chunk in enumerate(kb.chunks):
            matches = query_codes.intersection(chunk.codes)
            if chunk.section_code in query_codes:
                # El fragmento pertenece al archivo consultado, no solo lo menciona.
                scores[index] += section_bonus
            elif matches:
                # Referencias cruzadas y catálogos siguen siendo útiles, pero secundarias.
                if schedule_intent and "reglas" in chunk.source.lower():
                    reference_bonus = 0.35
                elif schedule_intent:
                    reference_bonus = 0.05
                else:
                    reference_bonus = 0.25
                scores[index] += reference_bonus + (0.05 * len(matches))

    best = np.argsort(scores)[::-1][:top_k]
    return [
        SearchResult(
            text=kb.chunks[index].text,
            source=kb.chunks[index].source,
            page=kb.chunks[index].page,
            score=float(scores[index]),
            codes=kb.chunks[index].codes,
            section_code=kb.chunks[index].section_code,
        )
        for index in best
    ]
