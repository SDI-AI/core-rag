from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

SUFFIXES = {".jsonl", ".txt", ".md", ".text"}


class DocumentError(ValueError):
    pass


@dataclass
class Document:
    doc_id: str
    source: str
    text: str


def iter_files(path: Path) -> list[Path]:
    path = path.resolve()
    if path.is_file():
        if path.suffix.lower() not in SUFFIXES:
            raise DocumentError(
                f"unsupported file type {path.suffix or '(none)'}: {path}. "
                "Use .jsonl, .txt, or .md."
            )
        return [path]
    if not path.is_dir():
        raise DocumentError(f"not found: {path}")
    files = [
        item
        for item in sorted(path.rglob("*"))
        if item.is_file() and not item.name.startswith(".") and item.suffix.lower() in SUFFIXES
    ]
    if not files:
        raise DocumentError(f"no .jsonl, .txt, or .md files in {path}")
    return files


def load_documents(path: Path) -> list[Document]:
    path = path.resolve()
    if path.suffix.lower() == ".jsonl":
        return _load_jsonl(path)
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        raise DocumentError(f"{path} is empty")
    return [Document(doc_id=path.stem, source=str(path), text=text)]


def _load_jsonl(path: Path) -> list[Document]:
    docs: list[Document] = []
    source = str(path)
    with path.open(encoding="utf-8") as handle:
        for lineno, line in enumerate(handle, 1):
            raw = line.strip()
            if not raw:
                continue
            try:
                item = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise DocumentError(f"{path}:{lineno}: {exc}") from exc
            if not isinstance(item, dict):
                raise DocumentError(f"{path}:{lineno}: each line must be a JSON object")
            text = item.get("text", item.get("content", item.get("body")))
            if not isinstance(text, str) or not text.strip():
                raise DocumentError(f"{path}:{lineno}: missing text")
            doc_id = item.get("id", item.get("doc_id"))
            if doc_id is None:
                doc_id = f"{path.stem}-{lineno}"
            docs.append(Document(doc_id=str(doc_id), source=source, text=text))
    if not docs:
        raise DocumentError(f"{path} has no documents")
    return docs
