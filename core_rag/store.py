"""SQLite FTS5 index.

One file, no native extension, and synchronous=FULL so a tablet that loses
power mid-write does not keep a half-committed journal.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from core_rag.chunk import chunk_text
from core_rag.documents import Document

_TOKEN = re.compile(r"[0-9A-Za-z]{2,40}")
_STOP = {
    "a",
    "about",
    "also",
    "an",
    "and",
    "are",
    "at",
    "be",
    "been",
    "being",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "her",
    "his",
    "how",
    "in",
    "into",
    "is",
    "its",
    "just",
    "not",
    "of",
    "on",
    "only",
    "or",
    "our",
    "over",
    "should",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "those",
    "to",
    "under",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "whom",
    "whose",
    "why",
    "with",
    "would",
    "you",
    "your",
}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
    body,
    doc_id UNINDEXED,
    source UNINDEXED,
    text UNINDEXED,
    tokenize='unicode61 remove_diacritics 2'
);
"""


class StoreError(RuntimeError):
    pass


@dataclass
class Passage:
    doc_id: str
    source: str
    text: str
    score: float


def fts_tokens(question: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN.finditer(question):
        token = match.group(0).lower()
        if token in _STOP or token in tokens:
            continue
        tokens.append(token)
        if len(tokens) == 12:
            break
    return tokens


class Index:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        # The tablet UI serves a request on another thread than the one that
        # opened the index. The lock below is what makes that safe.
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("DROP TABLE IF EXISTS fts5_probe")
        try:
            self.conn.execute("CREATE VIRTUAL TABLE fts5_probe USING fts5(x)")
        except sqlite3.OperationalError as exc:
            raise StoreError(
                "this Python's sqlite3 was built without FTS5, which core-rag needs for search"
            ) from exc
        self.conn.execute("DROP TABLE fts5_probe")
        # FULL sync: a Getac battery swap should not leave a torn index.
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.executescript(_SCHEMA)
        self.conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema', '1') "
            "ON CONFLICT(key) DO NOTHING"
        )
        self.conn.commit()

    def close(self) -> None:
        with self._lock:
            self.conn.close()

    def stats(self) -> dict[str, int]:
        with self._lock:
            chunks = self.conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
            sources = self.conn.execute(
                "SELECT count(DISTINCT source) AS n FROM chunks"
            ).fetchone()["n"]
        return {"chunks": int(chunks), "sources": int(sources)}

    def clear(self) -> None:
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM chunks")

    def ingest_documents(
        self,
        source: str,
        docs: list[Document],
        chunk_chars: int,
        chunk_overlap: int,
    ) -> int:
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM chunks WHERE source = ?", (source,))
            count = 0
            for doc in docs:
                parts = chunk_text(doc.text, chunk_chars, chunk_overlap)
                for ordinal, part in enumerate(parts):
                    doc_id = doc.doc_id if len(parts) == 1 else f"{doc.doc_id}#{ordinal}"
                    self.conn.execute(
                        "INSERT INTO chunks(body, doc_id, source, text) VALUES (?, ?, ?, ?)",
                        (f"{doc_id}\n{part}", doc_id, source, part),
                    )
                    count += 1
        return count

    def search(self, question: str, k: int) -> list[Passage]:
        tokens = fts_tokens(question)
        if not tokens:
            return []
        # AND the terms that actually occur. Extra words ("which", "operations")
        # would otherwise throw away the documents that match the rest. A wide
        # OR is worse: it returns passages that share one common word.
        with self._lock:
            present = self._terms_in_index(tokens)
            if not present:
                return []
            frequency = {token: self._count_token(token) for token in present}
            while present:
                match = " AND ".join(f'"{token}"' for token in present)
                hits = self._match(match, k)
                if hits:
                    return hits
                present.remove(max(present, key=lambda token: frequency[token]))
        return []

    def _terms_in_index(self, tokens: list[str]) -> list[str]:
        found: list[str] = []
        for token in tokens:
            if self._count_token(token):
                found.append(token)
                continue
            # "OPERATION" is indexed as "operation". Try the same word without
            # a trailing s so "operations" still matches.
            if len(token) > 4 and token.endswith("s") and self._count_token(token[:-1]):
                found.append(token[:-1])
        return found

    def _count_token(self, token: str) -> int:
        row = self.conn.execute(
            "SELECT count(*) AS n FROM chunks WHERE chunks MATCH ?",
            (f'"{token}"',),
        ).fetchone()
        return int(row["n"])

    def _match(self, match: str, k: int) -> list[Passage]:
        try:
            rows = self.conn.execute(
                "SELECT doc_id, source, text, bm25(chunks) AS bm25_score "
                "FROM chunks WHERE chunks MATCH ? ORDER BY bm25(chunks) LIMIT ?",
                (match, k),
            ).fetchall()
        except sqlite3.OperationalError as exc:
            raise StoreError(f"search failed: {exc}") from exc
        return [
            Passage(
                doc_id=row["doc_id"],
                source=row["source"],
                text=row["text"],
                score=round(-float(row["bm25_score"]), 4),
            )
            for row in rows
        ]
