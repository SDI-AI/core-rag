from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

from core_rag.config import Config
from core_rag.documents import iter_files, load_documents
from core_rag.generate import Generator
from core_rag.store import Index, Passage

NO_HITS = "No matching passages in the local index."
PASSAGES_ONLY = (
    "No local generation model is loaded. Showing matching passages from the tablet index."
)


class QuestionError(ValueError):
    pass


@dataclass
class Answer:
    question: str
    passages: list[Passage]
    generated: bool
    text: str | None
    notice: str | None

    def as_dict(self) -> dict:
        return {
            "question": self.question,
            "answer": self.text,
            "generated": self.generated,
            "notice": self.notice,
            "passages": [
                {
                    "doc_id": passage.doc_id,
                    "source": passage.source,
                    "text": passage.text,
                    "score": passage.score,
                }
                for passage in self.passages
            ],
        }


def clean_question(question: str) -> str:
    text = question.strip()
    if not text:
        raise QuestionError("empty question")
    if len(text) > 2000:
        raise QuestionError("question is longer than 2000 characters")
    return text


class Engine:
    def __init__(self, config: Config):
        self.config = config
        self.index = Index(config.index_path)
        self.generator = Generator(config)
        self._lock = threading.Lock()

    def close(self) -> None:
        self.index.close()

    def ingest(self, path: Path, reset: bool = False) -> int:
        files = iter_files(path)
        total = 0
        with self._lock:
            if reset:
                self.index.clear()
            for file in files:
                docs = load_documents(file)
                source = str(file.resolve())
                total += self.index.ingest_documents(
                    source,
                    docs,
                    self.config.chunk_chars,
                    self.config.chunk_overlap,
                )
                print(f"indexed {file} ({len(docs)} documents)", flush=True)
        print(f"index has {self.index.stats()['chunks']} chunks", flush=True)
        return total

    def ask(self, question: str, k: int | None = None, generate: bool = True) -> Answer:
        question = clean_question(question)
        limit = k or self.config.top_k
        with self._lock:
            passages = self.index.search(question, limit)
            if not passages:
                return Answer(question, [], False, NO_HITS, None)
            if not generate:
                return Answer(question, passages, False, None, None)
            ready, reason = self.generator.availability()
            if not ready:
                return Answer(question, passages, False, None, PASSAGES_ONLY + f" ({reason})")
            text = self.generator.answer(question, passages)
            return Answer(question, passages, True, text, None)
