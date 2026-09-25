"""Tablet-sized defaults.

A Getac F110 or UX10 is a Windows 11 tablet: Intel CPU, Intel graphics,
usually 16 GB of RAM (8 GB on some F110 configs, 32 GB optional). A 70B
model does not fit. Defaults keep the context, thread count, and GPU
offload inside that box. Override them in config.json when the tablet
has the RAM.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

_LOOPBACK = {"127.0.0.1", "localhost", "::1"}
# All-interfaces bind is opt-in for testing from another machine.
_LAN = {"0.0.0.0", "::"}


class ConfigError(ValueError):
    pass


@dataclass
class Config:
    root: Path
    llama_bin: str
    llm_path: Path
    gpu_layers: int
    context_tokens: int
    max_new_tokens: int
    threads: int
    temperature: float
    top_k: int
    host: str
    port: int
    chunk_chars: int
    chunk_overlap: int
    timeout_seconds: int
    index_path: Path


def _as_path(root: Path, value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _int(data: dict, key: str, default: int) -> int:
    value = data.get(key, default)
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be an integer") from exc


def _float(data: dict, key: str, default: float) -> float:
    value = data.get(key, default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{key} must be a number") from exc


def override(cfg: Config, **changes) -> Config:
    for key, value in changes.items():
        if value is None:
            continue
        if key in {"llm_path", "index_path"}:
            value = _as_path(cfg.root, value)
        setattr(cfg, key, value)
    _check(cfg)
    return cfg


def _check(cfg: Config) -> None:
    if cfg.host not in _LOOPBACK and cfg.host not in _LAN:
        raise ConfigError(
            f"host must be 127.0.0.1, localhost, ::1, or 0.0.0.0 (got {cfg.host}). "
            "Use 0.0.0.0 only when another machine on the network should open the page."
        )
    if not 1 <= cfg.port <= 65535:
        raise ConfigError("port must be between 1 and 65535")
    if cfg.threads < 1:
        raise ConfigError("threads must be at least 1")
    if cfg.context_tokens < 512:
        raise ConfigError("context_tokens must be at least 512")
    if not 16 <= cfg.max_new_tokens <= 1024:
        raise ConfigError("max_new_tokens must be between 16 and 1024")
    if not 0 <= cfg.temperature <= 2:
        raise ConfigError("temperature must be between 0 and 2")
    if not 1 <= cfg.top_k <= 20:
        raise ConfigError("top_k must be between 1 and 20")
    if cfg.chunk_chars < 200:
        raise ConfigError("chunk_chars must be at least 200")
    if cfg.chunk_overlap < 0 or cfg.chunk_overlap >= cfg.chunk_chars:
        raise ConfigError("chunk_overlap must be smaller than chunk_chars")
    if cfg.gpu_layers < 0:
        raise ConfigError("gpu_layers must be 0 or more")
    if cfg.timeout_seconds < 1:
        raise ConfigError("timeout_seconds must be at least 1")


def _read_json(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"{path} must contain a JSON object")
    return data


def load_config(root: Path | None = None) -> Config:
    root = (root or REPO_ROOT).resolve()
    data: dict = {}
    for name in ("config.json", "config.local.json"):
        path = root / name
        if path.is_file():
            data.update(_read_json(path))

    env_map = {
        "CORE_RAG_LLAMA_BIN": "llama_bin",
        "CORE_RAG_LLM": "llm_path",
        "CORE_RAG_GPU_LAYERS": "gpu_layers",
        "CORE_RAG_HOST": "host",
        "CORE_RAG_PORT": "port",
        "CORE_RAG_THREADS": "threads",
    }
    for env_name, key in env_map.items():
        if os.environ.get(env_name):
            data[key] = os.environ[env_name]

    cfg = Config(
        root=root,
        llama_bin=str(data.get("llama_bin", "llama-cli")),
        llm_path=_as_path(root, data.get("llm_path", "models/llm.gguf")),
        gpu_layers=_int(data, "gpu_layers", 0),
        context_tokens=_int(data, "context_tokens", 2048),
        max_new_tokens=_int(data, "max_new_tokens", 256),
        threads=_int(data, "threads", 4),
        temperature=_float(data, "temperature", 0.2),
        top_k=_int(data, "top_k", 4),
        host=str(data.get("host", "127.0.0.1")),
        port=_int(data, "port", 7860),
        chunk_chars=_int(data, "chunk_chars", 1200),
        chunk_overlap=_int(data, "chunk_overlap", 200),
        timeout_seconds=_int(data, "timeout_seconds", 180),
        index_path=_as_path(root, data.get("index_path", "index/core-rag.sqlite")),
    )
    _check(cfg)
    return cfg
