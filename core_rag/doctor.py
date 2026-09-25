from __future__ import annotations

import os
import sys
from dataclasses import dataclass

from core_rag.service import Engine


@dataclass
class Check:
    level: str
    name: str
    detail: str


def total_ram_bytes() -> int | None:
    if os.name == "nt":
        return _windows_ram_bytes()
    try:
        return int(os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _windows_ram_bytes() -> int | None:
    import ctypes

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = MEMORYSTATUSEX()
    status.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return int(status.ullTotalPhys)


def _ram_detail(ram: int | None, model_bytes: int | None) -> tuple[str, str]:
    if ram is None:
        return "warn", "could not read RAM. Use a 3B Q4 model and context 2048."
    gib = ram / (1024**3)
    if gib < 7.5:
        advice = "under 8 GB. Use a 1B-3B Q4 model, context 2048, and close other apps."
    elif gib < 14:
        advice = "8-16 GB class. A 3B Q4 model is the comfortable default. A 7B Q4 will be tight."
    elif gib < 28:
        advice = (
            "16-32 GB class. A 3B Q4 model is the responsive choice. "
            "A 7B Q4 can fit, but a 15 W tablet CPU will be slow."
        )
    else:
        advice = (
            "A 7B-8B Q4 fits in memory. "
            "On a Getac U-series chip, prefer a 3B model if you want answers in under a minute."
        )
    level = "pass"
    if model_bytes is not None and model_bytes > ram * 0.45:
        level = "warn"
        advice += " The model file is large for this machine."
    return level, f"{gib:.1f} GB. {advice}"


def checks(engine: Engine) -> list[Check]:
    cfg = engine.config
    found: list[Check] = []
    if sys.version_info < (3, 11):
        found.append(Check("fail", "python", f"{sys.version.split()[0]} is older than 3.11"))
    else:
        found.append(Check("pass", "python", sys.version.split()[0]))

    try:
        stats = engine.index.stats()
        found.append(
            Check(
                "pass" if stats["chunks"] else "warn",
                "index",
                f"{stats['chunks']} chunks from {stats['sources']} files at {cfg.index_path}",
            )
        )
    except Exception as exc:  # store already raised a clear error on open
        found.append(Check("fail", "index", str(exc)))

    ready, reason = engine.generator.availability()
    found.append(Check("pass" if ready else "warn", "model", reason))

    model_bytes = cfg.llm_path.stat().st_size if cfg.llm_path.is_file() else None
    level, detail = _ram_detail(total_ram_bytes(), model_bytes)
    found.append(Check(level, "memory", detail))
    if cfg.host in {"0.0.0.0", "::"}:
        bind_detail = (
            f"UI is listening on all interfaces, port {cfg.port}. "
            "Outbound connects are still refused."
        )
    else:
        bind_detail = f"UI is limited to {cfg.host}:{cfg.port}. Outbound connects are refused."
    found.append(Check("pass", "bind", bind_detail))
    found.append(
        Check(
            "pass",
            "cpu",
            f"{cfg.threads} threads, context {cfg.context_tokens}, gpu_layers {cfg.gpu_layers}",
        )
    )
    return found


def worst(items: list[Check]) -> str:
    if any(item.level == "fail" for item in items):
        return "fail"
    if any(item.level == "warn" for item in items):
        return "warn"
    return "pass"
