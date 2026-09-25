from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core_rag import __version__
from core_rag.airgap import AirgapError, install_airgap
from core_rag.config import Config, ConfigError, load_config, override
from core_rag.doctor import checks, worst
from core_rag.documents import DocumentError
from core_rag.generate import GenerateError
from core_rag.server import make_server
from core_rag.service import Engine, QuestionError
from core_rag.store import StoreError


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--root", type=Path, default=None, help="repo directory (default: this checkout)")
    parser.add_argument("--llama-bin", default=None, help="path to llama-cli or llama-cli.exe")
    parser.add_argument("--llm", default=None, help="path to the GGUF model")
    parser.add_argument("--gpu-layers", type=int, default=None, help="llama.cpp -ngl value (default 0)")
    parser.add_argument("--index", default=None, help="path to the sqlite index")


def _engine(args: argparse.Namespace) -> tuple[Engine, Config]:
    cfg = load_config(args.root)
    override(
        cfg,
        llama_bin=args.llama_bin,
        llm_path=args.llm,
        gpu_layers=args.gpu_layers,
        index_path=args.index,
        port=getattr(args, "port", None),
        host=getattr(args, "host", None),
    )
    return Engine(cfg), cfg


def _ipv4_addresses() -> list[str]:
    """LAN addresses, read from the interfaces. No name lookup."""
    try:
        import fcntl
        import socket
        import struct
    except ImportError:
        return []
    net = Path("/sys/class/net")
    if not net.is_dir():
        return []
    found: list[str] = []
    for iface in sorted(net.iterdir()):
        if iface.name == "lo":
            continue
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            packed = fcntl.ioctl(
                sock.fileno(),
                0x8915,
                struct.pack("256s", iface.name.encode()[:15]),
            )
        except OSError:
            continue
        finally:
            sock.close()
        found.append(socket.inet_ntoa(packed[20:24]))
    return found


def _cmd_ingest(args: argparse.Namespace) -> int:
    engine, _cfg = _engine(args)
    try:
        engine.ingest(Path(args.path), reset=args.reset)
    finally:
        engine.close()
    return 0


def _cmd_query(args: argparse.Namespace) -> int:
    engine, _cfg = _engine(args)
    try:
        answer = engine.ask(args.question, k=args.k, generate=not args.no_generate)
    finally:
        engine.close()
    if answer.notice:
        print(answer.notice)
    for number, passage in enumerate(answer.passages, 1):
        print(f"[{number}] {passage.doc_id}  {passage.source}")
        print(passage.text)
        print()
    if answer.generated and answer.text:
        print("Answer:")
        print(answer.text)
    elif answer.text:
        print(answer.text)
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    engine, cfg = _engine(args)
    try:
        httpd = make_server(engine)
    except Exception:
        engine.close()
        raise
    host, port = httpd.server_address[:2]
    if host in {"0.0.0.0", "::"}:
        print(f"core-rag {__version__} listening on all interfaces, port {port}", flush=True)
        for address in _ipv4_addresses():
            print(f"  http://{address}:{port}", flush=True)
        print("Outbound connections are still blocked.", flush=True)
    else:
        print(f"core-rag {__version__} at http://{host}:{port}", flush=True)
        print("This address is on this machine only.", flush=True)
    print(f"Model: {cfg.llm_path}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped", flush=True)
    finally:
        httpd.server_close()
        engine.close()
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    engine, _cfg = _engine(args)
    try:
        items = checks(engine)
    finally:
        engine.close()
    for item in items:
        print(f"{item.level:4}  {item.name:8}  {item.detail}")
    return 1 if worst(items) == "fail" else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="core-rag",
        description=(
            "Air-gapped RAG for a Getac tablet. Search is a local SQLite index. "
            "Answers come from a local GGUF. The process refuses non-loopback network."
        ),
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    ingest = sub.add_parser("ingest", parents=[], help="index .jsonl, .txt, or .md files")
    _add_common(ingest)
    ingest.add_argument("path", help="file or directory to index")
    ingest.add_argument("--reset", action="store_true", help="delete the index before ingesting")
    ingest.set_defaults(func=_cmd_ingest)

    query = sub.add_parser("query", help="search the local index and optionally generate an answer")
    _add_common(query)
    query.add_argument("question")
    query.add_argument("--k", type=int, default=None, help="passages to retrieve (1-20)")
    query.add_argument("--no-generate", action="store_true", help="return passages only")
    query.set_defaults(func=_cmd_query)

    serve = sub.add_parser("serve", help="open the query page")
    _add_common(serve)
    serve.add_argument("--port", type=int, default=None)
    serve.add_argument(
        "--host",
        default=None,
        help="127.0.0.1 (default) or 0.0.0.0 to open the page on the LAN",
    )
    serve.set_defaults(func=_cmd_serve)

    doctor = sub.add_parser("doctor", help="check Python, the index, the model, and RAM")
    _add_common(doctor)
    doctor.set_defaults(func=_cmd_doctor)
    return parser


def main(argv: list[str] | None = None) -> int:
    install_airgap()
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except (
        AirgapError,
        ConfigError,
        DocumentError,
        GenerateError,
        QuestionError,
        StoreError,
        OSError,
    ) as exc:
        print(exc, file=sys.stderr)
        return 1
