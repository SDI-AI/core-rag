import json
import socket
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

from core_rag.airgap import AirgapError, install_airgap, uninstall_airgap
from core_rag.chunk import chunk_text
from core_rag.cli import main
from core_rag.config import ConfigError, load_config
from core_rag.documents import DocumentError, load_documents
from core_rag.generate import PROMPT_MARK, Generator, extract_answer, render_prompt, subprocess_runner
from core_rag.server import make_server
from core_rag.service import NO_HITS, Engine
from core_rag.store import Index, Passage, fts_tokens


ROOT = Path(__file__).resolve().parents[1]


def _corpus(directory: Path) -> Path:
    path = directory / "notes.jsonl"
    rows = [
        {
            "id": "OP-1",
            "text": "OPERATION NORTH: Extract VIP. Assets: Acme, Globex. Threat: HIGH.",
        },
        {
            "id": "OP-2",
            "text": "OPERATION SOUTH: Secure perimeter. Assets: Initech. Threat: LOW.",
        },
        {
            "id": "OP-3",
            "text": "Logistics note: fuel convoy delayed at the depot.",
        },
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


class AirgapTests(unittest.TestCase):
    def setUp(self):
        install_airgap()

    def tearDown(self):
        uninstall_airgap()

    def test_blocks_name_lookup_and_outbound_connect(self):
        with self.assertRaises(AirgapError):
            socket.getaddrinfo("example.com", 80)
        sock = socket.socket()
        try:
            with self.assertRaises(AirgapError):
                sock.connect(("8.8.8.8", 53))
        finally:
            sock.close()

    def test_allows_loopback(self):
        socket.getaddrinfo("127.0.0.1", 9)
        socket.getaddrinfo("0.0.0.0", 9)
        sock = socket.socket()
        sock.settimeout(0.2)
        try:
            try:
                sock.connect(("127.0.0.1", 1))
            except AirgapError:
                self.fail("loopback connect was blocked")
            except OSError:
                pass
        finally:
            sock.close()

    def test_package_does_not_import_an_http_client(self):
        for path in (ROOT / "core_rag").glob("*.py"):
            text = path.read_text(encoding="utf-8")
            self.assertNotIn("urllib.request", text, path.name)
            self.assertNotIn("import requests", text, path.name)
            self.assertNotIn("import httpx", text, path.name)

    def test_page_has_no_remote_url(self):
        html = (ROOT / "core_rag" / "page.html").read_text(encoding="utf-8")
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)
        self.assertIn("/api/query", html)
        self.assertIn("Search", html)
        self.assertIn("Extract VIP, HIGH threat", html)
        self.assertIn("Which operations have a HIGH threat and Extract VIP?", html)
        self.assertIn("data-q=", html)


class ChunkAndConfigTests(unittest.TestCase):
    def test_short_text_is_one_chunk(self):
        self.assertEqual(chunk_text("  Extract VIP.  ", 100, 20), ["Extract VIP."])

    def test_long_text_advances(self):
        parts = chunk_text("a" * 5000, 1000, 200)
        self.assertGreaterEqual(len(parts), 5)
        self.assertTrue(all(len(part) <= 1000 for part in parts))

    def test_rejects_public_host(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "config.json").write_text('{"host": "8.8.8.8"}\n', encoding="utf-8")
            with self.assertRaises(ConfigError):
                load_config(root)
            (root / "config.json").write_text('{"host": "0.0.0.0"}\n', encoding="utf-8")
            self.assertEqual(load_config(root).host, "0.0.0.0")

    def test_bad_jsonl_names_the_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.jsonl"
            path.write_text("{not json}\n", encoding="utf-8")
            with self.assertRaises(DocumentError) as caught:
                load_documents(path)
            self.assertIn(":1:", str(caught.exception))


class IndexTests(unittest.TestCase):
    def test_search_replace_and_punctuation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _corpus(root)
            index = Index(root / "index.sqlite")
            try:
                docs = load_documents(corpus)
                index.ingest_documents(str(corpus.resolve()), docs, 1200, 200)
                hits = index.search("Extract VIP", 4)
                self.assertEqual([hit.doc_id for hit in hits], ["OP-1"])
                self.assertGreater(hits[0].score, 0)
                self.assertEqual(index.search('HIGH" OR ( *', 4)[0].doc_id, "OP-1")
                self.assertEqual(index.search("the", 4), [])
                self.assertEqual(fts_tokens("***"), [])
                wordy = index.search(
                    "Which operations have a HIGH threat and Extract VIP?", 4
                )
                self.assertEqual(wordy[0].doc_id, "OP-1")

                corpus.write_text(
                    json.dumps(
                        {
                            "id": "OP-9",
                            "text": "OPERATION NORTH: Extract VIP. Threat: MEDIUM.",
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
                index.ingest_documents(
                    str(corpus.resolve()), load_documents(corpus), 1200, 200
                )
                self.assertEqual(index.stats()["chunks"], 1)
                self.assertEqual(index.search("Globex", 4), [])
                self.assertEqual(index.search("MEDIUM", 4)[0].doc_id, "OP-9")
            finally:
                index.close()


class GenerateTests(unittest.TestCase):
    def test_prompt_and_argv_stay_offline(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = load_config(root)
            passage = Passage("OP-1", str(root / "notes.jsonl"), "Threat: HIGH.", 1.0)
            prompt = render_prompt("What is the threat?", [passage], cfg)
            self.assertIn("Threat: HIGH.", prompt)
            self.assertIn("[1]", prompt)
            self.assertTrue(prompt.rstrip().endswith(PROMPT_MARK))

            seen = {}

            def runner(argv, timeout):
                seen["argv"] = argv
                seen["prompt"] = Path(argv[argv.index("-f") + 1]).read_text(encoding="utf-8")
                seen["timeout"] = timeout
                return (
                    0,
                    "Loading model...\n> Question:\n"
                    + seen["prompt"]
                    + "The threat is HIGH [1]\n\n[ Prompt: 1 t/s | Generation: 2 t/s ]\nExiting...\n",
                    "",
                )

            gen = Generator(cfg, runner=runner)
            gen.availability = lambda: (True, "ready")
            gen.binary = lambda: "llama-cli"
            text = gen.answer("What is the threat?", [passage])
            self.assertEqual(text, "The threat is HIGH [1]")
            self.assertIn("--offline", seen["argv"])
            self.assertIn("--no-show-timings", seen["argv"])
            self.assertIn("--log-disable", seen["argv"])
            self.assertEqual(seen["argv"][seen["argv"].index("-ngl") + 1], "0")
            self.assertIn("Threat: HIGH.", seen["prompt"])
            self.assertFalse((root / "results" / "prompt.txt").exists())

    def test_extract_answer_without_banner(self):
        self.assertEqual(
            extract_answer("The threat is HIGH [1]\n[ Prompt: 1 t/s | Generation: 2 t/s ]\nExiting..."),
            "The threat is HIGH [1]",
        )
        banner = (
            "build      : test\navailable commands:\n\n> "
            + ("Question: " + "passage " * 40)
            + " ... (truncated)\n\nThe threat is HIGH [1]\n\nExiting...\n"
        )
        self.assertEqual(extract_answer(banner), "The threat is HIGH [1]")

    def test_runner_timeout(self):
        from core_rag.generate import GenerateError

        with self.assertRaises(GenerateError):
            subprocess_runner([ "python3", "-c", "import time; time.sleep(5)" ], 1)


class CliAndServerTests(unittest.TestCase):
    def test_cli_query_without_a_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _corpus(root)
            self.assertEqual(main(["ingest", str(corpus), "--root", str(root)]), 0)
            self.assertEqual(
                main(["query", "Extract VIP", "--root", str(root), "--no-generate"]),
                0,
            )
            self.assertEqual(main(["query", "no-such-token-zzz", "--root", str(root)]), 0)
            code = main(["doctor", "--root", str(root)])
            self.assertEqual(code, 0)

    def test_server_query_page(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            corpus = _corpus(root)
            cfg = load_config(root)
            engine = Engine(cfg)
            engine.ingest(corpus)
            httpd = make_server(engine, port=0)
            thread = threading.Thread(target=httpd.serve_forever, daemon=True)
            thread.start()
            host, port = httpd.server_address[:2]
            try:
                with self.assertRaises(AirgapError):
                    make_server(engine, host="203.0.113.10", port=0)
                lan = make_server(engine, host="0.0.0.0", port=0)
                lan.server_close()
                page = urllib.request.urlopen(f"http://{host}:{port}/", timeout=5)
                html = page.read().decode("utf-8")
                self.assertIn("core-rag", html)
                self.assertNotIn("https://", html)
                health = json.loads(
                    urllib.request.urlopen(f"http://{host}:{port}/api/health", timeout=5).read()
                )
                self.assertGreaterEqual(health["chunks"], 2)
                self.assertFalse(health["model_ready"])
                body = json.dumps({"question": "Extract VIP"}).encode("utf-8")
                req = urllib.request.Request(
                    f"http://{host}:{port}/api/query",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                payload = json.loads(urllib.request.urlopen(req, timeout=5).read())
                self.assertEqual(payload["passages"][0]["doc_id"], "OP-1")
                self.assertFalse(payload["generated"])
                self.assertIn("No local generation model", payload["notice"])
                missing = json.dumps({"question": "zzzz-not-a-token"}).encode("utf-8")
                req = urllib.request.Request(
                    f"http://{host}:{port}/api/query",
                    data=missing,
                    headers={"Content-Type": "application/json"},
                )
                empty = json.loads(urllib.request.urlopen(req, timeout=5).read())
                self.assertEqual(empty["answer"], NO_HITS)
            finally:
                httpd.shutdown()
                httpd.server_close()
                engine.close()

    def test_query_rejects_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = load_config(Path(tmp))
            engine = Engine(cfg)
            try:
                with self.assertRaises(Exception):
                    engine.ask("   ")
            finally:
                engine.close()


if __name__ == "__main__":
    unittest.main()
