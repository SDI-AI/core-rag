# core-rag

Air-gapped retrieval-augmented generation for a Getac rugged tablet.

A question is answered from documents on the machine. Search is a local SQLite index. When a GGUF is present, llama.cpp writes the answer from the passages it just retrieved. The app does not download a model and does not call a cloud API. Outbound connections from the Python process are refused, and llama.cpp is started with `--offline`.

## Stack

| Piece | What it does |
| --- | --- |
| Python 3.11+ | Ingest, search, and the query page. No third-party packages. |
| SQLite FTS5 | BM25 retrieval over `.jsonl`, `.txt`, and `.md`. |
| llama.cpp | Local generation. CPU build. `--offline` so the process stays on local files. |
| Qwen2.5-3B-Instruct Q4_K_M | The generation model. 1.8 GB. Linked or copied to `models/llm.gguf`. |
| Tablet browser | The page at `http://127.0.0.1:7860`, with sample queries. |

`config.json` is the tablet profile: 4 threads, 2048 tokens of context, 256 new tokens, KV cache in q8, 4 passages, `gpu_layers` 0. `config.local.json` overrides that file and is gitignored.

## It fits a Getac

A current F110 or UX10 can run this stack. Both are Windows 11 tablets with Intel CPUs and Intel graphics, and neither has an NVIDIA GPU.

| Tablet | CPU | RAM | Graphics |
| --- | --- | --- | --- |
| F110 | 13th-gen Core i5/i7 U-series, 15 W | 8, 16, or 32 GB | Intel UHD |
| UX10 | Core Ultra Series 2 | 16 or 32 GB soldered | Intel Arc |

The model file is 1.8 GB. Context is capped at 2048 with a q8 KV cache, and generation uses 4 CPU threads with no GPU offload. That fits an 8 GB F110 if other apps are closed, and it is comfortable in 16 GB. The UX10 neural processor is not used. A 7B Q4 file is about 4.7 GB and will be slow on a 15 W chip. A 70B model does not fit.

The same Qwen2.5-3B Q4 file was run on the development machine with llama.cpp and returned a cited answer from the demo corpus. The tablet copy is the Windows CPU build of `llama-cli.exe` plus this GGUF, with `gpu_layers` left at 0.

## Run on a Getac

Stage the folder on a connected PC, then carry it over on USB.

1. Install Python 3.11 or newer on the tablet image. The Windows embeddable package is enough.
2. Copy a Windows CPU build of [llama.cpp](https://github.com/ggml-org/llama.cpp/releases). The file you need is `llama-cli.exe`. Put it on `PATH`, or pass `--llama-bin`.
3. Download [Qwen2.5-3B-Instruct-Q4_K_M.gguf](https://huggingface.co/bartowski/Qwen2.5-3B-Instruct-GGUF/resolve/main/Qwen2.5-3B-Instruct-Q4_K_M.gguf) (1.8 GB) and save it as `models/llm.gguf`. The running app never fetches this.
4. Copy this repo, including `data/fake_cui.jsonl` if you want the demo corpus.

On the tablet, turn off Wi-Fi, Bluetooth, and the cellular radio. Then:

```bat
core-rag.bat doctor
core-rag.bat ingest data\fake_cui.jsonl
core-rag.bat serve
```

Open `http://127.0.0.1:7860` in the browser on the tablet. The page has three sample queries. Other machines cannot open that address.

If `llama-cli.exe` is not on `PATH`:

```bat
core-rag.bat serve --llama-bin C:\llama\llama-cli.exe --llm models\llm.gguf
```

`doctor` warns until both the binary and `models/llm.gguf` are in place. Search still works without them: the page shows the matching passages and says generation is unavailable.

## Run on a Linux machine

From this directory, for a rehearsal before the tablet copy:

```sh
python -m unittest discover -s tests -v
python -m core_rag doctor
python -m core_rag ingest data/fake_cui.jsonl
python -m core_rag serve
```

`./core-rag.sh` is the same entry point. The page listens on `127.0.0.1:7860`.

To open it from another computer on the same network:

```sh
python -m core_rag serve --host 0.0.0.0 --llm models/llm.gguf --llama-bin /path/to/llama-cli
```

Add `--gpu-layers 99` only on a machine whose llama.cpp build can use an NVIDIA GPU. Leave it off for the tablet. Outbound connections stay blocked either way.

A query from the shell:

```sh
python -m core_rag query "Which operations have a HIGH threat and Extract VIP?"
```

## Commands

| Command | What it does |
| --- | --- |
| `doctor` | Python, FTS5, index size, model, RAM, and whether the page is local or on the LAN. |
| `ingest PATH` | Index a `.jsonl`, `.txt`, or `.md` file, or a directory of them. Re-ingesting a file replaces that file's chunks. `--reset` clears the index first. |
| `query TEXT` | Retrieve passages and, if a model is loaded, answer from them. `--no-generate` skips the model. `--k` is 1 to 20. |
| `serve` | Query page on port 7860. `--host 0.0.0.0` opens it on the LAN. `--port` changes the port. |

JSONL records use `id` and `text` (`content` or `body` are accepted). The committed demo file is 10,000 synthetic records.

## How an answer is built

The question is reduced to words that appear in the index. A trailing "s" is removed when that is what makes the match, so "operations" finds "operation". Those words are required together. If nothing matches, the most common leftover word is dropped and the search is tried again.

The model is told to use only the retrieved passages and to say when they do not contain the answer. Passages are data, not instructions. If the index has no match, the model is not called.

Retrieval is lexical BM25, not a neural embedding model. An embedding model would be a second file in RAM. BM25 needs no model and runs on an 8 GB tablet.

## Demo corpus

`data/fake_cui.jsonl` is synthetic text for a dry run. It is not real CUI.

Regenerating it needs the `faker` package, which the tablet does not need:

```sh
python scripts/make_fake_cui.py
```

The script writes `data/fake_cui.jsonl` from any working directory and uses a fixed seed.

## Tests

```sh
python -m unittest discover -s tests -v
```

The tests do not need a model or a network.

## Layout

```text
core_rag/          application
config.json        tablet defaults (gpu_layers 0)
data/              demo JSONL
index/             core-rag.sqlite, created by ingest, not committed
models/llm.gguf    Qwen2.5-3B-Instruct Q4_K_M, not committed
results/           scratch during generation
```
