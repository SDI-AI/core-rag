[![@SDI_AI](https://img.shields.io/badge/%40SDI__AI-70B%20RAG-blue?style=flat&logo=x&logoColor=white)](https://x.com/SDI_AI) [![GitHub](https://img.shields.io/badge/GitHub-SDI--AI-black?style=flat&logo=github)](https://github.com/SDI-AI)

# core-rag
**Air-gapped 70B RAG engine on DGX Spark — the backbone of classified-rag, rail-rag, fin-rag, health-rag.**

> “One codebase. Four industries. Zero cloud. $1M+ brand.” — @SDI_AI

---

## 🚀 Features
- **70B Llama-3** (GGUF) — full GPU offload via NVLink  
- **FAISS-GPU** — 1M docs indexed in <30 min  
- **0.8s query latency** — no internet, no leaks  
- **Fine-tune in <12 hrs** — LoRA on DGX  
- **Auto-benchmark → X post** — `benchmark.py`

## ⚡ Stack
```text
PyTorch | Llama.cpp | FAISS-GPU | Flower | NVLink
