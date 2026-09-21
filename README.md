# ⚖️ Nya(AI) — AI Legal Intelligence Platform

> **AI-Assisted Legal Case Analysis & Decision-Support System**  
> Grounded in Real Indian Supreme Court & High Court Jurisprudence with Zero Hallucination Citation Discipline.

---

## 📂 Repository Contents

| Directory | Purpose |
| :--- | :--- |
| 🚀 **[`nya-ai/`](./nya-ai)** | **Core Application Source**: FastAPI backend, Vanilla JS dark-theme UI, vector retrieval, and multi-signal reranker. |
| 📊 **[`dataset - hugging face/`](./dataset%20-%20hugging%20face)** | **Indian Legal Dataset**: Parquet & Excel dumps containing ~12,000 Indian court judgments with citations. |

---

## ⚡ 1-Click Quick Launch

Launch the full platform (backend server + database + web frontend) in one single command:

```bash
cd nya-ai
python run.py
```

✨ **What this does:**
1. 🔍 Sets up the environment and checks SQLite database (**130 precedents, 10,949 chunks indexed**).
2. 🌐 Starts the Uvicorn server on `http://127.0.0.1:8000`.
3. 🚀 **Automatically opens your web browser** directly to the platform!

---

## 📖 Complete Documentation

For architectural deep-dives, API specifications, benchmark results, and setup options, explore:
👉 **[Full Documentation & Guide in nya-ai/README.md](./nya-ai/README.md)**
