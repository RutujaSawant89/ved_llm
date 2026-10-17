# TrustLM Web Interface & Backend

Minimal, dark-themed developer interface for **TrustLM** — custom Transformer language model with RAG retrieval, trust scoring, verification, and training telemetry.

---

## Architecture Overview

```
ved_llm/
├── api/                  # FastAPI backend server
│   ├── server.py         # Main application & CORS
│   ├── schemas.py        # Pydantic request/response schemas
│   ├── requirements.txt  # FastAPI dependencies
│   └── routes/           # REST endpoints
│       ├── health.py     # GET /api/health
│       ├── query.py      # POST /api/query
│       ├── documents.py  # Ingestion & deletion
│       ├── model.py      # Architecture info & checkpoint
│       ├── config.py     # Configuration & generation settings
│       └── training.py   # Subprocess training runner
│
├── ui/                   # Vite + React frontend
│   ├── src/
│   │   ├── components/   # Flat modular UI components
│   │   ├── hooks/        # Reactive hooks (useChat, useDocuments, etc.)
│   │   ├── pages/        # Chat, Knowledge Base, Model, Training
│   │   ├── utils/        # API client & formatters
│   │   └── index.css     # Monochromatic design system tokens
│   └── vite.config.js    # Dev proxy for /api -> localhost:8000
│
└── run_api.py            # Backend startup runner
```

---

## 1. Install Dependencies

In the root directory `ved_llm`:

```bash
# Core TrustLM dependencies (PyTorch, ChromaDB, etc.)
pip install -r requirements.txt

# API backend dependencies (FastAPI, Uvicorn, Multipart)
pip install -r api/requirements.txt
```

*(Or if you use `uv`:)*
```bash
uv pip install -r requirements.txt -r api/requirements.txt
```

---

## 2. Start the Backend API

Run from the root directory `ved_llm`:

```bash
python run_api.py
```

*Or via uvicorn directly:*
```bash
uvicorn api.server:app --reload --host 127.0.0.1 --port 8000
```

- API Server: `http://127.0.0.1:8000`
- Interactive Swagger Docs: `http://127.0.0.1:8000/docs`

---

## 3. Start the Frontend UI

In a separate terminal, navigate to `ui`:

```bash
cd ui
npm install
npm run dev
```

- Web UI: `http://localhost:5173`
- All `/api/*` requests from Vite automatically proxy to `http://127.0.0.1:8000`.
