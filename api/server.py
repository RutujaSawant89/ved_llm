"""TrustLM API — FastAPI application entry point."""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from api.routes import config as config_routes
from api.routes import documents, health, model, query, training


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize shared resources on startup."""
    # Lazy-build the pipeline once
    from rag.pipeline import RAGPipeline

    app.state.pipeline = RAGPipeline()
    app.state.train_process = None
    yield


app = FastAPI(title="TrustLM API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/api")
app.include_router(query.router, prefix="/api")
app.include_router(documents.router, prefix="/api")
app.include_router(model.router, prefix="/api")
app.include_router(config_routes.router, prefix="/api")
app.include_router(training.router, prefix="/api")
