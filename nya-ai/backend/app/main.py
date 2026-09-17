import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .database import Base, engine
from . import models  # noqa: F401  (ensures models are registered before create_all)
from .routers import cases, documents, analysis, historical

Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Nya(AI) — Justice. Reimagined.",
    description="AI-assisted legal case analysis and decision-support API (PoC).",
    version="0.1.0",
)

origins = os.getenv("CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(cases.router)
app.include_router(documents.router)
app.include_router(analysis.router)
app.include_router(historical.router)


@app.get("/api/health")
def health():
    return {"status": "ok", "product": "Nya(AI)", "tagline": "Justice. Reimagined."}


# Serve the static frontend (single-page app) if present, so the whole
# product can be run from one process for the demo.
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
