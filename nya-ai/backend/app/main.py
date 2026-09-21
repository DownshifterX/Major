import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .database import Base, engine, get_db
from . import models  # noqa: F401  (ensures models are registered before create_all)
from .routers import cases, documents, analysis, historical
from fastapi import Depends
from sqlalchemy.orm import Session

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
@app.get("/health")
def health():
    return {
        "status": "ok",
        "product": "Nya(AI)",
        "tagline": "Justice. Reimagined.",
        "version": "2.0",
        "llm_provider": os.getenv("LLM_PROVIDER", "mock"),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "sentence-transformers"),
    }


@app.get("/api/search")
def search_cases(q: str, limit: int = 10, db: Session = Depends(get_db)):
    """Lexical & semantic search across historical cases and statutes."""
    from .services.retrieval import retrieve_historical_chunks
    from . import models as _models
    results = retrieve_historical_chunks(db, q, top_k=limit)
    output = []
    for r in results:
        chunk = r.chunk
        hc = None
        if chunk.historical_case_id:
            hc = db.query(_models.HistoricalCase).get(chunk.historical_case_id)
        output.append({
            "historical_case_id": chunk.historical_case_id,
            "case_name": hc.case_name if hc else "Precedent",
            "court": hc.court if hc else None,
            "year": hc.year if hc else None,
            "doc_url": hc.doc_url if hc else None,
            "semantic_score": round(r.semantic_score, 3),
            "lexical_score": round(r.keyword_score, 3),
            "hybrid_score": round(r.hybrid_score, 3),
            "excerpt": chunk.text[:280],
        })
    return output


@app.get("/api/cases/{case_id}/reranked-precedents")
def reranked_precedents(case_id: str, top_k: int = 8, db: Session = Depends(get_db)):
    """Returns multi-signal reranked precedent candidates with transparent score breakdown."""
    from . import models as _models
    from .services.retrieval import retrieve_historical_chunks
    from .services.reranker import rerank_candidates
    case = db.get(_models.Case, case_id)
    if not case:
        from fastapi import HTTPException
        raise HTTPException(404, "Case not found")
    query_text = " ".join(filter(None, [case.facts or "", case.issues or ""]))
    retrieved = retrieve_historical_chunks(db, query_text, top_k=30)
    reranked = rerank_candidates(db, case, retrieved, top_k=top_k)
    return [r.to_dict() for r in reranked]



# Serve the static frontend (single-page app) if present, so the whole
# product can be run from one process for the demo.
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
