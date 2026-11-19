import os, sys, warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from pathlib import Path
from dotenv import load_dotenv
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

load_dotenv()

from .database import Base, engine, get_db, SessionLocal
from . import models
from .routers import cases, documents, analysis, historical

Base.metadata.create_all(bind=engine)

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(title="NyaAi — Justice. Reimagined.", version="0.1.0", lifespan=lifespan)

@app.middleware("http")
async def add_no_cache_headers(request, call_next):
    response = await call_next(request)
    path = request.url.path
    if path.endswith((".js", ".css", ".html")) or path in ("/", "/index.html"):
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

for r in [cases.router, documents.router, analysis.router, historical.router]:
    app.include_router(r)

try:
    eval_dir = Path(__file__).resolve().parents[2] / "evaluation"
    if str(eval_dir) not in sys.path: sys.path.insert(0, str(eval_dir))
    from eval_router import router as eval_router
    app.include_router(eval_router)
except Exception as _eval_err:
    import warnings; warnings.warn(f"Evaluation router not loaded: {_eval_err}")

@app.get("/api/health")
@app.get("/health")
def health():
    return {
        "status": "ok", "product": "NyaAi", "tagline": "Justice. Reimagined.", "version": "2.0",
        "llm_provider": os.getenv("LLM_PROVIDER", "mock"),
        "embedding_provider": os.getenv("EMBEDDING_PROVIDER", "sentence-transformers"),
    }

@app.get("/api/search")
def search_cases(q: str, limit: int = 10, db: Session = Depends(get_db)):
    from .services.retrieval import retrieve_historical_chunks
    results = retrieve_historical_chunks(db, q, top_k=limit)
    out = []
    for r in results:
        hc = db.get(models.HistoricalCase, r.chunk.historical_case_id) if r.chunk.historical_case_id else None
        out.append({
            "historical_case_id": r.chunk.historical_case_id, "case_name": hc.case_name if hc else "Precedent",
            "court": hc.court if hc else None, "year": hc.year if hc else None, "doc_url": hc.doc_url if hc else None,
            "semantic_score": round(r.semantic_score, 3), "lexical_score": round(r.keyword_score, 3),
            "hybrid_score": round(r.hybrid_score, 3), "excerpt": r.chunk.text[:280],
        })
    return out

@app.get("/api/cases/{case_id}/reranked-precedents")
def reranked_precedents(case_id: str, top_k: int = 8, db: Session = Depends(get_db)):
    from .services.retrieval import retrieve_historical_chunks
    from .services.reranker import rerank_candidates
    case = db.get(models.Case, case_id)
    if not case: raise HTTPException(404, "Case not found")
    q = " ".join(filter(None, [case.facts or "", case.issues or ""]))
    retrieved = retrieve_historical_chunks(db, q, top_k=30)
    return [r.to_dict() for r in rerank_candidates(db, case, retrieved, top_k=top_k)]

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
