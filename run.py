"""NyaAi — One-Click Unified Application Launcher.

Runs the complete end-to-end NyaAi application:
- Verifies python environment and core dependencies
- Ensures the data directory and SQLite database schema are initialized
- Verifies legal precedent corpus presence
- Automatically opens the web browser to the application
- Starts the production-grade Uvicorn server hosting both backend APIs & frontend UI

Usage:
    python run.py
    python run.py --port 8000 --host 127.0.0.1
    python run.py --no-browser
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import sys
import time
import socket
import argparse
import webbrowser
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
FRONTEND_DIR = ROOT_DIR / "frontend"
DATA_DIR = BACKEND_DIR / "data"

# Ensure backend is in python path
sys.path.insert(0, str(BACKEND_DIR))


def check_port_in_use(host: str, port: int) -> bool:
    """Check if the designated port is already bound."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex((host, port)) == 0


def print_banner(host: str, port: int):
    url = f"http://{host}:{port}"
    print("=" * 68)
    print("      NYAAI — Justice. Reimagined.      ")
    print("   AI-Assisted Legal Case Analysis & Decision-Support Platform   ")
    print("=" * 68)
    print(f" [*] Application URL:    {url}")
    print(f" [*] API Docs (Swagger): {url}/docs")
    print(f" [*] Health Endpoint:   {url}/api/health")
    print(f" [*] Legal Research:    {url}#search")
    print(f" [*] Backend root:      {BACKEND_DIR}")
    print(f" [*] Frontend root:     {FRONTEND_DIR}")
    print("=" * 68)


def initialize_environment():
    """Ensure data folders exist and environment variables are loaded."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Change working directory to backend so all relative paths resolve cleanly
    os.chdir(BACKEND_DIR)

    env_file = BACKEND_DIR / ".env"
    if env_file.exists():
        from dotenv import load_dotenv
        load_dotenv(dotenv_path=env_file)
    else:
        example_file = ROOT_DIR / ".env.example"
        if example_file.exists():
            import shutil
            shutil.copy(example_file, env_file)
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_file)

    os.environ.setdefault("EMBEDDING_PROVIDER", "sentence-transformers")
    os.environ.setdefault("LLM_PROVIDER", "mock")


def initialize_database():
    """Pre-initialize database tables and check precedent corpus."""
    try:
        try:
            from app.database import Base, engine, SessionLocal
            from app import models
        except ImportError:
            from backend.app.database import Base, engine, SessionLocal
            from backend.app import models

        Base.metadata.create_all(bind=engine)
        db = SessionLocal()
        try:
            case_count = db.query(models.HistoricalCase).count()
            chunk_count = db.query(models.Chunk).count()
            print(f" [OK] Database ready. Precedent library: {case_count} cases, {chunk_count} indexed chunks.")
        finally:
            db.close()
    except Exception as exc:
        print(f" [!] Warning during database initialization: {exc}")


def launch_browser(url: str, delay: float = 1.2):
    """Open browser in a background thread after a brief delay."""
    import threading

    def _open():
        time.sleep(delay)
        print(f" [*] Opening browser to {url} ...")
        try:
            webbrowser.open(url)
        except Exception:
            pass

    threading.Thread(target=_open, daemon=True).start()


def main():
    parser = argparse.ArgumentParser(description="NyaAi Unified Application Launcher")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--no-browser", action="store_true", help="Do not open web browser automatically")
    args = parser.parse_args()

    # Step 1: Environment Setup
    initialize_environment()

    # Step 2: Database & Corpus check
    initialize_database()

    # Step 3: Print UI Banner
    print_banner(args.host, args.port)

    # Step 4: Check Port Availability
    url = f"http://{args.host}:{args.port}"
    if check_port_in_use(args.host, args.port):
        print(f" [i] A service is already running on port {args.port}.")
        print(f"     Opening browser to {url} now...")
        if not args.no_browser:
            webbrowser.open(url)
        print("     To restart fresh, terminate the existing server and run again.")
        return

    # Step 5: Open Browser
    if not args.no_browser:
        launch_browser(url)

    # Step 6: Start FastAPI application via uvicorn
    import uvicorn
    os.chdir(BACKEND_DIR)
    
    print(" [*] Starting server (Press Ctrl+C to stop)...")
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
