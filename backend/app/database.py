import os
from sqlalchemy import create_engine, event
from sqlalchemy.orm import sessionmaker, declarative_base

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_DB_PATH = os.path.join(DATA_DIR, "nya.db").replace("\\", "/")
env_url = (os.getenv("DATABASE_URL") or "").strip()
if not env_url or env_url in ("sqlite:///./backend/data/nya.db", "sqlite:///./data/nya.db"):
    DATABASE_URL = f"sqlite:///{DEFAULT_DB_PATH}"
elif env_url.startswith("sqlite:///./"):
    rel = env_url[len("sqlite:///./"):].removeprefix("backend/")
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATABASE_URL = f"sqlite:///{os.path.abspath(os.path.join(base, rel)).replace(chr(92), '/')}"
else:
    DATABASE_URL = env_url

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False, "timeout": 60.0} if DATABASE_URL.startswith("sqlite") else {})

if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _set_pragmas(conn, _):
        cur = conn.cursor()
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA synchronous=NORMAL")
        cur.execute("PRAGMA busy_timeout=60000")
        cur.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try: yield db
    finally: db.close()
