import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
os.makedirs(DATA_DIR, exist_ok=True)

DEFAULT_DB_PATH = os.path.join(DATA_DIR, "nya.db").replace("\\", "/")
env_url = os.getenv("DATABASE_URL")
if not env_url or env_url.strip() in ("", "sqlite:///./backend/data/nya.db", "sqlite:///./data/nya.db"):
    DATABASE_URL = f"sqlite:///{DEFAULT_DB_PATH}"
elif env_url.startswith("sqlite:///./"):
    # Resolve relative sqlite path to absolute backend directory
    rel_path = env_url[len("sqlite:///./"):]
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # If starts with backend/, strip to avoid double backend
    if rel_path.startswith("backend/"):
        rel_path = rel_path[len("backend/"):]
    DATABASE_URL = f"sqlite:///{os.path.abspath(os.path.join(base_dir, rel_path)).replace(chr(92), '/')}"
else:
    DATABASE_URL = env_url

connect_args = {"check_same_thread": False, "timeout": 60.0} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)

if DATABASE_URL.startswith("sqlite"):
    from sqlalchemy import event
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA busy_timeout=60000")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
