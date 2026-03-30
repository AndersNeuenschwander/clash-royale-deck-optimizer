import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, DeclarativeBase
from dotenv import load_dotenv

load_dotenv()

# In development we use a local Postgres URL from .env
# In production Railway injects DATABASE_URL automatically
DATABASE_URL = os.getenv("DATABASE_URL", "")

if not DATABASE_URL:
    raise ValueError("DATABASE_URL not set in environment")

# SQLAlchemy engine — the connection pool to the database
# The intuition: this is like the API client for the database,
# we create it once and reuse it across all requests
engine = create_engine(DATABASE_URL)

# SessionLocal is a factory that creates database sessions
# Each request gets its own session — like a transaction scope
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


def get_db():
    """
    FastAPI dependency that provides a database session per request.
    The try/finally ensures the session is always closed after the
    request completes, even if an error occurs — preventing connection leaks.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()