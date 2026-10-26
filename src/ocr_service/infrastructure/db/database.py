"""SQLAlchemy engine and session management with SQLite WAL mode."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from ocr_service.config import Settings
from ocr_service.infrastructure.db.models import Base


def create_db_engine(settings: Settings) -> Engine:
    """Create a SQLAlchemy engine configured for multi-process SQLite WAL concurrency."""
    db_url = settings.database_url or f"sqlite:///{settings.temp_storage_path / 'ocr_state.sqlite3'}"
    if db_url.startswith("sqlite:///"):
        db_file = Path(db_url.replace("sqlite:///", "", 1))
        db_file.parent.mkdir(parents=True, exist_ok=True)

    connect_args: dict[str, Any] = {}
    if db_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False, "timeout": 30.0}

    engine = create_engine(
        db_url,
        connect_args=connect_args,
        pool_pre_ping=True,
        future=True,
    )

    if db_url.startswith("sqlite"):

        @event.listens_for(engine, "connect")
        def _set_sqlite_pragma(dbapi_connection: Any, _connection_record: Any) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA journal_mode=WAL;")
            cursor.execute("PRAGMA synchronous=NORMAL;")
            cursor.execute("PRAGMA busy_timeout=15000;")
            cursor.execute("PRAGMA foreign_keys=ON;")
            cursor.close()

    Base.metadata.create_all(bind=engine)
    return engine


class DatabaseSessionManager:
    """Manage database sessions for API and Worker processes."""

    def __init__(self, settings: Settings) -> None:
        self.engine = create_db_engine(settings)
        self._session_factory = sessionmaker(
            bind=self.engine,
            autocommit=False,
            autoflush=False,
            expire_on_commit=False,
        )

    @contextmanager
    def session(self) -> Generator[Session, None, None]:
        sess: Session = self._session_factory()
        try:
            yield sess
            sess.commit()
        except Exception:
            sess.rollback()
            raise
        finally:
            sess.close()

    def dispose(self) -> None:
        self.engine.dispose()
