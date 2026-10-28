"""SQLAlchemy engine and session management with SQLite WAL mode."""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sqlalchemy import Engine, create_engine, event, inspect, text
from sqlalchemy.exc import OperationalError
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
    _apply_additive_schema_upgrades(engine)
    return engine


def _apply_additive_schema_upgrades(engine: Engine) -> None:
    """Add columns and indexes introduced after a database file was first created.

    ``create_all`` only creates missing tables, so an existing state database keeps its old
    columns. New columns are always nullable or carry a server default, which makes plain
    ``ALTER TABLE ... ADD COLUMN`` safe. API and worker processes start concurrently, so a
    "duplicate column" race from another process is expected and ignored.
    """
    inspector = inspect(engine)
    for table in Base.metadata.sorted_tables:
        if not inspector.has_table(table.name):
            continue
        existing = {c["name"] for c in inspector.get_columns(table.name)}
        for column in table.columns:
            if column.name in existing:
                continue
            col_type = column.type.compile(dialect=engine.dialect)
            ddl = f'ALTER TABLE "{table.name}" ADD COLUMN "{column.name}" {col_type}'
            if column.server_default is not None:
                default_sql = column.server_default.arg  # type: ignore[attr-defined]
                ddl += f" NOT NULL DEFAULT '{default_sql}'"
            try:
                with engine.begin() as conn:
                    conn.execute(text(ddl))
            except OperationalError as exc:
                if "duplicate column" not in str(exc).lower():
                    raise
        for index in table.indexes:
            try:
                index.create(bind=engine, checkfirst=True)
            except OperationalError as exc:
                if "already exists" not in str(exc).lower():
                    raise


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
