import pytest
from alembic import command
from alembic.config import Config
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session

from app.config import settings
from app.models import User


@pytest.fixture(scope="session")
def engine():
    # Build the schema through the real migrations, so they are tested too.
    eng = create_engine(settings.test_database_url)
    with eng.begin() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE; CREATE SCHEMA public"))
    cfg = Config("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", settings.test_database_url)
    command.upgrade(cfg, "head")
    yield eng
    eng.dispose()


@pytest.fixture
def db(engine):
    """Each test runs inside a transaction that is rolled back afterwards.

    Code under test may call commit(); with create_savepoint those commits only
    release a SAVEPOINT, and the outer transaction still gets rolled back.
    """
    conn = engine.connect()
    trans = conn.begin()
    session = Session(bind=conn, join_transaction_mode="create_savepoint", expire_on_commit=False)
    yield session
    session.close()
    trans.rollback()
    conn.close()


@pytest.fixture
def make_user(db):
    counter = 0

    def _make(**kw) -> User:
        nonlocal counter
        counter += 1
        user = User(
            username=kw.pop("username", f"u{counter}"),
            email=kw.pop("email", f"u{counter}@example.com"),
            password_hash="x",
            **kw,
        )
        db.add(user)
        db.flush()
        return user

    return _make
