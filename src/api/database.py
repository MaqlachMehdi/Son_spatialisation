"""
database.py
-----------
Connexion PostgreSQL (SQLAlchemy async) et fabrique de sessions, utilisées
par l'authentification et, plus tard, par la persistance des scènes.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

_ROOT = Path(__file__).resolve().parents[2]  # src/api/database.py -> racine du repo
load_dotenv(_ROOT / ".env")

DATABASE_URL = os.environ["DATABASE_URL"]

engine = create_async_engine(DATABASE_URL)
async_session_maker = async_sessionmaker(engine, expire_on_commit=False)


class Base(DeclarativeBase):
    """Classe de base déclarative pour tous les modèles ORM."""


async def get_async_session() -> AsyncGenerator[AsyncSession, None]:
    async with async_session_maker() as session:
        yield session