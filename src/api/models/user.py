"""
models/user.py
--------------
Modèle ORM de la table `users`. id / email / hashed_password / is_active /
is_verified / is_superuser viennent du mixin fastapi-users ; on ajoute
seulement ce dont l'app a besoin en plus.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi_users.db import SQLAlchemyBaseUserTableUUID
from sqlalchemy import DateTime
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class User(SQLAlchemyBaseUserTableUUID, Base):
    __tablename__ = "users"

    display_name: Mapped[str | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )