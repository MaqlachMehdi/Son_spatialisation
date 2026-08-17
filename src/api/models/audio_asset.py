"""
models/audio_asset.py
----------------------
Sons importés par un utilisateur (upload WAV/MP3/...), convertis en WAV et
stockés sous UPLOADS_DIR/<user_id>/<id>.wav — voir routers/sounds.py pour
l'upload et routers/render.py pour la résolution sécurisée à la lecture.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import DateTime, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class AudioAsset(Base):
    __tablename__ = "audio_assets"

    # Id serveur (uuid4 hex) : contrairement à Source/Trajectory (créés
    # côté client, PK composite user_id+id), un asset n'existe qu'après
    # upload côté serveur — un id simple globalement unique suffit.
    id: Mapped[str] = mapped_column(primary_key=True)
    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    label: Mapped[str]
    storage_path: Mapped[str]  # relatif à UPLOADS_DIR, ex: "<user_id>/<id>.wav"
    # DateTime(timezone=True) explicite : doit matcher le type de colonne
    # créé par la migration (TIMESTAMP WITH TIME ZONE), sinon asyncpg refuse
    # d'y écrire un datetime.now(timezone.utc) (aware).
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
