"""
models/workspace.py
--------------------
Sources et trajectoires persistées par utilisateur — un espace de travail
unique par compte pour l'instant (pas encore de scènes multiples nommées,
voir le plan de mise en ligne). trajectory_id sur Source est une référence
"molle" (pas de contrainte FK) : elle reflète juste le lien déjà utilisé
côté frontend, sans imposer d'ordre d'insertion entre les deux tables.
"""

from __future__ import annotations

import uuid

from sqlalchemy import JSON, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column

from ..database import Base


class Trajectory(Base):
    __tablename__ = "trajectories"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    id: Mapped[str] = mapped_column(primary_key=True)

    name: Mapped[str]
    type: Mapped[str]
    distance: Mapped[float]
    center_azimuth: Mapped[float]
    center_elevation: Mapped[float]
    az_amplitude: Mapped[float]
    el_amplitude: Mapped[float]
    speed: Mapped[float]
    reverse: Mapped[bool]
    axis_azimuth: Mapped[float]
    axis_elevation: Mapped[float]
    offset_azimuth: Mapped[float]
    offset_elevation: Mapped[float]
    offset_distance: Mapped[float]
    points: Mapped[list] = mapped_column(JSON, default=list)


class Source(Base):
    __tablename__ = "sources"

    user_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), primary_key=True)
    id: Mapped[str] = mapped_column(primary_key=True)

    label: Mapped[str]
    path: Mapped[str]
    azimuth: Mapped[float]
    elevation: Mapped[float]
    distance: Mapped[float]
    gain: Mapped[float]
    color: Mapped[str]
    muted: Mapped[bool]
    model_id: Mapped[str | None]
    trajectory_id: Mapped[str | None]
