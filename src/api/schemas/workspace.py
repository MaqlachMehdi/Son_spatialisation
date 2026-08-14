"""
schemas/workspace.py
---------------------
Schémas Pydantic pour GET/PUT /workspace. Les noms de champs sont en
snake_case côté Python mais exposés en camelCase en JSON (alias_generator)
pour correspondre exactement aux DTO du frontend (types.ts) — aucun mapping
manuel de champ à champ nécessaire côté client.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True, from_attributes=True)


class WaypointPayload(CamelModel):
    azimuth: float
    elevation: float
    distance: float


class SourcePayload(CamelModel):
    id: str
    label: str
    path: str
    azimuth: float
    elevation: float
    distance: float
    gain: float
    color: str
    muted: bool
    model_id: str | None = None
    trajectory_id: str | None = None


class TrajectoryPayload(CamelModel):
    id: str
    name: str
    type: str
    distance: float
    center_azimuth: float
    center_elevation: float
    az_amplitude: float
    el_amplitude: float
    speed: float
    reverse: bool
    axis_azimuth: float
    axis_elevation: float
    offset_azimuth: float
    offset_elevation: float
    offset_distance: float
    points: list[WaypointPayload]


class WorkspacePayload(CamelModel):
    sources: list[SourcePayload]
    trajectories: list[TrajectoryPayload]
