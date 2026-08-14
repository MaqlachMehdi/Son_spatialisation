"""
routers/workspace.py
---------------------
Persistance des sources et trajectoires de l'utilisateur connecté.

Pas de CRUD granulaire (POST /sources, PATCH /trajectories/{id}, ...) : le
frontend envoie tout son état local d'un coup (debounced côté client) et le
serveur remplace entièrement les lignes existantes de l'utilisateur. Plus
simple qu'une synchronisation fine, largement suffisant pour un espace de
travail unique par compte.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import current_active_user
from ..database import get_async_session
from ..models import Source as SourceModel
from ..models import Trajectory as TrajectoryModel
from ..models import User
from ..schemas import SourcePayload, TrajectoryPayload, WorkspacePayload

router = APIRouter()


@router.get("/workspace", response_model=WorkspacePayload)
async def get_workspace(
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> WorkspacePayload:
    sources = (
        (await session.execute(select(SourceModel).where(SourceModel.user_id == user.id)))
        .scalars()
        .all()
    )
    trajectories = (
        (await session.execute(select(TrajectoryModel).where(TrajectoryModel.user_id == user.id)))
        .scalars()
        .all()
    )
    return WorkspacePayload(
        sources=[SourcePayload.model_validate(s) for s in sources],
        trajectories=[TrajectoryPayload.model_validate(t) for t in trajectories],
    )


@router.put("/workspace", status_code=204)
async def put_workspace(
    payload: WorkspacePayload,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    await session.execute(delete(SourceModel).where(SourceModel.user_id == user.id))
    await session.execute(delete(TrajectoryModel).where(TrajectoryModel.user_id == user.id))

    for t in payload.trajectories:
        session.add(
            TrajectoryModel(
                user_id=user.id,
                id=t.id,
                name=t.name,
                type=t.type,
                distance=t.distance,
                center_azimuth=t.center_azimuth,
                center_elevation=t.center_elevation,
                az_amplitude=t.az_amplitude,
                el_amplitude=t.el_amplitude,
                speed=t.speed,
                reverse=t.reverse,
                axis_azimuth=t.axis_azimuth,
                axis_elevation=t.axis_elevation,
                offset_azimuth=t.offset_azimuth,
                offset_elevation=t.offset_elevation,
                offset_distance=t.offset_distance,
                points=[wp.model_dump() for wp in t.points],
            )
        )

    for s in payload.sources:
        session.add(
            SourceModel(
                user_id=user.id,
                id=s.id,
                label=s.label,
                path=s.path,
                azimuth=s.azimuth,
                elevation=s.elevation,
                distance=s.distance,
                gain=s.gain,
                color=s.color,
                muted=s.muted,
                model_id=s.model_id,
                trajectory_id=s.trajectory_id,
            )
        )

    await session.commit()
