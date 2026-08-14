"""
schemas/render.py
------------------
Schéma de la requête POST /render : l'état complet de la scène (sources +
trajectoires), tel qu'il existe déjà côté frontend (sceneStore/trajectoryStore).
Mêmes DTO que /workspace (schemas/workspace.py) — une source ou une
trajectoire a une seule représentation entre les deux endpoints, aucun
mapping manuel côté client.
"""

from __future__ import annotations

from .workspace import CamelModel, SourcePayload, TrajectoryPayload


class RenderRequest(CamelModel):
    sources: list[SourcePayload]
    trajectories: list[TrajectoryPayload] = []
