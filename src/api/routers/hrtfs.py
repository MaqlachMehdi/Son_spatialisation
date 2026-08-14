from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from hrtf import HRTF, HRTFInterpolator

from ..config import DATASET_DIR

router = APIRouter()


class HrtfAsset(BaseModel):
    id: str
    label: str
    path: str
    active: bool = False


class SetActiveHrtfRequest(BaseModel):
    id: str


def _list_sofa_files() -> list[Path]:
    # Ne recurse pas dans Database-Master_V2-1/, qui est un jeu de données
    # BRIR distinct, hors périmètre de ce sélecteur.
    if not DATASET_DIR.exists():
        return []
    return sorted(DATASET_DIR.glob("*.sofa"))


@router.get("/hrtfs", response_model=list[HrtfAsset])
def list_hrtfs(http_request: Request) -> list[HrtfAsset]:
    active_path = http_request.app.state.active_hrtf_path
    return [
        HrtfAsset(id=file.stem, label=file.stem, path=file.name, active=file == active_path)
        for file in _list_sofa_files()
    ]


@router.put("/hrtfs/active", response_model=HrtfAsset)
def set_active_hrtf(payload: SetActiveHrtfRequest, http_request: Request) -> HrtfAsset:
    """
    Change la HRTF utilisée par /render pour tous les rendus suivants.

    On ne construit jamais le chemin fichier depuis payload.id directement
    (risque de path traversal) : on ne retient que le membre de
    _list_sofa_files() dont le nom correspond, sinon 404.
    """
    target = next((f for f in _list_sofa_files() if f.stem == payload.id), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"HRTF introuvable : {payload.id}")

    print(f"[api] Changement de HRTF active -> {target} …")
    # Un rendu déjà en cours garde sa propre référence locale à l'ancien
    # interpolator (capturée en début de fonction dans routers/render.py) :
    # remplacer ces deux attributs n'interrompt rien en vol.
    http_request.app.state.interpolator = HRTFInterpolator(HRTF.from_sofa(target), verbose=False)
    http_request.app.state.active_hrtf_path = target
    print("[api] HRTFInterpolator prêt.")

    return HrtfAsset(id=target.stem, label=target.stem, path=target.name, active=True)
