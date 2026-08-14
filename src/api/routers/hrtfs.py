from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ..config import DATASET_DIR, SOFA_PATH

router = APIRouter()


class HrtfAsset(BaseModel):
    id: str
    label: str
    path: str
    active: bool = False


# Liste les .sofa directement sous dataset/ (le HRTF_SOFA_PATH chargé au
# démarrage y pointe par défaut). Ne recurse pas dans Database-Master_V2-1/,
# qui est un jeu de données BRIR distinct, hors périmètre de ce sélecteur.
@router.get("/hrtfs", response_model=list[HrtfAsset])
def list_hrtfs() -> list[HrtfAsset]:
    if not DATASET_DIR.exists():
        return []
    assets = []
    for file in sorted(DATASET_DIR.glob("*.sofa")):
        assets.append(HrtfAsset(id=file.stem, label=file.stem, path=file.name, active=file == SOFA_PATH))
    return assets