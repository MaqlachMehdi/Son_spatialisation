from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from ..config import AUDIO_EXTENSIONS, SOUND_DIR

router = APIRouter()


class SoundAsset(BaseModel):
    id: str
    label: str
    path: str


@router.get("/sounds", response_model=list[SoundAsset])
def list_sounds() -> list[SoundAsset]:
    if not SOUND_DIR.exists():
        return []
    assets = []
    for file in sorted(SOUND_DIR.rglob("*")):
        if file.suffix.lower() not in AUDIO_EXTENSIONS:
            continue
        rel = file.relative_to(SOUND_DIR).as_posix()
        assets.append(SoundAsset(id=rel, label=file.stem, path=rel))
    return assets