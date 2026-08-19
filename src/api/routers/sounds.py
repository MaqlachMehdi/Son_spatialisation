from __future__ import annotations

import io
import os
import subprocess
import tempfile
import uuid
from pathlib import Path

import soundfile as sf
from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel
from sqlalchemy import delete as sa_delete
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..auth import current_active_user, current_active_user_optional
from ..config import AUDIO_EXTENSIONS, SOUND_DIR, UPLOAD_EXTENSIONS, UPLOAD_MAX_BYTES, UPLOADS_DIR
from ..database import get_async_session
from ..models import AudioAsset, User

router = APIRouter()


class SoundAsset(BaseModel):
    id: str
    label: str
    path: str
    personal: bool = False


def _decode_any_format(content: bytes, ext: str):
    """Décode le fichier uploadé quel que soit son format d'origine.

    soundfile (libsndfile) lit nativement wav/flac/ogg/aiff mais pas les
    formats compressés type mp3/m4a/aac/webm — pour ceux-là on appelle
    ffmpeg directement en sous-processus plutôt que de passer par le repli
    "audioread" de librosa : ce dernier a été retiré en librosa 1.0 (present
    en 0.10.x, marqué déprécié, absent depuis — cf. ModuleNotFoundError
    "audioread" observé une fois la dépendance mise à jour), donc plus fiable
    de ne pas en dépendre pour cette conversion.
    """
    try:
        audio, samplerate = sf.read(io.BytesIO(content), dtype="float32")
        return audio, samplerate
    except Exception:
        pass

    # ffmpeg a besoin d'un vrai fichier sur disque en entrée (pas de stdin
    # ici pour rester simple). delete=False + suppression manuelle : sur
    # Windows, un fichier encore ouvert par ce process ne peut pas être
    # relu par le sous-processus tant qu'on ne l'a pas fermé nous-même.
    in_path = out_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_in:
            tmp_in.write(content)
            in_path = tmp_in.name
        out_path = in_path + ".wav"

        result = subprocess.run(
            ["ffmpeg", "-y", "-i", in_path, "-f", "wav", out_path],
            capture_output=True,
            timeout=60,
        )
        if result.returncode != 0 or not os.path.exists(out_path):
            raise RuntimeError(result.stderr.decode(errors="replace"))

        audio, samplerate = sf.read(out_path, dtype="float32")
        return audio, samplerate
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Impossible de décoder ce fichier audio.") from exc
    finally:
        for p in (in_path, out_path):
            if p and os.path.exists(p):
                os.unlink(p)


@router.get("/sounds", response_model=list[SoundAsset])
async def list_sounds(
    user: User | None = Depends(current_active_user_optional),
    session: AsyncSession = Depends(get_async_session),
) -> list[SoundAsset]:
    assets: list[SoundAsset] = []

    if SOUND_DIR.exists():
        for file in sorted(SOUND_DIR.rglob("*")):
            if file.suffix.lower() not in AUDIO_EXTENSIONS:
                continue
            rel = file.relative_to(SOUND_DIR).as_posix()
            assets.append(SoundAsset(id=rel, label=file.stem, path=rel))

    if user is not None:
        rows = (
            (
                await session.execute(
                    select(AudioAsset).where(AudioAsset.user_id == user.id).order_by(AudioAsset.created_at)
                )
            )
            .scalars()
            .all()
        )
        for a in rows:
            assets.append(SoundAsset(id=f"personal:{a.id}", label=a.label, path=f"personal/{a.id}", personal=True))

    return assets


@router.post("/sounds/upload", response_model=SoundAsset, status_code=201)
async def upload_sound(
    file: UploadFile = File(...),
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> SoundAsset:
    filename = file.filename or "son importé"
    ext = Path(filename).suffix.lower()
    if ext not in UPLOAD_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté ({ext or 'inconnu'}). Formats acceptés : {', '.join(UPLOAD_EXTENSIONS)}.",
        )

    content = await file.read()
    if len(content) > UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"Fichier trop volumineux (max {UPLOAD_MAX_BYTES // (1024 * 1024)} Mo).")
    if not content:
        raise HTTPException(status_code=400, detail="Fichier vide.")

    audio, samplerate = _decode_any_format(content, ext)

    asset_id = uuid.uuid4().hex
    storage_path = f"{user.id}/{asset_id}.wav"
    full_path = UPLOADS_DIR / storage_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(full_path), audio, samplerate, subtype="PCM_16")

    label = Path(filename).stem or "Son importé"
    asset = AudioAsset(id=asset_id, user_id=user.id, label=label, storage_path=storage_path)
    session.add(asset)
    await session.commit()

    return SoundAsset(id=f"personal:{asset_id}", label=label, path=f"personal/{asset_id}", personal=True)


@router.delete("/sounds/upload/{asset_id}", status_code=204)
async def delete_sound(
    asset_id: str,
    user: User = Depends(current_active_user),
    session: AsyncSession = Depends(get_async_session),
) -> None:
    asset = (
        await session.execute(select(AudioAsset).where(AudioAsset.id == asset_id, AudioAsset.user_id == user.id))
    ).scalar_one_or_none()
    if asset is None:
        raise HTTPException(status_code=404, detail="Son introuvable.")

    full_path = UPLOADS_DIR / asset.storage_path
    full_path.unlink(missing_ok=True)

    await session.execute(sa_delete(AudioAsset).where(AudioAsset.id == asset_id, AudioAsset.user_id == user.id))
    await session.commit()
