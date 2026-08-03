"""
server.py
---------
API FastAPI qui expose le pipeline de spatialisation binaurale existant
(HRTFInterpolator, SoundSource, Soundscape) au frontend React/Three.js.

Lancement (depuis n'importe quel répertoire) :
    python -m uvicorn api.server:app --reload --port 8000 --app-dir src
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]  # src/api/server.py -> racine du repo
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

# Le pipeline existant (engine/Convolution.py, scene/*) logge en UTF-8 (ex: "→").
# Sous uvicorn sur Windows, stdout hérite du codepage cp1252 par défaut et plante
# sur ces caractères — on le force en UTF-8 ici plutôt que de toucher au pipeline.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import os

import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from hrtf import HRTF, HRTFInterpolator
from scene import SoundSource, Soundscape

SOUND_DIR = _ROOT / "sound"
SOFA_PATH = Path(os.environ.get("HRTF_SOFA_PATH", _ROOT / "dataset" / "generic.sofa"))
AUDIO_EXTENSIONS = (".wav", ".flac")

app = FastAPI(title="Spatialisation HRTF API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"[api] Chargement HRTF depuis {SOFA_PATH} …")
_hrtf = HRTF.from_sofa(SOFA_PATH)
_interpolator = HRTFInterpolator(_hrtf, verbose=False)
print("[api] HRTFInterpolator prêt.")


class SourcePayload(BaseModel):
    path: str
    azimuth: float
    elevation: float
    distance: float = 2.06
    gain: float = 1.0
    label: str = ""


class RenderRequest(BaseModel):
    sources: list[SourcePayload]


class SoundAsset(BaseModel):
    id: str
    label: str
    path: str


@app.get("/sounds", response_model=list[SoundAsset])
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


@app.post("/render")
def render(request: RenderRequest) -> StreamingResponse:
    if not request.sources:
        raise HTTPException(status_code=400, detail="Aucune source fournie.")

    soundscape = Soundscape(_interpolator)
    for src in request.sources:
        full_path = SOUND_DIR / src.path
        if not full_path.is_file():
            raise HTTPException(status_code=404, detail=f"Fichier introuvable : {src.path}")
        soundscape.add_source(
            SoundSource(
                azimuth=src.azimuth,
                elevation=src.elevation,
                distance=src.distance,
                gain=src.gain,
                path=str(full_path),
            )
        )

    mix = soundscape.render()  # (N, 2) float32

    buf = io.BytesIO()
    sf.write(buf, mix, _interpolator.sample_rate, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")