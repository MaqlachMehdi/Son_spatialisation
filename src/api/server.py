"""
server.py
---------
API FastAPI qui expose le pipeline de spatialisation binaurale existant
(HRTFInterpolator, SoundSource, Soundscape) au frontend React/Three.js, et
les comptes utilisateurs (fastapi-users) en vue de la persistance des scènes.

Lancement (depuis n'importe quel répertoire) :
    python -m uvicorn api.server:app --reload --port 8000 --app-dir src
"""

from __future__ import annotations

import os
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

from dotenv import load_dotenv

load_dotenv(_ROOT / ".env")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from hrtf import HRTF, HRTFInterpolator

from .auth import auth_router
from .config import SOFA_PATH
from .routers import hrtfs, render, sounds, workspace

app = FastAPI(title="Spatialisation HRTF API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_ORIGIN", "http://localhost:5173")],
    # Requis pour que le cookie de session (httpOnly) passe en cross-origin
    # entre le frontend (5173) et cette API (8000).
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print(f"[api] Chargement HRTF depuis {SOFA_PATH} …")
app.state.interpolator = HRTFInterpolator(HRTF.from_sofa(SOFA_PATH), verbose=False)
# Chemin du .sofa actif — mutable (cf. PUT /hrtfs/active dans routers/hrtfs.py),
# contrairement à SOFA_PATH qui reste le choix de démarrage figé.
app.state.active_hrtf_path = SOFA_PATH
print("[api] HRTFInterpolator prêt.")

app.include_router(auth_router)
app.include_router(sounds.router)
app.include_router(hrtfs.router)
app.include_router(render.router)
app.include_router(workspace.router)