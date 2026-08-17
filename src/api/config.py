"""
config.py
---------
Chemins et constantes partagés entre les routers.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]  # src/api/config.py -> racine du repo
SOUND_DIR = ROOT / "sound"
DATASET_DIR = ROOT / "dataset"
SOFA_PATH = Path(os.environ.get("HRTF_SOFA_PATH", DATASET_DIR / "generic.sofa"))
AUDIO_EXTENSIONS = (".wav", ".flac")

# Sons importés par les utilisateurs — volontairement HORS de SOUND_DIR : ce
# dernier est parcouru en entier (rglob) par /sounds pour le catalogue
# public, un sous-dossier ici serait exposé à tout le monde, pas seulement
# à son propriétaire.
UPLOADS_DIR = ROOT / "sound_uploads"
UPLOAD_MAX_BYTES = 50 * 1024 * 1024  # 50 Mo
UPLOAD_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")