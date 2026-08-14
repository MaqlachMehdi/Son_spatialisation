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