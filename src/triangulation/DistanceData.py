"""
DistanceData.py
---------------
Chargement et accès brut aux données JSON des RIR (rir_max_per_mic.json).

Responsabilité unique : lire et stocker les données. Aucun calcul.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

from .logger import logger

_SPEED_OF_SOUND = 343.0  # m/s


@dataclass
class DistanceData:
    """
    Données brutes issues du JSON rir_max_per_mic.

    Structure JSON attendue
    ----------------------
    {
        "instrument": {
            "mic_id": {"max_abs": float, "sample": int},
            ...
        },
        ...
    }

    Attributs
    ---------
    raw : dict
        Dictionnaire brut tel que chargé depuis le JSON.
    fs : int
        Fréquence d'échantillonnage en Hz (défaut 48 000).
    """

    raw: Dict[str, Dict[str, Dict[str, float]]]
    fs: int = 48_000

    # ──────────────────────────────────────────────────────────────────────
    # Constructeur alternatif
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str | Path, fs: int = 48_000) -> DistanceData:
        path = Path(path)
        logger.start(f"Chargement — {path.name}")

        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        obj = cls(raw=raw, fs=fs)

        logger.scalar("Instruments détectés", len(obj.instruments))
        logger.scalar("Microphones détectés", len(obj.all_mic_ids))
        logger.scalar("Fréquence d'échantillonnage", fs, "Hz")
        return obj

    # ──────────────────────────────────────────────────────────────────────
    # Propriétés d'accès
    # ──────────────────────────────────────────────────────────────────────

    @property
    def instruments(self) -> list[str]:
        return list(self.raw.keys())

    @property
    def all_mic_ids(self) -> list[str]:
        """Ensemble trié de tous les mic_id présents dans les données."""
        ids: set[str] = set()
        for mic_dict in self.raw.values():
            ids.update(mic_dict.keys())
        return sorted(ids, key=lambda x: int(x.split("_")[1]))

    # ──────────────────────────────────────────────────────────────────────
    # Accès par (instrument, mic_id)
    # ──────────────────────────────────────────────────────────────────────

    def get_sample(self, instrument: str, mic_id: str) -> int:
        """Numéro d'échantillon du pic de la RIR."""
        return int(self.raw[instrument][mic_id]["sample"])

    def get_distance(self, instrument: str, mic_id: str) -> float:
        """
        Distance en mètres calculée par TOA :
            d = sample / fs * c
        """
        return self.get_sample(instrument, mic_id) / self.fs * _SPEED_OF_SOUND

    def has_measurement(self, instrument: str, mic_id: str) -> bool:
        return mic_id in self.raw.get(instrument, {})
