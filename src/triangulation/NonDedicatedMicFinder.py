"""
NonDedicatedMicFinder.py
------------------------
Identifie les mic_ids qui ne sont pas des microphones dédiés (Phase 1)
et évalue leur coverage (nombre d'instruments fournissant une mesure TOA).

Un mic est « localisable » si au moins min_coverage instruments (≥ 4)
possèdent une mesure TOA vers ce mic — condition nécessaire pour une
trilatération 3D non-dégénérée.
"""

from __future__ import annotations

from dataclasses import dataclass

from .DistanceData import DistanceData
from .DedicatedMicMapper import DedicatedMicMapper
from .logger import logger

_EXCLUDED_MICS: frozenset[str]  = frozenset({"mic_05"})
_MIN_COVERAGE_DEFAULT: int       = 4


@dataclass
class NonDedicatedMicFinder:
    """
    Inventaire des microphones non-dédiés présents dans les données.

    Attributs
    ---------
    unknown_mic_ids : list[str]
        Tous les mic_ids non-dédiés (hors exclus), triés numériquement.
    coverage : dict[str, int]
        Nombre d'instruments ayant une mesure vers chaque mic non-dédié.
    low_coverage : list[str]
        Mics avec coverage < min_coverage : ne peuvent pas être trilatérés.
    min_coverage : int
        Seuil minimum d'ancres (défaut = 4).
    """

    unknown_mic_ids: list[str]
    coverage:        dict[str, int]
    low_coverage:    list[str]
    min_coverage:    int

    # ──────────────────────────────────────────────────────────────────────
    # Constructeur alternatif
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_data(
        cls,
        data:         DistanceData,
        mapper:       DedicatedMicMapper,
        min_coverage: int = _MIN_COVERAGE_DEFAULT,
    ) -> NonDedicatedMicFinder:
        logger.step(1, 3, "Inventaire des microphones non-dédiés")

        dedicated_ids = set(mapper.dedicated_mic_ids)
        candidate_ids = set(data.all_mic_ids) - dedicated_ids - _EXCLUDED_MICS

        n_instr = len(data.instruments)
        coverage: dict[str, int] = {
            mic_id: sum(
                1 for instr in data.instruments
                if data.has_measurement(instr, mic_id)
            )
            for mic_id in candidate_ids
        }

        unknown = sorted(candidate_ids, key=lambda x: int(x.split("_")[1]))
        low_cov = [m for m in unknown if coverage[m] < min_coverage]

        logger.scalar("Micros dédiés  (Phase 1)",  float(len(dedicated_ids)), "")
        logger.scalar("Micros candidats Phase 2",  float(len(unknown)),       "")

        for mic_id in unknown:
            cov = coverage[mic_id]
            logger.check(
                f"{mic_id}",
                ok=cov >= min_coverage,
                detail=f"coverage = {cov}/{n_instr} instruments",
            )

        if low_cov:
            logger.warn(
                f"{len(low_cov)} mic(s) avec coverage insuffisant (< {min_coverage}) : "
                + ", ".join(low_cov)
            )

        return cls(
            unknown_mic_ids=unknown,
            coverage=coverage,
            low_coverage=low_cov,
            min_coverage=min_coverage,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Utilitaires
    # ──────────────────────────────────────────────────────────────────────

    def localisable(self) -> list[str]:
        """Sous-liste des mics ayant un coverage suffisant pour la trilatération."""
        return [m for m in self.unknown_mic_ids if self.coverage[m] >= self.min_coverage]

    def __len__(self) -> int:
        return len(self.unknown_mic_ids)
