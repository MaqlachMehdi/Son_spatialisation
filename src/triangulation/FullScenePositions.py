"""
FullScenePositions.py
---------------------
Dataclass résultat agrégeant Phase 1 (micros dédiés) et Phase 2 (non-dédiés).

Usage
-----
    scene = loc2.run()          # Phase2Localizer

    print(scene)                # tableau Phase 1 + tableau Phase 2
    scene.all_positions()       # dict {mic_id → np.ndarray (3,)}
    scene.reliable(0.5)         # mics Phase 2 avec résiduel < 0.5 m
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .MicPositions import MicPositions


@dataclass
class NonDedMicResult:
    """
    Résultat de trilatération pour un micro non-dédié.

    Attributs
    ---------
    position     : np.ndarray (3,)  — coordonnées cartésiennes (m),
                                      même repère orienté que Phase 1.
    residual_rms : float            — RMS de |d_i - ||p_i - p̂||  (mètres).
                                      Indicateur de qualité : < 0.3 m = fiable.
    coverage     : int              — instruments ayant une mesure vers ce mic.
    n_used       : int              — ancres retenues après rejet d'outliers.
    """

    position:     np.ndarray
    residual_rms: float
    coverage:     int
    n_used:       int


@dataclass
class FullScenePositions:
    """
    Vue unifiée de tous les microphones : dédiés (Phase 1) + non-dédiés (Phase 2).

    Attributs
    ---------
    dedicated     : MicPositions
        Résultat Phase 1 — positions 3D + métadonnées MDS.
    non_dedicated : dict[str, NonDedMicResult]
        mic_id → résultat de trilatération Phase 2.
    """

    dedicated:     MicPositions
    non_dedicated: dict[str, NonDedMicResult]

    # ──────────────────────────────────────────────────────────────────────
    # Accès unifié
    # ──────────────────────────────────────────────────────────────────────

    def all_positions(self) -> dict[str, np.ndarray]:
        """Dict {mic_id → position (3,)} pour tous les micros (Phase 1 + 2)."""
        result: dict[str, np.ndarray] = {
            mic_id: self.dedicated.positions[i]
            for i, mic_id in enumerate(self.dedicated.mic_ids)
        }
        for mic_id, res in self.non_dedicated.items():
            result[mic_id] = res.position
        return result

    def reliable(self, threshold_m: float = 0.5) -> list[str]:
        """mic_ids Phase 2 dont le résiduel RMS < threshold_m mètres."""
        return [
            mic_id
            for mic_id, r in self.non_dedicated.items()
            if r.residual_rms < threshold_m
        ]

    # ──────────────────────────────────────────────────────────────────────
    # Représentation
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        lines = [str(self.dedicated), ""]

        n2 = len(self.non_dedicated)
        lines.append(f"Phase 2 — {n2} microphone(s) non-dédié(s) localisé(s)")

        if n2 == 0:
            lines.append("  (aucun)")
            return "\n".join(lines)

        header = (
            f"  {'Mic':<10} {'X (m)':>8} {'Y (m)':>8} {'Z (m)':>8}"
            f"  {'Résiduel':>9}  {'Ancres':>8}"
        )
        sep = (
            f"  {'─'*10} {'─'*8} {'─'*8} {'─'*8}  {'─'*9}  {'─'*8}"
        )
        lines += [header, sep]

        for mic_id in sorted(
            self.non_dedicated, key=lambda x: int(x.split("_")[1])
        ):
            r = self.non_dedicated[mic_id]
            x, y, z = float(r.position[0]), float(r.position[1]), float(r.position[2])
            lines.append(
                f"  {mic_id:<10} {x:>8.3f} {y:>8.3f} {z:>8.3f}"
                f"  {r.residual_rms:>8.3f} m  {r.n_used:>3}/{r.coverage:>3}"
            )

        n_reliable = len(self.reliable())
        lines.append(f"\n  Localisations fiables (< 0.5 m) : {n_reliable}/{n2}")
        return "\n".join(lines)
