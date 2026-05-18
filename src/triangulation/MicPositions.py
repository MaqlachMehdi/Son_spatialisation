"""
MicPositions.py
---------------
Dataclass résultat du pipeline MDS.

Traitée comme immuable : ne pas modifier les attributs après construction.
(frozen=True évité car np.ndarray n'est pas hashable.)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass


@dataclass
class MicPositions:
    """
    Résultat du pipeline Classical MDS.

    Attributs
    ---------
    positions : np.ndarray  shape (n, 3)
        Coordonnées cartésiennes [x, y, z] en mètres, centrées au centroïde.
    mic_ids : tuple[str, ...]
        Identifiants des micros, dans le même ordre que les lignes de positions.
    instrument_labels : tuple[str, ...]
        Noms des instruments associés.
    eigenvalues : np.ndarray  shape (n,)
        Valeurs propres de B, triées décroissantes.
    explained_variance_ratio : float
        (λ₁ + λ₂ + λ₃) / Σλₖ  — doit être proche de 1.0 pour valider 3D.
    """

    positions: np.ndarray
    mic_ids: tuple
    instrument_labels: tuple
    eigenvalues: np.ndarray
    explained_variance_ratio: float

    # ──────────────────────────────────────────────────────────────────────
    # Conversion sphérique (convention SOFA CCW)
    # ──────────────────────────────────────────────────────────────────────

    def to_spherical(self) -> dict[str, dict[str, float]]:
        """
        Convertit les positions cartésiennes en (az, el, dist).

        Convention SOFA CCW
        -------------------
        0°  = devant  (+x)
        90° = gauche  (+y)
        axe z = haut

        Retourne
        --------
        dict {mic_id: {"az": float°, "el": float°, "dist": float m}}
        """
        result: dict[str, dict[str, float]] = {}
        for mic_id, pos in zip(self.mic_ids, self.positions):
            x, y, z = float(pos[0]), float(pos[1]), float(pos[2])
            dist = float(np.linalg.norm(pos))
            az   = float(np.degrees(np.arctan2(y, x))) % 360.0
            el   = float(np.degrees(np.arcsin(np.clip(z / (dist + 1e-12), -1.0, 1.0))))
            result[mic_id] = {"az": az, "el": el, "dist": dist}
        return result

    # ──────────────────────────────────────────────────────────────────────
    # Représentation
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        sph   = self.to_spherical()
        lines = [
            f"MicPositions  — {len(self.mic_ids)} micros  "
            f"(variance 3D = {self.explained_variance_ratio * 100:.2f}%)",
            f"{'Micro':<12} {'Instrument':<22} {'Az':>8} {'El':>7} {'Dist':>8}",
            "─" * 62,
        ]
        for mic_id, instr in zip(self.mic_ids, self.instrument_labels):
            s = sph[mic_id]
            lines.append(
                f"{mic_id:<12} {instr:<22} "
                f"{s['az']:7.1f}°  {s['el']:6.1f}°  {s['dist']:6.2f} m"
            )
        return "\n".join(lines)
