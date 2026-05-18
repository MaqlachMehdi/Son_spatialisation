"""
DistanceMatrix.py
-----------------
Construction et validation de la matrice de distances 17×17 entre micros dédiés.

D[i][j] = min(s_ij, s_ji) / fs * c   (mètres)

Pourquoi min(s_ij, s_ji) et non s_ij seul ?
--------------------------------------------
Le JSON fournit le numéro d'échantillon du **maximum absolu** de la RIR, pas du
premier arrivé. Pour les paires distantes, une réflexion tardive peut dominer la
RIR (amplitude > trajet direct) → TOA mesuré >> TOA direct → distance surestimée.

Le trajet direct étant toujours le plus court, min(s_ij, s_ji) en est une borne
supérieure plus serrée. Cela réduit fortement le bruit de reconstruction MDS.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .DistanceData import DistanceData
from .DedicatedMicMapper import DedicatedMicMapper
from .logger import logger

_SYMMETRY_TOL = 0.10   # mètres — écart max toléré avant avertissement


@dataclass
class DistanceMatrix:
    """
    Matrice de distances (n×n) symétrisée entre les n micros dédiés.

    Attributs
    ---------
    matrix : np.ndarray  shape (n, n)
        Distances en mètres. Diagonale nulle, valeurs hors-diagonale > 0.
    mic_ids : list[str]
        Identifiants des micros, dans l'ordre des lignes/colonnes.
    instrument_labels : list[str]
        Noms des instruments associés, dans le même ordre.
    """

    matrix: np.ndarray
    mic_ids: list[str]
    instrument_labels: list[str]

    # ──────────────────────────────────────────────────────────────────────
    # Constructeur alternatif
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_data(
        cls,
        data: DistanceData,
        mapper: DedicatedMicMapper,
    ) -> DistanceMatrix:
        logger.step(2, 4, "Construction de la matrice de distances  (17×17, min bidirectionnel)")

        mic_ids     = mapper.dedicated_mic_ids
        instruments = [mapper.mic_to_instrument[m] for m in mic_ids]
        n           = len(mic_ids)

        D         = np.zeros((n, n))
        n_swapped = 0   # paires où le sens inverse était plus court (réflexion évitée)

        for i, (mic_i, instr_i) in enumerate(zip(mic_ids, instruments)):
            for j, (mic_j, instr_j) in enumerate(zip(mic_ids, instruments)):
                if i == j:
                    D[i, j] = 0.0
                    continue

                has_ij = data.has_measurement(instr_i, mic_j)
                has_ji = data.has_measurement(instr_j, mic_i)

                if has_ij and has_ji:
                    s_ij = data.get_sample(instr_i, mic_j)
                    s_ji = data.get_sample(instr_j, mic_i)
                    if i < j and s_ji < s_ij:   # compte une seule fois par paire
                        n_swapped += 1
                    D[i, j] = min(s_ij, s_ji) / data.fs * 343.0
                elif has_ij:
                    D[i, j] = data.get_sample(instr_i, mic_j) / data.fs * 343.0
                elif has_ji:
                    D[i, j] = data.get_sample(instr_j, mic_i) / data.fs * 343.0
                else:
                    logger.warn(f"Mesure manquante : {instr_i} ↔ {mic_j}  (mis à NaN)")
                    D[i, j] = np.nan

        logger.check(
            "min(s_ij, s_ji) appliqué",
            ok=True,
            detail=f"{n_swapped}/{n*(n-1)//2} paires corrigées par sens inverse",
        )
        logger.matrix_info("D brute", D)
        D = cls._validate_and_symmetrize(D, mic_ids)

        return cls(matrix=D, mic_ids=mic_ids, instrument_labels=instruments)

    # ──────────────────────────────────────────────────────────────────────
    # Validation et symétrisation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _validate_and_symmetrize(D: np.ndarray, mic_ids: list[str]) -> np.ndarray:
        n = D.shape[0]

        # 1. Diagonale nulle
        diag_ok = np.allclose(np.diag(D), 0.0)
        logger.check("Diagonale nulle", ok=diag_ok)

        # 2. Valeurs manquantes
        n_nan = int(np.isnan(D).sum())
        nan_ok = n_nan == 0
        logger.check(
            "Aucune valeur manquante",
            ok=nan_ok,
            detail=f"{n_nan} NaN" if not nan_ok else "",
        )
        if not nan_ok:
            D = np.nan_to_num(D, nan=0.0)

        # 3. Positivité hors-diagonale
        mask     = ~np.eye(n, dtype=bool)
        pos_ok   = bool(np.all(D[mask] > 0))
        logger.check("Valeurs strictement positives", ok=pos_ok)

        # 4. Symétrie — mesure de l'écart avant symétrisation
        asym     = np.abs(D - D.T)
        max_asym = float(asym.max())
        sym_ok   = max_asym < _SYMMETRY_TOL
        logger.check(
            "Symétrie",
            ok=sym_ok,
            detail=f"écart max = {max_asym:.4f} m",
        )
        if not sym_ok:
            logger.warn(
                f"Asymétrie > {_SYMMETRY_TOL} m — vérifie les données TOA. "
                "Symétrisation (D + Dᵀ)/2 appliquée."
            )

        # Symétrisation
        D_sym = (D + D.T) / 2.0
        np.fill_diagonal(D_sym, 0.0)

        logger.matrix_info("D symétrisée", D_sym)
        logger.scalar("Distance min (hors-diag)", float(D_sym[mask].min()), "m")
        logger.scalar("Distance max", float(D_sym.max()), "m")
        return D_sym

    # ──────────────────────────────────────────────────────────────────────
    # Utilitaires
    # ──────────────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.mic_ids)

    def get_index(self, mic_id: str) -> int:
        return self.mic_ids.index(mic_id)
