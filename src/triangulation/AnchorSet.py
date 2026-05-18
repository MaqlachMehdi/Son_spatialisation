"""
AnchorSet.py
------------
Encapsule les positions d'ancres Phase 1 et fournit, pour un mic cible,
les distances TOA et poids de confiance depuis chaque ancre.

Critère de poids
----------------
    w_ij = max_abs_ij / sample_ij

Fort quand le signal est à la fois précoce (sample bas) et intense
(max_abs élevé) → signature du trajet direct → mesure fiable.
C'est l'inverse du score de sélection de DedicatedMicMapper, ce qui
est cohérent : on récompense maintenant ce qu'on cherchait alors.

Ancres invalides
----------------
Exclues silencieusement si : mesure absente, sample ≤ 0, ou max_abs ≤ 0.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

from .MicPositions import MicPositions
from .DistanceData import DistanceData

_SPEED_OF_SOUND: float = 343.0  # m/s


@dataclass
class AnchorSet:
    """
    Référentiel d'ancres construit depuis les positions Phase 1.

    Attributs
    ---------
    anchor_positions : np.ndarray  shape (n, 3)
        Coordonnées cartésiennes des n instruments en mètres,
        dans le repère orienté de la Phase 1.
    anchor_labels : list[str]
        Noms des instruments, dans le même ordre que les lignes.
    anchor_mic_ids : list[str]
        mic_ids dédiés correspondants.
    data : DistanceData
        Données brutes pour interroger les TOA instrument → mic_cible.
    """

    anchor_positions: np.ndarray
    anchor_labels:    list[str]
    anchor_mic_ids:   list[str]
    data:             DistanceData

    # ──────────────────────────────────────────────────────────────────────
    # Constructeur alternatif
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_phase1(
        cls,
        positions: MicPositions,
        data:      DistanceData,
    ) -> AnchorSet:
        return cls(
            anchor_positions=positions.positions.copy(),
            anchor_labels=list(positions.instrument_labels),
            anchor_mic_ids=list(positions.mic_ids),
            data=data,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Requête principale
    # ──────────────────────────────────────────────────────────────────────

    def get_distances(
        self,
        mic_id: str,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Retourne (positions_k×3, distances_k, weights_k) pour les ancres valides.

        Parameters
        ----------
        mic_id : str
            Identifiant du micro cible (non-dédié).

        Returns
        -------
        positions : np.ndarray (k, 3)  — coordonnées des k ancres valides
        distances : np.ndarray (k,)    — distances TOA en mètres
        weights   : np.ndarray (k,)    — confiance par ancre (non normalisée)
        """
        pos_list: list[np.ndarray] = []
        d_list:   list[float]      = []
        w_list:   list[float]      = []

        for i, instr in enumerate(self.anchor_labels):
            if not self.data.has_measurement(instr, mic_id):
                continue
            raw     = self.data.raw[instr][mic_id]
            sample  = int(raw["sample"])
            max_abs = float(raw["max_abs"])
            if sample <= 0 or max_abs <= 0:
                continue
            pos_list.append(self.anchor_positions[i])
            d_list.append(sample / self.data.fs * _SPEED_OF_SOUND)
            w_list.append(max_abs / sample)

        return (
            np.array(pos_list, dtype=float),
            np.array(d_list,   dtype=float),
            np.array(w_list,   dtype=float),
        )

    # ──────────────────────────────────────────────────────────────────────
    # Utilitaires
    # ──────────────────────────────────────────────────────────────────────

    @property
    def n(self) -> int:
        """Nombre d'ancres disponibles."""
        return len(self.anchor_labels)
