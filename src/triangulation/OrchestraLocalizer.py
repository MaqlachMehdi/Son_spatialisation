"""
OrchestraLocalizer.py
---------------------
Façade principale du pipeline de localisation Phase 1.

Usage
-----
    from triangulation import OrchestraLocalizer

    loc       = OrchestraLocalizer.from_json("dataset_live/rir_max_per_mic.json")
    positions = loc.run()

    print(positions)
    loc.visualizer.plot_all()
"""

from __future__ import annotations

from pathlib import Path

from .DistanceData import DistanceData
from .DedicatedMicMapper import DedicatedMicMapper
from .DistanceMatrix import DistanceMatrix
from .ClassicalMDS import ClassicalMDS
from .AxisOrientator import OrchestralAxisOrientator
from .MicPositions import MicPositions
from .logger import logger


class OrchestraLocalizer:
    """
    Orchestre les trois étapes du pipeline :

        DistanceData → DedicatedMicMapper → DistanceMatrix → ClassicalMDS

    Chaque étape intermédiaire est stockée comme attribut public
    pour permettre l'inspection depuis un notebook après run().

    Attributs (disponibles après run())
    ------------------------------------
    distance_data   : DistanceData
    mapper          : DedicatedMicMapper
    distance_matrix : DistanceMatrix
    positions       : MicPositions
    """

    def __init__(self, distance_data: DistanceData) -> None:
        self.distance_data: DistanceData             = distance_data
        self.mapper:          DedicatedMicMapper | None = None
        self.distance_matrix: DistanceMatrix | None  = None
        self._positions:      MicPositions | None    = None
        self._mds:            ClassicalMDS | None    = None

    # ──────────────────────────────────────────────────────────────────────
    # Constructeur alternatif
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_json(cls, path: str | Path, fs: int = 48_000) -> OrchestraLocalizer:
        data = DistanceData.from_json(path, fs=fs)
        return cls(distance_data=data)

    # ──────────────────────────────────────────────────────────────────────
    # Pipeline
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> MicPositions:
        """
        Exécute le pipeline complet en 4 étapes et retourne les positions 3D.

        Étapes
        ------
        1. DedicatedMicMapper  — argmin(sample/max_abs) par instrument
        2. DistanceMatrix      — matrice 17×17 par min bidirectionnel des TOA
        3. ClassicalMDS        — double centrage + décomposition propre
        4. AxisOrientator      — orientation sémantique sans Kabsch
        """
        self.mapper          = DedicatedMicMapper.from_data(self.distance_data)
        self.distance_matrix = DistanceMatrix.from_data(self.distance_data, self.mapper)
        self._mds            = ClassicalMDS(self.distance_matrix)
        raw_positions        = self._mds.solve()
        self._orientator     = OrchestralAxisOrientator(raw_positions)
        self._positions      = self._orientator.orient()
        logger.done()
        return self._positions

    # ──────────────────────────────────────────────────────────────────────
    # Accès aux résultats
    # ──────────────────────────────────────────────────────────────────────

    @property
    def positions(self) -> MicPositions:
        if self._positions is None:
            raise RuntimeError("Appelez .run() avant d'accéder aux positions.")
        return self._positions

    @property
    def visualizer(self) -> "MicVisualizer":
        from .visualisation import MicVisualizer
        if self._positions is None:
            raise RuntimeError("Appelez .run() avant d'accéder au visualizer.")
        return MicVisualizer(
            positions=self._positions,
            mds=self._mds,
            distance_matrix=self.distance_matrix,
        )
