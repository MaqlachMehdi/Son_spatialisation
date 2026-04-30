"""
hrtf.py
-------
Dataclass représentant un dataset HRTF chargé depuis un fichier SOFA.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


@dataclass
class HRTF:
    """
    Contient toutes les données d'un fichier SOFA (SimpleFreeFieldHRIR).

    Attributs
    ---------
    hrir : np.ndarray
        Réponses impulsionnelles binaurales.
        Forme : (M, 2, N)  →  M positions x 2 oreilles x N échantillons.
    positions : np.ndarray
        Positions de source en coordonnées sphériques.
        Forme : (M, 3)  →  colonnes [azimut°, élévation°, distance_m].
    delay : np.ndarray
        Délai fractionnaire par position et par oreille.
        Forme : (M, 2).
    sample_rate : int
        Fréquence d'échantillonnage en Hz.
    source_path : Path
        Chemin du fichier SOFA source.
    metadata : dict
        Attributs globaux du fichier SOFA (convention, auteur, etc.).
    """

    hrir: np.ndarray
    positions: np.ndarray
    delay: np.ndarray
    sample_rate: int
    source_path: Path
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Propriétés dérivées
    # ------------------------------------------------------------------

    @property
    def n_positions(self) -> int:
        """Nombre de positions de mesure."""
        return self.hrir.shape[0]

    @property
    def n_samples(self) -> int:
        """Longueur des réponses impulsionnelles (en échantillons)."""
        return self.hrir.shape[2]

    @property
    def duration_ms(self) -> float:
        """Durée des HRIRs en millisecondes."""
        return self.n_samples / self.sample_rate * 1000

    @property
    def azimuths(self) -> np.ndarray:
        """Azimuts uniques présents dans le dataset (degrés)."""
        return np.unique(self.positions[:, 0])

    @property
    def elevations(self) -> np.ndarray:
        """Élévations uniques présentes dans le dataset (degrés)."""
        return np.unique(self.positions[:, 1])

    @property
    def time_axis_ms(self) -> np.ndarray:
        """Axe temporel en millisecondes correspondant aux HRIRs."""
        return np.arange(self.n_samples) / self.sample_rate * 1000

    @property
    def freq_axis_hz(self) -> np.ndarray:
        """Axe fréquentiel (rfft) en Hz."""
        from scipy.fft import rfftfreq
        return rfftfreq(self.n_samples, 1 / self.sample_rate)

    # ------------------------------------------------------------------
    # Méthodes utilitaires
    # ------------------------------------------------------------------

    def get_index(self, azimuth: float, elevation: float) -> int:
        """
        Renvoie l'indice de la position la plus proche de (azimuth, elevation).

        Paramètres
        ----------
        azimuth : float
            Azimut cible en degrés.
        elevation : float
            Élévation cible en degrés.

        Retourne
        --------
        int
            Indice dans `self.positions` / `self.hrir`.
        """
        diff = np.abs(self.positions[:, 0] - azimuth) + \
               np.abs(self.positions[:, 1] - elevation)
        return int(np.argmin(diff))

    def get_hrir(self, azimuth: float, elevation: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Retourne les HRIRs gauche et droite pour la position la plus proche.

        Retourne
        --------
        (hrir_left, hrir_right) : tuple de ndarray de forme (N,)
        """
        idx = self.get_index(azimuth, elevation)
        return self.hrir[idx, 0, :], self.hrir[idx, 1, :]

    def get_hrtf(self, azimuth: float, elevation: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Retourne les spectres de magnitude HRTF (en linéaire) gauche et droite.

        Retourne
        --------
        (hrtf_left, hrtf_right) : tuple de ndarray de forme (N//2+1,)
        """
        from scipy.fft import rfft
        l, r = self.get_hrir(azimuth, elevation)
        return np.abs(rfft(l)), np.abs(rfft(r))

    def horizontal_plane(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Retourne les indices et positions du plan horizontal (élévation = 0°).

        Retourne
        --------
        (indices, positions_subset) : arrays de longueur M_h
        """
        mask = self.positions[:, 1] == 0.0
        indices = np.where(mask)[0]
        return indices, self.positions[indices]

    # ------------------------------------------------------------------
    # Constructeur depuis fichier SOFA
    # ------------------------------------------------------------------

    @classmethod
    def from_sofa(cls, path: str | Path) -> "HRTF":
        """
        Charge un fichier SOFA et construit l'objet HRTF.

        Paramètres
        ----------
        path : str ou Path
            Chemin vers le fichier .sofa.

        Retourne
        --------
        HRTF
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier SOFA introuvable : {path}")

        with Dataset(str(path), "r") as f:
            hrir      = f.variables["Data.IR"][:]
            positions = f.variables["SourcePosition"][:]
            delay     = f.variables["Data.Delay"][:]
            sr        = int(f.variables["Data.SamplingRate"][:].flat[0])
            metadata  = {attr: getattr(f, attr) for attr in f.ncattrs()}

        return cls(
            hrir=np.asarray(hrir),
            positions=np.asarray(positions),
            delay=np.asarray(delay),
            sample_rate=sr,
            source_path=path,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Représentation
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"HRTF("
            f"positions={self.n_positions}, "
            f"samples={self.n_samples}, "
            f"sr={self.sample_rate} Hz, "
            f"source='{self.source_path.name}')"
        )
