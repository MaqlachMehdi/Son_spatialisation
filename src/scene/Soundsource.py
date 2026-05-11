"""
Soundsource.py
--------------
Dataclass représentant une source sonore positionnée dans l'espace 3D.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SoundSource:
    """
    Source sonore positionnée dans l'espace sphérique.

    Fournir soit `path` (fichier WAV sur disque) soit `signal` + `sr`
    (array numpy généré en mémoire, par ex. depuis GenerateSound).

    Attributs
    ---------
    azimuth : float
        Azimut en degrés  (convention SOFA : 0° = devant, 90° = gauche).
    elevation : float
        Élévation en degrés (-90° = bas, 0° = horizontal, 90° = dessus).
    path : str | None
        Chemin vers le fichier WAV source (mono ou stéréo).
    signal : np.ndarray | None
        Signal audio directement en mémoire (mono float32).
        Utilisé à la place de `path` pour éviter la lecture disque.
    sr : int
        Taux d'échantillonnage du signal en mémoire (ignoré si `path` fourni).
    distance : float
        Distance en mètres. Transmise à DistanceModel (atténuation 1/r,
        retard de propagation, absorption). Défaut : 2.06 m (distance IRCAM LISTEN).
    gain : float
        Gain linéaire appliqué après convolution (défaut : 1.0).
        Permet d'équilibrer les niveaux relatifs dans un paysage sonore.
    """

    azimuth:   float
    elevation: float
    path:      str | None       = None
    signal:    np.ndarray | None = field(default=None, compare=False, repr=False)
    sr:        int               = 44_100
    distance:  float             = 2.06
    gain:      float             = 1.0

    def __post_init__(self) -> None:
        if self.path is None and self.signal is None:
            raise ValueError(
                "SoundSource : fournissez soit 'path' (fichier WAV) "
                "soit 'signal' (array numpy depuis GenerateSound)."
            )
        if self.path is not None and self.signal is not None:
            raise ValueError(
                "SoundSource : fournissez 'path' OU 'signal', pas les deux."
            )

    # ──────────────────────────────────────────────────────────────────────
    # Constructeur alternatif — depuis un générateur GenerateSound
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_generator(
        cls,
        generator: Any,
        azimuth:   float,
        elevation: float,
        distance:  float = 2.06,
        gain:      float = 1.0,
    ) -> SoundSource:
        """
        Crée un SoundSource depuis un objet GenerateSound (Impulse, Note, BurstNote…).

        Appelle generator.generate() en mémoire — aucune écriture disque.

        Paramètres
        ----------
        generator : Impulse | Note | BurstNote
            N'importe quel objet possédant .generate() → np.ndarray et .sr : int.
        azimuth, elevation, distance, gain : cf. SoundSource.
        """
        signal = generator.generate()
        return cls(
            azimuth=azimuth,
            elevation=elevation,
            signal=signal,
            sr=generator.sr,
            distance=distance,
            gain=gain,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Représentation
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        source_info = (
            f"path='{self.path}'"
            if self.path is not None
            else f"signal=<array {len(self.signal)} samples @ {self.sr} Hz>"
        )
        return (
            f"SoundSource({source_info}, "
            f"az={self.azimuth}°, el={self.elevation}°, "
            f"dist={self.distance}m, gain={self.gain})"
        )
