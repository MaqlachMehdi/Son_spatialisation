"""
Soundsource.py
--------------
Dataclass représentant une source sonore positionnée dans l'espace 3D.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SoundSource:
    """
    Source sonore positionnée dans l'espace sphérique.

    Attributs
    ---------
    path : str
        Chemin vers le fichier WAV source (mono ou stéréo).
    azimuth : float
        Azimut en degrés  (convention SOFA : 0° = devant, 90° = gauche).
    elevation : float
        Élévation en degrés (-90° = bas, 0° = horizontal, 90° = dessus).
    distance : float
        Distance en mètres. Utilisée pour le gain d'atténuation 1/r.
        Défaut : 2.06 m (distance de mesure du dataset IRCAM LISTEN).
    gain : float
        Gain linéaire appliqué après convolution (défaut : 1.0).
        Permet d'équilibrer les niveaux relatifs dans un paysage sonore.
    """

    path: str
    azimuth: float
    elevation: float
    distance: float = 2.06
    gain: float = 1.0

    @property
    def distance_gain(self) -> float:
        """Gain d'atténuation par la distance (loi 1/r, référence = 2.06 m)."""
        return 2.06 / max(self.distance, 0.01)

    @property
    def total_gain(self) -> float:
        """Gain total : distance × gain manuel."""
        return self.distance_gain * self.gain

    def __repr__(self) -> str:
        return (
            f"SoundSource(path='{self.path}', "
            f"az={self.azimuth}°, el={self.elevation}°, "
            f"dist={self.distance}m, gain={self.gain})"
        )
