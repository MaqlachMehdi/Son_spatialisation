"""
Listener.py
-----------
Position et orientation de l'auditeur dans le repère monde, pour permettre à
l'auditeur (et pas seulement la source) de se déplacer dans la scène.

Un Listener convertit un instant t (secondes) en pose (position cartésienne
en mètres, rotation tête -> monde). DynamicConvolver l'utilise pour
transformer la position monde d'une source en position relative à la tête
(azimut, élévation, distance) avant l'appel HRTF — cf. geometry.py pour les
formules.

Classes
-------
Listener        (ABC) — interface commune
StaticListener  — position et orientation fixes (auditeur immobile)
MovingListener  — position et orientation interpolées entre waypoints
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .geometry import rotation_from_ypr, slerp_rotation


# ══════════════════════════════════════════════════════════════════════════════
# Classe de base abstraite
# ══════════════════════════════════════════════════════════════════════════════

class Listener(ABC):
    """
    Interface commune à tous les auditeurs.

    Convertit un instant t (secondes) en pose : position cartésienne monde
    (mètres, convention SOFA x=devant/y=gauche/z=haut) et matrice de rotation
    tête -> monde (3x3, orthonormale).
    """

    @abstractmethod
    def get_pose(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        """
        Retourne (position, rotation) à l'instant t.

        position : np.ndarray, shape (3,) — mètres, repère monde.
        rotation : np.ndarray, shape (3, 3) — tête -> monde.
        """
        ...

    def get_position(self, t: float) -> np.ndarray:
        """Raccourci : uniquement la position, shape (3,)."""
        return self.get_pose(t)[0]

    def get_rotation(self, t: float) -> np.ndarray:
        """Raccourci : uniquement la rotation, shape (3, 3)."""
        return self.get_pose(t)[1]


# ══════════════════════════════════════════════════════════════════════════════
# StaticListener — auditeur immobile (comportement historique)
# ══════════════════════════════════════════════════════════════════════════════

class StaticListener(Listener):
    """
    Auditeur fixe : position et orientation constantes.

    Avec les valeurs par défaut (position à l'origine, orientation neutre),
    StaticListener reproduit exactement le comportement historique du moteur
    (auditeur implicite à l'origine, regard selon +x).

    Paramètres
    ----------
    position : tuple[float, float, float]
        Position (x, y, z) en mètres (défaut : origine).
    yaw, pitch, roll : float
        Orientation de la tête en degrés (défaut : 0° — regard selon +x).
    """

    def __init__(
        self,
        position: tuple[float, float, float] = (0.0, 0.0, 0.0),
        yaw: float = 0.0,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> None:
        self._position = np.asarray(position, dtype=float)
        self._rotation = rotation_from_ypr(yaw, pitch, roll)

    def get_pose(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        return self._position, self._rotation

    def __repr__(self) -> str:
        x, y, z = self._position
        return f"StaticListener(position=({x:.2f}, {y:.2f}, {z:.2f}))"


# ══════════════════════════════════════════════════════════════════════════════
# MovingListener — auditeur mobile par waypoints
# ══════════════════════════════════════════════════════════════════════════════

class MovingListener(Listener):
    """
    Auditeur mobile : position et orientation interpolées entre waypoints.

    La position est interpolée linéairement (mouvement à vitesse constante par
    segment) ; l'orientation est interpolée par SLERP (quaternions), pour
    éviter les artefacts de l'interpolation linéaire d'angles d'Euler.

    La pose est tenue fixe avant le premier waypoint et après le dernier.

    Paramètres
    ----------
    waypoints : list[tuple[float, float, float, float, float, float, float]]
        Liste de (t_s, x, y, z, yaw°, pitch°, roll°), dans n'importe quel ordre.
        Au moins 2 waypoints sont requis.

    Exemple
    -------
    >>> listener = MovingListener([
    ...     (0.0, 0.0, 0.0, 0.0,   0.0, 0.0, 0.0),   # origine, regarde devant
    ...     (5.0, 3.0, 0.0, 0.0,  90.0, 0.0, 0.0),   # avance de 3m, tourne la tête à gauche
    ... ])
    >>> pos, rot = listener.get_pose(2.5)
    """

    def __init__(
        self,
        waypoints: list[tuple[float, float, float, float, float, float, float]],
    ) -> None:
        if len(waypoints) < 2:
            raise ValueError("Au moins 2 waypoints sont requis.")

        waypoints = sorted(waypoints, key=lambda w: w[0])
        self._t = np.array([w[0] for w in waypoints], dtype=float)
        self._pos = np.array([w[1:4] for w in waypoints], dtype=float)
        self._rot = np.array([
            rotation_from_ypr(w[4], w[5], w[6]) for w in waypoints
        ])

    @property
    def duration(self) -> float:
        return float(self._t[-1])

    def get_pose(self, t: float) -> tuple[np.ndarray, np.ndarray]:
        t = float(np.clip(t, self._t[0], self._t[-1]))

        idx = int(np.searchsorted(self._t, t, side="right")) - 1
        idx = int(np.clip(idx, 0, len(self._t) - 2))

        t0, t1 = self._t[idx], self._t[idx + 1]
        dt = t1 - t0
        alpha = (t - t0) / dt if dt > 1e-12 else 0.0

        position = self._pos[idx] + alpha * (self._pos[idx + 1] - self._pos[idx])
        rotation = slerp_rotation(self._rot[idx], self._rot[idx + 1], alpha)
        return position, rotation

    def n_waypoints(self) -> int:
        return len(self._t)

    def __repr__(self) -> str:
        return f"MovingListener({len(self._t)} waypoints, durée={self.duration}s)"
