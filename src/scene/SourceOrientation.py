"""
SourceOrientation.py
--------------------
Axe d'émission d'une source sonore ("où pointe la source") au fil du temps 
complément de Listener.py (qui gère la pose de l'auditeur), côté émission.

Un piston étant à symétrie de révolution (cf. Directivity.py), un simple
vecteur unitaire suffit à décrire l'orientation d'une source : pas besoin
d'une matrice de rotation 3x3 complète comme pour Listener (pas de notion
de "roll" pertinente pour un lobe axisymétrique).

Classes
-------
SourceOrientation    (ABC) — interface commune
FixedOrientation     — axe d'émission constant
VelocityOrientation  — la source pointe dans le sens de son déplacement
OrientationWaypoints — axe d'émission interpolé (grand cercle) entre waypoints
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .geometry import (
    cartesian_to_spherical,
    slerp_unit_vector,
    spherical_to_cartesian,
)


# ══════════════════════════════════════════════════════════════════════════════
# Classe de base abstraite
# ══════════════════════════════════════════════════════════════════════════════

class SourceOrientation(ABC):
    """Interface commune : direction d'émission (vecteur unitaire monde) à l'instant t."""

    @abstractmethod
    def get_direction(self, t: float) -> np.ndarray:
        """Vecteur unitaire (3,), repère monde (x=devant, y=gauche, z=haut)."""
        ...

    def get_direction_angles(self, t: float) -> tuple[float, float]:
        """Raccourci : direction d'émission sous forme (azimut°, élévation°)."""
        az, el, _ = cartesian_to_spherical(self.get_direction(t))
        return az, el


# ══════════════════════════════════════════════════════════════════════════════
# FixedOrientation — axe d'émission constant
# ══════════════════════════════════════════════════════════════════════════════

class FixedOrientation(SourceOrientation):
    """
    Axe d'émission constant, donné en (azimut°, élévation°) monde.

    Exemple
    -------
    >>> orient = FixedOrientation(azimuth=180.0, elevation=0.0)  # pointe vers -x
    """

    def __init__(self, azimuth: float = 0.0, elevation: float = 0.0) -> None:
        self.azimuth = float(azimuth)
        self.elevation = float(elevation)
        self._dir = spherical_to_cartesian(self.azimuth, self.elevation, 1.0)

    def get_direction(self, t: float) -> np.ndarray:
        return self._dir

    def __repr__(self) -> str:
        return f"FixedOrientation(az={self.azimuth}°, el={self.elevation}°)"


# ══════════════════════════════════════════════════════════════════════════════
# VelocityOrientation — la source pointe où elle va
# ══════════════════════════════════════════════════════════════════════════════

class VelocityOrientation(SourceOrientation):
    """
    L'axe d'émission suit la tangente de la trajectoire (dérivée numérique
    par différences finies centrées) : hypothèse "la source pointe dans le
    sens de son déplacement" (une voiture, un instrument en mouvement...).

    Ne convient pas à une source qui pivote sur place sans se déplacer (une
    tête qui parle en tournant, par ex.)  utiliser OrientationWaypoints
    ou FixedOrientation dans ce cas.

    Paramètres
    ----------
    trajectory : Trajectory
        Trajectoire de la source (interprétée en coordonnées MONDE, comme
        pour DynamicConvolver.listener — cf. get_distance()/R).
    R : float
        Rayon par défaut si la trajectoire n'expose pas get_distance()
        (défaut : 2.06 m, IRCAM LISTEN).
    dt : float
        Pas de temps pour la différence finie centrée (secondes, défaut 1 ms).
    """

    def __init__(self, trajectory, R: float = 2.06, dt: float = 1e-3) -> None:
        self.trajectory = trajectory
        self.R = float(R)
        self.dt = float(dt)

    def _world_position(self, t: float) -> np.ndarray:
        az, el = self.trajectory.get_position(t)
        get_distance = getattr(self.trajectory, "get_distance", None)
        r = get_distance(t) if callable(get_distance) else getattr(self.trajectory, "R", self.R)
        return spherical_to_cartesian(az, el, r)

    def get_direction(self, t: float) -> np.ndarray:
        duration = getattr(self.trajectory, "duration", None)
        t0 = max(t - self.dt / 2.0, 0.0)
        t1 = t + self.dt / 2.0
        if duration is not None:
            t1 = min(t1, float(duration))

        v = self._world_position(t1) - self._world_position(t0)
        norm = float(np.linalg.norm(v))
        if norm < 1e-9:
            return np.array([1.0, 0.0, 0.0])  # source immobile : convention "devant"
        return v / norm

    def __repr__(self) -> str:
        return f"VelocityOrientation(trajectory={self.trajectory!r})"


# ══════════════════════════════════════════════════════════════════════════════
# OrientationWaypoints  axe d'émission libre par waypoints
# ══════════════════════════════════════════════════════════════════════════════

class OrientationWaypoints(SourceOrientation):
    """
    Axe d'émission défini par des waypoints (t_s, azimut°, élévation°),
    interpolés par SLERP (grand cercle)  pas d'artefact près des pôles ni
    au franchissement de 0°/360°, contrairement à une interpolation linéaire
    directe des angles.

    La direction est tenue fixe avant le premier waypoint et après le dernier.

    Exemple
    -------
    >>> orient = OrientationWaypoints([
    ...     (0.0,   0.0, 0.0),   # pointe devant
    ...     (5.0, 180.0, 0.0),   # se retourne (pointe derrière) en 5 s
    ... ])
    """

    def __init__(self, waypoints: list[tuple[float, float, float]]) -> None:
        if len(waypoints) < 2:
            raise ValueError("Au moins 2 waypoints sont requis.")

        waypoints = sorted(waypoints, key=lambda w: w[0])
        self._t = np.array([w[0] for w in waypoints], dtype=float)
        self._dirs = np.array([
            spherical_to_cartesian(w[1], w[2], 1.0) for w in waypoints
        ])

    @property
    def duration(self) -> float:
        return float(self._t[-1])

    def get_direction(self, t: float) -> np.ndarray:
        t = float(np.clip(t, self._t[0], self._t[-1]))

        idx = int(np.searchsorted(self._t, t, side="right")) - 1
        idx = int(np.clip(idx, 0, len(self._t) - 2))

        t0, t1 = self._t[idx], self._t[idx + 1]
        dt = t1 - t0
        alpha = (t - t0) / dt if dt > 1e-12 else 0.0

        return slerp_unit_vector(self._dirs[idx], self._dirs[idx + 1], alpha)

    def __repr__(self) -> str:
        return f"OrientationWaypoints({len(self._t)} waypoints, durée={self.duration}s)"
