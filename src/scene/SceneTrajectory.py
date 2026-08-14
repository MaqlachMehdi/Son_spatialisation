"""
SceneTrajectory.py
-------------------
Rejoue côté serveur la trajectoire éditée dans BeInTheFlow (frontend), pour
que le rendu audio (DynamicSoundscape) suive exactement le même chemin que
l'aperçu 3D animé (TrajectoryPlayer.tsx).

Réplique trajectoryPointAt() (BeInTheFlow/src/utils/trajectorySampling.ts)
directement en coordonnées cartésiennes SOFA (x=devant, y=gauche, z=haut) —
pas de passage par les axes Three.js. Les deux repères sont reliés par une
transformation orthogonale (cf. BeInTheFlow/src/utils/sofaCoords.ts) : tilt
et lerp cartésien donnent donc géométriquement le même résultat calculés
directement en SOFA, sans conversion intermédiaire.

Trois types, un par valeur de TrajectoryDTO.type (BeInTheFlow/src/types.ts) :
  circular   — anneau à élévation constante, incliné (axis*) et décalé (offset*)
  elliptical — azimut/élévation en quadrature (sin/cos), pas de tilt/offset
  points     — boucle fermée de waypoints, interpolation linéaire cartésienne

IMPORTANT : BASE_PERIOD_SECONDS doit rester synchronisé avec la constante du
même nom dans BeInTheFlow/src/three/TrajectoryPlayer.tsx.
"""

from __future__ import annotations

import numpy as np

from .Trajectory import Trajectory

BASE_PERIOD_SECONDS = 8.0


def _cartesian_to_spherical(vec: np.ndarray) -> tuple[float, float, float]:
    """Cartésien SOFA (x,y,z) -> (azimut°, élévation°, distance)."""
    r = float(np.linalg.norm(vec))
    if r < 1e-9:
        return 0.0, 0.0, 0.0
    az = float(np.degrees(np.arctan2(vec[1], vec[0])) % 360.0)
    el = float(np.degrees(np.arcsin(np.clip(vec[2] / r, -1.0, 1.0))))
    return az, el, r


def _rotation_aligning(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Matrice 3x3 alignant le vecteur unitaire `a` sur `b` par la rotation de
    plus court angle — équivalent de THREE.Quaternion.setFromUnitVectors().
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    s = float(np.linalg.norm(v))

    if s < 1e-9:
        if c > 0.0:
            return np.eye(3)
        # a et b opposés : rotation de 180° autour d'un axe perpendiculaire
        axis = np.cross(a, [1.0, 0.0, 0.0])
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, [0.0, 1.0, 0.0])
        axis = axis / np.linalg.norm(axis)
        return 2.0 * np.outer(axis, axis) - np.eye(3)

    vx = np.array([
        [0.0, -v[2], v[1]],
        [v[2], 0.0, -v[0]],
        [-v[1], v[0], 0.0],
    ])
    return np.eye(3) + vx + vx @ vx * ((1.0 - c) / (s ** 2))


class SceneTrajectory(Trajectory):
    """
    Trajectoire construite depuis un TrajectoryPayload (schemas/render.py),
    tel qu'envoyé par le frontend — dict en snake_case (model_dump() de
    src/api/schemas/workspace.TrajectoryPayload).

    Paramètres
    ----------
    payload : dict
        Champs attendus : type, distance, center_azimuth, center_elevation,
        az_amplitude, el_amplitude, points, speed, reverse, axis_azimuth,
        axis_elevation, offset_azimuth, offset_elevation, offset_distance.
    duration_s : float
        Durée du signal audio à spatialiser (secondes) — informatif
        (get_position boucle indéfiniment, DynamicConvolver ne clippe pas).
    R : float
        Rayon de référence de la sphère HRTF (m), utilisé par
        DynamicConvolver pour le gain 1/r. Défaut : 2.06 m (IRCAM LISTEN).
    """

    def __init__(self, payload: dict, duration_s: float, R: float = 2.06) -> None:
        self.payload = payload
        self._duration = float(duration_s)
        self.type = payload["type"]
        self.speed = float(payload.get("speed", 1.0)) or 1.0
        self.reverse = bool(payload.get("reverse", False))
        self.period_s = BASE_PERIOD_SECONDS / max(self.speed, 0.01)
        self.R = float(R)

        if self.type == "points":
            points = payload.get("points") or []
            self._n = len(points)
            self._waypoints = np.array([
                self._to_cart_deg(p["azimuth"], p["elevation"], p["distance"])
                for p in points
            ]) if self._n else np.zeros((0, 3))
        elif self.type == "circular":
            axis_dir = self._to_cart_deg(payload["axis_azimuth"], payload["axis_elevation"], 1.0)
            self._tilt = _rotation_aligning(np.array([0.0, 0.0, 1.0]), axis_dir)
            self._offset = self._to_cart_deg(
                payload["offset_azimuth"], payload["offset_elevation"], payload["offset_distance"]
            )
        elif self.type != "elliptical":
            raise ValueError(f"Type de trajectoire inconnu : {self.type!r}")

    @staticmethod
    def _to_cart_deg(az_deg: float, el_deg: float, r: float) -> np.ndarray:
        az, el = np.deg2rad(az_deg), np.deg2rad(el_deg)
        return r * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])

    @property
    def duration(self) -> float:
        return self._duration

    def _u(self, t: float) -> float:
        """Temps normalisé [0,1) sur une boucle, en tenant compte de speed/reverse."""
        forward = (t / self.period_s) % 1.0
        return (1.0 - forward) if self.reverse else forward

    def _position_cartesian(self, t: float) -> np.ndarray:
        u = self._u(t)
        p = self.payload

        if self.type == "circular":
            ring_point = self._to_cart_deg(360.0 * u, p["center_elevation"], p["distance"])
            return self._tilt @ ring_point + self._offset

        if self.type == "elliptical":
            angle = 2.0 * np.pi * u
            az = p["center_azimuth"] + p["az_amplitude"] * np.sin(angle)
            el = p["center_elevation"] + p["el_amplitude"] * np.cos(angle)
            return self._to_cart_deg(az, el, p["distance"])

        # points
        if self._n == 0:
            return np.zeros(3)
        if self._n == 1:
            return self._waypoints[0]
        seg_t = u * self._n
        i = int(seg_t) % self._n
        frac = seg_t - int(seg_t)
        a, b = self._waypoints[i], self._waypoints[(i + 1) % self._n]
        return a + (b - a) * frac

    def get_position(self, t: float) -> tuple[float, float]:
        az, el, _ = _cartesian_to_spherical(self._position_cartesian(t))
        return az, el

    def get_distance(self, t: float) -> float:
        """Distance instantanée (m) — active le gain 1/r dans DynamicConvolver."""
        return float(np.linalg.norm(self._position_cartesian(t)))

    def __repr__(self) -> str:
        return (
            f"SceneTrajectory(type={self.type!r}, période={self.period_s:.2f}s, "
            f"reverse={self.reverse}, durée={self._duration:.2f}s)"
        )
