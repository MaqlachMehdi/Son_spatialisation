"""
geometry.py
-----------
Primitives géométriques partagées pour le passage repère monde <-> repère
auditeur (tête). Convention SOFA dans tout le module : x = devant, y = gauche,
z = haut ; azimut 0°=devant/90°=gauche/180°=derrière/270°=droite ;
élévation -90°=dessous/0°=horizontal/90°=dessus.

Ces fonctions sont volontairement indépendantes de Trajectory/Listener pour
rester réutilisables sans dépendance circulaire.
"""

from __future__ import annotations

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
# Conversions sphérique <-> cartésien
# ══════════════════════════════════════════════════════════════════════════════

def spherical_to_cartesian(az_deg: float, el_deg: float, r: float) -> np.ndarray:
    """(azimut°, élévation°, r) -> vecteur cartésien (x, y, z), shape (3,)."""
    az = np.deg2rad(az_deg)
    el = np.deg2rad(el_deg)
    return r * np.array([
        np.cos(el) * np.cos(az),
        np.cos(el) * np.sin(az),
        np.sin(el),
    ])


def cartesian_to_spherical(vec: np.ndarray) -> tuple[float, float, float]:
    """Vecteur cartésien (x, y, z) -> (azimut°, élévation°, r)."""
    r = float(np.linalg.norm(vec))
    if r < 1e-9:
        return 0.0, 0.0, 0.0
    az = float(np.degrees(np.arctan2(vec[1], vec[0])) % 360.0)
    el = float(np.degrees(np.arcsin(np.clip(vec[2] / r, -1.0, 1.0))))
    return az, el, r


# ══════════════════════════════════════════════════════════════════════════════
# Orientation — matrice de rotation tête -> monde
# ══════════════════════════════════════════════════════════════════════════════

def rotation_from_ypr(yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """
    Matrice de rotation R (3x3) tête -> monde, à partir des angles de Tait-Bryan
    lacet (yaw) / tangage (pitch) / roulis (roll), composée R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

    Conventions (cohérentes avec les signes d'azimut/élévation SOFA) :
      yaw   > 0 : la tête tourne vers la GAUCHE (le nez va vers +y).
      pitch > 0 : la tête se LÈVE (le nez va vers +z).
      roll  > 0 : la tête PENCHE À GAUCHE (l'oreille gauche descend vers -z).

    Un point du monde situé pile devant l'auditeur (az=0, el=0) et vu par une
    tête tournée de +yaw apparaît, dans le repère tête, à l'azimut -yaw (il
    semble s'être déplacé vers la droite relativement au nez) — comportement
    attendu physiquement.
    """
    y = np.deg2rad(yaw_deg)
    p = np.deg2rad(pitch_deg)
    r = np.deg2rad(roll_deg)

    Rz = np.array([
        [np.cos(y), -np.sin(y), 0.0],
        [np.sin(y),  np.cos(y), 0.0],
        [0.0,        0.0,       1.0],
    ])
    Ry = np.array([
        [ np.cos(p), 0.0, -np.sin(p)],
        [ 0.0,       1.0,  0.0      ],
        [ np.sin(p), 0.0,  np.cos(p)],
    ])
    Rx = np.array([
        [1.0, 0.0,        0.0      ],
        [0.0, np.cos(r),  np.sin(r)],
        [0.0, -np.sin(r), np.cos(r)],
    ])
    return Rz @ Ry @ Rx


# ══════════════════════════════════════════════════════════════════════════════
# Quaternions — pour une interpolation d'orientation sans gimbal lock
# ══════════════════════════════════════════════════════════════════════════════

def quaternion_from_rotation(R: np.ndarray) -> np.ndarray:
    """Matrice de rotation (3x3) -> quaternion unitaire [w, x, y, z]."""
    trace = np.trace(R)
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def rotation_from_quaternion(q: np.ndarray) -> np.ndarray:
    """Quaternion unitaire [w, x, y, z] -> matrice de rotation (3x3)."""
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])


def slerp_quaternion(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    """
    Interpolation sphérique (SLERP) entre deux quaternions unitaires.

    Évite les artefacts de l'interpolation linéaire d'angles d'Euler
    (vitesse de rotation non uniforme, gimbal lock) lors du lissage d'une
    orientation de tête entre deux échantillons de tracking.
    """
    q0 = q0 / np.linalg.norm(q0)
    q1 = q1 / np.linalg.norm(q1)

    dot = float(np.dot(q0, q1))
    if dot < 0.0:
        q1, dot = -q1, -dot  # plus court chemin sur l'hypersphère

    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        result = q0 + alpha * (q1 - q0)
        return result / np.linalg.norm(result)

    theta_0 = np.arccos(dot)
    theta   = theta_0 * alpha
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


def slerp_rotation(R0: np.ndarray, R1: np.ndarray, alpha: float) -> np.ndarray:
    """Interpolation sphérique entre deux matrices de rotation (via quaternions)."""
    q0 = quaternion_from_rotation(R0)
    q1 = quaternion_from_rotation(R1)
    return rotation_from_quaternion(slerp_quaternion(q0, q1, alpha))
