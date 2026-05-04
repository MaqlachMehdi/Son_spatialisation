"""
Trajectory.py
-------------
Définit comment une source sonore se déplace dans l'espace au fil du temps.

Convention angulaire (SOFA) utilisée dans tout ce module :
  Azimut   : 0° = devant  ·  90° = gauche  ·  180° = derrière  ·  270° = droite
  Élévation : −90° = dessous  ·  0° = horizontal  ·  90° = dessus

Classes
-------
Trajectory          (ABC) — interface commune à toutes les trajectoires
CircularTrajectory  — rotation circulaire à élévation constante
LinearTrajectory    — déplacement rectiligne entre deux positions angulaires
CustomTrajectory    — chemin libre défini par des points de passage interpolés
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ══════════════════════════════════════════════════════════════════════════════
# Classe de base abstraite
# ══════════════════════════════════════════════════════════════════════════════

class Trajectory(ABC):
    """
    Interface commune à toutes les trajectoires.

    Une trajectoire convertit un instant t (secondes) en position angulaire
    (azimut°, élévation°) dans la convention SOFA.

    Toutes les sous-classes doivent implémenter :
      • duration   (property) — durée totale en secondes
      • get_position(t)       — position à l'instant t

    Toutes les méthodes utilitaires et de visualisation sont héritées.
    """

    # ── Contrat à implémenter ──────────────────────────────────────────────

    @property
    @abstractmethod
    def duration(self) -> float:
        """Durée totale de la trajectoire en secondes."""
        ...

    @abstractmethod
    def get_position(self, t: float) -> tuple[float, float]:
        """Retourne (azimut°, élévation°) à l'instant t (secondes)."""
        ...

    # ── Méthodes utilitaires partagées ────────────────────────────────────

    def get_positions(self, t_array: np.ndarray) -> np.ndarray:
        """
        Évalue la trajectoire sur un tableau d'instants.

        Paramètres
        ----------
        t_array : np.ndarray, shape (N,)
            Instants en secondes.

        Retourne
        --------
        np.ndarray, shape (N, 2)  — colonnes [azimut°, élévation°]
        """
        return np.array([self.get_position(float(t)) for t in t_array])

    def sample_segments(self, signal_duration_s: float, segment_ms: float) -> np.ndarray:
        """
        Génère une position par segment audio, évaluée au centre du segment.

        Utilisé par DynamicConvolver pour associer une HRIR à chaque bloc.

        Paramètres
        ----------
        signal_duration_s : float
            Durée totale du signal audio à spatialiser (secondes).
        segment_ms : float
            Durée d'un segment en millisecondes.

        Retourne
        --------
        np.ndarray, shape (N_segments, 2)  — [azimut°, élévation°]
        """
        segment_s  = segment_ms / 1000.0
        n_segments = int(np.ceil(signal_duration_s / segment_s))
        centers    = np.arange(n_segments) * segment_s + segment_s / 2.0
        centers    = np.clip(centers, 0.0, self.duration)
        return self.get_positions(centers)

    # ── Helpers privés partagés ───────────────────────────────────────────

    @staticmethod
    def _to_cartesian(
        az_deg: np.ndarray, el_deg: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convertit (azimut°, élévation°) en coordonnées cartésiennes unitaires.

        Convention SOFA : x = devant, y = gauche, z = haut.
        """
        az = np.deg2rad(az_deg)
        el = np.deg2rad(el_deg)
        return (
            np.cos(el) * np.cos(az),   # x → devant
            np.cos(el) * np.sin(az),   # y → gauche
            np.sin(el),                # z → haut
        )

    @staticmethod
    def _make_segments_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Construit un tableau (N−1, 2, 2) de segments pour LineCollection."""
        pts = np.stack([a, b], axis=1).reshape(-1, 1, 2)
        return np.concatenate([pts[:-1], pts[1:]], axis=1)

    @staticmethod
    def _make_segments_3d(
        x: np.ndarray, y: np.ndarray, z: np.ndarray
    ) -> np.ndarray:
        """Construit un tableau (N−1, 2, 3) de segments pour Line3DCollection."""
        pts = np.stack([x, y, z], axis=1).reshape(-1, 1, 3)
        return np.concatenate([pts[:-1], pts[1:]], axis=1)

    # ── Visualisations ────────────────────────────────────────────────────

    def plot(self, n_points: int = 500) -> None:
        """
        Deux vues complémentaires :
          • Courbes temporelles azimut / élévation (domaine temps).
          • Vue polaire de dessus avec gradient de couleur (bleu→rouge = début→fin).

        Paramètres
        ----------
        n_points : int
            Résolution de l'affichage (nombre de points échantillonnés).
        """
        t      = np.linspace(0.0, self.duration, n_points)
        pos    = self.get_positions(t)
        az, el = pos[:, 0], pos[:, 1]
        t_mid  = (t[:-1] + t[1:]) / 2.0  # centre de chaque segment

        fig = plt.figure(figsize=(13, 5))
        fig.suptitle(repr(self), fontsize=11, fontweight="bold")

        # ── Courbes temporelles ───────────────────────────────────────────
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.plot(t, az, color="#2196F3", linewidth=1.5, label="Azimut (°)")
        ax1.plot(t, el, color="#FF9800", linewidth=1.5, linestyle="--", label="Élévation (°)")
        ax1.set_xlabel("Temps (s)")
        ax1.set_ylabel("Angle (°)")
        ax1.set_title("Évolution angulaire")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # ── Vue polaire (plan horizontal, vu de dessus) ───────────────────
        ax2 = fig.add_subplot(1, 2, 2, projection="polar")
        segs = self._make_segments_2d(np.deg2rad(az), np.ones(n_points))
        lc   = LineCollection(segs, cmap="cool", linewidth=2.5)
        lc.set_array(t_mid / self.duration)
        ax2.add_collection(lc)

        ax2.scatter(np.deg2rad(az[0]),  1.0, s=100, color="lime", zorder=5, label="Début")
        ax2.scatter(np.deg2rad(az[-1]), 1.0, s=100, color="red",  zorder=5, marker="s", label="Fin")

        # Convention visuelle : 0° = devant = haut, 90° = gauche = gauche
        ax2.set_theta_zero_location("N")
        ax2.set_theta_direction(1)       # CCW → 90° apparaît à gauche (SOFA)
        ax2.set_rlim(0.0, 1.4)
        ax2.set_rticks([])

        # Annotations cardinales
        for angle, label in [(0, "Devant"), (90, "Gauche"), (180, "Derrière"), (270, "Droite")]:
            ax2.annotate(
                label,
                xy=(np.deg2rad(angle), 1.35),
                ha="center", va="center", fontsize=8, color="dimgray",
            )

        ax2.set_title("Vue de dessus (plan horizontal)", pad=20)
        ax2.legend(loc="upper right", bbox_to_anchor=(1.45, 1.1))
        fig.colorbar(lc, ax=ax2, pad=0.2, shrink=0.55, label="Temps normalisé")

        plt.tight_layout()
        plt.show()

    def plot_sphere(self, n_points: int = 500) -> None:
        """
        Trajectoire projetée sur une sphère 3D unitaire.

        Le gradient de couleur (bleu→rouge) représente l'évolution temporelle.
        Les flèches grises indiquent les directions cardinales (SOFA).

        Paramètres
        ----------
        n_points : int
            Résolution de l'affichage.
        """
        t     = np.linspace(0.0, self.duration, n_points)
        pos   = self.get_positions(t)
        x, y, z = self._to_cartesian(pos[:, 0], pos[:, 1])
        t_mid   = (t[:-1] + t[1:]) / 2.0

        fig = plt.figure(figsize=(8, 7))
        ax  = fig.add_subplot(111, projection="3d")
        fig.suptitle(repr(self), fontsize=11, fontweight="bold")

        # ── Sphère de référence transparente ─────────────────────────────
        u  = np.linspace(0.0, 2.0 * np.pi, 40)
        v  = np.linspace(-np.pi / 2.0, np.pi / 2.0, 20)
        xs = np.outer(np.cos(v), np.cos(u))
        ys = np.outer(np.cos(v), np.sin(u))
        zs = np.outer(np.sin(v), np.ones_like(u))
        ax.plot_surface(xs, ys, zs, alpha=0.07, color="lightblue")
        ax.plot_wireframe(xs, ys, zs, alpha=0.06, color="gray", linewidth=0.4)

        # ── Axes cardinaux annotés ────────────────────────────────────────
        for (dx, dy, dz), lbl in [
            (( 1,  0,  0), "Devant"),
            ((-1,  0,  0), "Derrière"),
            (( 0,  1,  0), "Gauche"),
            (( 0, -1,  0), "Droite"),
            (( 0,  0,  1), "Haut"),
            (( 0,  0, -1), "Bas"),
        ]:
            ax.quiver(
                0, 0, 0, dx * 1.25, dy * 1.25, dz * 1.25,
                color="gray", alpha=0.35, arrow_length_ratio=0.12,
            )
            ax.text(
                dx * 1.5, dy * 1.5, dz * 1.5, lbl,
                fontsize=7, ha="center", va="center", color="dimgray",
            )

        # ── Trajectoire avec gradient couleur ────────────────────────────
        segs = self._make_segments_3d(x, y, z)
        lc   = Line3DCollection(segs, cmap="cool", linewidth=2.5)
        lc.set_array(t_mid / self.duration)
        ax.add_collection(lc)

        # ── Marqueurs début / fin ─────────────────────────────────────────
        ax.scatter(x[0],  y[0],  z[0],  s=90, color="lime", zorder=5, label="Début")
        ax.scatter(x[-1], y[-1], z[-1], s=90, color="red",  zorder=5, marker="s", label="Fin")

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.set_zlim(-1.3, 1.3)
        ax.set_xlabel("X (devant)")
        ax.set_ylabel("Y (gauche)")
        ax.set_zlabel("Z (haut)")
        ax.set_title("Trajectoire sur la sphère", pad=10)
        ax.legend(loc="upper left")
        fig.colorbar(lc, ax=ax, pad=0.1, shrink=0.55, label="Temps normalisé")

        plt.tight_layout()
        plt.show()

    def plot_all(self, n_points: int = 500) -> None:
        """Affiche plot() puis plot_sphere() en séquence."""
        self.plot(n_points=n_points)
        self.plot_sphere(n_points=n_points)


# ══════════════════════════════════════════════════════════════════════════════
# EllipseTrajectory — ellipse dans l'espace azimut / élévation
# ══════════════════════════════════════════════════════════════════════════════

class EllipseTrajectory(Trajectory):
    """
    Trajectoire elliptique paramétrée dans l'espace (azimut, élévation).

    La source décrit une ellipse dont le demi-grand axe contrôle l'amplitude
    azimutale et le demi-petit axe contrôle l'amplitude en élévation.

    Paramètres
    ----------
    duration_s : float
        Durée totale du rendu audio en secondes.
    period_s : float
        Durée d'un tour complet de l'ellipse (secondes).
    az_amplitude : float
        Demi-amplitude azimutale en degrés (défaut : 90°).
        L'azimut varie dans [center_az - az_amplitude, center_az + az_amplitude].
    el_amplitude : float
        Demi-amplitude en élévation en degrés (défaut : 30°).
        L'élévation varie dans [center_el - el_amplitude, center_el + el_amplitude].
    center_az : float
        Azimut central de l'ellipse en degrés (défaut : 0°).
    center_el : float
        Élévation centrale de l'ellipse en degrés (défaut : 0°).
    clockwise : bool
        True  → sens horaire vu de dessus.
        False → sens anti-horaire.

    Exemple
    -------
    >>> traj = EllipseTrajectory(duration_s=10, period_s=3,
    ...                          az_amplitude=90, el_amplitude=30)
    >>> az, el = traj.get_position(t=0.75)
    >>> points = traj.get_positions(np.linspace(0, 10, 500))
    """

    def __init__(
        self,
        duration_s: float,
        period_s: float,
        az_amplitude: float = 90.0,
        el_amplitude: float = 30.0,
        center_az: float = 0.0,
        center_el: float = 0.0,
        clockwise: bool = True,
    ) -> None:
        if duration_s <= 0.0:
            raise ValueError(f"duration_s doit être > 0 (reçu : {duration_s}).")
        if period_s <= 0.0:
            raise ValueError(f"period_s doit être > 0 (reçu : {period_s}).")

        self._duration    = float(duration_s)
        self.period_s     = float(period_s)
        self.az_amplitude = float(az_amplitude)
        self.el_amplitude = float(el_amplitude)
        self.center_az    = float(center_az)
        self.center_el    = float(center_el)
        self._direction   = -1.0 if clockwise else 1.0

    @property
    def duration(self) -> float:
        return self._duration

    def get_position(self, t: float) -> tuple[float, float]:
        """Retourne (azimut°, élévation°) à l'instant t."""
        angle = self._direction * 2.0 * np.pi * t / self.period_s
        az = (self.center_az + self.az_amplitude * np.sin(angle)) % 360.0
        el = float(np.clip(
            self.center_el + self.el_amplitude * np.cos(angle),
            -90.0, 90.0
        ))
        return float(az), el

    def sample_points(self, n_points: int = 360) -> np.ndarray:
        """
        Renvoie n_points positions régulièrement espacées sur un tour complet.

        Retourne
        --------
        np.ndarray, shape (n_points, 2)  — colonnes [azimut°, élévation°]
        """
        t = np.linspace(0.0, self.period_s, n_points, endpoint=False)
        return self.get_positions(t)

    def __repr__(self) -> str:
        sens = "horaire" if self._direction < 0 else "anti-horaire"
        return (
            f"EllipseTrajectory("
            f"durée={self._duration}s, période={self.period_s}s, "
            f"az±{self.az_amplitude}°, el±{self.el_amplitude}°, "
            f"centre=({self.center_az}°, {self.center_el}°), {sens})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CircularTrajectory — rotation circulaire
# ══════════════════════════════════════════════════════════════════════════════

class CircularTrajectory(Trajectory):
    """
    Rotation circulaire à élévation constante.

    Paramètres
    ----------
    duration_s : float
        Durée totale du rendu audio en secondes.
    period_s : float
        Durée d'un tour complet en secondes.
        Ex : period_s=2  → rotation rapide (1 tour en 2 s).
             period_s=10 → rotation lente  (1 tour en 10 s).
    elevation : float
        Élévation constante en degrés (défaut : 0° = plan horizontal).
    start_az : float
        Azimut de départ en degrés (défaut : 0° = devant).
    clockwise : bool
        True  → sens horaire vu de dessus : devant → droite → derrière → gauche.
        False → sens anti-horaire          : devant → gauche → derrière → droite.

    Exemple
    -------
    >>> traj = CircularTrajectory(duration_s=10, period_s=2.5)
    >>> az, el = traj.get_position(t=1.25)   # quart de tour
    """

    def __init__(
        self,
        duration_s: float,
        period_s: float,
        elevation: float = 0.0,
        start_az: float = 0.0,
        clockwise: bool = True,
    ) -> None:
        if duration_s <= 0.0:
            raise ValueError(f"duration_s doit être > 0 (reçu : {duration_s}).")
        if period_s <= 0.0:
            raise ValueError(f"period_s doit être > 0 (reçu : {period_s}).")

        self._duration = float(duration_s)
        self.period_s  = float(period_s)
        self.elevation = float(elevation)
        self.start_az  = float(start_az)
        self.clockwise = clockwise
        # Sens horaire → azimut diminue (0→270→180→90 en SOFA)
        self._direction = -1.0 if clockwise else 1.0

    @property
    def duration(self) -> float:
        return self._duration

    def get_position(self, t: float) -> tuple[float, float]:
        az = (self.start_az + self._direction * 360.0 * t / self.period_s) % 360.0
        return float(az), float(self.elevation)

    def n_rotations(self) -> float:
        """Nombre de tours complets effectués sur la durée totale."""
        return self._duration / self.period_s

    def __repr__(self) -> str:
        sens = "horaire" if self.clockwise else "anti-horaire"
        return (
            f"CircularTrajectory("
            f"durée={self._duration}s, "
            f"période={self.period_s}s, "
            f"élév={self.elevation}°, "
            f"start={self.start_az}°, "
            f"{sens})"
        )


# ══════════════════════════════════════════════════════════════════════════════
# LinearTrajectory — déplacement rectiligne A → B
# ══════════════════════════════════════════════════════════════════════════════

class LinearTrajectory(Trajectory):
    """
    Déplacement rectiligne d'une position angulaire à une autre.

    La trajectoire est tenue fixe avant t=0 et après t=duration_s.

    Paramètres
    ----------
    duration_s : float
        Durée totale du déplacement en secondes.
    start_az : float
        Azimut de départ en degrés.
    end_az : float
        Azimut d'arrivée en degrés.
    start_el : float
        Élévation de départ en degrés (défaut : 0°).
    end_el : float
        Élévation d'arrivée en degrés (défaut : 0°).

    Exemple
    -------
    >>> traj = LinearTrajectory(duration_s=5, start_az=0, end_az=180)
    >>> traj.get_position(2.5)   # (90.0, 0.0) — à mi-chemin
    """

    def __init__(
        self,
        duration_s: float,
        start_az: float,
        end_az: float,
        start_el: float = 0.0,
        end_el: float = 0.0,
    ) -> None:
        if duration_s <= 0.0:
            raise ValueError(f"duration_s doit être > 0 (reçu : {duration_s}).")

        self._duration = float(duration_s)
        self.start_az  = float(start_az)
        self.end_az    = float(end_az)
        self.start_el  = float(start_el)
        self.end_el    = float(end_el)

    @property
    def duration(self) -> float:
        return self._duration

    def get_position(self, t: float) -> tuple[float, float]:
        alpha = float(np.clip(t / self._duration, 0.0, 1.0))
        az    = self.start_az + alpha * (self.end_az - self.start_az)
        el    = self.start_el + alpha * (self.end_el - self.start_el)
        return float(az % 360.0), float(el)

    def __repr__(self) -> str:
        return (
            f"LinearTrajectory("
            f"durée={self._duration}s, "
            f"az: {self.start_az}°→{self.end_az}°, "
            f"él: {self.start_el}°→{self.end_el}°)"
        )


# ══════════════════════════════════════════════════════════════════════════════
# CustomTrajectory — chemin libre par waypoints
# ══════════════════════════════════════════════════════════════════════════════

class CustomTrajectory(Trajectory):
    """
    Trajectoire libre définie par des points de passage (waypoints) interpolés.

    Paramètres
    ----------
    waypoints : list[tuple[float, float, float]]
        Liste de points (t_s, azimut°, élévation°), dans n'importe quel ordre.
        Au moins 2 waypoints sont requis.
        Le premier waypoint définit t=0, le dernier définit la fin.
    interpolation : 'linear' | 'cosine'
        Méthode d'interpolation entre waypoints successifs.
        'linear' → vitesse constante entre les points.
        'cosine' → accélération/décélération douce aux extrémités de chaque segment.

    Exemple
    -------
    >>> traj = CustomTrajectory([
    ...     (0.0,   0.0,  0.0),   # devant, horizontal
    ...     (3.0,  90.0, 30.0),   # gauche, surélevé
    ...     (6.0, 180.0,  0.0),   # derrière, horizontal
    ... ], interpolation='cosine')
    """

    def __init__(
        self,
        waypoints: list[tuple[float, float, float]],
        interpolation: str = "linear",
    ) -> None:
        if len(waypoints) < 2:
            raise ValueError("Au moins 2 waypoints sont requis.")
        if interpolation not in ("linear", "cosine"):
            raise ValueError("interpolation doit être 'linear' ou 'cosine'.")

        waypoints = sorted(waypoints, key=lambda w: w[0])
        self._t            = np.array([w[0] for w in waypoints], dtype=float)
        self._az           = np.array([w[1] for w in waypoints], dtype=float)
        self._el           = np.array([w[2] for w in waypoints], dtype=float)
        self.interpolation = interpolation

    @property
    def duration(self) -> float:
        return float(self._t[-1])

    def get_position(self, t: float) -> tuple[float, float]:
        t = float(np.clip(t, self._t[0], self._t[-1]))

        # Segment encadrant t
        idx = int(np.searchsorted(self._t, t, side="right")) - 1
        idx = int(np.clip(idx, 0, len(self._t) - 2))

        t0, t1 = self._t[idx], self._t[idx + 1]
        dt     = t1 - t0
        alpha  = (t - t0) / dt if dt > 1e-12 else 0.0

        if self.interpolation == "cosine":
            # Lissage cosinus : accélération/décélération douce
            alpha = (1.0 - np.cos(np.pi * alpha)) / 2.0

        az = self._az[idx] + alpha * (self._az[idx + 1] - self._az[idx])
        el = self._el[idx] + alpha * (self._el[idx + 1] - self._el[idx])
        return float(az % 360.0), float(el)

    def n_waypoints(self) -> int:
        """Nombre de waypoints définis."""
        return len(self._t)

    def __repr__(self) -> str:
        return (
            f"CustomTrajectory("
            f"{len(self._t)} waypoints, "
            f"durée={self.duration}s, "
            f"interp='{self.interpolation}')"
        )
