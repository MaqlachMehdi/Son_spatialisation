"""
DynamicConvolver.py
-------------------
Spatialisation binaurale dynamique par convolution segmentee.

Supporte deux moteurs de convolution :
  • SegmentEngine  — moteur historique (segment_ms + overlap_ms + crossfade_type)
  • WOLAEngine     — Weighted Overlap-Add, reconstruction parfaite (hop_ms seul)

Pour utiliser WOLAEngine, passer uniquement hop_ms (segment_ms doit rester None).
Pour utiliser SegmentEngine, passer segment_ms + overlap_ms (comportement historique).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from .SegmentEngine import SegmentEngine
from hrtf import HRTF
from scene import Trajectory, Listener
from scene.geometry import spherical_to_cartesian, cartesian_to_spherical


class DynamicConvolver:
    """
    Convolution dynamique d'un signal mono selon une trajectoire.

    Workflow:
    1. Evaluer la trajectoire au centre de chaque bloc (segment ou hop).
    2. Recuperer une paire HRIR (L, R) par bloc.
    3. Rendre un signal stereo binaural via le moteur choisi.

    Paramètres
    ----------
    hrtf : HRTF | HRTFInterpolator | HRTFGen
        Source de HRIRs (tout objet exposant get_hrir()).
    signal : np.ndarray
        Signal mono (ou stereo converti en mono).
    sr : int
        Fréquence d'échantillonnage.
    trajectory : Trajectory
        Trajectoire spatiale de la source.
    listener : Listener | None
        Auditeur mobile (position + orientation dans le repère monde).
        Si None (défaut) : auditeur implicite fixe à l'origine, comportement
        historique inchangé — trajectory.get_position()/get_distance() sont
        utilisés directement comme avant.
        Si fourni : la position de la source est réinterprétée comme un point
        du repère MONDE (sur une sphère de rayon trajectory.R ou
        get_distance(), centrée à l'origine), puis reprojetée dans le repère
        de la tête à chaque bloc via la pose de l'auditeur.
    segment_ms : float | None
        [SegmentEngine] Durée d'un segment. Incompatible avec hop_ms.
    overlap_ms : float | None
        [SegmentEngine] Durée du recouvrement. Requis si segment_ms fourni.
    crossfade_type : str
        [SegmentEngine] Type d'enveloppe de crossfade.
    hop_ms : float | None
        [WOLAEngine] Pas de hop. Incompatible avec segment_ms.
        Si fourni, WOLAEngine est utilisé (reconstruction parfaite, COLA Hann).

    Attributs publics
    -----------------
    output : np.ndarray | None
        Signal stereo rendu, shape (N, 2), disponible apres run().
    """

    def __init__(
        self,
        hrtf:           HRTF,
        signal:         np.ndarray,
        sr:             int,
        trajectory:     Trajectory,
        segment_ms:     float | None = None,
        overlap_ms:     float | None = None,
        crossfade_type: str          = "cosine",
        hop_ms:         float | None = None,
        listener:       Listener | None = None,
    ) -> None:
        if hop_ms is not None and segment_ms is not None:
            raise ValueError(
                "hop_ms et segment_ms sont mutuellement exclusifs. "
                "Choisissez WOLAEngine (hop_ms) ou SegmentEngine (segment_ms + overlap_ms)."
            )
        if hop_ms is None and segment_ms is None:
            raise ValueError(
                "Fournissez soit hop_ms (WOLAEngine) soit segment_ms + overlap_ms (SegmentEngine)."
            )

        self.hrtf       = hrtf
        self.trajectory = trajectory
        self.listener   = listener
        self.sr         = int(sr)

        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 2:
            signal = signal.mean(axis=1).astype(np.float32)
        if signal.ndim != 1:
            raise ValueError("signal doit etre mono (1D) ou stereo (2D).")
        self.signal = signal

        if hop_ms is not None:
            from .WOLAEngine import WOLAEngine
            hrir_probe = hrtf.get_hrir(0.0, 0.0)[0]
            self.engine = WOLAEngine(
                sr       = self.sr,
                hop_ms   = hop_ms,
                hrir_len = len(hrir_probe),
            )
        else:
            if overlap_ms is None:
                raise ValueError("overlap_ms est requis quand segment_ms est fourni.")
            self.engine = SegmentEngine(
                sr             = self.sr,
                segment_ms     = segment_ms,
                overlap_ms     = overlap_ms,
                crossfade_type = crossfade_type,
            )

        self.output: np.ndarray | None = None

    # ──────────────────────────────────────────────────────────────────────

    @property
    def _stride_samples(self) -> int:
        """Pas en échantillons, compatible SegmentEngine et WOLAEngine."""
        return getattr(self.engine, "hop_samples", None) or self.engine.segment_samples

    def _listener_relative(self, t: float) -> tuple[float, float, float]:
        """
        Position de la source (az°, el°, r) dans le repère de la tête, à l'instant t.

        1. La position de la source est interprétée comme un point du repère
           MONDE : S(t), sur une sphère de rayon trajectory.get_distance(t)
           (ou trajectory.R à défaut, 2.06 m par défaut), centrée à l'origine.
        2. L'auditeur a une pose (position L(t), rotation R(t) tête->monde)
           dans ce même repère monde (Listener.get_pose).
        3. d(t) = R(t)ᵀ · (S(t) − L(t))  — position relative à la tête
           (Rᵀ = R⁻¹ car R est orthonormale).
        4. Conversion cartésien -> sphérique pour l'appel HRTF / le gain 1/r.

        Avec un StaticListener à l'origine et orientation neutre, ce calcul
        redonne exactement (az_source, el_source, R_ref) — comportement
        identique au cas sans auditeur, aux erreurs d'arrondi flottant près.
        """
        az_src, el_src = self.trajectory.get_position(t)
        get_distance = getattr(self.trajectory, "get_distance", None)
        r_src = get_distance(t) if callable(get_distance) else getattr(self.trajectory, "R", 2.06)

        S = spherical_to_cartesian(az_src, el_src, r_src)
        L, R = self.listener.get_pose(t)
        return cartesian_to_spherical(R.T @ (S - L))

    def _compute_segment_positions(self) -> list[tuple[float, float]]:
        """
        Evalue la position angulaire de la source pour chaque bloc.

        Sans listener : trajectory.get_position(t_center) directement
        (comportement historique).
        Avec listener : position reprojetée dans le repère de la tête,
        cf. _listener_relative().
        """
        n_blocks  = int(np.ceil(len(self.signal) / self._stride_samples))
        stride_s  = self._stride_samples / self.sr

        positions: list[tuple[float, float]] = []
        for i in range(n_blocks):
            t_center = (i + 0.5) * stride_s
            if self.listener is not None:
                az, el, _ = self._listener_relative(t_center)
            else:
                az, el = self.trajectory.get_position(t_center)
            positions.append((float(az), float(el)))

        return positions

    def _fetch_hrirs(
        self,
        positions: list[tuple[float, float]],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Recupere une paire HRIR (L, R) par position."""
        hrirs: list[tuple[np.ndarray, np.ndarray]] = []
        get_nearest = getattr(self.hrtf, "get_nearest_hrir", None)

        for az, el in positions:
            if callable(get_nearest):
                hrir_l, hrir_r = get_nearest(az, el)
            else:
                hrir_l, hrir_r = self.hrtf.get_hrir(azimuth=az, elevation=el)

            hrirs.append((
                np.asarray(hrir_l, dtype=np.float32),
                np.asarray(hrir_r, dtype=np.float32),
            ))

        return hrirs

    def _compute_segment_gains(self) -> list[float] | None:
        """
        Gain 1/r par bloc.

        Sans listener : uniquement si la trajectoire expose get_distance()
        (comportement historique).
        Avec listener : toujours calculé, distance mesurée depuis la pose de
        l'auditeur (cf. _listener_relative), même si la trajectoire de la
        source n'expose pas get_distance() (source alors supposée sur la
        sphère de rayon trajectory.R, 2.06 m par défaut).
        """
        if self.listener is None and not hasattr(self.trajectory, "get_distance"):
            return None

        n_blocks = int(np.ceil(len(self.signal) / self._stride_samples))
        stride_s = self._stride_samples / self.sr
        R_ref    = getattr(self.trajectory, "R", 2.06)

        gains = []
        for i in range(n_blocks):
            t_center = (i + 0.5) * stride_s
            if self.listener is not None:
                _, _, r = self._listener_relative(t_center)
            else:
                r = self.trajectory.get_distance(t_center)
            gains.append(float(R_ref / max(r, 0.01)))
        return gains

    def run(self) -> np.ndarray:
        """
        Orchestration complete: positions -> HRIRs -> gains -> rendu binaural.

        Retourne
        --------
        np.ndarray, shape (N, 2)
        """
        positions = self._compute_segment_positions()
        hrirs     = self._fetch_hrirs(positions)
        gains     = self._compute_segment_gains()

        from .WOLAEngine import WOLAEngine
        if isinstance(self.engine, WOLAEngine):
            self.output = self.engine.render(self.signal, hrirs, gains_per_hop=gains)
        else:
            self.output = self.engine.render(self.signal, hrirs, gains_per_segment=gains)

        return self.output

    def save(
        self,
        path_L:      str | Path,
        path_R:      str | Path,
        path_merged: str | Path,
    ) -> None:
        """Sauvegarde gauche, droite et stereo fusionne."""
        if self.output is None:
            raise RuntimeError("Appelez run() avant save().")

        left  = self.output[:, 0].astype(np.float32)
        right = self.output[:, 1].astype(np.float32)

        sf.write(str(path_L),      left,          self.sr)
        sf.write(str(path_R),      right,         self.sr)
        sf.write(str(path_merged), self.output.astype(np.float32), self.sr)

    def __repr__(self) -> str:
        done        = self.output is not None
        engine_name = type(self.engine).__name__
        from .WOLAEngine import WOLAEngine
        if isinstance(self.engine, WOLAEngine):
            params = f"hop={self.engine.hop_ms:.0f} ms"
        else:
            params = f"segment={self.engine.segment_ms:.0f} ms, overlap={self.engine.overlap_ms:.0f} ms"
        return (
            f"DynamicConvolver(sr={self.sr} Hz, "
            f"moteur={engine_name}, {params}, "
            f"auditeur={'mobile' if self.listener is not None else 'fixe'}, "
            f"rendu={'oui' if done else 'non'})"
        )
