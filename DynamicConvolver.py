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

from SegmentEngine import SegmentEngine
from hrtf import HRTF
from Trajectory import Trajectory


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
        Trajectoire spatiale.
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
        self.sr         = int(sr)

        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 2:
            signal = signal.mean(axis=1).astype(np.float32)
        if signal.ndim != 1:
            raise ValueError("signal doit etre mono (1D) ou stereo (2D).")
        self.signal = signal

        if hop_ms is not None:
            from WOLAEngine import WOLAEngine
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

    def _compute_segment_positions(self) -> list[tuple[float, float]]:
        """Evalue trajectory.get_position(t_center) pour chaque bloc."""
        n_blocks  = int(np.ceil(len(self.signal) / self._stride_samples))
        stride_s  = self._stride_samples / self.sr

        positions: list[tuple[float, float]] = []
        for i in range(n_blocks):
            t_center = (i + 0.5) * stride_s
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
        """Gain 1/r par bloc si la trajectoire expose get_distance()."""
        if not hasattr(self.trajectory, "get_distance"):
            return None

        n_blocks = int(np.ceil(len(self.signal) / self._stride_samples))
        stride_s = self._stride_samples / self.sr
        R_ref    = getattr(self.trajectory, "R", 2.06)

        gains = []
        for i in range(n_blocks):
            t_center = (i + 0.5) * stride_s
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

        from WOLAEngine import WOLAEngine
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
        from WOLAEngine import WOLAEngine
        if isinstance(self.engine, WOLAEngine):
            params = f"hop={self.engine.hop_ms:.0f} ms"
        else:
            params = f"segment={self.engine.segment_ms:.0f} ms, overlap={self.engine.overlap_ms:.0f} ms"
        return (
            f"DynamicConvolver(sr={self.sr} Hz, "
            f"moteur={engine_name}, {params}, "
            f"rendu={'oui' if done else 'non'})"
        )
