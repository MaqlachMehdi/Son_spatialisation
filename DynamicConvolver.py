"""
DynamicConvolver.py
-------------------
Spatialisation binaurale dynamique par convolution segmentee.
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
    1. Decouper le signal en segments via SegmentEngine.
    2. Evaluer la trajectoire au centre de chaque segment.
    3. Recuperer une paire HRIR (L, R) par segment.
    4. Rendre un signal stereo binaural avec crossfade.

    Attributs publics
    -----------------
    output : np.ndarray | None
        Signal stereo rendu, shape (N, 2), disponible apres run().
    """

    def __init__(
        self,
        hrtf: HRTF,
        signal: np.ndarray,
        sr: int,
        trajectory: Trajectory,
        segment_ms: float,
        overlap_ms: float,
        crossfade_type: str = "cosine",
    ) -> None:
        self.hrtf = hrtf
        self.trajectory = trajectory
        self.sr = int(sr)

        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 2:
            signal = signal.mean(axis=1).astype(np.float32)
        if signal.ndim != 1:
            raise ValueError("signal doit etre mono (1D) ou stereo (2D).")

        self.signal = signal

        self.engine = SegmentEngine(
            sr=self.sr,
            segment_ms=segment_ms,
            overlap_ms=overlap_ms,
            crossfade_type=crossfade_type,
        )

        self.output: np.ndarray | None = None

    def _compute_segment_positions(self) -> list[tuple[float, float]]:
        """
        Evalue trajectory.get_position(t_center) pour chaque segment.

        Retourne
        --------
        list[tuple[float, float]]
            Liste de paires (azimuth_deg, elevation_deg).
        """
        n_segments = int(np.ceil(len(self.signal) / self.engine.segment_samples))
        segment_s = self.engine.segment_samples / self.sr

        positions: list[tuple[float, float]] = []
        for i in range(n_segments):
            t_center = (i + 0.5) * segment_s
            az, el = self.trajectory.get_position(t_center)
            positions.append((float(az), float(el)))

        return positions

    def _fetch_hrirs(
        self,
        positions: list[tuple[float, float]],
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """
        Recupere une paire HRIR (L, R) par position.

        Priorite: hrtf.get_nearest_hrir(az, el)
        Fallback: hrtf.get_hrir(azimuth=..., elevation=...)
        """
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

    def run(self) -> np.ndarray:
        """
        Orchestration complete: positions -> HRIRs -> rendu binaural.

        Retourne
        --------
        np.ndarray
            Sortie stereo shape (N, 2), egalement stockee dans self.output.
        """
        positions = self._compute_segment_positions()
        hrirs = self._fetch_hrirs(positions)
        self.output = self.engine.render(self.signal, hrirs)
        return self.output

    def save(
        self,
        path_L: str | Path,
        path_R: str | Path,
        path_merged: str | Path,
    ) -> None:
        """
        Sauvegarde gauche, droite et stereo fusionne.

        path_L : WAV mono (canal gauche)
        path_R : WAV mono (canal droit)
        path_merged : WAV stereo (L, R)
        """
        if self.output is None:
            raise RuntimeError("Appelez run() avant save().")

        path_L = Path(path_L)
        path_R = Path(path_R)
        path_merged = Path(path_merged)

        left = self.output[:, 0].astype(np.float32)
        right = self.output[:, 1].astype(np.float32)

        sf.write(str(path_L), left, self.sr)
        sf.write(str(path_R), right, self.sr)
        sf.write(str(path_merged), self.output.astype(np.float32), self.sr)

    def __repr__(self) -> str:
        done = self.output is not None
        return (
            f"DynamicConvolver(sr={self.sr} Hz, "
            f"segment={self.engine.segment_ms:.0f} ms, "
            f"overlap={self.engine.overlap_ms:.0f} ms, "
            f"rendu={'oui' if done else 'non'})"
        )
