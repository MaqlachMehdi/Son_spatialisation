"""
Soundscape.py
-------------
Paysage sonore multi-source : combine N sources spatialisées en un mix binaural.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf

from hrtf import HRTF
from .Soundsource import SoundSource


class Soundscape:
    """
    Paysage sonore composé de plusieurs sources spatialisées.

    Chaque source est convoluée indépendamment avec ses HRIRs, puis
    les canaux gauche/droite sont sommés et normalisés en un seul mix binaural.

    Paramètres
    ----------
    hrtf : HRTF
        Dataset HRTF commun à toutes les sources.

    Attributs publics (disponibles après render())
    ----------------------------------------------
    mix_left  : np.ndarray  — canal gauche du mix final
    mix_right : np.ndarray  — canal droit du mix final
    rendered  : np.ndarray  — mix stéréo shape (N, 2), prêt à écouter / exporter
    """

    def __init__(self, hrtf: HRTF) -> None:
        self.hrtf = hrtf
        self.sources: list[SoundSource] = []
        self.mix_left:  np.ndarray | None = None
        self.mix_right: np.ndarray | None = None
        self.rendered:  np.ndarray | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Gestion des sources
    # ──────────────────────────────────────────────────────────────────────

    def add_source(self, source: SoundSource) -> None:
        """Ajoute une SoundSource au paysage sonore."""
        self.sources.append(source)
        print(f"[Soundscape] Source ajoutée : {source}")

    # ──────────────────────────────────────────────────────────────────────
    # Rendu
    # ──────────────────────────────────────────────────────────────────────

    def render(self) -> np.ndarray:
        """
        Convolue chaque source, aligne les longueurs, somme et normalise.

        Retourne
        --------
        np.ndarray, shape (N, 2)
            Mix binaural stéréo normalisé (float32).

        Lève
        ----
        RuntimeError
            Si aucune source n'a été ajoutée.
        """
        if not self.sources:
            raise RuntimeError("Aucune source ajoutée. Appelez add_source() d'abord.")

        lefts:  list[np.ndarray] = []
        rights: list[np.ndarray] = []

        for i, source in enumerate(self.sources):
            print(f"\n[Soundscape] Traitement source {i + 1}/{len(self.sources)} — {source}")
            from engine import HRTFConvolver
            convolver = HRTFConvolver.from_source(self.hrtf, source)
            convolver.run()
            lefts.append(convolver.output_left)
            rights.append(convolver.output_right)

        self.mix_left, self.mix_right = self._mix(lefts, rights)
        self.rendered = np.stack([self.mix_left, self.mix_right], axis=1)

        print(f"\n[Soundscape] Rendu terminé : {self.rendered.shape[0]} échantillons "
              f"({self.rendered.shape[0] / self.hrtf.sample_rate:.2f}s)")
        return self.rendered

    def _mix(
        self,
        lefts:  list[np.ndarray],
        rights: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """Zero-pad toutes les pistes à la longueur max, somme, normalise."""
        max_len = max(len(s) for s in lefts)

        def pad(sig: np.ndarray) -> np.ndarray:
            if len(sig) < max_len:
                return np.pad(sig, (0, max_len - len(sig)))
            return sig

        mix_l = sum(pad(s) for s in lefts)
        mix_r = sum(pad(s) for s in rights)

        peak = max(np.max(np.abs(mix_l)), np.max(np.abs(mix_r))) + 1e-10
        return (mix_l / peak).astype(np.float32), (mix_r / peak).astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Export
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path = "soundscape.wav") -> None:
        """
        Sauvegarde le mix binaural en fichier WAV stéréo.

        Lève
        ----
        RuntimeError
            Si render() n'a pas été appelé avant.
        """
        if self.rendered is None:
            raise RuntimeError("Appelez render() avant save().")

        sf.write(str(path), self.rendered, self.hrtf.sample_rate)
        print(f"[Soundscape] Sauvegardé : {path}")

    # ──────────────────────────────────────────────────────────────────────
    # Représentation
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"Soundscape("
            f"{len(self.sources)} source(s), "
            f"rendu={'oui' if self.rendered is not None else 'non'})"
        )
