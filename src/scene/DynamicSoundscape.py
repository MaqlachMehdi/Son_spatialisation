"""
DynamicSoundscape.py
--------------------
Paysage sonore dynamique multi-source : plusieurs sources se déplacent
simultanément dans l'espace selon leurs trajectoires respectives.

Chaque source est convoluée indépendamment via un DynamicConvolver,
puis les canaux gauche/droite sont sommés et normalisés.

Classe
------
DynamicSoundscape
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import numpy as np
import soundfile as sf

from hrtf import HRTF
from .Trajectory import Trajectory
from .Listener import Listener


# ──────────────────────────────────────────────────────────────────────────────
# Dataclass interne — une entrée de source
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class _DynamicSource:
    """Représente une source avec son signal chargé, sa trajectoire et son gain."""
    signal:     np.ndarray
    sr:         int
    trajectory: Trajectory
    gain:       float = 1.0
    label:      str   = ""


# ══════════════════════════════════════════════════════════════════════════════
# DynamicSoundscape
# ══════════════════════════════════════════════════════════════════════════════

class DynamicSoundscape:
    """
    Paysage sonore dynamique : N sources spatialisées avec trajectoires variables.

    Paramètres
    ----------
    hrtf : HRTF
        Dataset HRTF commun à toutes les sources.
    segment_ms : float
        Durée d'un segment de convolution en millisecondes.
        Pour l'IRCAM LISTEN (pas = 15°) :
          segment_ms ≈ period_s / 24 * 1000  → un segment par position HRTF.
    overlap_ms : float | None
        Durée du crossfade entre segments.
        Si None : max(segment_ms × 0.2 , hrir_duration_ms × 1.1).
        Doit couvrir la queue HRIR (11.6 ms pour IRCAM LISTEN 44100 Hz).
    crossfade_type : 'linear' | 'cosine' | 'equal_power'
        Forme de l'enveloppe de crossfade (défaut : 'cosine').
    listener : Listener | None
        Auditeur commun à toutes les sources de la scène (position +
        orientation mobiles). Si None (défaut) : auditeur implicite fixe à
        l'origine, comportement historique inchangé.

    Attributs publics (disponibles après render())
    ----------------------------------------------
    mix_left  : np.ndarray  — canal gauche normalisé
    mix_right : np.ndarray  — canal droit normalisé
    rendered  : np.ndarray  — mix stéréo (N, 2)
    """

    def __init__(
        self,
        hrtf: HRTF,
        segment_ms: float,
        overlap_ms: float | None = None,
        crossfade_type: str = "cosine",
        listener: Listener | None = None,
    ) -> None:
        self.hrtf          = hrtf
        self.segment_ms    = float(segment_ms)
        self.crossfade_type = crossfade_type
        self.listener       = listener

        # overlap automatique : au moins 110 % de la durée HRIR
        if overlap_ms is None:
            self.overlap_ms = max(segment_ms * 0.2, hrtf.duration_ms * 1.1)
        else:
            self.overlap_ms = float(overlap_ms)

        self._sources: list[_DynamicSource] = []

        self.mix_left:  np.ndarray | None = None
        self.mix_right: np.ndarray | None = None
        self.rendered:  np.ndarray | None = None

    # ══════════════════════════════════════════════════════════════════════
    # Gestion des sources
    # ══════════════════════════════════════════════════════════════════════

    def add_source(
        self,
        signal: Union[np.ndarray, str, Path],
        trajectory: Trajectory,
        gain: float = 1.0,
        label: str = "",
    ) -> None:
        """
        Ajoute une source dynamique au paysage sonore.

        Paramètres
        ----------
        signal : np.ndarray | str | Path
            Signal audio mono.
            • np.ndarray : array déjà chargé à hrtf.sample_rate.
            • str / Path : chemin vers un fichier WAV (rééchantillonnage auto).
        trajectory : Trajectory
            Trajectoire de la source (CircularTrajectory, LinearTrajectory, …).
        gain : float
            Gain linéaire relatif (défaut : 1.0).
        label : str
            Nom de la source pour les logs (optionnel).
        """
        sr_target = self.hrtf.sample_rate

        if isinstance(signal, (str, Path)):
            path = Path(signal)
            if not path.exists():
                raise FileNotFoundError(f"Fichier introuvable : {path}")

            audio, sr_file = sf.read(str(path), dtype="float32")

            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            if sr_file != sr_target:
                print(
                    f"[DynamicSoundscape] Rééchantillonnage '{path.name}' : "
                    f"{sr_file} Hz → {sr_target} Hz…"
                )
                try:
                    import librosa
                    audio = librosa.resample(audio, orig_sr=sr_file, target_sr=sr_target)
                except ImportError:
                    raise ImportError("librosa est requis pour le rééchantillonnage.")

            lbl = label or path.name
            print(
                f"[DynamicSoundscape] Source ajoutée : '{lbl}'  "
                f"{len(audio)} smp ({len(audio)/sr_target:.2f}s)  "
                f"gain={gain}  traj={trajectory!r}"
            )
            self._sources.append(
                _DynamicSource(signal=audio, sr=sr_target, trajectory=trajectory,
                               gain=gain, label=lbl)
            )

        else:
            audio = np.asarray(signal, dtype=np.float32)
            if audio.ndim == 2:
                audio = audio.mean(axis=1)

            lbl = label or f"source_{len(self._sources) + 1}"
            print(
                f"[DynamicSoundscape] Source ajoutée : '{lbl}'  "
                f"{len(audio)} smp ({len(audio)/sr_target:.2f}s)  "
                f"gain={gain}  traj={trajectory!r}"
            )
            self._sources.append(
                _DynamicSource(signal=audio, sr=sr_target, trajectory=trajectory,
                               gain=gain, label=lbl)
            )

    # ══════════════════════════════════════════════════════════════════════
    # Rendu
    # ══════════════════════════════════════════════════════════════════════

    def render(self) -> np.ndarray:
        """
        Convolue chaque source, aligne les longueurs, somme et normalise.

        Retourne
        --------
        np.ndarray, shape (N, 2)
            Mix binaural stéréo normalisé (float32).

        Lève
        ----
        RuntimeError  Si aucune source n'a été ajoutée.
        """
        if not self._sources:
            raise RuntimeError("Aucune source ajoutée. Appelez add_source() d'abord.")

        outputs: list[np.ndarray] = []

        for i, src in enumerate(self._sources):
            print(
                f"\n[DynamicSoundscape] ── Source {i + 1}/{len(self._sources)} "
                f"'{src.label}' ──"
            )
            from engine import DynamicConvolver
            convolver = DynamicConvolver(
                hrtf           = self.hrtf,
                signal         = src.signal,
                sr             = src.sr,
                trajectory     = src.trajectory,
                segment_ms     = self.segment_ms,
                overlap_ms     = self.overlap_ms,
                crossfade_type = self.crossfade_type,
                listener       = self.listener,
            )
            out = convolver.run()           # (M, 2)
            out = (out * src.gain).astype(np.float32)
            outputs.append(out)

        self.mix_left, self.mix_right = self._mix(outputs)
        self.rendered = np.stack([self.mix_left, self.mix_right], axis=1)

        duration = self.rendered.shape[0] / self.hrtf.sample_rate
        print(
            f"\n[DynamicSoundscape] Rendu terminé : "
            f"{self.rendered.shape[0]} smp ({duration:.2f}s)  "
            f"shape={self.rendered.shape}"
        )
        return self.rendered

    def _mix(
        self,
        outputs: list[np.ndarray],
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Zero-pad toutes les pistes à la longueur max, somme, normalise.

        Paramètres
        ----------
        outputs : list[np.ndarray]
            Liste de tableaux (N_i, 2).

        Retourne
        --------
        (mix_left, mix_right) : tuple de np.ndarray, shape (N_max,)
        """
        max_len = max(o.shape[0] for o in outputs)
        mix_l   = np.zeros(max_len, dtype=np.float64)
        mix_r   = np.zeros(max_len, dtype=np.float64)

        for out in outputs:
            n = out.shape[0]
            mix_l[:n] += out[:, 0].astype(np.float64)
            mix_r[:n] += out[:, 1].astype(np.float64)

        peak = max(np.max(np.abs(mix_l)), np.max(np.abs(mix_r))) + 1e-10
        return (mix_l / peak).astype(np.float32), (mix_r / peak).astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════
    # Export
    # ══════════════════════════════════════════════════════════════════════

    def save(self, path: str | Path = "dynamic_soundscape.wav") -> None:
        """
        Sauvegarde le mix binaural en WAV stéréo.

        Lève
        ----
        RuntimeError  Si render() n'a pas été appelé.
        """
        if self.rendered is None:
            raise RuntimeError("Appelez render() avant save().")

        sf.write(str(path), self.rendered, self.hrtf.sample_rate)
        print(f"[DynamicSoundscape] Sauvegardé : {path}")

    # ══════════════════════════════════════════════════════════════════════
    # Représentation
    # ══════════════════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        return (
            f"DynamicSoundscape("
            f"{len(self._sources)} source(s), "
            f"segment={self.segment_ms:.0f} ms, "
            f"overlap={self.overlap_ms:.1f} ms, "
            f"crossfade='{self.crossfade_type}', "
            f"auditeur={'mobile' if self.listener is not None else 'fixe'}, "
            f"rendu={'oui' if self.rendered is not None else 'non'})"
        )
