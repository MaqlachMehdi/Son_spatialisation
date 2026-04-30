"""
convolution.py
--------------
Convolution d'un signal mono avec les HRIRs d'un objet HRTF.

Classes
-------
HRTFConvolver
    Applique la convolution fréquentielle d'un signal mono avec une paire
    de HRIRs (gauche / droite) sélectionnée par azimut et élévation.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

from hrtf import HRTF


class HRTFConvolver:
    """
    Convolve un signal mono avec la paire de HRIRs d'un objet HRTF.

    Paramètres
    ----------
    hrtf : HRTF
        Dataset HRTF chargé.
    azimuth : float
        Azimut de la source virtuelle en degrés (défaut : 0°).
    elevation : float
        Élévation de la source virtuelle en degrés (défaut : 0°).

    Attributs publics (disponibles après run())
    -------------------------------------------
    signal       : np.ndarray  — signal mono normalisé (float32)
    hrir_left    : np.ndarray  — HRIR oreille gauche sélectionnée
    hrir_right   : np.ndarray  — HRIR oreille droite sélectionnée
    output_left  : np.ndarray  — signal convolué oreille gauche
    output_right : np.ndarray  — signal convolué oreille droite
    sample_rate  : int         — fréquence d'échantillonnage commune
    """

    def __init__(
        self,
        hrtf: HRTF,
        azimuth: float = 0.0,
        elevation: float = 0.0,
    ) -> None:
        self.hrtf      = hrtf
        self.azimuth   = azimuth
        self.elevation = elevation
        self.sample_rate = hrtf.sample_rate

        # Sélection de la paire HRIR la plus proche
        idx = hrtf.get_index(azimuth, elevation)
        self.hrir_left, self.hrir_right = hrtf.get_hrir(azimuth, elevation)
        az_real = hrtf.positions[idx, 0]
        el_real = hrtf.positions[idx, 1]
        print(
            f"[HRTFConvolver] Position sélectionnée : "
            f"Az={az_real}°  El={el_real}°  (indice {idx})"
        )

        # Initialisés après load_signal() et run()
        self.signal:       np.ndarray | None = None
        self.output_left:  np.ndarray | None = None
        self.output_right: np.ndarray | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Chargement du signal
    # ──────────────────────────────────────────────────────────────────────

    def load_signal(self, path: str | Path) -> None:
        """
        Charge un fichier WAV, le convertit en mono float32 et
        rééchantillonne si nécessaire pour correspondre au sr des HRIRs.

        Paramètres
        ----------
        path : str ou Path
            Chemin vers le fichier .wav source.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        audio, sr = sf.read(str(path), dtype="float32")

        # Stéréo → mono
        if audio.ndim == 2:
            print("[HRTFConvolver] Signal stéréo détecté → mixage en mono.")
            audio = audio.mean(axis=1)

        # Rééchantillonnage si nécessaire
        if sr != self.sample_rate:
            print(
                f"[HRTFConvolver] Rééchantillonnage {sr} Hz → {self.sample_rate} Hz."
            )
            try:
                import librosa
                audio = librosa.resample(
                    audio,
                    orig_sr=sr,
                    target_sr=self.sample_rate,
                )
            except ImportError:
                raise ImportError(
                    "librosa est requis pour le rééchantillonnage. "
                    "Installez-le avec : pip install librosa"
                )

        self.signal = self._normalize(audio)
        print(
            f"[HRTFConvolver] Signal chargé : {len(self.signal)} échantillons "
            f"({len(self.signal)/self.sample_rate:.2f}s) @ {self.sample_rate} Hz"
        )

    def load_array(self, audio: np.ndarray, sample_rate: int) -> None:
        """
        Charge un signal directement depuis un array numpy.

        Paramètres
        ----------
        audio : np.ndarray
            Signal mono ou stéréo.
        sample_rate : int
            Fréquence d'échantillonnage du signal.
        """
        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        if sample_rate != self.sample_rate:
            raise ValueError(
                f"Le sample rate du signal ({sample_rate} Hz) ne correspond pas "
                f"à celui des HRIRs ({self.sample_rate} Hz). "
                f"Utilisez load_signal() pour le rééchantillonnage automatique."
            )

        self.signal = self._normalize(audio.astype(np.float32))

    # ──────────────────────────────────────────────────────────────────────
    # Convolution
    # ──────────────────────────────────────────────────────────────────────

    def _convolution_freq(self, signal: np.ndarray, hrir: np.ndarray) -> np.ndarray:
        """
        Convolve un signal mono avec une HRIR dans le domaine fréquentiel.

        Utilise fftconvolve (scipy) — équivalent à une multiplication
        spectrale suivie d'une IFFT, avec gestion correcte des bords.

        Paramètres
        ----------
        signal : np.ndarray, shape (N,)
            Signal mono normalisé en float32.
        hrir : np.ndarray, shape (L,)
            Réponse impulsionnelle (HRIR) sélectionnée.

        Retourne
        --------
        np.ndarray, shape (N + L - 1,)
            Signal convolué en float32.
        """
        convolved = fftconvolve(signal, hrir, mode="full")
        return convolved.astype(np.float32)

    def _conv_freq_R_and_L(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Applique la convolution fréquentielle sur les deux canaux
        (oreille gauche et oreille droite) et normalise le résultat.

        Retourne
        --------
        (output_left, output_right) : tuple de np.ndarray shape (N + L - 1,)
            Signaux convolutés normalisés, prêts à être exportés en WAV.

        Lève
        ----
        RuntimeError
            Si load_signal() ou load_array() n'a pas été appelé avant.
        """
        if self.signal is None:
            raise RuntimeError(
                "Aucun signal chargé. Appelez load_signal() ou load_array() d'abord."
            )

        print("[HRTFConvolver] Convolution canal gauche...")
        left  = self._convolution_freq(self.signal, self.hrir_left)

        print("[HRTFConvolver] Convolution canal droit...")
        right = self._convolution_freq(self.signal, self.hrir_right)

        # Normalisation globale (même facteur pour les deux canaux
        # afin de préserver l'équilibre binaural)
        peak = max(np.max(np.abs(left)), np.max(np.abs(right))) + 1e-10
        left  = (left  / peak).astype(np.float32)
        right = (right / peak).astype(np.float32)

        print(
            f"[HRTFConvolver] Sortie : {len(left)} échantillons "
            f"({len(left)/self.sample_rate:.2f}s)"
        )
        return left, right

    # ──────────────────────────────────────────────────────────────────────
    # Exécution et export
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> None:
        """Lance la convolution et stocke les résultats dans output_left / output_right."""
        self.output_left, self.output_right = self._conv_freq_R_and_L()

    def save(
        self,
        path_left:  str | Path = "output_left.wav",
        path_right: str | Path = "output_right.wav",
        path_stereo: str | Path | None = "output_stereo.wav",
    ) -> None:
        """
        Sauvegarde les signaux convolutés en fichiers WAV.

        Paramètres
        ----------
        path_left : str ou Path
            Chemin du fichier WAV oreille gauche (mono).
        path_right : str ou Path
            Chemin du fichier WAV oreille droite (mono).
        path_stereo : str ou Path ou None
            Chemin du fichier WAV stéréo (L+R entrelacés). None pour ignorer.
        """
        if self.output_left is None or self.output_right is None:
            raise RuntimeError("Appelez run() avant save().")

        sf.write(str(path_left),  self.output_left,  self.sample_rate)
        sf.write(str(path_right), self.output_right, self.sample_rate)
        print(f"[HRTFConvolver] Sauvegardé : {path_left}")
        print(f"[HRTFConvolver] Sauvegardé : {path_right}")

        if path_stereo is not None:
            stereo = np.stack([self.output_left, self.output_right], axis=1)
            sf.write(str(path_stereo), stereo, self.sample_rate)
            print(f"[HRTFConvolver] Sauvegardé : {path_stereo}")

    # ──────────────────────────────────────────────────────────────────────
    # Utilitaires internes
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(signal: np.ndarray) -> np.ndarray:
        """Normalise un signal dans [-1, 1]."""
        peak = np.max(np.abs(signal)) + 1e-10
        return (signal / peak).astype(np.float32)

    def __repr__(self) -> str:
        loaded = self.signal is not None
        done   = self.output_left is not None
        return (
            f"HRTFConvolver("
            f"az={self.azimuth}°, el={self.elevation}°, "
            f"sr={self.sample_rate} Hz, "
            f"signal={'chargé' if loaded else 'non chargé'}, "
            f"convolution={'faite' if done else 'non faite'})"
        )