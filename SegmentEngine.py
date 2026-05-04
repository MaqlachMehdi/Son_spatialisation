"""
SegmentEngine.py
----------------
Moteur de convolution par segments avec crossfade pour la spatialisation dynamique.

Principe (overlap-add avec look-ahead) :
  1. Découper le signal en blocs de segment_ms avec overlap_ms de look-ahead.
  2. Convoluer chaque bloc avec la paire HRIR qui lui est associée.
  3. Appliquer le crossfade entre blocs adjacents dans la zone de recouvrement.
  4. Assembler par overlap-add → sortie binaural stéréo.

Représentation de la zone de crossfade entre segment i et i+1 :

  Segment i output  : |████████████▓▓▓▓|          (▓ = zone fade-out)
  Segment i+1 output:          |▓▓▓▓████████████|  (▓ = zone fade-in)
  Résultat final    : |████████████████████████|   (transition lisse)

  sortie[k] = (1 - t[k]) · seg_i[k]  +  t[k] · seg_j[k]   avec t ∈ [0, 1]

Classe
------
SegmentEngine
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve


_CROSSFADE_TYPES = ("linear", "cosine", "equal_power")


class SegmentEngine:
    """
    Moteur de convolution par segments avec crossfade.

    Paramètres
    ----------
    sr : int
        Fréquence d'échantillonnage en Hz.
    segment_ms : float
        Durée d'un segment en millisecondes.
        Plus court = résolution spatiale plus fine, coût CPU plus élevé.
        Règle pratique : segment_ms ≈ period_s / 360 × angle_step × 1000.
    overlap_ms : float
        Durée de la zone de crossfade entre segments (millisecondes).
        Recommandé : 10 à 25 % de segment_ms. Doit être < segment_ms.
    crossfade_type : 'linear' | 'cosine' | 'equal_power'
        Forme de l'enveloppe de crossfade (défaut : 'cosine').

    Attributs publics (disponibles après render())
    ----------------------------------------------
    output : np.ndarray, shape (M, 2) — dernier rendu binaural stéréo
    """

    def __init__(
        self,
        sr: int,
        segment_ms: float,
        overlap_ms: float,
        crossfade_type: str = "cosine",
    ) -> None:
        if crossfade_type not in _CROSSFADE_TYPES:
            raise ValueError(
                f"crossfade_type inconnu : '{crossfade_type}'. "
                f"Valeurs valides : {_CROSSFADE_TYPES}."
            )
        if overlap_ms <= 0:
            raise ValueError(f"overlap_ms doit être > 0 (reçu : {overlap_ms}).")
        if overlap_ms >= segment_ms:
            raise ValueError(
                f"overlap_ms ({overlap_ms} ms) doit être < segment_ms ({segment_ms} ms)."
            )

        self.sr             = int(sr)
        self.segment_ms     = float(segment_ms)
        self.overlap_ms     = float(overlap_ms)
        self.crossfade_type = crossfade_type

        self.segment_samples = int(sr * segment_ms / 1000)
        self.overlap_samples = int(sr * overlap_ms / 1000)

        self.output: np.ndarray | None = None

    # ══════════════════════════════════════════════════════════════════════
    # Découpage
    # ══════════════════════════════════════════════════════════════════════

    def _split_segments(self, signal: np.ndarray) -> list[np.ndarray]:
        """
        Découpe le signal en segments avec look-ahead (overlap à la fin).

        Chaque segment a une longueur de :
          segment_samples + overlap_samples

        L'overlap final est du contenu "emprunté" au segment suivant.
        Il sert de zone de crossfade après convolution.
        Le dernier segment est zero-paddé si le signal est trop court.

        Paramètres
        ----------
        signal : np.ndarray, shape (N,)
            Signal mono en float32.

        Retourne
        --------
        list[np.ndarray]
            N_segments tableaux de longueur (segment_samples + overlap_samples).
        """
        n          = len(signal)
        n_segments = max(1, int(np.ceil(n / self.segment_samples)))
        block_len  = self.segment_samples + self.overlap_samples

        segments = []
        for i in range(n_segments):
            start = i * self.segment_samples
            end   = min(start + block_len, n)
            block = np.zeros(block_len, dtype=np.float32)
            block[: end - start] = signal[start:end]
            segments.append(block)

        return segments

    # ══════════════════════════════════════════════════════════════════════
    # Enveloppe de crossfade
    # ══════════════════════════════════════════════════════════════════════

    def _make_envelope(
        self, N: int, fade_type: str | None = None
    ) -> np.ndarray:
        """
        Retourne un tableau fade-in [0 → 1] de longueur N.

        Le fade-out correspondant est `1 - envelope` (sauf en equal_power,
        où le couple sin/cos est utilisé pour conserver l'énergie).

        Paramètres
        ----------
        N : int
            Nombre d'échantillons de l'enveloppe.
        fade_type : str | None
            Type d'enveloppe. Si None, utilise self.crossfade_type.

        Retourne
        --------
        np.ndarray, shape (N,), valeurs dans [0.0, 1.0]

        Types
        -----
        'linear'      : rampe linéaire t ∈ [0, 1].
        'cosine'      : S-curve (1 − cos(πt)) / 2  — recommandé.
        'equal_power' : sin(π/2 · t)  — conservation d'énergie au croisement.
        """
        if N == 0:
            return np.array([], dtype=np.float32)

        t  = np.linspace(0.0, 1.0, N)
        ft = fade_type or self.crossfade_type

        if ft == "linear":
            env = t
        elif ft == "cosine":
            env = (1.0 - np.cos(np.pi * t)) / 2.0
        elif ft == "equal_power":
            env = np.sin(np.pi / 2.0 * t)
        else:
            raise ValueError(f"fade_type inconnu : '{ft}'.")

        return env.astype(np.float32)

    # ══════════════════════════════════════════════════════════════════════
    # Convolution d'un segment
    # ══════════════════════════════════════════════════════════════════════

    def _convolve_segment(
        self,
        segment: np.ndarray,
        hrir_L: np.ndarray,
        hrir_R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convolue un segment mono avec les HRIRs gauche et droite.

        Paramètres
        ----------
        segment : np.ndarray, shape (M,)
            Segment audio mono (segment_samples + overlap_samples).
        hrir_L : np.ndarray, shape (H,)
            HRIR oreille gauche.
        hrir_R : np.ndarray, shape (H,)
            HRIR oreille droite.

        Retourne
        --------
        (conv_L, conv_R) : tuple de np.ndarray, shape (M + H − 1,)
        """
        conv_L = fftconvolve(segment, hrir_L).astype(np.float32)
        conv_R = fftconvolve(segment, hrir_R).astype(np.float32)
        return conv_L, conv_R

    # ══════════════════════════════════════════════════════════════════════
    # Crossfade entre deux sorties convoluées
    # ══════════════════════════════════════════════════════════════════════

    def _crossfade_segments(
        self,
        out_i: np.ndarray,
        out_j: np.ndarray,
        envelope: np.ndarray,
    ) -> None:
        """
        Applique le crossfade entre la queue de out_i et la tête de out_j.

        Formule dans la zone overlap :
          out_i[k] ← (1 − t[k]) · out_i[k]   (fade-out)
          out_j[k] ←       t[k] · out_j[k]   (fade-in)

        La somme lors de l'overlap-add donnera :
          (1 − t) · out_i  +  t · out_j   ✓

        Cette méthode modifie out_i et out_j EN PLACE.

        Paramètres
        ----------
        out_i : np.ndarray
            Sortie convoluée du segment i (mono). Modifiée en place.
        out_j : np.ndarray
            Sortie convoluée du segment j (mono). Modifiée en place.
        envelope : np.ndarray, shape (overlap_samples,)
            Enveloppe fade-in [0 → 1] retournée par _make_envelope().
        """
        N = len(envelope)
        if N == 0:
            return

        if self.crossfade_type == "equal_power":
            # equal-power: fade_in^2 + fade_out^2 = 1
            fade_out = np.sqrt(np.clip(1.0 - envelope ** 2, 0.0, 1.0))
        else:
            fade_out = 1.0 - envelope
        fade_in  = envelope

        # ── Fade-out sur out_i — après la frontière du segment ───────────
        #    Zone : [segment_samples, segment_samples + overlap_samples]
        i_start  = self.segment_samples
        i_end    = min(i_start + N, len(out_i))
        actual_n = i_end - i_start
        if actual_n > 0:
            out_i[i_start:i_end] *= fade_out[:actual_n]
            out_i[i_end:] = 0.0  # queue HRIR résiduelle au-delà du fade : coupée

        # ── Fade-in sur out_j — au début du prochain segment ─────────────
        #    Zone : [0, overlap_samples]
        j_end = min(N, len(out_j))
        if j_end > 0:
            out_j[:j_end] *= fade_in[:j_end]

    # ══════════════════════════════════════════════════════════════════════
    # Boucle principale
    # ══════════════════════════════════════════════════════════════════════

    def render(
        self,
        signal: np.ndarray,
        hrirs_per_segment: list[tuple[np.ndarray, np.ndarray]],
        gains_per_segment: list[float] | None = None,
    ) -> np.ndarray:
        """
        Boucle principale : découpe → convolue → crossfade → overlap-add.

        Paramètres
        ----------
        signal : np.ndarray, shape (N,)
            Signal audio mono (float32).
        hrirs_per_segment : list[tuple[hrir_L, hrir_R]]
            Une paire HRIR par segment.
            Généré typiquement via :
              positions = trajectory.sample_segments(duration_s, segment_ms)
              hrirs = [hrtf.get_hrir(az, el) for az, el in positions]
        gains_per_segment : list[float] | None
            Gain scalaire par segment (ex. loi 1/r pour RectilinearTrajectory).
            Si None, aucun gain additionnel n'est appliqué (équivalent à 1.0).

        Retourne
        --------
        np.ndarray, shape (M, 2)
            Signal binaural stéréo normalisé (float32).
            M ≈ len(signal) + longueur_HRIR − 1.

        Lève
        ----
        RuntimeError  Si le signal est vide.
        ValueError    Si le nombre de HRIRs ≠ nombre de segments.
        """
        if len(signal) == 0:
            raise RuntimeError("Le signal fourni est vide.")

        # — 1. Découpage ──────────────────────────────────────────────────
        segments   = self._split_segments(signal)
        n_segments = len(segments)

        if len(hrirs_per_segment) != n_segments:
            raise ValueError(
                f"Nombre de HRIRs ({len(hrirs_per_segment)}) ≠ "
                f"nombre de segments ({n_segments}).\n"
                f"  → Utilisez trajectory.sample_segments("
                f"len(signal)/sr={len(signal)/self.sr:.2f}, "
                f"segment_ms={self.segment_ms}) pour générer la liste."
            )

        hrir_len = len(hrirs_per_segment[0][0])

        print(
            f"[SegmentEngine] Démarrage : {n_segments} segments × "
            f"{self.segment_ms:.0f} ms  |  "
            f"overlap={self.overlap_ms:.0f} ms  |  "
            f"HRIR={hrir_len} smp  |  "
            f"crossfade='{self.crossfade_type}'"
        )

        # — 2. Convolution de tous les segments ───────────────────────────
        conv_outs: list[list[np.ndarray]] = []
        for segment, (hrir_L, hrir_R) in zip(segments, hrirs_per_segment):
            cL, cR = self._convolve_segment(segment, hrir_L, hrir_R)
            conv_outs.append([cL, cR])

        # — 3. Crossfade entre segments adjacents ─────────────────────────
        envelope = self._make_envelope(self.overlap_samples)
        for i in range(n_segments - 1):
            cL_i, cR_i = conv_outs[i]
            cL_j, cR_j = conv_outs[i + 1]
            self._crossfade_segments(cL_i, cL_j, envelope)
            self._crossfade_segments(cR_i, cR_j, envelope)

        # — 4. Overlap-add ────────────────────────────────────────────────
        out_len = len(signal) + hrir_len + self.overlap_samples
        out_L   = np.zeros(out_len, dtype=np.float32)
        out_R   = np.zeros(out_len, dtype=np.float32)

        for i, (cL, cR) in enumerate(conv_outs):
            g   = gains_per_segment[i] if gains_per_segment is not None else 1.0
            pos = i * self.segment_samples
            end = min(pos + len(cL), out_len)
            n   = end - pos
            out_L[pos:end] += cL[:n] * g
            out_R[pos:end] += cR[:n] * g

        # — 5. Normalisation et trim ──────────────────────────────────────
        peak = max(np.max(np.abs(out_L)), np.max(np.abs(out_R))) + 1e-10
        out_L = (out_L / peak).astype(np.float32)
        out_R = (out_R / peak).astype(np.float32)

        # On garde signal + queue HRIR naturelle (hrir_len - 1 échantillons)
        trim   = min(len(signal) + hrir_len - 1, out_len)
        output = np.stack([out_L[:trim], out_R[:trim]], axis=1)

        print(
            f"[SegmentEngine] Rendu terminé : {output.shape[0]} échantillons "
            f"({output.shape[0] / self.sr:.2f}s)"
        )

        self.output = output
        return output

    # ══════════════════════════════════════════════════════════════════════
    # Utilitaire statique
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def optimal_segment_ms(
        period_s: float,
        angular_resolution_deg: float = 5.0,
    ) -> float:
        """
        Calcule la durée de segment nécessaire pour une résolution spatiale donnée.

        Paramètres
        ----------
        period_s : float
            Durée d'un tour complet en secondes.
        angular_resolution_deg : float
            Résolution angulaire souhaitée en degrés (défaut : 5°).

        Retourne
        --------
        float : durée de segment en millisecondes.

        Exemple
        -------
        >>> SegmentEngine.optimal_segment_ms(period_s=2, angular_resolution_deg=5)
        27.77...   # → utiliser segment_ms=25 ou 30
        """
        return (angular_resolution_deg / 360.0) * period_s * 1000.0

    # ══════════════════════════════════════════════════════════════════════
    # Visualisation
    # ══════════════════════════════════════════════════════════════════════

    def plot_envelopes(self) -> None:
        """
        Visualise les 3 types d'enveloppe disponibles.

        Graphique gauche : courbes fade-in (0 → 1).
        Graphique droit  : conservation d'énergie √(fade_in² + fade_out²).
          • equal_power = 1.0 constant → énergie parfaitement conservée.
          • cosine ≈ 0.97 au centre   → légère baisse, très acceptable.
          • linear ≈ 0.71 au centre   → baisse notable (creux perceptible).
        """
        N = max(self.overlap_samples, 200)
        t = np.linspace(0.0, self.overlap_ms, N)

        styles = [
            ("linear",      "#2196F3", "Linéaire"),
            ("cosine",      "#FF9800", "Cosinus (recommandé)"),
            ("equal_power", "#4CAF50", "Puissance égale"),
        ]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(
            f"Enveloppes de crossfade — overlap = {self.overlap_ms:.0f} ms",
            fontsize=11, fontweight="bold",
        )

        # ── Fade-in ───────────────────────────────────────────────────────
        for ft, color, label in styles:
            env = self._make_envelope(N, fade_type=ft)
            if ft == "equal_power":
                fade_out = np.sqrt(np.clip(1.0 - env ** 2, 0.0, 1.0))
            else:
                fade_out = 1.0 - env

            ax1.plot(t, env, color=color, linewidth=2.0, label=f"{label} (fade-in)")
            ax1.plot(t, fade_out, color=color, linewidth=2.0, linestyle="--", alpha=0.5)

        ax1.set_xlabel("Temps dans l'overlap (ms)")
        ax1.set_ylabel("Gain")
        ax1.set_title("Enveloppes fade-in (plein) et fade-out (tirets)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Conservation d'énergie : √(fade_in² + fade_out²) ─────────────
        for ft, color, label in styles:
            env = self._make_envelope(N, fade_type=ft)
            fade_in = env
            if ft == "equal_power":
                fade_out = np.sqrt(np.clip(1.0 - fade_in ** 2, 0.0, 1.0))
            else:
                fade_out = 1.0 - fade_in

            power = np.sqrt(fade_in ** 2 + fade_out ** 2)
            ax2.plot(t, power, color=color, linewidth=2.0, label=label)

        ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Cible = 1.0")
        ax2.set_xlabel("Temps dans l'overlap (ms)")
        ax2.set_ylabel("√(fade_in² + fade_out²)")
        ax2.set_title("Conservation d'énergie au croisement")
        ax2.set_ylim(0.6, 1.1)
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ══════════════════════════════════════════════════════════════════════
    # Représentation
    # ══════════════════════════════════════════════════════════════════════

    def __repr__(self) -> str:
        done = self.output is not None
        return (
            f"SegmentEngine("
            f"sr={self.sr} Hz, "
            f"segment={self.segment_ms:.0f} ms ({self.segment_samples} smp), "
            f"overlap={self.overlap_ms:.0f} ms ({self.overlap_samples} smp), "
            f"crossfade='{self.crossfade_type}', "
            f"rendu={'oui' if done else 'non'})"
        )
