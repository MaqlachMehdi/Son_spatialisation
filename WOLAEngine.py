"""
WOLAEngine.py
--------------
Moteur de convolution dynamique par Weighted Overlap-Add (WOLA).

Principe :
  1. Découper le signal en trames de `frame_samples` avec un pas de `hop_samples`
     (50 % de recouvrement → propriété COLA de la fenêtre de Hann).
  2. Appliquer la fenêtre de Hann à chaque trame AVANT convolution (input windowing).
  3. Convoluer chaque trame avec la paire HRIR correspondante.
  4. Overlap-add avec le même pas hop_samples.

Propriété fondamentale (COLA — Constant Overlap-Add) :
  Si toutes les HRIRs sont identiques, le signal est parfaitement reconstruit :
    w[n] + w[n + hop_samples] = 1.0   ∀ n   (fenêtre de Hann périodique, sym=False)

  Avec des HRIRs changeantes, la transition entre deux trames est implicitement
  équivalente à une interpolation linéaire des deux HRIRs dans la zone de recouvrement :
    sortie[k] ≈ x[k] · (w_out · HRIR_i + w_in · HRIR_{i+1})
  → aucun filtre en peigne, aucune coupure abrupte de queue HRIR.

Comparaison avec SegmentEngine :
  • Un seul paramètre de timing (hop_ms) au lieu de deux (segment_ms + overlap_ms).
  • frame_samples = 2 × hop_samples (50 % overlap fixe — pas de choix à faire).
  • Pas de crossfade_type : la fenêtre de Hann est optimale pour la COLA.
  • Reconstruction parfaite garantie quand la HRIR est stationnaire.
  • Contrainte : hop_ms ≥ hrir_len / sr × 500 ms.

Classe
------
WOLAEngine
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import fftconvolve
from scipy.signal.windows import hann as hann_window


class WOLAEngine:
    """
    Moteur de convolution par blocs avec Weighted Overlap-Add (WOLA).

    Paramètres
    ----------
    sr : int
        Fréquence d'échantillonnage en Hz.
    hop_ms : float
        Pas (stride) entre deux trames consécutives en millisecondes.
        Contrôle la résolution spatiale : plus petit = transitions plus fines.
        La trame a toujours une longueur de 2 × hop_ms (50 % de recouvrement).
    hrir_len : int
        Longueur de la HRIR en échantillons.
        Contrainte : frame_samples (= 2 × hop_samples) ≥ hrir_len.

    Attributs publics
    -----------------
    output : np.ndarray | None
        Dernier rendu binaural (N, 2). None avant le premier appel à render().
    """

    def __init__(self, sr: int, hop_ms: float, hrir_len: int) -> None:
        self.sr       = int(sr)
        self.hop_ms   = float(hop_ms)
        self.hrir_len = int(hrir_len)

        self.hop_samples   = int(sr * hop_ms / 1000)
        self.frame_samples = self.hop_samples * 2  # 50 % overlap → COLA Hann

        if self.frame_samples < hrir_len:
            min_hop_ms = hrir_len / sr * 500.0
            raise ValueError(
                f"frame_samples ({self.frame_samples} smp = {self.frame_samples/sr*1000:.1f} ms) "
                f"< hrir_len ({hrir_len} smp = {hrir_len/sr*1000:.1f} ms).\n"
                f"  → hop_ms minimum : {min_hop_ms:.1f} ms"
            )

        # Fenêtre de Hann périodique (sym=False) : COLA exacte
        #   w[n] + w[n + hop_samples] = 1.0  ∀ n  — voir plot_cola_property()
        self._window = hann_window(self.frame_samples, sym=False).astype(np.float32)

        self.output: np.ndarray | None = None

    # ══════════════════════════════════════════════════════════════════════
    # Découpage et fenêtrage
    # ══════════════════════════════════════════════════════════════════════

    def _frame_signal(self, signal: np.ndarray) -> list[np.ndarray]:
        """
        Découpe le signal en trames et applique la fenêtre de Hann (input windowing).

        Chaque trame a une longueur de frame_samples et est extraite avec un pas
        de hop_samples. Le signal est zero-paddé pour que le dernier hop soit complet.

        Paramètres
        ----------
        signal : np.ndarray, shape (N,)
            Signal mono float32.

        Retourne
        --------
        list[np.ndarray]
            Trames fenêtrées, chacune de longueur frame_samples, dtype float32.
        """
        n_hops  = max(1, int(np.ceil(len(signal) / self.hop_samples)))
        pad_len = (n_hops - 1) * self.hop_samples + self.frame_samples
        padded  = np.zeros(pad_len, dtype=np.float32)
        padded[: len(signal)] = signal

        frames = []
        for i in range(n_hops):
            start = i * self.hop_samples
            frame = padded[start : start + self.frame_samples].copy()
            frame *= self._window  # fenêtrage in-place
            frames.append(frame)

        return frames

    # ══════════════════════════════════════════════════════════════════════
    # Convolution d'une trame
    # ══════════════════════════════════════════════════════════════════════

    def _convolve_frame(
        self,
        frame:  np.ndarray,
        hrir_L: np.ndarray,
        hrir_R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Convolue une trame fenêtrée avec les HRIRs gauche et droite.

        Retourne
        --------
        (conv_L, conv_R) de longueur frame_samples + hrir_len − 1.
        """
        conv_L = fftconvolve(frame, hrir_L).astype(np.float32)
        conv_R = fftconvolve(frame, hrir_R).astype(np.float32)
        return conv_L, conv_R

    # ══════════════════════════════════════════════════════════════════════
    # Overlap-Add
    # ══════════════════════════════════════════════════════════════════════

    def _overlap_add(
        self,
        conv_frames:   list[tuple[np.ndarray, np.ndarray]],
        gains_per_hop: list[float] | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Assemble les trames convoluées par overlap-add.

        Chaque trame est placée à l'offset i × hop_samples. La propriété COLA
        garantit que les contributions se somment sans creux ni pic d'énergie
        dans les zones de recouvrement.

        Paramètres
        ----------
        conv_frames : list of (conv_L, conv_R)
        gains_per_hop : list[float] | None
            Gain scalaire par trame (modèle de distance 1/r). None = 1.0.

        Retourne
        --------
        (out_L, out_R) : np.ndarray float32 non normalisé.
        """
        n_hops       = len(conv_frames)
        frame_out    = len(conv_frames[0][0])           # frame_samples + hrir_len - 1
        out_len      = (n_hops - 1) * self.hop_samples + frame_out

        out_L = np.zeros(out_len, dtype=np.float32)
        out_R = np.zeros(out_len, dtype=np.float32)

        for i, (cL, cR) in enumerate(conv_frames):
            g   = gains_per_hop[i] if gains_per_hop is not None else 1.0
            pos = i * self.hop_samples
            end = min(pos + len(cL), out_len)
            n   = end - pos
            out_L[pos:end] += cL[:n] * g
            out_R[pos:end] += cR[:n] * g

        return out_L, out_R

    # ══════════════════════════════════════════════════════════════════════
    # Boucle principale
    # ══════════════════════════════════════════════════════════════════════

    def render(
        self,
        signal:        np.ndarray,
        hrirs_per_hop: list[tuple[np.ndarray, np.ndarray]],
        gains_per_hop: list[float] | None = None,
    ) -> np.ndarray:
        """
        Pipeline WOLA complet : fenêtrage → convolution → overlap-add.

        Paramètres
        ----------
        signal : np.ndarray, shape (N,)
            Signal audio mono float32.
        hrirs_per_hop : list[tuple[hrir_L, hrir_R]]
            Une paire HRIR par hop. Générer avec :
              trajectory.sample_segments(duration_s, segment_ms=hop_ms)
        gains_per_hop : list[float] | None
            Gain par hop pour le modèle de distance. None = pas de gain.

        Retourne
        --------
        np.ndarray, shape (M, 2)
            Signal binaural stéréo normalisé float32.

        Lève
        ----
        RuntimeError  Si le signal est vide.
        ValueError    Si len(hrirs_per_hop) ≠ nombre de hops calculé.
        """
        if len(signal) == 0:
            raise RuntimeError("Le signal fourni est vide.")

        # — 1. Découpage et fenêtrage ─────────────────────────────────────
        frames  = self._frame_signal(signal)
        n_hops  = len(frames)

        if len(hrirs_per_hop) != n_hops:
            raise ValueError(
                f"Nombre de HRIRs ({len(hrirs_per_hop)}) ≠ nombre de hops ({n_hops}).\n"
                f"  → Utilisez trajectory.sample_segments("
                f"duration_s={len(signal)/self.sr:.3f}, "
                f"segment_ms={self.hop_ms}) pour générer la liste."
            )
        if gains_per_hop is not None and len(gains_per_hop) != n_hops:
            raise ValueError(
                f"Nombre de gains ({len(gains_per_hop)}) ≠ nombre de hops ({n_hops})."
            )

        print(
            f"[WOLAEngine] Démarrage : {n_hops} hops × {self.hop_ms:.0f} ms  |  "
            f"frame={self.frame_samples} smp ({self.frame_samples/self.sr*1000:.1f} ms)  |  "
            f"HRIR={self.hrir_len} smp  |  fenêtre=Hann (COLA)"
        )

        # — 2. Convolution de toutes les trames ───────────────────────────
        conv_frames: list[tuple[np.ndarray, np.ndarray]] = [
            self._convolve_frame(frame, hrir_L, hrir_R)
            for frame, (hrir_L, hrir_R) in zip(frames, hrirs_per_hop)
        ]

        # — 3. Overlap-add ────────────────────────────────────────────────
        out_L, out_R = self._overlap_add(conv_frames, gains_per_hop)

        # — 4. Normalisation et trim ──────────────────────────────────────
        peak   = max(float(np.max(np.abs(out_L))), float(np.max(np.abs(out_R)))) + 1e-10
        out_L  = (out_L / peak).astype(np.float32)
        out_R  = (out_R / peak).astype(np.float32)

        trim   = min(len(signal) + self.hrir_len - 1, len(out_L))
        output = np.stack([out_L[:trim], out_R[:trim]], axis=1)

        print(
            f"[WOLAEngine] Rendu terminé : {output.shape[0]} échantillons "
            f"({output.shape[0] / self.sr:.2f} s)"
        )

        self.output = output
        return output

    # ══════════════════════════════════════════════════════════════════════
    # Utilitaire statique
    # ══════════════════════════════════════════════════════════════════════

    @staticmethod
    def optimal_hop_ms(
        period_s: float,
        angular_resolution_deg: float = 5.0,
    ) -> float:
        """
        Durée de hop pour une résolution spatiale angulaire donnée.

        Paramètres
        ----------
        period_s : float
            Durée d'un tour complet en secondes.
        angular_resolution_deg : float
            Résolution souhaitée en degrés (défaut : 5°).

        Retourne
        --------
        float : hop_ms recommandé.

        Exemple
        -------
        >>> WOLAEngine.optimal_hop_ms(period_s=4.0, angular_resolution_deg=5)
        55.55...   # → utiliser hop_ms=50 ou 55
        """
        return (angular_resolution_deg / 360.0) * period_s * 1000.0

    # ══════════════════════════════════════════════════════════════════════
    # Visualisation
    # ══════════════════════════════════════════════════════════════════════

    def plot_cola_property(self) -> None:
        """
        Vérifie et visualise la propriété COLA de la fenêtre de Hann.

        Gauche : fenêtre w[n] et sa voisine décalée de hop_samples.
        Droite : somme w[n] + w[n + hop_samples] — doit valoir 1.0 partout.
        """
        w    = self._window.astype(np.float64)
        hop  = self.hop_samples
        N    = self.frame_samples

        t_ms     = np.arange(N)    / self.sr * 1000.0
        t_over   = np.arange(hop)  / self.sr * 1000.0
        t_shift  = np.arange(hop, hop + N) / self.sr * 1000.0

        cola_sum = w[:hop] + w[hop:]
        max_err  = float(np.max(np.abs(cola_sum - 1.0)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(
            f"Propriété COLA — fenêtre de Hann  "
            f"(frame = {N} smp = {N/self.sr*1000:.1f} ms,  "
            f"hop = {hop} smp = {self.hop_ms:.1f} ms)",
            fontsize=11, fontweight="bold",
        )

        # ── Fenêtres ──────────────────────────────────────────────────────
        ax1.plot(t_ms,    w, color="#2196F3", linewidth=2.0, label="w[n]  (trame i)")
        ax1.plot(t_shift, w, color="#FF9800", linewidth=2.0,
                 linestyle="--", label=f"w[n + {hop} smp]  (trame i+1)")
        ax1.axvspan(0, t_over[-1], alpha=0.08, color="#4CAF50",
                    label="Zone de recouvrement")
        ax1.set_xlabel("Temps (ms)")
        ax1.set_ylabel("Amplitude")
        ax1.set_title("Trames consécutives (50 % overlap)")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # ── Somme COLA ────────────────────────────────────────────────────
        ax2.plot(t_over, cola_sum, color="#4CAF50", linewidth=2.5,
                 label=f"w[n] + w[n+hop]  (erreur max = {max_err:.2e})")
        ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.6,
                    label="Cible = 1.0")
        ax2.set_ylim(0.85, 1.15)
        ax2.set_xlabel("Temps (ms)")
        ax2.set_ylabel("Somme")
        ax2.set_title("Vérification COLA : w[n] + w[n + hop] = 1.0")
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
            f"WOLAEngine("
            f"sr={self.sr} Hz, "
            f"hop={self.hop_ms:.0f} ms ({self.hop_samples} smp), "
            f"frame={self.frame_samples} smp ({self.frame_samples/self.sr*1000:.1f} ms), "
            f"hrir_len={self.hrir_len} smp, "
            f"rendu={'oui' if done else 'non'})"
        )
