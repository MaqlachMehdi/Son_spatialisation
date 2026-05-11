"""
visualisation.py
----------------
Visualisations classiques pour un dataset HRTF.
Chaque visualisation est encapsulée dans sa propre classe.

Usage rapide
------------
    from hrtf import HRTF
    from visualisation import HRTFVisualizer

    hrtf = HRTF.from_sofa("dataset/IRC_1002_C_44100.sofa")
    viz  = HRTFVisualizer(hrtf)
    viz.plot_all()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import rfft
from scipy.signal import correlate

from hrtf import HRTF


# ══════════════════════════════════════════════════════════════════════════════
# Classe de base
# ══════════════════════════════════════════════════════════════════════════════

class _BasePlot:
    """
    Classe de base partagée par toutes les visualisations.
    Fournit l'accès à l'objet HRTF et des utilitaires communs.
    """

    def __init__(self, hrtf: HRTF) -> None:
        self.hrtf = hrtf

    # --- utilitaires internes ---

    def _magnitude_db(self, signal: np.ndarray) -> np.ndarray:
        """Magnitude en dB d'un signal réel via rfft."""
        return 20 * np.log10(np.abs(rfft(signal)) + 1e-10)

    def _savefig(self, fig: plt.Figure, path: str | None) -> None:
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Figure sauvegardée : {path}")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Distribution des positions de mesure
# ══════════════════════════════════════════════════════════════════════════════

class PositionPlot(_BasePlot):
    """
    Affiche la grille de mesure Az/El (2D) et la distribution sphérique (3D).
    """

    def plot(self, save_path: str | None = None) -> plt.Figure:
        """
        Paramètres
        ----------
        save_path : str, optionnel
            Chemin de sauvegarde de la figure.
        """
        pos = self.hrtf.positions
        fig = plt.figure(figsize=(14, 5))
        fig.suptitle("Distribution des positions de mesure", fontsize=13, fontweight="bold")

        # --- 2D Az / El ---
        ax1 = fig.add_subplot(1, 2, 1)
        ax1.scatter(pos[:, 0], pos[:, 1], s=40, color="steelblue", edgecolors="white", linewidths=0.5)
        ax1.set_xlabel("Azimut (°)")
        ax1.set_ylabel("Élévation (°)")
        ax1.set_title("Grille Az / El")
        ax1.grid(True, linestyle="--", alpha=0.5)

        # --- 3D sphérique ---
        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        az = np.radians(pos[:, 0])
        el = np.radians(pos[:, 1])
        r  = pos[:, 2]
        x  = r * np.cos(el) * np.cos(az)
        y  = r * np.cos(el) * np.sin(az)
        z  = r * np.sin(el)
        ax2.scatter(x, y, z, s=25, color="steelblue", edgecolors="white", linewidths=0.3)
        ax2.set_title("Distribution sphérique 3D")
        ax2.set_xlabel("X (m)")
        ax2.set_ylabel("Y (m)")
        ax2.set_zlabel("Z (m)")

        plt.tight_layout()
        self._savefig(fig, save_path)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. HRIR dans le domaine temporel
# ══════════════════════════════════════════════════════════════════════════════

class HRIRTimePlot(_BasePlot):
    """
    Affiche les réponses impulsionnelles gauche/droite pour une position donnée.
    """

    def plot(
        self,
        azimuth: float = 0.0,
        elevation: float = 0.0,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Paramètres
        ----------
        azimuth : float
            Azimut cible en degrés (défaut : 0°).
        elevation : float
            Élévation cible en degrés (défaut : 0°).
        save_path : str, optionnel
            Chemin de sauvegarde.
        """
        idx      = self.hrtf.get_index(azimuth, elevation)
        az_real  = self.hrtf.positions[idx, 0]
        el_real  = self.hrtf.positions[idx, 1]
        t        = self.hrtf.time_axis_ms

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(t, self.hrtf.hrir[idx, 0, :], label="Gauche",  color="steelblue")
        ax.plot(t, self.hrtf.hrir[idx, 1, :], label="Droite", color="tomato", alpha=0.8)
        ax.set_xlabel("Temps (ms)")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"HRIR — Az={az_real}°  El={el_real}°")
        ax.legend()
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        self._savefig(fig, save_path)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. HRTF dans le domaine fréquentiel
# ══════════════════════════════════════════════════════════════════════════════

class HRTFFreqPlot(_BasePlot):
    """
    Affiche la magnitude fréquentielle (HRTF) gauche/droite pour une position.
    """

    def plot(
        self,
        azimuth: float = 0.0,
        elevation: float = 0.0,
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Paramètres
        ----------
        azimuth : float
            Azimut cible en degrés.
        elevation : float
            Élévation cible en degrés.
        save_path : str, optionnel
            Chemin de sauvegarde.
        """
        idx     = self.hrtf.get_index(azimuth, elevation)
        az_real = self.hrtf.positions[idx, 0]
        el_real = self.hrtf.positions[idx, 1]
        freqs   = self.hrtf.freq_axis_hz

        mag_l = self._magnitude_db(self.hrtf.hrir[idx, 0, :])
        mag_r = self._magnitude_db(self.hrtf.hrir[idx, 1, :])

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.semilogx(freqs, mag_l, label="Gauche",  color="steelblue")
        ax.semilogx(freqs, mag_r, label="Droite", color="tomato", alpha=0.8)
        ax.set_xlabel("Fréquence (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.set_title(f"HRTF — Az={az_real}°  El={el_real}°")
        ax.set_xlim([100, self.hrtf.sample_rate / 2])
        ax.legend()
        ax.grid(True, which="both", linestyle="--", alpha=0.4)

        plt.tight_layout()
        self._savefig(fig, save_path)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. ITD — Interaural Time Difference
# ══════════════════════════════════════════════════════════════════════════════

class ITDPlot(_BasePlot):
    """
    Calcule et affiche l'ITD en fonction de l'azimut sur le plan horizontal.
    """

    @staticmethod
    def _compute_itd(hrir_l: np.ndarray, hrir_r: np.ndarray, sr: int) -> float:
        """Calcule l'ITD (en µs) par intercorrélation."""
        corr = correlate(hrir_l, hrir_r, mode="full")
        lag  = np.argmax(corr) - (len(hrir_l) - 1)
        return lag / sr * 1e6

    def plot(self, save_path: str | None = None) -> plt.Figure:
        """
        Paramètres
        ----------
        save_path : str, optionnel
            Chemin de sauvegarde.
        """
        indices, pos_h = self.hrtf.horizontal_plane()
        sort_order     = np.argsort(pos_h[:, 0])
        indices_sorted = indices[sort_order]
        az_sorted      = pos_h[sort_order, 0]

        itds = [
            self._compute_itd(
                self.hrtf.hrir[i, 0, :],
                self.hrtf.hrir[i, 1, :],
                self.hrtf.sample_rate,
            )
            for i in indices_sorted
        ]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(az_sorted, itds, "o-", color="steelblue", markersize=4)
        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
        ax.set_xlabel("Azimut (°)")
        ax.set_ylabel("ITD (µs)")
        ax.set_title("ITD en fonction de l'azimut — plan horizontal (El=0°)")
        ax.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        self._savefig(fig, save_path)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# 5. Colormap Azimut × Fréquence (plan horizontal)
# ══════════════════════════════════════════════════════════════════════════════

class HRTFColormapPlot(_BasePlot):
    """
    Colormap de la magnitude HRTF sur tout le plan horizontal.
    Axes : azimut (x) × fréquence log (y), couleur = dB.
    """

    def plot(
        self,
        ear: str = "left",
        save_path: str | None = None,
    ) -> plt.Figure:
        """
        Paramètres
        ----------
        ear : str
            'left' ou 'right'.
        save_path : str, optionnel
            Chemin de sauvegarde.
        """
        ear_idx = 0 if ear.lower() == "left" else 1
        indices, pos_h = self.hrtf.horizontal_plane()
        sort_order     = np.argsort(pos_h[:, 0])
        indices_sorted = indices[sort_order]
        az_sorted      = pos_h[sort_order, 0]
        freqs          = self.hrtf.freq_axis_hz

        hrtf_mag = np.array([
            self._magnitude_db(self.hrtf.hrir[i, ear_idx, :])
            for i in indices_sorted
        ])  # (M_h, F)

        fig, ax = plt.subplots(figsize=(13, 5))
        pcm = ax.pcolormesh(
            az_sorted,
            freqs / 1000,
            hrtf_mag.T,
            shading="auto",
            cmap="inferno",
            vmin=-40,
            vmax=20,
        )
        cbar = fig.colorbar(pcm, ax=ax, label="Magnitude (dB)")
        ax.set_yscale("log")
        ax.set_ylim([0.1, self.hrtf.sample_rate / 2000])
        ax.set_xlabel("Azimut (°)")
        ax.set_ylabel("Fréquence (kHz)")
        ax.set_title(
            f"HRTF oreille {'gauche' if ear_idx == 0 else 'droite'} — plan horizontal"
        )

        plt.tight_layout()
        self._savefig(fig, save_path)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# Facade : toutes les visualisations réunies
# ══════════════════════════════════════════════════════════════════════════════

class HRTFVisualizer:
    """
    Point d'entrée unique pour toutes les visualisations HRTF.

    Exemple
    -------
        hrtf = HRTF.from_sofa("dataset/IRC_1002_C_44100.sofa")
        viz  = HRTFVisualizer(hrtf)

        viz.positions.plot()
        viz.hrir_time.plot(azimuth=90, elevation=0)
        viz.hrtf_freq.plot(azimuth=0, elevation=0)
        viz.itd.plot()
        viz.colormap.plot(ear="left")

        viz.plot_all()   # génère toutes les figures d'un coup
    """

    def __init__(self, hrtf: HRTF) -> None:
        self.hrtf      = hrtf
        self.positions = PositionPlot(hrtf)
        self.hrir_time = HRIRTimePlot(hrtf)
        self.hrtf_freq = HRTFFreqPlot(hrtf)
        self.itd       = ITDPlot(hrtf)
        self.colormap  = HRTFColormapPlot(hrtf)

    def plot_all(
        self,
        azimuth: float = 0.0,
        elevation: float = 0.0,
        save_dir: str | None = None,
    ) -> None:
        """
        Génère les 5 visualisations classiques.

        Paramètres
        ----------
        azimuth : float
            Azimut de référence pour les plots par position.
        elevation : float
            Élévation de référence.
        save_dir : str, optionnel
            Dossier de sauvegarde. Si None, les figures s'affichent seulement.
        """
        from pathlib import Path

        def _path(name):
            if save_dir:
                return str(Path(save_dir) / name)
            return None

        print(f"Dataset : {self.hrtf}")
        self.positions.plot(save_path=_path("1_positions.png"))
        self.hrir_time.plot(azimuth, elevation, save_path=_path("2_hrir_time.png"))
        self.hrtf_freq.plot(azimuth, elevation, save_path=_path("3_hrtf_freq.png"))
        self.itd.plot(save_path=_path("4_itd.png"))
        self.colormap.plot(ear="left",  save_path=_path("5_colormap_L.png"))
        self.colormap.plot(ear="right", save_path=_path("5_colormap_R.png"))
        plt.show()


# ══════════════════════════════════════════════════════════════════════════════
# Exécution directe
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "dataset/IRC_1002_C_44100.sofa"
    hrtf = HRTF.from_sofa(path)
    viz  = HRTFVisualizer(hrtf)
    viz.plot_all()
