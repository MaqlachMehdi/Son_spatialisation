"""
SoundVisu.py
------------
Visualisation d'un Soundscape rendu : représentation temporelle,
spectrale (Fourier) et spectrogramme pour l'oreille gauche, droite
et le mix stéréo.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.fft import rfft, rfftfreq
from scipy.signal import spectrogram as scipy_spectrogram

from Soundscape import Soundscape


class SoundVisu:
    """
    Visualiseur pour un Soundscape rendu.

    Paramètres
    ----------
    scene : Soundscape
        Paysage sonore ayant déjà appelé render().

    Méthodes publiques
    ------------------
    plot_temporal()     Formes d'onde (domaine temporel).
    plot_fourier()      Spectres de magnitude FFT en dB.
    plot_spectrogram()  Spectrogrammes temps-fréquence en dB.
    plot_all()          Les trois analyses dans une unique figure 3 x 3.
    plot_scene_sphere() Sphère 3D avec positions d'émission des sources.
    """

    _COLOR_LEFT  = "#1f77b4"   # bleu
    _COLOR_RIGHT = "#d62728"   # rouge

    def __init__(self, scene: Soundscape) -> None:
        if scene.rendered is None:
            raise RuntimeError(
                "Le Soundscape n'a pas encore été rendu. "
                "Appelez scene.render() avant SoundVisu(scene)."
            )
        self.scene = scene
        self.sr    = scene.hrtf.sample_rate
        self.left  = scene.mix_left          # shape (N,)
        self.right = scene.mix_right         # shape (N,)
        self.n     = len(self.left)
        self.t     = np.arange(self.n) / self.sr   # axe temporel en secondes

    # ──────────────────────────────────────────────────────────────────────
    # Helpers privés
    # ──────────────────────────────────────────────────────────────────────

    def _fft_db(self, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """FFT réelle → (fréquences Hz, magnitude en dB)."""
        mag   = np.abs(rfft(signal)) * 2 / self.n
        freqs = rfftfreq(self.n, 1.0 / self.sr)
        return freqs, 20.0 * np.log10(np.maximum(mag, 1e-10))

    def _spectrogram(
        self, signal: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """scipy.spectrogram → (fréquences, temps, puissance en dB)."""
        nperseg  = min(1024, self.n // 4)
        noverlap = nperseg // 2
        f, t, Sxx = scipy_spectrogram(
            signal, fs=self.sr, nperseg=nperseg, noverlap=noverlap, window="hann"
        )
        return f, t, 10.0 * np.log10(np.maximum(Sxx, 1e-12))

    @staticmethod
    def _decimate_for_plot(t: np.ndarray, sig: np.ndarray, max_pts: int = 50_000):
        """Sous-échantillonne pour garder les tracés fluides."""
        step = max(1, len(sig) // max_pts)
        return t[::step], sig[::step]

    # ──────────────────────────────────────────────────────────────────────
    # Représentation temporelle
    # ──────────────────────────────────────────────────────────────────────

    def plot_temporal(
        self,
        ax_l: plt.Axes | None = None,
        ax_r: plt.Axes | None = None,
        ax_s: plt.Axes | None = None,
    ) -> None:
        """Formes d'onde gauche, droite et stéréo superposée."""
        standalone = ax_l is None
        if standalone:
            fig, (ax_l, ax_r, ax_s) = plt.subplots(
                1, 3, figsize=(15, 3), sharey=True
            )
            fig.suptitle("Représentation temporelle", fontsize=12, fontweight="bold")

        t_d, l_d = self._decimate_for_plot(self.t, self.left)
        t_d, r_d = self._decimate_for_plot(self.t, self.right)

        for ax, sig, color, title in [
            (ax_l, l_d, self._COLOR_LEFT,  "Oreille gauche"),
            (ax_r, r_d, self._COLOR_RIGHT, "Oreille droite"),
        ]:
            ax.plot(t_d, sig, color=color, linewidth=0.6)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Temps (s)", fontsize=8)
            ax.set_xlim(0, self.t[-1])
            ax.set_ylim(-1.05, 1.05)
            ax.grid(True, alpha=0.25, linewidth=0.5)

        ax_l.set_ylabel("Amplitude", fontsize=8)

        ax_s.plot(t_d, l_d, color=self._COLOR_LEFT,  linewidth=0.6, alpha=0.7, label="G")
        ax_s.plot(t_d, r_d, color=self._COLOR_RIGHT, linewidth=0.6, alpha=0.7, label="D")
        ax_s.set_title("Mix stéréo", fontsize=9)
        ax_s.set_xlabel("Temps (s)", fontsize=8)
        ax_s.set_xlim(0, self.t[-1])
        ax_s.grid(True, alpha=0.25, linewidth=0.5)
        ax_s.legend(loc="upper right", fontsize=7)

        if standalone:
            plt.tight_layout()
            plt.show()

    # ──────────────────────────────────────────────────────────────────────
    # Représentation de Fourier
    # ──────────────────────────────────────────────────────────────────────

    def plot_fourier(
        self,
        ax_l: plt.Axes | None = None,
        ax_r: plt.Axes | None = None,
        ax_s: plt.Axes | None = None,
    ) -> None:
        """Spectres de magnitude FFT (dB, échelle log en fréquence)."""
        standalone = ax_l is None
        if standalone:
            fig, (ax_l, ax_r, ax_s) = plt.subplots(
                1, 3, figsize=(15, 3), sharey=True
            )
            fig.suptitle(
                "Spectre de Fourier — magnitude", fontsize=12, fontweight="bold"
            )

        f_l, db_l = self._fft_db(self.left)
        f_r, db_r = self._fft_db(self.right)
        fmax = self.sr / 2

        for ax, db, color, title in [
            (ax_l, db_l, self._COLOR_LEFT,  "Oreille gauche"),
            (ax_r, db_r, self._COLOR_RIGHT, "Oreille droite"),
        ]:
            ax.semilogx(f_l, db, color=color, linewidth=0.8)
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Fréquence (Hz)", fontsize=8)
            ax.set_xlim(20, fmax)
            ax.grid(True, which="both", alpha=0.25, linewidth=0.5)

        ax_l.set_ylabel("Magnitude (dB)", fontsize=8)

        ax_s.semilogx(f_l, db_l, color=self._COLOR_LEFT,  linewidth=0.8, alpha=0.8, label="G")
        ax_s.semilogx(f_r, db_r, color=self._COLOR_RIGHT, linewidth=0.8, alpha=0.8, label="D")
        ax_s.set_title("Mix stéréo", fontsize=9)
        ax_s.set_xlabel("Fréquence (Hz)", fontsize=8)
        ax_s.set_xlim(20, fmax)
        ax_s.grid(True, which="both", alpha=0.25, linewidth=0.5)
        ax_s.legend(loc="upper right", fontsize=7)

        if standalone:
            plt.tight_layout()
            plt.show()

    # ──────────────────────────────────────────────────────────────────────
    # Spectrogramme
    # ──────────────────────────────────────────────────────────────────────

    def plot_spectrogram(
        self,
        ax_l: plt.Axes | None = None,
        ax_r: plt.Axes | None = None,
        ax_s: plt.Axes | None = None,
    ):
        """
        Spectrogrammes temps-fréquence (dB, colormap magma).

        La colonne stéréo affiche la différence inter-auriculaire L − D (dB),
        utile pour visualiser les indices spatiaux en temps-fréquence.
        Retourne l'objet pcolormesh pour permettre une colorbar externe.
        """
        standalone = ax_l is None
        if standalone:
            fig, (ax_l, ax_r, ax_s) = plt.subplots(1, 3, figsize=(15, 4))
            fig.suptitle("Spectrogramme", fontsize=12, fontweight="bold")

        f_l, t_l, S_l = self._spectrogram(self.left)
        f_r, t_r, S_r = self._spectrogram(self.right)

        vmin = min(S_l.min(), S_r.min())
        vmax = max(S_l.max(), S_r.max())

        kw = dict(shading="gouraud", cmap="magma")
        im_l = ax_l.pcolormesh(t_l, f_l, S_l, vmin=vmin, vmax=vmax, **kw)
        im_r = ax_r.pcolormesh(t_r, f_r, S_r, vmin=vmin, vmax=vmax, **kw)

        # Différence inter-auriculaire L − D (dB) — révèle les indices de spatialisation
        diff   = S_l - S_r
        abs_lim = np.percentile(np.abs(diff), 99)
        im_s = ax_s.pcolormesh(
            t_l, f_l, diff,
            cmap="RdBu_r", vmin=-abs_lim, vmax=abs_lim,
            shading="gouraud",
        )

        labels = [
            (ax_l, "Oreille gauche",          "Fréquence (Hz)", True),
            (ax_r, "Oreille droite",           None,             False),
            (ax_s, "Différence G − D (dB)",    None,             False),
        ]
        for ax, title, ylabel, show_ylabel in labels:
            ax.set_title(title, fontsize=9)
            ax.set_xlabel("Temps (s)", fontsize=8)
            ax.set_ylim(0, self.sr / 2)
            ax.tick_params(labelsize=7)
            if show_ylabel and ylabel:
                ax.set_ylabel(ylabel, fontsize=8)

        if standalone:
            plt.colorbar(im_l, ax=[ax_l, ax_r], label="Puissance (dB)")
            plt.colorbar(im_s, ax=ax_s, label="L − D (dB)")
            plt.tight_layout()
            plt.show()

        return im_l, im_s

    # ──────────────────────────────────────────────────────────────────────
    # Affichage global
    # ──────────────────────────────────────────────────────────────────────

    def plot_all(self) -> None:
        """
        Affiche les 3 analyses dans une unique figure 3 × 3 :
        lignes = Temporel / Fourier / Spectrogramme
        colonnes = Oreille gauche / Oreille droite / Mix stéréo
        """
        n_src = len(self.scene.sources)
        dur   = self.n / self.sr
        fig = plt.figure(figsize=(17, 11))
        fig.suptitle(
            f"Analyse Soundscape — {n_src} source(s)  ·  {dur:.2f} s  ·  {self.sr} Hz",
            fontsize=13,
            fontweight="bold",
            y=0.98,
        )

        gs = gridspec.GridSpec(
            3, 3,
            figure=fig,
            hspace=0.55,
            wspace=0.38,
            left=0.08, right=0.96,
            top=0.92, bottom=0.07,
        )

        # ── Ligne 0 : temporel ────────────────────────────────────────────
        ax_tl = fig.add_subplot(gs[0, 0])
        ax_tr = fig.add_subplot(gs[0, 1], sharey=ax_tl)
        ax_ts = fig.add_subplot(gs[0, 2], sharey=ax_tl)
        self.plot_temporal(ax_tl, ax_tr, ax_ts)
        _row_label(ax_tl, "Temporel")

        # ── Ligne 1 : Fourier ─────────────────────────────────────────────
        ax_fl = fig.add_subplot(gs[1, 0])
        ax_fr = fig.add_subplot(gs[1, 1], sharey=ax_fl)
        ax_fs = fig.add_subplot(gs[1, 2], sharey=ax_fl)
        self.plot_fourier(ax_fl, ax_fr, ax_fs)
        _row_label(ax_fl, "Fourier")

        # ── Ligne 2 : spectrogramme ───────────────────────────────────────
        ax_sl = fig.add_subplot(gs[2, 0])
        ax_sr = fig.add_subplot(gs[2, 1])
        ax_ss = fig.add_subplot(gs[2, 2])
        im_pwr, im_diff = self.plot_spectrogram(ax_sl, ax_sr, ax_ss)
        _row_label(ax_sl, "Spectrogramme")

        # Colorbars
        cbar_pwr = fig.colorbar(
            im_pwr, ax=[ax_sl, ax_sr],
            label="Puissance (dB)", shrink=0.85, pad=0.04,
        )
        cbar_pwr.ax.tick_params(labelsize=7)

        cbar_diff = fig.colorbar(
            im_diff, ax=ax_ss,
            label="G − D (dB)", shrink=0.85, pad=0.04,
        )
        cbar_diff.ax.tick_params(labelsize=7)

        plt.show()

    def plot_scene_sphere(self) -> None:
        """
        Affiche la sphère de mesure HRTF et les positions d'émission
        de toutes les sources de la scène.

        Repère d'affichage (perspective naturelle de l'auditeur) :
          • axe horizontal   → Gauche (-) / Droite (+)   [SOFA  -Y]
          • axe profondeur   → Derrière (-) / Devant (+)  [SOFA  +X]
          • axe vertical     → Bas (-) / Haut (+)         [SOFA  +Z]

        Les points bleus = positions du dataset HRTF.
        Les marqueurs rouges = sources actives de la scène.
        """
        if not self.scene.sources:
            raise RuntimeError(
                "La scène ne contient aucune source. "
                "Ajoutez des sources avant d'appeler plot_scene_sphere()."
            )

        def sofa_to_display(az_rad, el_rad, r):
            """
            Sphérique SOFA → Cartésien d'affichage.
            SOFA : X=devant, Y=gauche, Z=haut.
            Affichage : dx=droite, dy=devant, dz=haut.
              dx = -Y_sofa  (gauche SOFA → côté gauche de l'écran)
              dy =  X_sofa  (devant SOFA → profondeur de l'écran)
              dz =  Z_sofa
            """
            x_s = r * np.cos(el_rad) * np.cos(az_rad)   # devant SOFA
            y_s = r * np.cos(el_rad) * np.sin(az_rad)   # gauche SOFA
            z_s = r * np.sin(el_rad)                     # haut SOFA
            return -y_s, x_s, z_s                        # dx, dy, dz

        pos   = self.scene.hrtf.positions
        az_r  = np.radians(pos[:, 0])
        el_r  = np.radians(pos[:, 1])
        r_arr = pos[:, 2]
        dx, dy, dz = sofa_to_display(az_r, el_r, r_arr)

        fig = plt.figure(figsize=(9, 7))
        ax  = fig.add_subplot(111, projection="3d")
        ax.scatter(dx, dy, dz, s=16, color="#76a9d5",
                   edgecolors="white", linewidths=0.25, alpha=0.8,
                   label="Positions HRTF")

        src_pts = []
        for i, src in enumerate(self.scene.sources, start=1):
            az_s  = np.radians(src.azimuth)
            el_s  = np.radians(src.elevation)
            r_s   = float(src.distance)
            sdx, sdy, sdz = sofa_to_display(az_s, el_s, r_s)
            src_pts.append((sdx, sdy, sdz))

            ax.scatter([sdx], [sdy], [sdz], s=120, marker="o",
                       color="#e74c3c", edgecolors="black", linewidths=0.8,
                       zorder=5)
            ax.text(sdx, sdy, sdz, f"  S{i}\n  Az={src.azimuth}° Él={src.elevation}°",
                    fontsize=8, fontweight="bold", color="#222222")

        ax.scatter([], [], [], s=85, marker="o", color="#e74c3c",
                   edgecolors="black", linewidths=0.6, label="Sources de la scène")

        # Auditeur au centre
        ax.scatter([0], [0], [0], s=200, marker="^", color="#2ecc71",
                   edgecolors="black", linewidths=0.8, zorder=6, label="Auditeur")

        # Cadre isotrope
        all_dx = np.append(dx, [p[0] for p in src_pts])
        all_dy = np.append(dy, [p[1] for p in src_pts])
        all_dz = np.append(dz, [p[2] for p in src_pts])
        max_r  = 0.5 * max(np.ptp(all_dx), np.ptp(all_dy), np.ptp(all_dz), 0.1)
        cx, cy, cz = 0.0, 0.0, 0.0
        ax.set_xlim(cx - max_r, cx + max_r)
        ax.set_ylim(cy - max_r, cy + max_r)
        ax.set_zlim(cz - max_r, cz + max_r)

        ax.set_xlabel("← Gauche  |  Droite →", fontsize=8)
        ax.set_ylabel("← Derrière  |  Devant →", fontsize=8)
        ax.set_zlabel("↑ Haut", fontsize=8)
        ax.set_title("Sphère HRTF et positions des sources\n"
                     "(vue depuis la droite de l'auditeur, légèrement en hauteur)",
                     fontsize=11, fontweight="bold")

        # Angle de vue : légèrement au-dessus, depuis l'avant-droite
        ax.view_init(elev=20, azim=200)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper right", fontsize=8)

        plt.tight_layout()
        plt.show()


# ──────────────────────────────────────────────────────────────────────────
# Utilitaire interne
# ──────────────────────────────────────────────────────────────────────────

def _row_label(ax: plt.Axes, label: str) -> None:
    """Étiquette verticale à gauche d'une ligne de subplots."""
    ax.annotate(
        label,
        xy=(-0.28, 0.5),
        xycoords="axes fraction",
        ha="center",
        va="center",
        rotation=90,
        fontsize=9,
        fontweight="bold",
        color="#444444",
    )
