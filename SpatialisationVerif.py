"""
SpatialisationVerif.py
----------------------
Outils de vérification de la spatialisation HRTF.

Deux niveaux de vérification :
  1. Niveau HRIR  — vérifie que le dataset est chargé et sélectionné correctement.
  2. Niveau pipeline — vérifie que la convolution produit bien les bons indices spatiaux.

Usage notebook :
    from SpatialisationVerif import SpatialisationVerif
    verif = SpatialisationVerif(hrtf)
    verif.plot_all()
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import fftconvolve

from hrtf import HRTF


class SpatialisationVerif:
    """
    Vérifie que la spatialisation HRTF fonctionne correctement.

    Paramètres
    ----------
    hrtf : HRTF
        Dataset chargé via HRTF.from_sofa().
    elevation : float
        Élévation du plan de balayage (défaut 0° = plan horizontal).
    """

    def __init__(self, hrtf: HRTF, elevation: float = 0.0) -> None:
        self.hrtf      = hrtf
        self.elevation = elevation
        self._azimuths, self._ild_db, self._itd_ms = self._sweep()

    # ──────────────────────────────────────────────────────────────────────
    # Calculs
    # ──────────────────────────────────────────────────────────────────────

    def _sweep(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Balaye tous les azimuts disponibles à l'élévation cible.
        Calcule ILD et ITD depuis les HRIR brutes (sans signal de test).
        """
        # Filtrer les positions proches de l'élévation cible (±5°)
        mask = np.abs(self.hrtf.positions[:, 1] - self.elevation) <= 5.0
        subset = self.hrtf.positions[mask]
        indices = np.where(mask)[0]

        # Trier par azimut croissant
        order    = np.argsort(subset[:, 0])
        azimuths = subset[order, 0]
        indices  = indices[order]

        ild_db = np.zeros(len(indices))
        itd_ms = np.zeros(len(indices))

        for k, idx in enumerate(indices):
            hrir_l = self.hrtf.hrir[idx, 0, :].astype(np.float64)
            hrir_r = self.hrtf.hrir[idx, 1, :].astype(np.float64)

            # ILD = différence de niveau en dB (énergie gauche vs droite)
            energy_l = np.sum(hrir_l ** 2) + 1e-30
            energy_r = np.sum(hrir_r ** 2) + 1e-30
            ild_db[k] = 10.0 * np.log10(energy_l / energy_r)

            # ITD via intercorrélation : lag > 0 → L arrive en premier → source à gauche
            xcorr  = np.correlate(hrir_l, hrir_r, mode="full")
            lag    = np.argmax(xcorr) - (len(hrir_r) - 1)
            itd_ms[k] = lag / self.hrtf.sample_rate * 1000.0

        return azimuths, ild_db, itd_ms

    # ──────────────────────────────────────────────────────────────────────
    # Plots individuels
    # ──────────────────────────────────────────────────────────────────────

    # ── helpers communs aux deux polaires ─────────────────────────────────

    @staticmethod
    def _polar_setup(ax: plt.Axes) -> None:
        """0° en haut, sens anti-horaire → 90° à gauche (convention SOFA CCW)."""
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(1)   # CCW : 90° apparaît à gauche ✓

    @staticmethod
    def _add_caption(ax: plt.Axes, text: str) -> None:
        """Texte descriptif centré sous un axes polaire."""
        ax.text(
            0.5, -0.10, text,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=7.5, style="italic", color="#555555",
        )

    def _highlight_on_polar(
        self,
        ax: plt.Axes,
        azimuth: float,
        values: np.ndarray,
        label: str,
        color: str = "#f5a623",
    ) -> None:
        """Marque un azimut cible sur un diagramme polaire."""
        # Trouver la valeur la plus proche dans les données mesurées
        idx   = int(np.argmin(np.abs(self._azimuths - azimuth % 360)))
        az_r  = np.deg2rad(self._azimuths[idx])
        value = np.abs(values[idx])

        # Ligne radiale pointillée jusqu'au bord du graphe
        r_max = ax.get_rmax() if ax.get_rmax() > 0 else value * 1.2
        ax.plot([az_r, az_r], [0, r_max], color=color,
                linewidth=1.4, linestyle="--", zorder=4)
        # Marqueur à la valeur ILD/ITD
        ax.plot(az_r, value, marker="*", markersize=12,
                color=color, zorder=5, label=f"Az={azimuth}°")
        ax.legend(loc="lower right", fontsize=7)

    # ──────────────────────────────────────────────────────────────────────

    def plot_ild_polar(
        self,
        ax: plt.Axes | None = None,
        highlight_az: float | None = None,
    ) -> None:
        """
        Diagramme polaire de l'ILD (Interaural Level Difference).

        Paramètres
        ----------
        highlight_az : float | None
            Azimut (degrés) à marquer d'une étoile sur le diagramme.
        """
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 7))
            fig.suptitle("ILD — Interaural Level Difference", fontweight="bold")

        order = np.argsort(self._azimuths)
        az_r  = np.deg2rad(self._azimuths[order])
        ild   = self._ild_db[order]

        # Fermer la courbe
        az_r = np.append(az_r, az_r[0])
        ild  = np.append(ild,  ild[0])

        ild_pos = np.where(ild >= 0, ild,  0)
        ild_neg = np.where(ild <  0, -ild, 0)

        ax.plot(az_r, np.abs(ild), color="#333333", linewidth=1.0, zorder=3)
        ax.fill(az_r, ild_pos, alpha=0.35, color="#1f77b4", label="Gauche dominant")
        ax.fill(az_r, ild_neg, alpha=0.35, color="#d62728", label="Droite dominant")

        self._polar_setup(ax)
        ax.set_title(f"ILD (dB) — élév. {self.elevation}°", fontsize=9, pad=14)
        ax.legend(loc="lower right", fontsize=7)
        self._add_caption(ax, "Distance au centre = différence de niveau entre les deux oreilles (dB)")

        if highlight_az is not None:
            self._highlight_on_polar(ax, highlight_az, self._ild_db, "ILD")

        if standalone:
            plt.tight_layout()
            plt.show()

    def plot_itd_polar(
        self,
        ax: plt.Axes | None = None,
        highlight_az: float | None = None,
    ) -> None:
        """
        Diagramme polaire de l'ITD (Interaural Time Difference) en ms.

        Paramètres
        ----------
        highlight_az : float | None
            Azimut (degrés) à marquer d'une étoile sur le diagramme.
        """
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(6, 7))
            fig.suptitle("ITD — Interaural Time Difference", fontweight="bold")

        order = np.argsort(self._azimuths)
        az_r  = np.deg2rad(self._azimuths[order])
        itd   = self._itd_ms[order]

        az_r = np.append(az_r, az_r[0])
        itd  = np.append(itd,  itd[0])

        itd_pos = np.where(itd >= 0,  itd, 0)
        itd_neg = np.where(itd <  0, -itd, 0)

        ax.plot(az_r, np.abs(itd), color="#333333", linewidth=1.0, zorder=3)
        ax.fill(az_r, itd_pos, alpha=0.35, color="#1f77b4", label="Gauche en avance")
        ax.fill(az_r, itd_neg, alpha=0.35, color="#d62728", label="Droite en avance")

        self._polar_setup(ax)
        ax.set_title(f"ITD (ms) — élév. {self.elevation}°", fontsize=9, pad=14)
        ax.legend(loc="lower right", fontsize=7)
        self._add_caption(ax, "Distance au centre = délai d'arrivée entre les deux oreilles (ms)")

        if highlight_az is not None:
            self._highlight_on_polar(ax, highlight_az, self._itd_ms, "ITD")

        if standalone:
            plt.tight_layout()
            plt.show()

    def plot_hrir_pair(self, azimuth: float, ax_l: plt.Axes | None = None,
                       ax_r: plt.Axes | None = None) -> None:
        """
        Affiche la paire HRIR (gauche / droite) pour un azimut donné.
        Utile pour inspecter visuellement la réponse impulsionnelle.
        """
        standalone = ax_l is None
        if standalone:
            fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
            fig.suptitle(
                f"HRIR — Az={azimuth}°  Él={self.elevation}°", fontweight="bold"
            )

        idx    = self.hrtf.get_index(azimuth, self.elevation)
        hrir_l = self.hrtf.hrir[idx, 0, :]
        hrir_r = self.hrtf.hrir[idx, 1, :]
        t_ms   = self.hrtf.time_axis_ms

        az_real = self.hrtf.positions[idx, 0]
        el_real = self.hrtf.positions[idx, 1]

        for ax, hrir, color, label in [
            (ax_l, hrir_l, "#1f77b4", f"Gauche — Az={az_real}° Él={el_real}°"),
            (ax_r, hrir_r, "#d62728", f"Droite — Az={az_real}° Él={el_real}°"),
        ]:
            ax.plot(t_ms, hrir, color=color, linewidth=0.8)
            ax.set_title(label, fontsize=9)
            ax.set_xlabel("Temps (ms)", fontsize=8)
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.grid(True, alpha=0.25)

        ax_l.set_ylabel("Amplitude", fontsize=8)

        if standalone:
            plt.tight_layout()
            plt.show()

    def plot_hrtf_spectra(self, azimuth: float, ax: plt.Axes | None = None) -> None:
        """
        Spectres de magnitude des HRIR gauche et droite pour un azimut.
        La différence de coloration fréquentielle est la signature de la spatialisation.
        """
        standalone = ax is None
        if standalone:
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.suptitle(
                f"Spectres HRTF — Az={azimuth}°  Él={self.elevation}°",
                fontweight="bold",
            )

        mag_l, mag_r = self.hrtf.get_hrtf(azimuth, self.elevation)
        freqs        = self.hrtf.freq_axis_hz
        db_l = 20 * np.log10(np.maximum(mag_l, 1e-10))
        db_r = 20 * np.log10(np.maximum(mag_r, 1e-10))

        ax.semilogx(freqs, db_l, color="#1f77b4", linewidth=1.0, label="Gauche")
        ax.semilogx(freqs, db_r, color="#d62728", linewidth=1.0, label="Droite", alpha=0.8)
        ax.fill_between(freqs, db_l, db_r, alpha=0.15, color="purple",
                         label="|G−D| (zone colorée)")
        ax.set_xlabel("Fréquence (Hz)", fontsize=8)
        ax.set_ylabel("Magnitude (dB)", fontsize=8)
        ax.set_xlim(200, self.hrtf.sample_rate / 2)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8)
        ax.set_title(f"Az={azimuth}°", fontsize=9)

        if standalone:
            plt.tight_layout()
            plt.show()

    def plot_pipeline_check(self, azimuth: float) -> None:
        """
        Vérifie la pipeline complète : convolve un Dirac avec les HRIR,
        mesure l'énergie L et R, et affiche le résultat attendu vs obtenu.

        Critères de validation :
          • Az = 0°   → ILD ≈ 0 dB
          • Az = 90°  → ILD > 0 dB  (gauche dominant)
          • Az = 270° → ILD < 0 dB  (droite dominant)
        """
        hrir_l, hrir_r = self.hrtf.get_hrir(azimuth, self.elevation)

        # Signal de test : Dirac
        dirac = np.zeros(512, dtype=np.float32)
        dirac[0] = 1.0

        out_l = fftconvolve(dirac, hrir_l, mode="full").astype(np.float32)
        out_r = fftconvolve(dirac, hrir_r, mode="full").astype(np.float32)

        e_l = np.sum(out_l ** 2)
        e_r = np.sum(out_r ** 2)
        ild = 10 * np.log10((e_l + 1e-30) / (e_r + 1e-30))

        # Diagnostic
        az_norm = azimuth % 360
        if 15 <= az_norm <= 165:
            expected = "ILD > 0 dB  (source à gauche)"
            ok = ild > 0
        elif 195 <= az_norm <= 345:
            expected = "ILD < 0 dB  (source à droite)"
            ok = ild < 0
        else:
            expected = "ILD ≈ 0 dB  (source devant/derrière)"
            ok = abs(ild) < 3

        status = "✓ OK" if ok else "✗ ERREUR"
        print(f"[Pipeline check] Az={azimuth}°  Él={self.elevation}°")
        print(f"  ILD mesuré  : {ild:+.2f} dB")
        print(f"  Attendu     : {expected}")
        print(f"  Résultat    : {status}")

        # Waveformes
        t = np.arange(len(out_l)) / self.hrtf.sample_rate * 1000
        fig, axes = plt.subplots(1, 2, figsize=(10, 3), sharey=True)
        fig.suptitle(
            f"Vérification pipeline — Az={azimuth}°  |  ILD={ild:+.2f} dB  {status}",
            fontweight="bold",
        )
        for ax, sig, color, label in [
            (axes[0], out_l, "#1f77b4", "Gauche"),
            (axes[1], out_r, "#d62728", "Droite"),
        ]:
            ax.plot(t, sig, color=color, linewidth=0.8)
            rms = np.sqrt(np.mean(sig ** 2))
            ax.set_title(f"{label}  (RMS={rms:.4f})", fontsize=9)
            ax.set_xlabel("Temps (ms)", fontsize=8)
            ax.axhline(0, color="gray", linewidth=0.5)
            ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("Amplitude", fontsize=8)
        plt.tight_layout()
        plt.show()

    # ──────────────────────────────────────────────────────────────────────
    # Dashboard complet
    # ──────────────────────────────────────────────────────────────────────

    def plot_all(
        self,
        test_azimuths: list[float] | None = None,
        highlight_az: float | None = None,
    ) -> None:
        """
        Dashboard de vérification complet.

        Paramètres
        ----------
        test_azimuths : list[float] | None
            Azimuts pour la vérification pipeline (défaut : [0, 90, 270]).
        highlight_az : float | None
            Azimut à marquer d'une étoile sur les diagrammes polaires.
            Exemple : highlight_az=75 pour voir où se situe une source à 75°.
        """
        if test_azimuths is None:
            test_azimuths = [0.0, 90.0, 270.0]

        # ── Figure 1 : polaires ILD + ITD ─────────────────────────────────
        fig1, (ax_ild, ax_itd) = plt.subplots(
            1, 2,
            subplot_kw={"projection": "polar"},
            figsize=(12, 6),
        )
        fig1.suptitle(
            f"Indices spatiaux — plan horizontal ({self.elevation}°)",
            fontsize=13, fontweight="bold",
        )
        self.plot_ild_polar(ax_ild, highlight_az=highlight_az)
        self.plot_itd_polar(ax_itd, highlight_az=highlight_az)
        plt.tight_layout()
        plt.show()

        # ── Figure 2 : spectres HRTF pour 3 positions ─────────────────────
        fig2, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
        fig2.suptitle("Spectres HRTF — coloration fréquentielle par position",
                      fontsize=12, fontweight="bold")
        for ax, az in zip(axes, [0.0, 90.0, 270.0]):
            self.plot_hrtf_spectra(az, ax)
        plt.tight_layout()
        plt.show()

        # ── Vérification pipeline ──────────────────────────────────────────
        print("\n── Vérification pipeline (convolution Dirac → ILD) ──────────")
        for az in test_azimuths:
            self.plot_pipeline_check(az)
