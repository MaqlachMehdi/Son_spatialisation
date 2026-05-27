from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from .sh_transform import SphericalHarmonicsTransform, _real_sph_harm


class SHVisualizer:
    """Visualisation de la décomposition en harmoniques sphériques.

    Usage
    -----
    sh  = SphericalHarmonicsTransform(order=5)
    viz = SHVisualizer(sh)

    viz.plot_basis(l=2, m=1)
    viz.plot_coefficients(coeffs)
    viz.plot_reconstruction(hrtf_orig, coeffs, positions_deg)
    viz.plot_reconstruction_error(hrtf_orig, coeffs, positions_deg)
    """

    def __init__(self, transform: SphericalHarmonicsTransform | None = None) -> None:
        self._sh = transform or SphericalHarmonicsTransform()

    # ------------------------------------------------------------------

    def plot_basis(self, l: int, m: int, resolution: int = 200) -> None:
        """Affiche Y_l^m sur la sphère (projection azimut / élévation).

        Parameters
        ----------
        l, m       : ordre et degré de l'harmonique
        resolution : nombre de points sur chaque axe
        """
        az = np.linspace(0.0, 360.0, resolution)
        el = np.linspace(-90.0, 90.0, resolution)
        AZ, EL = np.meshgrid(az, el)

        phi   = np.radians(AZ.ravel())
        theta = np.radians(90.0 - EL.ravel())
        vals  = _real_sph_harm(l, m, phi, theta).reshape(resolution, resolution)

        vmax = np.abs(vals).max()
        fig, ax = plt.subplots(figsize=(9, 4))
        im = ax.pcolormesh(az, el, vals,
                           cmap='RdBu_r', shading='auto',
                           vmin=-vmax, vmax=vmax)
        plt.colorbar(im, ax=ax, label='Amplitude')
        ax.set_xlabel('Azimut (°)')
        ax.set_ylabel('Élévation (°)')
        ax.set_title(rf'Harmonique sphérique $Y_{l}^{{{m}}}$  (ordre={l}, degré={m})')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------

    def plot_coefficients(self, coeffs: np.ndarray, sr: int = 44100) -> None:
        """Heatmap des coefficients SH [n_sh, N_freqs, 2].

        Chaque ligne = un mode spatial, chaque colonne = une fréquence.
        Permet de voir quels modes dominent et à quelles fréquences.
        """
        n_sh, N_freqs, _ = coeffs.shape
        freqs = np.fft.rfftfreq(2 * (N_freqs - 1), d=1.0 / sr) / 1000.0  # kHz

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for i, (ax, label) in enumerate(zip(axes, ['Gauche', 'Droite'])):
            data = coeffs[:, :, i]
            vmax = np.abs(data).max()
            im   = ax.imshow(
                data, aspect='auto', origin='lower',
                extent=[freqs[0], freqs[-1], -0.5, n_sh - 0.5],
                cmap='RdBu_r', vmin=-vmax, vmax=vmax,
            )
            plt.colorbar(im, ax=ax, label='dB')
            ax.set_xlabel('Fréquence (kHz)')
            ax.set_ylabel('Indice SH')
            ax.set_title(f'Coefficients SH — Oreille {label}')
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------

    def plot_reconstruction(
        self,
        hrtf_orig:    np.ndarray,
        coeffs:       np.ndarray,
        positions_deg: np.ndarray,
        position_idx: int = 0,
        sr:           int = 44100,
    ) -> None:
        """Superpose le spectre original et reconstruit pour une direction.

        Parameters
        ----------
        hrtf_orig    : [N_pos, N_freqs, 2]  spectre original
        coeffs       : [n_sh, N_freqs, 2]   coefficients SH
        positions_deg: [N_pos, 2]            positions en degrés
        position_idx : indice de la direction à afficher
        """
        hrtf_recon = self._sh.decode(coeffs, positions_deg)
        N_freqs    = hrtf_orig.shape[1]
        freqs      = np.fft.rfftfreq(2 * (N_freqs - 1), d=1.0 / sr)

        az, el = positions_deg[position_idx]
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))
        for i, (ax, label) in enumerate(zip(axes, ['Gauche', 'Droite'])):
            ax.plot(freqs, hrtf_orig[position_idx, :, i],
                    label='Original', linewidth=1.5)
            ax.plot(freqs, hrtf_recon[position_idx, :, i],
                    '--', label=f'Reconstruit (L={self._sh.order})', linewidth=1.5)
            ax.set_xscale('log')
            ax.set_xlabel('Fréquence (Hz)')
            ax.set_ylabel('Magnitude (dB)')
            ax.set_title(f'Az={az:.0f}°  El={el:.0f}° — Oreille {label}')
            ax.legend()
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------

    def plot_reconstruction_error(
        self,
        hrtf_orig:    np.ndarray,
        coeffs:       np.ndarray,
        positions_deg: np.ndarray,
    ) -> None:
        """Carte de Log-Spectral Distortion (LSD) sur toute la sphère.

        Chaque point = une position de mesure, coloré par l'erreur de
        reconstruction. Utile pour choisir l'ordre L optimal.
        """
        hrtf_recon = self._sh.decode(coeffs, positions_deg)
        # LSD par position : RMSE sur fréquences et oreilles
        lsd = np.sqrt(np.mean((hrtf_orig - hrtf_recon) ** 2, axis=(1, 2)))

        fig, ax = plt.subplots(figsize=(11, 5))
        sc = ax.scatter(
            positions_deg[:, 0], positions_deg[:, 1],
            c=lsd, cmap='hot_r', s=8, vmin=0,
        )
        plt.colorbar(sc, ax=ax, label='LSD (dB)')
        ax.set_xlabel('Azimut (°)')
        ax.set_ylabel('Élévation (°)')
        ax.set_title(
            f'Erreur de reconstruction LSD — ordre L={self._sh.order}'
            f'  (mean={lsd.mean():.2f} dB, max={lsd.max():.2f} dB)'
        )
        plt.tight_layout()
        plt.show()
