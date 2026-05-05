"""
DistanceModel.py
----------------
Modèle de propagation acoustique en champ libre pour étendre une HRTF
mesurée à r₀ vers une distance cible r > r₀.

Trois effets physiques :
  1. Atténuation d'amplitude  — loi 1/r
  2. Retard de propagation    — Δt = (r - r₀) / c
  3. Absorption atmosphérique — atténuation haute-fréquence (ISO 9613, optionnel)

Note sur la dispersion : le son dans l'air est NON-DISPERSIF (ω = c·k, c constant
pour toutes les fréquences en champ libre). Il n'y a donc pas d'étalement temporel
dû à des vitesses de phase différentes entre fréquences.

Usage :
    model = DistanceModel(ref_distance=2.06)
    left, right = model.apply(left, right, r=10.0, sr=44100, user_gain=1.0)
"""

from __future__ import annotations

import numpy as np
from scipy.signal import fftconvolve


class DistanceModel:
    """
    Modèle de propagation pour sources à distance r quelconque.

    Paramètres
    ----------
    ref_distance : float
        Distance de mesure HRTF en mètres (r₀). Défaut : 2.06 m (IRCAM LISTEN).
    speed_of_sound : float
        Vitesse du son en m/s. Défaut : 343.0 m/s (air, 20 °C).
    apply_absorption : bool
        Active l'absorption atmosphérique (ISO 9613).
        Négligeable pour r < 50 m ; pertinent pour des scènes à grande échelle.
    temperature_c : float
        Température en °C pour le calcul de l'absorption. Défaut : 20.0 °C.
    humidity_pct : float
        Humidité relative en % pour l'absorption. Défaut : 50.0 %.
    """

    def __init__(
        self,
        ref_distance:    float = 2.06,
        speed_of_sound:  float = 343.0,
        apply_absorption: bool = False,
        temperature_c:   float = 20.0,
        humidity_pct:    float = 50.0,
    ) -> None:
        if ref_distance <= 0:
            raise ValueError("ref_distance doit être > 0.")
        self.ref_distance     = ref_distance
        self.speed_of_sound   = speed_of_sound
        self.apply_absorption = apply_absorption
        self.temperature_c    = temperature_c
        self.humidity_pct     = humidity_pct

    # ──────────────────────────────────────────────────────────────────────
    # Effet 1 — Atténuation d'amplitude (loi 1/r)
    # ──────────────────────────────────────────────────────────────────────

    def compute_gain(self, r: float) -> float:
        """
        Gain d'atténuation par la distance (loi 1/r).

        Retourne r₀/r, normalisé à 1.0 à r = r₀.
        Pour r < r₀, le gain est > 1 (source plus proche que la mesure).
        """
        return self.ref_distance / max(r, 1e-6)

    # ──────────────────────────────────────────────────────────────────────
    # Effet 2 — Retard de propagation
    # ──────────────────────────────────────────────────────────────────────

    def compute_delay_samples(self, r: float, sr: int) -> int:
        """
        Retard supplémentaire en échantillons dû à la distance additionnelle.

        Δt = (r - r₀) / c     [secondes]
        ΔN = round(Δt × sr)   [échantillons]

        Retourne 0 si r ≤ r₀ (pas de retard supplémentaire).
        """
        delta_r = max(r - self.ref_distance, 0.0)
        return round(delta_r / self.speed_of_sound * sr)

    # ──────────────────────────────────────────────────────────────────────
    # Effet 3 — Absorption atmosphérique (ISO 9613-1 simplifié)
    # ──────────────────────────────────────────────────────────────────────

    def compute_absorption_filter(self, r: float, sr: int, n_taps: int = 255) -> np.ndarray:
        """
        Filtre FIR représentant l'absorption atmosphérique pour la distance Δr = r - r₀.

        Modèle ISO 9613-1 simplifié :
          α(f) ≈ α_classic(f) + α_relaxation_O2(f) + α_relaxation_N2(f)  [dB/m]

        Le filtre est calculé dans le domaine fréquentiel puis converti en FIR par
        IFFT windowed (fenêtre de Hann).

        Retourne un filtre FIR normalisé (énergie ≈ 1 si Δr = 0).
        """
        delta_r = max(r - self.ref_distance, 0.0)
        if delta_r < 0.1:
            # Pas d'absorption significative : filtre identité
            h = np.zeros(n_taps, dtype=np.float32)
            h[n_taps // 2] = 1.0
            return h

        # Axe fréquentiel pour le filtre (rfft)
        n_fft  = 2 ** int(np.ceil(np.log2(n_taps)) + 1)
        freqs  = np.fft.rfftfreq(n_fft, d=1.0 / sr)  # Hz

        # α(f) en dB/m — approximation ISO 9613 pour conditions standard
        T_K    = self.temperature_c + 273.15
        T_ref  = 293.15  # 20°C en K
        hr     = self.humidity_pct

        # Fréquences de relaxation moléculaire (O2 et N2)
        f_rO2 = 24.0 + 4.04e4 * hr * (0.02 + hr) / (0.391 + hr)
        f_rN2 = (T_K / T_ref) ** (-0.5) * (
            9.0 + 280.0 * hr * np.exp(-4.17 * ((T_K / T_ref) ** (-1.0/3) - 1.0))
        )

        f = np.maximum(freqs, 1.0)  # éviter log(0)
        alpha_dB = (
            8.686 * f**2
            * (
                1.84e-11 * (T_K / T_ref) ** 0.5
                + (T_K / T_ref) ** (-2.5)
                * (
                    0.01275 * np.exp(-2239.1 / T_K) / (f_rO2 + f**2 / f_rO2)
                    + 0.1068  * np.exp(-3352.0 / T_K) / (f_rN2 + f**2 / f_rN2)
                )
            )
        )  # dB/m

        # Réponse en amplitude du filtre
        H_mag = 10.0 ** (-alpha_dB * delta_r / 20.0)
        H_mag[0] = 1.0  # DC inchangé

        # IFFT → réponse impulsionnelle, windowing
        h_full = np.fft.irfft(H_mag, n=n_fft)
        h_full = np.roll(h_full, n_taps // 2)[:n_taps]
        h_full *= np.hanning(n_taps)
        h_full /= np.sum(h_full) + 1e-30  # normalisation énergie DC

        return h_full.astype(np.float32)

    # ──────────────────────────────────────────────────────────────────────
    # Application complète
    # ──────────────────────────────────────────────────────────────────────

    def apply(
        self,
        left:       np.ndarray,
        right:      np.ndarray,
        r:          float,
        sr:         int,
        user_gain:  float = 1.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Applique les trois effets de distance au signal stéréo (L, R).

        Ordre des opérations :
          1. Retard de propagation : prepend ΔN échantillons de silence
          2. Atténuation 1/r  × gain manuel de l'utilisateur
          3. Absorption atmosphérique (si apply_absorption=True et r > r₀)

        Paramètres
        ----------
        left, right : np.ndarray
            Canaux gauche et droite après convolution HRTF.
        r : float
            Distance cible en mètres.
        sr : int
            Taux d'échantillonnage en Hz.
        user_gain : float
            Gain supplémentaire défini par l'utilisateur (défaut : 1.0).

        Retourne
        --------
        (left, right) : tuple de np.ndarray float32
        """
        # ── 1. Retard de propagation ──────────────────────────────────────
        delay = self.compute_delay_samples(r, sr)
        if delay > 0:
            pad   = np.zeros(delay, dtype=np.float32)
            left  = np.concatenate([pad, left])
            right = np.concatenate([pad, right])
            print(f"[DistanceModel] Retard : {delay} éch. "
                  f"({delay / sr * 1000:.1f} ms, Δr={r - self.ref_distance:.2f} m)")

        # ── 2. Atténuation 1/r + gain utilisateur ────────────────────────
        g     = self.compute_gain(r) * user_gain
        left  = (left  * g).astype(np.float32)
        right = (right * g).astype(np.float32)
        print(f"[DistanceModel] Gain : {g:.4f}  "
              f"(r={r:.2f} m → {20 * np.log10(max(g, 1e-10)):.1f} dB)")

        # ── 3. Absorption atmosphérique (optionnel) ───────────────────────
        if self.apply_absorption and r > self.ref_distance + 1.0:
            h     = self.compute_absorption_filter(r, sr)
            left  = fftconvolve(left,  h, mode="full").astype(np.float32)
            right = fftconvolve(right, h, mode="full").astype(np.float32)
            print(f"[DistanceModel] Absorption atmosphérique appliquée (Δr={r - self.ref_distance:.1f} m)")

        return left, right

    # ──────────────────────────────────────────────────────────────────────
    # Représentation
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"DistanceModel(r₀={self.ref_distance} m, "
            f"c={self.speed_of_sound} m/s, "
            f"absorption={'oui' if self.apply_absorption else 'non'})"
        )
