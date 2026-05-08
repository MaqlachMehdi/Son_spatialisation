"""
hrtf_utils.py
-------------
Fonctions utilitaires partagées entre HRTFInterpolator et HRTFGen.

Toutes les fonctions sont pures (pas d'état, pas de dépendance de classe) :
elles peuvent être appelées indépendamment ou importées par n'importe quel module.

Fonctions
---------
build_mp_window              — fenêtre cepstrale pour la phase minimum
detect_onset                 — retard d'onset τ = argmax |h(n)|
minimum_phase_from_magnitude — spectre complexe phase-min depuis un module
reconstruct_hrir             — pipeline complet : magnitude + onset → HRIR
"""

from __future__ import annotations

import numpy as np
from scipy.fft import rfft, irfft


def build_mp_window(n_samples: int) -> np.ndarray:
    """
    Construit la fenêtre cepstrale pour la reconstruction phase-minimum.

    Fenêtre w_mp (Oppenheim & Schafer) :
      w[0]         = 1
      w[1:N//2]    = 2
      w[N//2]      = 1   (si N pair)
      w[N//2+1:]   = 0

    Paramètres
    ----------
    n_samples : int
        Longueur N de la HRIR (nombre d'échantillons).

    Retourne
    --------
    np.ndarray, shape (N,), dtype float64
    """
    w = np.zeros(n_samples, dtype=np.float64)
    w[0] = 1.0
    w[1 : n_samples // 2] = 2.0
    if n_samples % 2 == 0:
        w[n_samples // 2] = 1.0
    return w


def detect_onset(hrir: np.ndarray) -> float:
    """
    Détecte le retard d'onset τ d'une HRIR.

    Définition : τ = argmax_n |h(n)|

    Paramètres
    ----------
    hrir : np.ndarray, shape (N,)
        Réponse impulsionnelle (mono, une oreille).

    Retourne
    --------
    float
        Indice du pic d'amplitude (en échantillons).
    """
    return float(np.argmax(np.abs(hrir)))


def minimum_phase_from_magnitude(
    mag:       np.ndarray,
    n_samples: int,
    mp_window: np.ndarray,
) -> np.ndarray:
    """
    Reconstruit le spectre complexe phase-minimum depuis un spectre de magnitude.

    Méthode du cepstre réel (Oppenheim & Schafer) :
      1. log_mag = log(mag + ε)
      2. c       = IRFFT(log_mag)            — cepstre réel, shape (N,)
      3. c_mp    = c × w_mp                  — annule la partie anti-causale
      4. H_mp    = exp(RFFT(c_mp))           — spectre complexe phase-minimum

    Propriété clé : |H_mp(f)| = mag(f) par construction.
    Tous les zéros de H_mp sont à l'intérieur du cercle unité.

    Paramètres
    ----------
    mag : np.ndarray, shape (N//2+1,), dtype float64
        Spectre de magnitude (valeurs ≥ 0).
    n_samples : int
        Longueur N de la HRIR cible.
    mp_window : np.ndarray, shape (N,)
        Fenêtre phase-minimum retournée par build_mp_window(N).

    Retourne
    --------
    np.ndarray complexe, shape (N//2+1,)
    """
    log_mag = np.log(mag + 1e-30)
    c       = irfft(log_mag, n=n_samples)
    c_mp    = c * mp_window
    return np.exp(rfft(c_mp, n=n_samples))


def reconstruct_hrir(
    mag:       np.ndarray,
    tau:       float,
    n_samples: int,
    mp_window: np.ndarray,
    rfft_freqs: np.ndarray,
) -> np.ndarray:
    """
    Reconstruit une HRIR depuis son spectre de magnitude et son onset.

    Pipeline :
      1. H_mp    = MinimumPhase(mag)
      2. H_final = H_mp · exp(−j2πf·τ)   — retard fractionnaire exact
      3. h       = Re{IRFFT(H_final)}

    Paramètres
    ----------
    mag : np.ndarray, shape (N//2+1,), dtype float64
        Spectre de magnitude interpolé ou moyenné.
    tau : float
        Retard d'onset en échantillons (peut être fractionnaire).
    n_samples : int
        Longueur N de la HRIR cible.
    mp_window : np.ndarray, shape (N,)
        Fenêtre phase-minimum (build_mp_window(N)).
    rfft_freqs : np.ndarray, shape (N//2+1,)
        Fréquences normalisées rfftfreq(N) = [0, 1/N, 2/N, …, 0.5].

    Retourne
    --------
    np.ndarray, shape (N,), dtype float32
    """
    H_mp    = minimum_phase_from_magnitude(mag, n_samples, mp_window)
    H_final = H_mp * np.exp(-1j * 2.0 * np.pi * rfft_freqs * tau)
    return np.real(irfft(H_final, n=n_samples)).astype(np.float32)
