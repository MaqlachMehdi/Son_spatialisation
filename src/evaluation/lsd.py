from __future__ import annotations

import numpy as np


def log_spectral_distance(
    hrtf_pred: np.ndarray,   # [n_sh, n_freqs, 2]
    hrtf_true: np.ndarray,   # [n_sh, n_freqs, 2]
    eps: float = 1e-8,
) -> float:
    """Log Spectral Distance entre deux HRTFs en coefficients SH.

    Mesure la distorsion spectrale moyenne en dB.
    Valeur < 5 dB = acoustiquement acceptable.

    Le calcul est fait directement sur les coefficients SH — pas besoin
    de décoder vers des positions spécifiques. On compare les profils
    spectraux coefficient par coefficient, puis on moyenne.

    Parameters
    ----------
    hrtf_pred : coefficients SH de la HRTF prédite  [n_sh, n_freqs, 2]
    hrtf_true : coefficients SH de la HRTF réelle   [n_sh, n_freqs, 2]
    eps       : évite log(0)

    Returns
    -------
    float  LSD en dB
    """
    mag_pred = np.abs(hrtf_pred) + eps   # [n_sh, n_freqs, 2]
    mag_true = np.abs(hrtf_true) + eps

    log_diff = 20.0 * np.log10(mag_pred) - 20.0 * np.log10(mag_true)
    return float(np.sqrt(np.mean(log_diff ** 2)))


def lsd_dataset(
    hrtf_preds: np.ndarray,   # [N, n_sh, n_freqs, 2]
    hrtf_trues: np.ndarray,   # [N, n_sh, n_freqs, 2]
) -> dict:
    """Calcule la LSD sur un ensemble de sujets.

    Returns
    -------
    dict : mean, std, min, max
    """
    scores = [
        log_spectral_distance(hrtf_preds[i], hrtf_trues[i])
        for i in range(len(hrtf_preds))
    ]
    return {
        'lsd_mean': float(np.mean(scores)),
        'lsd_std' : float(np.std(scores)),
        'lsd_min' : float(np.min(scores)),
        'lsd_max' : float(np.max(scores)),
    }