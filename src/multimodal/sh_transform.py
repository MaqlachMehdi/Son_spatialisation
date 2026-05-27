from __future__ import annotations

from math import factorial, sqrt, pi
from pathlib import Path

import numpy as np
import sofa
from scipy.special import lpmv


class SphericalHarmonicsTransform:
    """Décompose une HRTF en coefficients d'harmoniques sphériques.

    Usage
    -----
    sh = SphericalHarmonicsTransform(order=5)

    # Pipeline complet depuis un fichier SOFA
    coeffs = sh.encode_subject(sofa_path)        # [(L+1)², N_freqs, 2]

    # Ou étape par étape
    sh.fit(positions_deg)                         # précalcule Y⁺
    coeffs = sh.encode(hrtf)                      # [N_pos, N_freqs, 2] → [(L+1)², N_freqs, 2]

    # Reconstruction à n'importe quelle direction
    hrtf_recon = sh.decode(coeffs, new_positions_deg)
    """

    def __init__(self, order: int = 5) -> None:
        self.order = order
        self.n_sh  = (order + 1) ** 2
        self._Y_pinv: np.ndarray | None = None

    # ------------------------------------------------------------------
    # API publique
    # ------------------------------------------------------------------

    def fit(self, positions_deg: np.ndarray) -> SphericalHarmonicsTransform:
        """Précalcule la pseudo-inverse Y⁺ pour les positions de mesure.

        Parameters
        ----------
        positions_deg : [N_pos, 2]  azimut et élévation en degrés
        """
        Y            = self._build_Y(positions_deg)   # [N_pos, n_sh]
        self._Y_pinv = np.linalg.pinv(Y)              # [n_sh,  N_pos]
        return self

    def encode(self, hrtf: np.ndarray) -> np.ndarray:
        """Projette une HRTF sur la base SH.

        Calcule A = Y⁺ · H  où  Y⁺ est la pseudo-inverse précalculée par fit().

        Parameters
        ----------
        hrtf : [N_pos, N_freqs, 2]  log-magnitude en dB

        Returns
        -------
        coeffs : [n_sh, N_freqs, 2]  float32
        """
        if self._Y_pinv is None:
            raise RuntimeError("Appeler fit() avant encode()")
        N_pos, N_freqs, _ = hrtf.shape

        # Y⁺ · H :  [n_sh, N_pos] @ [N_pos, N_freqs*2]  =  [n_sh, N_freqs*2]
        A = self._Y_pinv @ hrtf.reshape(N_pos, N_freqs * 2)
        return A.reshape(self.n_sh, N_freqs, 2).astype(np.float32)

    def decode(self, coeffs: np.ndarray, positions_deg: np.ndarray) -> np.ndarray:
        """Reconstruit la HRTF à des directions arbitraires.

        Calcule H = Y · A  pour les nouvelles positions.

        Parameters
        ----------
        coeffs        : [n_sh, N_freqs, 2]
        positions_deg : [M, 2]  azimut et élévation cibles en degrés

        Returns
        -------
        hrtf_recon : [M, N_freqs, 2]  float32
        """
        n_sh, N_freqs, _ = coeffs.shape
        Y = self._build_Y(positions_deg)                        # [M, n_sh]

        # Y · A :  [M, n_sh] @ [n_sh, N_freqs*2]  =  [M, N_freqs*2]
        H = Y @ coeffs.reshape(n_sh, N_freqs * 2)
        return H.reshape(len(positions_deg), N_freqs, 2).astype(np.float32)

    def encode_subject(self, sofa_path: Path) -> np.ndarray:
        """Pipeline complet : ouvre le SOFA, fit, encode.

        Returns
        -------
        coeffs : [n_sh, N_freqs, 2]  float32
        """
        positions, hrtf = self._load_sofa(sofa_path)
        self.fit(positions)
        return self.encode(hrtf)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _load_sofa(self, sofa_path: Path) -> tuple[np.ndarray, np.ndarray]:
        """Charge positions et log-magnitudes depuis un fichier SOFA."""
        db        = sofa.Database.open(str(sofa_path))
        positions = db.SourcePosition.get_values()[:, :2].copy()  # [N_pos, 2] az/el degrés
        irs       = db.Data.IR.get_values()                        # [N_pos, 2, N_taps]
        db.close()

        spectrum = np.fft.rfft(irs, axis=-1)
        log_mag  = 20.0 * np.log10(np.abs(spectrum) + 1e-8)       # [N_pos, 2, N_freqs]
        hrtf     = np.transpose(log_mag, (0, 2, 1)).astype(np.float32)  # [N_pos, N_freqs, 2]
        return positions, hrtf

    def _build_Y(self, positions_deg: np.ndarray) -> np.ndarray:
        """Construit la matrice de design SH  Y [N_pos, n_sh].

        Chaque colonne = une harmonique sphérique évaluée sur toutes les positions.
        """
        phi   = np.radians(positions_deg[:, 0] % 360)       # azimut    [0, 2π]
        theta = np.radians(90.0 - positions_deg[:, 1])       # colatitude [0, π]

        Y   = np.zeros((len(positions_deg), self.n_sh), dtype=np.float64)
        idx = 0
        for l in range(self.order + 1):
            for m in range(-l, l + 1):
                Y[:, idx] = _real_sph_harm(l, m, phi, theta)
                idx += 1
        return Y


# ------------------------------------------------------------------
# Harmonique sphérique réelle — fonction module-level (utilisée aussi par SHVisualizer)
# ------------------------------------------------------------------

def _real_sph_harm(l: int, m: int, phi: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """Harmonique sphérique réelle Y_l^m(phi, theta).

    phi   : azimut en radians [0, 2π]
    theta : colatitude en radians [0, π]

    Formule :
        K(l, m) = sqrt( (2l+1)/(4π) · (l−|m|)! / (l+|m|)! )

        m > 0  →  sqrt(2) · K · cos(m·phi) · P_l^|m|(cos theta)
        m = 0  →  K · P_l^0(cos theta)
        m < 0  →  sqrt(2) · K · sin(|m|·phi) · P_l^|m|(cos theta)
    """
    abs_m = abs(m)
    K     = sqrt((2 * l + 1) / (4 * pi) * factorial(l - abs_m) / factorial(l + abs_m))
    P     = lpmv(abs_m, l, np.cos(theta))   # polynôme de Legendre associé P_l^|m|(cos θ)

    if m > 0:
        return sqrt(2) * K * np.cos(m * phi) * P
    elif m == 0:
        return K * P
    else:
        return sqrt(2) * K * np.sin(abs_m * phi) * P
