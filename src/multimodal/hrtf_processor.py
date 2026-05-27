from pathlib import Path

import numpy as np
import sofa

from .sh_transform import SphericalHarmonicsTransform


class HRTFProcessor:
    """Charge un fichier SOFA et retourne une représentation fixe de la HRTF.

    Parameters
    ----------
    mode : 'sh' | 'fft'
        'sh'  — décomposition en harmoniques sphériques → shape [(L+1)², N_freqs, 2]
                Recommandé : shape fixe quelle que soit la grille de mesure.
        'fft' — log-magnitude brute, moyenne sur les positions → shape [N_freqs, 2]
                Rapide mais perd l'information spatiale.
    sh_order : int
        Ordre de troncature SH (utilisé seulement si mode='sh').
    """

    def __init__(self, mode: str = 'sh', sh_order: int = 5) -> None:
        if mode not in ('sh', 'fft'):
            raise ValueError("mode doit être 'sh' ou 'fft'")
        self.mode = mode
        self._sh  = SphericalHarmonicsTransform(order=sh_order) if mode == 'sh' else None

    def load(self, sofa_path: Path) -> np.ndarray:
        """Charge et encode la HRTF depuis un fichier SOFA.

        Returns
        -------
        mode='sh'  → [n_sh, N_freqs, 2]    n_sh = (sh_order+1)²
        mode='fft' → [N_freqs, 2]           moyenne sur les positions
        """
        if self.mode == 'sh':
            return self._sh.encode_subject(sofa_path)

        # mode='fft' — log-magnitude moyennée sur toutes les positions
        db      = sofa.Database.open(str(sofa_path))
        irs     = db.Data.IR.get_values()   # [N_pos, 2, N_taps]
        db.close()
        spectrum = np.fft.rfft(irs, axis=-1)
        log_mag  = 20.0 * np.log10(np.abs(spectrum) + 1e-8)
        features = np.transpose(log_mag, (0, 2, 1)).astype(np.float32)  # [N_pos, N_freqs, 2]
        return features.mean(axis=0)   # [N_freqs, 2]
