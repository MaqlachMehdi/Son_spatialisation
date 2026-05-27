from abc import ABC, abstractmethod

import numpy as np


class AugmentationBase(ABC):
    """Classe de base pour toutes les augmentations.

    Deux modes d'utilisation :

    1. Online  — apply() / __call__()
       Transforme UN sample à la volée dans tf.data.map().
       Ne change pas le nombre d'exemples.

    2. Offline — expand()
       Prend les tableaux numpy complets et retourne des tableaux augmentés.
       Peut augmenter le nombre d'exemples (ex. doubler pour la réflexion).
       Appelé UNE FOIS dans MultimodalDataset.build(), avant le split.
    """

    def __init__(self, p: float = 0.5) -> None:
        if not 0.0 <= p <= 1.0:
            raise ValueError(f"p doit être dans [0, 1], reçu {p}")
        self.p = p

    @abstractmethod
    def apply(
        self,
        ear_L: np.ndarray,
        ear_R: np.ndarray,
        hrtf:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Transforme un sample (online). À implémenter dans chaque sous-classe."""
        ...

    def expand(
        self,
        hrtf_array:  np.ndarray,   # [N, n_sh, N_freqs, 2]
        img_L_array: np.ndarray,   # [N, 224, 224, 3]
        img_R_array: np.ndarray,   # [N, 224, 224, 3]
        subject_ids: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Expansion offline des tableaux numpy (optionnel).

        Par défaut retourne les arrays inchangés.
        Les sous-classes qui génèrent de nouveaux exemples surchargent cette méthode.
        """
        return hrtf_array, img_L_array, img_R_array, subject_ids

    def __call__(
        self,
        ear_L: np.ndarray,
        ear_R: np.ndarray,
        hrtf:  np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if np.random.random() < self.p:
            return self.apply(ear_L, ear_R, hrtf)
        return ear_L, ear_R, hrtf