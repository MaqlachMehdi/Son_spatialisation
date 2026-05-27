import numpy as np
from scipy.ndimage import rotate

from .base import AugmentationBase


class ImageTransforms(AugmentationBase):
    """Augmentations photométriques et géométriques sur les images d'oreilles.

    Ne modifie jamais la HRTF — uniquement les images.
    Justification : luminosité, contraste, bruit et légère rotation
    ne changent pas la morphologie de l'oreille, donc pas la HRTF.

    Deux modes :
        expand(n_augmentations=3) → offline, génère N copies par sujet
        apply()                   → online, transformation aléatoire à chaque appel

    Transforms appliquées dans l'ordre :
        1. Brightness  — shift aléatoire indépendant par oreille
        2. Contrast    — facteur multiplicatif indépendant par oreille
        3. Bruit       — bruit gaussien indépendant par oreille
        4. Rotation    — même angle pour L et R (simule inclinaison de tête)

    Parameters
    ----------
    brightness_range   : amplitude max du shift de luminosité (±)
    contrast_range     : amplitude max de la variation de contraste (±)
    noise_sigma        : écart-type du bruit gaussien (images dans [0,1])
    max_rotation_deg   : angle max en degrés pour la rotation (±)
    n_augmentations    : nombre de copies augmentées par sujet (pour expand)
    p                  : probabilité d'application (pour apply online)
    """

    def __init__(
        self,
        brightness_range:  float = 0.15,
        contrast_range:    float = 0.15,
        noise_sigma:       float = 0.02,
        max_rotation_deg:  float = 5.0,
        n_augmentations:   int   = 3,
        p:                 float = 0.6,
    ) -> None:
        super().__init__(p)
        self.brightness_range  = brightness_range
        self.contrast_range    = contrast_range
        self.noise_sigma       = noise_sigma
        self.max_rotation_deg  = max_rotation_deg
        self.n_augmentations   = n_augmentations

    # ------------------------------------------------------------------
    # Mode offline : expansion du dataset
    # ------------------------------------------------------------------

    def expand(
        self,
        hrtf_array:  np.ndarray,   # [N, n_sh, N_freqs, 2]
        img_L_array: np.ndarray,   # [N, 224, 224, 3]
        img_R_array: np.ndarray,   # [N, 224, 224, 3]
        subject_ids: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Génère n_augmentations copies augmentées par sujet.

        La HRTF est identique pour chaque copie — seules les images changent.
        N sujets → N × (1 + n_augmentations) sujets au total.
        """
        N = len(subject_ids)

        aug_hrtf  = []
        aug_img_L = []
        aug_img_R = []
        aug_ids   = []

        for k in range(1, self.n_augmentations + 1):
            for i in range(N):
                img_L_aug, img_R_aug = self._transform_pair(
                    img_L_array[i], img_R_array[i]
                )
                aug_hrtf.append(hrtf_array[i])
                aug_img_L.append(img_L_aug)
                aug_img_R.append(img_R_aug)
                aug_ids.append(f'{subject_ids[i]}_aug{k}')

        hrtf_out  = np.concatenate([hrtf_array,
                                    np.array(aug_hrtf,  dtype=np.float32)], axis=0)
        img_L_out = np.concatenate([img_L_array,
                                    np.array(aug_img_L, dtype=np.float32)], axis=0)
        img_R_out = np.concatenate([img_R_array,
                                    np.array(aug_img_R, dtype=np.float32)], axis=0)
        ids_out   = subject_ids + aug_ids

        print(f'  ImageTransforms.expand : {N} → {len(ids_out)} sujets '
              f'({self.n_augmentations} copies par sujet)')
        return hrtf_out, img_L_out, img_R_out, ids_out

    # ------------------------------------------------------------------
    # Mode online : transformation d'un sample
    # ------------------------------------------------------------------

    def apply(
        self,
        ear_L: np.ndarray,   # [224, 224, 3]
        ear_R: np.ndarray,   # [224, 224, 3]
        hrtf:  np.ndarray,   # [n_sh, N_freqs, 2]  — inchangé
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        ear_L_aug, ear_R_aug = self._transform_pair(ear_L, ear_R)
        return ear_L_aug, ear_R_aug, hrtf

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _transform_pair(
        self,
        ear_L: np.ndarray,
        ear_R: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Applique les 4 transforms sur la paire (L, R)."""

        # 1. Brightness — shift indépendant par oreille
        ear_L = self._brightness(ear_L)
        ear_R = self._brightness(ear_R)

        # 2. Contrast — facteur indépendant par oreille
        ear_L = self._contrast(ear_L)
        ear_R = self._contrast(ear_R)

        # 3. Bruit gaussien — indépendant par oreille
        ear_L = self._gaussian_noise(ear_L)
        ear_R = self._gaussian_noise(ear_R)

        # 4. Rotation — même angle pour L et R (inclinaison de tête)
        angle = np.random.uniform(-self.max_rotation_deg, self.max_rotation_deg)
        ear_L = self._rotate(ear_L, angle)
        ear_R = self._rotate(ear_R, angle)

        return ear_L, ear_R

    def _brightness(self, img: np.ndarray) -> np.ndarray:
        delta = np.random.uniform(-self.brightness_range, self.brightness_range)
        return np.clip(img + delta, 0.0, 1.0).astype(np.float32)

    def _contrast(self, img: np.ndarray) -> np.ndarray:
        factor = np.random.uniform(1.0 - self.contrast_range, 1.0 + self.contrast_range)
        mean   = img.mean()
        return np.clip(mean + factor * (img - mean), 0.0, 1.0).astype(np.float32)

    def _gaussian_noise(self, img: np.ndarray) -> np.ndarray:
        noise = np.random.normal(0.0, self.noise_sigma, img.shape).astype(np.float32)
        return np.clip(img + noise, 0.0, 1.0).astype(np.float32)

    def _rotate(self, img: np.ndarray, angle: float) -> np.ndarray:
        # reshape=False conserve la taille 224×224
        # cval=0.0 remplit les bords avec du noir (cohérent avec le fond des photos)
        rotated = rotate(img, angle=angle, axes=(0, 1),
                         reshape=False, cval=0.0, order=1)
        return rotated.astype(np.float32)