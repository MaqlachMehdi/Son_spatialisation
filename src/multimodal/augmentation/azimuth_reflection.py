import numpy as np

from .base import AugmentationBase


class AzimuthalReflection(AugmentationBase):
    """Réflexion azimutale gauche/droite — crée un sujet miroir.

    Transformation appliquée simultanément sur images et HRTF :

    Images :
        oreille gauche  →  flip horizontal de l'oreille droite originale
        oreille droite  →  flip horizontal de l'oreille gauche originale

    Coefficients SH (transformation φ → −φ, exacte) :
        coefficients m < 0  (termes en sin) : changent de signe
        coefficients m ≥ 0  (termes en cos et DC) : inchangés
        puis échange des deux oreilles (axe 2 du tenseur)

    Deux usages :
        expand()  →  N sujets → 2N sujets (offline, avant le split)
        apply()   →  1 sample → 1 sample  (online, dans tf.data.map)

    Parameters
    ----------
    sh_order : ordre de la décomposition SH utilisée pour encoder la HRTF
    p        : probabilité d'application pour le mode online (apply)
               ignoré en mode offline (expand applique toujours la réflexion)
    """

    def __init__(self, sh_order: int, p: float = 0.5) -> None:
        super().__init__(p)
        self.sh_order   = sh_order
        self._sign_mask = self._build_sign_mask(sh_order)  # [n_sh]

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
        """Génère les sujets miroirs et les concatène aux originaux.

        N sujets  →  2N sujets  (original + réfléchi pour chacun)
        Les IDs des sujets miroirs reçoivent le suffixe '_mirror'.
        """
        N = len(subject_ids)

        # Appliquer la réflexion sur chaque sujet
        hrtf_ref  = np.empty_like(hrtf_array)
        img_L_ref = np.empty_like(img_L_array)
        img_R_ref = np.empty_like(img_R_array)

        for i in range(N):
            img_L_ref[i], img_R_ref[i], hrtf_ref[i] = self.apply(
                img_L_array[i], img_R_array[i], hrtf_array[i]
            )

        ids_ref = [f'{sid}_mirror' for sid in subject_ids]

        # Concaténation : originaux + réfléchis
        hrtf_out  = np.concatenate([hrtf_array,  hrtf_ref],  axis=0)
        img_L_out = np.concatenate([img_L_array, img_L_ref], axis=0)
        img_R_out = np.concatenate([img_R_array, img_R_ref], axis=0)
        ids_out   = subject_ids + ids_ref

        print(f'  AzimuthalReflection.expand : {N} → {len(ids_out)} sujets')
        return hrtf_out, img_L_out, img_R_out, ids_out

    # ------------------------------------------------------------------
    # Mode online : transformation d'un sample
    # ------------------------------------------------------------------

    def apply(
        self,
        ear_L: np.ndarray,   # [224, 224, 3]
        ear_R: np.ndarray,   # [224, 224, 3]
        hrtf:  np.ndarray,   # [n_sh, N_freqs, 2]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:

        # Images : swap + flip horizontal (axis=1 = colonnes)
        ear_L_new = np.ascontiguousarray(np.flip(ear_R, axis=1))
        ear_R_new = np.ascontiguousarray(np.flip(ear_L, axis=1))

        # HRTF : masque de signe [n_sh] sur axe 0, puis swap oreilles
        hrtf_ref = hrtf * self._sign_mask[:, np.newaxis, np.newaxis]
        hrtf_ref = np.ascontiguousarray(hrtf_ref[:, :, ::-1])

        return ear_L_new, ear_R_new, hrtf_ref

    # ------------------------------------------------------------------

    @staticmethod
    def _build_sign_mask(sh_order: int) -> np.ndarray:
        """Masque de signe pour la réflexion φ → −φ dans la base SH réelle.

        m < 0  →  sin(|m|φ) → −sin(|m|φ)  →  signe −1
        m ≥ 0  →  cos(m φ)  →  cos(m φ)   →  signe +1
        """
        n_sh = (sh_order + 1) ** 2
        mask = np.ones(n_sh, dtype=np.float32)
        idx  = 0
        for l in range(sh_order + 1):
            for m in range(-l, l + 1):
                if m < 0:
                    mask[idx] = -1.0
                idx += 1
        return mask