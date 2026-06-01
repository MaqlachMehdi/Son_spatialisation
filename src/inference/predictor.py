from __future__ import annotations

from pathlib import Path

import numpy as np

from .gallery import Gallery


class HRTFPredictor:
    """Pipeline d'inférence complet : photo d'oreille → HRTF personnalisée.

    Usage
    -----
    # Construire
    predictor = HRTFPredictor.build(model, dataset, sh_order=10)
    predictor.save_gallery('checkpoints/gallery.npz')

    # Inférence depuis des chemins d'images
    result = predictor.predict('ear_L.png', 'ear_R.png')
    print(result['subject_id'])    # 'H10'
    print(result['similarity'])    # 0.87
    print(result['hrtf_coeffs'])   # [121, 129, 2]

    # Inférence depuis des arrays numpy déjà chargés [224, 224, 3] float32 [0,1]
    result = predictor.predict(img_L_array, img_R_array)

    # Décoder vers des positions spécifiques
    hrtf = predictor.predict_at_positions(
        'ear_L.png', 'ear_R.png',
        positions_deg=my_grid,   # [N_pos, 2]  colonnes = [azimut, élévation]
    )
    # hrtf : [N_pos, N_freqs, 2]
    """

    def __init__(
        self,
        model,
        gallery:  Gallery,
        sh_order: int = 10,
    ) -> None:
        self._model   = model
        self._gallery = gallery
        self._sh_order = sh_order

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        model,
        dataset,
        split:    str = 'train',
        sh_order: int = 10,
    ) -> HRTFPredictor:
        """Construit la gallery et retourne un prédicteur prêt à l'emploi."""
        gallery = Gallery.build(model, dataset, split=split)
        return cls(model, gallery, sh_order=sh_order)

    # ------------------------------------------------------------------
    # Inférence principale
    # ------------------------------------------------------------------

    def predict(
        self,
        img_L,   # chemin (str | Path) ou np.ndarray [224, 224, 3] float32 [0,1]
        img_R,
    ) -> dict:
        """Retourne le sujet le plus proche et ses coefficients SH HRTF.

        Returns
        -------
        {
            'subject_id'  : str          — identifiant du sujet matché
            'similarity'  : float        — score cosine ∈ [-1, 1]
            'hrtf_coeffs' : np.ndarray   — [n_sh, n_freqs, 2]
        }
        """
        import tensorflow as tf

        ear_L, ear_R = self._preprocess(img_L, img_R)

        batch = {
            'ear_left':  tf.constant(ear_L[np.newaxis]),   # [1, 224, 224, 3]
            'ear_right': tf.constant(ear_R[np.newaxis]),
        }
        z_ear = self._model.ear_model(batch, training=False)[0].numpy()  # [128]

        subject_id, hrtf_coeffs, similarity = self._gallery.find_nearest(z_ear)

        return {
            'subject_id'  : subject_id,
            'similarity'  : similarity,
            'hrtf_coeffs' : hrtf_coeffs,
        }

    def predict_at_positions(
        self,
        img_L,
        img_R,
        positions_deg: np.ndarray,   # [N_pos, 2]  colonnes = [azimut, élévation]
    ) -> np.ndarray:
        """Retourne la HRTF décodée aux positions demandées.

        Returns
        -------
        np.ndarray [N_pos, N_freqs, 2]
        """
        from src.multimodal.sh_transform import SphericalHarmonicsTransform

        result = self.predict(img_L, img_R)
        sh     = SphericalHarmonicsTransform(order=self._sh_order)
        hrtf   = sh.decode(result['hrtf_coeffs'], positions_deg)

        print(f"  Sujet matché  : {result['subject_id']}")
        print(f"  Similarité    : {result['similarity']:.3f}")
        print(f"  HRTF décodée  : {hrtf.shape}")
        return hrtf

    # ------------------------------------------------------------------
    # Prétraitement des images
    # ------------------------------------------------------------------

    def _preprocess(
        self,
        img_L,
        img_R,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Accepte des chemins ou des arrays numpy déjà normalisés."""
        from pathlib import Path as _Path

        if isinstance(img_L, (str, _Path)):
            from src.multimodal.ear_processor import EarProcessor
            proc  = EarProcessor(img_size=224)
            ear_L, ear_R = proc.load(_Path(img_L), _Path(img_R))
        else:
            ear_L = np.asarray(img_L, dtype=np.float32)
            ear_R = np.asarray(img_R, dtype=np.float32)

        return ear_L, ear_R

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def save_gallery(self, path: str | Path) -> None:
        self._gallery.save(path)

    @classmethod
    def load(
        cls,
        gallery_path: str | Path,
        model,
        sh_order: int = 10,
    ) -> HRTFPredictor:
        """Recharge un prédicteur depuis une gallery sauvegardée."""
        gallery = Gallery.load(gallery_path)
        return cls(model, gallery, sh_order=sh_order)

    def __repr__(self) -> str:
        return (f'HRTFPredictor(gallery={self._gallery}, '
                f'sh_order={self._sh_order})')