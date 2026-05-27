from __future__ import annotations

import numpy as np

from .base import AugmentationBase


class AugmentationPipeline:
    """Compose plusieurs augmentations et s'intègre dans tf.data.Dataset.map().

    Usage
    -----
    from src.multimodal.augmentation import AugmentationPipeline, AzimuthalReflection

    pipeline = AugmentationPipeline([
        AzimuthalReflection(sh_order=5, p=0.5),
    ])

    ds = MultimodalDataset.build(db_root, augmentation=pipeline)
    # → appliqué automatiquement sur ds.train, jamais sur ds.val / ds.test
    """

    def __init__(self, augmentations: list[AugmentationBase]) -> None:
        self.augmentations = augmentations

    def expand_arrays(
        self,
        hrtf_array:  np.ndarray,
        img_L_array: np.ndarray,
        img_R_array: np.ndarray,
        subject_ids: list[str],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
        """Applique l'expansion offline de toutes les augmentations.

        Appelé une seule fois dans MultimodalDataset.build(), avant le split.
        Chaque augmentation peut augmenter le nombre de sujets.
        """
        for aug in self.augmentations:
            hrtf_array, img_L_array, img_R_array, subject_ids = aug.expand(
                hrtf_array, img_L_array, img_R_array, subject_ids
            )
        return hrtf_array, img_L_array, img_R_array, subject_ids

    def __call__(self, sample: dict, label):
        """Appelé par tf.data.Dataset.map().

        Reçoit des tenseurs TF, applique les augmentations numpy via
        tf.numpy_function, retourne des tenseurs TF de même shape.
        """
        import tensorflow as tf  # import paresseux — TF requis seulement ici

        def _apply(ear_L: np.ndarray, ear_R: np.ndarray, hrtf: np.ndarray):
            for aug in self.augmentations:
                ear_L, ear_R, hrtf = aug(ear_L, ear_R, hrtf)
            return (
                ear_L.astype(np.float32),
                ear_R.astype(np.float32),
                hrtf.astype(np.float32),
            )

        ear_L, ear_R, hrtf = tf.numpy_function(
            _apply,
            [sample['ear_left'], sample['ear_right'], sample['hrtf']],
            [tf.float32, tf.float32, tf.float32],
        )

        # tf.numpy_function perd les shapes statiques — on les restaure
        # pour que le reste du pipeline TF puisse les inférer
        ear_L.set_shape(sample['ear_left'].shape)
        ear_R.set_shape(sample['ear_right'].shape)
        hrtf.set_shape(sample['hrtf'].shape)

        return {'ear_left': ear_L, 'ear_right': ear_R, 'hrtf': hrtf}, label
