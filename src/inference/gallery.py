from __future__ import annotations

from pathlib import Path

import numpy as np


class Gallery:
    """Base de référence : z_hrtf + coefficients SH de tous les sujets connus.

    Construite une seule fois depuis le train set.
    Rechargeable sans recalculer les embeddings.

    Usage
    -----
    gallery = Gallery.build(model, dataset, split='train')
    gallery.save('checkpoints/gallery.npz')

    gallery = Gallery.load('checkpoints/gallery.npz')
    subject_id, hrtf_coeffs, score = gallery.find_nearest(z_ear)
    """

    def __init__(
        self,
        subject_ids:  list[str],
        z_hrtf:       np.ndarray,   # [N, 128]  — L2-normalisé
        hrtf_coeffs:  np.ndarray,   # [N, n_sh, n_freqs, 2]  — dénormalisé
        train_mean:   float,
        train_std:    float,
    ) -> None:
        self.subject_ids = subject_ids
        self.z_hrtf      = z_hrtf
        self.hrtf_coeffs = hrtf_coeffs
        self.train_mean  = train_mean
        self.train_std   = train_std

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        model,      # ContrastiveModel
        dataset,    # MultimodalDataset
        split: str = 'train',
    ) -> Gallery:
        """Construit la gallery depuis un split du dataset.

        Pour l'évaluation : split='train'  (pas de fuite vers val/test)
        Pour la production : split='all'   (tous les sujets connus)
        """
        import tensorflow as tf

        split_map = {
            'train': dataset.train_idx,
            'val':   dataset.val_idx,
            'test':  dataset.test_idx,
        }

        if split == 'all':
            idx = np.concatenate([
                dataset.train_idx, dataset.val_idx, dataset.test_idx
            ])
        else:
            idx = split_map[split]

        hrtf_norm   = dataset._hrtf[idx].astype(np.float32)
        subject_ids = [dataset.subject_ids[i] for i in idx]

        # Passer les HRTF normalisées dans l'encodeur
        z_hrtf_list = []
        BATCH = 16
        for start in range(0, len(idx), BATCH):
            batch = tf.constant(hrtf_norm[start:start + BATCH])
            z     = model.hrtf_model(batch, training=False)
            z_hrtf_list.extend(z.numpy())

        z_hrtf = np.array(z_hrtf_list, dtype=np.float32)

        # Dénormaliser les coefficients pour un usage audio réel
        hrtf_coeffs = (hrtf_norm * (dataset.train_std + 1e-8)
                       + dataset.train_mean)

        print(f'  Gallery construite — {len(subject_ids)} sujets '
              f'(split={split})')
        return cls(subject_ids, z_hrtf, hrtf_coeffs,
                   dataset.train_mean, dataset.train_std)

    # ------------------------------------------------------------------
    # Recherche
    # ------------------------------------------------------------------

    def find_nearest(
        self,
        z_ear: np.ndarray,   # [128]  — L2-normalisé
    ) -> tuple[str, np.ndarray, float]:
        """Retourne (subject_id, hrtf_coeffs, similarity_score).

        similarity_score ∈ [-1, 1] — proche de 1 = match très fiable.
        """
        scores = self.z_hrtf @ z_ear            # [N]  cosine similarity
        best   = int(np.argmax(scores))
        return (
            self.subject_ids[best],
            self.hrtf_coeffs[best],
            float(scores[best]),
        )

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            str(path),
            subject_ids = np.array(self.subject_ids),
            z_hrtf      = self.z_hrtf,
            hrtf_coeffs = self.hrtf_coeffs,
            train_mean  = np.array([self.train_mean], dtype=np.float32),
            train_std   = np.array([self.train_std],  dtype=np.float32),
        )
        print(f'  Gallery sauvegardée → {path}')

    @classmethod
    def load(cls, path: str | Path) -> Gallery:
        data = np.load(str(path), allow_pickle=True)
        return cls(
            subject_ids = list(data['subject_ids']),
            z_hrtf      = data['z_hrtf'],
            hrtf_coeffs = data['hrtf_coeffs'],
            train_mean  = float(data['train_mean'].flat[0]),
            train_std   = float(data['train_std'].flat[0]),
        )

    def __repr__(self) -> str:
        return (f'Gallery(n={len(self.subject_ids)}, '
                f'D={self.z_hrtf.shape[1]})')