from __future__ import annotations

from pathlib import Path

import numpy as np


class EmbeddingSpace:
    """Calcule et stocke les embeddings z_ear et z_hrtf pour tout le dataset.

    Usage
    -----
    # Calculer (une seule fois — coûteux)
    space = EmbeddingSpace.compute(model, dataset)
    space.save('checkpoints/embeddings.npz')

    # Recharger (rapide)
    space = EmbeddingSpace.load('checkpoints/embeddings.npz')

    # Utiliser
    visualizer = EmbeddingVisualizer(space)
    visualizer.plot_alignment(split='test')
    """

    def __init__(
        self,
        z_ear:       np.ndarray,   # [N, D]
        z_hrtf:      np.ndarray,   # [N, D]
        subject_ids: list[str],    # longueur N
        splits:      list[str],    # 'train' | 'val' | 'test'  longueur N
    ) -> None:
        self.z_ear       = z_ear
        self.z_hrtf      = z_hrtf
        self.subject_ids = subject_ids
        self.splits      = splits

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def compute(
        cls,
        model,     # ContrastiveModel
        dataset,   # MultimodalDataset
        batch_size: int = 16,
    ) -> EmbeddingSpace:
        """Passe tout le dataset dans le modèle et collecte les embeddings."""
        import tensorflow as tf

        z_ears, z_hrtfs, ids, split_labels = [], [], [], []

        for split_name, idx in [
            ('train', dataset.train_idx),
            ('val',   dataset.val_idx),
            ('test',  dataset.test_idx),
        ]:
            ear_L   = dataset._img_L[idx].astype(np.float32)
            ear_R   = dataset._img_R[idx].astype(np.float32)
            hrtf    = dataset._hrtf[idx].astype(np.float32)
            subj    = [dataset.subject_ids[i] for i in idx]

            for start in range(0, len(idx), batch_size):
                end   = start + batch_size
                batch = {
                    'ear_left':  tf.constant(ear_L[start:end]),
                    'ear_right': tf.constant(ear_R[start:end]),
                    'hrtf':      tf.constant(hrtf[start:end]),
                }
                z_e, z_h = model(batch, training=False)

                z_ears.extend(z_e.numpy())
                z_hrtfs.extend(z_h.numpy())
                ids.extend(subj[start:end])
                split_labels.extend([split_name] * len(subj[start:end]))

            print(f'  {split_name:5s} — {len(idx)} sujets encodés')

        return cls(
            z_ear       = np.array(z_ears,  dtype=np.float32),
            z_hrtf      = np.array(z_hrtfs, dtype=np.float32),
            subject_ids = ids,
            splits      = split_labels,
        )

    # ------------------------------------------------------------------
    # Persistance
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            str(path),
            z_ear       = self.z_ear,
            z_hrtf      = self.z_hrtf,
            subject_ids = np.array(self.subject_ids),
            splits      = np.array(self.splits),
        )
        print(f'  EmbeddingSpace sauvegardé → {path}')

    @classmethod
    def load(cls, path: str | Path) -> EmbeddingSpace:
        data = np.load(str(path), allow_pickle=True)
        return cls(
            z_ear       = data['z_ear'],
            z_hrtf      = data['z_hrtf'],
            subject_ids = list(data['subject_ids']),
            splits      = list(data['splits']),
        )

    # ------------------------------------------------------------------
    # Filtrage
    # ------------------------------------------------------------------

    def filter(self, split: str) -> EmbeddingSpace:
        """Retourne un sous-espace filtré sur un split donné."""
        mask = [s == split for s in self.splits]
        idx  = np.where(mask)[0]
        return EmbeddingSpace(
            z_ear       = self.z_ear[idx],
            z_hrtf      = self.z_hrtf[idx],
            subject_ids = [self.subject_ids[i] for i in idx],
            splits      = [self.splits[i] for i in idx],
        )

    def __repr__(self) -> str:
        counts = {s: self.splits.count(s) for s in ['train', 'val', 'test']}
        return (f'EmbeddingSpace(N={len(self.subject_ids)}, '
                f'D={self.z_ear.shape[1]}, splits={counts})')