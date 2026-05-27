from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from .ear_processor import EarProcessor
from .hrtf_processor import HRTFProcessor
from .subject_record import SubjectRecord

_SOFA_SUFFIX = '_44K_16bit_256tap_FIR_SOFA.sofa'


class MultimodalDataset:
    """Dataset binaural cross-modal : photos d'oreilles ↔ HRTF.

    Deux points d'entrée :
        MultimodalDataset.build(db_root)   # depuis les fichiers bruts
        MultimodalDataset.load(npz_path)   # depuis un .npz prétraité

    Propriétés publiques :
        ds.train / ds.val / ds.test   → tf.data.Dataset
        ds.n_subjects                 → int
        ds.hrtf_shape                 → tuple
    """

    def __init__(
        self,
        hrtf_norm:    np.ndarray,
        img_L:        np.ndarray,
        img_R:        np.ndarray,
        subject_ids:  list[str],
        train_idx:    np.ndarray,
        val_idx:      np.ndarray,
        test_idx:     np.ndarray,
        train_mean:   float,
        train_std:    float,
        batch_size:   int = 4,
        sh_order:     int = 5,
        augmentation = None,   # AugmentationPipeline | None
    ) -> None:
        self._hrtf         = hrtf_norm
        self._img_L        = img_L
        self._img_R        = img_R
        self.subject_ids   = subject_ids
        self.train_idx     = train_idx
        self.val_idx       = val_idx
        self.test_idx      = test_idx
        self.train_mean    = train_mean
        self.train_std     = train_std
        self.batch_size    = batch_size
        self.sh_order      = sh_order
        self._augmentation = augmentation

    # ------------------------------------------------------------------
    # Constructeurs alternatifs
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        db_root:     str | Path,
        sh_order:    int = 5,
        img_size:    int = 224,
        batch_size:  int = 4,
        val_size:    float = 0.14,
        test_size:   float = 0.14,
        seed:        int = 42,
        augmentation = None,   # AugmentationPipeline | None
    ) -> MultimodalDataset:
        """Charge tous les sujets depuis db_root et construit le dataset."""
        db_root       = Path(db_root)
        hrtf_proc     = HRTFProcessor(mode='sh', sh_order=sh_order)
        ear_proc      = EarProcessor(img_size=img_size)

        records = _discover_subjects(db_root)
        print(f'{len(records)} sujets découverts\n')

        all_hrtf, all_img_L, all_img_R, ids = [], [], [], []

        for rec in records:
            print(f'  {rec.id:4s} ...', end=' ')
            try:
                hrtf         = hrtf_proc.load(rec.sofa)
                img_L, img_R = ear_proc.load(rec.img_L, rec.img_R)
                all_hrtf.append(hrtf)
                all_img_L.append(img_L)
                all_img_R.append(img_R)
                ids.append(rec.id)
                print(f'OK  {hrtf.shape}')
            except Exception as exc:
                print(f'ERREUR : {exc}')

        hrtf_array  = np.array(all_hrtf,  dtype=np.float32)
        img_L_array = np.array(all_img_L, dtype=np.float32)
        img_R_array = np.array(all_img_R, dtype=np.float32)

        # Expansion offline : génère les nouvelles paires avant le split
        if augmentation is not None:
            hrtf_array, img_L_array, img_R_array, ids = augmentation.expand_arrays(
                hrtf_array, img_L_array, img_R_array, ids
            )
            print(f'Dataset après expansion : {len(ids)} sujets')

        # Split sur les indices sujets
        indices = np.arange(len(ids))
        train_idx, temp_idx = train_test_split(indices, test_size=val_size + test_size, random_state=seed)
        val_idx, test_idx   = train_test_split(temp_idx, test_size=0.5, random_state=seed)

        # Z-score calculé sur le train uniquement
        train_mean = float(hrtf_array[train_idx].mean())
        train_std  = float(hrtf_array[train_idx].std())
        hrtf_norm  = (hrtf_array - train_mean) / (train_std + 1e-8)

        return cls(hrtf_norm, img_L_array, img_R_array, ids,
                   train_idx, val_idx, test_idx,
                   train_mean, train_std, batch_size, sh_order, augmentation)

    @classmethod
    def load(cls, npz_path: str | Path, batch_size: int = 4) -> MultimodalDataset:
        """Recharge un dataset prétraité depuis un fichier .npz."""
        data = np.load(str(npz_path), allow_pickle=True)
        return cls(
            hrtf_norm   = data['hrtf'],
            img_L       = data['img_L'],
            img_R       = data['img_R'],
            subject_ids = list(data['subject_ids']),
            train_idx   = data['train_idx'],
            val_idx     = data['val_idx'],
            test_idx    = data['test_idx'],
            train_mean  = float(data['train_mean']),
            train_std   = float(data['train_std']),
            batch_size  = batch_size,
            sh_order    = int(data['sh_order']),
        )

    # ------------------------------------------------------------------
    # Sauvegarde
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            str(path),
            hrtf        = self._hrtf.astype(np.float32),
            img_L       = self._img_L.astype(np.float32),
            img_R       = self._img_R.astype(np.float32),
            train_idx   = self.train_idx,
            val_idx     = self.val_idx,
            test_idx    = self.test_idx,
            subject_ids = np.array(self.subject_ids),
            train_mean  = np.array([self.train_mean], dtype=np.float32),
            train_std   = np.array([self.train_std],  dtype=np.float32),
            sh_order    = np.array([self.sh_order],   dtype=np.int32),
        )
        print(f'Dataset sauvegardé → {path}')

    # ------------------------------------------------------------------
    # tf.data.Dataset
    # ------------------------------------------------------------------

    @property
    def train(self):
        return self._make_ds(self.train_idx, shuffle=True)

    @property
    def val(self):
        return self._make_ds(self.val_idx)

    @property
    def test(self):
        return self._make_ds(self.test_idx)

    def _make_ds(self, idx: np.ndarray, shuffle: bool = False):
        import tensorflow as tf  # import paresseux : TF requis seulement ici
        ds = tf.data.Dataset.from_tensor_slices((
            {
                'ear_left':  self._img_L[idx].astype(np.float32),
                'ear_right': self._img_R[idx].astype(np.float32),
                'hrtf':      self._hrtf[idx].astype(np.float32),
            },
            np.arange(len(idx), dtype=np.int32),
        ))
        if shuffle:
            ds = ds.shuffle(buffer_size=len(idx), reshuffle_each_iteration=True)
            if self._augmentation is not None:
                ds = ds.map(self._augmentation, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    # ------------------------------------------------------------------
    # Propriétés utilitaires
    # ------------------------------------------------------------------

    @property
    def n_subjects(self) -> int:
        return len(self.subject_ids)

    @property
    def hrtf_shape(self) -> tuple:
        return tuple(self._hrtf.shape[1:])

    def __repr__(self) -> str:
        return (
            f'MultimodalDataset('
            f'n_subjects={self.n_subjects}, '
            f'sh_order={self.sh_order}, '
            f'hrtf_shape={self.hrtf_shape}, '
            f'train={len(self.train_idx)} / val={len(self.val_idx)} / test={len(self.test_idx)})'
        )


# ------------------------------------------------------------------
# Fonction privée : découverte des sujets
# ------------------------------------------------------------------

def _discover_subjects(db_root: Path) -> list[SubjectRecord]:
    records = []
    for subject_dir in sorted(db_root.iterdir()):
        if not subject_dir.is_dir():
            continue
        sid = subject_dir.name

        sofa_file = subject_dir / f'{sid}_HRIR_SOFA' / f'{sid}{_SOFA_SUFFIX}'
        if not sofa_file.exists():
            continue

        scan_dir = subject_dir / f'{sid}_Scans'
        img_L = scan_dir / f'{sid} (L).png'
        img_R = scan_dir / f'{sid} (R).png'
        if not img_L.exists():
            img_L = scan_dir / f'{sid}_(L).png'
            img_R = scan_dir / f'{sid}_(R).png'
        if not img_L.exists() or not img_R.exists():
            continue

        records.append(SubjectRecord(id=sid, sofa=sofa_file, img_L=img_L, img_R=img_R))
    return records
