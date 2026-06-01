from __future__ import annotations

import numpy as np


def _base_id(subject_id: str) -> str:
    """Extrait l'identifiant de base : 'H10_mirror_aug1' → 'H10'."""
    return subject_id.split('_')[0]


class RecallCallback:
    """Callback Keras : calcule Recall@1 sur le val set à chaque epoch.

      Note sur le data leakage partiel :
    Le dataset a été splitté APRÈS augmentation. Certains sujets du val set
    ont leurs versions augmentées (mirror, aug) dans le train set.
    Pour mitiger cela, on considère qu'une prédiction est correcte si
    le base_id correspond (ex: H10_mirror → H10 est correct).
    Cela reste une évaluation optimiste — un split avant augmentation
    serait plus rigoureux.

    La gallery est recalculée à chaque epoch car les poids du modèle changent.
    """

    def __init__(
        self,
        model,       # ContrastiveModel
        dataset,     # MultimodalDataset
        log_mlflow:  bool = True,
        batch_size:  int  = 16,
    ) -> None:
        self._model      = model
        self._dataset    = dataset
        self._log_mlflow = log_mlflow
        self._batch_size = batch_size

    # Keras callback interface
    def set_model(self, model):   pass
    def set_params(self, params): pass
    def on_train_begin(self, logs=None): pass
    def on_train_end(self, logs=None):   pass
    def on_batch_begin(self, batch, logs=None): pass
    def on_batch_end(self, batch, logs=None):   pass
    def on_epoch_begin(self, epoch, logs=None): pass

    def on_epoch_end(self, epoch, logs=None) -> None:
        import tensorflow as tf

        # ------------------------------------------------------------------
        # 1. Construire la gallery depuis le TRAIN set uniquement
        # ------------------------------------------------------------------
        train_idx = self._dataset.train_idx
        hrtf_norm = self._dataset._hrtf[train_idx].astype(np.float32)
        train_ids = [self._dataset.subject_ids[i] for i in train_idx]

        z_hrtf_list = []
        for start in range(0, len(train_idx), self._batch_size):
            batch = tf.constant(hrtf_norm[start:start + self._batch_size])
            z     = self._model.hrtf_model(batch, training=False)
            z_hrtf_list.extend(z.numpy())
        z_hrtf_train = np.array(z_hrtf_list, dtype=np.float32)

        # ------------------------------------------------------------------
        # 2. Encoder les oreilles du VAL set
        # ------------------------------------------------------------------
        val_idx = self._dataset.val_idx
        ear_L   = self._dataset._img_L[val_idx].astype(np.float32)
        ear_R   = self._dataset._img_R[val_idx].astype(np.float32)
        val_ids = [self._dataset.subject_ids[i] for i in val_idx]

        z_ear_list = []
        for start in range(0, len(val_idx), self._batch_size):
            batch = {
                'ear_left':  tf.constant(ear_L[start:start + self._batch_size]),
                'ear_right': tf.constant(ear_R[start:start + self._batch_size]),
            }
            z = self._model.ear_model(batch, training=False)
            z_ear_list.extend(z.numpy())
        z_ear_val = np.array(z_ear_list, dtype=np.float32)

        # ------------------------------------------------------------------
        # 3. Recall@1 — critère : même base_id (gère le leakage partiel)
        # ------------------------------------------------------------------
        sim    = z_ear_val @ z_hrtf_train.T   # [N_val, N_train]
        hits   = 0
        for i, val_id in enumerate(val_ids):
            best_train_idx = int(np.argmax(sim[i]))
            predicted_base = _base_id(train_ids[best_train_idx])
            true_base      = _base_id(val_id)
            if predicted_base == true_base:
                hits += 1

        recall1 = hits / len(val_ids)

        # ------------------------------------------------------------------
        # 4. Logger
        # ------------------------------------------------------------------
        if logs is not None:
            logs['val_recall@1'] = recall1

        if self._log_mlflow:
            try:
                import mlflow
                mlflow.log_metric('val_recall@1', recall1, step=epoch)
            except Exception:
                pass