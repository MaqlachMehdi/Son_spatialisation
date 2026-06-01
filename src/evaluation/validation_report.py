from __future__ import annotations

import numpy as np

from .lsd import log_spectral_distance
from .metrics import recall_at_k, mrr


def _base_id(subject_id: str) -> str:
    return subject_id.split('_')[0]


class ValidationReport:
    """Rapport complet après chaque phase d'entraînement.

    Métriques calculées
    -------------------
    Recall@1, @3       : retrieval le bon sujet est-il retrouvé ?
    MRR                : rang moyen de la bonne paire
    Similarity S+      : cosine similarity moyenne des paires correctes
    Similarity S-      : cosine similarity moyenne des paires incorrectes
    Ratio S+/S-        : discrimination — > 1.5 est un bon signal
    LSD (dB)           : distorsion spectrale de la HRTF récupérée

    Note sur le data leakage :
    Le split est fait après augmentation. Des versions augmentées de sujets
    du val set existent dans le train set. Les métriques sont donc légèrement
    optimistes. Le critère "correct = même base_id" atténue cet effet mais
    ne l'élimine pas. Pour une évaluation rigoureuse, utiliser le test set
    et faire le split avant augmentation.

    Usage
    -----
    report = ValidationReport(model, dataset)
    results = report.run(phase=1, log_mlflow=True)
    """

    def __init__(self, model, dataset, batch_size: int = 16) -> None:
        self._model      = model
        self._dataset    = dataset
        self._batch_size = batch_size

    def run(self, phase: int = 0, log_mlflow: bool = True) -> dict:
        import tensorflow as tf

        # ------------------------------------------------------------------
        # 1. Gallery depuis le TRAIN set uniquement
        # ------------------------------------------------------------------
        train_idx  = self._dataset.train_idx
        hrtf_norm  = self._dataset._hrtf[train_idx].astype(np.float32)
        hrtf_raw   = (hrtf_norm * (self._dataset.train_std + 1e-8)
                      + self._dataset.train_mean)
        train_ids  = [self._dataset.subject_ids[i] for i in train_idx]

        z_hrtf_train = self._encode_hrtf(hrtf_norm)

        # ------------------------------------------------------------------
        # 2. Encoder le VAL set
        # ------------------------------------------------------------------
        val_idx = self._dataset.val_idx
        ear_L   = self._dataset._img_L[val_idx].astype(np.float32)
        ear_R   = self._dataset._img_R[val_idx].astype(np.float32)
        hrtf_val_norm = self._dataset._hrtf[val_idx].astype(np.float32)
        hrtf_val_raw  = (hrtf_val_norm * (self._dataset.train_std + 1e-8)
                         + self._dataset.train_mean)
        val_ids = [self._dataset.subject_ids[i] for i in val_idx]

        z_ear_val = self._encode_ear(ear_L, ear_R)

        # ------------------------------------------------------------------
        # 3. Matrice de similarité [N_val, N_train]
        # ------------------------------------------------------------------
        sim = z_ear_val @ z_hrtf_train.T

        # ------------------------------------------------------------------
        # 4. Recall@1 et @3 avec critère base_id
        # ------------------------------------------------------------------
        recall1, recall3 = self._recall_base_id(sim, val_ids, train_ids, ks=[1, 3])
        mrr_score        = self._mrr_base_id(sim, val_ids, train_ids)

        # ------------------------------------------------------------------
        # 5. Similarity S+ et S-
        # ------------------------------------------------------------------
        s_pos, s_neg = self._similarity_ratio(sim, val_ids, train_ids)
        ratio = s_pos / (s_neg + 1e-8)

        # ------------------------------------------------------------------
        # 6. LSD — HRTF prédite vs vraie HRTF du val sujet
        # ------------------------------------------------------------------
        lsd_scores = []
        for i, val_id in enumerate(val_ids):
            best_train = int(np.argmax(sim[i]))
            lsd_val    = log_spectral_distance(
                hrtf_raw[best_train],
                hrtf_val_raw[i],
            )
            lsd_scores.append(lsd_val)

        lsd_mean = float(np.mean(lsd_scores))
        lsd_std  = float(np.std(lsd_scores))

        # ------------------------------------------------------------------
        # 7. Affichage
        # ------------------------------------------------------------------
        results = {
            'phase'      : phase,
            'recall@1'   : recall1,
            'recall@3'   : recall3,
            'mrr'        : mrr_score,
            'sim_pos'    : s_pos,
            'sim_neg'    : s_neg,
            'ratio_s'    : ratio,
            'lsd_mean'   : lsd_mean,
            'lsd_std'    : lsd_std,
        }

        self._print(results, len(val_ids))

        if log_mlflow:
            self._log(results, phase)

        return results

    # ------------------------------------------------------------------
    # Métriques avec critère base_id
    # ------------------------------------------------------------------

    def _recall_base_id(
        self,
        sim:       np.ndarray,
        val_ids:   list[str],
        train_ids: list[str],
        ks:        list[int],
    ) -> tuple[float, ...]:
        recalls = []
        for k in ks:
            hits = 0
            for i, val_id in enumerate(val_ids):
                top_k = np.argsort(sim[i])[-k:]
                if any(_base_id(train_ids[j]) == _base_id(val_id)
                       for j in top_k):
                    hits += 1
            recalls.append(hits / len(val_ids))
        return tuple(recalls)

    def _mrr_base_id(
        self,
        sim:       np.ndarray,
        val_ids:   list[str],
        train_ids: list[str],
    ) -> float:
        total = 0.0
        for i, val_id in enumerate(val_ids):
            sorted_idx = np.argsort(sim[i])[::-1]
            for rank, j in enumerate(sorted_idx, start=1):
                if _base_id(train_ids[j]) == _base_id(val_id):
                    total += 1.0 / rank
                    break
        return total / len(val_ids)

    def _similarity_ratio(
        self,
        sim:       np.ndarray,
        val_ids:   list[str],
        train_ids: list[str],
    ) -> tuple[float, float]:
        pos_scores, neg_scores = [], []
        for i, val_id in enumerate(val_ids):
            base = _base_id(val_id)
            for j, tid in enumerate(train_ids):
                if _base_id(tid) == base:
                    pos_scores.append(sim[i, j])
                else:
                    neg_scores.append(sim[i, j])
        return float(np.mean(pos_scores)), float(np.mean(neg_scores))

    # ------------------------------------------------------------------
    # Encodage
    # ------------------------------------------------------------------

    def _encode_hrtf(self, hrtf_norm: np.ndarray) -> np.ndarray:
        import tensorflow as tf
        z_list = []
        for start in range(0, len(hrtf_norm), self._batch_size):
            z = self._model.hrtf_model(
                tf.constant(hrtf_norm[start:start + self._batch_size]),
                training=False,
            )
            z_list.extend(z.numpy())
        return np.array(z_list, dtype=np.float32)

    def _encode_ear(
        self, ear_L: np.ndarray, ear_R: np.ndarray
    ) -> np.ndarray:
        import tensorflow as tf
        z_list = []
        for start in range(0, len(ear_L), self._batch_size):
            batch = {
                'ear_left':  tf.constant(ear_L[start:start + self._batch_size]),
                'ear_right': tf.constant(ear_R[start:start + self._batch_size]),
            }
            z = self._model.ear_model(batch, training=False)
            z_list.extend(z.numpy())
        return np.array(z_list, dtype=np.float32)

    # ------------------------------------------------------------------
    # Affichage et logging
    # ------------------------------------------------------------------

    def _print(self, r: dict, n_val: int) -> None:
        print(f'\n  {"─"*52}')
        print(f'  RAPPORT DE VALIDATION — Phase {r["phase"]}'
              f'  ({n_val} sujets val)')
        print(f'  {"─"*52}')
        print(f'  Recall@1        : {r["recall@1"]:.3f}  '
              f'({int(r["recall@1"]*n_val)}/{n_val})')
        print(f'  Recall@3        : {r["recall@3"]:.3f}')
        print(f'  MRR             : {r["mrr"]:.3f}')
        print(f'  {"─"*52}')
        print(f'  Sim. correctes  : {r["sim_pos"]:.3f}')
        print(f'  Sim. incorrectes: {r["sim_neg"]:.3f}')
        print(f'  Ratio S+/S-     : {r["ratio_s"]:.2f}'
              f'  {"✓" if r["ratio_s"] > 1.5 else "⚠ modèle peu discriminant"}')
        print(f'  {"─"*52}')
        print(f'  LSD moyen       : {r["lsd_mean"]:.2f} dB ± {r["lsd_std"]:.2f}'
              f'  {"✓ < 5 dB" if r["lsd_mean"] < 5 else "⚠ > 5 dB"}')
        print(f'  {"─"*52}\n')

    def _log(self, r: dict, phase: int) -> None:
        try:
            import mlflow
            mlflow.log_metrics({
                f'phase{phase}_recall@1': r['recall@1'],
                f'phase{phase}_recall@3': r['recall@3'],
                f'phase{phase}_mrr'     : r['mrr'],
                f'phase{phase}_ratio_s' : r['ratio_s'],
                f'phase{phase}_lsd_mean': r['lsd_mean'],
            })
        except Exception:
            pass