from __future__ import annotations

import numpy as np


def recall_at_k(z_ear: np.ndarray, z_hrtf: np.ndarray, k: int = 1) -> float:
    """Recall@K : proportion de sujets dont le bon z_hrtf est dans le top-K.

    Pour chaque z_ear[i], cherche parmi tous les z_hrtf le plus similaire.
    Correct si z_hrtf[i] figure dans les K premiers.
    """
    sim  = z_ear @ z_hrtf.T   # [N, N] — cosine similarity (vecteurs L2-normalisés)
    N    = len(z_ear)
    hits = 0
    for i in range(N):
        top_k = np.argsort(sim[i])[-k:]
        if i in top_k:
            hits += 1
    return hits / N


def mrr(z_ear: np.ndarray, z_hrtf: np.ndarray) -> float:
    """Mean Reciprocal Rank : rang moyen de la bonne paire (1/rang)."""
    sim   = z_ear @ z_hrtf.T
    N     = len(z_ear)
    total = 0.0
    for i in range(N):
        sorted_idx = np.argsort(sim[i])[::-1]          # ordre décroissant
        rank       = int(np.where(sorted_idx == i)[0][0]) + 1  # 1-indexé
        total     += 1.0 / rank
    return total / N


def evaluate(space, split: str = 'test') -> dict:
    """Rapport complet sur un split donné.

    Returns
    -------
    dict avec keys : recall@1, recall@3, recall@5, mrr
    """
    sub = space.filter(split)

    if len(sub.subject_ids) == 0:
        print(f'  Aucun sujet dans le split "{split}".')
        return {}

    r1 = recall_at_k(sub.z_ear, sub.z_hrtf, k=1)
    r3 = recall_at_k(sub.z_ear, sub.z_hrtf, k=3)
    r5 = recall_at_k(sub.z_ear, sub.z_hrtf, k=5)
    m  = mrr(sub.z_ear, sub.z_hrtf)

    results = {'recall@1': r1, 'recall@3': r3, 'recall@5': r5, 'mrr': m}

    print(f'\n  Évaluation — split : {split}  ({len(sub.subject_ids)} sujets)')
    print(f'  {"─"*40}')
    print(f'  Recall@1  : {r1:.3f}   ({int(r1 * len(sub.subject_ids))}/{len(sub.subject_ids)} sujets)')
    print(f'  Recall@3  : {r3:.3f}')
    print(f'  Recall@5  : {r5:.3f}')
    print(f'  MRR       : {m:.3f}')

    return results