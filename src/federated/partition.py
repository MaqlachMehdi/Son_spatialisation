from __future__ import annotations

import numpy as np

from src.multimodal import MultimodalDataset


def partition_dataset(
    dataset:   MultimodalDataset,
    n_clients: int,
    seed:      int = 42,
    val_size:  float = 0.15,
    test_size: float = 0.15,
) -> list[MultimodalDataset]:
    """Partitionne le dataset en n_clients sous-datasets balancés.

    La partition se fait par sujet de BASE (H10, H11...) pour que H10,
    H10_mirror et H10_aug1 restent toujours dans le même client.
    Chaque client recalcule son propre split train/val/test local.

    Parameters
    ----------
    dataset   : dataset complet chargé depuis le .npz
    n_clients : nombre de partitions à créer
    seed      : reproductibilité
    val_size  : fraction de sujets de base réservés à la validation locale
    test_size : fraction réservée au test local

    Returns
    -------
    list[MultimodalDataset] un dataset par client
    """
    all_ids   = dataset.subject_ids
    base_ids  = sorted({sid.split("_")[0] for sid in all_ids})
    n_bases   = len(base_ids)

    min_per_client = max(4, round(1 / (val_size + test_size)) + 2)
    if n_bases < n_clients * min_per_client:
        raise ValueError(
            f"{n_bases} sujets de base insuffisants pour {n_clients} clients "
            f"(minimum {n_clients * min_per_client}). "
            f"Réduis n_clients ou augmente le dataset."
        )

    rng      = np.random.default_rng(seed)
    shuffled = np.array(base_ids, dtype=object)
    rng.shuffle(shuffled)
    groups   = np.array_split(shuffled, n_clients)

    partitions = []
    for i, group_bases in enumerate(groups):
        group_set  = set(group_bases.tolist())
        group_list = list(group_bases)
        n          = len(group_list)

        # Split par sujet de base pour éviter le leakage inter-split
        rng_i  = np.random.default_rng(seed + i)
        perm   = rng_i.permutation(n)
        n_val  = max(1, round(n * val_size))
        n_test = max(1, round(n * test_size))

        train_bases = {group_list[j] for j in perm[:n - n_val - n_test]}
        val_bases   = {group_list[j] for j in perm[n - n_val - n_test: n - n_test]}
        test_bases  = {group_list[j] for j in perm[n - n_test:]}

        def _idx(bases):
            return np.array([
                j for j, sid in enumerate(all_ids)
                if sid.split("_")[0] in bases
            ])

        train_idx = _idx(train_bases)
        val_idx   = _idx(val_bases)
        test_idx  = _idx(test_bases)

        partitions.append(MultimodalDataset(
            hrtf_norm   = dataset._hrtf,
            img_L       = dataset._img_L,
            img_R       = dataset._img_R,
            subject_ids = dataset.subject_ids,
            train_idx   = train_idx,
            val_idx     = val_idx,
            test_idx    = test_idx,
            train_mean  = dataset.train_mean,
            train_std   = dataset.train_std,
            batch_size  = dataset.batch_size,
            sh_order    = dataset.sh_order,
        ))

        print(
            f"  Client {i} : {len(train_bases)} sujets train "
            f"/ {len(val_bases)} val / {len(test_bases)} test "
            f"({len(train_idx) + len(val_idx) + len(test_idx)} samples total)"
        )

    return partitions