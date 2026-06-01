from __future__ import annotations


def supervised_nt_xent_loss(
    z_ear,
    z_hrtf,
    labels,
    temperature: float = 0.07,
):
    """Supervised Contrastive Loss cross-modal.

    Contrairement à NT-Xent standard, les versions augmentées du même
    sujet (H10_mirror, H10_aug1) sont traitées comme des POSITIFS
    supplémentaires — pas comme des négatifs.

    Pour chaque ancre z_ear[i], les positifs sont tous les z_hrtf[j]
    dont le label correspond au même sujet de base.

    Formule (Khosla et al., 2020 — adapté cross-modal)
    ---------------------------------------------------
    L_i = -1/|P(i)| · Σ_{p∈P(i)} log( exp(sim(i,p)/τ) / Σ_k exp(sim(i,k)/τ) )

    Où P(i) = {j : labels[j] == labels[i]}

    Parameters
    ----------
    z_ear     : [N, D]  embeddings oreilles,  L2-normalisés
    z_hrtf    : [N, D]  embeddings HRTF,       L2-normalisés
    labels    : [N]     entiers — même label = même sujet de base
    temperature : float  τ

    Returns
    -------
    loss : scalar
    """
    import tensorflow as tf

    labels = tf.cast(labels, tf.int32)

    # Masque positif [N, N] : True là où labels[i] == labels[j]
    pos_mask = tf.cast(
        tf.equal(
            tf.expand_dims(labels, 1),   # [N, 1]
            tf.expand_dims(labels, 0),   # [1, N]
        ),
        tf.float32,
    )  # [N, N]

    n_positives = tf.reduce_sum(pos_mask, axis=1)   # [N]

    # Similarité cosinus [N, N] — vecteurs déjà L2-normalisés
    sim = tf.matmul(z_ear, z_hrtf, transpose_b=True) / temperature  # [N, N]

    # log( Σ_k exp(sim[i,k]) )  pour chaque ancre i
    log_denom = tf.reduce_logsumexp(sim, axis=1, keepdims=True)   # [N, 1]

    # Pour chaque ancre : moyenne des log-probabilités des positifs
    # L_i = -mean_{p∈P(i)} (sim[i,p] - log_denom[i])
    loss_per_anchor = -tf.reduce_sum(
        pos_mask * (sim - log_denom), axis=1
    ) / (n_positives + 1e-8)   # [N]

    # Symétrique : hrtf → ear
    sim_T        = tf.transpose(sim)
    log_denom_T  = tf.reduce_logsumexp(sim_T, axis=1, keepdims=True)
    loss_per_anchor_T = -tf.reduce_sum(
        pos_mask * (sim_T - log_denom_T), axis=1
    ) / (n_positives + 1e-8)

    return (tf.reduce_mean(loss_per_anchor) + tf.reduce_mean(loss_per_anchor_T)) / 2.0