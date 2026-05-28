from __future__ import annotations


def nt_xent_loss(z_ear, z_hrtf, temperature: float = 0.07):
    """NT-Xent loss cross-modal : aligne z_ear et z_hrtf du même sujet.

    Parameters
    ----------
    z_ear         : [N, D]  embeddings oreilles,  L2-normalisés
    z_hrtf        : [N, D]  embeddings HRTF,       L2-normalisés
    temperature   : float   τ — contrôle la netteté de la distribution

    Returns
    -------
    loss : scalar  moyenne des 2N termes de cross-entropy
    """
    import tensorflow as tf

    N = tf.shape(z_ear)[0]

    # Matrice de similarité cosinus [N, N]
    # Les vecteurs étant L2-normalisés, cos(u,v) = u · v
    sim = tf.matmul(z_ear, z_hrtf, transpose_b=True) / temperature

    # Labels : la paire correcte (i, i) est sur la diagonale
    labels = tf.range(N)

    # Loss dans les deux sens
    loss_ear  = tf.keras.losses.sparse_categorical_crossentropy(
        labels, sim, from_logits=True
    )
    loss_hrtf = tf.keras.losses.sparse_categorical_crossentropy(
        labels, tf.transpose(sim), from_logits=True
    )

    return (tf.reduce_mean(loss_ear) + tf.reduce_mean(loss_hrtf)) / 2.0
