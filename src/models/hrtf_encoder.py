from __future__ import annotations


class HRTFEncoder:
    """Encodeur HRTF : profils spectraux SH → vecteur latent.

    Architecture
    ------------
    [B, n_sh, n_freqs, 2]
        │  Reshape : fusionne freq × oreilles
    [B, n_sh, n_freqs*2]
        │  Dense(32, relu) — mêmes poids appliqués sur les n_sh lignes
    [B, n_sh, 32]
        │  Flatten
    [B, n_sh * 32]
        │  Dense(256, relu) → Dropout → Dense(128) → L2 normalize
    z_hrtf [B, 128]
    """

    def __init__(
        self,
        n_sh:          int   = 121,
        n_freqs:       int   = 129,
        embedding_dim: int   = 128,
        dropout_rate:  float = 0.3,
    ) -> None:
        self.n_sh          = n_sh
        self.n_freqs       = n_freqs
        self.embedding_dim = embedding_dim
        self.dropout_rate  = dropout_rate
        self.model         = self._build()

    def _build(self):
        import tensorflow as tf
        from tensorflow.keras import layers

        inputs = layers.Input(shape=(self.n_sh, self.n_freqs, 2), name='hrtf')

        # Fusionne freq × oreilles → [B, n_sh, n_freqs*2]
        x = layers.Reshape((self.n_sh, self.n_freqs * 2))(inputs)

        # Dense partagé sur les n_sh lignes — [B, n_sh, 32]
        x = layers.Dense(32, activation='relu')(x)

        # Agrège tout → [B, n_sh * 32]
        x = layers.Flatten()(x)

        # MLP classique
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(self.embedding_dim)(x)

        # L2 normalize — requis pour NT-Xent
        z = layers.UnitNormalization(axis=-1, name='embedding')(x)

        return tf.keras.Model(inputs=inputs, outputs=z, name='HRTFEncoder')

    def summary(self) -> None:
        self.model.summary()