from __future__ import annotations

import tensorflow as tf

from .nt_xent import nt_xent_loss


class ContrastiveModel(tf.keras.Model):
    """Modèle contrastif cross-modal : EarEncoder + HRTFEncoder alignés via NT-Xent.

    Usage
    -----
    enc_ear  = EarEncoder(embedding_dim=128)
    enc_hrtf = HRTFEncoder(n_sh=121, n_freqs=129, embedding_dim=128)

    model = ContrastiveModel(enc_ear, enc_hrtf, temperature=0.07)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    history = model.fit(ds.train, validation_data=ds.val, epochs=50)
    """

    def __init__(
        self,
        ear_encoder,
        hrtf_encoder,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.ear_model   = ear_encoder.model
        self.hrtf_model  = hrtf_encoder.model
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def call(self, inputs, training: bool = False):
        z_ear = self.ear_model(
            {'ear_left': inputs['ear_left'], 'ear_right': inputs['ear_right']},
            training=training,
        )
        z_hrtf = self.hrtf_model(inputs['hrtf'], training=training)
        return z_ear, z_hrtf

    # ------------------------------------------------------------------
    # Boucle d'entraînement custom
    # ------------------------------------------------------------------

    def train_step(self, data):
        inputs, _ = data

        with tf.GradientTape() as tape:
            z_ear, z_hrtf = self(inputs, training=True)
            loss = nt_xent_loss(z_ear, z_hrtf, self.temperature)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        return {'loss': loss}

    def test_step(self, data):
        inputs, _ = data
        z_ear, z_hrtf = self(inputs, training=False)
        loss = nt_xent_loss(z_ear, z_hrtf, self.temperature)
        return {'val_loss': loss}