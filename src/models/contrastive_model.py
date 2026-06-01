from __future__ import annotations

import tensorflow as tf

from .nt_xent import nt_xent_loss
from .supervised_nt_xent import supervised_nt_xent_loss


class ContrastiveModel(tf.keras.Model):
    """Modèle contrastif cross-modal : EarEncoder + HRTFEncoder.

    Deux modes de loss :
        supervised=False  → NT-Xent standard (toutes augmentations = négatifs)
        supervised=True   → Supervised NT-Xent (augmentations = positifs)
                            Nécessite que ds._make_ds() passe des base_labels.

    Usage
    -----
    # Mode standard
    model = ContrastiveModel(enc_ear, enc_hrtf, temperature=0.07)

    # Mode supervisé (recommandé si dataset augmenté)
    model = ContrastiveModel(enc_ear, enc_hrtf, temperature=0.07, supervised=True)

    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    model.fit(ds.train, validation_data=ds.val, epochs=50)
    """

    def __init__(
        self,
        ear_encoder,
        hrtf_encoder,
        temperature: float = 0.07,
        supervised:  bool  = True,
    ) -> None:
        super().__init__()
        self.ear_model    = ear_encoder.model
        self.hrtf_model   = hrtf_encoder.model
        self.temperature  = temperature
        self.supervised   = supervised
        self.loss_tracker = tf.keras.metrics.Mean(name='loss')

    # ------------------------------------------------------------------
    # Métriques
    # ------------------------------------------------------------------

    @property
    def metrics(self):
        return [self.loss_tracker]

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
    # Calcul de la loss selon le mode
    # ------------------------------------------------------------------

    def _compute_loss(self, z_ear, z_hrtf, labels) -> tf.Tensor:
        if self.supervised:
            return supervised_nt_xent_loss(
                z_ear, z_hrtf, labels, self.temperature
            )
        return nt_xent_loss(z_ear, z_hrtf, self.temperature)

    # ------------------------------------------------------------------
    # Boucle d'entraînement
    # ------------------------------------------------------------------

    def train_step(self, data):
        inputs, labels = data

        with tf.GradientTape() as tape:
            z_ear, z_hrtf = self(inputs, training=True)
            loss = self._compute_loss(z_ear, z_hrtf, labels)

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        inputs, labels = data
        z_ear, z_hrtf = self(inputs, training=False)
        loss = self._compute_loss(z_ear, z_hrtf, labels)

        self.loss_tracker.update_state(loss)
        return {m.name: m.result() for m in self.metrics}