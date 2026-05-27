from __future__ import annotations


class EarEncoder:
    """Encodeur d'oreilles : EfficientNet-B0 (poids partagés L/R) + tête MLP.

    Usage
    -----
    enc = EarEncoder(embedding_dim=128)
    enc.set_phase(1)          # backbone gelé, entraîne le head seulement
    enc.model.compile(...)
    enc.model.fit(...)

    enc.set_phase(2)          # dégèle les 20 dernières couches du backbone
    enc.model.compile(...)    # recompiler obligatoire après changement de phase
    enc.model.fit(...)

    Phases
    ------
    1 : backbone entièrement gelé  → entraîne uniquement les ~1.4M params du head
    2 : 20 dernières couches dégelées → fine-tune le dernier bloc EfficientNet
    3 : backbone entier dégelé     → fine-tune global à faible lr

    Architecture
    ------------
    ear_left  [B, 224, 224, 3] ──┐
                                  ├── backbone (shared) ── pool ── concat [B, 2560]
    ear_right [B, 224, 224, 3] ──┘                                    │
                                                        Dense(512) → Dropout
                                                        Dense(256)
                                                        Dense(embedding_dim)
                                                        L2-normalize
                                                        z_ear [B, embedding_dim]
    """

    def __init__(
        self,
        embedding_dim: int   = 128,
        dropout_rate:  float = 0.3,
    ) -> None:
        self.embedding_dim = embedding_dim
        self.dropout_rate  = dropout_rate
        self.model, self._backbone = self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self):
        import tensorflow as tf
        from tensorflow.keras import layers
        from tensorflow.keras.applications import EfficientNetB0

        backbone = EfficientNetB0(
            include_top=False,
            weights='imagenet',
            input_shape=(224, 224, 3),
        )
        backbone.trainable = False  # phase 1 par défaut

        ear_L = layers.Input(shape=(224, 224, 3), name='ear_left')
        ear_R = layers.Input(shape=(224, 224, 3), name='ear_right')

        # EfficientNet attend des pixels en [0, 255] — nos images sont en [0, 1]
        rescale = layers.Rescaling(255.0)

        pool = layers.GlobalAveragePooling2D()

        # Backbone partagé : mêmes poids appliqués sur L et R
        feat_L = pool(backbone(rescale(ear_L)))   # (B, 1280)
        feat_R = pool(backbone(rescale(ear_R)))   # (B, 1280)

        x = layers.Concatenate()([feat_L, feat_R])  # (B, 2560)
        x = layers.Dense(512, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        x = layers.Dense(256, activation='relu')(x)
        x = layers.Dense(self.embedding_dim)(x)

        # Normalisation L2 — requis pour NT-Xent (cosine similarity)
        z = layers.UnitNormalization(axis=-1, name='embedding')(x)

        model = tf.keras.Model(
            inputs={'ear_left': ear_L, 'ear_right': ear_R},
            outputs=z,
            name='EarEncoder',
        )
        return model, backbone

    # ------------------------------------------------------------------
    # Gestion des phases de fine-tuning
    # ------------------------------------------------------------------

    def set_phase(self, phase: int) -> None:
        """Gèle ou dégèle le backbone selon la phase.

        Appeler model.compile() après chaque changement de phase.
        """
        if phase == 1:
            self._backbone.trainable = False

        elif phase == 2:
            self._backbone.trainable = True
            # Gèle tout sauf les 20 dernières couches (dernier bloc MBConv)
            for layer in self._backbone.layers[:-20]:
                layer.trainable = False

        elif phase == 3:
            self._backbone.trainable = True

        else:
            raise ValueError(f'phase doit être 1, 2 ou 3 — reçu {phase}')

        n_trainable = sum(1 for l in self._backbone.layers if l.trainable)
        n_total     = len(self._backbone.layers)
        n_params    = self._count_trainable_params()
        print(
            f'Phase {phase} — backbone : {n_trainable}/{n_total} couches entraînables'
            f'  |  total paramètres entraînables : {n_params:,}'
        )

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def _count_trainable_params(self) -> int:
        import tensorflow as tf
        return int(sum(tf.size(w).numpy() for w in self.model.trainable_weights))

    def summary(self) -> None:
        self.model.summary()