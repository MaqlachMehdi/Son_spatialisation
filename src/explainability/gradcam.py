from __future__ import annotations

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


class GradCAM:
    """Grad-CAM adapté à l'architecture EarEncoder (EfficientNetB0 + tête MLP contrastive).

    Utilisation rapide
    ------------------
    gradcam = GradCAM(ear_encoder)

    # Score A : norme de l'embedding (quels pixels activent le réseau ?)
    heatmap = gradcam.compute(img_L, img_R, ear='left')

    # Score B : similarité avec un HRTF cible (quels pixels prédisent ce profil ?)
    score_fn = GradCAM.score_similarity(hrtf_embedding)
    heatmap  = gradcam.compute(img_L, img_R, ear='left', score_fn=score_fn)

    # Visualisation
    overlaid = GradCAM.overlay(img_L, heatmap)
    plt.imshow(overlaid)

    Stratégie de calcul
    -------------------
    Le backbone EfficientNetB0 est souvent frozen (trainable=False en phase 1).
    Un appel monolithique `tape.gradient(score, feat_maps)` à travers un modèle
    Keras figé retourne des gradients nuls car la tape ne surveille pas les
    variables non-trainable — et l'appel du modèle crée une frontière opaque.

    Solution : on sépare explicitement le forward en deux étapes :
      1. Backbone hors tape  → feat_maps  (la trainabilité n'a pas d'importance)
      2. Tête dans tape      → embedding  (couches appelées directement, pas
                                           via un tf.keras.Model wrapper)
    tape.watch(feat_maps) + appel direct des couches garantit que la chaîne
    feat_maps → pool → concat → MLP → score est entièrement enregistrée.
    """

    def __init__(self, ear_encoder, hrtf_encoder=None) -> None:
        self._ear_enc  = ear_encoder
        self._hrtf_enc = hrtf_encoder
        self._rescale, self._backbone, self._pool, self._concat, self._head_layers = (
            self._extract_layers()
        )

    # ──────────────────────────────────────────────────────────────────────
    # Extraction des références de couches
    # ──────────────────────────────────────────────────────────────────────

    def _extract_layers(self):
        """Récupère les instances de couches depuis l'EarEncoder.

        On travaille directement avec les instances (mêmes poids que le modèle
        entraîné) sans construire de modèle Keras intermédiaire.
        """
        model    = self._ear_enc.model
        backbone = self._ear_enc._backbone

        rescale = next(
            l for l in model.layers if isinstance(l, tf.keras.layers.Rescaling)
        )
        pool = next(
            l for l in model.layers
            if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)
        )
        concat = next(
            l for l in model.layers if isinstance(l, tf.keras.layers.Concatenate)
        )
        head_layers = [
            l for l in model.layers
            if isinstance(l, (
                tf.keras.layers.Dense,
                tf.keras.layers.Dropout,
                tf.keras.layers.UnitNormalization,
            ))
        ]
        return rescale, backbone, pool, concat, head_layers

    # ──────────────────────────────────────────────────────────────────────
    # Calcul de la heatmap
    # ──────────────────────────────────────────────────────────────────────

    def compute(
        self,
        img_L:    np.ndarray,
        img_R:    np.ndarray,
        ear:      str = "left",
        score_fn  = None,
    ) -> np.ndarray:
        """Calcule la heatmap Grad-CAM pour une paire d'images.

        Paramètres
        ----------
        img_L, img_R : images [0, 1], shape (H, W, 3) ou (1, H, W, 3)
        ear          : "left" ou "right"  oreille à expliquer
        score_fn     : callable(embedding) → scalar tensor.
                       Par défaut : score_norm (norme de l'embedding).
                       Utiliser score_similarity(hrtf_emb) pour le score B.

        Retour
        ------
        heatmap : np.ndarray shape (224, 224), valeurs [0, 1]
        """
        if img_L.ndim == 3:
            img_L = img_L[np.newaxis]
        if img_R.ndim == 3:
            img_R = img_R[np.newaxis]

        img_L = tf.cast(img_L, tf.float32)
        img_R = tf.cast(img_R, tf.float32)

        score_fn = score_fn or self.score_norm

        # ── Étape 1 : backbone → feat_maps, hors de la tape ───────────
        # Le backbone peut être frozen (trainable=False) : aucune importance
        # car on ne demande jamais de gradients w.r.t. ses variables.
        fm_L = self._backbone(self._rescale(img_L), training=False)  # (1,7,7,1280)
        fm_R = self._backbone(self._rescale(img_R), training=False)  # (1,7,7,1280)
        fm_target = fm_L if ear == "left" else fm_R

        # ── Étape 2 : tête MLP dans la tape, couches appelées directement ─
        # tape.watch(fm_target) + appel direct (sans wrapper tf.keras.Model)
        # garantit que TF trace la chaîne fm_target → pool → concat → MLP.
        # L'appel via un Model Keras crée une frontière opaque qui bloque
        # le gradient même avec tape.watch → d'où les heatmaps vides.
        with tf.GradientTape() as tape:
            tape.watch(fm_target)

            feat_L = self._pool(fm_L)                   # (1, 1280)
            feat_R = self._pool(fm_R)                   # (1, 1280)
            x      = self._concat([feat_L, feat_R])     # (1, 2560)
            for layer in self._head_layers:
                x = layer(x, training=False)            # Dense / Dropout / UnitNorm
            score = score_fn(x)

        grads = tape.gradient(score, fm_target)         # (1, 7, 7, 1280)

        if grads is None:
            raise RuntimeError(
                "GradientTape a retourné None.\n"
                "Vérifie que score_fn retourne bien un scalaire TF qui dépend de x."
            )

        # ── Grad-CAM : pondération des feature maps ────────────────────
        # α^k = moyenne spatiale des gradients pour le filtre k → (1, 1280)
        alphas = tf.reduce_mean(grads, axis=[1, 2])

        # Somme pondérée des 1280 cartes → (1, 7, 7)
        cam = tf.reduce_sum(
            fm_target * alphas[:, tf.newaxis, tf.newaxis, :],
            axis=-1,
        )
        cam = tf.nn.relu(cam).numpy()[0]  # (7, 7) — ReLU : activations positives seulement

        if cam.max() > 0:
            cam = cam / cam.max()

        return self._resize(cam, size=224)

    # ──────────────────────────────────────────────────────────────────────
    # Fonctions de score
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def score_norm(embedding: tf.Tensor) -> tf.Tensor:
        """Score A : activation moyenne des composantes de l'embedding.

        NOTE : mean(embedding²) = ||embedding||²/dim = 1/dim est une CONSTANTE
        quand le dernier layer est UnitNormalization → gradient nul → heatmap vide.
        On utilise sum(|embedding|) à la place : varie sur la sphère unité,
        gradient non nul, et préserve l'interprétation "quels pixels activent
        le réseau le plus fortement ?".
        """
        return tf.reduce_sum(tf.abs(embedding))

    @staticmethod
    def score_similarity(hrtf_embedding: np.ndarray):
        """Score B : similarité cosinus avec un HRTF cible.

        Retourne une fonction de score pré-configurée avec l'embedding HRTF
        de référence. Les deux embeddings étant L2-normalisés (UnitNormalization),
        la similarité cosinus est équivalente au produit scalaire.

        Usage
        -----
        score_fn = GradCAM.score_similarity(hrtf_encoder.model.predict(hrtf_sample))
        heatmap  = gradcam.compute(img_L, img_R, score_fn=score_fn)
        """
        ref = tf.constant(hrtf_embedding.flatten(), dtype=tf.float32)

        def _score(embedding: tf.Tensor) -> tf.Tensor:
            return tf.reduce_sum(tf.reshape(embedding, [-1]) * ref)

        return _score

    @staticmethod
    def score_dim(dim: int):
        """Score C : activation d'une dimension spécifique de l'embedding.

        Utile pour explorer ce qu'encode chaque direction de l'espace latent.

        Usage
        -----
        score_fn = GradCAM.score_dim(42)
        heatmap  = gradcam.compute(img_L, img_R, score_fn=score_fn)
        """
        def _score(embedding: tf.Tensor) -> tf.Tensor:
            return embedding[0, dim]

        return _score

    # ──────────────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def overlay(
        image:    np.ndarray,
        heatmap:  np.ndarray,
        alpha:    float = 0.45,
        colormap: str   = "jet",
    ) -> np.ndarray:
        """Superpose la heatmap sur l'image originale.

        Paramètres
        ----------
        image   : (H, W, 3), valeurs [0, 1]
        heatmap : (H, W),    valeurs [0, 1]
        alpha   : intensité de la heatmap (0 = invisible, 1 = opaque)

        Retour
        ------
        (H, W, 3), valeurs [0, 1]
        """
        cmap        = plt.get_cmap(colormap)
        heatmap_rgb = cmap(heatmap)[:, :, :3]
        overlaid    = alpha * heatmap_rgb + (1 - alpha) * image
        return np.clip(overlaid, 0.0, 1.0)

    # ──────────────────────────────────────────────────────────────────────
    # Utilitaires internes
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _resize(arr: np.ndarray, size: int) -> np.ndarray:
        """Resize 2D array (H, W) vers (size, size) par interpolation bilinéaire."""
        t = tf.image.resize(
            arr[np.newaxis, :, :, np.newaxis],
            [size, size],
            method="bilinear",
        )
        return t.numpy()[0, :, :, 0]