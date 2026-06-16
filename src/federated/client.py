from __future__ import annotations

import os

# (segfault/abort) sous Windows en simulation Ray.
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

import flwr as fl

from src.models import EarEncoder, HRTFEncoder, ContrastiveModel
from src.multimodal import MultimodalDataset


class HRTFFlowerClient(fl.client.NumPyClient):
    """Client Flower pour l'entraînement fédéré du ContrastiveModel HRTF.

    Chaque instance représente un PC avec son propre dataset local.
    Flower gère la communication — cette classe gère uniquement la logique métier.
    """

    def __init__(
        self,
        model,
        ear_encoder,
        dataset:   MultimodalDataset,
        client_id: str,
    ) -> None:
        super().__init__()
        self._model     = model
        self._enc_ear   = ear_encoder
        self._dataset   = dataset
        self._client_id = client_id

    # ------------------------------------------------------------------
    # Interface Flower
    # ------------------------------------------------------------------

    def get_parameters(self, config):
        """Renvoie les poids actuels du modèle au serveur.

        Flower appelle cette méthode avec un argument nommé (config=...) —
        le paramètre doit s'appeler exactement `config`, même si inutilisé ici.
        """
        return self._model.get_weights()

    def fit(self, parameters, config):
        """Reçoit les poids globaux, entraîne localement, renvoie les nouveaux poids.

        Parameters
        ----------
        parameters : poids globaux envoyés par le serveur (liste de np.ndarray)
        config     : dict du serveur {"phase": 1, "lr": 1e-3, "local_epochs": 5}

        Returns
        -------
        (nouveaux_poids, n_samples, métriques)
        n_samples sert à FedAvg pour pondérer la moyenne.
        """
        self._model.set_weights(parameters)

        phase        = config.get("phase", 1)
        local_epochs = config.get("local_epochs", 5)
        lr           = config.get("lr", 1e-3)

        self._enc_ear.set_phase(phase)
        self._model.compile(optimizer=tf.keras.optimizers.Adam(float(lr)))

        history = self._model.fit(
            self._dataset.train,
            validation_data = self._dataset.val,
            epochs          = local_epochs,
            callbacks       = [
                tf.keras.callbacks.EarlyStopping(
                    monitor              = "val_loss",
                    patience             = 3,
                    restore_best_weights = True,
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor  = "val_loss",
                    factor   = 0.5,
                    patience = 2,
                    min_lr   = 1e-7,
                ),
            ],
            verbose = 0,
        )

        loss     = float(history.history["loss"][-1])
        val_loss = float(history.history["val_loss"][-1])

        return (
            self._model.get_weights(),
            len(self._dataset.train_idx),
            {"loss": loss, "val_loss": val_loss},
        )

    def evaluate(self, parameters, config):
        """Évalue les poids globaux sur le val set local.

        Permet au serveur de suivre la val_loss par client sans partager les données.

        """
        self._model.set_weights(parameters)
        self._model.compile(optimizer=tf.keras.optimizers.Adam())
        results = self._model.evaluate(
            self._dataset.val,
            return_dict = True,
            verbose     = 0,
        )
        loss = float(results.get("loss", 0.0))
        return loss, len(self._dataset.val_idx), {"loss": loss}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _build_model() -> tuple:
    """Construit ContrastiveModel et effectue le dummy pass obligatoire."""
    enc_ear  = EarEncoder(embedding_dim=128)
    enc_hrtf = HRTFEncoder(n_sh=121, n_freqs=129, embedding_dim=128)
    model    = ContrastiveModel(enc_ear, enc_hrtf, temperature=0.07, supervised=True)

    dummy = {
        "ear_left":  np.zeros((1, 224, 224, 3), dtype="float32"),
        "ear_right": np.zeros((1, 224, 224, 3), dtype="float32"),
        "hrtf":      np.zeros((1, 121, 129, 2), dtype="float32"),
    }
    model(dummy, training=False)
    return model, enc_ear


def client_fn(partitions: list[MultimodalDataset]):
    """Closure factory compatible avec fl.simulation.start_simulation().

    Usage
    -----
    fl.simulation.start_simulation(
        client_fn = client_fn(partitions),
        num_clients = len(partitions),
        ...
    )

    Note : depuis Flower 1.x, client_fn(context) reçoit un objet Context —
    plus un simple client_id str. L'identifiant de partition se lit dans
    context.node_config["partition-id"].
    """
    def _fn(context) -> fl.client.Client:
        partition_id = int(context.node_config["partition-id"])
        model, enc_ear = _build_model()
        dataset = partitions[partition_id]
        client  = HRTFFlowerClient(model, enc_ear, dataset, str(partition_id))
        return client.to_client()

    return _fn