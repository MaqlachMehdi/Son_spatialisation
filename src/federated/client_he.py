from __future__ import annotations

import flwr as fl

from src.multimodal import MultimodalDataset
from src.federated.client import HRTFFlowerClient, _build_model
from src.federated.he_context import encrypt_weights


class HRTFFlowerClientHE(HRTFFlowerClient):
    """Client FL avec chiffrement homomorphique des poids sortants.

    Flux par round
    --------------
    1. Serveur → client : poids globaux en CLAIR (le serveur les déchiffre
       avant envoi voir FedAvgHE.aggregate_fit)
    2. Client set_weights() + entraînement local : identique au client standard
    3. Client → serveur : poids w_i CHIFFRÉS avec la clé publique HE

    Le contexte passé ici est PUBLIC (sans clé privée)  le client peut
    chiffrer mais jamais déchiffrer les contributions des autres clients.
    """

    def __init__(
        self,
        model,
        ear_encoder,
        dataset: MultimodalDataset,
        client_id: str,
        he_context,
    ) -> None:
        super().__init__(model, ear_encoder, dataset, client_id)
        self._he_ctx = he_context

    def get_parameters(self, config):
        """Retourne les poids en clair identique au client standard.

        Les paramètres initiaux doivent être en clair : le serveur les stocke
        puis les redistribue aux clients dans fit(). Seul fit() retourne des
        poids chiffrés.
        """
        return self._model.get_weights()

    def fit(self, parameters, config):
        """Entraînement standard, retour des poids chiffrés.

        `parameters` reçu du serveur = poids en CLAIR (le serveur a déchiffré
        enc_avg avant de renvoyer, set_weights fonctionne normalement).
        Seul le retour est chiffré.
        """
        weights, n_samples, metrics = super().fit(parameters, config)
        encrypted_weights = encrypt_weights(self._he_ctx, weights)
        return encrypted_weights, n_samples, metrics

    # evaluate() héritée sans modification :
    # - reçoit des poids en CLAIR → set_weights() fonctionne
    # - retourne une loss scalaire → pas de chiffrement nécessaire


# ──────────────────────────────────────────────────────────────────────────────
# Factory de clients HE pour fl.simulation.start_simulation()
# ──────────────────────────────────────────────────────────────────────────────

def client_fn_he(
    partitions: list[MultimodalDataset],
    public_he_context,
):
    """Closure compatible avec fl.simulation.start_simulation(client_fn=...).

    Identique à client_fn() mais instancie HRTFFlowerClientHE
    avec le contexte HE public passé en argument.
    """
    # TenSEALContext n'est pas picklable par Ray. On sérialise en bytes
    # (picklable) avant que la closure le capture, et on désérialise
    # à l'intérieur du worker Ray.
    import tenseal as ts
    he_ctx_bytes = public_he_context.serialize()

    def _fn(context) -> fl.client.Client:
        local_he_ctx = ts.context_from(he_ctx_bytes)
        partition_id = int(context.node_config["partition-id"])
        model, enc_ear = _build_model()
        dataset = partitions[partition_id]
        client = HRTFFlowerClientHE(
            model, enc_ear, dataset, str(partition_id), local_he_ctx
        )
        return client.to_client()
    return _fn