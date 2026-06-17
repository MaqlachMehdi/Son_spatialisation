from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import flwr as fl


# ──────────────────────────────────────────────────────────────────────────────
# Sauvegarde des poids globaux
# ──────────────────────────────────────────────────────────────────────────────

class FedAvgWithCheckpoint(fl.server.strategy.FedAvg):
    """FedAvg avec sauvegarde automatique du modèle global après chaque round.

    Sauvegarde :
      - checkpoints/round_001.npz, round_002.npz … (un par round)
      - checkpoints/best_weights.npz               (meilleur val_loss)

    Usage
    -----
    strategy = FedAvgWithCheckpoint(checkpoint_dir="checkpoints/fl", **kwargs)
    weights  = load_weights_from_npz("checkpoints/fl/best_weights.npz")
    model.set_weights(weights)
    """

    def __init__(self, checkpoint_dir: str = "checkpoints/federated", **kwargs) -> None:
        super().__init__(**kwargs)
        self._ckpt_dir = Path(checkpoint_dir)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._best_loss = float("inf")

    def aggregate_fit(self, server_round: int, results, failures):
        aggregated_params, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_params is None:
            return aggregated_params, metrics

        ndarrays = fl.common.parameters_to_ndarrays(aggregated_params)
        self._save(ndarrays, f"round_{server_round:03d}.npz")

        loss = float(metrics.get("val_loss", metrics.get("loss", float("inf")))) if metrics else float("inf")
        if loss < self._best_loss:
            self._best_loss = loss
            self._save(ndarrays, "best_weights.npz")
            print(f"  [Round {server_round:2d}] Meilleur modèle → val_loss={loss:.4f}")

        return aggregated_params, metrics

    def _save(self, ndarrays: list[np.ndarray], filename: str) -> None:
        # Clés à 4 chiffres pour que le tri alphabétique = tri par index
        path = self._ckpt_dir / filename
        np.savez(str(path), **{f"arr_{i:04d}": arr for i, arr in enumerate(ndarrays)})

    @property
    def best_checkpoint_path(self) -> Path:
        return self._ckpt_dir / "best_weights.npz"


def load_weights_from_npz(path: str | Path) -> list[np.ndarray]:
    """Charge des poids depuis un .npz sauvegardé par FedAvgWithCheckpoint.

    Retourne une liste de np.ndarray directement utilisable avec model.set_weights().
    """
    with np.load(str(path)) as data:
        return [data[k] for k in sorted(data.files)]


# ──────────────────────────────────────────────────────────────────────────────
# Multi-Krum : agrégation robuste aux clients Byzantine
# ──────────────────────────────────────────────────────────────────────────────

def _multikrum_select(
    weights_list: list[list[np.ndarray]],
    num_examples: list[int],
    f: int,
) -> tuple[list[list[np.ndarray]], list[int]]:
    """Sélectionne les m = n - f clients avec le meilleur score Krum.

    Score Krum du client i = somme de ses distances L²² aux (n - f - 2) plus
    proches voisins. Un score bas = client central = probablement honnête.

    Paramètres
    ----------
    weights_list : poids de chaque client (liste de np.ndarray par client)
    num_examples : nombre d'exemples de chaque client (pour la moyenne pondérée)
    f            : nombre supposé de clients malveillants

    Conditions de validité
    ----------------------
    n_clients > 2 * f + 2  (sinon Krum ne peut pas isoler les attaquants)
    """
    n = len(weights_list)
    m = max(1, n - f)       # nb clients à retenir pour la moyenne
    k = max(0, n - f - 2)   # nb voisins utilisés pour scorer chaque client

    # Aplatir chaque vecteur de poids en un unique vecteur 1D
    flat = [
        np.concatenate([w.flatten() for w in client_weights])
        for client_weights in weights_list
    ]

    # Matrice de distances L²² (symétrique — calcul du triangle supérieur)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.sum((flat[i] - flat[j]) ** 2))
            dist[i, j] = d
            dist[j, i] = d

    # Score Krum de chaque client
    scores = np.zeros(n)
    for i in range(n):
        row = dist[i].copy()
        row[i] = np.inf     # s'exclut lui-même du calcul
        if k > 0:
            scores[i] = np.sum(np.sort(row)[:k])
        # Si k=0 (ex: n=2, f=0) : scores tous nuls → on retient tous les clients (= FedAvg)

    selected_idx = list(np.argsort(scores)[:m])
    return [weights_list[i] for i in selected_idx], [num_examples[i] for i in selected_idx]


class MultiKrum(fl.server.strategy.FedAvg):
    """Remplace la moyenne FedAvg par l'agrégation Multi-Krum.

    Résistant aux attaques Byzantine : jusqu'à f clients malveillants peuvent
    envoyer des poids arbitraires sans corrompre le modèle global, car
    Multi-Krum les exclut automatiquement avant de faire la moyenne.

    Paramètres
    ----------
    num_malicious : nb de clients malveillants supposés (f dans la littérature).
                    Doit satisfaire n_clients > 2 * f + 2.

    Notes
    -----
    Les métriques (loss, val_loss) sont agrégées sur TOUS les clients — y compris
    les exclus — pour ne pas biaiser le suivi de la courbe d'entraînement.
    """

    def __init__(self, num_malicious: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self._f = num_malicious

    def aggregate_fit(self, server_round: int, results, failures):
        if not results:
            return None, {}

        weights_results = [
            (fl.common.parameters_to_ndarrays(fit_res.parameters), fit_res.num_examples)
            for _, fit_res in results
        ]
        weights_list = [w for w, _ in weights_results]
        num_examples = [n for _, n in weights_results]

        selected_w, selected_n = _multikrum_select(weights_list, num_examples, self._f)

        # Moyenne pondérée par n_samples des clients sélectionnés uniquement
        total = sum(selected_n)
        n_layers = len(selected_w[0])
        aggregated = [
            sum(w[layer] * n for w, n in zip(selected_w, selected_n)) / total
            for layer in range(n_layers)
        ]

        parameters = fl.common.ndarrays_to_parameters(aggregated)

        # Métriques sur TOUS les clients (pas seulement les sélectionnés)
        metrics: dict = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
            ]
            metrics = self.fit_metrics_aggregation_fn(fit_metrics)

        n_excluded = len(weights_list) - len(selected_w)
        if n_excluded > 0:
            print(
                f"  [MultiKrum Round {server_round}] "
                f"{n_excluded}/{len(weights_list)} client(s) exclus (f={self._f})"
            )

        return parameters, metrics


# ──────────────────────────────────────────────────────────────────────────────
# FedAvg + chiffrement homomorphique (Option A : serveur honnête-mais-curieux)
# ──────────────────────────────────────────────────────────────────────────────

class FedAvgHE(fl.server.strategy.FedAvg):
    """FedAvg avec agrégation homomorphique des poids chiffrés.

    Flux par round
    --------------
    1. Clients chiffrent w_i avec la clé publique → envoient des uint8 arrays
    2. aggregate_fit() reçoit ces uint8 arrays et fait la moyenne pondérée
       SANS déchiffrer les contributions individuelles (homomorphic_average)
    3. Déchiffre uniquement le résultat agrégé enc_avg → poids en clair
    4. Renvoie les poids en clair aux clients (format Flower standard)

    Modèle de sécurité
    ------------------
    Honnête-mais-curieux : le serveur possède la clé privée mais
    suit le protocole, il n'utilise la clé privée que pour déchiffrer le
    résultat AGRÉGÉ, jamais les contributions individuelles.
    La protection est protocolaire, pas cryptographique : un serveur malveillant
    pourrait déchiffrer les contributions individuelles (il a la clé).

    Paramètres
    ----------
    he_context     : contexte CKKS COMPLET (clé publique + privée) : côté serveur
    checkpoint_dir : dossier de sauvegarde des poids (None = désactivé)
    """

    def __init__(
        self,
        he_context,
        checkpoint_dir: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._ctx = he_context
        self._ckpt_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self._ckpt_dir:
            self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._best_loss = float("inf")

        # Construire un modèle factice pour récupérer les shapes des couches.
        # Nécessaire pour reconstruire les numpy arrays après déchiffrement.
        # (importé ici pour éviter les imports circulaires au niveau module)
        from src.federated.client import _build_model
        model, _ = _build_model()
        self._shapes = [w.shape for w in model.get_weights()]
        del model

    def aggregate_fit(self, server_round: int, results, failures):
        if not results:
            return None, {}

        # 1. Extraire les uint8 arrays chiffrés et les n_samples de chaque client
        all_enc: list[list[np.ndarray]] = []
        all_n:   list[int] = []
        for _, fit_res in results:
            all_enc.append(fl.common.parameters_to_ndarrays(fit_res.parameters))
            all_n.append(fit_res.num_examples)

        # 2. Moyenne pondérée homomorphique → toujours chiffré
        from src.federated.he_context import homomorphic_average, decrypt_weights
        enc_avg = homomorphic_average(self._ctx, all_enc, all_n)

        # 3. Déchiffrement du résultat agrégé (clé privée nécessaire ici)
        decrypted = decrypt_weights(self._ctx, enc_avg, self._shapes)

        # 4. Convertir en Parameters Flower standard (plaintext)
        aggregated_params = fl.common.ndarrays_to_parameters(decrypted)

        # 5. Agréger les métriques scalaires (loss, val_loss) toujours en clair
        metrics: dict = {}
        if self.fit_metrics_aggregation_fn:
            fit_metrics = [
                (fit_res.num_examples, fit_res.metrics) for _, fit_res in results
            ]
            metrics = self.fit_metrics_aggregation_fn(fit_metrics)

        # 6. Checkpoint optionnel
        if self._ckpt_dir:
            self._save_ckpt(decrypted, server_round, metrics)

        return aggregated_params, metrics

    def _save_ckpt(self, ndarrays: list[np.ndarray], server_round: int, metrics: dict) -> None:
        path = self._ckpt_dir / f"round_{server_round:03d}.npz"
        np.savez(str(path), **{f"arr_{i:04d}": arr for i, arr in enumerate(ndarrays)})
        loss = float(metrics.get("val_loss", metrics.get("loss", float("inf")))) if metrics else float("inf")
        if loss < self._best_loss:
            self._best_loss = loss
            np.savez(
                str(self._ckpt_dir / "best_weights.npz"),
                **{f"arr_{i:04d}": arr for i, arr in enumerate(ndarrays)},
            )
            print(f"  [HE Round {server_round:2d}] Meilleur modèle → val_loss={loss:.4f}")

    @property
    def best_checkpoint_path(self) -> Path | None:
        return (self._ckpt_dir / "best_weights.npz") if self._ckpt_dir else None
