from __future__ import annotations

from pathlib import Path

import numpy as np
import flwr as fl

from src.multimodal import MultimodalDataset
from src.federated.partition import partition_dataset
from src.federated.client import client_fn
from src.federated.client_he import client_fn_he
from src.federated.strategies import (
    FedAvgWithCheckpoint,
    FedAvgHE,
    MultiKrum,
    load_weights_from_npz,
)


# ------------------------------------------------------------------
# Config envoyée aux clients avant chaque round
# ------------------------------------------------------------------

def fit_config(server_round: int) -> dict:
    """Pilote les 3 phases d'entraînement depuis le serveur.

    Rounds 1-5   → Phase 1 : backbone gelé,    lr élevé
    Rounds 6-10  → Phase 2 : dégel partiel,    lr moyen
    Rounds 11-15 → Phase 3 : dégel complet,    lr faible
    """
    if server_round <= 5:
        return {"phase": 1, "local_epochs": 5, "lr": 1e-3}
    elif server_round <= 10:
        return {"phase": 2, "local_epochs": 5, "lr": 1e-4}
    else:
        return {"phase": 3, "local_epochs": 3, "lr": 1e-5}


# ------------------------------------------------------------------
# Agrégation des métriques (loss, val_loss) retournées par les clients
# ------------------------------------------------------------------

def weighted_average(metrics: list[tuple[int, dict]]) -> dict:
    """Moyenne pondérée par n_samples de toutes les métriques numériques.

    FedAvg agrège les poids automatiquement.
    Cette fonction agrège les métriques du dict retourné par fit() / evaluate().

    metrics : [(n_samples_client_0, {"loss": 2.3, "val_loss": 2.5}), ...]
    """
    total = sum(n for n, _ in metrics)
    aggregated = {}
    for key, value in metrics[0][1].items():
        if isinstance(value, (int, float)):
            aggregated[key] = sum(n * m[key] for n, m in metrics if key in m) / total
    return aggregated


# ------------------------------------------------------------------
# Stratégie d'agrégation
# ------------------------------------------------------------------

def build_strategy(
    min_clients:    int = 2,
    strategy_type:  str = "fedavg",
    checkpoint_dir: str | None = "checkpoints/federated",
    num_malicious:  int = 1,
    he_context                 = None,
) -> fl.server.strategy.Strategy:
    """Construit la stratégie d'agrégation.

    Paramètres
    ----------
    strategy_type  : "fedavg"     → FedAvg standard avec checkpoint
                     "krum"       → Multi-Krum, robuste aux attaques Byzantine
                     "fedavg_he"  → FedAvg + chiffrement homomorphique 
    checkpoint_dir : dossier de sauvegarde des poids (None = désactivé).
    num_malicious  : nb de clients malveillants supposés (pour "krum" uniquement).
    he_context     : contexte CKKS complet (obligatoire pour "fedavg_he").
    """
    common = dict(
        fraction_fit                    = 1.0,
        fraction_evaluate               = 1.0,
        min_fit_clients                 = min_clients,
        min_evaluate_clients            = min_clients,
        min_available_clients           = min_clients,
        on_fit_config_fn                = fit_config,
        fit_metrics_aggregation_fn      = weighted_average,
        evaluate_metrics_aggregation_fn = weighted_average,
    )

    if strategy_type == "krum":
        return MultiKrum(num_malicious=num_malicious, **common)

    if strategy_type == "fedavg_he":
        if he_context is None:
            raise ValueError(
                "he_context obligatoire pour strategy_type='fedavg_he'. "
                "Utiliser build_he_context() pour le créer."
            )
        return FedAvgHE(
            he_context     = he_context,
            checkpoint_dir = checkpoint_dir,
            **common,
        )

    if checkpoint_dir:
        return FedAvgWithCheckpoint(checkpoint_dir=checkpoint_dir, **common)

    return fl.server.strategy.FedAvg(**common)


# ------------------------------------------------------------------
# Point d'entrée simulation
# ------------------------------------------------------------------

def run_simulation(
    dataset_path:   str,
    n_clients:      int = 3,
    num_rounds:     int = 15,
    seed:           int = 42,
    strategy_type:  str = "fedavg",
    checkpoint_dir: str | None = "checkpoints/federated",
    num_malicious:  int = 1,
) -> tuple[object, list[np.ndarray] | None]:
    """Lance une simulation FL complète sur une seule machine.

    Retour
    ------
    (history, final_weights)
        history       : objet Flower History avec les métriques par round
        final_weights : poids du meilleur modèle global (list[np.ndarray]),
                        prêt pour model.set_weights(). None si pas de checkpoint.

    Usage
    -----
    # FedAvg avec sauvegarde des poids
    history, weights = run_simulation(
        "dataset/preprocessed_dataset.npz",
        n_clients=3, strategy_type="fedavg", checkpoint_dir="checkpoints/fl"
    )
    model.set_weights(weights)

    # Multi-Krum (robuste Byzantine)
    history, _ = run_simulation(..., strategy_type="krum", num_malicious=1)
    """
    print(f"\n{'='*55}")
    print(f"  SIMULATION FEDERATED LEARNING")
    print(f"  stratégie={strategy_type.upper()}  |  {n_clients} clients  |  {num_rounds} rounds")
    print(f"{'='*55}\n")

    print(f"Chargement du dataset : {dataset_path}")
    dataset = MultimodalDataset.load(dataset_path)
    print(f"{dataset}\n")

    print(f"Partitionnement en {n_clients} clients...")
    partitions = partition_dataset(dataset, n_clients=n_clients, seed=seed)

    # Préparer le contexte HE si nécessaire
    he_context  = None
    _client_fn  = client_fn(partitions)          # factory par défaut (sans HE)

    if strategy_type == "fedavg_he":
        from src.federated.he_context import build_he_context, make_public_context
        print("Génération du contexte CKKS (clé publique + privée)...")
        he_context  = build_he_context()
        public_ctx  = make_public_context(he_context)
        _client_fn  = client_fn_he(partitions, public_ctx)
        print("Contexte public distribué aux clients.\n")

    strategy = build_strategy(
        min_clients    = n_clients,
        strategy_type  = strategy_type,
        checkpoint_dir = checkpoint_dir,
        num_malicious  = num_malicious,
        he_context     = he_context,
    )

    print(f"\nDémarrage de la simulation...\n")
    history = fl.simulation.start_simulation(
        client_fn        = _client_fn,
        num_clients      = n_clients,
        strategy         = strategy,
        config           = fl.server.ServerConfig(num_rounds=num_rounds),
        # Sans ça, Ray détecte tous les CPU de la machine (ex: 8) et lance
        # 8 process séparés (chacun avec sa propre copie de TensorFlow) même
        # si on n'a que n_clients clients réels — ça sature la RAM et fait
        # crasher les workers. On force le cluster à n_clients CPU exactement.
        ray_init_args    = {
            "num_cpus":            n_clients,
            "ignore_reinit_error": True,
            "include_dashboard":   False,
        },
        client_resources = {"num_cpus": 1, "num_gpus": 0.0},
    )

    _print_history(history)

    # Récupérer les poids du meilleur modèle depuis le checkpoint
    final_weights = None
    if checkpoint_dir and hasattr(strategy, "best_checkpoint_path"):
        best_path = strategy.best_checkpoint_path
        if best_path.exists():
            final_weights = load_weights_from_npz(best_path)
            print(f"  Poids finaux chargés depuis : {best_path}")
        else:
            print("  Aucun checkpoint trouvé (val_loss jamais améliorée).")

    return history, final_weights


def _print_history(history) -> None:
    print(f"\n{'='*55}")
    print("  RÉSULTATS DE LA SIMULATION")
    print(f"{'='*55}")
    for round_num, (_, loss) in enumerate(history.losses_distributed, start=1):
        print(f"  Round {round_num:2d}  →  loss agrégée : {loss:.4f}")
    print(f"{'='*55}\n")


# ------------------------------------------------------------------
# Mode distribué (un seul PC joue le rôle de serveur)
# ------------------------------------------------------------------

def run_server(
    server_address: str = "0.0.0.0:8080",
    num_rounds:     int = 15,
    n_clients:      int = 2,
) -> None:
    """Lance le serveur en mode distribué (vrais PCs réseau).

    Lancer sur le PC serveur :
        python -m src.federated.server

    Lancer sur chaque PC client :
        python -m src.federated.run_client --server 192.168.X.X:8080 --dataset ...
    """
    strategy = build_strategy(min_clients=n_clients, checkpoint_dir=None)
    fl.server.start_server(
        server_address = server_address,
        strategy       = strategy,
        config         = fl.server.ServerConfig(num_rounds=num_rounds),
    )


if __name__ == "__main__":
    run_server()