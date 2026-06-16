from __future__ import annotations

import flwr as fl

from src.multimodal import MultimodalDataset
from src.federated.partition import partition_dataset
from src.federated.client import client_fn


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
# Stratégie FedAvg
# ------------------------------------------------------------------

def build_strategy(min_clients: int = 2) -> fl.server.strategy.FedAvg:
    """Construit la stratégie FedAvg.

    Pour passer à Krum plus tard, remplacer par :
        fl.server.strategy.Krum(num_malicious_clients=1, ...)
    """
    return fl.server.strategy.FedAvg(
        fraction_fit                    = 1.0,
        fraction_evaluate               = 1.0,
        min_fit_clients                 = min_clients,
        min_evaluate_clients            = min_clients,
        min_available_clients           = min_clients,
        on_fit_config_fn                = fit_config,
        fit_metrics_aggregation_fn      = weighted_average,
        evaluate_metrics_aggregation_fn = weighted_average,
    )


# ------------------------------------------------------------------
# Point d'entrée simulation
# ------------------------------------------------------------------

def run_simulation(
    dataset_path: str,
    n_clients:   int = 3,
    num_rounds:  int = 15,
    seed:        int = 42,
):
    """Lance une simulation FL complète sur une seule machine.

    Usage
    -----
    from src.federated.server import run_simulation
    history = run_simulation("dataset/preprocessed_dataset.npz", n_clients=3)
    """
    print(f"\n{'='*55}")
    print(f"  SIMULATION FEDERATED LEARNING")
    print(f"  {n_clients} clients  |  {num_rounds} rounds")
    print(f"{'='*55}\n")

    print(f"Chargement du dataset : {dataset_path}")
    dataset = MultimodalDataset.load(dataset_path)
    print(f"{dataset}\n")

    print(f"Partitionnement en {n_clients} clients...")
    partitions = partition_dataset(dataset, n_clients=n_clients, seed=seed)

    strategy = build_strategy(min_clients=n_clients)

    print(f"\nDémarrage de la simulation...\n")
    history = fl.simulation.start_simulation(
        client_fn        = client_fn(partitions),
        num_clients      = n_clients,
        strategy         = strategy,
        config           = fl.server.ServerConfig(num_rounds=num_rounds),
   
        # virtuel à n'avoir que n_clients CPU → exactement n_clients acteurs.
        ray_init_args    = {
            "num_cpus":           n_clients,
            "ignore_reinit_error": True,
            "include_dashboard":   False,
        },
        client_resources = {"num_cpus": 1, "num_gpus": 0.0},
    )

    _print_history(history)
    return history


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
    strategy = build_strategy(min_clients=n_clients)
    fl.server.start_server(
        server_address = server_address,
        strategy       = strategy,
        config         = fl.server.ServerConfig(num_rounds=num_rounds),
    )


if __name__ == "__main__":
    run_server()