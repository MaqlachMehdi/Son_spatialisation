"""Script client pour le mode distribué (un vrai PC = un client).

Usage
-----
python -m src.federated.run_client \
    --server 192.168.1.10:8080 \
    --dataset dataset/preprocessed_dataset.npz \
    --client-id 0
"""
from __future__ import annotations

import argparse

import flwr as fl
import numpy as np

from src.models import EarEncoder, HRTFEncoder, ContrastiveModel
from src.multimodal import MultimodalDataset
from src.federated.client import HRTFFlowerClient, _build_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server",    default="127.0.0.1:8080")
    parser.add_argument("--dataset",   required=True)
    parser.add_argument("--client-id", default="0")
    args = parser.parse_args()

    print(f"Client {args.client_id} — connexion à {args.server}")
    print(f"Dataset : {args.dataset}\n")

    model, enc_ear = _build_model()
    dataset = MultimodalDataset.load(args.dataset)

    client = HRTFFlowerClient(model, enc_ear, dataset, args.client_id)

    fl.client.start_client(
        server_address = args.server,
        client         = client.to_client(),
    )


if __name__ == "__main__":
    main()