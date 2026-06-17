from __future__ import annotations

import math
import numpy as np
import tenseal as ts


# Max floats par vecteur CKKS (= poly_modulus_degree // 2)
CHUNK_SIZE = 4096


# ──────────────────────────────────────────────────────────────────────────────
# Contexte CKKS
# ──────────────────────────────────────────────────────────────────────────────

def build_he_context(poly_modulus_degree: int = 8192, scale_bits: int = 40) -> "ts.Context":
    """Crée un contexte CKKS complet (clé publique + privée) — côté serveur.

    poly_modulus_degree=8192 → 4096 floats chiffrés par vecteur.
    scale_bits=40 → précision ~12 décimales (suffisant pour des poids f32).
    """
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree = poly_modulus_degree,
        coeff_mod_bit_sizes = [60, scale_bits, scale_bits, 60],
    )
    context.global_scale = 2 ** scale_bits
    context.generate_galois_keys()
    context.generate_relin_keys()
    return context


def make_public_context(context: "ts.Context") -> "ts.Context":
    """Supprime la clé privée du contexte → safe à envoyer aux clients."""
    pub = context.copy()
    pub.make_context_public()
    return pub


# ──────────────────────────────────────────────────────────────────────────────
# Chiffrement / Déchiffrement
# ──────────────────────────────────────────────────────────────────────────────

def encrypt_weights(
    context: "ts.Context",
    weights: list[np.ndarray],
) -> list[np.ndarray]:
    """Chiffre les poids w_i → liste de uint8 arrays (un par chunk CKKS).

    Tous les poids sont aplatis en un seul vecteur 1D avant découpage.
    Chaque chunk chiffré est sérialisé en bytes puis encapsulé dans un
    uint8 array : format opaque mais compatible avec np.save/load de Flower.
    """
    flat = np.concatenate([w.flatten() for w in weights]).astype(np.float64).tolist()
    encrypted: list[np.ndarray] = []
    for i in range(0, max(len(flat), 1), CHUNK_SIZE):
        chunk = flat[i : i + CHUNK_SIZE]
        enc = ts.ckks_vector(context, chunk)
        encrypted.append(np.frombuffer(enc.serialize(), dtype=np.uint8))
    return encrypted


def decrypt_weights(
    context: "ts.Context",
    enc_arrays: list[np.ndarray],
    shapes: list[tuple],
) -> list[np.ndarray]:
    """Déchiffre les uint8 arrays → numpy arrays reconstruits par couche.

    Requiert la clé privée dans le contexte → serveur uniquement.
    """
    flat: list[float] = []
    for arr in enc_arrays:
        enc = ts.ckks_vector_from(context, arr.tobytes())
        flat.extend(enc.decrypt())

    weights: list[np.ndarray] = []
    idx = 0
    for shape in shapes:
        n = int(np.prod(shape))
        weights.append(
            np.array(flat[idx : idx + n], dtype=np.float32).reshape(shape)
        )
        idx += n
    return weights


# ──────────────────────────────────────────────────────────────────────────────
# Agrégation homomorphique — cœur de FedAvg + HE
# ──────────────────────────────────────────────────────────────────────────────

def homomorphic_average(
    context: "ts.Context",
    all_enc_arrays: list[list[np.ndarray]],
    all_n: list[int],
) -> list[np.ndarray]:
    """Moyenne pondérée FedAvg sur poids CHIFFRÉS sans jamais déchiffrer.

    Pour chaque chunk à la position k :
        enc_avg[k] = Σ_i ( enc_w_i[k] × (n_i / total) )

    CKKS supporte nativement :
      enc + enc  →  addition homomorphique   (pas de niveau consommé)
      enc × float → mult. scalaire plaintext (pas de niveau consommé)

    Le serveur ne voit que le résultat agrégé enc_avg, jamais les
    contributions individuelles enc_w_0, enc_w_1...
    """
    total = sum(all_n)
    n_chunks = len(all_enc_arrays[0])
    result: list[np.ndarray] = []

    for chunk_idx in range(n_chunks):
        enc_sum = None
        for client_enc, n_ex in zip(all_enc_arrays, all_n):
            enc_chunk = ts.ckks_vector_from(context, client_enc[chunk_idx].tobytes())
            enc_weighted = enc_chunk * (n_ex / total)
            enc_sum = enc_weighted if enc_sum is None else enc_sum + enc_weighted
        result.append(np.frombuffer(enc_sum.serialize(), dtype=np.uint8))

    return result


def n_chunks_for(weights: list[np.ndarray]) -> int:
    """Nombre de chunks nécessaires pour chiffrer un modèle donné."""
    total = sum(int(np.prod(w.shape)) for w in weights)
    return math.ceil(total / CHUNK_SIZE)