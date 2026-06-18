from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .gradcam import GradCAM


def _save(save_path: str) -> None:
    """Crée les dossiers parents si nécessaire puis sauvegarde la figure."""
    p = Path(save_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(p, dpi=150, bbox_inches="tight")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers internes
# ──────────────────────────────────────────────────────────────────────────────

def _get_subject(dataset, subject_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Retourne (img_L, img_R, hrtf) pour un sujet donné."""
    try:
        idx = dataset.subject_ids.index(subject_id)
    except ValueError:
        raise ValueError(
            f"Sujet '{subject_id}' introuvable.\n"
            f"Sujets disponibles : {dataset.subject_ids}"
        )
    return dataset._img_L[idx], dataset._img_R[idx], dataset._hrtf[idx]


def _ax_img(ax, img: np.ndarray, title: str, vmin=None, vmax=None) -> None:
    ax.imshow(np.clip(img, 0, 1), vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.axis("off")


# ──────────────────────────────────────────────────────────────────────────────
# Expérience 1 — Heatmap moyenne sur un ensemble de sujets
# ──────────────────────────────────────────────────────────────────────────────

def experiment_average_heatmap(
    gradcam:     GradCAM,
    dataset,
    subject_ids: list[str] | None = None,
    ear:         str = "left",
    score_fn     = None,
    figsize:     tuple = (14, 5),
    save_path:   str | None = None,
) -> np.ndarray:
    """Heatmap consensus : zones toujours importantes, quel que soit l'individu.

    Calcule la heatmap Grad-CAM pour chaque sujet de la liste, puis en fait
    la moyenne pixel à pixel. Les zones chaudes dans l'image finale sont celles
    que le réseau regarde systématiquement pas juste pour un sujet particulier.

    Paramètres
    ----------
    gradcam     : instance GradCAM initialisée
    dataset     : MultimodalDataset
    subject_ids : liste de sujets à inclure (None → tous)
    ear         : "left" ou "right"
    score_fn    : fonction de score (None → score_norm)
    save_path   : chemin PNG optionnel

    Retour
    ------
    avg_heatmap : (224, 224) np.ndarray, valeurs [0, 1]
    """
    if subject_ids is None:
        subject_ids = dataset.subject_ids

    heatmaps = []
    for sid in subject_ids:
        img_L, img_R, _ = _get_subject(dataset, sid)
        hm = gradcam.compute(img_L, img_R, ear=ear, score_fn=score_fn)
        heatmaps.append(hm)

    avg_heatmap = np.mean(heatmaps, axis=0)
    avg_heatmap /= max(avg_heatmap.max(), 1e-8)

    # Affichage
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle(
        f"Heatmap consensus ({ear}) — {len(subject_ids)} sujets",
        fontsize=11, fontweight="bold",
    )

    # Exemple : premier sujet
    img_L_ex, _, _ = _get_subject(dataset, subject_ids[0])
    _ax_img(axes[0], img_L_ex if ear == "left" else _get_subject(dataset, subject_ids[0])[1],
            f"Oreille ({ear})\nsujet : {subject_ids[0]}")
    _ax_img(axes[1], avg_heatmap, f"Heatmap moyenne\n(score_norm)", vmin=0, vmax=1)
    _ax_img(axes[2], GradCAM.overlay(img_L_ex, avg_heatmap),
            f"Superposition\nsur sujet {subject_ids[0]}")

    plt.tight_layout()
    if save_path:
        _save(save_path)
    plt.show()

    return avg_heatmap


# ──────────────────────────────────────────────────────────────────────────────
# Expérience 2 — Comparaison entre sujets
# ──────────────────────────────────────────────────────────────────────────────

def experiment_compare_subjects(
    gradcam:     GradCAM,
    dataset,
    subject_ids: list[str],
    ear:         str = "left",
    score_fn     = None,
    figsize:     tuple | None = None,
    save_path:   str | None = None,
) -> dict[str, np.ndarray]:
    """Compare les heatmaps de plusieurs sujets côte à côte.

    Pour chaque sujet : image brute | heatmap | superposition.
    Permet de voir si le réseau regarde les mêmes zones sur des oreilles
    morphologiquement différentes.

    Retour
    ------
    heatmaps : dict {subject_id: (224, 224) heatmap}
    """
    n = len(subject_ids)
    if figsize is None:
        figsize = (4 * 3, 3 * n)

    fig, axes = plt.subplots(n, 3, figsize=figsize, squeeze=False)
    fig.suptitle(
        f"Comparaison inter-sujets — oreille {ear}",
        fontsize=11, fontweight="bold",
    )

    col_titles = ["Image brute", "Heatmap Grad-CAM", "Superposition"]
    for col, t in enumerate(col_titles):
        axes[0, col].set_title(t, fontsize=9, pad=4)

    heatmaps = {}
    for row, sid in enumerate(subject_ids):
        img_L, img_R, _ = _get_subject(dataset, sid)
        img = img_L if ear == "left" else img_R
        hm  = gradcam.compute(img_L, img_R, ear=ear, score_fn=score_fn)
        heatmaps[sid] = hm

        axes[row, 0].imshow(np.clip(img, 0, 1))
        axes[row, 0].set_ylabel(sid, fontsize=8, rotation=0, labelpad=40, va="center")
        axes[row, 0].axis("off")

        axes[row, 1].imshow(hm, cmap="jet", vmin=0, vmax=1)
        axes[row, 1].axis("off")

        axes[row, 2].imshow(np.clip(GradCAM.overlay(img, hm), 0, 1))
        axes[row, 2].axis("off")

    plt.tight_layout()
    if save_path:
        _save(save_path)
    plt.show()

    return heatmaps


# ──────────────────────────────────────────────────────────────────────────────
# Expérience 3 — Oreille gauche vs droite (asymétrie)
# ──────────────────────────────────────────────────────────────────────────────

def experiment_left_vs_right(
    gradcam:    GradCAM,
    dataset,
    subject_id: str,
    score_fn    = None,
    figsize:    tuple = (14, 8),
    save_path:  str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compare les heatmaps oreille gauche / droite pour un même sujet.

    L'oreille droite étant le miroir anatomique de la gauche, img_R et hm_R
    sont retournés horizontalement (np.fliplr) avant comparaison, de façon à
    aligner les structures anatomiques (hélix, conque, anti-hélix…) sur le
    même côté de l'image.

    Disposition :
      Ligne 0 — oreille gauche (référence, non modifiée)
      Ligne 1 — oreille droite retournée  (alignée anatomiquement)
      Ligne 2 — diff |hm_L − hm_R_flipped|  sur la superposition gauche

    Retour
    ------
    (heatmap_L, heatmap_R_flipped)
        hm_R_flipped est déjà retourné pour être directement comparable à hm_L
        pixel à pixel (même orientation anatomique).
    """
    img_L, img_R, _ = _get_subject(dataset, subject_id)

    hm_L = gradcam.compute(img_L, img_R, ear="left",  score_fn=score_fn)
    hm_R = gradcam.compute(img_L, img_R, ear="right", score_fn=score_fn)

    # Alignement anatomique : retournement horizontal de l'oreille droite
    img_R_flip = np.fliplr(img_R)
    hm_R_flip  = np.fliplr(hm_R)
    diff       = np.abs(hm_L - hm_R_flip)

    fig, axes = plt.subplots(3, 3, figsize=figsize)
    fig.suptitle(
        f"Asymétrie gauche / droite — sujet {subject_id}",
        fontsize=11, fontweight="bold",
    )

    col_titles = ["Image brute", "Heatmap Grad-CAM", "Superposition"]
    for col, t in enumerate(col_titles):
        axes[0, col].set_title(t, fontsize=9)

    row_labels = [
        "Gauche\n(référence)",
        "Droite\n(retournée)",
        "Diff |G − D|\n(aligné)",
    ]
    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=8, rotation=0, labelpad=55, va="center")

    # Ligne 0 — gauche
    axes[0, 0].imshow(np.clip(img_L, 0, 1));                                axes[0, 0].axis("off")
    axes[0, 1].imshow(hm_L,      cmap="jet",    vmin=0, vmax=1);            axes[0, 1].axis("off")
    axes[0, 2].imshow(np.clip(GradCAM.overlay(img_L, hm_L), 0, 1));        axes[0, 2].axis("off")

    # Ligne 1 — droite retournée
    axes[1, 0].imshow(np.clip(img_R_flip, 0, 1));                           axes[1, 0].axis("off")
    axes[1, 1].imshow(hm_R_flip, cmap="jet",    vmin=0, vmax=1);            axes[1, 1].axis("off")
    axes[1, 2].imshow(np.clip(GradCAM.overlay(img_R_flip, hm_R_flip), 0, 1)); axes[1, 2].axis("off")

    # Ligne 2 — diff
    axes[2, 0].axis("off")
    axes[2, 1].imshow(diff, cmap="RdBu_r", vmin=0, vmax=1);                 axes[2, 1].axis("off")
    axes[2, 2].imshow(np.clip(GradCAM.overlay(img_L, diff), 0, 1));         axes[2, 2].axis("off")

    plt.tight_layout()
    if save_path:
        _save(save_path)
    plt.show()

    return hm_L, hm_R_flip


# ──────────────────────────────────────────────────────────────────────────────
# Expérience 4 — Évolution des heatmaps au cours de l'entraînement
# ──────────────────────────────────────────────────────────────────────────────

def experiment_training_evolution(
    ear_encoders: list,
    dataset,
    subject_id:   str,
    ear:          str = "left",
    epoch_labels: list[str] | None = None,
    score_fn      = None,
    figsize:      tuple | None = None,
    save_path:    str | None = None,
) -> list[np.ndarray]:
    """Montre comment l'attention du réseau évolue pendant l'entraînement.

    Charge une liste de checkpoints de l'EarEncoder (époque 1, 5, 10…) et
    calcule la heatmap Grad-CAM à chaque étape. Permet de visualiser le
    curriculum d'apprentissage : quelles zones le réseau apprend-il en premier ?

    Paramètres
    ----------
    ear_encoders  : liste d'instances EarEncoder (une par checkpoint)
    dataset       : MultimodalDataset
    subject_id    : sujet de référence pour la comparaison
    epoch_labels  : étiquettes des panneaux ["Init", "Ep 5", "Ep 10", ...]
    score_fn      : fonction de score (None → score_norm)
    save_path     : chemin PNG optionnel

    Usage
    -----
    from src.models import EarEncoder
    enc_init = EarEncoder(); enc_init.model.load_weights("checkpoints/epoch_0.weights.h5")
    enc_ep10 = EarEncoder(); enc_ep10.model.load_weights("checkpoints/epoch_10.weights.h5")
    experiment_training_evolution([enc_init, enc_ep10], dataset, "H10",
                                   epoch_labels=["Init", "Ep 10"])

    Retour
    ------
    liste de (224, 224) heatmaps, une par checkpoint
    """
    n = len(ear_encoders)
    if epoch_labels is None:
        epoch_labels = [f"Checkpoint {i}" for i in range(n)]
    if len(epoch_labels) != n:
        raise ValueError("epoch_labels doit avoir la même longueur que ear_encoders")

    img_L, img_R, _ = _get_subject(dataset, subject_id)
    img = img_L if ear == "left" else img_R

    heatmaps = []
    for enc in ear_encoders:
        gc  = GradCAM(enc)
        hm  = gc.compute(img_L, img_R, ear=ear, score_fn=score_fn)
        heatmaps.append(hm)

    # Disposition : 2 lignes × n colonnes (image brute en colonne 0)
    cols    = n + 1
    figsize = figsize or (3 * cols, 6)
    fig, axes = plt.subplots(2, cols, figsize=figsize, squeeze=False)
    fig.suptitle(
        f"Évolution de l'attention — sujet {subject_id} / oreille {ear}",
        fontsize=11, fontweight="bold",
    )

    # Colonne 0 : image de référence
    axes[0, 0].imshow(np.clip(img, 0, 1))
    axes[0, 0].set_title("Image brute", fontsize=9)
    axes[0, 0].axis("off")
    axes[1, 0].axis("off")   # cellule vide

    for col, (hm, label) in enumerate(zip(heatmaps, epoch_labels), start=1):
        axes[0, col].imshow(hm, cmap="jet", vmin=0, vmax=1)
        axes[0, col].set_title(label, fontsize=9)
        axes[0, col].axis("off")

        axes[1, col].imshow(np.clip(GradCAM.overlay(img, hm), 0, 1))
        axes[1, col].axis("off")

    # Étiquettes de ligne
    axes[0, 0].set_ylabel("Heatmap", fontsize=8, rotation=90)
    axes[1, 0].set_ylabel("Superposition", fontsize=8, rotation=90)

    plt.tight_layout()
    if save_path:
        _save(save_path)
    plt.show()

    return heatmaps