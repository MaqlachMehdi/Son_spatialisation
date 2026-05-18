"""
AxisOrientator.py
-----------------
Orientation sémantique des axes MDS par contraintes orchestrales connues.

Principe
--------
Le cMDS fournit une configuration correcte aux distances près, mais dans un
repère arbitraire (rotation et réflexion quelconques).  Sans Kabsch (qui
nécessite des points d'ancrage avec coordonnées 3D réelles), on contraint
les axes via la disposition connue d'un orchestre symphonique :

    Axe profondeur (X) :  cordes en avant (X < 0), cuivres+percus en arrière (X > 0)
    Axe latéral   (Y) :  Vln_1 à gauche du chef (Y > 0), Vcl à droite (Y < 0)
    Axe hauteur   (Z) :  axe restant — variance physique minimale sur scène

Ces trois contraintes permettent :
  1. d'identifier quel axe MDS correspond à chaque direction physique
     (via les centroïdes de familles),
  2. de corriger le signe (retournement) de chaque axe.

Limites
-------
Cette méthode suppose que la disposition de l'orchestre est standard. Elle
ne corrige pas la rotation dans le plan (angle de la scène par rapport à
l'axe profondeur) — pour cela, Kabsch ou un point d'ancrage GPS est requis.
"""

from __future__ import annotations

import numpy as np

from .MicPositions import MicPositions
from .logger import logger

_STRINGS    = frozenset({"Vln_1", "Vln_2", "Vla", "Vcl", "DBass_1", "DBass_2"})
_BRASS_PERC = frozenset({"Horns", "Trumpets", "Trombones", "Tuba", "Timpani", "Bass_Drum"})


class OrchestralAxisOrientator:
    """
    Assigne chaque axe MDS à une direction physique (profondeur / latéral /
    hauteur) en exploitant la disposition connue de l'orchestre symphonique.

    Usage
    -----
        orientator  = OrchestralAxisOrientator(raw_positions)
        oriented    = orientator.orient()
    """

    def __init__(self, positions: MicPositions) -> None:
        self._pos = positions

    def orient(self) -> MicPositions:
        logger.step(4, 4, "Orientation sémantique des axes  (contraintes orchestrales)")

        X      = self._pos.positions.copy()
        labels = list(self._pos.instrument_labels)

        # ── Indices des groupes ───────────────────────────────────────────────
        idx_strings    = [i for i, l in enumerate(labels) if l in _STRINGS]
        idx_brass_perc = [i for i, l in enumerate(labels) if l in _BRASS_PERC]
        idx_vln1       = labels.index("Vln_1")
        idx_vcl        = labels.index("Vcl")

        c_str = X[idx_strings].mean(axis=0)       # centroïde cordes
        c_bp  = X[idx_brass_perc].mean(axis=0)    # centroïde cuivres+percus

        # ── Identification des axes ───────────────────────────────────────────
        # Profondeur : axe qui sépare le plus cordes / cuivres+percus
        diff_depth = np.abs(c_bp - c_str)
        depth_axis = int(np.argmax(diff_depth))

        # Latéral : parmi les deux axes restants, celui avec la plus grande valeur propre
        # (plus de variance = plus grande étendue latérale sur scène)
        # Hauteur : le troisième (variance physique minimale sur une scène plane)
        remaining  = [a for a in range(3) if a != depth_axis]
        lambdas    = self._pos.eigenvalues
        if lambdas[remaining[0]] >= lambdas[remaining[1]]:
            lat_axis, height_axis = remaining[0], remaining[1]
        else:
            lat_axis, height_axis = remaining[1], remaining[0]

        logger.info(
            f"Axes MDS identifiés : "
            f"profondeur=λ{depth_axis+1}  latéral=λ{lat_axis+1}  hauteur=λ{height_axis+1}"
        )
        logger.scalar(
            f"Séparation cordes/cuivres sur λ{depth_axis+1}",
            float(diff_depth[depth_axis]), "m",
        )
        logger.scalar(f"λ{lat_axis+1}    (latéral)",  float(lambdas[lat_axis]),    "")
        logger.scalar(f"λ{height_axis+1}    (hauteur)", float(lambdas[height_axis]),  "")

        # ── Permutation des colonnes ──────────────────────────────────────────
        perm = [depth_axis, lat_axis, height_axis]
        X    = X[:, perm]

        c_str_p = c_str[perm]
        c_bp_p  = c_bp[perm]

        # ── Correction des signes ─────────────────────────────────────────────
        # Profondeur : cuivres+percus derrière → X positif
        if c_bp_p[0] < c_str_p[0]:
            X[:, 0] *= -1
            logger.check("Profondeur  (cuivres/percus X > 0)", ok=True, detail="signe inversé")
        else:
            logger.check("Profondeur  (cuivres/percus X > 0)", ok=True)

        # Latéral : Vln_1 à gauche du chef → Y positif (SOFA 90°=gauche)
        if X[idx_vln1, 1] < X[idx_vcl, 1]:
            X[:, 1] *= -1
            logger.check("Latéral  (Vln_1 Y > 0, gauche chef)", ok=True, detail="signe inversé")
        else:
            logger.check("Latéral  (Vln_1 Y > 0, gauche chef)", ok=True)

        logger.positions_table(self._pos.mic_ids, labels, X)

        return MicPositions(
            positions=X,
            mic_ids=self._pos.mic_ids,
            instrument_labels=self._pos.instrument_labels,
            eigenvalues=self._pos.eigenvalues,
            explained_variance_ratio=self._pos.explained_variance_ratio,
        )
