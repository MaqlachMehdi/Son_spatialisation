"""
ClassicalMDS.py
---------------
Implémentation des trois étapes mathématiques du Classical MDS.

Étape 1 — Mise au carré
    D²[i,j] = d²_ij
    Fait apparaître le produit scalaire xᵢ·xⱼ dans le développement
    ||xᵢ - xⱼ||² = ||xᵢ||² - 2 xᵢ·xⱼ + ||xⱼ||²

Étape 2 — Double centrage
    B = -½ J D² J     J = I - (1/n) 11ᵀ
    Preuve : J annule les termes ||xᵢ||² (car J·1 = 0)
    Résultat : B = G = X Xᵀ  (matrice de Gram)

Étape 3 — Décomposition propre
    G = V Λ Vᵀ  (V orthogonale, ||vₖ|| = 1)
    X = V₃ · Λ₃^½     col_k(X) = √λₖ · vₖ
    → λₖ = ||col_k(X)||² = Σᵢ x²ᵢₖ  (somme des carrés sur l'axe k)
"""

from __future__ import annotations

import numpy as np

from .DistanceMatrix import DistanceMatrix
from .MicPositions import MicPositions
from .logger import logger


class ClassicalMDS:
    """
    Classical Multidimensional Scaling (cMDS).

    Les résultats intermédiaires sont stockés comme attributs publics
    pour inspection depuis un notebook ou les visualisations.

    Attributs (remplis après solve())
    ----------------------------------
    D_sq   : np.ndarray (n×n)   — distances au carré
    B      : np.ndarray (n×n)   — matrice de Gram
    V      : np.ndarray (n×n)   — vecteurs propres (colonnes, normalisés)
    Lambda : np.ndarray (n,)    — valeurs propres décroissantes
    """

    def __init__(self, distance_matrix: DistanceMatrix) -> None:
        self._dm   = distance_matrix
        self.n     = len(distance_matrix)
        self.D_sq:  np.ndarray | None = None
        self.B:     np.ndarray | None = None
        self.V:     np.ndarray | None = None
        self.Lambda: np.ndarray | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Pipeline principal
    # ──────────────────────────────────────────────────────────────────────

    def solve(self) -> MicPositions:
        logger.step(3, 4, "Classical MDS")

        self.D_sq            = self._square()
        self.B               = self._double_center(self.D_sq)
        self.V, self.Lambda  = self._decompose(self.B)
        positions            = self._extract_3d(self.V, self.Lambda)

        ev_ratio = float(self.Lambda[:3].sum() / (self.Lambda.sum() + 1e-12))

        return MicPositions(
            positions=positions,
            mic_ids=tuple(self._dm.mic_ids),
            instrument_labels=tuple(self._dm.instrument_labels),
            eigenvalues=self.Lambda.copy(),
            explained_variance_ratio=ev_ratio,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Étape 1 : mise au carré
    # ──────────────────────────────────────────────────────────────────────

    def _square(self) -> np.ndarray:
        """
        D²[i,j] = d_ij²

        Le carré permet d'écrire :
            d²_ij = ||xᵢ||² + ||xⱼ||² - 2 xᵢ·xⱼ

        Sans le carré, aucun développement algébrique de ce type
        n'est possible avec les distances euclidiennes.
        """
        D_sq = self._dm.matrix ** 2
        logger.matrix_info("D²", D_sq)
        return D_sq

    # ──────────────────────────────────────────────────────────────────────
    # Étape 2 : double centrage
    # ──────────────────────────────────────────────────────────────────────

    def _double_center(self, D_sq: np.ndarray) -> np.ndarray:
        """
        B = -½ J D² J

        Avec J = I - (1/n) 11ᵀ (projecteur de centrage).

        Propriétés utilisées
        --------------------
        J · 1 = 0  →  annule les termes r_i = ||x_i||² de chaque ligne
        1ᵀ · J = 0ᵀ →  annule les termes r_j de chaque colonne
        Sous hypothèse de centrage Σxᵢ = 0 : J G J = G

        Résultat : B = G = X Xᵀ
        """
        n = self.n
        J = np.eye(n) - np.ones((n, n)) / n

        # Vérification numérique : J · 1 = 0
        ones          = np.ones(n)
        j_ones_norm   = float(np.linalg.norm(J @ ones))
        logger.check(
            "J · 1 = 0",
            ok=j_ones_norm < 1e-9,
            detail=f"||J·1|| = {j_ones_norm:.2e}",
        )

        B = -0.5 * (J @ D_sq @ J)

        # Vérifications sur B
        sym_err = float(np.max(np.abs(B - B.T)))
        logger.check(
            "B symétrique",
            ok=sym_err < 1e-8,
            detail=f"erreur = {sym_err:.2e}",
        )
        logger.scalar("tr(B) = Σλₖ", float(np.trace(B)))
        logger.matrix_info("B (Gram)", B)
        return B

    # ──────────────────────────────────────────────────────────────────────
    # Étape 3 : décomposition propre
    # ──────────────────────────────────────────────────────────────────────

    def _decompose(self, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        B = V Λ Vᵀ

        np.linalg.eigh garantit des valeurs propres réelles pour une
        matrice symétrique et est plus stable numériquement que eig.

        Normalisation ||vₖ|| = 1
        ------------------------
        Nécessaire pour que la décomposition B = V Λ Vᵀ soit valide
        (V doit être orthogonale : VᵀV = I).
        C'est aussi ce qui garantit :
            col_k(X) = √λₖ · vₖ  →  ||col_k(X)||² = λₖ · ||vₖ||² = λₖ
        """
        raw_vals, raw_vecs = np.linalg.eigh(B)

        # eigh renvoie ordre croissant → on renverse pour avoir λ₁ ≥ λ₂ ≥ ...
        idx    = np.argsort(raw_vals)[::-1]
        Lambda = raw_vals[idx]
        V      = raw_vecs[:, idx]

        # Valeurs propres légèrement négatives = bruit numérique
        n_neg = int(np.sum(Lambda < -1e-6))
        if n_neg:
            logger.warn(f"{n_neg} valeur(s) propre(s) légèrement négative(s) → clippées à 0")
        Lambda = np.maximum(Lambda, 0.0)

        # Rang effectif
        noise_floor = Lambda[0] * 1e-6
        rank        = int(np.sum(Lambda > noise_floor))
        logger.scalar("Rang effectif de B", rank)
        logger.check(
            "Hypothèse 3D valide  (rang ≥ 3)",
            ok=rank >= 3,
            detail=f"rang = {rank}",
        )

        # Vérification orthogonalité : VᵀV ≈ I
        VtV     = V.T @ V
        orth_err = float(np.max(np.abs(VtV - np.eye(self.n))))
        logger.check(
            "V orthogonale  (VᵀV ≈ I)",
            ok=orth_err < 1e-8,
            detail=f"erreur max = {orth_err:.2e}",
        )

        logger.eigenvalues(Lambda)
        return V, Lambda

    # ──────────────────────────────────────────────────────────────────────
    # Extraction 3D
    # ──────────────────────────────────────────────────────────────────────

    def _extract_3d(self, V: np.ndarray, Lambda: np.ndarray) -> np.ndarray:
        """
        X = V₃ · Λ₃^½     shape (n, 3)

        col_k(X) = √λₖ · vₖ
        →  λₖ = ||col_k(X)||²
                = Σᵢ x²ᵢₖ
                = somme des carrés des coordonnées sur l'axe k

        Les axes sont ordonnés par variance décroissante :
        axe 1 = direction de plus grande dispersion (profondeur scène)
        axe 2 = deuxième direction (largeur)
        axe 3 = troisième direction (hauteur)
        """
        V3      = V[:, :3]
        L3_sqrt = np.sqrt(Lambda[:3])
        X       = V3 * L3_sqrt[np.newaxis, :]   # broadcast colonne par colonne

        # Vérification : les colonnes de X ont bien la bonne norme
        for k in range(3):
            col_norm_sq = float(np.sum(X[:, k] ** 2))
            logger.check(
                f"||col_{k+1}(X)||² ≈ λ_{k+1}",
                ok=abs(col_norm_sq - Lambda[k]) < 1e-6,
                detail=f"||col||² = {col_norm_sq:.4f},  λ = {Lambda[k]:.4f}",
            )

        centroid_norm = float(np.linalg.norm(X.mean(axis=0)))
        logger.check(
            "Configuration centrée  (||centroïde|| ≈ 0)",
            ok=centroid_norm < 1e-6,
            detail=f"||Σxᵢ/n|| = {centroid_norm:.2e}",
        )
        logger.matrix_info("X positions 3D", X)
        logger.positions_table(self._dm.mic_ids, self._dm.instrument_labels, X)
        return X
