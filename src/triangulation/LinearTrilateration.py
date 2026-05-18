"""
LinearTrilateration.py
----------------------
Solveur de trilatération 3D par moindres carrés pondérés linéarisés (WLS).

Algorithme
----------
Système non-linéaire pour k ancres de positions p_i et distances d_i :

    ||p_i - p||² = d_i²

Linéarisation par soustraction de l'équation de l'ancre de référence p_ref :

    2(p_i - p_ref)ᵀ p = d_ref² - d_i² + ||p_i||² - ||p_ref||²

→ Système Ap = b  (k-1 équations, 3 inconnues)
→ Solution WLS : p̂ = (AᵀWA)⁻¹ AᵀWb

L'ancre de référence est celle de poids maximal (mesure la plus fiable).

Rejet d'outliers (solve_with_rejection)
----------------------------------------
Passe 1 : solve() sur toutes les k ancres.
Rejet   : ancres dont |d_i - ||p_i - p̂₁||| > sigma_threshold × std(residuals).
Passe 2 : solve() sur les ancres survivantes (≥ 4 requises).
"""

from __future__ import annotations

import numpy as np


class LinearTrilateration:
    """
    Solveur stateless — toutes les méthodes sont des @staticmethod.
    Pas d'état interne, utilisable sans instanciation.
    """

    @staticmethod
    def solve(
        positions: np.ndarray,
        distances: np.ndarray,
        weights:   np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """
        Trilatération WLS linéarisée — passe unique.

        Parameters
        ----------
        positions : (k, 3)  — coordonnées des ancres connues (mètres)
        distances : (k,)    — distances estimées ancre → cible (mètres)
        weights   : (k,)    — confiance par ancre (None = uniforme)

        Returns
        -------
        position     : np.ndarray (3,)
        residual_rms : float  — RMS de |d_i - ||p_i - p̂||  (mètres)

        Raises
        ------
        ValueError  si k < 4 ou si le système est de rang < 3.
        """
        k = len(positions)
        if k < 4:
            raise ValueError(
                f"Trilatération 3D requiert ≥ 4 ancres valides, reçu {k}."
            )

        w = np.ones(k) if weights is None else weights / (weights.max() + 1e-12)

        # Ancre de référence : la plus fiable (poids max)
        ref  = int(np.argmax(w))
        p_ref = positions[ref]
        d_ref = distances[ref]

        rows = [i for i in range(k) if i != ref]
        P_i  = positions[rows]   # (k-1, 3)
        d_i  = distances[rows]   # (k-1,)
        w_i  = w[rows]           # (k-1,)

        # Matrice du système linéarisé
        A = 2.0 * (P_i - p_ref)
        b = (
            d_ref**2 - d_i**2
            + np.einsum("ij,ij->i", P_i, P_i)   # ||p_i||²
            - float(np.dot(p_ref, p_ref))         # ||p_ref||²
        )

        W    = np.diag(w_i)
        AtWA = A.T @ W @ A   # (3, 3)
        AtWb = A.T @ W @ b   # (3,)

        cond = np.linalg.cond(AtWA)
        if cond > 1e12:
            # Ancres quasi-coplanaires — fallback lstsq
            p_hat, _, rank, _ = np.linalg.lstsq(
                A * w_i[:, None], b * w_i, rcond=None
            )
            if rank < 3:
                raise ValueError(
                    "Système de rang < 3 : ancres coplanaires ou géométrie dégénérée."
                )
        else:
            p_hat = np.linalg.solve(AtWA, AtWb)

        residuals    = np.abs(distances - np.linalg.norm(positions - p_hat, axis=1))
        residual_rms = float(np.sqrt(np.mean(residuals**2)))

        return p_hat, residual_rms

    @staticmethod
    def solve_with_rejection(
        positions:       np.ndarray,
        distances:       np.ndarray,
        weights:         np.ndarray | None = None,
        sigma_threshold: float             = 2.0,
    ) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Trilatération en deux passes avec rejet d'outliers.

        Passe 1 : solve() sur toutes les ancres.
        Rejet   : ancres dont résiduel > sigma_threshold × std(residuals).
        Passe 2 : solve() sur les ancres survivantes.

        Si le rejet est trop agressif (< 4 survivants), toutes les ancres
        sont conservées (passe 1 sert de résultat final).

        Returns
        -------
        position     : np.ndarray (3,)
        residual_rms : float
        mask_used    : np.ndarray (k,) bool  — ancres retenues en passe 2
        """
        k = len(positions)

        # Passe 1
        p1, _ = LinearTrilateration.solve(positions, distances, weights)
        res1  = np.abs(distances - np.linalg.norm(positions - p1, axis=1))
        std1  = res1.std() + 1e-6
        mask  = res1 <= sigma_threshold * std1

        if mask.sum() < 4:
            mask = np.ones(k, dtype=bool)

        # Passe 2 sur ancres survivantes
        w_filt = weights[mask] if weights is not None else None
        p2, rms2 = LinearTrilateration.solve(
            positions[mask], distances[mask], w_filt
        )

        return p2, rms2, mask
