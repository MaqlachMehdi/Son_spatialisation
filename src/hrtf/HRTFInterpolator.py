"""
HRTFInterpolator.py
-------------------
Interpolation sphérique barycentrique des HRIRs — sans filtre en peigne.

Wrape un objet HRTF et expose la même interface (get_hrir, get_index, sample_rate,
positions, …) — drop-in replacement de HRTF dans HRTFConvolver et DynamicConvolver.

Méthode
-------
1. Triangulation de Delaunay sphérique via convex hull 3D.

2. Coordonnées barycentriques sphériques (formule de Van Oosterom) :
     ω_A = Ω(P,B,C) / Ω(A,B,C)

3. Interpolation magnitude + onset séparés :
     M_interp(f)   = Σ ωi · |H_i(f)|          ← pas de filtre en peigne
     τ_interp      = Σ ωi · τ_i
     H_mp(f)       = exp(RFFT(w_mp · IRFFT(log M_interp)))   ← phase-minimum
     H_final(f)    = H_mp(f) · exp(−j2πf·τ_interp)          ← retard exact
     h_interp      = Re{IRFFT(H_final)}

   Contrairement à l'interpolation complexe directe (≡ moyenne temporelle),
   interpoler les MODULES évite toute annulation spectrale : M_interp > 0 par
   construction. Le retard d'onset est appliqué séparément en fréquentiel,
   ce qui équivaut à un filtre sinc parfait (retard fractionnaire exact).

Usage
-----
    from hrtf import HRTF
    from HRTFInterpolator import HRTFInterpolator

    hrtf  = HRTF.from_sofa("IRC_1002_C_44100.sofa")
    interp = HRTFInterpolator(hrtf)

    # Position hors grille — HRIR synthétisée sans artefact de peigne
    left, right = interp.get_hrir(42.5, 37.5)

    # Drop-in dans HRTFConvolver / DynamicConvolver (interface identique à HRTF)
    conv = DynamicConvolver(hrtf=interp, signal=signal, ...)
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from scipy.fft import rfft, irfft

from .hrtf import HRTF
from .hrtf_utils import (
    build_mp_window,
    detect_onset,
    minimum_phase_from_magnitude,
    reconstruct_hrir,
)


class HRTFInterpolator:
    """
    Interpolateur HRTF sphérique barycentrique — drop-in replacement de HRTF.

    Paramètres
    ----------
    hrtf : HRTF
        Dataset HRTF chargé (données brutes non modifiées).

    Attributs internes précalculés à l'init
    ----------------------------------------
    _vecs       : ndarray (M, 3)         — vecteurs unitaires des positions
    _hull       : ConvexHull             — triangulation de Delaunay sphérique
    _faces      : ndarray (F, 3)         — faces orientées CCW
    _onsets     : ndarray (M, 2)         — retard d'onset τ en échantillons
    _magnitudes : ndarray (M, 2, N//2+1) — |RFFT(hrir)| précalculé
    _mp_window  : ndarray (N,)           — fenêtre phase-minimum
    _rfft_freqs : ndarray (N//2+1,)      — fréquences normalisées [0, 0.5]
    """

    def __init__(self, hrtf: HRTF, verbose: bool = True) -> None:
        self.hrtf    = hrtf
        self.verbose = verbose
        N = hrtf.n_samples

        # ── Triangulation ──────────────────────────────────────────────────
        self._vecs: np.ndarray = self._to_unit_vectors(
            hrtf.positions[:, 0], hrtf.positions[:, 1]
        )
        self._hull  = ConvexHull(self._vecs)
        self._faces = self._orient_faces()

        # ── Précalculs pour l'interpolation ────────────────────────────────
        self._onsets:     np.ndarray = self._compute_onsets()
        self._magnitudes: np.ndarray = self._compute_magnitudes()

        # Fenêtre phase-minimum (réutilisée à chaque appel)
        self._mp_window = build_mp_window(N)

        # Fréquences normalisées pour l'application du retard fractionnaire
        # rfftfreq(N)[k] = k/N  → phase = exp(−j2π·(k/N)·τ) = exp(−j2πkτ/N)
        self._rfft_freqs = np.fft.rfftfreq(N)  # shape (N//2+1,)

    # ── Délégation vers HRTF sous-jacent ─────────────────────────────────

    def __getattr__(self, name: str):
        """
        Délègue tout attribut inconnu à self.hrtf.
        Permet d'utiliser HRTFInterpolator exactement comme HRTF :
        positions, sample_rate, n_positions, n_samples, duration_ms, …
        """
        if name == "hrtf":
            raise AttributeError(name)
        return getattr(self.hrtf, name)

    # ── Interface publique — même signature que HRTF ─────────────────────

    def get_index(self, azimuth: float, elevation: float) -> int:
        """Indice du plus proche voisin (conservé pour affichage et logs)."""
        return self.hrtf.get_index(azimuth, elevation)

    def get_hrir(
        self, azimuth: float, elevation: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Retourne les HRIRs (gauche, droite) interpolées à (azimuth, elevation).

        Si le triangle encadrant n'est pas trouvé (cas dégénéré numérique),
        repli sur le plus proche voisin mesuré.

        Paramètres
        ----------
        azimuth : float
            Azimut en degrés (SOFA : 0°=devant, 90°=gauche).
        elevation : float
            Élévation en degrés (-90°=bas, 0°=horizontal, 90°=zénith).

        Retourne
        --------
        (hrir_left, hrir_right) : tuple de ndarray float32, shape (N,)
        """
        p    = self._az_el_to_vec(azimuth, elevation)
        face = self._find_triangle(p)

        if face is None:
            idx = self.hrtf.get_index(azimuth, elevation)
            if self.verbose:
                pos = self.hrtf.positions[idx]
                print(f"[HRTFInterpolator] Cible ({azimuth:.1f}, {elevation:.1f}) deg"
                      f" -- triangle non trouve, repli NN :"
                      f" Az={pos[0]:.1f} El={pos[1]:.1f} deg (indice {idx})")
            return (
                self.hrtf.hrir[idx, 0, :].copy(),
                self.hrtf.hrir[idx, 1, :].copy(),
            )

        i_A, i_B, i_C = face
        weights = self._barycentric_weights(
            p,
            self._vecs[i_A],
            self._vecs[i_B],
            self._vecs[i_C],
        )

        if self.verbose:
            pa = self.hrtf.positions[i_A]
            pb = self.hrtf.positions[i_B]
            pc = self.hrtf.positions[i_C]
            dominant = int(weights.argmax())
            if weights[dominant] > 1.0 - 1e-6:
                print(f"[HRTFInterpolator] Cible ({azimuth:.1f}, {elevation:.1f}) deg"
                      f" -- sur grille, HRIR mesuree :"
                      f" Az={self.hrtf.positions[face[dominant], 0]:.1f}"
                      f" El={self.hrtf.positions[face[dominant], 1]:.1f} deg")
            else:
                print(f"[HRTFInterpolator] Cible ({azimuth:.1f}, {elevation:.1f}) deg"
                      f" -- triangle :")
                print(f"  A  Az={pa[0]:6.1f} El={pa[1]:5.1f} deg"
                      f"  w={weights[0]:.3f}"
                      f"  tauL={self._onsets[i_A,0]:.0f}  tauR={self._onsets[i_A,1]:.0f} smp")
                print(f"  B  Az={pb[0]:6.1f} El={pb[1]:5.1f} deg"
                      f"  w={weights[1]:.3f}"
                      f"  tauL={self._onsets[i_B,0]:.0f}  tauR={self._onsets[i_B,1]:.0f} smp")
                print(f"  C  Az={pc[0]:6.1f} El={pc[1]:5.1f} deg"
                      f"  w={weights[2]:.3f}"
                      f"  tauL={self._onsets[i_C,0]:.0f}  tauR={self._onsets[i_C,1]:.0f} smp")
                tau_L = float(weights @ self._onsets[face, 0])
                tau_R = float(weights @ self._onsets[face, 1])
                print(f"  -> tau_interp  L={tau_L:.2f} smp  R={tau_R:.2f} smp")

        left  = self._interp_channel(0, (i_A, i_B, i_C), weights)
        right = self._interp_channel(1, (i_A, i_B, i_C), weights)
        return left, right

    def get_hrtf(
        self, azimuth: float, elevation: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spectres de magnitude HRTF interpolés (linéaire, rfft)."""
        left, right = self.get_hrir(azimuth, elevation)
        return np.abs(rfft(left)), np.abs(rfft(right))

    # ── Initialisation : triangulation ───────────────────────────────────

    @staticmethod
    def _to_unit_vectors(az_deg: np.ndarray, el_deg: np.ndarray) -> np.ndarray:
        """Coordonnées sphériques → vecteurs 3D unitaires (convention SOFA)."""
        az = np.deg2rad(az_deg)
        el = np.deg2rad(el_deg)
        return np.column_stack([
            np.cos(el) * np.cos(az),   # x = devant
            np.cos(el) * np.sin(az),   # y = gauche
            np.sin(el),                # z = haut
        ])

    def _orient_faces(self) -> np.ndarray:
        """
        Oriente chaque face du convex hull CCW vue de l'extérieur.

        ConvexHull garantit une normale sortante via equations[:, :3] mais
        n'impose pas l'ordre CCW des sommets dans simplices.
        On vérifie : si (B−A)×(C−A) pointe dans le sens opposé à la normale,
        on permute B et C.
        """
        faces = []
        for k, face in enumerate(self._hull.simplices):
            normal = self._hull.equations[k, :3]
            A = self._vecs[face[0]]
            B = self._vecs[face[1]]
            C = self._vecs[face[2]]
            if np.dot(np.cross(B - A, C - A), normal) < 0:
                face = np.array([face[0], face[2], face[1]])
            faces.append(face)
        return np.array(faces)

    # ── Initialisation : précalculs pour l'interpolation ─────────────────

    def _compute_onsets(self) -> np.ndarray:
        """
        Précalcule le retard d'onset τ (en échantillons) pour chaque HRIR.

        Délègue à hrtf_utils.detect_onset.
        Retourne ndarray shape (M, 2), dtype float64.
        """
        M   = self.hrtf.n_positions
        out = np.zeros((M, 2), dtype=np.float64)
        for i in range(M):
            for ear in range(2):
                out[i, ear] = detect_onset(self.hrtf.hrir[i, ear, :])
        return out

    def _compute_magnitudes(self) -> np.ndarray:
        """
        Précalcule |RFFT(hrir)| pour chaque (position, oreille).

        Retourne ndarray shape (M, 2, N//2+1), dtype float64.
        Ces tableaux sont réutilisés à chaque appel à _interp_channel.
        """
        M     = self.hrtf.n_positions
        n_fft = self.hrtf.n_samples // 2 + 1
        out   = np.zeros((M, 2, n_fft), dtype=np.float64)
        for i in range(M):
            for ear in range(2):
                out[i, ear] = np.abs(
                    rfft(self.hrtf.hrir[i, ear, :].astype(np.float64))
                )
        return out

    # ── Géométrie sphérique ───────────────────────────────────────────────

    @staticmethod
    def _az_el_to_vec(az_deg: float, el_deg: float) -> np.ndarray:
        """Scalaire (az, el) → vecteur 3D unitaire."""
        az = np.deg2rad(az_deg)
        el = np.deg2rad(el_deg)
        return np.array([
            np.cos(el) * np.cos(az),
            np.cos(el) * np.sin(az),
            np.sin(el),
        ])

    def _find_triangle(self, p: np.ndarray) -> np.ndarray | None:
        """
        Cherche la face orientée CCW dont le triangle sphérique contient p.

        Un point P est à l'intérieur du triangle (A, B, C) si :
            (A × B) · P ≥ 0
            (B × C) · P ≥ 0
            (C × A) · P ≥ 0
        Tolérance ε=−1e-10 pour les arêtes.

        Retourne le triplet d'indices [i_A, i_B, i_C] ou None.
        """
        eps = -1e-10
        for face in self._faces:
            A = self._vecs[face[0]]
            B = self._vecs[face[1]]
            C = self._vecs[face[2]]
            if (
                np.dot(np.cross(A, B), p) >= eps
                and np.dot(np.cross(B, C), p) >= eps
                and np.dot(np.cross(C, A), p) >= eps
            ):
                return face
        return None

    @staticmethod
    def _solid_angle(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
        """
        Angle solide du triangle sphérique (u, v, w).
        Formule de Van Oosterom & Strackee (1983) :
            Ω = 2·arctan( |u·(v×w)| / (1 + u·v + u·w + v·w) )
        """
        num = abs(float(np.dot(u, np.cross(v, w))))
        den = 1.0 + float(np.dot(u, v)) + float(np.dot(u, w)) + float(np.dot(v, w))
        return 2.0 * np.arctan2(num, max(den, 1e-30))

    def _barycentric_weights(
        self,
        P: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
    ) -> np.ndarray:
        """
        Poids barycentriques sphériques de P dans le triangle (A, B, C).

            ω_A = Ω(P,B,C) / Ω(A,B,C)
            ω_B = Ω(A,P,C) / Ω(A,B,C)
            ω_C = Ω(A,B,P) / Ω(A,B,C)

        Retourne ndarray([ω_A, ω_B, ω_C]), normalisé à somme = 1.
        """
        omega_total = self._solid_angle(A, B, C)
        if omega_total < 1e-30:
            return np.array([1 / 3, 1 / 3, 1 / 3])

        w = np.array([
            self._solid_angle(P, B, C),
            self._solid_angle(A, P, C),
            self._solid_angle(A, B, P),
        ]) / omega_total

        return w / w.sum()   # normalisation numérique

    # ── Interpolation phase-minimum + onset ───────────────────────────────

    def _minimum_phase_from_magnitude(self, mag: np.ndarray) -> np.ndarray:
        """
        Spectre complexe phase-minimum depuis un spectre de magnitude.

        Délègue à hrtf_utils.minimum_phase_from_magnitude.
        """
        return minimum_phase_from_magnitude(
            mag, self.hrtf.n_samples, self._mp_window
        )

    def _interp_channel(
        self,
        ear:     int,                       # 0 = gauche, 1 = droite
        indices: tuple[int, int, int],      # indices des 3 vertices
        weights: np.ndarray,                # shape (3,), somme = 1
    ) -> np.ndarray:
        """
        Interpolation d'un canal HRIR par décomposition phase-minimum / onset.

        Étapes :
          1. M_interp = Σ ωi · |H_i(f)|     — interpolation des modules
          2. H_mp     = MinPhase(M_interp)   — reconstruction phase-minimum
          3. τ_interp = Σ ωi · τ_i           — interpolation des onset
          4. H_final  = H_mp · exp(−j2πf·τ) — retard fractionnaire exact
          5. h_interp = Re{IRFFT(H_final)}

        Garantie : M_interp > 0 → pas de zéro spectral → pas de filtre en peigne.

        Retourne ndarray float32, shape (N,).
        """
        i_A, i_B, i_C = indices
        w_A, w_B, w_C = weights

        # ── Shortcut sur-grille ────────────────────────────────────────────
        # Si un vertex domine quasi-totalement (point exactement sur la grille),
        # retourner la HRIR mesurée sans reconstruction pour préserver la mesure.
        dominant = int(np.argmax(weights))
        if weights[dominant] > 1.0 - 1e-6:
            return self.hrtf.hrir[indices[dominant], ear, :].copy().astype(np.float32)

        # ── 1. Interpolation des modules (précalculés) ────────────────────
        M_interp = (
            w_A * self._magnitudes[i_A, ear]
            + w_B * self._magnitudes[i_B, ear]
            + w_C * self._magnitudes[i_C, ear]
        )

        # ── 2. Reconstruction du spectre phase-minimum ────────────────────
        H_mp = self._minimum_phase_from_magnitude(M_interp)

        # ── 3. Interpolation du retard d'onset ────────────────────────────
        tau_interp = (
            w_A * self._onsets[i_A, ear]
            + w_B * self._onsets[i_B, ear]
            + w_C * self._onsets[i_C, ear]
        )

        # ── 4. Retard fractionnaire exact dans le domaine fréquentiel ──────
        # rfftfreq(N)[k] = k/N  →  phase shift = exp(−j·2π·(k/N)·τ)
        H_final = H_mp * np.exp(-1j * 2.0 * np.pi * self._rfft_freqs * tau_interp)

        # ── 5. IFFT et partie réelle ──────────────────────────────────────
        return np.real(irfft(H_final, n=self.hrtf.n_samples)).astype(np.float32)

    # ── Représentation ────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"HRTFInterpolator("
            f"positions={self.hrtf.n_positions}, "
            f"triangles={len(self._hull.simplices)}, "
            f"sr={self.hrtf.sample_rate} Hz, "
            f"source='{self.hrtf.source_path.name}')"
        )
