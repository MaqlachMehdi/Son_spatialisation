"""
HRTFGen.py
----------
HRTF générique construite par moyenne pondérée de N sujets SOFA.

Algorithme (par position et par oreille) :
  1. M_avg(f) = Σᵢ wᵢ · |RFFT(hᵢ(f))|   — moyenne des modules (jamais négative)
  2. τ_avg    = Σᵢ wᵢ · τᵢ               — onset moyen pondéré
  3. H_mp     = MinimumPhase(M_avg)        — reconstruction phase-minimum
  4. H_final  = H_mp · exp(−j2πf·τ_avg)  — retard fractionnaire exact
  5. h        = Re{IRFFT(H_final)}

Contrairement à la moyenne de spectres complexes (risque de filtre en peigne),
la moyenne des modules garantit M_avg > 0 — aucune annulation spectrale possible.

Persistance :
  Sauvegarde en .sofa (SOFA SimpleFreeFieldHRIR, netCDF4).
  Rechargeable via HRTFGen.from_sofa() ou HRTF.from_sofa().
  Compatible avec HRTFInterpolator(gen) sans modification.

Usage
-----
    # Calcul une seule fois :
    gen = HRTFGen.from_directory("dataset/", pattern="IRC_*.sofa")
    gen.save("dataset/generic.sofa")

    # Utilisation rapide (rechargement instantané) :
    gen   = HRTFGen.from_sofa("dataset/generic.sofa")
    interp = HRTFInterpolator(gen)
    conv   = DynamicConvolver(hrtf=interp, ...)
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset as NC4Dataset
from scipy.fft import rfft

from .hrtf import HRTF
from .hrtf_utils import build_mp_window, detect_onset, reconstruct_hrir


class HRTFGen:
    """
    HRTF générique construite par moyenne de modules sur N sujets.

    Expose la même interface que HRTF — drop-in replacement pour
    HRTFConvolver, DynamicConvolver et HRTFInterpolator.

    Attributs
    ---------
    hrir         : np.ndarray (M, 2, N)  — HRIRs génériques (gauche, droite)
    positions    : np.ndarray (M, 3)    — [azimut°, élévation°, distance_m]
    sample_rate  : int
    n_subjects   : int                  — nombre de sujets contribuants
    source_names : list[str]            — noms des fichiers SOFA source
    """

    # ──────────────────────────────────────────────────────────────────────
    # Constructeur interne
    # ──────────────────────────────────────────────────────────────────────

    def __init__(
        self,
        hrir:         np.ndarray,
        positions:    np.ndarray,
        sample_rate:  int,
        n_subjects:   int,
        source_names: list[str],
    ) -> None:
        self.hrir         = hrir
        self.positions    = positions
        self.sample_rate  = int(sample_rate)
        self.n_subjects   = int(n_subjects)
        self.source_names = list(source_names)

    # ──────────────────────────────────────────────────────────────────────
    # Propriétés — interface identique à HRTF
    # ──────────────────────────────────────────────────────────────────────

    @property
    def n_positions(self) -> int:
        return self.hrir.shape[0]

    @property
    def n_samples(self) -> int:
        return self.hrir.shape[2]

    @property
    def duration_ms(self) -> float:
        return self.n_samples / self.sample_rate * 1000

    @property
    def azimuths(self) -> np.ndarray:
        return np.unique(self.positions[:, 0])

    @property
    def elevations(self) -> np.ndarray:
        return np.unique(self.positions[:, 1])

    @property
    def time_axis_ms(self) -> np.ndarray:
        return np.arange(self.n_samples) / self.sample_rate * 1000

    @property
    def freq_axis_hz(self) -> np.ndarray:
        from scipy.fft import rfftfreq
        return rfftfreq(self.n_samples, 1 / self.sample_rate)

    # ──────────────────────────────────────────────────────────────────────
    # Méthodes — interface identique à HRTF
    # ──────────────────────────────────────────────────────────────────────

    def get_index(self, azimuth: float, elevation: float) -> int:
        """Indice du plus proche voisin par distance grand-cercle."""
        az_rad = np.deg2rad(self.positions[:, 0])
        el_rad = np.deg2rad(self.positions[:, 1])
        az_tgt = np.deg2rad(azimuth)
        el_tgt = np.deg2rad(elevation)
        cos_d  = (
            np.sin(el_tgt) * np.sin(el_rad)
            + np.cos(el_tgt) * np.cos(el_rad) * np.cos(az_tgt - az_rad)
        )
        return int(np.argmin(np.arccos(np.clip(cos_d, -1.0, 1.0))))

    def get_hrir(
        self, azimuth: float, elevation: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Retourne (hrir_gauche, hrir_droite) pour la position la plus proche."""
        idx = self.get_index(azimuth, elevation)
        return self.hrir[idx, 0, :].copy(), self.hrir[idx, 1, :].copy()

    def get_hrtf(
        self, azimuth: float, elevation: float
    ) -> tuple[np.ndarray, np.ndarray]:
        """Spectres de magnitude HRTF génériques (linéaire, rfft)."""
        left, right = self.get_hrir(azimuth, elevation)
        return np.abs(rfft(left)), np.abs(rfft(right))

    # ──────────────────────────────────────────────────────────────────────
    # Fabriques
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        sofa_paths: list[str | Path],
        weights:    list[float] | None = None,
    ) -> "HRTFGen":
        """
        Construit la HRTF générique depuis une liste de fichiers SOFA.

        Paramètres
        ----------
        sofa_paths : list[str | Path]
            Chemins vers les fichiers .sofa des sujets.
        weights : list[float] | None
            Poids par sujet (ex. scores anthropométriques, similarité).
            None → moyenne uniforme 1/N.

        Retourne
        --------
        HRTFGen prêt à l'emploi (non encore sauvegardé sur disque).

        Lève
        ----
        FileNotFoundError  Si aucun fichier n'est valide.
        ValueError         Si les sample rates ou longueurs HRIR diffèrent.
        """
        print(f"[HRTFGen] Chargement de {len(sofa_paths)} fichiers SOFA...")
        subjects = cls._load_subjects(sofa_paths)

        print(f"[HRTFGen] Validation de la grille ({len(subjects)} sujets)...")
        common_positions = cls._validate_grid(subjects)

        print(
            f"[HRTFGen] Calcul de la HRTF générique "
            f"({common_positions.shape[0]} positions × 2 oreilles)..."
        )
        hrir_generic = cls._compute_generic(subjects, common_positions, weights)

        source_names = [s.source_path.name for s in subjects]
        print(
            f"[HRTFGen] Terminé — {len(subjects)} sujets, "
            f"{common_positions.shape[0]} positions."
        )
        return cls(
            hrir         = hrir_generic,
            positions    = common_positions,
            sample_rate  = subjects[0].sample_rate,
            n_subjects   = len(subjects),
            source_names = source_names,
        )

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        pattern:   str = "*.sofa",
        weights:   list[float] | None = None,
    ) -> "HRTFGen":
        """
        Construit la HRTF générique depuis tous les .sofa d'un dossier.

        Paramètres
        ----------
        directory : str | Path
            Dossier contenant les fichiers SOFA des sujets.
        pattern : str
            Glob pattern pour filtrer les fichiers (défaut : '*.sofa').
        weights : list[float] | None
            Poids par fichier, dans l'ordre alphabétique des chemins.

        Retourne
        --------
        HRTFGen prêt à l'emploi.
        """
        directory = Path(directory)
        paths = sorted(directory.glob(pattern))
        if not paths:
            raise FileNotFoundError(
                f"Aucun fichier '{pattern}' trouvé dans : {directory}"
            )
        print(f"[HRTFGen] {len(paths)} fichiers trouvés dans '{directory.name}/'")
        return cls.build(paths, weights=weights)

    @classmethod
    def from_sofa(cls, path: str | Path) -> "HRTFGen":
        """
        Recharge une HRTF générique depuis un fichier .sofa sauvegardé par save().

        Paramètres
        ----------
        path : str | Path
            Chemin vers le fichier .sofa générique.

        Retourne
        --------
        HRTFGen reconstruit identique à l'objet sauvegardé.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Fichier introuvable : {path}")

        # Données HRTF via le chargeur standard (gère convention CW/CCW, swap oreilles)
        hrtf = HRTF.from_sofa(path)

        # Métadonnées spécifiques HRTFGen stockées dans les attributs globaux SOFA
        with NC4Dataset(str(path), "r") as ds:
            n_subjects   = int(getattr(ds, "HRTFGen_nsubjects", "0"))
            sources_raw  = getattr(ds, "HRTFGen_sources", "")
            source_names = [s for s in sources_raw.split(";") if s]

        print(
            f"[HRTFGen] Chargé depuis cache : {path.name} "
            f"({n_subjects} sujets, {hrtf.n_positions} positions)"
        )
        return cls(
            hrir         = hrtf.hrir,
            positions    = hrtf.positions,
            sample_rate  = hrtf.sample_rate,
            n_subjects   = n_subjects,
            source_names = source_names,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Persistance
    # ──────────────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """
        Sauvegarde la HRTF générique en fichier SOFA SimpleFreeFieldHRIR.

        Le fichier est rechargeable via :
          • HRTFGen.from_sofa(path)          — reconstruit un HRTFGen
          • HRTF.from_sofa(path)             — reconstruit un HRTF standard
          • HRTFInterpolator(HRTFGen.from_sofa(path))  — interpolation complète

        Paramètres
        ----------
        path : str | Path
            Chemin de destination (extension .sofa recommandée).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._write_sofa(path)
        print(f"[HRTFGen] Sauvegardé : {path}")

    # ──────────────────────────────────────────────────────────────────────
    # Visualisation
    # ──────────────────────────────────────────────────────────────────────

    def plot_comparison(
        self,
        other:     HRTF,
        azimuth:   float = 0.0,
        elevation: float = 0.0,
    ) -> None:
        """
        Compare la HRTF générique avec celle d'un sujet individuel.

        Affiche en parallèle :
          • Spectres de magnitude gauche et droite (dB vs kHz)
          • HRIRs gauche et droite (amplitude vs ms, zoom 0–5 ms)

        Paramètres
        ----------
        other : HRTF
            HRTF d'un sujet individuel (ou d'un autre HRTFGen).
        azimuth, elevation : float
            Position de comparaison en degrés (convention SOFA).
        """
        import matplotlib.pyplot as plt

        freqs_khz = self.freq_axis_hz / 1000

        gen_mag_l, gen_mag_r = self.get_hrtf(azimuth, elevation)
        ind_mag_l, ind_mag_r = other.get_hrtf(azimuth, elevation)
        gen_ir_l,  gen_ir_r  = self.get_hrir(azimuth, elevation)
        ind_ir_l,  ind_ir_r  = other.get_hrir(azimuth, elevation)

        def _db(x: np.ndarray) -> np.ndarray:
            return 20 * np.log10(np.maximum(x, 1e-10))

        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        fig.suptitle(
            f"Générique ({self.n_subjects} sujets)  vs  Individuel\n"
            f"Az={azimuth}°  El={elevation}°",
            fontsize=11, fontweight="bold",
        )

        ind_label = getattr(other, "source_path", Path("individuel")).name

        for col, (gen_mag, ind_mag, gen_ir, ind_ir, ear) in enumerate(zip(
            [gen_mag_l, gen_mag_r],
            [ind_mag_l, ind_mag_r],
            [gen_ir_l,  gen_ir_r],
            [ind_ir_l,  ind_ir_r],
            ["Gauche", "Droite"],
        )):
            # ── Spectre de magnitude ──────────────────────────────────
            ax = axes[0, col]
            ax.plot(freqs_khz, _db(ind_mag),
                    color="#9E9E9E", linewidth=1.0, alpha=0.8, label=ind_label)
            ax.plot(freqs_khz, _db(gen_mag),
                    color="#2196F3", linewidth=1.8,
                    label=f"Générique ({self.n_subjects} sujets)")
            ax.set_xlim(0, self.sample_rate / 2000)
            ax.set_xlabel("Fréquence (kHz)")
            ax.set_ylabel("Magnitude (dB)")
            ax.set_title(f"Spectre HRTF — oreille {ear}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

            # ── HRIR ─────────────────────────────────────────────────
            ax = axes[1, col]
            t_ms = self.time_axis_ms
            ax.plot(t_ms, ind_ir,
                    color="#9E9E9E", linewidth=0.8, alpha=0.8, label=ind_label)
            ax.plot(t_ms, gen_ir,
                    color="#2196F3", linewidth=1.2,
                    label=f"Générique")
            ax.set_xlim(0, min(5.0, float(t_ms[-1])))
            ax.set_xlabel("Temps (ms)")
            ax.set_ylabel("Amplitude")
            ax.set_title(f"HRIR — oreille {ear}")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    # ──────────────────────────────────────────────────────────────────────
    # Méthodes internes
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_subjects(sofa_paths: list[str | Path]) -> list[HRTF]:
        """
        Charge chaque fichier SOFA via HRTF.from_sofa().

        Valide la cohérence du sample_rate et de n_samples entre sujets.
        Les fichiers invalides sont ignorés avec un avertissement.

        Lève RuntimeError si aucun sujet n'a pu être chargé.
        Lève ValueError si sample_rate ou n_samples diffèrent entre sujets.
        """
        subjects: list[HRTF] = []
        for p in sofa_paths:
            try:
                h = HRTF.from_sofa(p)
                subjects.append(h)
                print(
                    f"  ✓  {Path(p).name:<35} "
                    f"{h.n_positions} pos  "
                    f"sr={h.sample_rate} Hz  "
                    f"N={h.n_samples} smp"
                )
            except Exception as exc:
                print(f"  ✗  {Path(p).name} — ignoré : {exc}")

        if not subjects:
            raise RuntimeError(
                "Aucun fichier SOFA valide chargé. Vérifiez les chemins fournis."
            )

        srs = {s.sample_rate for s in subjects}
        if len(srs) > 1:
            raise ValueError(
                f"Sample rates hétérogènes entre sujets : {srs}. "
                "Tous les sujets doivent partager le même taux d'échantillonnage."
            )

        lens = {s.n_samples for s in subjects}
        if len(lens) > 1:
            raise ValueError(
                f"Longueurs HRIR hétérogènes entre sujets : {lens} échantillons. "
                "Tous les sujets doivent avoir des HRIRs de même longueur."
            )

        return subjects

    @staticmethod
    def _validate_grid(
        subjects:  list[HRTF],
        tolerance: float = 0.5,
    ) -> np.ndarray:
        """
        Vérifie que tous les sujets partagent une grille de positions compatible
        et retourne les positions communes.

        Stratégie :
          - Le premier sujet définit la grille de référence.
          - Pour chaque autre sujet, chaque position de référence est validée
            si son plus proche voisin dans ce sujet est à moins de `tolerance`°
            (distance grand-cercle).
          - Les positions invalides pour au moins un sujet sont exclues.

        Paramètres
        ----------
        subjects : list[HRTF]
        tolerance : float
            Seuil de distance angulaire en degrés (défaut : 0.5°).

        Retourne
        --------
        np.ndarray, shape (M_common, 3) — positions communes validées.
        """
        ref          = subjects[0]
        common_mask  = np.ones(ref.n_positions, dtype=bool)

        for subj in subjects[1:]:
            for p_idx in np.where(common_mask)[0]:
                az  = float(ref.positions[p_idx, 0])
                el  = float(ref.positions[p_idx, 1])
                idx = subj.get_index(az, el)
                naz = float(subj.positions[idx, 0])
                nel = float(subj.positions[idx, 1])

                # Distance grand-cercle en degrés
                cos_d = (
                    np.sin(np.deg2rad(el))  * np.sin(np.deg2rad(nel))
                    + np.cos(np.deg2rad(el)) * np.cos(np.deg2rad(nel))
                    * np.cos(np.deg2rad(az - naz))
                )
                dist_deg = float(np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0))))
                if dist_deg > tolerance:
                    common_mask[p_idx] = False

        n_common = int(common_mask.sum())
        n_total  = ref.n_positions

        if n_common < n_total:
            print(
                f"[HRTFGen] Grilles hétérogènes : {n_total - n_common} position(s) "
                f"exclue(s) (> {tolerance}° d'écart). "
                f"Grille commune retenue : {n_common} positions."
            )
        else:
            print(f"[HRTFGen] Grille homogène ✓  ({n_common} positions)")

        return ref.positions[common_mask].copy()

    @staticmethod
    def _compute_generic(
        subjects:         list[HRTF],
        common_positions: np.ndarray,
        weights:          list[float] | None = None,
    ) -> np.ndarray:
        """
        Calcule les HRIRs génériques par moyenne pondérée des modules.

        Pour chaque position p et chaque oreille e :
          M_avg(f) = Σᵢ wᵢ · |RFFT(hᵢ(p,e))|
          τ_avg    = Σᵢ wᵢ · τᵢ(p,e)
          hrir_out(p,e) = reconstruct_hrir(M_avg, τ_avg)

        Paramètres
        ----------
        subjects : list[HRTF]
        common_positions : np.ndarray (M, 3)
        weights : list[float] | None

        Retourne
        --------
        np.ndarray, shape (M, 2, N), dtype float32
        """
        n   = len(subjects)
        N   = subjects[0].n_samples
        M   = len(common_positions)

        # Poids normalisés
        w = (
            np.ones(n, dtype=np.float64) / n
            if weights is None
            else np.asarray(weights, dtype=np.float64) / np.sum(weights)
        )

        # Précalculs partagés (construits une seule fois)
        mp_window  = build_mp_window(N)
        rfft_freqs = np.fft.rfftfreq(N)
        n_fft      = N // 2 + 1

        hrir_out = np.zeros((M, 2, N), dtype=np.float32)

        for p_idx in range(M):
            az = float(common_positions[p_idx, 0])
            el = float(common_positions[p_idx, 1])

            for ear in range(2):
                mag_avg = np.zeros(n_fft, dtype=np.float64)
                tau_avg = 0.0

                for s_idx, subj in enumerate(subjects):
                    idx = subj.get_index(az, el)
                    h   = subj.hrir[idx, ear, :].astype(np.float64)
                    mag_avg += w[s_idx] * np.abs(rfft(h))
                    tau_avg += w[s_idx] * detect_onset(h)

                hrir_out[p_idx, ear, :] = reconstruct_hrir(
                    mag_avg, tau_avg, N, mp_window, rfft_freqs
                )

            if (p_idx + 1) % 30 == 0 or (p_idx + 1) == M:
                print(f"  [{p_idx + 1:>3}/{M}] positions traitées")

        return hrir_out

    def _write_sofa(self, path: Path) -> None:
        """
        Écrit un fichier SOFA SimpleFreeFieldHRIR valide (netCDF4).

        Convention de canaux dans ReceiverPosition :
          index 0 → oreille gauche  (y = +0.0875 m > 0)
          index 1 → oreille droite  (y = −0.0875 m < 0)
        Après squeeze() dans HRTF.from_sofa() : y0 > y1 → pas de swap → index 0 = gauche ✓
        """
        M, _, N = self.hrir.shape
        R = 2
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        with NC4Dataset(str(path), "w", format="NETCDF4") as ds:
            # ── Attributs globaux SOFA (SimpleFreeFieldHRIR 1.0) ─────────
            ds.Conventions             = "SOFA"
            ds.Version                 = "2.1"
            ds.SOFAConventions         = "SimpleFreeFieldHRIR"
            ds.SOFAConventionsVersion  = "1.0"
            ds.APIName                 = "HRTFGen"
            ds.APIVersion              = "1.0"
            ds.ApplicationName         = "son_spatialisation"
            ds.ApplicationVersion      = "1.0"
            ds.AuthorContact           = ""
            ds.Comment                 = (
                f"HRTF générique — moyenne de {self.n_subjects} sujets. "
                "Méthode : moyenne des modules + reconstruction phase-minimum."
            )
            ds.DataType                = "FIR"
            ds.History                 = ""
            ds.License                 = "CC BY 4.0"
            ds.Organization            = ""
            ds.References              = ""
            ds.RoomType                = "free field"
            ds.Origin                  = ""
            ds.DateCreated             = now
            ds.DateModified            = now
            ds.Title                   = f"Generic HRTF ({self.n_subjects} subjects)"
            ds.ListenerShortName       = "Generic"
            ds.DatabaseName            = "HRTFGen"
            # Métadonnées propres à HRTFGen (relues par from_sofa)
            ds.HRTFGen_nsubjects       = str(self.n_subjects)
            ds.HRTFGen_sources         = ";".join(self.source_names)

            # ── Dimensions ───────────────────────────────────────────────
            ds.createDimension("M", M)
            ds.createDimension("R", R)
            ds.createDimension("N", N)
            ds.createDimension("I", 1)
            ds.createDimension("E", 1)
            ds.createDimension("C", 3)

            # ── Data.IR (M, R, N) ─────────────────────────────────────────
            v    = ds.createVariable("Data.IR", "f8", ("M", "R", "N"))
            v[:] = self.hrir.astype(np.float64)

            # ── Data.SamplingRate (I,) ────────────────────────────────────
            v         = ds.createVariable("Data.SamplingRate", "f8", ("I",))
            v[:]      = [float(self.sample_rate)]
            v.Units   = "hertz"

            # ── Data.Delay (I, R) — zéro : l'onset est déjà dans la HRIR ─
            v         = ds.createVariable("Data.Delay", "f8", ("I", "R"))
            v[:]      = [[0.0, 0.0]]
            v.Units   = "samples"

            # ── SourcePosition (M, C) ─────────────────────────────────────
            v         = ds.createVariable("SourcePosition", "f8", ("M", "C"))
            v[:]      = self.positions.astype(np.float64)
            v.Type    = "spherical"
            v.Units   = "degree, degree, metre"

            # ── ListenerPosition (I, C) ───────────────────────────────────
            v         = ds.createVariable("ListenerPosition", "f8", ("I", "C"))
            v[:]      = [[0.0, 0.0, 0.0]]
            v.Type    = "cartesian"
            v.Units   = "metre"

            # ── ListenerView (I, C) ───────────────────────────────────────
            v         = ds.createVariable("ListenerView", "f8", ("I", "C"))
            v[:]      = [[1.0, 0.0, 0.0]]
            v.Type    = "cartesian"
            v.Units   = "metre"

            # ── ListenerUp (I, C) ─────────────────────────────────────────
            v         = ds.createVariable("ListenerUp", "f8", ("I", "C"))
            v[:]      = [[0.0, 0.0, 1.0]]
            v.Type    = "cartesian"
            v.Units   = "metre"

            # ── ReceiverPosition (R, C, I) ────────────────────────────────
            rp              = np.zeros((R, 3, 1), dtype=np.float64)
            rp[0, :, 0]     = [ 0.0,  0.0875, 0.0]   # gauche  (y > 0)
            rp[1, :, 0]     = [ 0.0, -0.0875, 0.0]   # droite  (y < 0)
            v               = ds.createVariable("ReceiverPosition", "f8", ("R", "C", "I"))
            v[:]            = rp
            v.Type          = "cartesian"
            v.Units         = "metre"

            # ── EmitterPosition (E, C, I) ─────────────────────────────────
            v         = ds.createVariable("EmitterPosition", "f8", ("E", "C", "I"))
            v[:]      = np.zeros((1, 3, 1), dtype=np.float64)
            v.Type    = "cartesian"
            v.Units   = "metre"

    # ──────────────────────────────────────────────────────────────────────
    # Représentation
    # ──────────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"HRTFGen("
            f"sujets={self.n_subjects}, "
            f"positions={self.n_positions}, "
            f"samples={self.n_samples}, "
            f"sr={self.sample_rate} Hz)"
        )
