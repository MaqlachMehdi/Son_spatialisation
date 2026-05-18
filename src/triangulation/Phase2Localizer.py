"""
Phase2Localizer.py
------------------
Façade principale du pipeline Phase 2 — localisation des micros non-dédiés
par trilatération linéaire pondérée depuis les ancres Phase 1.

Usage
-----
    from triangulation import OrchestraLocalizer, Phase2Localizer

    loc1  = OrchestraLocalizer.from_json("dataset_live/rir_max_per_mic.json")
    pos1  = loc1.run()

    loc2  = Phase2Localizer.from_phase1(loc1)
    scene = loc2.run()

    print(scene)
    loc2.visualizer.plot_all(save_dir="figures/triangulation/")

Accès aux intermédiaires (notebook)
------------------------------------
    loc2.finder   # NonDedicatedMicFinder — inventaire et coverage
    loc2.anchors  # AnchorSet             — positions + requêtes de distance
"""

from __future__ import annotations

from .OrchestraLocalizer import OrchestraLocalizer
from .MicPositions import MicPositions
from .DistanceData import DistanceData
from .DedicatedMicMapper import DedicatedMicMapper
from .NonDedicatedMicFinder import NonDedicatedMicFinder
from .AnchorSet import AnchorSet
from .LinearTrilateration import LinearTrilateration
from .FullScenePositions import FullScenePositions, NonDedMicResult
from .logger import logger


class Phase2Localizer:
    """
    Orchestre le pipeline Phase 2 en 3 étapes :

        NonDedicatedMicFinder → AnchorSet → LinearTrilateration

    Chaque intermédiaire est accessible comme attribut public après run().

    Attributs (disponibles après run())
    ------------------------------------
    finder  : NonDedicatedMicFinder
    anchors : AnchorSet
    results : FullScenePositions
    """

    def __init__(
        self,
        dedicated_positions: MicPositions,
        distance_data:       DistanceData,
        mapper:              DedicatedMicMapper,
    ) -> None:
        self.dedicated_positions: MicPositions             = dedicated_positions
        self.distance_data:       DistanceData             = distance_data
        self.mapper:              DedicatedMicMapper       = mapper
        self.finder:              NonDedicatedMicFinder | None = None
        self.anchors:             AnchorSet | None         = None
        self._results:            FullScenePositions | None = None

    # ──────────────────────────────────────────────────────────────────────
    # Constructeur alternatif
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_phase1(cls, localizer: OrchestraLocalizer) -> Phase2Localizer:
        """Construit un Phase2Localizer depuis un OrchestraLocalizer déjà run()."""
        return cls(
            dedicated_positions=localizer.positions,
            distance_data=localizer.distance_data,
            mapper=localizer.mapper,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Pipeline
    # ──────────────────────────────────────────────────────────────────────

    def run(self) -> FullScenePositions:
        """
        Exécute le pipeline Phase 2 en 3 étapes.

        Étapes
        ------
        1. NonDedicatedMicFinder  — inventaire des mics non-dédiés + coverage
        2. AnchorSet              — référentiel 3D depuis les positions Phase 1
        3. LinearTrilateration    — WLS avec rejet d'outliers (σ = 2.0) par mic
        """
        logger.start("Phase 2 — Localisation des microphones non-dédiés")

        # ── Étape 1 : inventaire ──────────────────────────────────────────
        self.finder = NonDedicatedMicFinder.from_data(
            self.distance_data, self.mapper
        )

        # ── Étape 2 : référentiel d'ancres ────────────────────────────────
        logger.step(2, 3, "Construction du référentiel d'ancres (Phase 1 → 3D)")
        self.anchors = AnchorSet.from_phase1(
            self.dedicated_positions, self.distance_data
        )
        logger.scalar("Ancres 3D disponibles", float(self.anchors.n), "")

        # ── Étape 3 : trilatération ────────────────────────────────────────
        logger.step(3, 3, "Trilatération WLS avec rejet d'outliers (σ = 2.0)")

        non_dedicated: dict[str, NonDedMicResult] = {}
        localisable = self.finder.localisable()

        for mic_id in localisable:
            positions, distances, weights = self.anchors.get_distances(mic_id)

            try:
                pos, rms, mask = LinearTrilateration.solve_with_rejection(
                    positions, distances, weights
                )
            except ValueError as exc:
                logger.warn(f"{mic_id} ignoré : {exc}")
                continue

            result = NonDedMicResult(
                position=pos,
                residual_rms=rms,
                coverage=self.finder.coverage[mic_id],
                n_used=int(mask.sum()),
            )
            non_dedicated[mic_id] = result
            logger.check(
                f"{mic_id}",
                ok=rms < 0.5,
                detail=(
                    f"résiduel = {rms:.3f} m  "
                    f"ancres = {result.n_used}/{len(distances)}"
                ),
            )

        self._results = FullScenePositions(
            dedicated=self.dedicated_positions,
            non_dedicated=non_dedicated,
        )
        logger.done()
        return self._results

    # ──────────────────────────────────────────────────────────────────────
    # Accès aux résultats
    # ──────────────────────────────────────────────────────────────────────

    @property
    def results(self) -> FullScenePositions:
        if self._results is None:
            raise RuntimeError("Appelez .run() avant d'accéder aux résultats.")
        return self._results

    @property
    def visualizer(self) -> "FullSceneVisualizer":
        from .full_scene_visualisation import FullSceneVisualizer
        if self._results is None:
            raise RuntimeError("Appelez .run() avant d'accéder au visualizer.")
        return FullSceneVisualizer(self._results)
