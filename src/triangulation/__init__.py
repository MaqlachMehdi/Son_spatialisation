"""
triangulation
-------------
Pipeline en deux phases de localisation 3D des microphones d'orchestre.

Phase 1 — Classical MDS (micros dédiés)
----------------------------------------
    from triangulation import OrchestraLocalizer

    loc1      = OrchestraLocalizer.from_json("dataset_live/rir_max_per_mic.json")
    positions = loc1.run()

    print(positions)              # tableau az/el/dist par micro dédié
    loc1.visualizer.plot_all()    # 4 figures : 3D, vue dessus, eigenspectrum, heatmap

Phase 2 — Trilatération (micros non-dédiés)
--------------------------------------------
    from triangulation import Phase2Localizer

    loc2  = Phase2Localizer.from_phase1(loc1)
    scene = loc2.run()

    print(scene)                  # Phase 1 + Phase 2 (positions + résiduels)
    loc2.visualizer.plot_all()    # 2 figures : vue complète, bar chart résiduels

Accès aux intermédiaires
-------------------------
    loc1.mapper            # DedicatedMicMapper — bijection instrument ↔ mic
    loc1.distance_matrix   # DistanceMatrix     — matrice 17×17 symétrisée
    loc1._mds.Lambda       # np.ndarray         — valeurs propres

    loc2.finder            # NonDedicatedMicFinder — inventaire + coverage
    loc2.anchors           # AnchorSet             — positions + requêtes TOA

Contrôle du logger
------------------
    from triangulation.logger import logger
    logger.set_debug()    # affiche shapes et stats intermédiaires
    logger.set_info()     # retour au niveau normal
"""

from .OrchestraLocalizer import OrchestraLocalizer
from .Phase2Localizer import Phase2Localizer
from .MicPositions import MicPositions
from .FullScenePositions import FullScenePositions, NonDedMicResult
from .AxisOrientator import OrchestralAxisOrientator
from .logger import logger

__all__ = [
    "OrchestraLocalizer",
    "Phase2Localizer",
    "MicPositions",
    "FullScenePositions",
    "NonDedMicResult",
    "OrchestralAxisOrientator",
    "logger",
]
