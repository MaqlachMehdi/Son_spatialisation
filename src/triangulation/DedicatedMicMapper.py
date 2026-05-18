"""
DedicatedMicMapper.py
---------------------
Identifie le microphone dédié de chaque instrument : argmin(sample / max_abs).

Le critère sample / max_abs combine précocité (sample bas) et intensité
(max_abs élevé). Un micro éloigné peut présenter un échantillon très précoce
(artefact de déclenchement ou pré-écho) mais avec une amplitude infime — ce
ratio le pénalise correctement par rapport au vrai micro dédié.

Règle d'exclusion : mic_05 est systématiquement écarté avant le calcul
(distances apparentes 45–215 m, physiquement impossibles — cf. LaTeX §1.4).
"""

from __future__ import annotations

from dataclasses import dataclass

from .DistanceData import DistanceData
from .logger import logger

_EXCLUDED_MICS: frozenset[str] = frozenset({"mic_05"})


@dataclass
class DedicatedMicMapper:
    """
    Bijection instrument ↔ microphone dédié.

    Attributs
    ---------
    instrument_to_mic : dict[str, str]
        {"Vln_1": "mic_06", "Vln_2": "mic_07", ...}
    mic_to_instrument : dict[str, str]
        Inverse : {"mic_06": "Vln_1", ...}
    """

    instrument_to_mic: dict[str, str]
    mic_to_instrument: dict[str, str]

    # ──────────────────────────────────────────────────────────────────────
    # Constructeur alternatif
    # ──────────────────────────────────────────────────────────────────────

    @classmethod
    def from_data(cls, data: DistanceData) -> DedicatedMicMapper:
        logger.step(1, 4, "Identification des microphones dédiés  (argmin sample/max_abs)")

        instrument_to_mic: dict[str, str] = {}

        for instrument in data.instruments:
            raw_mics = {
                mic_id: data.raw[instrument][mic_id]
                for mic_id in data.raw[instrument]
                if mic_id not in _EXCLUDED_MICS
            }
            if not raw_mics:
                logger.warn(f"Aucun candidat pour {instrument} après exclusions.")
                continue

            # score = sample / max_abs : faible quand le signal est à la fois
            # précoce (sample bas) et fort (max_abs élevé) → vrai micro dédié
            scores = {
                mic_id: vals["sample"] / vals["max_abs"]
                for mic_id, vals in raw_mics.items()
            }
            dedicated = min(scores, key=lambda m: scores[m])
            instrument_to_mic[instrument] = dedicated

            sample  = raw_mics[dedicated]["sample"]
            max_abs = raw_mics[dedicated]["max_abs"]
            logger.check(
                f"{instrument:<22s} → {dedicated}",
                ok=True,
                detail=f"sample={sample}  max_abs={max_abs:.4f}  score={scores[dedicated]:.1f}",
            )

        mic_to_instrument = {v: k for k, v in instrument_to_mic.items()}

        # Vérification d'unicité : deux instruments ne doivent pas partager
        # le même micro dédié (sinon la matrice 17×17 aurait une ligne dupliquée)
        n_unique = len(set(instrument_to_mic.values()))
        n_total  = len(instrument_to_mic)
        unique_ok = n_unique == n_total
        logger.check(
            "Unicité des micros dédiés",
            ok=unique_ok,
            detail=f"{n_unique}/{n_total} uniques",
        )
        if not unique_ok:
            logger.warn(
                "Deux instruments partagent le même micro dédié. "
                "La matrice 17×17 aura des lignes dupliquées — vérifiez les données."
            )

        return cls(
            instrument_to_mic=instrument_to_mic,
            mic_to_instrument=mic_to_instrument,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Propriétés utilitaires
    # ──────────────────────────────────────────────────────────────────────

    @property
    def dedicated_mic_ids(self) -> list[str]:
        """Liste des mic_id dédiés, triés numériquement."""
        return sorted(
            self.instrument_to_mic.values(),
            key=lambda x: int(x.split("_")[1]),
        )

    @property
    def n(self) -> int:
        return len(self.instrument_to_mic)
