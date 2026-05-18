"""
full_scene_visualisation.py
---------------------------
Visualisations de la scène complète : Phase 1 (micros dédiés) + Phase 2
(micros non-dédiés localisés par trilatération).

Usage
-----
    loc2  = Phase2Localizer.from_phase1(loc1)
    scene = loc2.run()
    viz   = loc2.visualizer

    viz.top_view.plot()
    viz.residual_chart.plot()
    viz.plot_all(save_dir="figures/triangulation/")
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe

from .FullScenePositions import FullScenePositions
from .visualisation import (
    _BG_DARK, _BG_PANEL, _SPINE,
    _FAMILY_COLORS, _FAMILIES,
    _family_color, _family_of, _style_ax,
)

# ── Seuils et couleurs de qualité de trilatération ───────────────────────────
_THR_GOOD: float = 0.30   # résiduel < 0.3 m → fiable
_THR_MED:  float = 0.80   # résiduel < 0.8 m → acceptable

_COLOR_GOOD = "#56C272"   # vert
_COLOR_MED  = "#E8A23C"   # orange
_COLOR_BAD  = "#E85C5C"   # rouge


def _residual_color(rms: float) -> str:
    if rms < _THR_GOOD:
        return _COLOR_GOOD
    if rms < _THR_MED:
        return _COLOR_MED
    return _COLOR_BAD


# ══════════════════════════════════════════════════════════════════════════════
# Classe de base
# ══════════════════════════════════════════════════════════════════════════════

class _BaseScenePlot:
    def __init__(self, scene: FullScenePositions) -> None:
        self.scene = scene

    def _savefig(self, fig: plt.Figure, path: str | None) -> None:
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Vue de dessus — scène complète
# ══════════════════════════════════════════════════════════════════════════════

class FullSceneTopView(_BaseScenePlot):
    """
    Vue de dessus de la scène orchestrale complète.

    Convention d'affichage
    ----------------------
    Axe X du plot (horizontal, inversé) : latéral Y_MDS
        → valeurs positives (Vln_1) à gauche du plot
    Axe Y du plot (vertical)            : profondeur X_MDS
        → valeurs négatives (avant-scène) en bas

    Micros dédiés     ○  : cercles colorés par famille instrumentale (Phase 1)
    Micros non-dédiés □  : carrés colorés par qualité de trilatération (Phase 2)
                           + disque d'incertitude de rayon = résiduel_rms
    """

    def plot(self, save_path: str | None = None) -> plt.Figure:
        ded  = self.scene.dedicated
        nded = self.scene.non_dedicated

        pos_ded   = ded.positions          # (n, 3)
        mics_ded  = ded.mic_ids
        instr_ded = ded.instrument_labels

        fig, ax = plt.subplots(figsize=(14, 11))
        fig.patch.set_facecolor(_BG_DARK)
        ax.set_facecolor(_BG_PANEL)
        fig.suptitle(
            "Vue de dessus — scène complète"
            "  (○ Phase 1 : dédiés · □ Phase 2 : non-dédiés)",
            fontsize=12, fontweight="bold", color="white",
        )

        # ── Micros dédiés (Phase 1) ───────────────────────────────────────
        plotted: set[str] = set()
        for i, (mic_id, instrument) in enumerate(zip(mics_ded, instr_ded)):
            color  = _family_color(instrument)
            family = _family_of(instrument)
            label  = family.capitalize() if family not in plotted else None
            plotted.add(family)

            ax.scatter(
                pos_ded[i, 1], pos_ded[i, 0],
                c=color, s=160, edgecolors="white", linewidths=0.7,
                marker="o", label=label, zorder=5,
            )
            num = mic_id.split("_")[1]
            ax.annotate(
                num,
                xy=(pos_ded[i, 1], pos_ded[i, 0]),
                xytext=(0, 0), textcoords="offset points",
                ha="center", va="center",
                fontsize=5.5, fontweight="bold", color="white", zorder=6,
            )
            ax.annotate(
                instrument,
                xy=(pos_ded[i, 1], pos_ded[i, 0]),
                xytext=(6, 6), textcoords="offset points",
                ha="left", va="bottom",
                fontsize=6.5, color=color,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=_BG_PANEL)],
                zorder=7,
            )

        # ── Micros non-dédiés (Phase 2) ───────────────────────────────────
        for mic_id, res in nded.items():
            xp    = float(res.position[1])   # latéral → X plot
            yp    = float(res.position[0])   # profondeur → Y plot
            color = _residual_color(res.residual_rms)

            ax.scatter(
                xp, yp,
                c=color, s=190, edgecolors="white", linewidths=0.9,
                marker="s", zorder=5,
            )
            num = mic_id.split("_")[1]
            ax.annotate(
                num,
                xy=(xp, yp),
                xytext=(0, 0), textcoords="offset points",
                ha="center", va="center",
                fontsize=5.5, fontweight="bold", color="white", zorder=6,
            )
            ax.annotate(
                mic_id,
                xy=(xp, yp),
                xytext=(6, -8), textcoords="offset points",
                ha="left", va="top",
                fontsize=6.5, color=color,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=_BG_PANEL)],
                zorder=7,
            )
            # Disque d'incertitude de rayon = résiduel RMS
            if res.residual_rms > 0:
                ax.add_patch(mpatches.Circle(
                    (xp, yp), radius=res.residual_rms,
                    color=color, alpha=0.12, zorder=2,
                ))

        # ── Centroïde global ──────────────────────────────────────────────
        ax.scatter(0, 0, c="white", s=90, marker="+", zorder=10, label="centroïde")

        # ── Limites ───────────────────────────────────────────────────────
        lat_coords = list(pos_ded[:, 1])
        dep_coords = list(pos_ded[:, 0])
        if nded:
            lat_coords += [r.position[1] for r in nded.values()]
            dep_coords += [r.position[0] for r in nded.values()]

        lat_arr = np.array(lat_coords)
        dep_arr = np.array(dep_coords)
        margin  = 1.5
        ax.set_xlim(lat_arr.min() - margin, lat_arr.max() + margin)
        ax.set_ylim(dep_arr.min() - margin, dep_arr.max() + margin)
        ax.invert_xaxis()

        # ── Annotations directionnelles ───────────────────────────────────
        xlo, xhi = lat_arr.min(), lat_arr.max()
        ylo, yhi = dep_arr.min(), dep_arr.max()
        ax.text(xhi + 0.7, 0, "Gauche chef\n(Vln_1)",
                ha="center", va="center", fontsize=7, color="#888899")
        ax.text(xlo - 0.7, 0, "Droite chef\n(Vcl)",
                ha="center", va="center", fontsize=7, color="#888899")
        ax.text(0, ylo - 0.8, "Avant-scène\n(cordes)",
                ha="center", va="top",    fontsize=7, color="#888899")
        ax.text(0, yhi + 0.8, "Arrière\n(cuivres/percus)",
                ha="center", va="bottom", fontsize=7, color="#888899")

        # ── Légende Phase 2 ───────────────────────────────────────────────
        p2_legend = [
            mpatches.Patch(color=_COLOR_GOOD, label=f"Phase 2 — résiduel < {_THR_GOOD} m"),
            mpatches.Patch(color=_COLOR_MED,  label=f"Phase 2 — résiduel {_THR_GOOD}–{_THR_MED} m"),
            mpatches.Patch(color=_COLOR_BAD,  label=f"Phase 2 — résiduel > {_THR_MED} m"),
        ]

        # ── Style ─────────────────────────────────────────────────────────
        ax.axhline(0, color=_SPINE, linewidth=0.6, linestyle="--", zorder=1)
        ax.axvline(0, color=_SPINE, linewidth=0.6, linestyle="--", zorder=1)
        ax.set_xlabel("Latéral Y (m)  [← gauche chef  |  droite chef →]", color="white")
        ax.set_ylabel("Profondeur X (m)  [bas=avant  |  haut=arrière]",   color="white")
        ax.grid(linestyle="--", alpha=0.2, color=_SPINE)
        _style_ax(ax)

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles + p2_legend,
            labels  + [p.get_label() for p in p2_legend],
            loc="lower right", fontsize=7.5, framealpha=0.3,
            labelcolor="white", facecolor=_BG_PANEL, edgecolor=_SPINE,
        )

        plt.tight_layout()
        self._savefig(fig, save_path)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. Bar chart des résiduels de trilatération
# ══════════════════════════════════════════════════════════════════════════════

class TrilaterationResidualPlot(_BaseScenePlot):
    """
    Bar chart horizontal des résiduels RMS par micro non-dédié.

    Trié par résiduel croissant pour identifier rapidement les localisations
    fiables vs. dégradées.  Chaque barre indique le ratio ancres_utilisées /
    coverage total.
    """

    def plot(self, save_path: str | None = None) -> plt.Figure:
        nded = self.scene.non_dedicated

        if not nded:
            fig, ax = plt.subplots(figsize=(8, 4))
            fig.patch.set_facecolor(_BG_DARK)
            ax.set_facecolor(_BG_PANEL)
            ax.text(
                0.5, 0.5, "Aucun micro non-dédié localisé.",
                ha="center", va="center", color="white", fontsize=12,
                transform=ax.transAxes,
            )
            _style_ax(ax)
            self._savefig(fig, save_path)
            return fig

        sorted_items = sorted(nded.items(), key=lambda kv: kv[1].residual_rms)
        mic_ids  = [kv[0]                      for kv in sorted_items]
        rms_vals = [kv[1].residual_rms         for kv in sorted_items]
        n_used   = [kv[1].n_used               for kv in sorted_items]
        coverage = [kv[1].coverage             for kv in sorted_items]
        colors   = [_residual_color(r)         for r in rms_vals]

        n = len(mic_ids)
        fig_h = max(4.0, n * 0.55 + 2.0)
        fig, ax = plt.subplots(figsize=(10, fig_h))
        fig.patch.set_facecolor(_BG_DARK)
        ax.set_facecolor(_BG_PANEL)
        fig.suptitle(
            "Résiduels de trilatération — Phase 2  (ordre croissant)",
            fontsize=12, fontweight="bold", color="white",
        )

        y_pos = np.arange(n)
        bars  = ax.barh(
            y_pos, rms_vals,
            color=colors, edgecolor="white", linewidth=0.4,
            height=0.6, zorder=5,
        )

        x_max = max(rms_vals) if rms_vals else 1.0
        for bar, nu, cov, rms in zip(bars, n_used, coverage, rms_vals):
            ax.text(
                min(rms + 0.02, x_max * 1.02),
                bar.get_y() + bar.get_height() / 2,
                f"{nu}/{cov} ancres",
                va="center", ha="left", fontsize=7.5, color="white",
            )

        ax.axvline(
            _THR_GOOD, color=_COLOR_GOOD, linestyle="--", linewidth=1.2,
            label=f"{_THR_GOOD} m — fiable",
        )
        ax.axvline(
            _THR_MED, color=_COLOR_MED, linestyle="--", linewidth=1.2,
            label=f"{_THR_MED} m — limite",
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(mic_ids, fontsize=9, color="white")
        ax.set_xlabel("Résiduel RMS (m)", color="white")
        ax.set_xlim(left=0, right=x_max * 1.25)
        ax.grid(axis="x", linestyle="--", alpha=0.3, color=_SPINE)
        _style_ax(ax)
        ax.legend(
            labelcolor="white", facecolor=_BG_PANEL, edgecolor=_SPINE,
            fontsize=9, framealpha=0.3,
        )

        plt.tight_layout()
        self._savefig(fig, save_path)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# Façade
# ══════════════════════════════════════════════════════════════════════════════

class FullSceneVisualizer:
    """
    Point d'entrée unique pour les visualisations Phase 2.

    Attributs
    ---------
    top_view       : FullSceneTopView
    residual_chart : TrilaterationResidualPlot

    Exemple
    -------
        viz = loc2.visualizer
        viz.top_view.plot()
        viz.residual_chart.plot()
        viz.plot_all(save_dir="figures/triangulation/")
    """

    def __init__(self, scene: FullScenePositions) -> None:
        self.top_view       = FullSceneTopView(scene)
        self.residual_chart = TrilaterationResidualPlot(scene)

    def plot_all(self, save_dir: str | None = None) -> None:
        from pathlib import Path

        def _path(name: str) -> str | None:
            return str(Path(save_dir) / name) if save_dir else None

        self.top_view.plot(save_path=_path("5_full_scene_top_view.png"))
        self.residual_chart.plot(save_path=_path("6_trilateration_residuals.png"))
        plt.show()
