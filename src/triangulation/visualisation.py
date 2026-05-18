"""
visualisation.py
----------------
Visualisations pour le pipeline de localisation de microphones.

Usage
-----
    loc = OrchestraLocalizer.from_json(...)
    pos = loc.run()
    viz = loc.visualizer

    viz.mic_3d.plot()
    viz.eigenspectrum.plot()
    viz.distance_heatmap.plot()
    viz.plot_all(save_dir="figures/")
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  — active la projection 3D

from .MicPositions import MicPositions
from .ClassicalMDS import ClassicalMDS
from .DistanceMatrix import DistanceMatrix

# ── Palette sombre cohérente avec le reste du projet ──────────────────────────
_BG_DARK  = "#12121f"
_BG_PANEL = "#1a1a2e"
_SPINE    = "#444466"

_FAMILY_COLORS = {
    "strings":    "#4C9BE8",
    "woodwinds":  "#56C272",
    "brass":      "#E8A23C",
    "percussion": "#E85C5C",
}
_FAMILIES = {
    "strings":    {"Vln_1", "Vln_2", "Vla", "Vcl", "DBass_1", "DBass_2"},
    "woodwinds":  {"Flute", "Oboe", "Clarinet", "Bassoon"},
    "brass":      {"Horns", "Trumpets", "Trombones", "Tuba"},
    "percussion": {"Timpani", "Bass_Drum", "Harp"},
}


def _family_color(instrument: str) -> str:
    for family, members in _FAMILIES.items():
        if instrument in members:
            return _FAMILY_COLORS[family]
    return "#AAAAAA"


def _family_of(instrument: str) -> str:
    for family, members in _FAMILIES.items():
        if instrument in members:
            return family
    return "autre"


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(_BG_PANEL)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor(_SPINE)


# ══════════════════════════════════════════════════════════════════════════════
# Classe de base
# ══════════════════════════════════════════════════════════════════════════════

class _BaseMicPlot:
    def __init__(
        self,
        positions: MicPositions,
        mds: ClassicalMDS | None = None,
        distance_matrix: DistanceMatrix | None = None,
    ) -> None:
        self.positions       = positions
        self.mds             = mds
        self.distance_matrix = distance_matrix

    def _savefig(self, fig: plt.Figure, path: str | None) -> None:
        if path:
            fig.savefig(path, dpi=150, bbox_inches="tight")


# ══════════════════════════════════════════════════════════════════════════════
# 1. Scatter 3D des positions
# ══════════════════════════════════════════════════════════════════════════════

class MicPlot3D(_BaseMicPlot):
    """
    Nuage de points 3D des micros dédiés.

    Chaque micro est coloré par famille instrumentale.
    Une annotation (mic_id + instrument) est affichée à côté de chaque point.
    """

    def plot(self, save_path: str | None = None) -> plt.Figure:
        pos   = self.positions.positions
        mics  = self.positions.mic_ids
        instr = self.positions.instrument_labels
        ev    = self.positions.explained_variance_ratio * 100

        fig = plt.figure(figsize=(13, 9))
        fig.patch.set_facecolor(_BG_DARK)
        ax = fig.add_subplot(111, projection="3d")
        ax.set_facecolor(_BG_PANEL)
        fig.suptitle(
            f"Positions 3D des microphones dédiés — variance expliquée 3D : {ev:.1f}%",
            fontsize=12, fontweight="bold", color="white",
        )

        # --- points par famille (pour légende propre) ---
        plotted: set[str] = set()
        for i, (mic_id, instrument) in enumerate(zip(mics, instr)):
            color  = _family_color(instrument)
            family = _family_of(instrument)
            label  = family.capitalize() if family not in plotted else None
            plotted.add(family)

            ax.scatter(
                pos[i, 0], pos[i, 1], pos[i, 2],
                c=color, s=130, edgecolors="white", linewidths=0.6,
                label=label, depthshade=True, zorder=5,
            )
            ax.text(
                pos[i, 0], pos[i, 1], pos[i, 2] + 0.08,
                f"{mic_id}\n{instrument}",
                fontsize=6, ha="center", va="bottom", color="white",
                path_effects=[pe.withStroke(linewidth=2, foreground=color)],
            )

        # --- centroïde ---
        centroid = pos.mean(axis=0)
        ax.scatter(*centroid, c="white", s=80, marker="+", zorder=10, label="centroïde")

        ax.set_xlabel("X (m)", color="white")
        ax.set_ylabel("Y (m)", color="white")
        ax.set_zlabel("Z (m)", color="white")
        ax.tick_params(colors="white")
        ax.legend(
            loc="upper left", fontsize=9, framealpha=0.3,
            labelcolor="white", facecolor=_BG_PANEL, edgecolor=_SPINE,
        )

        plt.tight_layout()
        self._savefig(fig, save_path)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# 2. Spectre des valeurs propres
# ══════════════════════════════════════════════════════════════════════════════

class EigenSpectrumPlot(_BaseMicPlot):
    """
    Deux panneaux :
    - Gauche : bar chart de toutes les valeurs propres (signal 3D en bleu, bruit en gris)
    - Droite : variance cumulée (%)
    """

    def plot(self, save_path: str | None = None) -> plt.Figure:
        if self.mds is None or self.mds.Lambda is None:
            raise RuntimeError(
                "ClassicalMDS avec Lambda calculé requis pour EigenSpectrumPlot."
            )

        Lambda   = self.mds.Lambda
        n        = len(Lambda)
        total    = float(Lambda.sum()) + 1e-12
        ratio_3d = Lambda[:3].sum() / total * 100

        colors = [_FAMILY_COLORS["strings"]] * 3 + ["#444466"] * (n - 3)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor(_BG_DARK)
        fig.suptitle(
            "Spectre des valeurs propres de B (matrice de Gram)",
            fontsize=12, fontweight="bold", color="white",
        )

        # --- Valeurs propres ---
        ax1.bar(range(1, n + 1), Lambda, color=colors, edgecolor="white", linewidth=0.4)
        ax1.axvline(3.5, color="#E85C5C", linestyle="--", linewidth=1.5, label="Coupure 3D")
        ax1.set_xlabel("Rang k")
        ax1.set_ylabel("λₖ")
        ax1.set_title("Valeurs propres")
        ax1.legend(labelcolor="white", facecolor=_BG_PANEL, edgecolor=_SPINE)
        ax1.grid(axis="y", linestyle="--", alpha=0.3, color=_SPINE)
        ax1.text(
            2, Lambda[0] * 0.85,
            f"{ratio_3d:.1f}% de\nvariance\nexpliquée",
            ha="center", color=_FAMILY_COLORS["strings"], fontsize=9, fontweight="bold",
        )
        _style_ax(ax1)

        # --- Variance cumulée ---
        cumvar = np.cumsum(Lambda) / total * 100
        ax2.plot(range(1, n + 1), cumvar, "o-", color=_FAMILY_COLORS["woodwinds"], markersize=5)
        ax2.axhline(99, color=_FAMILY_COLORS["brass"], linestyle="--", linewidth=1.2, label="99%")
        ax2.axvline(3.5, color="#E85C5C", linestyle="--", linewidth=1.5, label="Coupure 3D")
        ax2.set_xlabel("Rang k")
        ax2.set_ylabel("Variance cumulée (%)")
        ax2.set_title("Variance cumulée")
        ax2.set_ylim([0, 101])
        ax2.legend(labelcolor="white", facecolor=_BG_PANEL, edgecolor=_SPINE)
        ax2.grid(linestyle="--", alpha=0.3, color=_SPINE)
        _style_ax(ax2)

        plt.tight_layout()
        self._savefig(fig, save_path)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# 3. Heatmap de la matrice de distances
# ══════════════════════════════════════════════════════════════════════════════

class DistanceHeatmap(_BaseMicPlot):
    """
    Heatmap annotée de la matrice 17×17 en mètres.
    Les axes sont étiquetés par nom d'instrument.
    """

    def plot(self, save_path: str | None = None) -> plt.Figure:
        if self.distance_matrix is None:
            raise RuntimeError(
                "DistanceMatrix requis pour DistanceHeatmap."
            )

        D      = self.distance_matrix.matrix
        labels = self.distance_matrix.instrument_labels
        n      = len(labels)

        fig, ax = plt.subplots(figsize=(13, 11))
        fig.patch.set_facecolor(_BG_DARK)
        fig.suptitle(
            "Matrice de distances entre microphones dédiés (mètres)",
            fontsize=12, fontweight="bold", color="white",
        )

        im   = ax.imshow(D, cmap="plasma", aspect="equal")
        cbar = fig.colorbar(im, ax=ax, label="Distance (m)", fraction=0.046, pad=0.04)
        cbar.ax.yaxis.label.set_color("white")
        cbar.ax.tick_params(colors="white")

        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8, color="white")
        ax.set_yticklabels(labels, fontsize=8, color="white")

        threshold = D.max() * 0.55
        for i in range(n):
            for j in range(n):
                txt_color = "white" if D[i, j] < threshold else "black"
                ax.text(
                    j, i, f"{D[i, j]:.1f}",
                    ha="center", va="center", fontsize=5.5, color=txt_color,
                )

        ax.set_facecolor(_BG_PANEL)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor(_SPINE)

        plt.tight_layout()
        self._savefig(fig, save_path)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# 4. Vue de dessus (plan XY : profondeur × latéral)
# ══════════════════════════════════════════════════════════════════════════════

class TopViewPlot(_BaseMicPlot):
    """
    Vue de dessus du plateau orchestral : plan XY après orientation sémantique.

    Axes
    ----
    X (horizontal) : profondeur — négatif = avant-scène (cordes),
                                  positif = arrière-scène (cuivres/percus)
    Y (vertical)   : latéral   — positif = gauche chef (Vln_1),
                                  négatif = droite chef (Vcl)

    Chaque point est coloré par famille.  Les centroïdes de familles sont
    affichés en plus petit avec un contour pointillé.  Les numéros de micro
    sont indiqués dans des bulles colorées pour limiter l'encombrement.
    """

    def plot(self, save_path: str | None = None) -> plt.Figure:
        pos   = self.positions.positions          # (n, 3)
        mics  = self.positions.mic_ids
        instr = self.positions.instrument_labels

        fig, ax = plt.subplots(figsize=(13, 10))
        fig.patch.set_facecolor(_BG_DARK)
        ax.set_facecolor(_BG_PANEL)
        fig.suptitle(
            "Vue de dessus — plan scène orchestrale"
            "  (gauche=Vln_1, droite=Vcl, bas=avant-scène, haut=arrière)",
            fontsize=12, fontweight="bold", color="white",
        )

        # ── Points par instrument ─────────────────────────────────────────────
        # Convention d'affichage :
        #   axe X du plot = Y_MDS = latéral  (inversé : positif → gauche chef)
        #   axe Y du plot = X_MDS = profondeur (négatif=avant=bas, positif=arrière=haut)
        plotted: set[str] = set()
        for i, (mic_id, instrument) in enumerate(zip(mics, instr)):
            color  = _family_color(instrument)
            family = _family_of(instrument)
            label  = family.capitalize() if family not in plotted else None
            plotted.add(family)

            ax.scatter(
                pos[i, 1], pos[i, 0],          # lat → X plot, depth → Y plot
                c=color, s=160, edgecolors="white", linewidths=0.7,
                label=label, zorder=5,
            )
            num = mic_id.split("_")[1]
            ax.annotate(
                num,
                xy=(pos[i, 1], pos[i, 0]),
                xytext=(0, 0), textcoords="offset points",
                ha="center", va="center",
                fontsize=5.5, fontweight="bold", color="white", zorder=6,
            )
            ax.annotate(
                instrument,
                xy=(pos[i, 1], pos[i, 0]),
                xytext=(6, 6), textcoords="offset points",
                ha="left", va="bottom",
                fontsize=6.5, color=color,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=_BG_PANEL)],
                zorder=7,
            )

        # ── Centroïdes de familles ────────────────────────────────────────────
        for family, members in _FAMILIES.items():
            idxs = [i for i, ins in enumerate(instr) if ins in members]
            if not idxs:
                continue
            cx = float(pos[idxs, 1].mean())    # latéral → X plot
            cy = float(pos[idxs, 0].mean())    # profondeur → Y plot
            color = _FAMILY_COLORS[family]
            ax.scatter(cx, cy, c=color, s=300, marker="D",
                       edgecolors="white", linewidths=1.0,
                       alpha=0.35, zorder=3)
            ax.annotate(
                family.capitalize(),
                xy=(cx, cy), xytext=(0, -14), textcoords="offset points",
                ha="center", fontsize=7, color=color, alpha=0.7,
                path_effects=[pe.withStroke(linewidth=1.5, foreground=_BG_PANEL)],
            )

        # ── Centroïde global ──────────────────────────────────────────────────
        ax.scatter(0, 0, c="white", s=90, marker="+", zorder=10, label="centroïde")

        # ── Limites + inversion de l'axe latéral ─────────────────────────────
        lat_min, lat_max = pos[:, 1].min(), pos[:, 1].max()   # Y_MDS → X plot
        dep_min, dep_max = pos[:, 0].min(), pos[:, 0].max()   # X_MDS → Y plot
        margin = 1.2
        ax.set_xlim(lat_min - margin, lat_max + margin)
        ax.set_ylim(dep_min - margin, dep_max + margin)
        # Inversion : valeurs lat positives (Vln_1) passent à gauche du plot
        ax.invert_xaxis()

        # ── Annotations directionnelles ───────────────────────────────────────
        # Avant-scène : profondeur négative → bas du plot
        ax.annotate("", xy=(0, dep_min - 0.6), xytext=(0, dep_min),
                    arrowprops=dict(arrowstyle="->", color="#888899", lw=1.2))
        ax.text(0, dep_min - 0.75, "Avant-scène\n(cordes)",
                ha="center", va="top", fontsize=7, color="#888899")
        # Arrière-scène : profondeur positive → haut du plot
        ax.annotate("", xy=(0, dep_max + 0.6), xytext=(0, dep_max),
                    arrowprops=dict(arrowstyle="->", color="#888899", lw=1.2))
        ax.text(0, dep_max + 0.75, "Arrière\n(cuivres/percus)",
                ha="center", va="bottom", fontsize=7, color="#888899")
        # Gauche chef : lat positif → après inversion, côté gauche du plot (x_max en data)
        ax.text(lat_max + 0.6, 0, "Gauche chef\n(Vln_1)",
                ha="center", va="center", fontsize=7, color="#888899")
        # Droite chef : lat négatif → après inversion, côté droit du plot (x_min en data)
        ax.text(lat_min - 0.6, 0, "Droite chef\n(Vcl)",
                ha="center", va="center", fontsize=7, color="#888899")

        # ── Style ─────────────────────────────────────────────────────────────
        ax.axhline(0, color=_SPINE, linewidth=0.6, linestyle="--", zorder=1)
        ax.axvline(0, color=_SPINE, linewidth=0.6, linestyle="--", zorder=1)
        ax.set_xlabel("Latéral Y (m)  [← gauche chef  |  droite chef →]", color="white")
        ax.set_ylabel("Profondeur X (m)  [bas=avant  |  haut=arrière]",   color="white")
        ax.grid(linestyle="--", alpha=0.2, color=_SPINE)
        _style_ax(ax)
        ax.legend(
            loc="lower right", fontsize=8, framealpha=0.3,
            labelcolor="white", facecolor=_BG_PANEL, edgecolor=_SPINE,
        )

        plt.tight_layout()
        self._savefig(fig, save_path)
        return fig


# ══════════════════════════════════════════════════════════════════════════════
# Façade
# ══════════════════════════════════════════════════════════════════════════════

class MicVisualizer:
    """
    Point d'entrée unique pour toutes les visualisations de localisation.

    Exemple
    -------
        loc = OrchestraLocalizer.from_json(...)
        pos = loc.run()
        viz = loc.visualizer

        viz.mic_3d.plot()
        viz.top_view.plot()
        viz.eigenspectrum.plot()
        viz.distance_heatmap.plot()
        viz.plot_all(save_dir="figures/triangulation/")
    """

    def __init__(
        self,
        positions: MicPositions,
        mds: ClassicalMDS | None = None,
        distance_matrix: DistanceMatrix | None = None,
    ) -> None:
        self.mic_3d           = MicPlot3D(positions, mds, distance_matrix)
        self.top_view         = TopViewPlot(positions, mds, distance_matrix)
        self.eigenspectrum    = EigenSpectrumPlot(positions, mds, distance_matrix)
        self.distance_heatmap = DistanceHeatmap(positions, mds, distance_matrix)

    def plot_all(self, save_dir: str | None = None) -> None:
        from pathlib import Path

        def _path(name: str) -> str | None:
            return str(Path(save_dir) / name) if save_dir else None

        self.mic_3d.plot(save_path=_path("1_mic_positions_3d.png"))
        self.top_view.plot(save_path=_path("2_top_view.png"))
        self.eigenspectrum.plot(save_path=_path("3_eigenspectrum.png"))
        self.distance_heatmap.plot(save_path=_path("4_distance_heatmap.png"))
        plt.show()
