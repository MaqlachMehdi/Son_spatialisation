from __future__ import annotations

import numpy as np

from .embedding_space import EmbeddingSpace


def _umap_2d(
    vectors:    np.ndarray,
    n_neighbors: int   = 15,
    min_dist:    float = 0.3,
    random_state: int  = 42,
) -> np.ndarray:
    """Réduit un ensemble de vecteurs en 2D via UMAP."""
    try:
        import umap
    except ImportError as e:
        raise ImportError(
            'umap-learn non installé. Lance : pip install umap-learn'
        ) from e

    reducer = umap.UMAP(
        n_components  = 2,
        n_neighbors   = n_neighbors,
        min_dist      = min_dist,
        random_state  = random_state,
        metric        = 'cosine',
    )
    return reducer.fit_transform(vectors)


class EmbeddingVisualizer:
    """Visualisation interactive de l'espace latent via UMAP + Plotly.

    Usage
    -----
    space = EmbeddingSpace.load('checkpoints/embeddings.npz')
    viz   = EmbeddingVisualizer(space)

    viz.plot_alignment(split='test')        # z_ear + z_hrtf, paires reliées
    viz.plot_subjects(split='test')         # z_ear uniquement
    viz.plot_augmentations('H10')           # un sujet + ses versions augmentées
    """

    def __init__(self, space: EmbeddingSpace) -> None:
        self._space = space

    # ------------------------------------------------------------------
    # Vue 1 — Alignement des paires (vue principale)
    # ------------------------------------------------------------------

    def plot_alignment(
        self,
        split:       str   = 'test',
        n_neighbors: int   = 15,
        min_dist:    float = 0.3,
    ) -> None:
        """UMAP sur z_ear + z_hrtf combinés. Relie les paires du même sujet."""
        import plotly.graph_objects as go

        sub      = self._space.filter(split)
        N        = len(sub.subject_ids)
        combined = np.concatenate([sub.z_ear, sub.z_hrtf], axis=0)  # [2N, D]

        print(f'  UMAP en cours sur {2*N} vecteurs...')
        coords = _umap_2d(combined, n_neighbors=n_neighbors, min_dist=min_dist)

        ear_xy  = coords[:N]    # [N, 2]
        hrtf_xy = coords[N:]    # [N, 2]

        colors = _palette(sub.subject_ids)
        fig    = go.Figure()

        # Traits reliant les paires du même sujet
        for i in range(N):
            fig.add_trace(go.Scatter(
                x=[ear_xy[i, 0], hrtf_xy[i, 0]],
                y=[ear_xy[i, 1], hrtf_xy[i, 1]],
                mode='lines',
                line=dict(color=colors[i], width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip',
            ))

        # Points z_ear (cercles)
        fig.add_trace(go.Scatter(
            x=ear_xy[:, 0], y=ear_xy[:, 1],
            mode='markers',
            name='z_ear',
            marker=dict(symbol='circle', size=10,
                        color=colors, line=dict(width=1, color='white')),
            text=sub.subject_ids,
            hovertemplate='%{text} — ear<extra></extra>',
        ))

        # Points z_hrtf (carrés)
        fig.add_trace(go.Scatter(
            x=hrtf_xy[:, 0], y=hrtf_xy[:, 1],
            mode='markers',
            name='z_hrtf',
            marker=dict(symbol='square', size=10,
                        color=colors, line=dict(width=1, color='white')),
            text=sub.subject_ids,
            hovertemplate='%{text} — hrtf<extra></extra>',
        ))

        fig.update_layout(
            title=f'Alignement z_ear / z_hrtf — split {split}',
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            width=850, height=650,
        )
        fig.show()

    # ------------------------------------------------------------------
    # Vue 2 — Structure par sujets (une seule modalité)
    # ------------------------------------------------------------------

    def plot_subjects(
        self,
        modality:    str   = 'ear',
        split:       str   = 'test',
        n_neighbors: int   = 15,
        min_dist:    float = 0.3,
    ) -> None:
        """UMAP sur une seule modalité, colorée par sujet."""
        import plotly.express as px
        import pandas as pd

        sub      = self._space.filter(split)
        vectors  = sub.z_ear if modality == 'ear' else sub.z_hrtf

        print(f'  UMAP en cours sur {len(sub.subject_ids)} vecteurs z_{modality}...')
        coords = _umap_2d(vectors, n_neighbors=n_neighbors, min_dist=min_dist)

        df = pd.DataFrame({
            'x':          coords[:, 0],
            'y':          coords[:, 1],
            'sujet':      sub.subject_ids,
            'base_sujet': [s.split('_')[0] for s in sub.subject_ids],
        })

        fig = px.scatter(
            df, x='x', y='y',
            color='base_sujet',
            hover_data=['sujet'],
            title=f'z_{modality} par sujet — split {split}',
            labels={'x': 'UMAP 1', 'y': 'UMAP 2'},
            width=800, height=600,
        )
        fig.update_traces(marker=dict(size=10))
        fig.show()

    # ------------------------------------------------------------------
    # Vue 3 — Zoom sur un sujet et ses augmentations
    # ------------------------------------------------------------------

    def plot_augmentations(
        self,
        base_subject_id: str,
        n_neighbors:     int   = 15,
        min_dist:        float = 0.1,
    ) -> None:
        """UMAP sur un sujet + toutes ses versions augmentées (mirror, aug1...)."""
        import plotly.graph_objects as go

        # Filtrer : sujet de base + ses variantes
        mask = [
            sid == base_subject_id or sid.startswith(f'{base_subject_id}_')
            for sid in self._space.subject_ids
        ]
        idx  = np.where(mask)[0]

        if len(idx) == 0:
            print(f'  Sujet "{base_subject_id}" introuvable.')
            return

        sub_ids  = [self._space.subject_ids[i] for i in idx]
        combined = np.concatenate([
            self._space.z_ear[idx],
            self._space.z_hrtf[idx],
        ], axis=0)

        print(f'  UMAP sur {len(idx)} variantes de {base_subject_id}...')
        coords  = _umap_2d(combined, n_neighbors=max(5, len(idx)-1),
                           min_dist=min_dist)
        N       = len(idx)
        ear_xy  = coords[:N]
        hrtf_xy = coords[N:]

        symbols = _augmentation_symbols(sub_ids)
        fig     = go.Figure()

        for i, sid in enumerate(sub_ids):
            fig.add_trace(go.Scatter(
                x=[ear_xy[i, 0]], y=[ear_xy[i, 1]],
                mode='markers+text',
                name=f'{sid} ear',
                marker=dict(symbol=symbols[i], size=14, color='steelblue'),
                text=[sid], textposition='top center',
                showlegend=True,
            ))
            fig.add_trace(go.Scatter(
                x=[hrtf_xy[i, 0]], y=[hrtf_xy[i, 1]],
                mode='markers',
                name=f'{sid} hrtf',
                marker=dict(symbol='square', size=14, color='tomato'),
                showlegend=True,
            ))
            fig.add_trace(go.Scatter(
                x=[ear_xy[i, 0], hrtf_xy[i, 0]],
                y=[ear_xy[i, 1], hrtf_xy[i, 1]],
                mode='lines',
                line=dict(color='gray', width=1, dash='dot'),
                showlegend=False,
                hoverinfo='skip',
            ))

        fig.update_layout(
            title=f'Augmentations de {base_subject_id}',
            xaxis_title='UMAP 1', yaxis_title='UMAP 2',
            width=750, height=600,
        )
        fig.show()


# ------------------------------------------------------------------
# Utilitaires internes
# ------------------------------------------------------------------

def _palette(subject_ids: list[str]) -> list[str]:
    """Génère une couleur hexadécimale par sujet de base (ignore _mirror, _aug)."""
    import hashlib

    base_ids = [sid.split('_')[0] for sid in subject_ids]
    unique   = sorted(set(base_ids))

    # Palette de couleurs Plotly étendue
    plotly_colors = [
        '#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A',
        '#19D3F3', '#FF6692', '#B6E880', '#FF97FF', '#FECB52',
        '#1F77B4', '#FF7F0E', '#2CA02C', '#D62728', '#9467BD',
        '#8C564B', '#E377C2', '#7F7F7F', '#BCBD22', '#17BECF',
    ]

    color_map = {uid: plotly_colors[i % len(plotly_colors)]
                 for i, uid in enumerate(unique)}
    return [color_map[b] for b in base_ids]


def _augmentation_symbols(subject_ids: list[str]) -> list[str]:
    """Symbole différent selon le type d'augmentation."""
    symbols = []
    for sid in subject_ids:
        if '_mirror' in sid:
            symbols.append('diamond')
        elif '_aug' in sid:
            symbols.append('star')
        else:
            symbols.append('circle')
    return symbols