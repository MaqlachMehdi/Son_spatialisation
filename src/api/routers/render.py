from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from scene import DynamicSoundscape, SceneTrajectory, SoundSource, Soundscape

from ..config import SOUND_DIR
from ..schemas import RenderRequest, SourcePayload, TrajectoryPayload

router = APIRouter()

# Résolution temporelle des blocs de convolution dynamique. Les positions
# sont interpolées en continu (HRTFInterpolator, pas de grille figée), donc
# pas besoin de coller à la grille de mesure IRCAM (15°) — juste assez fin
# pour un mouvement audible fluide.
DYNAMIC_SEGMENT_MS = 25.0


def _resolve_path(rel_path: str) -> Path:
    full_path = SOUND_DIR / rel_path
    if not full_path.is_file():
        raise HTTPException(status_code=404, detail=f"Fichier introuvable : {rel_path}")
    return full_path


def _mix_down(mixes: list[np.ndarray]) -> np.ndarray:
    """Zero-pad à la longueur max, somme, normalise (même logique que Soundscape._mix)."""
    max_len = max(m.shape[0] for m in mixes)
    left = np.zeros(max_len, dtype=np.float64)
    right = np.zeros(max_len, dtype=np.float64)
    for m in mixes:
        n = m.shape[0]
        left[:n] += m[:, 0].astype(np.float64)
        right[:n] += m[:, 1].astype(np.float64)
    peak = max(np.max(np.abs(left)), np.max(np.abs(right))) + 1e-10
    return np.stack([left / peak, right / peak], axis=1).astype(np.float32)


@router.post("/render")
def render(request: RenderRequest, http_request: Request) -> StreamingResponse:
    active_sources = [s for s in request.sources if not s.muted]
    if not active_sources:
        raise HTTPException(status_code=400, detail="Aucune source active fournie.")

    interpolator = http_request.app.state.interpolator
    trajectories_by_id = {t.id: t for t in request.trajectories}

    static_sources: list[SourcePayload] = []
    dynamic_sources: list[tuple[SourcePayload, TrajectoryPayload]] = []
    for s in active_sources:
        trajectory = trajectories_by_id.get(s.trajectory_id) if s.trajectory_id else None
        if trajectory is None or (trajectory.type == "points" and not trajectory.points):
            # Pas de trajectoire, ou trajectoire "points" vide (aucun mouvement possible).
            static_sources.append(s)
        else:
            dynamic_sources.append((s, trajectory))

    mixes: list[np.ndarray] = []

    if static_sources:
        soundscape = Soundscape(interpolator)
        for s in static_sources:
            full_path = _resolve_path(s.path)
            soundscape.add_source(
                SoundSource(
                    azimuth=s.azimuth,
                    elevation=s.elevation,
                    distance=s.distance,
                    gain=s.gain,
                    path=str(full_path),
                )
            )
        mixes.append(soundscape.render())

    if dynamic_sources:
        dynamic_scape = DynamicSoundscape(interpolator, segment_ms=DYNAMIC_SEGMENT_MS)
        for s, trajectory in dynamic_sources:
            full_path = _resolve_path(s.path)
            duration_s = sf.info(str(full_path)).duration
            scene_trajectory = SceneTrajectory(trajectory.model_dump(), duration_s=duration_s)
            dynamic_scape.add_source(
                signal=full_path,
                trajectory=scene_trajectory,
                gain=s.gain,
                label=s.label,
            )
        mixes.append(dynamic_scape.render())

    mix = mixes[0] if len(mixes) == 1 else _mix_down(mixes)

    buf = io.BytesIO()
    sf.write(buf, mix, interpolator.sample_rate, format="WAV")
    buf.seek(0)
    return StreamingResponse(buf, media_type="audio/wav")
