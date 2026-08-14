import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useTrajectoryStore } from "../store/trajectoryStore";
import { useSceneStore } from "../store/sceneStore";
import { trajectoryPointAt } from "../utils/trajectorySampling";
import { threeToSofa } from "../utils/sofaCoords";
import { getAudioElement } from "../utils/audioEngine";
import type { TrajectoryDTO } from "../types";

const BASE_PERIOD_SECONDS = 8; // durée d'une boucle complète à vitesse = 1
const point = new THREE.Vector3();

function moveSourceAlong(
  trajectory: TrajectoryDTO,
  t: number,
  sourceId: string,
  updateSource: (id: string, patch: { azimuth: number; elevation: number; distance: number }) => void,
) {
  const p = trajectoryPointAt(trajectory, t);
  point.set(p.x, p.y, p.z);
  const sofa = threeToSofa(point);
  updateSource(sourceId, {
    azimuth: Math.round(sofa.azimuth * 10) / 10,
    elevation: Math.round(sofa.elevation * 10) / 10,
    distance: Math.round(sofa.distance * 100) / 100,
  });
}

// Anime les sources associées à une trajectoire, selon deux modes
// indépendants (lus directement depuis les stores à chaque frame plutôt que
// par abonnement, pour ne pas re-render ce composant à 60fps) :
//
//  - Aperçu manuel : bouton "Lire la trajectoire" du panneau, horloge
//    interne (elapsed), une seule trajectoire à la fois.
//  - Lecture audio : bouton "Play" principal — toutes les sources ayant une
//    trajectoire la suivent, synchronisées sur audio.currentTime, pour que
//    le son (spatialisation HRTF rendue) et le visuel avancent ensemble.
export default function TrajectoryPlayer() {
  const elapsed = useRef(0);
  const lastPlayingId = useRef<string | null>(null);

  useFrame((_, delta) => {
    const { playingTrajectoryId, trajectories } = useTrajectoryStore.getState();
    const { sources, updateSource } = useSceneStore.getState();

    if (playingTrajectoryId !== lastPlayingId.current) {
      elapsed.current = 0;
      lastPlayingId.current = playingTrajectoryId;
    }
    if (playingTrajectoryId) {
      const trajectory = trajectories.find((t) => t.id === playingTrajectoryId);
      if (trajectory) {
        elapsed.current += delta;
        const period = BASE_PERIOD_SECONDS / Math.max(trajectory.speed, 0.01);
        const t = elapsed.current / period;
        sources
          .filter((s) => s.trajectoryId === trajectory.id)
          .forEach((s) => moveSourceAlong(trajectory, t, s.id, updateSource));
      }
    }

    const audio = getAudioElement();
    if (audio && !audio.paused) {
      for (const source of sources) {
        // Déjà pris en charge par l'aperçu manuel ci-dessus : éviter que les
        // deux modes se disputent la même source le même frame.
        if (!source.trajectoryId || source.trajectoryId === playingTrajectoryId) continue;
        const trajectory = trajectories.find((t) => t.id === source.trajectoryId);
        if (!trajectory) continue;
        const period = BASE_PERIOD_SECONDS / Math.max(trajectory.speed, 0.01);
        const t = audio.currentTime / period;
        moveSourceAlong(trajectory, t, source.id, updateSource);
      }
    }
  });

  return null;
}
