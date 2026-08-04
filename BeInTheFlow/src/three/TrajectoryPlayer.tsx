import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import * as THREE from "three";
import { useTrajectoryStore } from "../store/trajectoryStore";
import { useSceneStore } from "../store/sceneStore";
import { trajectoryPointAt } from "../utils/trajectorySampling";
import { threeToSofa } from "../utils/sofaCoords";

const BASE_PERIOD_SECONDS = 8; // durée d'une boucle complète à vitesse = 1
const point = new THREE.Vector3();

// Anime les sources associées à la trajectoire en cours de lecture. Lit
// directement le state des stores à chaque frame (plutôt que de s'y
// abonner) pour ne pas re-render ce composant à 60fps.
export default function TrajectoryPlayer() {
  const elapsed = useRef(0);
  const lastPlayingId = useRef<string | null>(null);

  useFrame((_, delta) => {
    const { playingTrajectoryId, trajectories } = useTrajectoryStore.getState();

    if (playingTrajectoryId !== lastPlayingId.current) {
      elapsed.current = 0;
      lastPlayingId.current = playingTrajectoryId;
    }
    if (!playingTrajectoryId) return;

    const trajectory = trajectories.find((t) => t.id === playingTrajectoryId);
    if (!trajectory) return;

    elapsed.current += delta;
    const period = BASE_PERIOD_SECONDS / Math.max(trajectory.speed, 0.01);
    const t = elapsed.current / period;

    const p = trajectoryPointAt(trajectory, t);
    point.set(p.x, p.y, p.z);
    const sofa = threeToSofa(point);
    const azimuth = Math.round(sofa.azimuth * 10) / 10;
    const elevation = Math.round(sofa.elevation * 10) / 10;
    const distance = Math.round(sofa.distance * 100) / 100;

    const { sources, updateSource } = useSceneStore.getState();
    sources
      .filter((s) => s.trajectoryId === trajectory.id)
      .forEach((s) => updateSource(s.id, { azimuth, elevation, distance }));
  });

  return null;
}
