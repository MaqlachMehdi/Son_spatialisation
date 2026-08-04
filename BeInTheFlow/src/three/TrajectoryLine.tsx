import { Line } from "@react-three/drei";
import type { TrajectoryDTO } from "../types";
import { sampleTrajectory } from "../utils/trajectorySampling";

interface TrajectoryLineProps {
  trajectory: TrajectoryDTO;
}

// Trace la trajectoire sélectionnée en pointillé autour de la sphère de
// référence. Purement décoratif : ne doit jamais intercepter les
// clics/drags des sources.
export default function TrajectoryLine({ trajectory }: TrajectoryLineProps) {
  const points = sampleTrajectory(trajectory);
  if (points.length < 2) return null;

  return (
    <Line
      points={points.map((p) => [p.x, p.y, p.z])}
      color="#c3073f"
      lineWidth={1.5}
      dashed
      dashSize={0.08}
      gapSize={0.06}
      raycast={() => null}
    />
  );
}
