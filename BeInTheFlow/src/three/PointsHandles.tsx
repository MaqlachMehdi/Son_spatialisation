import type { TrajectoryDTO } from "../types";
import { sofaToThree } from "../utils/sofaCoords";
import { useTrajectoryStore } from "../store/trajectoryStore";
import { useDraggablePoint } from "./useDraggablePoint";

interface WaypointHandleProps {
  trajectory: TrajectoryDTO;
  index: number;
  onDragStateChange: (dragging: boolean) => void;
}

function WaypointHandle({ trajectory, index, onDragStateChange }: WaypointHandleProps) {
  const updateTrajectory = useTrajectoryStore((s) => s.updateTrajectory);
  const wp = trajectory.points[index];
  const pos = sofaToThree(wp.azimuth, wp.elevation, wp.distance);

  const { handleGroundDown, handleElevationDown } = useDraggablePoint(
    pos,
    (sofa) => {
      const points = trajectory.points.map((p, i) => (i === index ? sofa : p));
      updateTrajectory(trajectory.id, { points });
    },
    onDragStateChange,
  );

  return (
    <group position={[pos.x, pos.y, pos.z]}>
      <mesh onPointerDown={handleGroundDown}>
        <sphereGeometry args={[0.08, 20, 20]} />
        <meshBasicMaterial color="#f2b100" />
      </mesh>
      <group onPointerDown={handleElevationDown}>
        <mesh position={[0, 0.24, 0]} visible={false}>
          <cylinderGeometry args={[0.08, 0.08, 0.4, 8]} />
          <meshBasicMaterial />
        </mesh>
        <mesh position={[0, 0.26, 0]} raycast={() => null}>
          <cylinderGeometry args={[0.01, 0.01, 0.2, 8]} />
          <meshBasicMaterial color="#f2b100" />
        </mesh>
        <mesh position={[0, 0.4, 0]} raycast={() => null}>
          <coneGeometry args={[0.035, 0.1, 12]} />
          <meshBasicMaterial color="#f2b100" />
        </mesh>
      </group>
    </group>
  );
}

interface PointsHandlesProps {
  trajectory: TrajectoryDTO;
  onDragStateChange: (dragging: boolean) => void;
}

// Une poignée par point du nuage, chacune déplaçable comme une source
// (sol + élévation) — met à jour ce point précis dans trajectory.points.
export default function PointsHandles({ trajectory, onDragStateChange }: PointsHandlesProps) {
  return (
    <group>
      {trajectory.points.map((_, i) => (
        <WaypointHandle key={i} trajectory={trajectory} index={i} onDragStateChange={onDragStateChange} />
      ))}
    </group>
  );
}
