import type { TrajectoryDTO } from "../types";
import { sofaToThree } from "../utils/sofaCoords";
import { useTrajectoryStore } from "../store/trajectoryStore";
import { useDraggablePoint } from "./useDraggablePoint";
import { useSphereDrag } from "./useSphereDrag";

interface EllipseHandlesProps {
  trajectory: TrajectoryDTO;
  onDragStateChange: (dragging: boolean) => void;
}

function angularDiff(a: number, b: number) {
  return Math.abs((((a - b + 540) % 360) + 360) % 360 - 180);
}

// Poignées 3D d'une trajectoire elliptique : une pour déplacer son centre
// (comme une source), deux contraintes à la sphère (comme l'axe du cercle)
// pour régler l'amplitude en azimut et en élévation.
export default function EllipseHandles({ trajectory, onDragStateChange }: EllipseHandlesProps) {
  const updateTrajectory = useTrajectoryStore((s) => s.updateTrajectory);

  const centerPos = sofaToThree(trajectory.centerAzimuth, trajectory.centerElevation, trajectory.distance);
  const azPos = sofaToThree(
    trajectory.centerAzimuth + trajectory.azAmplitude,
    trajectory.centerElevation,
    trajectory.distance,
  );
  const elPos = sofaToThree(
    trajectory.centerAzimuth,
    trajectory.centerElevation + trajectory.elAmplitude,
    trajectory.distance,
  );

  const { handleGroundDown, handleElevationDown } = useDraggablePoint(
    centerPos,
    ({ azimuth, elevation, distance }) =>
      updateTrajectory(trajectory.id, { centerAzimuth: azimuth, centerElevation: elevation, distance }),
    onDragStateChange,
  );

  const azDrag = useSphereDrag(
    trajectory.distance,
    ({ azimuth }) => {
      const azAmplitude = Math.round(angularDiff(azimuth, trajectory.centerAzimuth) * 10) / 10;
      updateTrajectory(trajectory.id, { azAmplitude });
    },
    onDragStateChange,
  );

  const elDrag = useSphereDrag(
    trajectory.distance,
    ({ elevation }) => {
      const elAmplitude = Math.round(Math.min(90, Math.abs(elevation - trajectory.centerElevation)) * 10) / 10;
      updateTrajectory(trajectory.id, { elAmplitude });
    },
    onDragStateChange,
  );

  return (
    <group>
      {/* Poignée de centre : déplace toute l'ellipse, comme une source. */}
      <group position={[centerPos.x, centerPos.y, centerPos.z]}>
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

      {/* Poignée d'amplitude azimut. */}
      <mesh position={[azPos.x, azPos.y, azPos.z]} onPointerDown={azDrag.handleDown}>
        <sphereGeometry args={[0.09, 20, 20]} />
        <meshBasicMaterial color="#7fd6ff" />
      </mesh>

      {/* Poignée d'amplitude élévation. */}
      <mesh position={[elPos.x, elPos.y, elPos.z]} onPointerDown={elDrag.handleDown}>
        <sphereGeometry args={[0.09, 20, 20]} />
        <meshBasicMaterial color="#8fff8f" />
      </mesh>
    </group>
  );
}
