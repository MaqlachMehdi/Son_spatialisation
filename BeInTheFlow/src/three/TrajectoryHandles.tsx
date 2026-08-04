import { useEffect, useRef } from "react";
import { useThree, type ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import type { TrajectoryDTO } from "../types";
import { sofaToThree, threeToSofa } from "../utils/sofaCoords";
import { useTrajectoryStore } from "../store/trajectoryStore";

interface TrajectoryHandlesProps {
  trajectory: TrajectoryDTO;
  onDragStateChange: (dragging: boolean) => void;
}

// Réutilisés à chaque frame de drag pour éviter des allocations répétées
// (même stratégie que SourceNode.tsx).
const dragPlane = new THREE.Plane();
const dragSphere = new THREE.Sphere();
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const intersection = new THREE.Vector3();
const UP = new THREE.Vector3(0, 1, 0);

type DragMode = "none" | "offset-ground" | "offset-elevation" | "axis";

// Poignées 3D d'une trajectoire circulaire : une pour décaler son centre
// (comme une source, drag libre au sol + flèche d'élévation), une pour
// incliner son axe (drag contraint à la sphère de rayon = trajectory.distance
// centrée sur le centre décalé).
export default function TrajectoryHandles({ trajectory, onDragStateChange }: TrajectoryHandlesProps) {
  const { camera, gl } = useThree();
  const updateTrajectory = useTrajectoryStore((s) => s.updateTrajectory);
  const dragMode = useRef<DragMode>("none");
  const startWorldPos = useRef(new THREE.Vector3());
  const axisCenter = useRef(new THREE.Vector3());

  const offsetPos = sofaToThree(trajectory.offsetAzimuth, trajectory.offsetElevation, trajectory.offsetDistance);
  const axisDir = sofaToThree(trajectory.axisAzimuth, trajectory.axisElevation, 1);
  const axisPos = {
    x: axisDir.x * trajectory.distance + offsetPos.x,
    y: axisDir.y * trajectory.distance + offsetPos.y,
    z: axisDir.z * trajectory.distance + offsetPos.z,
  };

  const applyOffset = (point: THREE.Vector3) => {
    const { azimuth, elevation, distance } = threeToSofa(point);
    updateTrajectory(trajectory.id, {
      offsetAzimuth: Math.round(azimuth * 10) / 10,
      offsetElevation: Math.round(elevation * 10) / 10,
      offsetDistance: Math.round(distance * 100) / 100,
    });
  };

  const applyAxis = (point: THREE.Vector3) => {
    const relative = point.clone().sub(axisCenter.current);
    const { azimuth, elevation } = threeToSofa(relative);
    updateTrajectory(trajectory.id, {
      axisAzimuth: Math.round(azimuth * 10) / 10,
      axisElevation: Math.round(elevation * 10) / 10,
    });
  };

  const handleOffsetGroundDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    dragPlane.setFromNormalAndCoplanarPoint(UP, new THREE.Vector3(offsetPos.x, offsetPos.y, offsetPos.z));
    dragMode.current = "offset-ground";
    onDragStateChange(true);
    gl.domElement.setPointerCapture(e.pointerId);
  };

  const handleOffsetElevationDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    startWorldPos.current.set(offsetPos.x, offsetPos.y, offsetPos.z);
    dragMode.current = "offset-elevation";
    onDragStateChange(true);
    gl.domElement.setPointerCapture(e.pointerId);
  };

  const handleAxisDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    axisCenter.current.set(offsetPos.x, offsetPos.y, offsetPos.z);
    dragSphere.center.copy(axisCenter.current);
    dragSphere.radius = trajectory.distance;
    dragMode.current = "axis";
    onDragStateChange(true);
    gl.domElement.setPointerCapture(e.pointerId);
  };

  useEffect(() => {
    const dom = gl.domElement;

    const handleMove = (e: PointerEvent) => {
      if (dragMode.current === "none") return;

      const rect = dom.getBoundingClientRect();
      ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(ndc, camera);

      if (dragMode.current === "offset-ground") {
        if (raycaster.ray.intersectPlane(dragPlane, intersection)) {
          applyOffset(intersection);
        }
        return;
      }

      if (dragMode.current === "offset-elevation") {
        // Point le plus proche entre la droite verticale passant par le
        // centre et le rayon caméra->souris (cf. SourceNode.tsx).
        const p0 = startWorldPos.current;
        const rOrigin = raycaster.ray.origin;
        const rDir = raycaster.ray.direction;
        const w0y = p0.y - rOrigin.y;
        const w0x = p0.x - rOrigin.x;
        const w0z = p0.z - rOrigin.z;
        const b = rDir.y;
        const d = w0y;
        const e_ = rDir.x * w0x + rDir.y * w0y + rDir.z * w0z;
        const denom = 1 - b * b;
        if (Math.abs(denom) > 1e-6) {
          const t = (b * e_ - d) / denom;
          intersection.set(p0.x, p0.y + t, p0.z);
          applyOffset(intersection);
        }
        return;
      }

      if (dragMode.current === "axis") {
        if (raycaster.ray.intersectSphere(dragSphere, intersection)) {
          applyAxis(intersection);
        }
      }
    };

    const handleUp = (e: PointerEvent) => {
      if (dragMode.current === "none") return;
      dragMode.current = "none";
      onDragStateChange(false);
      dom.releasePointerCapture(e.pointerId);
    };

    dom.addEventListener("pointermove", handleMove);
    dom.addEventListener("pointerup", handleUp);
    return () => {
      dom.removeEventListener("pointermove", handleMove);
      dom.removeEventListener("pointerup", handleUp);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [camera, gl, trajectory.id, updateTrajectory, onDragStateChange]);

  return (
    <group>
      {/* Poignée de centre : décale le cercle, comme une source (sol + élévation). */}
      <group position={[offsetPos.x, offsetPos.y, offsetPos.z]}>
        <mesh onPointerDown={handleOffsetGroundDown}>
          <sphereGeometry args={[0.08, 20, 20]} />
          <meshBasicMaterial color="#f2b100" />
        </mesh>
        <group onPointerDown={handleOffsetElevationDown}>
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

      {/* Poignée d'axe : incline le cercle autour de son centre. */}
      <mesh position={[axisPos.x, axisPos.y, axisPos.z]} onPointerDown={handleAxisDown}>
        <sphereGeometry args={[0.09, 20, 20]} />
        <meshBasicMaterial color="#7fd6ff" />
      </mesh>
    </group>
  );
}
