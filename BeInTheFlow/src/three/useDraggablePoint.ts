import { useEffect, useRef } from "react";
import { useThree, type ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { threeToSofa, type Vec3 } from "../utils/sofaCoords";

interface SofaValue {
  azimuth: number;
  elevation: number;
  distance: number;
}

type DragMode = "none" | "ground" | "elevation";

// Drag libre d'un point dans l'espace (sol horizontal + flèche d'élévation),
// exactement comme une source (cf. SourceNode.tsx). Réutilisé pour toute
// poignée 3D qui représente une position complète (azimut+élévation+distance).
export function useDraggablePoint(
  worldPos: Vec3,
  onChange: (v: SofaValue) => void,
  onDragStateChange: (dragging: boolean) => void,
) {
  const { camera, gl } = useThree();
  const dragMode = useRef<DragMode>("none");
  const startWorldPos = useRef(new THREE.Vector3());
  const dragPlane = useRef(new THREE.Plane()).current;
  const raycaster = useRef(new THREE.Raycaster()).current;
  const ndc = useRef(new THREE.Vector2()).current;
  const intersection = useRef(new THREE.Vector3()).current;
  const UP = useRef(new THREE.Vector3(0, 1, 0)).current;

  const apply = (point: THREE.Vector3) => {
    const { azimuth, elevation, distance } = threeToSofa(point);
    onChange({
      azimuth: Math.round(azimuth * 10) / 10,
      elevation: Math.round(elevation * 10) / 10,
      distance: Math.round(distance * 100) / 100,
    });
  };

  const handleGroundDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    dragPlane.setFromNormalAndCoplanarPoint(UP, new THREE.Vector3(worldPos.x, worldPos.y, worldPos.z));
    dragMode.current = "ground";
    onDragStateChange(true);
    gl.domElement.setPointerCapture(e.pointerId);
  };

  const handleElevationDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    startWorldPos.current.set(worldPos.x, worldPos.y, worldPos.z);
    dragMode.current = "elevation";
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

      if (dragMode.current === "ground") {
        if (raycaster.ray.intersectPlane(dragPlane, intersection)) {
          apply(intersection);
        }
        return;
      }

      // "elevation" : point le plus proche entre la droite verticale passant
      // par le point de départ et le rayon caméra->souris.
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
        apply(intersection);
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
  }, [camera, gl, onChange, onDragStateChange]);

  return { handleGroundDown, handleElevationDown };
}
