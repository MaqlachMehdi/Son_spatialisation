import { useEffect, useRef } from "react";
import { useThree, type ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { threeToSofa } from "../utils/sofaCoords";

interface SofaValue {
  azimuth: number;
  elevation: number;
  distance: number;
}

// Drag contraint à la surface d'une sphère de rayon fixe centrée sur
// l'auditeur (ex: poignées d'amplitude d'une ellipse). Contrairement à
// useDraggablePoint, la distance ne change jamais — seule la direction
// (azimut/élévation) varie.
export function useSphereDrag(
  radius: number,
  onChange: (v: SofaValue) => void,
  onDragStateChange: (dragging: boolean) => void,
) {
  const { camera, gl } = useThree();
  const dragging = useRef(false);
  const sphere = useRef(new THREE.Sphere()).current;
  const raycaster = useRef(new THREE.Raycaster()).current;
  const ndc = useRef(new THREE.Vector2()).current;
  const intersection = useRef(new THREE.Vector3()).current;

  const handleDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    sphere.center.set(0, 0, 0);
    sphere.radius = radius;
    dragging.current = true;
    onDragStateChange(true);
    gl.domElement.setPointerCapture(e.pointerId);
  };

  useEffect(() => {
    const dom = gl.domElement;

    const handleMove = (e: PointerEvent) => {
      if (!dragging.current) return;

      const rect = dom.getBoundingClientRect();
      ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
      ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(ndc, camera);

      if (raycaster.ray.intersectSphere(sphere, intersection)) {
        const { azimuth, elevation, distance } = threeToSofa(intersection);
        onChange({
          azimuth: Math.round(azimuth * 10) / 10,
          elevation: Math.round(elevation * 10) / 10,
          distance: Math.round(distance * 100) / 100,
        });
      }
    };

    const handleUp = (e: PointerEvent) => {
      if (!dragging.current) return;
      dragging.current = false;
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

  return { handleDown };
}
