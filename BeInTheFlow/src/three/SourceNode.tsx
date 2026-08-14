import { Suspense, useEffect, useRef } from "react";
import { useThree, type ThreeEvent } from "@react-three/fiber";
import * as THREE from "three";
import { Html } from "@react-three/drei";
import type { SoundSourceDTO } from "../types";
import { sofaToThree, threeToSofa } from "../utils/sofaCoords";
import { useSceneStore } from "../store/sceneStore";
import { getInstrument } from "./instrumentCatalog";
import InstrumentModel from "./InstrumentModel";

interface SourceNodeProps {
  source: SoundSourceDTO;
  selected: boolean;
  onSelect: () => void;
  onDragStateChange: (dragging: boolean) => void;
}

// Réutilisés à chaque frame de drag pour éviter des allocations répétées.
const dragPlane = new THREE.Plane();
const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
const intersection = new THREE.Vector3();
const UP = new THREE.Vector3(0, 1, 0);

type DragMode = "none" | "ground" | "elevation";

export default function SourceNode({ source, selected, onSelect, onDragStateChange }: SourceNodeProps) {
  const { camera, gl } = useThree();
  const updateSource = useSceneStore((s) => s.updateSource);
  const pos = sofaToThree(source.azimuth, source.elevation, source.distance);
  const instrument = getInstrument(source.modelId);
  const dragMode = useRef<DragMode>("none");
  const startWorldPos = useRef(new THREE.Vector3());

  const applyIntersection = (point: THREE.Vector3) => {
    const { azimuth, elevation, distance } = threeToSofa(point);
    updateSource(source.id, {
      azimuth: Math.round(azimuth * 10) / 10,
      elevation: Math.round(elevation * 10) / 10,
      distance: Math.round(distance * 100) / 100,
    });
  };

  // Déplacement libre sur le plan horizontal (hauteur Y constante, celle de la
  // source au clic) : attraper directement la sphère contrôle azimut + distance
  // ensemble, sans jamais être contraint à un axe ou à une sphère.
  const handleGroundPointerDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    onSelect();
    dragPlane.setFromNormalAndCoplanarPoint(UP, new THREE.Vector3(pos.x, pos.y, pos.z));
    dragMode.current = "ground";
    onDragStateChange(true);
    gl.domElement.setPointerCapture(e.pointerId);
  };

  // Flèche d'élévation : translation le long de l'axe Y monde. Le point le plus
  // proche entre le rayon de la souris et cette droite verticale donne la
  // nouvelle position — comme les poignées d'axe d'un gizmo 3D classique, ça
  // reste précis quel que soit l'angle de la caméra (pas juste le delta Y brut
  // en pixels).
  const handleElevationPointerDown = (e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    onSelect();
    startWorldPos.current.set(pos.x, pos.y, pos.z);
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
          applyIntersection(intersection);
        }
        return;
      }

      // "elevation" : point le plus proche entre la droite verticale (p0, UP)
      // et le rayon caméra->souris (formule classique de distance entre deux droites).
      const p0 = startWorldPos.current;
      const rOrigin = raycaster.ray.origin;
      const rDir = raycaster.ray.direction;
      const w0y = p0.y - rOrigin.y;
      const w0x = p0.x - rOrigin.x;
      const w0z = p0.z - rOrigin.z;
      const b = rDir.y; // UP·rDir
      const d = w0y; // UP·w0
      const e_ = rDir.x * w0x + rDir.y * w0y + rDir.z * w0z; // rDir·w0
      const denom = 1 - b * b;
      if (Math.abs(denom) > 1e-6) {
        const t = (b * e_ - d) / denom;
        intersection.set(p0.x, p0.y + t, p0.z);
        applyIntersection(intersection);
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
  }, [camera, gl, source.id, updateSource, onDragStateChange]);

  return (
    <group position={[pos.x, pos.y, pos.z]}>
      <mesh onPointerDown={handleGroundPointerDown} visible={!instrument}>
        <sphereGeometry args={[0.15, 24, 24]} />
        <meshStandardMaterial
          color={source.color}
          transparent
          opacity={source.muted ? 0.3 : selected ? 1 : 0.75}
        />
      </mesh>

      {instrument && (
        <>
          {/* Zone de préhension invisible (même geste que le cercle) : le modèle
              lui-même ignore le raycast pour éviter les surprises de hit-test
              sur des maillages complexes. */}
          <mesh onPointerDown={handleGroundPointerDown} visible={false}>
            <sphereGeometry args={[0.2, 12, 12]} />
            <meshBasicMaterial />
          </mesh>
          <Suspense fallback={null}>
            <InstrumentModel entry={instrument} position={pos} selected={selected} />
          </Suspense>
        </>
      )}

      {selected && (
        <>
          <mesh raycast={() => null}>
            <torusGeometry args={[0.24, 0.012, 8, 48]} />
            <meshBasicMaterial color="#c3073f" />
          </mesh>

          {/* Flèche d'élévation : zone de préhension invisible plus large + visuel fin. */}
          <group onPointerDown={handleElevationPointerDown}>
            <mesh position={[0, 0.32, 0]} visible={false}>
              <cylinderGeometry args={[0.09, 0.09, 0.6, 8]} />
              <meshBasicMaterial />
            </mesh>
            <mesh position={[0, 0.34, 0]} raycast={() => null}>
              <cylinderGeometry args={[0.012, 0.012, 0.28, 8]} />
              <meshBasicMaterial color="#4caf50" />
            </mesh>
            <mesh position={[0, 0.55, 0]} raycast={() => null}>
              <coneGeometry args={[0.045, 0.14, 12]} />
              <meshBasicMaterial color="#4caf50" />
            </mesh>
          </group>
        </>
      )}

      <Html distanceFactor={8} position={[0, selected ? 0.72 : 0.28, 0]} style={{ pointerEvents: "none" }}>
        <div className="source-label">{source.label}</div>
      </Html>
    </group>
  );
}