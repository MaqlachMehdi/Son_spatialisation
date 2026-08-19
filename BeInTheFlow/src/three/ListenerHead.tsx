import { useRef } from "react";
import { useFrame } from "@react-three/fiber";
import type { Group } from "three";
import { Model } from "./EvilScreamingManModel";
import { useListenerStore } from "../store/listenerStore";
import { sofaCartesianToThree, yprToThreeQuaternion } from "../utils/sofaCoords";

export default function ListenerHead() {
  const groupRef = useRef<Group>(null);

  // Lecture directe du store (pas d'abonnement React) à chaque frame — même
  // pattern que TrajectoryPlayer.tsx, pour ne pas re-render tout le composant
  // à la fréquence des échantillons d'orientation (potentiellement > 30Hz).
  useFrame(() => {
    const group = groupRef.current;
    if (!group) return;
    const { currentPose } = useListenerStore.getState();
    const p = sofaCartesianToThree(currentPose.x, currentPose.y, 0);
    group.position.set(p.x, p.y, p.z);
    group.quaternion.copy(yprToThreeQuaternion(currentPose.yaw, currentPose.pitch, currentPose.roll));
  });

  return (
    <group ref={groupRef}>
      {/* Nez orienté vers -Z, comme la flèche "devant" ci-dessous. */}
      <Model rotation={[0, 0, 0]} scale={4} />
      {/* petite flèche vers -Z : direction "devant" du listener */}
      <mesh position={[0, 0, -0.35]} rotation={[-Math.PI / 2, 0, 0]} raycast={() => null}>
        <coneGeometry args={[0.06, 0.18, 12]} />
        <meshStandardMaterial color="#999999" />
      </mesh>
    </group>
  );
}
