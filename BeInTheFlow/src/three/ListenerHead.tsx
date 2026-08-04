import { Model } from "./EvilScreamingManModel";

export default function ListenerHead() {
  return (
    <group>
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
