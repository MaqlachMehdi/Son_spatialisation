export default function ListenerHead() {
  return (
    <group>
      <mesh raycast={() => null}>
        <sphereGeometry args={[0.22, 32, 32]} />
        <meshStandardMaterial color="#cccccc" wireframe />
      </mesh>
      {/* petite flèche vers -Z : direction "devant" du listener */}
      <mesh position={[0, 0, -0.35]} rotation={[-Math.PI / 2, 0, 0]} raycast={() => null}>
        <coneGeometry args={[0.06, 0.18, 12]} />
        <meshStandardMaterial color="#999999" />
      </mesh>
    </group>
  );
}
