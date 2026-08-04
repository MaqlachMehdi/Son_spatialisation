interface ReferenceSphereProps {
  radius: number;
}

// Sphère de repère visuel à la distance de mesure HRTF par défaut (ex: 2.06 m IRCAM LISTEN).
// raycast désactivé : purement décoratif, ne doit jamais intercepter les clics/drags des sources.
export default function ReferenceSphere({ radius }: ReferenceSphereProps) {
  return (
    <mesh raycast={() => null}>
      <sphereGeometry args={[radius, 24, 16]} />
      <meshBasicMaterial color="#6f2232" wireframe transparent opacity={0.35} />
    </mesh>
  );
}